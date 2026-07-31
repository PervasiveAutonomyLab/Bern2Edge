"""ooc: Vivado out-of-context synthesis + place&route of the HLS-generated RTL.

Post-route numbers are the trustworthy counterpart to the csynth estimates:
csynth LUT counts in particular are known to be pessimistic (~0.5-0.8x after
routing on KV260), while DSP counts are near-exact. For each project that has
completed csynth, this runs Vivado OOC synth -> opt -> place -> phys_opt ->
route on solution1/syn/verilog/ and writes util/timing/power reports to
<proj>/ooc/, where `collect --ooc` picks them up and emits comparison ratios.

The TCL and report parser are vendored (adapted from
lowpower_fpga/common/{ooc_impl.tcl,parse_reports.py}) to keep the artifact
self-contained. Source the matching Vivado settings64.sh before running.
"""

import glob
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Adapted from lowpower_fpga/common/ooc_impl.tcl (KV260-validated flow).
OOC_TCL = """\
# Out-of-context synthesis + implementation of Vitis HLS generated RTL.
# usage: vivado -mode batch -source ooc_impl.tcl -tclargs <top> <part> <clk_ns> <rtl_dir> <out_dir>
# NOTE: must be launched with cwd = <rtl_dir> so $readmemh .dat paths resolve.
lassign $argv top part clk_ns rtl out
file mkdir $out

read_verilog [glob -nocomplain $rtl/*.v]
synth_design -top $top -part $part -mode out_of_context
create_clock -period $clk_ns -name ap_clk [get_ports ap_clk]
set_clock_uncertainty 0.2 [get_clocks ap_clk]

report_utilization -file $out/util_synth.rpt

opt_design
place_design
phys_opt_design
route_design

report_utilization     -file $out/util.rpt -hierarchical
report_timing_summary  -file $out/timing.rpt
report_power           -file $out/power.rpt
report_design_analysis -file $out/design_analysis.rpt
write_checkpoint -force $out/routed.dcp

# one-line machine-readable summary
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "OOC_SUMMARY top=$top part=$part clk_ns=$clk_ns wns=$wns"
exit
"""


def find_rtl_dir(proj):
    matches = glob.glob(os.path.join(proj, 'script', '*_hls', 'solution1', 'syn', 'verilog'))
    return matches[0] if matches else None


def find_top(proj):
    tcl = os.path.join(proj, 'script', 'run_csynth.tcl')
    if os.path.isfile(tcl):
        m = re.search(r'^set_top\s+(\S+)', open(tcl).read(), re.M)
        if m:
            return m.group(1)
    return None


def discover_ooc_targets(roots):
    """Projects with csynth RTL available: [(proj, rtl_dir, top), ...]."""
    targets = []
    for root in roots:
        for name in sorted(os.listdir(root)):
            proj = os.path.join(root, name)
            rtl = find_rtl_dir(proj)
            top = find_top(proj)
            if rtl and top:
                targets.append((proj, rtl, top))
    return targets


def run_one(proj, rtl, top, part, clk_ns, vivado):
    out = os.path.join(proj, 'ooc')
    os.makedirs(out, exist_ok=True)
    tcl_path = os.path.join(out, 'ooc_impl.tcl')
    with open(tcl_path, 'w') as f:
        f.write(OOC_TCL)
    log_path = os.path.join(out, 'run.log')
    cmd = [vivado, '-mode', 'batch', '-nojournal',
           '-log', os.path.join(out, 'vivado.log'),
           '-source', tcl_path,
           '-tclargs', top, part, str(clk_ns), rtl, out]
    with open(log_path, 'w') as log:
        r = subprocess.run(cmd, cwd=rtl, stdout=log, stderr=subprocess.STDOUT)
    ok = r.returncode == 0 and os.path.isfile(os.path.join(out, 'util.rpt'))
    return proj, ok, log_path


def run_ooc(roots, only=None, jobs=1, part='xck26-sfvc784-2LV-c', clk_ns=10,
            vivado='vivado', force=False):
    targets = discover_ooc_targets(roots)
    if only:
        targets = [t for t in targets if os.path.basename(t[0]) == only]
    if not targets:
        print(f"No projects with csynth RTL (script/*_hls/solution1/syn/verilog) "
              f"under: {roots} — run csynth first.")
        return 1

    todo = []
    for proj, rtl, top in targets:
        if not force and os.path.isfile(os.path.join(proj, 'ooc', 'util.rpt')):
            print(f"SKIP {os.path.basename(proj)} (ooc/util.rpt exists; use --force to redo)")
            continue
        todo.append((proj, rtl, top))
    if not todo:
        print("Nothing to do.")
        return 0

    print(f"Running Vivado OOC synth+route on {len(todo)} project(s), "
          f"part={part}, clock={clk_ns}ns, {jobs} job(s)")
    results = []
    if jobs <= 1:
        for i, (proj, rtl, top) in enumerate(todo):
            print(f"[{i+1}/{len(todo)}] {os.path.basename(proj)} ({top}) ...", flush=True)
            results.append(run_one(proj, rtl, top, part, clk_ns, vivado))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(run_one, proj, rtl, top, part, clk_ns, vivado)
                    for proj, rtl, top in todo]
            for i, fut in enumerate(as_completed(futs)):
                proj, ok, log = fut.result()
                print(f"[{i+1}/{len(todo)}] {os.path.basename(proj)}: "
                      f"{'OK' if ok else 'FAIL'}", flush=True)
                results.append((proj, ok, log))

    fails = [(p, log) for p, ok, log in results if not ok]
    print(f"\nOOC: {len(results) - len(fails)}/{len(results)} succeeded")
    for p, log in fails:
        print(f"  FAIL {os.path.basename(p)}, log: {log}")
    return 1 if fails else 0


# =====================================================================
# Report parsing (vendored from lowpower_fpga/common/parse_reports.py)
# =====================================================================

def parse_ooc(outdir):
    """Parse Vivado OOC reports into a flat dict (post-route util/WNS/power)."""
    d = {}
    util = os.path.join(outdir, 'util.rpt')
    if os.path.exists(util):
        txt = open(util).read()
        for key, pat in [('lut', r'\| Slice LUTs\*?\s+\|\s+(\d+)'),
                         ('ff', r'\| Slice Registers\s+\|\s+(\d+)'),
                         ('bram_tile', r'\| Block RAM Tile\s+\|\s+(\d+)'),
                         ('dsp', r'\| DSPs\s+\|\s+(\d+)')]:
            m = re.search(pat, txt)
            if m:
                d[key] = m.group(1)
        if 'lut' not in d:
            # -hierarchical report: header-driven parse of the first (top) data
            # row. Column sets differ per family (UltraScale+ inserts URAM
            # between RAMB18 and DSP; names vary: 'DSP Blocks' vs 'DSPs').
            lines = txt.splitlines()
            header_cols = None
            for line in lines:
                if header_cols is None:
                    if line.startswith('|') and 'Instance' in line and 'Module' in line:
                        header_cols = [c.strip() for c in line.split('|')]
                    continue
                cells = [c.strip() for c in line.split('|')]
                if len(cells) == len(header_cols) and '(top)' in cells[2]:
                    byname = dict(zip(header_cols, cells))
                    def col(*names):
                        for n in names:
                            if byname.get(n, '').isdigit():
                                return byname[n]
                        return None
                    lut = col('Total LUTs', 'CLB LUTs', 'Slice LUTs')
                    ff = col('FFs', 'CLB Registers', 'Slice Registers')
                    ramb36 = col('RAMB36')
                    ramb18 = col('RAMB18')
                    dsp = col('DSP Blocks', 'DSPs', 'DSP48 Blocks')
                    if lut:
                        d['lut'] = lut
                    if ff:
                        d['ff'] = ff
                    if ramb36 is not None and ramb18 is not None:
                        d['bram18k'] = str(int(ramb36) * 2 + int(ramb18))
                    if dsp:
                        d['dsp'] = dsp
                    break
    log = os.path.join(outdir, 'vivado.log')
    if os.path.exists(log):
        m = re.search(r'OOC_SUMMARY top=\S+ part=(\S+) clk_ns=(\S+) wns=([\d.-]+)',
                      open(log).read())
        if m:
            d['part'] = m.group(1)
            d['clk_ns'] = m.group(2)
            d['wns_ns'] = m.group(3)
            try:
                d['fmax_mhz'] = f"{1000.0 / (float(m.group(2)) - float(m.group(3))):.1f}"
            except (ValueError, ZeroDivisionError):
                pass
    power = os.path.join(outdir, 'power.rpt')
    if os.path.exists(power):
        txt = open(power).read()
        for key, pat in [('total_power_w', r'\| Total On-Chip Power \(W\)\s+\|\s+([\d.]+)'),
                         ('dynamic_power_w', r'\| Dynamic \(W\)\s+\|\s+([\d.]+)'),
                         ('static_power_w', r'\| Device Static \(W\)\s+\|\s+([\d.]+)')]:
            m = re.search(pat, txt)
            if m:
                d[key] = m.group(1)
    return d
