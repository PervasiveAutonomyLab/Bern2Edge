"""collect: unified synthesis-results collector.

Scans one or more roots for Vitis HLS reports at
<proj>/script/*_hls/solution1/syn/report/csynth.xml (also matches bare
*/solution1/syn/report/csynth.xml layouts), parses the metrics the
per-dataset collect_results.py scripts extracted, labels each row with
dataset / model / activation / degree / arch derived from the project dir
name, and writes one CSV. With --ooc it additionally fills post-route
columns (WNS/fmax/power) when Vivado OOC reports exist next to a design
(reusing lowpower_fpga/common/parse_reports.py conventions).
"""

import csv
import glob
import os
import re
import xml.etree.ElementTree as ET

NAME_RE = re.compile(r'^(?P<act>bern|relu)(?:_d(?P<deg>\d+))?_(?P<arch>\d+(?:x\d+)+)$')

ACC_COLUMNS = ['csim_acc_pct', 'csim_correct', 'csim_samples',
               'ref_acc_pct', 'ref_agree_pct', 'ref_mismatch', 'csim_verdict']

CSV_COLUMNS = [
    'dataset', 'model', 'act', 'degree', 'arch', 'status',
    'latency_cycles', 'clock_ns', 'latency_us', 'interval',
    'BRAM_18K', 'DSP', 'FF', 'LUT', 'URAM',
    'BRAM_pct', 'DSP_pct', 'FF_pct', 'LUT_pct',
    'part', 'top_fn',
    # optional OOC post-route columns (+ post-route/csynth ratios)
    'ooc_lut', 'ooc_ff', 'ooc_dsp', 'ooc_bram18k',
    'lut_ratio', 'ff_ratio', 'dsp_ratio', 'bram_ratio',
    'wns_ns', 'fmax_mhz', 'total_power_w', 'dynamic_power_w', 'static_power_w',
]
CSV_COLUMNS = CSV_COLUMNS + ACC_COLUMNS


def parse_csynth_xml(xml_path):
    """Parse a csynth.xml file and extract key metrics.

    Lifted from adult-HW/sweep_test_all/collect_results.py.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    result = {}

    perf = root.find('PerformanceEstimates')
    if perf is not None:
        timing = perf.find('SummaryOfTimingAnalysis')
        if timing is not None:
            clk = timing.find('EstimatedClockPeriod')
            if clk is not None:
                result['clock_ns'] = float(clk.text)

        latency = perf.find('SummaryOfOverallLatency')
        if latency is not None:
            for tag in ['Worst-caseLatency', 'Best-caseLatency']:
                el = latency.find(tag)
                if el is not None:
                    result[tag] = int(el.text)
            interval_min = latency.find('Interval-min')
            if interval_min is not None:
                result['interval'] = int(interval_min.text)

    area = root.find('AreaEstimates')
    if area is not None:
        resources = area.find('Resources')
        available = area.find('AvailableResources')
        for tag in ['BRAM_18K', 'DSP', 'FF', 'LUT', 'URAM']:
            el = resources.find(tag) if resources is not None else None
            if el is not None:
                result[tag] = int(el.text)
            avail_el = available.find(tag) if available is not None else None
            if avail_el is not None:
                result[f'{tag}_avail'] = int(avail_el.text)

    ua = root.find('UserAssignments')
    if ua is not None:
        part = ua.find('Part')
        if part is not None:
            result['part'] = part.text
        top = ua.find('TopModelName')
        if top is not None:
            result['top_fn'] = top.text

    return result


def find_reports(root, all_solutions=False):
    """Yield (label, csynth_xml_path) pairs under a root.

    A project directory can hold more than one synthesized solution. The
    low-power sweeps rely on this: one env-parameterized TCL builds
    `pn_14x16x2_q16_8_xc7s15_hls`, `..._q32_16_xc7s15_hls` and so on side by
    side, so a single directory carries up to nine results.

    By default one row per project is emitted, which is what the flow-1 sweeps
    want (one design, one solution). Pass all_solutions=True to emit every
    solution, labelled by its `*_hls` directory — without it a 105-row sweep
    silently collapses to 23 rows, which reads as complete.
    """
    patterns = [
        os.path.join(root, '*', 'script', '*_hls', 'solution1', 'syn', 'report', 'csynth.xml'),
        os.path.join(root, '*', '*_hls', 'solution1', 'syn', 'report', 'csynth.xml'),
        os.path.join(root, '*_hls', 'solution1', 'syn', 'report', 'csynth.xml'),
        # one level deeper: sweeps that group their points in a subdirectory
        # (bern_lut_sweep/lut32/, fallback variants' fb_only/, ...)
        os.path.join(root, '*', '*', 'script', '*_hls', 'solution1', 'syn', 'report',
                     'csynth.xml'),
    ]
    seen = set()
    for pattern in patterns:
        for xml_path in sorted(glob.glob(pattern)):
            rel = os.path.relpath(xml_path, root)
            parts = rel.split(os.sep)
            # the project is everything above script/<sol>_hls/
            if 'script' in parts:
                proj = os.sep.join(parts[:parts.index('script')])
            else:
                proj = parts[0]
            if proj.endswith('_hls') and rel.count(os.sep) <= 4:
                proj = proj[:-len('_hls')]
            if all_solutions:
                # name the row after the solution, so quantization/part sweep
                # points stay distinguishable
                sol = next((p for p in parts if p.endswith('_hls')), None)
                label = f'{proj}/{sol[:-len("_hls")]}' if sol else proj
                if label in seen:
                    continue
                seen.add(label)
                yield label, xml_path
            elif proj not in seen:
                seen.add(proj)
                yield proj, xml_path


# csim prints accuracy in one of three shapes, depending on which testbench a
# flow uses. All are scraped so a results table does not need a per-flow parser.
ACC_PATTERNS = [
    # low-power accuracy TB:  ACCURACY: 7681/9045 = 84.9198%
    re.compile(r'ACCURACY:\s*(?P<n>\d+)\s*/\s*(?P<total>\d+)\s*=\s*(?P<pct>[\d.]+)%'),
    # rule classifier TB:     HLS  acc : 7438/9045 (82.23%)
    re.compile(r'HLS\s+acc\s*:\s*(?P<n>\d+)\s*/\s*(?P<total>\d+)\s*\((?P<pct>[\d.]+)%\)'),
    # k-sweep TB:             HLS accuracy:  149/200 (74.50%)
    re.compile(r'HLS\s+accuracy:\s*(?P<n>\d+)\s*/\s*(?P<total>\d+)\s*\((?P<pct>[\d.]+)%\)'),
]
# the k-sweep TB reports the golden reference's accuracy rather than agreement
REF_ACC_PATTERN = re.compile(
    r'Ref\s+accuracy:\s*(?P<n>\d+)\s*/\s*(?P<total>\d+)\s*\((?P<pct>[\d.]+)%\)')
# rule TB also reports agreement with the golden reference
AGREE_PATTERN = re.compile(
    r'HLS==Ref\s*:\s*(?P<n>\d+)\s*/\s*(?P<total>\d+)\s*\((?P<pct>[\d.]+)%\)'
    r'\s*mismatch=(?P<mismatch>\d+)')
# flow-1 self-checking TB reports a verdict rather than an accuracy
VERDICT_PATTERN = re.compile(r'RESULT: (ALL TESTS PASSED|ALL MATCH|OK|[^\n]*)')

# csim output lands in different places depending on which driver ran it
LOG_GLOBS = ['csim_log.txt', 'synthesis_log.txt', 'script/*.log', '*.log',
             'script/synthesis_log.txt']


def scrape_accuracy(proj_dir, solution=None):
    """Pull csim accuracy out of whatever log the flow left behind.

    `solution` matters when a project directory holds several: the logs sit
    side by side (synth_q16_8_xc7s15.log, synth_q32_16_xc7s15.log, ...) and
    reading all of them would attribute one solution's accuracy to every
    quantization point. When a solution is named, only logs mentioning it are
    read; a project with a single solution keeps reading everything.
    """
    out = {}
    paths = []
    for pattern in LOG_GLOBS:
        paths.extend(sorted(glob.glob(os.path.join(proj_dir, pattern))))
    if solution:
        matched = [p for p in paths if solution in os.path.basename(p)]
        if matched:
            paths = matched
        else:
            return {}          # no log for this solution: report nothing, not someone else's
    texts = []
    for path in paths:
        try:
            texts.append(open(path, errors='replace').read())
        except OSError:
            pass
    for text in texts:
        for pat in ACC_PATTERNS:
            m = None
            for m in pat.finditer(text):     # last occurrence wins: latest run
                pass
            if m:
                out['csim_acc_pct'] = float(m.group('pct'))
                out['csim_correct'] = int(m.group('n'))
                out['csim_samples'] = int(m.group('total'))
        m = None
        for m in AGREE_PATTERN.finditer(text):
            pass
        if m:
            out['ref_agree_pct'] = float(m.group('pct'))
            out['ref_mismatch'] = int(m.group('mismatch'))
        m = None
        for m in REF_ACC_PATTERN.finditer(text):
            pass
        if m:
            out['ref_acc_pct'] = float(m.group('pct'))
        m = VERDICT_PATTERN.search(text)
        if m and 'csim_verdict' not in out:
            out['csim_verdict'] = m.group(1).strip()
    return out


def dataset_label(root):
    """Derive a dataset label from a root path (e.g. .../adult-HW/sweep_test_all -> adult)."""
    parts = os.path.abspath(root).split(os.sep)
    for p in reversed(parts):
        low = p.lower()
        for ds in ('adult', 'covertype', 'higgs'):
            if ds in low:
                return ds
    return os.path.basename(os.path.abspath(root))


def parse_project_name(name):
    m = NAME_RE.match(name)
    if not m:
        return {'act': '', 'degree': '', 'arch': ''}
    d = m.groupdict()
    return {'act': d['act'], 'degree': d['deg'] or '', 'arch': d['arch']}


def try_parse_ooc(proj_dir):
    """Fill post-route columns if Vivado OOC reports exist for this design."""
    from .ooc import parse_ooc
    candidates = [d for d in sorted(glob.glob(os.path.join(proj_dir, 'ooc*')))
                  if os.path.isdir(d)]
    if not candidates:
        return {}
    row = parse_ooc(candidates[0])
    bram = row.get('bram18k', '')
    if not bram and 'bram_tile' in row:
        bram = str(int(row['bram_tile']) * 2)
    return {
        'ooc_lut': row.get('lut', ''), 'ooc_ff': row.get('ff', ''),
        'ooc_dsp': row.get('dsp', ''), 'ooc_bram18k': bram,
        'wns_ns': row.get('wns_ns', ''), 'fmax_mhz': row.get('fmax_mhz', ''),
        'total_power_w': row.get('total_power_w', ''),
        'dynamic_power_w': row.get('dynamic_power_w', ''),
        'static_power_w': row.get('static_power_w', ''),
    }


def add_ratios(row):
    """Post-route / csynth resource ratios where both sides are present."""
    for cs, pr, col in [('LUT', 'ooc_lut', 'lut_ratio'), ('FF', 'ooc_ff', 'ff_ratio'),
                        ('DSP', 'ooc_dsp', 'dsp_ratio'),
                        ('BRAM_18K', 'ooc_bram18k', 'bram_ratio')]:
        try:
            row[col] = round(int(row[pr]) / int(row[cs]), 2)
        except (ValueError, KeyError, ZeroDivisionError, TypeError):
            pass


def collect_rows(roots, with_ooc=False, all_solutions=False, accuracy=False):
    rows = []
    for root in roots:
        ds = dataset_label(root)
        for proj, xml_path in find_reports(root, all_solutions):
            row = {c: '' for c in CSV_COLUMNS}
            row.update({'dataset': ds, 'model': proj})
            row.update(parse_project_name(proj))
            try:
                data = parse_csynth_xml(xml_path)
            except Exception as e:
                row['status'] = f'ERROR: {e}'
                rows.append(row)
                continue
            row['status'] = 'OK'
            lat = data.get('Worst-caseLatency')
            clk = data.get('clock_ns')
            row['latency_cycles'] = lat if lat is not None else ''
            row['clock_ns'] = clk if clk is not None else ''
            if lat is not None and clk is not None:
                row['latency_us'] = lat * clk / 1000.0
            row['interval'] = data.get('interval', '')
            for tag in ['BRAM_18K', 'DSP', 'FF', 'LUT', 'URAM']:
                row[tag] = data.get(tag, '')
            for tag, col in [('BRAM_18K', 'BRAM_pct'), ('DSP', 'DSP_pct'),
                             ('FF', 'FF_pct'), ('LUT', 'LUT_pct')]:
                if tag in data and data.get(f'{tag}_avail'):
                    row[col] = round(100.0 * data[tag] / data[f'{tag}_avail'], 2)
            row['part'] = data.get('part', '')
            row['top_fn'] = data.get('top_fn', '')
            # `proj` may be "<project>/<solution>" under --all-solutions
            proj_dir = os.path.join(root, proj.split('/')[0])
            if with_ooc:
                row.update(try_parse_ooc(proj_dir))
                add_ratios(row)
            if accuracy:
                sol = proj.split('/', 1)[1] if '/' in proj else None
                row.update(scrape_accuracy(proj_dir, sol))
            rows.append(row)
    return rows


def print_table(rows):
    header = (f"{'dataset':<10s} {'model':<26s} {'latency':>9s} {'clk_ns':>7s} "
              f"{'lat_us':>8s} {'BRAM':>5s} {'DSP':>5s} {'FF':>8s} {'LUT':>8s}")
    print(header)
    print('-' * len(header))
    for r in rows:
        if r['status'] != 'OK':
            print(f"{r['dataset']:<10s} {r['model']:<26s} {r['status']}")
            continue
        lat_us = f"{r['latency_us']:.2f}" if r['latency_us'] != '' else ''
        print(f"{r['dataset']:<10s} {r['model']:<26s} {r['latency_cycles']:>9} "
              f"{r['clock_ns']:>7} {lat_us:>8s} {r['BRAM_18K']:>5} {r['DSP']:>5} "
              f"{r['FF']:>8} {r['LUT']:>8}")
    ok = sum(1 for r in rows if r['status'] == 'OK')
    print('-' * len(header))
    print(f"{ok}/{len(rows)} reports parsed")


def print_comparison_table(rows):
    """csynth estimates vs Vivado OOC post-route actuals, for rows that have both."""
    rows = [r for r in rows if r.get('ooc_lut')]
    if not rows:
        return
    header = (f"{'model':<26s} {'LUT cs/pr (x)':>17s} {'FF cs/pr (x)':>17s} "
              f"{'DSP cs/pr':>11s} {'BRAM cs/pr':>11s} {'WNS':>6s} {'fmax':>6s} {'P(W)':>6s}")
    print(f"\ncsynth estimate vs Vivado OOC post-route")
    print(header)
    print('-' * len(header))
    for r in rows:
        lut = f"{r['LUT']}/{r['ooc_lut']} ({r.get('lut_ratio', '?')})"
        ff = f"{r['FF']}/{r['ooc_ff']} ({r.get('ff_ratio', '?')})"
        dsp = f"{r['DSP']}/{r['ooc_dsp']}"
        bram = f"{r['BRAM_18K']}/{r['ooc_bram18k']}"
        print(f"{r['model']:<26s} {lut:>17s} {ff:>17s} {dsp:>11s} {bram:>11s} "
              f"{r.get('wns_ns', ''):>6s} {r.get('fmax_mhz', ''):>6s} "
              f"{r.get('total_power_w', ''):>6s}")
    print('-' * len(header))


def run_collect(roots, csv_path=None, with_ooc=False, all_solutions=False,
                accuracy=False):
    rows = collect_rows(roots, with_ooc=with_ooc, all_solutions=all_solutions,
                        accuracy=accuracy)
    if not rows:
        print(f"No csynth.xml reports found under: {roots}")
        return 1
    print_table(rows)
    if with_ooc:
        print_comparison_table(rows)
    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV saved to: {csv_path}")
    return 0
