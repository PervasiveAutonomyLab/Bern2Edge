"""synth: run Vitis HLS (csim/csynth) over generated projects.

Replaces the per-dataset run_all_synthesis.sh scripts. A "project" is any
immediate subdirectory of a root that contains script/run_csynth.tcl; the
invocation matches the original flow: cd <proj>/script && vitis_hls -f <tcl>,
teeing output to <proj>/{mode}_log.txt.
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def discover_projects(roots):
    projects = []
    for root in roots:
        for name in sorted(os.listdir(root)):
            proj = os.path.join(root, name)
            if os.path.isfile(os.path.join(proj, 'script', 'run_csynth.tcl')):
                projects.append(proj)
    return projects


def hls_command(vitis_hls, tcl):
    """Build the invocation for either CLI style.

    Vitis HLS <= 2023.x ships a `vitis_hls` binary (vitis_hls -f run.tcl);
    2024.x replaced it with `vitis-run --mode hls --tcl run.tcl`. Source the
    matching settings64.sh before invoking synth.
    """
    cmd = vitis_hls.split()
    if 'vitis-run' in os.path.basename(cmd[0]):
        return cmd + ['--mode', 'hls', '--tcl', tcl]
    return cmd + ['-f', tcl]


# Named sweeps over the env-parameterized TCL. One project directory ends up
# holding one synthesized solution per point, which is why collect needs
# --all-solutions to report them all.
MATRICES = {
    # the quantization study: five fixed-point widths on the Spartan-7
    'lp_quant': [{'QW': str(w), 'QI': str(i), 'HLS_PART': 'xc7s15ftgb196-1',
                  'HLS_TAG': 'xc7s15', 'HLS_PERIOD': '15'}
                 for w, i in ((32, 16), (24, 12), (16, 8), (12, 7), (10, 6))],
    # the same design on both low-power parts, at the width that fits
    'lp_parts': [{'QW': '16', 'QI': '8', 'HLS_PART': part, 'HLS_TAG': tag,
                  'HLS_PERIOD': '15'}
                 for part, tag in (('xc7s15ftgb196-1', 'xc7s15'),
                                   ('xc7a15tcpg236-1', 'xc7a15t'))],
}


def parse_env(pairs):
    out = {}
    for p in pairs or []:
        if '=' not in p:
            raise ValueError(f"--env expects KEY=VALUE, got {p!r}")
        k, v = p.split('=', 1)
        out[k] = v
    return out


def point_tag(env):
    """A filename-safe label for one sweep point."""
    bits = []
    if 'QW' in env:
        bits.append(f"q{env['QW']}_{env.get('QI', '')}")
    if 'HLS_TAG' in env:
        bits.append(env['HLS_TAG'])
    return '_'.join(b for b in bits if b) or 'run'


def run_one(proj, mode, vitis_hls, env=None):
    scr = os.path.join(proj, 'script')
    tcl = f'run_{mode}.tcl'
    suffix = f'_{point_tag(env)}' if env else ''
    log_path = os.path.join(proj, f'{mode}{suffix}_log.txt')
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
        # the env-parameterized TCL decides whether to csim; keep the two
        # modes distinguishable when a sweep drives run_csynth.tcl only
        run_env.setdefault('HLS_SKIP_CSIM', '0' if mode == 'csim' else '0')
    with open(log_path, 'w') as log:
        r = subprocess.run(hls_command(vitis_hls, tcl), cwd=scr,
                           stdout=log, stderr=subprocess.STDOUT, env=run_env)
    return proj, r.returncode, log_path


def run_synth(roots, only=None, csim=False, jobs=1, vitis_hls='vitis_hls',
              matrix=None, env=None):
    mode = 'csim' if csim else 'csynth'
    projects = discover_projects(roots)
    if only:
        projects = [p for p in projects if os.path.basename(p) == only]
    if not projects:
        print(f"No projects with script/run_csynth.tcl found under: {roots}")
        return 1

    base = parse_env(env)
    if matrix:
        try:
            points = [{**pt, **base} for pt in MATRICES[matrix]]
        except KeyError:
            raise ValueError(f"unknown matrix '{matrix}' "
                             f"(available: {', '.join(sorted(MATRICES))})") from None
    else:
        points = [base] if base else [None]

    jobs_list = [(p, pt) for p in projects for pt in points]
    print(f"Running {mode} on {len(projects)} project(s) x {len(points)} point(s) "
          f"= {len(jobs_list)} run(s), {jobs} job(s), using '{vitis_hls}'")
    results = []
    if jobs <= 1:
        for i, (proj, pt) in enumerate(jobs_list):
            label = f'{os.path.basename(proj)}' + (f' [{point_tag(pt)}]' if pt else '')
            print(f"[{i+1}/{len(jobs_list)}] {label} ...", flush=True)
            results.append(run_one(proj, mode, vitis_hls, pt))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(run_one, p, mode, vitis_hls, pt): (p, pt)
                    for p, pt in jobs_list}
            for i, fut in enumerate(as_completed(futs)):
                proj, rc, log = fut.result()
                print(f"[{i+1}/{len(jobs_list)}] {os.path.basename(proj)}: "
                      f"{'OK' if rc == 0 else f'FAIL (rc={rc})'}", flush=True)
                results.append((proj, rc, log))

    fails = [(p, rc, log) for p, rc, log in results if rc != 0]
    print(f"\n{mode}: {len(results) - len(fails)}/{len(results)} succeeded")
    for p, rc, log in fails:
        print(f"  FAIL {os.path.basename(p)} (rc={rc}), log: {log}")
    return 1 if fails else 0
