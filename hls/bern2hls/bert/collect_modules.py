"""Per-module area and latency breakdown of a synthesized encoder layer.

This is the evidence behind the transformer claim. The whole-layer totals move
very little between activations, which on its own says nothing about where the
cost is. Breaking the layer into modules shows that attention, softmax,
LayerNorm and both matmuls come out byte-for-byte identical across variants and
that the entire difference sits in one module — so the comparison really is
measuring the activation and not some second-order effect.

Two things about Vitis reports that are easy to get wrong, and would quietly
produce a wrong table:

* Area per module is HIERARCHICAL, i.e. inclusive of children. layer_top's
  numbers already contain attention_sublayer's, which already contain
  proj_stream's. Summing a column is meaningless; the rows are a decomposition
  to read down, not to add up.
* `Worst-caseLatency` is per *call*. The FFN leaves run once per token, so a
  leaf's contribution to the layer is lat x SEQ_LEN, while the sublayer and top
  reports are already whole-invocation figures.
"""

import csv
import glob
import os
import xml.etree.ElementTree as ET

METRICS = ['BRAM_18K', 'DSP', 'FF', 'LUT']

# Leaves the top-level FFN loop invokes once per token; the activation module
# is added per variant since its name depends on the activation.
PER_TOKEN_FFN = {'load_input', 'fc1_stream', 'fc2_stream'}

_ROLES = {
    'layer_top': ('top', 'layer_top (whole layer)'),
    'attention_sublayer': ('attention', 'attention_sublayer (Q/K/V·softmax·Wo+LN1)'),
    'proj_stream': ('attention', '  proj_stream (Q/K/V/Wo matmul)'),
    'softmax_row': ('attention', '  softmax_row (LUT exp + recip)'),
    'layernorm_1': ('attention', '  layernorm (LN1)'),
    'load_input': ('ffn', 'load_input (per-token DRAM load)'),
    'layernorm': ('ffn', 'layernorm (LN2)'),
}


def classify(module, act_module, hidden_dim, d_model=312):
    if module in _ROLES:
        return _ROLES[module]
    if module in ('fc1_stream', 'fc2_stream'):
        dims = (f'{d_model}→{hidden_dim}' if module == 'fc1_stream'
                else f'{hidden_dim}→{d_model}')
        return ('ffn', f'{module} ({dims} matmul)')
    if module == act_module:
        return ('ffn', f'{act_module} (ACTIVATION)')
    return ('other', module)


def parse_module_report(path):
    root = ET.parse(path).getroot()
    out = {}
    perf = root.find('PerformanceEstimates')
    lat = perf.find('SummaryOfOverallLatency/Worst-caseLatency') if perf is not None else None
    out['lat'] = int(lat.text) if lat is not None else 0
    area = root.find('AreaEstimates/Resources')
    for m in METRICS:
        e = area.find(m) if area is not None else None
        out[m] = int(e.text) if e is not None else 0
    return out


def module_reports(report_dir):
    """Top-level module reports only.

    Vitis also emits a report per pipelined loop (`*_Pipeline_*`), which are
    fragments of their parent rather than modules; including them would double
    count and clutter the table.
    """
    mods = {}
    for xml in sorted(glob.glob(os.path.join(report_dir, '*_csynth.xml'))):
        name = os.path.basename(xml)[:-len('_csynth.xml')]
        if '_Pipeline_' in name:
            continue
        mods[name] = parse_module_report(xml)
    return mods


def breakdown(spec, report_dir):
    """Rows for one variant, plus the measured/modelled latency reconciliation."""
    act_module = 'bernstein_layer' if spec.activation == 'bern' else 'gelu_layer'
    mods = module_reports(report_dir)
    per_token = PER_TOKEN_FFN | {act_module}
    rows, leaf_contrib = [], 0
    for name in sorted(mods):
        group, role = classify(name, act_module, spec.hidden_dim)
        m = mods[name]
        calls = spec.seq_len if name in per_token else 1
        contrib = m['lat'] * calls if name in per_token else None
        if contrib is not None:
            leaf_contrib += contrib
        rows.append(dict(variant=spec.name, group=group, module=name, role=role,
                         lat_call=m['lat'], calls=calls,
                         lat_contrib=(contrib if contrib is not None else ''),
                         **{k: m[k] for k in METRICS}))
    top, attn = mods.get('layer_top'), mods.get('attention_sublayer')
    summary = None
    if top and attn:
        summary = dict(total=top['lat'], attention=attn['lat'],
                       ffn=top['lat'] - attn['lat'], ffn_leaf_model=leaf_contrib)
    return rows, summary


FIELDS = ['variant', 'group', 'module', 'role'] + METRICS + \
         ['lat_call', 'calls', 'lat_contrib']


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    return path
