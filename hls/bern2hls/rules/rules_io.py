"""Decode a distilled rule set (`rules_int8.json`) into ROM-ready tables.

A rule is a conjunction of band conditions on *linear projections* of the
input, not axis-aligned thresholds:

    rule fires  <=>  for every condition c:  lo_c <= w_c . x < hi_c

with the bias folded into the band, and either bound possibly absent. The
winning rule is the firing one with the highest purity; if nothing fires the
input falls through to a Phase-3 fallback.

Two storage forms, both shipped:

* dense  — `COND_W8[R][C][N_FEATURES]`, the full quantized projection.
* sparse — `COND_W8[R][C][K]` plus a 4-bit `COND_IDX[R][C][K]`, keeping only
  the K largest-magnitude weights per condition.

NOTE: the extraction algorithm that *produces* these JSON files is not part of
this artifact — see the README. This module consumes a released rule set.
"""

import json


def fp_to_int8(weight_vector, scale):
    """Symmetric int8 quantization; `scale` is absmax/127 from the JSON."""
    vals = []
    for w in weight_vector:
        if abs(w) < 1e-12:
            vals.append(0)
        else:
            vals.append(max(-128, min(127, int(round(w / scale)))))
    return vals


class RuleModel:
    """A decoded rules_int8.json."""

    def __init__(self, path):
        with open(path) as f:
            self.raw = json.load(f)
        self.feature_names = self.raw['feature_names']
        self.feat_idx = {n: i for i, n in enumerate(self.feature_names)}
        self.config = self.raw.get('config', {})
        self.metrics = self.raw.get('metrics', {})
        self.fallback = self.raw.get('fallback', {})
        # Rules without conditions are the default/uncovered marker, not rules.
        self.rules = [r for r in self.raw['rules'] if r.get('conditions')]

    @property
    def n_features(self):
        return len(self.feature_names)

    @property
    def n_rules(self):
        return len(self.rules)

    @property
    def max_conds(self):
        return max(r['n_cond'] for r in self.rules)

    @property
    def sparsity_k(self):
        """K for the sparse form: the widest `sparse_weights` map in the set.

        Taken from the data rather than config['sparsity_k'] because the
        k-sweep JSONs use an older schema where that key is absent.
        """
        return max(len(c['sparse_weights'])
                   for r in self.rules for c in r['conditions'])

    @property
    def is_sparse(self):
        return bool(self.config.get('sparsified'))

    def dense_weight(self, cond):
        """The condition's projection as a dense vector over feature_names.

        `weight_vector` is already dense. `sparse_weights` is a {name: w} map
        of the retained top-k. Which one a suite dots with is a per-suite
        choice that changes the golden reference (see spec.RuleSpec.dot_source)
        — getting it backwards silently changes test_output_ref.txt.
        """
        return cond['weight_vector']

    def sparse_weight(self, cond):
        sw = cond.get('sparse_weights')
        if sw is None:
            return cond['weight_vector']
        return [sw.get(n, 0.0) for n in self.feature_names]

    def rom_tables(self, k_sparse=None):
        """Per-rule ROM tables, padded to MAX_CONDS.

        With `k_sparse` set, emit the sparse form: the K largest-magnitude
        weights and their feature indices. Otherwise the dense form.
        """
        nf, mc = self.n_features, self.max_conds
        width = k_sparse or nf
        t = {k: [] for k in ('cond_w8', 'cond_idx', 'cond_scale', 'band_lo',
                             'band_hi', 'has_lo', 'has_hi',
                             'labels', 'purity', 'n_cond')}
        for r in self.rules:
            t['labels'].append(r['label'])
            t['purity'].append(int(round(r['purity'] * 10000)))
            t['n_cond'].append(r['n_cond'])
            rw, ridx, rs = [], [], []
            rlo, rhi, rhlo, rhhi = [], [], [], []
            for c in r['conditions']:
                dense = c['weight_vector']
                if k_sparse:
                    # Retained weights come from `sparse_weights` in JSON order.
                    # Re-selecting the top-k from the dense vector would give a
                    # different column order and a different ROM.
                    names = list(c['sparse_weights'])
                    w8 = fp_to_int8([c['sparse_weights'][n] for n in names],
                                    c['int8_scale'])
                    idx = [self.feat_idx[n] for n in names]
                    while len(w8) < k_sparse:
                        w8.append(0)
                        idx.append(0)
                    rw.append(w8)
                    ridx.append(idx)
                else:
                    rw.append(fp_to_int8(dense, c['int8_scale']))
                    ridx.append([0] * width)
                rs.append(c['int8_scale'])
                rlo.append(c['band_lo'])
                rhi.append(c['band_hi'])
                rhlo.append(c['band_lo'] is not None)
                rhhi.append(c['band_hi'] is not None)
            while len(rw) < mc:      # pad unused condition slots
                rw.append([0] * width)
                ridx.append([0] * width)
                rs.append(0.0)
                rlo.append(0.0)
                rhi.append(0.0)
                rhlo.append(False)
                rhhi.append(False)
            for key, val in (('cond_w8', rw), ('cond_idx', ridx), ('cond_scale', rs),
                             ('band_lo', rlo), ('band_hi', rhi),
                             ('has_lo', rhlo), ('has_hi', rhhi)):
                t[key].append(val)
        return t
