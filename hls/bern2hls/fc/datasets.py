"""Dataset registry: per-dataset HLS template parameters and curated model lists.

The MODELS_* lists are lifted verbatim from the three original generators
({adult,covertype,higgs}-HW/sweep_test_all/generate_all_models.py) and define
the exact checkpoints that produced the shipped ground-truth projects.
`compile` can also run on arbitrary .pth files (config auto-extracted from
checkpoint metadata), in which case these lists are not consulted.
"""

import os
from dataclasses import dataclass, field

ARTIFACT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(ARTIFACT_DIR)

NEURON_LUT_SIZE = 50


@dataclass
class DatasetSpec:
    name: str                 # dataset key, also used for top fn / file stems
    input_dim: int
    output_dim: int
    family: str               # 'rom' (on-chip weight ROM) or 'axi' (m_axi streamed)
    num_samples: int          # test samples used by the shipped projects
    ckpt_dir: str             # default checkpoint dir, relative to repo root
    ground_truth_dir: str     # shipped projects dir, relative to repo root
    unroll_strategy: str = 'tier1'   # rom family only
    lut_grid: str = 'torch1'         # linspace variant of the shipping generation env
    models: list = field(default_factory=list)

    @property
    def top_name(self):
        return f'{self.name}_top'

    def default_ckpt_dir(self):
        """Curated-checkpoint location: the bundled artifact/checkpoints/<ds>/
        copy if present (self-contained clone), else the original repo dir."""
        bundled = os.path.join(ARTIFACT_DIR, 'checkpoints', self.name)
        if os.path.isdir(bundled):
            return bundled
        return os.path.join(REPO_ROOT, self.ckpt_dir)


MODELS_ADULT = [
    # --- Bernstein deg3, single hidden ---
    {'name': 'bern_d3_14x4x2',   'layer_sizes': [14,4,2],   'act': 'bern', 'degree': 3,
     'file': 'kd_fc_14x4x2_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed9.pth'},
    {'name': 'bern_d3_14x8x2',   'layer_sizes': [14,8,2],   'act': 'bern', 'degree': 3,
     'file': 'kd_fc_14x8x2_bern_deg3_alpha0.85_T2_lr0.006_wd0.0001_seed7.pth'},
    {'name': 'bern_d3_14x16x2',  'layer_sizes': [14,16,2],  'act': 'bern', 'degree': 3,
     'file': 'kd_fc_14x16x2_bern_deg3_alpha0.85_T2_lr0.006_wd0.0001_seed8.pth'},
    {'name': 'bern_d3_14x32x2',  'layer_sizes': [14,32,2],  'act': 'bern', 'degree': 3,
     'file': 'kd_fc_14x32x2_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed7.pth'},
    {'name': 'bern_d3_14x64x2',  'layer_sizes': [14,64,2],  'act': 'bern', 'degree': 3,
     'file': 'kd_fc_14x64x2_bern_deg3_alpha0_T1_lr0.006_wd0.0001_seed9.pth'},
    {'name': 'bern_d3_14x128x2', 'layer_sizes': [14,128,2], 'act': 'bern', 'degree': 3,
     'file': 'kd_fc_14x128x2_bern_deg3_alpha0_T1_lr0.006_wd0.0001_seed7.pth'},
    # --- Bernstein deg5, single hidden ---
    {'name': 'bern_d5_14x4x2',   'layer_sizes': [14,4,2],   'act': 'bern', 'degree': 5,
     'file': 'kd_fc_14x4x2_bern_deg5_alpha0.5_T2_lr0.006_wd0.0001_seed7.pth'},
    {'name': 'bern_d5_14x8x2',   'layer_sizes': [14,8,2],   'act': 'bern', 'degree': 5,
     'file': 'kd_fc_14x8x2_bern_deg5_alpha0.5_T2_lr0.006_wd0.0001_seed10.pth'},
    {'name': 'bern_d5_14x16x2',  'layer_sizes': [14,16,2],  'act': 'bern', 'degree': 5,
     'file': 'kd_fc_14x16x2_bern_deg5_alpha0.5_T2_lr0.006_wd0.0001_seed10.pth'},
    {'name': 'bern_d5_14x32x2',  'layer_sizes': [14,32,2],  'act': 'bern', 'degree': 5,
     'file': 'kd_fc_14x32x2_bern_deg5_alpha0.5_T2_lr0.006_wd0.0001_seed9.pth'},
    {'name': 'bern_d5_14x64x2',  'layer_sizes': [14,64,2],  'act': 'bern', 'degree': 5,
     'file': 'kd_fc_14x64x2_bern_deg5_alpha0.5_T2_lr0.006_wd0.0001_seed9.pth'},
    {'name': 'bern_d5_14x128x2', 'layer_sizes': [14,128,2], 'act': 'bern', 'degree': 5,
     'file': 'kd_fc_14x128x2_bern_deg5_alpha0.5_T2_lr0.006_wd0.0001_seed9.pth'},
    # --- ReLU, single hidden ---
    {'name': 'relu_14x4x2',   'layer_sizes': [14,4,2],   'act': 'relu', 'degree': None,
     'file': 'kd_fc_14x4x2_relu_degNA_alpha0.85_T2_lr0.006_wd0.0001_seed6.pth'},
    {'name': 'relu_14x8x2',   'layer_sizes': [14,8,2],   'act': 'relu', 'degree': None,
     'file': 'kd_fc_14x8x2_relu_degNA_alpha0_T1_lr0.006_wd0.0001_seed7.pth'},
    {'name': 'relu_14x16x2',  'layer_sizes': [14,16,2],  'act': 'relu', 'degree': None,
     'file': 'kd_fc_14x16x2_relu_degNA_alpha0.5_T2_lr0.006_wd0.0001_seed7.pth'},
    {'name': 'relu_14x32x2',  'layer_sizes': [14,32,2],  'act': 'relu', 'degree': None,
     'file': 'kd_fc_14x32x2_relu_degNA_alpha0.85_T2_lr0.006_wd0.0001_seed6.pth'},
    {'name': 'relu_14x64x2',  'layer_sizes': [14,64,2],  'act': 'relu', 'degree': None,
     'file': 'kd_fc_14x64x2_relu_degNA_alpha0.85_T2_lr0.006_wd0.0001_seed9.pth'},
    {'name': 'relu_14x128x2', 'layer_sizes': [14,128,2], 'act': 'relu', 'degree': None,
     'file': 'kd_fc_14x128x2_relu_degNA_alpha0.85_T2_lr0.006_wd0.0001_seed9.pth'},
    # --- Bernstein deg3, two hidden ---
    {'name': 'bern_d3_14x16x8x2',   'layer_sizes': [14,16,8,2],   'act': 'bern', 'degree': 3,
     'file': 'kd_fc_14x16x8x2_bern_deg3_alpha0.85_T2_lr0.006_wd0.0001_seed8.pth'},
    {'name': 'bern_d3_14x32x16x2',  'layer_sizes': [14,32,16,2],  'act': 'bern', 'degree': 3,
     'file': 'kd_fc_14x32x16x2_bern_deg3_alpha0.85_T2_lr0.006_wd0.0001_seed7.pth'},
    {'name': 'bern_d3_14x64x32x2',  'layer_sizes': [14,64,32,2],  'act': 'bern', 'degree': 3,
     'file': 'kd_fc_14x64x32x2_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed8.pth'},
    # --- Bernstein deg5, two hidden ---
    {'name': 'bern_d5_14x16x8x2',   'layer_sizes': [14,16,8,2],   'act': 'bern', 'degree': 5,
     'file': 'kd_fc_14x16x8x2_bern_deg5_alpha0.5_T2_lr0.006_wd0.0001_seed9.pth'},
    {'name': 'bern_d5_14x32x16x2',  'layer_sizes': [14,32,16,2],  'act': 'bern', 'degree': 5,
     'file': 'kd_fc_14x32x16x2_bern_deg5_alpha0.5_T2_lr0.006_wd0.0001_seed6.pth'},
    {'name': 'bern_d5_14x64x32x2',  'layer_sizes': [14,64,32,2],  'act': 'bern', 'degree': 5,
     'file': 'kd_fc_14x64x32x2_bern_deg5_alpha0.5_T2_lr0.006_wd0.0001_seed9.pth'},
    # --- ReLU, two hidden ---
    {'name': 'relu_14x16x8x2',   'layer_sizes': [14,16,8,2],   'act': 'relu', 'degree': None,
     'file': 'kd_fc_14x16x8x2_relu_degNA_alpha0.5_T2_lr0.006_wd0.0001_seed9.pth'},
    {'name': 'relu_14x32x16x2',  'layer_sizes': [14,32,16,2],  'act': 'relu', 'degree': None,
     'file': 'kd_fc_14x32x16x2_relu_degNA_alpha0.5_T2_lr0.006_wd0.0001_seed6.pth'},
    {'name': 'relu_14x64x32x2',  'layer_sizes': [14,64,32,2],  'act': 'relu', 'degree': None,
     'file': 'kd_fc_14x64x32x2_relu_degNA_alpha0.5_T2_lr0.006_wd0.0001_seed7.pth'},
]

MODELS_HIGGS = [
    # --- Bernstein deg3, single hidden ---
    {'name': 'bern_d3_28x4x2',   'layer_sizes': [28,4,2],   'act': 'bern', 'degree': 3,
     'file': 'kd_fc_28x4x2_bern_deg3_alpha0_T1_lr0.006_wd0_seed42.pth'},
    {'name': 'bern_d3_28x8x2',   'layer_sizes': [28,8,2],   'act': 'bern', 'degree': 3,
     'file': 'kd_fc_28x8x2_bern_deg3_alpha0.5_T2_lr0.006_wd0_seed45.pth'},
    {'name': 'bern_d3_28x16x2',  'layer_sizes': [28,16,2],  'act': 'bern', 'degree': 3,
     'file': 'kd_fc_28x16x2_bern_deg3_alpha0.5_T2_lr0.006_wd0_seed43.pth'},
    {'name': 'bern_d3_28x64x2',  'layer_sizes': [28,64,2],  'act': 'bern', 'degree': 3,
     'file': 'kd_fc_28x64x2_bern_deg3_alpha0.5_T2_lr0.006_wd0_seed43.pth'},
    {'name': 'bern_d3_28x128x2', 'layer_sizes': [28,128,2], 'act': 'bern', 'degree': 3,
     'file': 'kd_fc_28x128x2_bern_deg3_alpha0.85_T2_lr0.006_wd0_seed46.pth'},
    # --- ReLU, single hidden ---
    {'name': 'relu_28x4x2',   'layer_sizes': [28,4,2],   'act': 'relu', 'degree': None,
     'file': 'kd_fc_28x4x2_relu_degNA_alpha0_T1_lr0.006_wd0_seed45.pth'},
    {'name': 'relu_28x8x2',   'layer_sizes': [28,8,2],   'act': 'relu', 'degree': None,
     'file': 'kd_fc_28x8x2_relu_degNA_alpha0.85_T2_lr0.006_wd0_seed46.pth'},
    {'name': 'relu_28x16x2',  'layer_sizes': [28,16,2],  'act': 'relu', 'degree': None,
     'file': 'kd_fc_28x16x2_relu_degNA_alpha0.85_T2_lr0.006_wd0_seed46.pth'},
    {'name': 'relu_28x64x2',  'layer_sizes': [28,64,2],  'act': 'relu', 'degree': None,
     'file': 'kd_fc_28x64x2_relu_degNA_alpha0.5_T2_lr0.006_wd0_seed43.pth'},
    {'name': 'relu_28x128x2', 'layer_sizes': [28,128,2], 'act': 'relu', 'degree': None,
     'file': 'kd_fc_28x128x2_relu_degNA_alpha0.85_T2_lr0.006_wd0_seed46.pth'},
    # --- Bernstein deg3, two hidden ---
    {'name': 'bern_d3_28x16x8x2',   'layer_sizes': [28,16,8,2],   'act': 'bern', 'degree': 3,
     'file': 'kd_fc_28x16x8x2_bern_deg3_alpha0_T1_lr0.006_wd0_seed45.pth'},
    {'name': 'bern_d3_28x32x16x2',  'layer_sizes': [28,32,16,2],  'act': 'bern', 'degree': 3,
     'file': 'kd_fc_28x32x16x2_bern_deg3_alpha0.5_T2_lr0.006_wd0_seed43.pth'},
    {'name': 'bern_d3_28x64x32x2',  'layer_sizes': [28,64,32,2],  'act': 'bern', 'degree': 3,
     'file': 'kd_fc_28x64x32x2_bern_deg3_alpha0.5_T2_lr0.006_wd0_seed46.pth'},
    # --- ReLU, two hidden ---
    {'name': 'relu_28x16x8x2',   'layer_sizes': [28,16,8,2],   'act': 'relu', 'degree': None,
     'file': 'kd_fc_28x16x8x2_relu_degNA_alpha0.85_T2_lr0.006_wd0_seed46.pth'},
    {'name': 'relu_28x32x16x2',  'layer_sizes': [28,32,16,2],  'act': 'relu', 'degree': None,
     'file': 'kd_fc_28x32x16x2_relu_degNA_alpha0.5_T2_lr0.006_wd0_seed46.pth'},
    {'name': 'relu_28x64x32x2',  'layer_sizes': [28,64,32,2],  'act': 'relu', 'degree': None,
     'file': 'kd_fc_28x64x32x2_relu_degNA_alpha0.85_T2_lr0.006_wd0_seed44.pth'},
]

MODELS_COVERTYPE = [
    # --- Bernstein deg3, single hidden ---
    {'name': 'bern_d3_54x8x7',     'layer_sizes': [54,8,7],     'act': 'bern', 'degree': 3,
     'file': 'kd_fc_54x8x7_bern_deg3_alpha0.85_T2_lr0.006_wd0.0001_seed1002.pth'},
    {'name': 'bern_d3_54x12x7',    'layer_sizes': [54,12,7],    'act': 'bern', 'degree': 3,
     'file': 'kd_fc_54x12x7_bern_deg3_alpha0.85_T4_lr0.006_wd0.0001_seed1004.pth'},
    {'name': 'bern_d3_54x32x7',    'layer_sizes': [54,32,7],    'act': 'bern', 'degree': 3,
     'file': 'kd_fc_54x32x7_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed1001.pth'},
    {'name': 'bern_d3_54x64x7',    'layer_sizes': [54,64,7],    'act': 'bern', 'degree': 3,
     'file': 'kd_fc_54x64x7_bern_deg3_alpha0.9_T2_lr0.006_wd0.0001_seed1000.pth'},
    {'name': 'bern_d3_54x128x7',   'layer_sizes': [54,128,7],   'act': 'bern', 'degree': 3,
     'file': 'kd_fc_54x128x7_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed1001.pth'},
    {'name': 'bern_d3_54x256x7',   'layer_sizes': [54,256,7],   'act': 'bern', 'degree': 3,
     'file': 'kd_fc_54x256x7_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed1002.pth'},
    # --- Bernstein deg5, single hidden ---
    {'name': 'bern_d5_54x12x7',    'layer_sizes': [54,12,7],    'act': 'bern', 'degree': 5,
     'file': 'kd_fc_54x12x7_bern_deg5_alpha0.5_T2_lr0.006_wd0.0001_seed1004.pth'},
    {'name': 'bern_d5_54x256x7',   'layer_sizes': [54,256,7],   'act': 'bern', 'degree': 5,
     'file': 'kd_fc_54x256x7_bern_deg5_alpha0.5_T2_lr0.006_wd0.0001_seed1001.pth'},
    # --- Bernstein deg3, two hidden ---
    {'name': 'bern_d3_54x64x32x7',   'layer_sizes': [54,64,32,7],   'act': 'bern', 'degree': 3,
     'file': 'kd_fc_54x64x32x7_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed1004.pth'},
    {'name': 'bern_d3_54x128x64x7',  'layer_sizes': [54,128,64,7],  'act': 'bern', 'degree': 3,
     'file': 'kd_fc_54x128x64x7_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed1001.pth'},
    {'name': 'bern_d3_54x256x128x7', 'layer_sizes': [54,256,128,7], 'act': 'bern', 'degree': 3,
     'file': 'kd_fc_54x256x128x7_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed1002.pth'},
    # --- Bernstein deg5, two hidden ---
    {'name': 'bern_d5_54x128x64x7',  'layer_sizes': [54,128,64,7],  'act': 'bern', 'degree': 5,
     'file': 'kd_fc_54x128x64x7_bern_deg5_alpha0.5_T2_lr0.006_wd0.0001_seed1001.pth'},
    {'name': 'bern_d5_54x256x128x7', 'layer_sizes': [54,256,128,7], 'act': 'bern', 'degree': 5,
     'file': 'kd_fc_54x256x128x7_bern_deg5_alpha0.5_T2_lr0.006_wd0.0001_seed1000.pth'},
    # --- Bernstein deg3, three hidden ---
    {'name': 'bern_d3_54x128x64x64x7', 'layer_sizes': [54,128,64,64,7], 'act': 'bern', 'degree': 3,
     'file': 'kd_fc_54x128x64x64x7_bern_deg3_alpha0.5_T2_lr0.006_wd0.0001_seed1002.pth'},
    # --- ReLU, single hidden ---
    {'name': 'relu_54x8x7',     'layer_sizes': [54,8,7],     'act': 'relu', 'degree': None,
     'file': 'kd_fc_54x8x7_relu_degNA_alpha0.85_T2_lr0.006_wd0.0001_seed1000.pth'},
    {'name': 'relu_54x12x7',    'layer_sizes': [54,12,7],    'act': 'relu', 'degree': None,
     'file': 'kd_fc_54x12x7_relu_degNA_alpha0.85_T2_lr0.006_wd0.0001_seed1001.pth'},
    {'name': 'relu_54x32x7',    'layer_sizes': [54,32,7],    'act': 'relu', 'degree': None,
     'file': 'kd_fc_54x32x7_relu_degNA_alpha0.9_T2_lr0.006_wd0.0001_seed1000.pth'},
    {'name': 'relu_54x64x7',    'layer_sizes': [54,64,7],    'act': 'relu', 'degree': None,
     'file': 'kd_fc_54x64x7_relu_degNA_alpha0.9_T2_lr0.006_wd0.0001_seed1002.pth'},
    {'name': 'relu_54x128x7',   'layer_sizes': [54,128,7],   'act': 'relu', 'degree': None,
     'file': 'kd_fc_54x128x7_relu_degNA_alpha0.5_T2_lr0.006_wd0.0001_seed1000.pth'},
    {'name': 'relu_54x256x7',   'layer_sizes': [54,256,7],   'act': 'relu', 'degree': None,
     'file': 'kd_fc_54x256x7_relu_degNA_alpha0.3_T2_lr0.006_wd0.0001_seed1002.pth'},
    # --- ReLU, two hidden ---
    {'name': 'relu_54x64x32x7',   'layer_sizes': [54,64,32,7],   'act': 'relu', 'degree': None,
     'file': 'kd_fc_54x64x32x7_relu_degNA_alpha0.5_T2_lr0.006_wd0.0001_seed1003.pth'},
    {'name': 'relu_54x128x64x7',  'layer_sizes': [54,128,64,7],  'act': 'relu', 'degree': None,
     'file': 'kd_fc_54x128x64x7_relu_degNA_alpha0.5_T2_lr0.006_wd0.0001_seed1000.pth'},
    {'name': 'relu_54x256x128x7', 'layer_sizes': [54,256,128,7], 'act': 'relu', 'degree': None,
     'file': 'kd_fc_54x256x128x7_relu_degNA_alpha0.5_T2_lr0.006_wd0.0001_seed1004.pth'},
    # --- ReLU, three hidden ---
    {'name': 'relu_54x128x64x64x7', 'layer_sizes': [54,128,64,64,7], 'act': 'relu', 'degree': None,
     'file': 'kd_fc_54x128x64x64x7_relu_degNA_alpha0.5_T2_lr0.006_wd0.0001_seed1002.pth'},
]

DATASET_SPECS = {
    'adult': DatasetSpec(
        name='adult', input_dim=14, output_dim=2, family='rom', num_samples=20,
        ckpt_dir='adult-SW/saved_models_kd_adult',
        ground_truth_dir='adult-HW/sweep_test_all',
        unroll_strategy='tier1', models=MODELS_ADULT),
    'higgs': DatasetSpec(
        name='higgs', input_dim=28, output_dim=2, family='rom', num_samples=20,
        ckpt_dir='higgs-SW/saved_models_kd',
        ground_truth_dir='higgs-HW/sweep_test_all',
        unroll_strategy='tier1', lut_grid='torch2', models=MODELS_HIGGS),
    'covertype': DatasetSpec(
        name='covertype', input_dim=54, output_dim=7, family='axi', num_samples=10,
        ckpt_dir='covertype-SW/sweep_test_sw/saved_models_kd_covertype',
        ground_truth_dir='covertype-HW/sweep_test_all',
        models=MODELS_COVERTYPE),
}
