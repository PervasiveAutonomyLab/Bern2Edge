# Table I compression results

This directory reproduces the model metrics in paper Table I for Adult,
Covertype, and HIGGS-Small:

```bash
python table_i_compression/reproduce_table_i.py
```

The script evaluates the shipped checkpoints stored in the dataset directories
and writes the aggregate CSV files here. FPGA measurements are read from
`table_i_hls_results.csv`; synthesis is not rerun.

Student training remains in each dataset directory:

- `Adult/run_kd_experiments.py`
- `cover_type/run_kd_experiments.py`
- `higgs_small/run_kd_experiments.py`
