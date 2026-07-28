# Installation

Run all commands from the repository root.

## 1. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Python 3.10 is recommended. See `REQUIREMENTS.md` for supported versions,
hardware, storage, and network requirements.

## 2. Smoke test

These commands render committed paper results without downloading data or
training models:

```bash
python Adult/make_table3.py
python Adult/make_table8.py
python MAGIC/make_table5.py
python MAGIC/make_table_x.py
python ACS/make_table_xi.py
python Transformer/make_table_xii.py
```

Each command should print a Markdown table and write or update its corresponding
LaTeX/Markdown result file. For example, the first Transformer row should be:

```text
| SST-2 | Acc. (%) | 90.37 | 90.02 | 90.48 | 89.11 | 90.02 |
```

If the tables render without an exception, the installation is working.

## 3. Basic functional test

Recompute Covertype Table II from the shipped checkpoints:

```bash
python cover_type/reproduce_table_ii.py
```

The dataset is downloaded and cached on first use. A successful run ends with:

```text
Verified 50 checkpoints and all 10 five-fold means.
Wrote cover_type/table_ii_checkpoint_results.csv
Wrote cover_type/table_ii_results.csv
```

Additional reproduction commands are organized by paper result in `README.md`.

## Optional MAGIC certification dependency

`auto_LiRPA` is needed only to recompute the ReLU certification column in
MAGIC Table X. It is not needed to render the committed table.

```bash
pip install --no-deps git+https://github.com/Verified-Intelligence/auto_LiRPA.git
pip install graphviz
```

The GitHub installation is used because the older PyPI package pins an
incompatible PyTorch version.
