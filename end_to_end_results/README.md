# Table VI — end-to-end results

This directory reproduces the software accuracy provenance and renders the
post-synthesis end-to-end comparison across HIGGS-Small, Covertype, and Adult.
The selected network accuracies are evaluated from the committed teacher and
five-fold Bernstein student checkpoints. The Adult Rules row uses the
`14x16x8x2` artifact with `same_cov_alpha=0.5` and `conflict_alpha=0.1`.

## Reproduce

From the repository root, after following `INSTALL.md`:

```bash
python end_to_end_results/reproduce_table_vi.py
```

The command writes:

- `table_vi_artifact_metrics.csv`: one live evaluation per checkpoint/artifact;
- `table_vi_values.csv`: software metrics joined with the supplied hardware
  results and their relative reductions;
- `table_vi.tex`: the rendered table.

The first run may download HIGGS-Small, Covertype, and Adult through the existing
dataset loaders. Subsequent runs use the local caches.

## Provenance

- HIGGS-Small LUT: five `28x16x8x2`, degree-3 Bernstein checkpoints.
- Covertype LUT: five `54x256x128x7`, degree-5 Bernstein checkpoints.
- Adult LUT: five `14x16x2`, degree-3 Bernstein checkpoints.
- Adult Rules: `Adult/rule_jsons/...14x16x8x2...sca0.5_ca0.1/rules_float.json`
  and its committed CART sidecar.
- Teachers: `higgs_small/higgs_small_teacher.pth`,
  `cover_type/covertype_teacher_weights.pth`, and
  `Adult/adult_teacher_ordinal.pt`.
- `hardware_results.csv`: post-synthesis accuracy and hardware values supplied
  for Table VI.

The script evaluates software artifacts but does not rerun HLS synthesis.
`hardware_results.csv` is therefore the authoritative source for the displayed
post-synthesis accuracy and hardware columns.

Both generated CSVs retain an explicit `architecture` column. The detailed CSV
also records the direct project-relative `.pth` or rule JSON path for every
evaluation; the final CSV preserves all five checkpoint paths for averaged LUT
rows.
