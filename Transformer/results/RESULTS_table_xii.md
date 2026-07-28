# TABLE XII -- SST-2 accuracy, latency, and FPGA resource utilization

Full 4-layer FFN substitution in TinyBERT4. FFN: FFN sublayers only;
Full: complete encoder. Resource changes are relative to TinyBERT4.

| Scope | Metric | TinyBERT4 | h=600 GeLU | h=600 Bern | h=312 GeLU | h=312 Bern |
|---|---|---|---|---|---|---|
| SST-2 | Acc. (%) | 90.37 | 90.02 | 90.48 | 89.11 | 90.02 |
| FFN | Lat. (cycles) | 6,413,520 | 3,282,640 (↓48.8%) | 3,278,288 (↓48.9%) | 1,789,648 (↓72.1%) | 1,785,296 (↓72.2%) |
| FFN | DSP | 121 | 121 (0.0%) | 80 (↓33.9%) | 121 (0.0%) | 80 (↓33.9%) |
| FFN | BRAM | 155 | 82 (↓47.1%) | 128 (↓17.4%) | 80 (↓48.4%) | 108 (↓30.3%) |
| FFN | FF | 17,746 | 19,580 (↑10.3%) | 15,654 (↓11.8%) | 19,528 (↑10.0%) | 15,568 (↓12.3%) |
| FFN | LUT | 34,744 | 35,165 (↑1.2%) | 30,070 (↓13.5%) | 34,924 (↑0.5%) | 29,789 (↓14.3%) |
| Full | Lat. (cycles) | 7,591,584 | 4,460,704 (↓41.2%) | 4,456,352 (↓41.3%) | 2,967,712 (↓60.9%) | 2,963,360 (↓61.0%) |
| Full | DSP | 319 | 319 (0.0%) | 278 (↓12.9%) | 319 (0.0%) | 278 (↓12.9%) |
| Full | BRAM | 164 | 91 (↓44.5%) | 137 (↓16.5%) | 89 (↓45.7%) | 117 (↓28.7%) |
| Full | FF | 40,302 | 42,049 (↑4.3%) | 38,210 (↓5.2%) | 41,997 (↑4.2%) | 38,037 (↓5.6%) |
| Full | LUT | 81,034 | 81,378 (↑0.4%) | 76,360 (↓5.8%) | 81,137 (↑0.1%) | 76,002 (↓6.2%) |

The **SST-2 Acc.** row is recomputed from the shipped weights by
`eval_release.py`. The latency and resource rows are **transcribed from the
paper**: HLS synthesis is outside the scope of this artifact.
