"""bern2hls — automated Bernstein-NN (.pth) to Vitis HLS project compiler.

Reproduces the per-dataset HLS kernels in {adult,covertype,higgs}-HW/sweep_test_all
byte-for-byte from the trained checkpoints, and provides unified synthesis
orchestration, result collection and plotting.
"""

__version__ = '1.0'
