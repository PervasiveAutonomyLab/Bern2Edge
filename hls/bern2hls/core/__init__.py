"""Flow-agnostic machinery: source emission, verification, and the
Vitis HLS / Vivado / results pipeline.

Nothing here knows about Bernstein networks, rule sets or transformers —
these modules only assume the emitted project layout
``{include,src,tb,data,script}/`` with ``script/run_{csim,csynth}.tcl``.
"""
