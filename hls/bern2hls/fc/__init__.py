"""Front-end 1: fully-connected Bernstein networks (.pth) -> HLS projects.

Covers the adult, higgs and covertype sweeps via two code generators
(``codegen.rom_family`` for on-chip weight ROMs, ``codegen.axi_family``
for m_axi-streamed weights).
"""
