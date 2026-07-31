"""Shared codegen helpers (lifted verbatim from the original generators)."""

import os


def ceildiv(a, b):
    return (a + b - 1) // b


def arch_str(layer_sizes):
    return 'x'.join(str(s) for s in layer_sizes)


def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)


def write_flat(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for v in data.flatten():
            f.write(f'{v:.10f}\n')
