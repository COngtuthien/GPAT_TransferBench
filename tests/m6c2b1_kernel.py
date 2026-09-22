"""Synthetic-only ABI harness for the UNCHANGED pinned Sim3DR C++ kernel.

This is not a production fallback for the missing Sim3DR_Cython extension.
The tiny C linkage shim does no rendering math; binaries live only in /tmp.
"""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from methods.physics_std.source import validate_source


class SyntheticOfficialKernel:
    def __init__(self, config):
        compiler = shutil.which('c++')
        if compiler is None:
            raise RuntimeError('CPU c++ compiler unavailable for official-kernel synthetic smoke')
        source = validate_source(config)
        self.tmp = tempfile.TemporaryDirectory(prefix='gpat-e04-kernel-')
        root = Path(source['root']) / 'Sim3DR/lib'
        shim = Path(self.tmp.name) / 'shim.cpp'
        shim.write_text('''#include "rasterize.h"
extern "C" void gpat_rasterize(unsigned char* image, float* vertices,
int* triangles, float* colors, float* buffer, int n, int h, int w, int c,
float alpha, bool reverse) {
    _rasterize(image, vertices, triangles, colors, buffer, n, h, w, c, alpha, reverse);
}
''')
        library = Path(self.tmp.name) / 'kernel.so'
        subprocess.run([compiler, '-std=c++11', '-shared', '-fPIC', '-I', str(root),
                        str(shim), str(root / 'rasterize_kernel.cpp'), '-o', str(library)],
                       check=True, capture_output=True)
        self.library = ctypes.CDLL(str(library))
        self.function = self.library.gpat_rasterize
        self.function.restype = None
        self.function.argtypes = [np.ctypeslib.ndpointer(dtype=d, flags='C_CONTIGUOUS')
                                  for d in (np.uint8, np.float32, np.int32, np.float32, np.float32)] + [
                                  ctypes.c_int]*4 + [ctypes.c_float, ctypes.c_bool]

    def rasterize(self, image, vertices, triangles, colors, buffer, n, h, w, c,
                  alpha=1, reverse=False):
        self.function(image, vertices, triangles, colors, buffer, n, h, w, c, alpha, reverse)

    def close(self):
        self.tmp.cleanup()
