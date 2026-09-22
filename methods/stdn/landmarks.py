"""M2 landmarks_px256 (iBUG-68 xy pixels) -> pinned STDN normalized xy."""
import ast
from pathlib import Path
import numpy as np

from methods.common.learned import PreparationError
from .source import validate_source


class LandmarkAdapter:
    def __init__(self, config, *, source_root=None):
        source = validate_source(config, source_root=source_root)
        tree = ast.parse((Path(source['root']) / 'model/dataset.py').read_text())
        lists = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'lm_reverse_list' for t in node.targets):
                # Exact pinned expression np.array([one-based indices]) - 1.
                expr = node.value
                if not (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Sub)
                        and ast.literal_eval(expr.right) == 1 and isinstance(expr.left, ast.Call)):
                    raise PreparationError('unrecognized pinned landmark permutation expression')
                lists.append(np.array(ast.literal_eval(expr.left.args[0]), dtype=np.int64) - 1)
        if not lists or any(not np.array_equal(x, lists[0]) for x in lists):
            raise PreparationError('pinned train/VAL flip permutations missing or disagree')
        self.permutation = lists[0]
        if not np.array_equal(np.sort(self.permutation), np.arange(68)):
            raise PreparationError('source permutation is not iBUG-68')
        if not np.array_equal(self.permutation[self.permutation], np.arange(68)):
            raise PreparationError('source permutation is not an involution')
        self.permutation.setflags(write=False)
        self.width = config['training']['input_resolution']

    @staticmethod
    def _points(points):
        points = np.asarray(points, dtype=np.float32)
        if points.shape != (68, 2) or not np.isfinite(points).all():
            raise ValueError('landmarks must be finite 68x2 xy coordinates')
        return points.copy()

    def flip_normalized(self, points):
        result = self._points(points)
        # Deliberately 1-x, NOT (width-1-x)/width: upstream uses this convention.
        result[:, 0] = 1 - result[:, 0]
        return result[self.permutation]

    def from_cache(self, geometry, *, flip=False):
        """Accept only the canonical pixel-space field; never guess coordinate units."""
        if 'landmarks_px256' not in geometry:
            raise ValueError('M2 landmarks_px256 is required; normalized detector output is not interchangeable')
        points = self._points(geometry['landmarks_px256']) / self.width
        return self.flip_normalized(points) if flip else points
