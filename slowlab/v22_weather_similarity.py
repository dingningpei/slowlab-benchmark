"""Outcome-blind exact/near replay audit for paired multichannel weather windows."""
from __future__ import annotations

import numpy as np


def nearest_window_distances(generated, source, scale, *, rows: int, stride: int,
                             chunk_size: int = 16, source_stride: int | None = None):
    """Return each generated window's minimum standardized RMSE to source windows.

    The source and generated arrays have shape (time, channels). We compare
    every source window at ``source_stride`` spacing in bounded generated
    chunks, without crop or agent outcomes. By default source stride equals
    generated stride, preserving the original aligned-window audit. Set
    ``source_stride=1`` for an exhaustive source-offset audit on a small slice.
    ``scale`` must have been fitted on development weather only.
    """
    generated = np.asarray(generated, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    source_stride = stride if source_stride is None else source_stride
    if (generated.ndim != 2 or source.ndim != 2 or generated.shape[1] != source.shape[1]
            or scale.shape != (generated.shape[1],) or rows <= 0 or stride <= 0
            or source_stride <= 0
            or chunk_size <= 0 or len(generated) < rows or len(source) < rows
            or not np.isfinite(generated).all() or not np.isfinite(source).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError('invalid weather similarity inputs')
    g_starts = range(0, len(generated) - rows + 1, stride)
    s_starts = range(0, len(source) - rows + 1, source_stride)
    s = np.stack([(source[i:i + rows] / scale).ravel() for i in s_starts])
    s_sq = np.sum(s * s, axis=1)
    answers = []
    dimension = rows * generated.shape[1]
    for offset in range(0, len(g_starts), chunk_size):
        starts = list(g_starts[offset:offset + chunk_size])
        g = np.stack([(generated[i:i + rows] / scale).ravel() for i in starts])
        g_sq = np.sum(g * g, axis=1)
        distances_sq = np.maximum(g_sq[:, None] + s_sq[None, :] - 2 * g @ s.T, 0.0)
        answers.extend(np.sqrt(np.min(distances_sq, axis=1) / dimension).tolist())
    return np.asarray(answers)


def audit_nonreplay(generated, sources, scale, *, threshold: float = 0.08):
    """Check 24h and 6h windows against every archived source path."""
    if not 0 < threshold < 1 or not sources:
        raise ValueError('invalid preregistered similarity threshold or empty source set')
    minima = {}
    for rows, label in ((144, '24h'), (36, '6h')):
        nearest = None
        for year, source in sorted(sources.items()):
            distances = nearest_window_distances(generated, source, scale,
                                                 rows=rows, stride=rows)
            nearest = distances if nearest is None else np.minimum(nearest, distances)
        minima[label] = {'minimum_standardized_rmse': float(np.min(nearest)),
                         'near_duplicate_windows': int(np.sum(nearest < threshold)),
                         'windows': len(nearest)}
    return {'passed': all(row['near_duplicate_windows'] == 0 for row in minima.values()),
            'threshold': threshold, 'by_window': minima}
