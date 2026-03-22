import numpy as np
from scipy import signal


def pixel_to_series_exp(pixel, zero_mv, length):
    """Convert segmentation mask to pixel y-positions via weighted average.

    Args:
        pixel: (4, H, W) mask predictions
        zero_mv: not used in computation, kept for API compat
        length: target length after resampling (None to skip)

    Returns:
        series: (4, length) pixel y-positions
    """
    _, H, W = pixel.shape
    eps = 1e-8
    y_idx = np.arange(H, dtype=np.float32)[:, None]

    series = []
    for j in range(4):
        p = pixel[j]
        denom = p.sum(axis=0)
        y_exp = (p * y_idx).sum(axis=0) / (denom + eps)
        series.append(y_exp)
    series = np.stack(series).astype(np.float32)

    if length is not None and length != W:
        resampled = []
        for s in series:
            rs = signal.resample(s, length).astype(np.float32)
            resampled.append(rs)
        series = np.stack(resampled)

    return series