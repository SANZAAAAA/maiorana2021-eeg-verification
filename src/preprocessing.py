"""Signal-processing pipeline of Maiorana (2021), Section 3.1.

Steps, in the order reported in the paper:

    1. band-pass filtering to the alpha-beta sub-band [8, 30] Hz;
    2. downsampling to S = 64 Hz (Nyquist-Shannon, safe after step 1);
    3. common average reference (CAR) spatial filtering;
    4. segmentation into H = 5 s frames, with an 80 % overlap between
       consecutive frames.

A frame therefore contains C sequences of ``S * H = 320`` samples, one per
electrode, exactly as required by the CNN input of Table 1.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def bandpass(x: np.ndarray, fs_in: float, band=(8.0, 30.0), order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter (step 1)."""
    nyq = 0.5 * fs_in
    low, high = band[0] / nyq, band[1] / nyq
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def downsample(x: np.ndarray, fs_in: float, fs_out: float = 64.0) -> np.ndarray:
    """Polyphase resampling 160 Hz -> 64 Hz (step 2)."""
    if fs_in == fs_out:
        return x.astype(np.float32)
    # resample_poly works with rational up/down ratios (160 -> 64 = 5/2).
    from math import gcd

    a, b = int(round(fs_out)), int(round(fs_in))
    g = gcd(a, b)
    up, down = a // g, b // g
    return signal.resample_poly(x, up, down, axis=-1).astype(np.float32)


def common_average_reference(x: np.ndarray) -> np.ndarray:
    """CAR: subtract, at each time instant, the mean across electrodes (step 3)."""
    return (x - x.mean(axis=0, keepdims=True)).astype(np.float32)


def preprocess_stream(x: np.ndarray, fs_in: float, fs_out: float = 64.0, band=(8.0, 30.0)):
    """Apply steps 1-3 to a continuous multi-channel stream.

    Parameters
    ----------
    x : (n_channels, n_samples) array at ``fs_in`` Hz.

    Returns
    -------
    (n_channels, n_samples') array at ``fs_out`` Hz, CAR-referenced.
    """
    x = np.asarray(x, dtype=np.float64)
    x = bandpass(x, fs_in, band=band)
    x = downsample(x, fs_in, fs_out)
    x = common_average_reference(x)
    return x


def frame_stream(x: np.ndarray, fs: float, win_s: float = 5.0, overlap: float = 0.8):
    """Segment a continuous stream into frames (step 4).

    Parameters
    ----------
    x : (n_channels, n_samples) array at ``fs`` Hz.
    win_s : frame length in seconds (H = 5 s).
    overlap : fraction of overlap between consecutive frames (0.8 -> 80 %).

    Returns
    -------
    frames : (n_frames, n_channels, win_len) array, C-order.

    Notes
    -----
    The paper uses an 80 % overlap, i.e. a hop of ``0.2 * win_len`` samples.
    """
    win = int(round(win_s * fs))
    hop = int(round(win * (1.0 - overlap)))
    hop = max(hop, 1)
    n_channels, n_samples = x.shape
    if n_samples < win:
        return np.empty((0, n_channels, win), dtype=np.float32)
    starts = range(0, n_samples - win + 1, hop)
    frames = np.stack([x[:, s : s + win] for s in starts], axis=0)
    return frames.astype(np.float32)
