"""Hand-crafted representations used as the paper's comparison baseline.

The 2021 article compares its deep representations against a "fusion of AR and
MFCC features as in [17]" (Table 3a / Table 4a), where [17] is Maiorana &
Campisi, *Longitudinal evaluation of EEG-based biometric recognition*, IEEE
TIFS 13(5), 2018.

The 2021 paper does not restate the exact AR order / MFCC configuration, so
this module implements an equivalent hand-crafted representation:

    * **AR coefficients** of order ``P = 8`` per frame and per channel,
      estimated with the Yule-Walker equations (autocorrelation + Toeplitz
      solve);
    * **MFCC-style cepstral coefficients** (``N = 13``) obtained from a
      mel-spaced filter-bank over the retained [8, 30] Hz band, followed by a
      log compression and a DCT-II.

Each frame/channel is therefore described by a 21-dimensional vector, and the
same one-class-SVM verification scheme of the deep system is applied.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, rfft

AR_ORDER = 8
N_MFCC = 13
N_FFT = 512
N_MEL_FILTERS = 20
MEL_BAND = (8.0, 30.0)


def ar_coefficients(frame: np.ndarray, order: int = AR_ORDER) -> np.ndarray:
    """Yule-Walker AR coefficients of a single 1-D frame."""
    x = np.asarray(frame, dtype=float)
    x = x - x.mean()
    n = len(x)
    # biased autocorrelation is enough for stable AR estimation
    r = np.correlate(x, x, mode="full")[n - 1:] / n
    if r[0] <= 0:
        return np.zeros(order)
    R = np.empty((order, order))
    for i in range(order):
        R[i] = r[np.abs(np.arange(order) - i)]
    try:
        a = np.linalg.solve(R, r[1:order + 1])
    except np.linalg.LinAlgError:
        a = np.zeros(order)
    return a


def _mel_filterbank(fs: float, n_fft: int, n_filters: int,
                    fmin: float, fmax: float) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_filters + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / fs).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        if centre == left:
            centre = left + 1
        if right == centre:
            right = centre + 1
        for k in range(left, centre):
            fb[m - 1, k] = (k - left) / (centre - left)
        for k in range(centre, right):
            fb[m - 1, k] = (right - k) / (right - centre)
    return fb


_FB_CACHE: dict = {}


def mfcc_coefficients(frame: np.ndarray, fs: float = 64.0,
                      n_mfcc: int = N_MFCC) -> np.ndarray:
    """Mel-frequency cepstral coefficients of a single 1-D frame."""
    key = (fs, N_FFT, N_MEL_FILTERS, MEL_BAND)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _mel_filterbank(fs, N_FFT, N_MEL_FILTERS, *MEL_BAND)
    fb = _FB_CACHE[key]
    x = np.asarray(frame, dtype=float)
    x = x - x.mean()
    x = x * np.hanning(len(x))
    spec = np.abs(rfft(x, n=N_FFT)) ** 2
    energies = fb @ spec
    log_e = np.log(energies + 1e-12)
    ceps = dct(log_e, type=2, norm="ortho")
    return ceps[:n_mfcc]


def handcrafted_features(frames_ch: np.ndarray, fs: float = 64.0) -> np.ndarray:
    """Concatenate AR and MFCC features for ``(N, 320)`` single-channel frames."""
    frames_ch = np.asarray(frames_ch, dtype=float)
    out = np.empty((len(frames_ch), AR_ORDER + N_MFCC), dtype=np.float32)
    for i, frame in enumerate(frames_ch):
        out[i, :AR_ORDER] = ar_coefficients(frame)
        out[i, AR_ORDER:] = mfcc_coefficients(frame, fs=fs)
    return out
