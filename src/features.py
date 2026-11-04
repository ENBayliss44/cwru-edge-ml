"""Per-window features. Written to mirror firmware/edge_ml_bench/src/edge_ml/edge_ml.c:
same order, same definitions, float32 inputs.

Features (23):
  0 log10_rms        1 log10_p2p        2 crest           3 kurtosis
  4 skewness         5 impulse          6 shape
  7-22: 16 log10 band powers of a Hann-windowed, zero-padded 2048-pt FFT,
  each band 64 bins (52 Hz) wide over 0-833 Hz (DC bin excluded).
"""
import numpy as np

NFFT = 2048
N_BANDS = 16
BINS_PER_BAND = 64
EPS = 1e-12

TIME_NAMES = ["log10_rms", "log10_p2p", "crest", "kurtosis",
              "skewness", "impulse", "shape"]
BAND_NAMES = [f"band_{int(b * BINS_PER_BAND * 1666 / NFFT)}_"
              f"{int((b + 1) * BINS_PER_BAND * 1666 / NFFT)}Hz"
              for b in range(N_BANDS)]
NAMES = TIME_NAMES + BAND_NAMES
N_FEATURES = len(NAMES)


def hann(n):
    k = np.arange(n, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(2 * np.pi * k / (n - 1))).astype(np.float32)


def extract(w, nfft=NFFT):
    """w: (n_windows, N) array in g. Returns (n_windows, 23) float32.
    nfft: FFT length (2048 for the 1666 Hz deployment; larger only for the
    sample-rate ablation, where bands still split 0..fs/2 into 16 equal parts)."""
    w = np.asarray(w, dtype=np.float32)
    bpb = nfft // 2 // N_BANDS
    w = w - w.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(w * w, axis=1))
    mx, mn = w.max(axis=1), w.min(axis=1)
    peak = np.maximum(mx, -mn)
    mabs = np.mean(np.abs(w), axis=1)
    m3 = np.mean(w ** 3, axis=1)
    m4 = np.mean(w ** 4, axis=1)
    time = np.stack([
        np.log10(rms + EPS),
        np.log10(mx - mn + EPS),
        peak / rms,
        m4 / rms ** 4,
        m3 / rms ** 3,
        peak / mabs,
        rms / mabs,
    ], axis=1)

    spec = np.fft.rfft(w * hann(w.shape[1]), n=nfft, axis=1)
    p = (spec.real ** 2 + spec.imag ** 2)[:, :N_BANDS * bpb]
    p[:, 0] = 0.0                                     # drop DC
    bands = p.reshape(len(w), N_BANDS, bpb).sum(axis=2)
    return np.concatenate([time, np.log10(bands + EPS)], axis=1).astype(np.float32)
