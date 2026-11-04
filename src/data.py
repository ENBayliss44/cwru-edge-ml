"""CWRU loading, resampling to the LSM6DSOX rate, and windowing.

Sample-rate note: the CWRU "normal baseline" files (97-100) are sampled at
48 kHz, not 12 kHz. Evidence (see notebooks/README): read at 48 kHz, the
normal files show the same speed-locked rig peaks (5.40x and 6.40x shaft) at the
same absolute frequency as the 12 kHz fault files (155.27 Hz in both 100.mat and
133.mat at 1725 rpm), and their spectral lines land on mains harmonics. Normal
files are therefore decimated 48k -> 12k first, then follow the identical
12k -> 1666 Hz path as the fault files.
"""
import csv
from math import gcd
import pathlib

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

ROOT = pathlib.Path(__file__).resolve().parents[1]
FS_TARGET = 1666            # LSM6DSOX ODR used by the EE284A firmware
WINDOW = 1666               # ~1 s, matches WINDOW_SAMPLES_TARGET in firmware
CLASSES = ["normal", "inner", "outer", "ball"]


def manifest():
    with open(ROOT / "data" / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["file_id"] = int(r["file_id"])
        r["load_hp"] = int(r["load_hp"])
        r["fault_size_in"] = float(r["fault_size_in"])
        r["fs_native"] = 48000 if r["label"] == "normal" else 12000
        r["y"] = CLASSES.index(r["label"])
    return rows


def load_de(file_id):
    """Drive-end signal in g. Some files contain other recordings' keys
    (e.g. 99.mat also holds X098_DE_time), so match the file id exactly."""
    m = sio.loadmat(ROOT / "data" / "raw" / f"{file_id}.mat")
    key = f"X{file_id:03d}_DE_time"
    return m[key].ravel().astype(np.float64)


def to_target_rate(x, fs_native, fs=FS_TARGET):
    """All recordings pass through 12 kHz, then a common 12k -> fs path."""
    if fs_native == 48000:
        x = resample_poly(x, 1, 4)          # 48k -> 12k
    elif fs_native != 12000:
        raise ValueError(fs_native)
    if fs == 12000:
        return x
    g = gcd(fs, 12000)
    return resample_poly(x, fs // g, 12000 // g)


def windows(x, hop, n_win=WINDOW):
    n = 1 + (len(x) - n_win) // hop
    idx = np.arange(n_win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def load_all(fs=FS_TARGET, cache=True):
    """Return list of recordings: dict(meta..., x=signal at fs)."""
    cache_path = ROOT / "data" / f"resampled_{fs}.npz"
    rows = manifest()
    if cache and cache_path.exists():
        z = np.load(cache_path)
        for r in rows:
            r["x"] = z[str(r["file_id"])]
        return rows
    for r in rows:
        r["x"] = to_target_rate(load_de(r["file_id"]), r["fs_native"], fs)
    if cache:
        np.savez(cache_path, **{str(r["file_id"]): r["x"] for r in rows})
    return rows
