"""Stream held-out CWRU windows to the ESP32 and compare with Python.

usage: python tools/stream_bench.py /dev/cu.usbserial-XXXX 6664_mlp_load3
Requires build/<tag>/ from src/export_verify.py and the matching model flashed
(see firmware/edge_ml_bench/src/edge_ml/model_config.h).
Writes results/device_<tag>.json.
"""
import json
import pathlib
import struct
import sys
import time

import numpy as np
import serial

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_line(ser, prefix, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = ser.readline().decode(errors="replace").strip()
        if line.startswith(prefix):
            return line.split(",")[1:]
    raise TimeoutError(f"no '{prefix}' line")


def main(port, tag):
    vec = ROOT / "build" / tag
    with open(vec / "windows.bin", "rb") as f:
        n, N = struct.unpack("<ii", f.read(8))
        W = np.frombuffer(f.read(), dtype="<f4").reshape(n, N)
    ref = np.load(vec / "python_ref.npz")

    ser = serial.Serial(port, 921600, timeout=2)
    time.sleep(2.0)                      # board resets on open
    ser.reset_input_buffer()
    ser.write(b"I")
    info = list(map(int, read_line(ser, "I")))
    fs, win = info[0], info[1]
    assert win == N, f"board model window {win} != vectors {N} (wrong model flashed?)"

    cls, logits, t_feat, t_inf, t_tot = [], [], [], [], []
    for i in range(n):
        ser.write(b"W" + W[i].astype("<f4").tobytes())
        r = list(map(int, read_line(ser, "R")))
        cls.append(r[0]); logits.append(r[1:5])
        t_feat.append(r[5]); t_inf.append(r[6]); t_tot.append(r[7])
    ser.write(b"I")
    info_after = list(map(int, read_line(ser, "I")))
    ser.close()

    cls, logits = np.array(cls), np.array(logits)
    y = ref["y"]

    def stats(t):
        t = np.array(t)
        return {"median_us": float(np.median(t)), "p99_us": float(np.percentile(t, 99)),
                "max_us": int(t.max()), "min_us": int(t.min())}

    out = {
        "tag": tag, "n_windows": int(n), "fs": fs,
        "nfft": info[2], "edge_ml_static_ram_bytes": info[3],
        "free_heap_bytes": info_after[4], "min_free_heap_bytes": info_after[5],
        "loop_stack_high_water_mark_bytes": info_after[6], "cpu_mhz": info[7],
        "latency_features": stats(t_feat), "latency_inference": stats(t_inf),
        "latency_total": stats(t_tot),
        "device_acc": float((cls == y).mean()),
        "python_int8_acc": float((ref["cls"] == y).mean()),
        "device_vs_python_class_agreement": float((cls == ref["cls"]).mean()),
        "device_logits_equal_python": float((logits == ref["logits"]).all(axis=1).mean()),
        "device_far": float((cls[y == 0] != 0).mean()),
        "device_tpr": float((cls[y != 0] != 0).mean()),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"device_{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
