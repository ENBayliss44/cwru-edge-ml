"""Hardware checks for the LSM6DSOX SPI + FIFO path.

usage:
  python tools/sensor_check.py <port> selftest        # achieved ODR, overruns, gaps
  python tools/sensor_check.py <port> live [seconds]  # log live windows to CSV

selftest passes when: |measured - expected| / expected < 0.5 % (expected from
the sensor's INTERNAL_FREQ_FINE trim), and overruns, tag gaps, unexpected FIFO
words and bus errors are all 0. Writes results/sensor_selftest.json or
results/live_<timestamp>.csv.
"""
import csv
import json
import pathlib
import sys
import time

import serial

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASSES = ["normal", "inner", "outer", "ball"]


def open_port(port):
    ser = serial.Serial(port, 921600, timeout=2)
    time.sleep(2.0)
    ser.reset_input_buffer()
    return ser


def wait_for(ser, prefixes, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = ser.readline().decode(errors="replace").strip()
        if line[:2] in prefixes:
            return line
    raise TimeoutError(f"no {prefixes} line in {timeout} s")


def selftest(ser):
    ser.write(b"S")
    line = wait_for(ser, ("S,", "E,"), 30)
    if line.startswith("E,"):
        sys.exit(f"device error: {line}")
    f = line.split(",")[1:]
    r = dict(zip(["measured_hz", "expected_hz", "freq_fine", "samples", "overruns",
                  "tag_gaps", "other_words", "bus_errors"],
                 [float(f[0]), float(f[1])] + [int(v) for v in f[2:]]))
    r["rate_error_pct"] = 100 * (r["measured_hz"] - r["expected_hz"]) / r["expected_hz"]
    r["pass"] = (abs(r["rate_error_pct"]) < 0.5 and r["overruns"] == 0 and
                 r["tag_gaps"] == 0 and r["other_words"] == 0 and r["bus_errors"] == 0)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "sensor_selftest.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    return r["pass"]


def live(ser, seconds):
    out = ROOT / "results" / f"live_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out.parent.mkdir(exist_ok=True)
    cols = ["cls", "l0", "l1", "l2", "l3", "t_total_us", "rms_g", "window_hz",
            "overruns", "tag_gaps", "dropped"]
    ser.write(b"L")
    t0 = time.time()
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["host_time_s", "label"] + cols)
        while time.time() - t0 < seconds:
            line = ser.readline().decode(errors="replace").strip()
            if line.startswith("E,"):
                sys.exit(f"device error: {line}")
            if not line.startswith("L,"):
                continue
            f = line.split(",")[1:]
            w.writerow([f"{time.time() - t0:.2f}", CLASSES[int(f[0])]] + f)
            print(f"{CLASSES[int(f[0])]:<7} {f[5]:>6} us  rms={f[6]} g  "
                  f"{f[7]} Hz  ovr={f[8]} gaps={f[9]} dropped={f[10]}")
    ser.write(b"x")
    print(f"wrote {out}")


if __name__ == "__main__":
    port, mode = sys.argv[1], sys.argv[2]
    ser = open_port(port)
    if mode == "selftest":
        sys.exit(0 if selftest(ser) else 1)
    live(ser, float(sys.argv[3]) if len(sys.argv) > 3 else 30)
