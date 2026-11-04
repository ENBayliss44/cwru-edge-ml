"""Export int8 models to C headers and verify the firmware C against Python.

For each sample rate x model x leave-one-load-out fold:
  1. write firmware/edge_ml_bench/src/edge_ml/models/model_<fs>_<model>_<fold>.h
  2. write the fold's held-out windows + Python int8 inputs as test vectors
  3. compile firmware/host/host_main.c with that header (clang -O2) and run it
  4. check (a) integer inference bit-exact: C logits from Python's int8 inputs
     == Python logits, and (b) end-to-end raw-window -> class agreement.
Results: results/c_verification.json
"""
import json
import subprocess

import numpy as np

import data
import experiments as ex
import features
from quantize import evaluate

ROOT = data.ROOT
FW = ROOT / "firmware"
MODELS = FW / "edge_ml_bench" / "src" / "edge_ml" / "models"
BUILD = ROOT / "build"
F, C = features.N_FEATURES, len(data.CLASSES)


def write_vectors(path, W):
    with open(path, "wb") as f:
        np.array([len(W), W.shape[1]], np.int32).tofile(f)
        W.astype(np.float32).tofile(f)


def read_out(path, n):
    rec = np.dtype([("feat", np.float32, F), ("zq", np.int8, F),
                    ("logits", np.int32, C), ("cls", np.int32),
                    ("logits_ref", np.int32, C), ("cls_ref", np.int32)])
    return np.fromfile(path, dtype=rec, count=n)


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(exist_ok=True)
    report = {}
    for fs in (1666, 6664):
        _, fitted = evaluate(fs)
        ex.CFG["fs"] = fs
        nfft = 1 << (fs - 1).bit_length()
        for (fold, m), (pipe, q, te) in fitted.items():
            tag = f"{fs}_{m}_{fold}"
            Wte = np.concatenate([data.windows(r["x"], fs, fs) for r in te])
            yte = np.concatenate([[r["y"]] * (1 + (len(r["x"]) - fs) // fs) for r in te])
            Xte = features.extract(Wte, nfft)
            zq_py = q.quant_input(Xte)
            logits_py = q.logits_from_q(zq_py)
            cls_py = np.argmax(logits_py, axis=1)

            hdr = MODELS / f"model_{tag}.h"
            hdr.write_text(q.to_c_header(tag, fs, nfft, {
                "model": m, "fs_hz": fs, "fold_held_out": fold,
                "train": "CWRU 12k drive-end, loads != held-out",
            }))
            vec = BUILD / tag
            vec.mkdir(exist_ok=True)
            write_vectors(vec / "windows.bin", Wte)
            zq_py.tofile(vec / "zq_ref.bin")
            np.savez(vec / "python_ref.npz", y=yte, X=Xte, zq=zq_py,
                     logits=logits_py, cls=cls_py)

            exe = vec / "host_main"
            subprocess.run(["clang", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                            f"-DEM_MODEL_HEADER=\"models/model_{tag}.h\"",
                            "-I", str(FW / "edge_ml_bench" / "src" / "edge_ml"),
                            str(FW / "host" / "host_main.c"),
                            str(FW / "edge_ml_bench" / "src" / "edge_ml" / "edge_ml.c"),
                            "-lm", "-o", str(exe)], check=True)
            subprocess.run([str(exe), str(vec / "windows.bin"), str(vec / "zq_ref.bin"),
                            str(vec / "c_out.bin")], check=True, capture_output=True)
            o = read_out(vec / "c_out.bin", len(Wte))

            rel = np.abs(o["feat"] - Xte) / np.maximum(np.abs(Xte), 1e-3)
            report[tag] = {
                "n": int(len(Wte)),
                "int_inference_bit_exact": bool(np.array_equal(o["logits_ref"], logits_py)),
                "c_vs_python_class_agreement": float((o["cls"] == cls_py).mean()),
                "zq_mismatch_fraction": float((o["zq"] != zq_py).mean()),
                "zq_max_abs_diff": int(np.abs(o["zq"].astype(int) - zq_py).max()),
                "feature_max_rel_err": float(rel.max()),
                "c_acc": float((o["cls"] == yte).mean()),
                "python_int8_acc": float((cls_py == yte).mean()),
            }
            r = report[tag]
            print(f"{tag:<22} n={r['n']:<4} bit_exact={r['int_inference_bit_exact']} "
                  f"class_agree={r['c_vs_python_class_agreement']:.3f} "
                  f"zq_mismatch={r['zq_mismatch_fraction']:.4f} (max {r['zq_max_abs_diff']}) "
                  f"feat_rel_err={r['feature_max_rel_err']:.1e} "
                  f"acc C={r['c_acc']:.3f} py={r['python_int8_acc']:.3f}", flush=True)
    ex.CFG["fs"] = data.FS_TARGET
    (ROOT / "results" / "c_verification.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
