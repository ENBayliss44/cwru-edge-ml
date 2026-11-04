"""Numbers quoted in README.md: int8 leave-one-load-out results with counts,
95 % Wilson intervals and confusion matrices. Writes results/summary.json."""
import json
import math

import numpy as np

import data
import experiments as ex
from quantize import evaluate


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    out = {}
    for fs in (1666, 6664):
        _, fitted = evaluate(fs)
        for m in ("logreg", "mlp"):
            ys, ps = [], []
            for (fold, mm), (pipe, q, te) in fitted.items():
                if mm != m:
                    continue

                ex.CFG["fs"] = fs
                X, y, _ = ex.featurize(te, ex.TEST)
                ys.append(y); ps.append(q.predict(X))
            ex.CFG["fs"] = data.FS_TARGET
            y, p = np.concatenate(ys), np.concatenate(ps)
            cm = np.zeros((4, 4), int)
            for a, b in zip(y, p):
                cm[a, b] += 1
            k_acc, n = int((y == p).sum()), len(y)
            k_fa, n_norm = int((p[y == 0] != 0).sum()), int((y == 0).sum())
            k_det, n_fault = int((p[y != 0] != 0).sum()), int((y != 0).sum())
            out[f"{fs}_{m}"] = {
                "accuracy": [k_acc, n, *wilson(k_acc, n)],
                "false_alarms": [k_fa, n_norm, *wilson(k_fa, n_norm)],
                "fault_detected": [k_det, n_fault, *wilson(k_det, n_fault)],
                "confusion_rows_true_cols_pred": cm.tolist(),
                "classes": data.CLASSES,
            }
            r = out[f"{fs}_{m}"]
            print(f"{fs} {m:<7} acc {k_acc}/{n} [{r['accuracy'][2]:.2f},{r['accuracy'][3]:.2f}]  "
                  f"FA {k_fa}/{n_norm} [{r['false_alarms'][2]:.2f},{r['false_alarms'][3]:.2f}]  "
                  f"det {k_det}/{n_fault} [{r['fault_detected'][2]:.3f},{r['fault_detected'][3]:.3f}]")
            print("   ", cm.tolist())
    (data.ROOT / "results" / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
