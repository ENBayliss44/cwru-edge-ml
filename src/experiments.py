"""Leakage-safe evaluation of threshold baseline vs logistic regression vs MLP.

Protocols
  random  : pooled windows, random 70/30 split.  LEAKY - shown only to quantify
            how much window-level splitting inflates CWRU results.
  lolo    : leave-one-load-out. Train on 3 motor loads, test on the 4th.
  lolo_ls : leave-one-load-AND-fault-size-out. Train on loads != L and fault
            sizes != s; test on load L, size s (+ normal at load L). 12 folds.
No recording ever contributes windows to both train and test in lolo / lolo_ls.

Hyperparameters are fixed a priori (not tuned on test folds).
Train windows: 75 % overlap (hop 416). Test windows: non-overlapping.
"""
import json
import pathlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data
import features

ROOT = data.ROOT
SEED = 0
# Sample rate under evaluation. 1666 Hz = deployed hardware; others = ablation.
# Window is always 1 s (fs samples); train hop fs/4 (75 % overlap); test hop fs.
CFG = {"fs": data.FS_TARGET}
TRAIN, TEST = "train", "test"


def featurize(recs, mode):
    fs = CFG["fs"]
    hop = fs // 4 if mode == TRAIN else fs
    nfft = 1 << (fs - 1).bit_length()
    X, y, rid = [], [], []
    for r in recs:
        F = features.extract(data.windows(r["x"], hop, fs), nfft)
        X.append(F)
        y += [r["y"]] * len(F)
        rid += [r["file_id"]] * len(F)
    return np.concatenate(X), np.array(y), np.array(rid)


def balance(X, y, rng):
    """Duplicate minority-class windows so every class has equal weight."""
    n = np.bincount(y).max()
    idx = np.concatenate([rng.choice(np.where(y == c)[0], n, replace=True)
                          for c in np.unique(y)])
    return X[idx], y[idx]


def make_models():
    return {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(C=1.0, max_iter=5000)),
        "mlp": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-3,
                                           max_iter=3000, random_state=SEED)),
    }


def threshold_baseline(Xtr, ytr, Xte, n_sigma=3.0):
    """EE284A detector: mean + 3 sigma of healthy-window RMS. Binary only."""
    rms_tr = 10 ** Xtr[ytr == 0, 0]
    thr = rms_tr.mean() + n_sigma * rms_tr.std()
    return (10 ** Xte[:, 0] > thr).astype(int), thr


def folds(recs, protocol):
    if protocol == "lolo":
        for L in range(4):
            yield f"load{L}", [r for r in recs if r["load_hp"] != L], \
                              [r for r in recs if r["load_hp"] == L]
    elif protocol == "lolo_ls":
        for L in range(4):
            for s in (0.007, 0.014, 0.021):
                tr = [r for r in recs if r["load_hp"] != L and
                      r["fault_size_in"] != s]
                te = [r for r in recs if r["load_hp"] == L and
                      r["fault_size_in"] in (0.0, s)]
                yield f"load{L}_size{s}", tr, te


def summarize(y, p, pbin=None):
    out = {
        "n": int(len(y)),
        "accuracy": float((y == p).mean()),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "recall": {c: float((p[y == i] == i).mean())
                   for i, c in enumerate(data.CLASSES)},
        "confusion": confusion_matrix(y, p, labels=range(4)).tolist(),
    }
    yb = (y != 0).astype(int)
    pb = (p != 0).astype(int) if pbin is None else pbin
    out["detect_tpr"] = float(pb[yb == 1].mean())
    out["false_alarm_rate"] = float(pb[yb == 0].mean())
    out["n_normal"] = int((yb == 0).sum())
    return out


def run_protocol(recs, protocol):
    rng = np.random.default_rng(SEED)
    preds = {m: [] for m in ["logreg", "mlp"]}
    thr_pred, ys, per_fold = [], [], []
    for name, tr, te in folds(recs, protocol):
        Xtr, ytr, _ = featurize(tr, TRAIN)
        Xte, yte, _ = featurize(te, TEST)
        Xb, yb = balance(Xtr, ytr, rng)
        ys.append(yte)
        tp, thr = threshold_baseline(*featurize([r for r in tr if r["y"] == 0],
                                                TEST)[:2], Xte)
        thr_pred.append(tp)
        fold = {"fold": name, "n_test": int(len(yte)), "rms_threshold_g": float(thr)}
        for m, model in make_models().items():
            model.fit(Xb, yb)
            p = model.predict(Xte)
            preds[m].append(p)
            fold[m + "_acc"] = float((p == yte).mean())
        per_fold.append(fold)
    y = np.concatenate(ys)
    res = {m: summarize(y, np.concatenate(p)) for m, p in preds.items()}
    tb = np.concatenate(thr_pred)
    res["threshold"] = {
        "detect_tpr": float(tb[y != 0].mean()),
        "false_alarm_rate": float(tb[y == 0].mean()),
        "n": int(len(y)), "n_normal": int((y == 0).sum()),
    }
    res["folds"] = per_fold
    return res


def run_random(recs):
    X, y, _ = featurize(recs, TRAIN)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=SEED, stratify=y)
    Xb, yb = balance(Xtr, ytr, np.random.default_rng(SEED))
    res = {}
    for m, model in make_models().items():
        model.fit(Xb, yb)
        res[m] = summarize(yte, model.predict(Xte))
    return res


def main():
    recs = data.load_all()
    results = {"random_LEAKY": run_random(recs)}
    for proto in ("lolo", "lolo_ls"):
        results[proto] = run_protocol(recs, proto)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1))

    print(f"{'protocol':<14}{'model':<10}{'acc':>7}{'macroF1':>9}"
          f"{'TPR':>7}{'FAR':>7}   recall n/i/o/b")
    for proto, res in results.items():
        for m in ("threshold", "logreg", "mlp"):
            if m not in res:
                continue
            r = res[m]
            acc = f"{r['accuracy']:.3f}" if "accuracy" in r else "  -  "
            f1 = f"{r['macro_f1']:.3f}" if "macro_f1" in r else "  -  "
            rec = " ".join(f"{v:.2f}" for v in r["recall"].values()) \
                if "recall" in r else ""
            print(f"{proto:<14}{m:<10}{acc:>7}{f1:>9}{r['detect_tpr']:>7.3f}"
                  f"{r['false_alarm_rate']:>7.3f}   {rec}")
    for proto in ("lolo", "lolo_ls"):
        print(f"\n{proto} per fold:")
        for f in results[proto]["folds"]:
            print(f"  {f['fold']:<18} n={f['n_test']:<4} thr={f['rms_threshold_g']:.4f}g"
                  f"  logreg={f['logreg_acc']:.3f}  mlp={f['mlp_acc']:.3f}")


def ablation(rates=(1666, 3332, 6664, 12000)):
    """Same pipeline, same models, different sample rate (leave-one-load-out)."""
    out = {}
    for fs in rates:
        CFG["fs"] = fs
        res = run_protocol(data.load_all(fs), "lolo")
        res2 = run_protocol(data.load_all(fs), "lolo_ls")
        out[fs] = {"lolo": res, "lolo_ls": res2}
        for proto, r in (("lolo", res), ("lolo_ls", res2)):
            print(f"fs={fs:>5} {proto:<8} thr TPR={r['threshold']['detect_tpr']:.3f} "
                  f"FAR={r['threshold']['false_alarm_rate']:.3f} | " +
                  " | ".join(f"{m} acc={r[m]['accuracy']:.3f} F1={r[m]['macro_f1']:.3f} "
                             f"FAR={r[m]['false_alarm_rate']:.3f}"
                             for m in ("logreg", "mlp")), flush=True)
    CFG["fs"] = data.FS_TARGET
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "ablation_fs.json").write_text(
        json.dumps({str(k): {p: {m: {kk: vv for kk, vv in v[p][m].items()
                                     if kk != "confusion"}
                                 for m in ("threshold", "logreg", "mlp")}
                             for p in v} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    import sys
    ablation() if "ablation" in sys.argv else main()
