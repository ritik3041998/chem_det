"""Test on our Raman data -> outputs/results.json, predictions.csv, confusion_matrix.png, model.pt

1. Cross-instrument check: our scans of chemicals that also exist in the public library are matched
   against the public library only (the model has never seen our instrument).
2. Leave-sessions-out cross-validation on our labelled scans. The reference library is the public
   library (thousands of classes, as distractors) + our scans from the *other* sessions.
   Compared: cosine similarity (classic hit-quality index), the public-pretrained matcher, and the
   matcher fine-tuned on our training sessions.
3. Final model fine-tuned on all our labelled scans, and what it makes of the 'not in lib' scans.
"""
import argparse
import csv
import json
from collections import Counter

import numpy as np
import torch

from common import ALIASES, GRID, OUT_DIR, UNKNOWN, cap_per_class, load_ours, load_public
from model import Siamese
from train import DEV, PairSampler, train


def cosine_scores(xq, xr):
    a = xq / np.linalg.norm(xq, axis=1, keepdims=True).clip(1e-9)
    b = xr / np.linalg.norm(xr, axis=1, keepdims=True).clip(1e-9)
    return a @ b.T


def ranked_classes(scores, ref_label, k=5):
    """Best k distinct substances per query, with the score of each one's closest reference."""
    out = []
    for s in scores:
        best = {}
        for j in np.argsort(-s)[:300]:
            if ref_label[j] not in best:
                best[ref_label[j]] = float(s[j])
                if len(best) == k:
                    break
        out.append(list(best.items()))
    return out


def summarise(y, ranked):
    top1 = np.array([r[0][0] for r in ranked])
    hit1 = top1 == y
    hit5 = np.array([y[i] in [n for n, _ in r] for i, r in enumerate(ranked)])
    per_class = {c: float(hit1[y == c].mean()) for c in np.unique(y)}
    return {"n": int(len(y)), "top1": float(hit1.mean()), "top5": float(hit5.mean()),
            "balanced_top1": float(np.mean(list(per_class.values()))), "per_class": per_class}, top1


def load_model(state):
    m = Siamese(len(GRID)).to(DEV)
    m.load_state_dict(state)
    return m


def session_folds(label, session, n_folds, seed=0):
    """Whole sessions go to one fold; every tested class keeps >= 1 other session in the library."""
    rng = np.random.default_rng(seed)
    fold = np.full(len(label), -1)
    for c in np.unique(label):
        sess = rng.permutation(np.unique(session[label == c]))
        if len(sess) < 2:
            continue  # single session: stays in the library, cannot be tested fairly
        for i, s in enumerate(sess):
            fold[(label == c) & (session == s)] = i % n_folds
    return fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ft_steps", type=int, default=400)
    ap.add_argument("--lib_per_class", type=int, default=6)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    xp_all, lp_all, _, _ = load_public()  # everything is used for fine-tuning
    lib = cap_per_class(lp_all, args.lib_per_class)  # a few references per class are enough for matching
    xp, lp = xp_all[lib], lp_all[lib]
    xo, lo, so, row = load_ours()
    ck = torch.load(OUT_DIR / "pretrained.pt", weights_only=False)
    pre = load_model(ck["state"])
    known = lo != UNKNOWN
    results = {"public_library": {"spectra": int(len(xp)), "classes": int(len(np.unique(lp)))},
               "ours": {"scans": int(len(xo)), "labelled": int(known.sum()),
                        "chemicals": int(len(np.unique(lo[known])))}}

    # ---- 1. cross-instrument: ours vs public library only
    shared = sorted(set(ALIASES.values()) & set(lo) & set(lp) | (set(lo[known]) & set(lp)))
    print(f"\n[1] Cross-instrument check on {shared} (reference = public library only, "
          f"{len(np.unique(lp))} classes)")
    q = np.isin(lo, shared)
    results["cross_instrument"] = {}
    for name, sc in [("cosine", cosine_scores(xo[q], xp)), ("pretrained", pre.match(xo[q], xp).numpy())]:
        res, _ = summarise(lo[q], ranked_classes(sc, lp))
        results["cross_instrument"][name] = res
        print(f"  {name:11s} top-1 {res['top1']:.3f}  top-5 {res['top5']:.3f}  (n={res['n']})  "
              + "  ".join(f"{c}: {a:.2f}" for c, a in res["per_class"].items()))

    # ---- 2. leave-sessions-out CV on our labelled scans
    xk, lk, sk, rk = xo[known], lo[known], so[known], row[known]
    fold = session_folds(lk, sk, args.folds)
    tested = fold >= 0
    untestable = sorted(set(lk[~tested]))
    print(f"\n[2] Leave-sessions-out CV: {int(tested.sum())} scans of {len(set(lk[tested]))} chemicals tested; "
          f"single-session chemicals kept only as library entries: {untestable}")
    ranked = {m: [None] * len(lk) for m in ("cosine", "pretrained", "finetuned")}
    for f in range(args.folds):
        te = fold == f
        tr = ~te
        xr, lr = np.concatenate([xp, xk[tr]]), np.concatenate([lp, lk[tr]])
        print(f" fold {f}: test {int(te.sum())} scans, library {len(xr)} spectra / {len(np.unique(lr))} classes")
        torch.manual_seed(f)
        ft = train([PairSampler(xk[tr], lk[tr], seed=f), PairSampler(xp_all, lp_all, seed=f)], args.ft_steps,
                   lr=5e-5, init=ck["state"], log_every=args.ft_steps)
        for name, sc in [("cosine", cosine_scores(xk[te], xr)), ("pretrained", pre.match(xk[te], xr).numpy()),
                         ("finetuned", ft.match(xk[te], xr).numpy())]:
            for i, r in zip(np.flatnonzero(te), ranked_classes(sc, lr)):
                ranked[name][i] = r
        del ft
    results["cv"] = {"folds": args.folds, "untestable_single_session": untestable}
    idx = np.flatnonzero(tested)
    preds = {}
    print(f"\n  {'method':11s} top-1   top-5   balanced top-1")
    for name in ranked:
        res, preds[name] = summarise(lk[idx], [ranked[name][i] for i in idx])
        results["cv"][name] = res
        print(f"  {name:11s} {res['top1']:.3f}   {res['top5']:.3f}   {res['balanced_top1']:.3f}")
    counts = Counter(lk[idx])
    print(f"\n  per-chemical top-1 (scans)        cosine  pretrained  finetuned")
    for c, n in counts.most_common():
        print(f"  {c:22s} {n:4d}       " + "      ".join(f"{results['cv'][m]['per_class'][c]:.2f}" for m in ranked))

    with open(OUT_DIR / "predictions.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["raman_row", "true", "cosine", "pretrained", "finetuned", "finetuned_score", "finetuned_top5"])
        for j, i in enumerate(idx):
            r = ranked["finetuned"][i]
            w.writerow([int(rk[i]), lk[i], preds["cosine"][j], preds["pretrained"][j], preds["finetuned"][j],
                        round(r[0][1], 2), " | ".join(n for n, _ in r)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    true_c = [c for c, _ in counts.most_common()]
    pred_c = true_c + sorted(set(preds["finetuned"]) - set(true_c))
    cm = np.array([[np.sum((lk[idx] == t) & (preds["finetuned"] == p)) for p in pred_c] for t in true_c])
    fig, ax = plt.subplots(figsize=(0.45 * len(pred_c) + 3, 0.4 * len(true_c) + 3))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for a in range(len(true_c)):
        for b in range(len(pred_c)):
            if cm[a, b]:
                ax.text(b, a, cm[a, b], ha="center", va="center", fontsize=7,
                        color="white" if cm[a, b] / cm[a].sum() > 0.5 else "black")
    ax.set_xticks(range(len(pred_c)), pred_c, rotation=90, fontsize=8)
    ax.set_yticks(range(len(true_c)), true_c, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (instrument label)")
    ax.set_title(f"Fine-tuned matcher, leave-sessions-out CV (top-1 {results['cv']['finetuned']['top1']:.1%})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)

    # ---- 3. final model on all labelled scans + the 'not in lib' scans
    print(f"\n[3] Final model: fine-tuning on all {len(xk)} labelled scans")
    torch.manual_seed(0)
    final = train([PairSampler(xk, lk), PairSampler(xp_all, lp_all)], args.ft_steps, lr=5e-5, init=ck["state"],
                  log_every=args.ft_steps)
    xlib, llib = np.concatenate([xp, xk]), np.concatenate([lp, lk])
    torch.save({"state": final.state_dict()}, OUT_DIR / "model.pt")
    np.savez_compressed(OUT_DIR / "library.npz", x=xlib.astype(np.float16), label=llib)
    if (~known).any():
        r = ranked_classes(final.match(xo[~known], xlib).numpy(), llib, k=1)
        guess = Counter(x[0][0] for x in r)
        results["not_in_lib"] = {"n": int((~known).sum()), "best_match_counts": dict(guess.most_common())}
        print(f"  best match for the {int((~known).sum())} '{UNKNOWN}' scans: {guess.most_common(8)}")
    json.dump(results, open(OUT_DIR / "results.json", "w"), indent=2)
    print(f"\nsaved results.json, predictions.csv, confusion_matrix.png, model.pt, library.npz in {OUT_DIR}")


if __name__ == "__main__":
    main()
