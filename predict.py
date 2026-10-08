"""Identify the chemical in a Raman spectrum.

    python predict.py scan.csv                      # file with wavenumber + intensity
    python predict.py scan.npy --wn wn.npy          # intensity only, wavenumber axis in a second file
    python predict.py raman/spec_log.csv --wn raman/wnums_log.csv --row 10 --labels raman/chem_log.csv
    python predict.py --values "12.5, 14.1, 13.9, ..."

Accepted files: .csv / .txt / .tsv (comma, semicolon, tab or space separated, header lines are
skipped), .npy, .npz. Accepted layouts:
  - two columns (or two rows): wavenumber, intensity            -> one scan
  - one row / one column / 1-D array of intensities             -> one scan
  - many rows of intensities (like raman/spec_log.csv)          -> one scan per row
When no wavenumber axis is given, a scan of 3648 values is assumed to come from our instrument
(its usual axis is used) and a scan of 1600 values is assumed to already be on the 200-1799 grid.
Uses outputs/model.pt and outputs/library.npz written by evaluate.py.
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from common import GRID, OUT_DIR, RAMAN_DIR, normalise, resample
from evaluate import ranked_classes
from model import Siamese
from prepare_ours import RENAME


def read_array(path):
    """Numeric content of a csv/txt/npy/npz file as a float array (1-D or 2-D)."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if path.suffix.lower() == ".npz":
        d = np.load(path, allow_pickle=False)
        keys = list(d.keys())
        if {"wn", "spec"} <= set(keys):
            return np.stack([np.ravel(d["wn"]), np.ravel(d["spec"])], axis=1).astype(float)
        return np.asarray(d["x" if "x" in keys else keys[0]], dtype=float)
    rows = []
    for line in open(path, encoding="utf-8-sig", errors="replace"):
        try:
            vals = [float(v) for v in re.split(r"[,;\s]+", line.strip()) if v]
        except ValueError:
            continue  # header or comment line
        if vals:
            rows.append(vals)
    if not rows:
        sys.exit(f"no numbers found in {path}")
    if len({len(r) for r in rows}) > 1:
        sys.exit(f"{path}: rows have different lengths")
    return np.array(rows)


def default_axis(n):
    """Wavenumber axis to assume for a scan of n intensities given without one."""
    if n == len(GRID):
        return GRID
    cache = OUT_DIR / "instrument_axis.npy"
    if not cache.exists() and (RAMAN_DIR / "wnums_log.csv").exists():
        rows = [line.strip() for line in open(RAMAN_DIR / "wnums_log.csv")]
        np.save(cache, np.array(Counter(rows).most_common(1)[0][0].split(","), dtype=float))
    if cache.exists() and len(np.load(cache)) == n:
        return np.load(cache)
    sys.exit(f"the scan has {n} values and no wavenumber axis: pass one with --wn, "
             f"or give a file with two columns (wavenumber, intensity)")


def to_scans(a, wn):
    """-> list of (wavenumber, intensity) pairs."""
    a = np.squeeze(np.asarray(a, dtype=float))
    if wn is None and a.ndim == 2 and 2 in a.shape and max(a.shape) > 2:
        a = a if a.shape[1] == 2 else a.T  # (wavenumber, intensity) columns
        return [(a[:, 0], a[:, 1])]
    spec = np.atleast_2d(a)
    if wn is None:
        return [(default_axis(spec.shape[1]), s) for s in spec]
    wn = np.atleast_2d(np.squeeze(wn))
    if wn.shape[1] != spec.shape[1]:
        sys.exit(f"wavenumber axis has {wn.shape[1]} values but the scan has {spec.shape[1]}")
    if len(wn) not in (1, len(spec)):
        sys.exit(f"--wn has {len(wn)} rows but there are {len(spec)} scans")
    return [(wn[i if len(wn) > 1 else 0], s) for i, s in enumerate(spec)]


def main():
    ap = argparse.ArgumentParser(description="Identify the chemical in a Raman spectrum.")
    ap.add_argument("file", nargs="?", help="spectrum file: .csv .txt .npy .npz")
    ap.add_argument("--wn", help="file with the wavenumber axis (one row, or one row per scan)")
    ap.add_argument("--values", help='intensities typed directly, e.g. "1.2, 3.4, 5.6"')
    ap.add_argument("--row", type=int, nargs="*", help="only these scans (row numbers, first row = 0)")
    ap.add_argument("--labels", help="optional chem_log.csv with the true names, to compare")
    ap.add_argument("--top", type=int, default=5, help="how many candidates to show")
    ap.add_argument("--out", help="also write the predictions to this csv file")
    args = ap.parse_args()
    if not args.file and not args.values:
        ap.error("give a spectrum file or --values")
    a = np.array([float(v) for v in re.split(r"[,;\s]+", args.values.strip()) if v]) if args.values \
        else read_array(args.file)
    scans = to_scans(a, read_array(args.wn) if args.wn else None)
    rows = list(range(len(scans))) if not args.row else args.row
    truth = None
    if args.labels:
        truth = [RENAME.get(r[1].strip(), r[1].strip()) for r in csv.reader(open(args.labels, encoding="utf-8",
                                                                                 errors="replace"))]

    x = normalise(np.stack([resample(*scans[i]) for i in rows]))
    lib = np.load(OUT_DIR / "library.npz")
    model = Siamese(len(GRID)).to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(OUT_DIR / "model.pt", weights_only=False)["state"])
    ranked = ranked_classes(model.match(x, lib["x"].astype(np.float32)).numpy(), lib["label"], k=args.top)

    records = []
    for i, r, xi in zip(rows, ranked, x):
        print(f"\nscan {i}")
        if xi.max() <= 0:
            print("  no signal between 200 and 1800 cm^-1 - cannot identify")
            continue
        # match score = the model's similarity logit: higher is a closer match, below 0 is a poor one
        print(f"  predicted chemical : {r[0][0]}   (match score {r[0][1]:.1f})")
        if truth is not None and i < len(truth):
            print(f"  label in file      : {truth[i]}   -> {'correct' if truth[i] == r[0][0] else 'different'}")
        print("  other candidates   : " + ",  ".join(f"{n} ({s:.1f})" for n, s in r[1:]))
        records.append([i, r[0][0], round(r[0][1], 2), truth[i] if truth and i < len(truth) else "",
                        " | ".join(n for n, _ in r)])
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["scan", "predicted", "match_score", "label_in_file", f"top{args.top}"])
            w.writerows(records)
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
