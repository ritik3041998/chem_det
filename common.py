"""Shared settings and preprocessing so public spectra and our spectra land in the same space."""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAMAN_DIR = ROOT / "raman"
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CACHE_DIR = Path(os.environ.get("CHM_CACHE", Path.home() / ".cache" / "chm_detection"))

# Common wavenumber axis (cm^-1, 1 cm^-1 step). Chosen as the overlap of our instrument and RRUFF.
WN_MIN, WN_MAX = 200, 1800
GRID = np.arange(WN_MIN, WN_MAX, dtype=np.float32)


UNKNOWN = "not in lib"
# Public-library names that are the same substance as one of our chemicals (mineral name -> ours).
ALIASES = {"Niter": "Potassium Nitrate", "Nitratine": "Sodium Nitrate", "Sulphur": "Sulphur",
           "Acetone": "Acetone", "2 - Propanol": "Isopropyl Alcohol", "Ethanol": "Ethyl Alcohol"}


def load_public():
    d = np.load(DATA_DIR / "public.npz")
    label = np.array([ALIASES.get(n, n) for n in d["label"]])
    return d["x"].astype(np.float32), label, d["group"], d["source"]


def cap_per_class(label, n, seed=0):
    """Indices keeping at most n randomly chosen spectra of each class."""
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.permutation(np.flatnonzero(label == c))[:n] for c in np.unique(label)]))


def load_ours():
    d = np.load(DATA_DIR / "ours.npz")
    return d["x"], d["label"], d["session"], d["row"]


def resample(wn, inten, grid=GRID):
    """Linear interpolation onto the common grid; outside the measured range is filled with 0."""
    wn = np.asarray(wn, dtype=np.float64)
    inten = np.asarray(inten, dtype=np.float64)
    order = np.argsort(wn)
    wn, inten = wn[order], inten[order]
    return np.interp(grid, wn, inten, left=0.0, right=0.0).astype(np.float32)


def normalise(x):
    """Clip negatives (baseline-corrected spectra dip below 0) and scale each spectrum to max 1."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0, None)
    m = x.max(axis=-1, keepdims=True)
    return x / np.where(m > 0, m, 1)
