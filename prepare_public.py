"""Build the public training library on the common wavenumber grid -> data/public.npz

Sources
  RRUFF   https://rruff.info/zipped_data_files/raman/  (zips expected in CACHE_DIR/rruff)
          processed (baseline-corrected) spectra of ~2500 mineral species
  raman-data (pip install raman-bench) - experimental organic / pharmaceutical / biomolecule
          reference sets, which cover the 1300-1800 cm^-1 region that most RRUFF spectra lack
"""
import zipfile

import numpy as np

from common import CACHE_DIR, DATA_DIR, GRID, WN_MAX, WN_MIN, normalise, resample

RAMAN_DATA_SETS = ["organic_compounds_preprocess", "pharmaceutical_ingredients", "biomolecules_reference"]
MIN_COVERAGE = 800  # cm^-1 of the common grid a spectrum must actually cover


def read_rruff():
    xs, labels, groups = [], [], []
    for z in sorted((CACHE_DIR / "rruff").glob("*.zip")):
        zf = zipfile.ZipFile(z)
        n_ok = 0
        for name in zf.namelist():
            if not name.endswith(".txt") or "Processed" not in name:
                continue
            parts = name.split("/")[-1].split("__")
            wn, inten = [], []
            for line in zf.read(name).decode("latin-1").splitlines():
                if line[:1].isdigit() and "," in line:
                    a, b = line.split(",")[:2]
                    wn.append(float(a))
                    inten.append(float(b))
            if len(wn) < 100 or min(max(wn), WN_MAX) - max(min(wn), WN_MIN) < MIN_COVERAGE:
                continue
            x = resample(wn, inten)
            if x.max() <= 0:
                continue
            xs.append(x)
            labels.append(parts[0])
            groups.append(parts[1])  # RRUFF specimen id
            n_ok += 1
        print(f"  {z.name:26s} {n_ok:6d} spectra")
    return np.stack(xs), labels, groups


def read_raman_data():
    from raman_data import raman_data
    xs, labels, groups = [], [], []
    for ds_name in RAMAN_DATA_SETS:
        try:
            ds = raman_data(ds_name)
            spectra, shifts = np.asarray(ds.spectra, dtype=float), np.asarray(ds.raman_shifts, dtype=float)
            names = [str(n).strip() for n in ds.target_names]
            if all(n.isdigit() for n in names):  # anonymous compound ids
                names = [f"{ds_name.split('_')[0]}_{n}" for n in names]
            targets = np.asarray(ds.targets).astype(int)
            gid = getattr(ds, "group_ids", None)
            n_ok = 0
            for i, s in enumerate(spectra):
                if min(shifts.max(), WN_MAX) - max(shifts.min(), WN_MIN) < MIN_COVERAGE or not np.isfinite(s).all():
                    continue
                x = resample(shifts, s - np.min(s))
                if x.max() <= 0:
                    continue
                xs.append(x)
                labels.append(names[targets[i]])
                groups.append(f"{ds_name}:{gid[i] if gid is not None else i}")
                n_ok += 1
            print(f"  {ds_name:26s} {n_ok:6d} spectra, {len(set(labels[-n_ok:])) if n_ok else 0} classes")
        except Exception as e:  # a dataset that fails to download must not block the rest
            print(f"  {ds_name:26s} SKIPPED ({type(e).__name__}: {e})")
    return (np.stack(xs) if xs else np.zeros((0, len(GRID)), np.float32)), labels, groups


def main():
    print("RRUFF")
    x1, l1, g1 = read_rruff()
    print("raman-data")
    x2, l2, g2 = read_raman_data()
    x = normalise(np.concatenate([x1, x2]))
    source = np.array(["rruff"] * len(l1) + ["raman-data"] * len(l2))
    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(DATA_DIR / "public.npz", x=x.astype(np.float16), label=np.array(l1 + l2),
                        group=np.array(g1 + g2), source=source)
    print(f"saved {len(x)} spectra, {len(set(l1 + l2))} classes -> {DATA_DIR / 'public.npz'}")


if __name__ == "__main__":
    main()
