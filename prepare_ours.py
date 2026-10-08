"""Turn raman/*.csv into one clean array on the common wavenumber grid -> data/ours.npz

Uses spec_log.csv (intensity), wnums_log.csv (wavenumber per pixel) and chem_log.csv
(timestamp, chemical name[, ON/OFF]) - these three are row-aligned. spec_wl/spec_wol are not.
"""
import csv
from datetime import datetime

import numpy as np

from common import DATA_DIR, RAMAN_DIR, normalise, resample

# Spelling / casing fixes for the names written by the instrument.
RENAME = {"UREA": "Urea", "Napthalene": "Naphthalene", "Nirobenzene": "Nitrobenzene",
          "Sodium chlorate": "Sodium Chlorate"}
UNKNOWN = "not in lib"


def parse_time(s):
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%d-%m-%Y %H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    raise ValueError(f"unrecognised timestamp {s!r}")


def main():
    rows = list(csv.reader(open(RAMAN_DIR / "chem_log.csv", encoding="utf-8", errors="replace")))
    times = [parse_time(r[0]) for r in rows]
    labels = np.array([RENAME.get(r[1].strip(), r[1].strip()) for r in rows])
    flag = np.array([r[2].strip() if len(r) > 2 else "" for r in rows])
    spec = np.loadtxt(RAMAN_DIR / "spec_log.csv", delimiter=",")
    wn = np.loadtxt(RAMAN_DIR / "wnums_log.csv", delimiter=",")
    assert len(spec) == len(wn) == len(rows), "files are not row-aligned"

    # A session = consecutive scans of the same chemical less than 30 min apart. Scans within a
    # session are near-duplicates, so the evaluation keeps a whole session on one side of a split.
    session = np.zeros(len(rows), dtype=int)
    for i in range(1, len(rows)):
        gap = abs((times[i] - times[i - 1]).total_seconds())
        session[i] = session[i - 1] + (labels[i] != labels[i - 1] or gap > 1800)

    x = np.stack([resample(w, s) for w, s in zip(wn, spec)])
    # Blank scans: no peak standing clearly above the pixel-to-pixel noise in the common window.
    noise = np.diff(x, n=2, axis=1).std(axis=1) / np.sqrt(6)
    snr = x.max(axis=1) / np.where(noise > 0, noise, 1)
    keep = snr > 15

    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(DATA_DIR / "ours.npz", x=normalise(x[keep]), label=labels[keep], session=session[keep],
                        flag=flag[keep], time=np.array([t.isoformat() for t in times])[keep],
                        row=np.flatnonzero(keep), snr=snr[keep])
    print(f"{len(rows)} scans read, {int((~keep).sum())} blank/low-signal dropped, {int(keep.sum())} kept")
    known = labels[keep] != UNKNOWN
    print(f"{int(known.sum())} labelled scans in {len(set(labels[keep][known]))} chemicals, "
          f"{int((~known).sum())} '{UNKNOWN}'")
    for name in sorted(set(labels[keep]), key=lambda n: -(labels[keep] == n).sum()):
        m = labels[keep] == name
        print(f"  {int(m.sum()):4d} scans  {len(set(session[keep][m])):3d} sessions  {name}")


if __name__ == "__main__":
    main()
