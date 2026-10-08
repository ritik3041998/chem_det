# Chemical detection from Raman spectra

Give the program a Raman spectrum (wavenumber and intensity) and it prints the name of the
chemical. It works by comparing the spectrum with a reference library of 10,345 spectra in
2,908 classes: public mineral and organic spectra plus our own scans of explosives and precursors
(RDX, PETN, TNT, HMX, TATB, ammonium nitrate, urea and others).

## Quick start: identify a chemical

Open a terminal in the project folder and activate the environment once per terminal:

```powershell
cd "C:\Users\Ritik Madhasia\OneDrive\Desktop\chm_detection"
& "$env:USERPROFILE\.venvs\chm_detection\Scripts\Activate.ps1"
```

Then run `predict.py` on your spectrum file:

```powershell
python predict.py samples/two_column.csv
```

Output:

```
scan 0
  predicted chemical : Ethylene Diamine   (match score 7.5)
  other candidates   : Wurtzite (4.9),  Pseudowollastonite (4.8),  ...
```

The match score is the model's similarity value. Higher means a closer match. A score below 0, or
a first candidate barely ahead of the second, means the result should not be trusted.

If activation is blocked by PowerShell, skip it and call the environment's Python directly:

```powershell
& "$env:USERPROFILE\.venvs\chm_detection\Scripts\python.exe" predict.py samples/two_column.csv
```

## Commands for every input format

Ready-made example files for each case are in `samples/`.

| Your input | Command |
|---|---|
| CSV with two columns: wavenumber, intensity | `python predict.py samples/two_column.csv` |
| CSV with one row of intensities | `python predict.py samples/intensity_only.csv` |
| TXT with one intensity per line | `python predict.py samples/intensity_column.txt` |
| NPY intensity array + NPY wavenumber array | `python predict.py samples/intensity.npy --wn samples/wavenumber.npy` |
| NPY array of shape (N, 2) or (2, N) | `python predict.py samples/two_column.npy` |
| NPZ with arrays named `wn` and `spec` | `python predict.py samples/scan.npz` |
| Many scans in one file, one per row | `python predict.py samples/many_scans.npy --wn samples/many_wavenumbers.npy` |
| Numbers typed in the terminal | `python predict.py --values="12.5, 14.1, 13.9, ..."` |

Notes on input:

- CSV and TXT files may be separated by commas, semicolons, tabs or spaces. Header lines are skipped.
- **Wavenumber axis.** The model needs to know the wavenumber of each intensity value. Give it
  either as a first column or with `--wn`. If you give intensities only, the scan must have 3,648
  values (our instrument; its usual axis is assumed) or 1,600 values (already on the 200-1799 cm⁻¹
  grid). Any other length without an axis is rejected with a message.
- Write `--values="..."` with the equals sign. Without it, a list starting with a negative number
  is misread as an option.
- The intensities should be baseline-corrected, like `raman/spec_log.csv`.

### Testing on the `raman` folder

```powershell
# one scan (row numbers start at 0)
python predict.py raman/spec_log.csv --wn raman/wnums_log.csv --row 10

# several scans, compared with the names in chem_log.csv
python predict.py raman/spec_log.csv --wn raman/wnums_log.csv --row 0 3 100 500 --labels raman/chem_log.csv

# every scan in the folder, saved to a csv
python predict.py raman/spec_log.csv --wn raman/wnums_log.csv --labels raman/chem_log.csv --out outputs/raman_folder_predictions.csv
```

All options:

| Option | Meaning |
|---|---|
| `--wn FILE` | wavenumber axis: one row for all scans, or one row per scan |
| `--row N [N ...]` | predict only these rows |
| `--labels FILE` | a `chem_log.csv` with the true names; prints correct / different |
| `--top K` | number of candidates to show (default 5) |
| `--out FILE` | also save the predictions as CSV |
| `--values="..."` | intensities typed directly instead of a file |

### Using it from Python

```python
import numpy as np, torch
from common import GRID, OUT_DIR, normalise, resample
from evaluate import ranked_classes
from model import Siamese

wavenumber = np.load("samples/wavenumber.npy")   # your arrays
intensity = np.load("samples/intensity.npy")

x = normalise(resample(wavenumber, intensity))[None]
lib = np.load(OUT_DIR / "library.npz")
model = Siamese(len(GRID)).to("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(OUT_DIR / "model.pt", weights_only=False)["state"])
scores = model.match(x, lib["x"].astype(np.float32)).numpy()
print(ranked_classes(scores, lib["label"], k=5)[0])   # [(name, score), ...]
```

## How accurate it is

Tested on our own scans with whole measurement sessions held out (553 scans, 19 chemicals,
matched against 2,908 classes):

| Method | Top-1 | Top-5 | Mean over chemicals |
|---|---|---|---|
| Cosine similarity (classical) | 90.2% | 97.8% | 76.4% |
| Model trained on public data only | 86.4% | 94.0% | 62.6% |
| Model fine-tuned on our scans (the saved model) | 90.1% | 97.5% | 74.4% |

- **Reliable:** urea, naphthalene, RDX, PETN, HMX, TNB, acetone (100%), hexamine (97%), PNBA (94%).
- **Unreliable:** ammonium nitrate and potassium nitrate are confused with each other; TNT (38%),
  potassium chlorate (38%), H2O2 (0%), ethylene diamine (0%). These have 4-8 scans each, except
  the nitrates.
- **Not tested:** ethyl alcohol, isopropyl alcohol, nitrobenzene, nitromethane, potassium
  carbonate and sulphur have a single session each. DNAN's only scan was blank, so DNAN is not in
  the library.
- On the `raman` folder itself the saved model agrees with the label on 539 of 560 labelled scans
  (96%). That figure is optimistic, because those scans are in the reference library. The 90.1%
  above is the fair estimate for a new measurement.
- A chemical that is not in the library cannot be named. The program will still print its nearest
  match, usually with a low score.

## What is in the project

| File | Purpose |
|---|---|
| `predict.py` | identify the chemical in a spectrum (the command you run) |
| `prepare_ours.py` | cleans `raman/` into `data/ours.npz` |
| `prepare_public.py` | builds the public training library `data/public.npz` |
| `train.py` | trains the model on the public library -> `outputs/pretrained.pt` |
| `evaluate.py` | tests on our data, fine-tunes, saves `outputs/model.pt` and `outputs/library.npz` |
| `model.py` | the neural network |
| `common.py` | shared settings: wavenumber grid, resampling, normalisation, name aliases |
| `samples/` | example input files in every supported format |
| `outputs/` | model, library, `results.json`, `predictions.csv`, `confusion_matrix.png` |
| `external/` | clone of the reference repository the model comes from |

### Our data (`raman/`)

| File | Content | Used |
|---|---|---|
| `spec_log.csv` | intensity, 3,648 values per scan, baseline-corrected | yes |
| `wnums_log.csv` | wavenumber of each value, about -106 to 3,314 cm⁻¹ | yes |
| `chem_log.csv` | timestamp, chemical name, ON/OFF flag | yes |
| `spec_wl.csv`, `spec_wol.csv` | other intensity versions, not row-aligned with the names | no |
| `peaks_amp.csv`, `peaks_pxl.csv` | detected peak heights and positions | no |

753 scans; 20 blank ones are dropped; 560 carry a chemical name (25 chemicals) and 173 are
marked `not in lib`. The names appear to be the instrument's own match, not hand-checked.

### Public data

| Source | Spectra | Classes |
|---|---|---|
| [RRUFF](https://rruff.info/zipped_data_files/raman/) minerals | 13,843 | about 2,340 |
| Organic compounds, via [raman-data](https://github.com/ml-lab-htw/raman_data) | 5,278 | 377 |
| Pharmaceutical solvents and reagents, via raman-data | 3,510 | 32 |
| Biomolecules, via raman-data | 202 | 141 |
| **Total** | **22,833** | **2,889** |

No public dataset contains explosives. Those spectra come only from `raman/`.

### Method

1. Every spectrum, public or ours, is resampled to one axis (200-1799 cm⁻¹, 1 cm⁻¹ steps), negative
   values are clipped and the spectrum is scaled to a maximum of 1.
2. The model is the 1D Xception Siamese network from
   [lyn1874/raman_spectra_matching_with_contrastive_learning](https://github.com/lyn1874/raman_spectra_matching_with_contrastive_learning)
   (MIT licence). It takes two spectra and scores whether they are the same substance.
3. It is trained on the public library with augmentations that imitate a handheld instrument
   (calibration shift, broader peaks, noise, baseline, shorter range), then fine-tuned on our scans.
4. To identify a scan, it is scored against every library spectrum and the best-scoring substance
   is reported.

## Setup on a new machine

Needs Python 3.10 or newer and, for training, an NVIDIA GPU (prediction also runs on CPU, slower).

```powershell
python -m venv "$env:USERPROFILE\.venvs\chm_detection"
& "$env:USERPROFILE\.venvs\chm_detection\Scripts\Activate.ps1"
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The environment is kept outside the project folder so OneDrive does not sync it.

## Rebuilding everything from scratch

Only needed after adding new scans to `raman/` or changing the model. Prediction alone needs
just `outputs/model.pt` and `outputs/library.npz`, which already exist.

```powershell
# 1. download the RRUFF archives (about 450 MB) into the cache folder
$d = "$env:USERPROFILE\.cache\chm_detection\rruff"; mkdir $d -Force
foreach ($f in 'excellent_unoriented','excellent_oriented','fair_unoriented','fair_oriented','poor_unoriented','unrated_unoriented','unrated_oriented') {
    curl.exe -L -o "$d\$f.zip" "https://rruff.info/zipped_data_files/raman/$f.zip" }

# 2. prepare the data
python prepare_ours.py        # raman/ -> data/ours.npz
python prepare_public.py      # RRUFF + raman-data -> data/public.npz

# 3. train on the public library (about 6 minutes on an RTX 3050)
python train.py --steps 6000

# 4. test on our data and save the final model (about 15 minutes)
python evaluate.py --ft_steps 1500
```

### Adding a new chemical

Append its scans to `raman/spec_log.csv` and `raman/wnums_log.csv`, add matching rows
(timestamp, name) to `raman/chem_log.csv`, then run steps 2 and 4 above (`prepare_ours.py` and
`evaluate.py`). Several scans from separate sessions are needed for the chemical to be tested.

## Related repositories

| Repository | What it offers |
|---|---|
| [lyn1874/raman_spectra_matching_with_contrastive_learning](https://github.com/lyn1874/raman_spectra_matching_with_contrastive_learning) | library matching on RRUFF, organic and bacteria sets; the model used here |
| [ml-lab-htw/RamanBench](https://github.com/ml-lab-htw/RamanBench) | benchmark of 28 models on public Raman datasets |
| [ml-lab-htw/raman_data](https://github.com/ml-lab-htw/raman_data) | loader for the public datasets |
| [SaashaJoshi/raman-spetroscopy-chemical-detection](https://github.com/SaashaJoshi/raman-spetroscopy-chemical-detection) | CNN on RRUFF, framed around explosive chemicals |
| [zshicode/Raman-Spectra-Deep-Learning](https://github.com/zshicode/Raman-Spectra-Deep-Learning) | CNN, LSTM, Transformer, GNN and contrastive models in PyTorch |
| [XiaqiongFan/DeepRaman](https://github.com/XiaqiongFan/DeepRaman) | component identification in mixtures |
| [XiaqiongFan/DeepCID](https://github.com/XiaqiongFan/DeepCID) | per-compound CNNs for mixtures |
| [nibtehaz/RamanNet](https://github.com/nibtehaz/RamanNet) | Raman-specific network architecture |
| [csho33/bacteria-ID](https://github.com/csho33/bacteria-ID) | 1D ResNet for Raman (bacteria) |
| [barahona-research-group/RamanSPy](https://github.com/barahona-research-group/RamanSPy) | preprocessing and analysis toolkit |
| [h2020charisma/ramanchada2](https://github.com/h2020charisma/ramanchada2) | calibration and harmonisation between instruments |
| [aaroncelestian/RamanLab](https://github.com/aaroncelestian/RamanLab) | desktop app for matching against a database |
| [charlesll/RaMin](https://github.com/charlesll/RaMin) | mineral recognition notebooks on RRUFF |
