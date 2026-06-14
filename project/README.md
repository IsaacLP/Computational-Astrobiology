<img src="https://img.shields.io/badge/MASS--UBMATF-Computational_Astrobiology_2026-blue" alt="MASS-UBMATF Badge">

# Recovering Planetary Transits in Active PLATO-like Stars

A simulation-to-ML pipeline for detecting exoplanet transits in the presence of stellar activity, built around ESA PLATO mission preparation tools.

## Overview

Stellar activity — spots, rotation, flares, granulation — can mask or mimic the tiny brightness dips produced by transiting planets. This project generates synthetic PLATO-like light curves using **PSLS** (PLATO Solar-like Light-curve Simulator), injects planetary transits across a range of activity regimes, and trains machine-learning classifiers to recover planet signals where classical methods struggle.

The core question: *Can ML reliably detect transiting planets when the host star is realistically active?*

## Dataset

~1500 labeled light curves across four classes:

| Class | Description |
|-------|-------------|
| 0 | Quiet star (mild activity), no planet |
| 1 | Active star (strong activity), no planet — false positive regime |
| 2 | Planet + mild activity |
| 3 | Planet + strong activity |

Planet parameters span periods 2–50 d and radii 0.8–4 R⊕. Stellar parameters are drawn from the PLATO target range (Teff 5200–6400 K, log g 4.20–4.55).

## Repository Structure

```
project/
├── psls/                        # PSLS simulator (vendored, upstream package)
│   ├── psls.py                  # Main entry point — called as subprocess
│   ├── sls.py                   # Solar-like star light curve engine
│   ├── transit.py               # Transit injection (Mandel & Agol 2002)
│   ├── spotintime.py            # Spot modulation (Dorren 1987)
│   ├── flares.py                # Flare generation
│   ├── models/        [request] # CESAM2K stellar grid (HDF5) + ADIPLS modes — not redistributed
│   ├── systematics/             # PLATO instrumental systematics (.npy tables)
│   └── examples/                # Example PSLS configs and outputs
├── simulation/
│   ├── generate_dataset.py      # Full pipeline: sample → PSLS → metadata.csv
│   ├── run_one.py               # Re-run PSLS for a single existing config
│   ├── build_dataset.py         # Pack .npz light curves → single HDF5
│   └── read_dataset.py          # Utility to load the packed HDF5
├── data/
│   ├── metadata.csv             # One row per LC: labels, sampled parameters, derived SNR/detection flags
│   ├── bls_results.csv          # Per-LC BLS output: period, power, depth, recovery flags (Phase 3)
│   ├── dataset.h5         [LFS] # Packed HDF5 of all light curves (downsampled) — 201 MB
│   ├── lightcurves/   [ignored] # Per-star .npz files — 12 GB, regenerate with Phase 2
│   ├── configs/       [ignored] # Per-star PSLS YAMLs — regenerate with Phase 2
│   └── models/            [LFS] # Trained ML model weights and split indices (Phase 4)
├── notebooks/
│   ├── phase2_simulation_pipeline.ipynb   # Dataset generation walkthrough + health checks
│   ├── phase3_bls_baseline.ipynb          # Classical BLS baseline (Wotan detrend → BLS → recovery stats)
│   └── phase4_ml_classification.ipynb     # ML classifiers: Logistic Regression, RF, XGBoost, CNN
└── reports/
    └── phase1_literature_review.md        # Literature review
```

## Setup

### 0. Git LFS and external data

Large binary files (`psls/systematics/`, `data/dataset.h5`, `data/models/`) are stored in Git LFS. Install LFS and pull them after cloning:

```bash
git lfs install
git lfs pull
```

**PSLS stellar grid (`psls/models/`)** — the CESAM2K grid is not redistributed in this repository. It was provided by Reza Samadi (reza.samadi@obspm.fr) and is documented at the [PSLS website](https://sites.lesia.obspm.fr/psls/documentation/). Contact him directly to obtain it, then place the files under `psls/models/`.

### 1. Python environment

All dependencies are managed in a dedicated virtual environment:

```bash
python3.12 -m venv ~/.venvs/plato-sim
source ~/.venvs/plato-sim/bin/activate
```

### 2. Install PSLS

PSLS is vendored in `psls/`. Install it in editable mode:

```bash
cd psls
pip install -e .
cd ..
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **macOS note:** XGBoost requires OpenMP. Install via Homebrew: `brew install libomp`

## Reproducing Results

### Phase 1 — Literature Review

See `reports/phase1_literature_review.md`.

### Phase 2 — Dataset Generation

Generate ~1500 light curves (375 per class) and pack them into an HDF5 file:

```bash
cd simulation

# Generate light curves (runs PSLS for each config, ~minutes to hours depending on N)
python generate_dataset.py 375

# Pack into a single HDF5 at 10-min cadence (24× downsampling from 25-s PSLS output)
python build_dataset.py --downsample 24
```

Outputs: `data/lightcurves/`, `data/configs/`, `data/metadata.csv`, `data/dataset.h5`

To re-run a single light curve (its config YAML must already exist in `data/configs/`):

```bash
python run_one.py <star_id>
```

Walk through the full generation pipeline and dataset health checks in:

```
notebooks/phase2_simulation_pipeline.ipynb
```

### Phase 3 — BLS Classical Baseline

Run the notebook to execute a BLS transit search on the full dataset:

```
notebooks/phase3_bls_baseline.ipynb
```

Pipeline per light curve: bin to 1-hour cadence → Wotan biweight detrend → BLS search via `astropy.timeseries.BoxLeastSquares` → threshold at 95th-percentile power of Class 0 (caps FPR ≤ 5%).

Output: `data/bls_results.csv`

### Phase 4 — ML Classification

Run the notebook to train and evaluate all classifiers:

```
notebooks/phase4_ml_classification.ipynb
```

Models trained (binary planet/no-planet and 4-class):
- Logistic Regression (tabular BLS features)
- Random Forest (tabular BLS features)
- XGBoost (tabular BLS features)
- 1D CNN (raw phase-folded flux)

Trained weights are saved to `data/models/`. The notebook compares all models against the BLS baseline recovery fractions.

## PSLS Quick Reference

Run the bundled PSLS example directly (must run from `psls/` as working directory):

```bash
cd psls
python psls.py -P -V examples/psls.yaml
```

The simulator resolves `models/` and `systematics/` relative to its own directory, so it must always be invoked with `psls/` as the working directory (the pipeline scripts in `simulation/` handle this automatically).
