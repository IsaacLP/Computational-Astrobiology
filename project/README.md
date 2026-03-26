# Recovering Planetary Transits in Active PLATO-like Stars

A simulation-to-ML pipeline for detecting exoplanet transits in the presence of stellar activity, built around ESA PLATO mission preparation tools.

## Overview

Stellar activity — spots, rotation, flares, granulation — can mask or mimic the tiny brightness dips produced by transiting planets. This project generates synthetic PLATO-like light curves using **PSLS** (PLATO Solar-like Light-curve Simulator), injects planetary transits across a range of activity regimes, and trains machine-learning classifiers to recover planet signals where classical methods struggle.

The core question: *Can ML reliably detect transiting planets when the host star is realistically active?*

## Dataset

~6000–7000 labeled light curves across four classes:

| Class | Description |
|-------|-------------|
| 0 | Quiet star, no planet |
| 1 | Activity only (false positive regime) |
| 2 | Planet + mild activity |
| 3 | Planet + strong activity |

## Pipeline

1. **Simulate** — PSLS generates PLATO-like light curves with spots, rotation, flares, granulation, and instrumental noise
2. **Inject** — Planet transits are injected across a structured parameter grid (periods 2–50 d, radii 0.8–4 R⊕)
3. **Baseline** — TLS transit search with optional Wotan detrending establishes a classical benchmark
4. **ML** — Models trained on engineered tabular features and/or raw 1D sequences
5. **Evaluate** — Recovery fraction as a function of activity level, planet radius, and transit depth