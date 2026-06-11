# Simulation Pipeline

The goal in this phase is to generate the dataset of LC's that will be used to train the classfication model.

The four classes are:
| | No planet | Planet |
|---|---|---|
|Mild activity | Class 0 | Class 2 |
|Strong activity | Class 1 | Class 3 |

## Parameter sampling
The simulation pipeline was fully automated in `generate_dataset.py`. The default run produces **1500 LCs per class (6000 total)**, all parameters drawn from continuous distributions so the classifier must learn physical phenomena rather than memorise discrete grid values. All sampling is seeded from a single master RNG (default seed = 42) for full reproducibility.

### Stellar parameters (every LC)

| Parameter | Distribution | Range |
|---|---|---|
| Teff | uniform | 5200–6400 K |
| Logg | uniform | 4.20–4.55 |

PSLS snaps (Teff, Logg) to the nearest CESAM2K main-sequence track in `grid_v0.1_ov0-plato.hdf5` using a Chi2 metric identical to PSLS's internal `search_model_hdf5`. The snapped stellar mass and radius are recovered from the grid and stored in `metadata.csv`. 

### Activity parameters (class-dependent)

Two activity levels are defined internally — `mild` (Classes 0 & 2) and `strong` (Classes 1 & 3). Spots and flares are **always enabled** for both levels; the distinction is purely in amplitude and timescale.

| Parameter | mild | strong |
|---|---|---|
| Rotation period [d] | uniform [20, 35] | uniform [8, 22] |
| Activity σ [ppm] | log-uniform [5, 40] | log-uniform [25, 150] |
| Activity τ [d] | uniform [0.5, 2.5] | uniform [1.0, 5.0] |
| Spot count | uniform int [1, 3] | uniform int [2, 5] |
| Spot radius [°] | uniform [0.5, 2.0] | uniform [1.5, 3.0] |
| Spot contrast | uniform [0.62, 0.80] | uniform [0.40, 0.65] |
| Spot lifetime | uniform [0.2, 0.8] × P_rot | uniform [0.5, 1.5] × P_rot |
| Spot latitude [°] | uniform [−20, 20] | uniform [−35, 35] |
| Spot dΩ | uniform [0.12, 0.25] | uniform [0.05, 0.15] |
| Flare mean period [d] | log-uniform [20, 80] | log-uniform [1, 6] |
| Flare amplitude [ppm] | power-law [15, 100], α=2 | power-law [100, 1500], α=2 |

Mild and strong distributions **overlap deliberately** on rotation period ([20, 22] d), σ ([25, 40] ppm), spot radius ([1.5, 2.0]°), and spot contrast ([0.62, 0.65]), preventing the classifier from trivially thresholding on any single activity indicator. Spot lifetime is expressed as a fraction of P_rot so coherence (lifetime/P_rot) is preserved independently of the rotation period draw. Flare amplitudes follow a power-law (Davenport 2016); mild-class flares occur every 20–80 days (essentially absent over the 270-day baseline for most draws), while strong-class flares occur every 1–6 days.

### Inclination

Drawn using the physical prior — uniform in cos(*i*) — over [5°, 90°] for all classes.

### Planet parameters (Classes 2 & 3)

| Parameter | Distribution | Range |
|---|---|---|
| Orbital period [d] | log-uniform | 3–50 |
| Planet radius [R_Jup] | log-uniform | 0.089–0.356 (≈ 1.0–3.9 R⊕) |
| Semi-major axis [AU] | Kepler's 3rd law | derived from grid stellar mass |
| Orbital angle [°] | uniform | 0–360 |

The log-uniform draws reflect roughly flat occurrence rates in log-period and log-radius; the 3–50 d window ensures at least ~5 transits in the 270-day baseline. The semi-major axis is computed from Kepler's 3rd law using the stellar mass recovered from the CESAM2K grid. Transit depth is likewise computed using the grid stellar radius and stored in `metadata.csv`.
