# Simulation Pipeline

The goal in this phase is to generate the dataset of LC's that will be used to train the classfication model.

The four classes are:
| | No planet | Planet
|---|---|---|
|Mild activity | Class 0 | Class 1 |
|Strong activity | Class 2 | Class 3 |

## Parameter sampling
The simulation pipeline was fully automated in `generate_dataset.py`. The default run produces **1500 LCs per class (6000 total)**, all parameters drawn from continuous distributions so the classifier must learn physical phenomena rather than memorise discrete grid values. All sampling is seeded from a single master RNG (default seed = 42) for full reproducibility.

### Stellar parameters (every LC)

| Parameter | Distribution | Range |
|---|---|---|
| Teff | uniform | 5200–6400 K |
| Logg | uniform | 4.20–4.55 |

PSLS snaps (Teff, Logg) to the nearest CESAM2K main-sequence track in `grid_v0.1_ov0-plato.hdf5`; mass, radius, numax, and Δν are read from the grid and are **not** set in the YAML.

### Activity parameters (class-dependent)

Two activity levels are defined internally — `mild` (Classes 0 & 2) and `strong` (Classes 1 & 3). Spots and flares are **always enabled** for both levels; the distinction is purely in amplitude and timescale.

| Parameter | mild | strong |
|---|---|---|
| Rotation period [d] | uniform [15, 30] | uniform [5, 18] |
| Activity σ [ppm] | log-uniform [25, 100] | log-uniform [60, 300] |
| Activity τ [d] | uniform [0.10, 1.00] | uniform [0.30, 3.00] |
| Spot radius [°] | uniform [1.5, 5.0] | uniform [3.5, 12.0] |
| Spot contrast | uniform [0.60, 0.78] | uniform [0.25, 0.55] |
| Spot lifetime [d] | uniform [20, 100] | uniform [30, 200] |
| Spot latitude [°] | uniform [−25, 25] | uniform [−35, 35] |
| Spot dΩ | uniform [0.00, 0.25] | uniform [0.00, 0.15] |
| Flare mean period [d] | log-uniform [60, 500] | log-uniform [0.3, 2] |
| Flare amplitude [ppm] | log-uniform [200, 800] | log-uniform [800, 5000] |

Mild and strong distributions **overlap deliberately** on rotation period (15–18 d) and σ (60–80 ppm). Spot radius also overlaps on [3.5, 5.0]°, preventing the classifier from trivially thresholding on activity level. Mild-class flares are extremely rare (mean inter-flare interval 60–500 d), making them essentially absent over the 270-day baseline, while strong-class flares occur every 0.3–2 days.

### Inclination

Drawn using the physical prior — uniform in cos(*i*) — within class-dependent ranges:

- mild: [25°, 85°]
- strong: [45°, 90°] — biased equatorial to make spot modulation prominent

### Planet parameters (Classes 2 & 3)

| Parameter | Distribution | Range |
|---|---|---|
| Orbital period [d] | log-uniform | 3–50 |
| Planet radius [R⊕] | log-uniform | 1.0–4.0 (= 0.089–0.356 R_Jup) |
| Semi-major axis [AU] | Kepler's 3rd law | derived (1 M☉ host) |
| Orbital angle [°] | uniform | 0–360 |

The log-uniform draws reflect roughly flat occurrence rates in log-period and log-radius; the 3–50 d window ensures at least ~5 transits in the 270-day baseline.