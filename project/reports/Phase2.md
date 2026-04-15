# Simulation Pipeline

The goal in this phase is to generate the dataset of LC's that will be used to train the classfication model.

The four classes are:
- **Class 0**: Quiet star, no planet
- **Class 1**: Activity only (false positive regime)
- **Class 2**: Planet + mild activity
- **Class 3**: Planet + strong activity

## Initial test (Only 1 file per class) 

**Key choices**
- All use `ModelType: grid` pointing at `grid_v0.1_ov0-plato.hdf5`
- Classes 0 and 2 have minimal activity; Classes 1 and 3 have strong activity (spots + stochastic + flares)
- Classes 2 and 3 enable transit; Classes 0 and 1 do not
- Each gets a distinct `MasterSeed` so simulations are independent


**Shared baseline**
- the same star (solar-like, V=10, Teff=5778 K, logg=4.438, main sequence)
- same grid model
- same 270-day baseline (3×90-day quarters)
- same PLATO instrumental noise setup (24 cameras, `PLATO_SIMU` random noise, BOL systematics table).

**What distinguishes the classes:**

| Parameter | Class 0 | Class 1 | Class 2 | Class 3 |
|---|---|---|---|---|
| `Transit/Enable` | 0 | 0 | 1 | 1 |
| `Activity/Enable` (stochastic) | 0 | 1 (σ=150 ppm) | 1 (σ=40 ppm) | 1 (σ=150 ppm) |
| `Spot/Enable` | 0 | 1 (3 large spots) | 0 | 1 (3 large spots) |
| `Flare/Enable` | 0 | 1 (every ~3 days) | 0 | 1 (every ~3 days) |
| `SurfaceRotationPeriod` [days] | 0 (off) | 12 (fast) | 26 (Sun-like) | 12 (fast) |
| `Inclination` [°] | 0 | 60 | 30 | 60 |

**Planet** 
(Classes 2 & 3)
- 0.2 R_Jup (~2.25 R⊕)
- 15-day period at 0.12 AU 