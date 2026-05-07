"""
Phase 2 — PLATO Transit Detection Dataset Generator
=====================================================
Generates ~6000-7000 labeled light curves across 4 classes using PSLS:
  Class 0: no planet + mild activity
  Class 1: no planet + strong activity
  Class 2: planet + mild activity
  Class 3: planet + strong activity

Usage
-----
  python generate_dataset.py                   # full run (~6000 LCs)
  python generate_dataset.py --n_per_class 50  # quick smoke-test
  python generate_dataset.py --out_dir /data/plato_lcs

The script writes:
  <out_dir>/lightcurves/   ← one .dat per LC  (raw PSLS output)
  <out_dir>/metadata.csv   ← one row per LC with all labels & parameters
  <out_dir>/configs/       ← the PSLS YAML used for each LC (reproducibility)
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Observation / instrument constants (never sampled)
# ---------------------------------------------------------------------------

QUARTER_DURATION = [90.0, 90.0, 90.0]   # 3 × 90 days = 270-day baseline
SAMPLING         = 25                    # cadence in seconds
MAG              = 10.0                  # V magnitude — bright PLATO-like target
STAR_ID_BASE     = 0                     # sequential ID counter base


# ---------------------------------------------------------------------------
# Stellar grid parameter ranges
# ---------------------------------------------------------------------------
# PSLS ModelType: grid snaps to the nearest (Teff, Logg) point in the
# CESAM2K HDF5 grid (grid_v0.1_ov0-plato.hdf5).  
#
# Valid CESAM2K MS range: Teff 3973–6611 K, Logg 3.39–4.60,
#                         mass 0.90–1.20 Msun (from grid.ipynb exploration).
# We stay in the PLATO solar-like sweet spot:
#   Teff  5200–6400 K   (F5 – K0 main-sequence)
#   Logg  4.20–4.55     (dense MS band)

TEFF_RANGE  = (5200.0, 6400.0)   # K   — uniform draw
LOGG_RANGE  = (4.20,   4.55)     # cgs — uniform draw

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Draw one sample from a log-uniform distribution on [lo, hi]."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _powerlaw(rng: np.random.Generator, lo: float, hi: float, alpha: float) -> float:
    """Draw one sample from a truncated power law p(x) ∝ x^{-alpha} on [lo, hi].
    Uses inverse-CDF method; degenerates to log-uniform when alpha == 1."""
    if abs(alpha - 1.0) < 1e-10:
        return _loguniform(rng, lo, hi)
    exp = 1.0 - alpha
    u = rng.uniform(0.0, 1.0)
    return float((u * (hi**exp - lo**exp) + lo**exp) ** (1.0 / exp))


def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _inclination_from_cosi(rng: np.random.Generator,
                            i_min_deg: float, i_max_deg: float) -> float:
    """
    Draw inclination using the physical prior: uniform in cos(i).
    """
    cos_lo = np.cos(np.radians(i_max_deg))   # note: cos is decreasing
    cos_hi = np.cos(np.radians(i_min_deg))
    cos_i  = rng.uniform(cos_lo, cos_hi)
    return float(np.degrees(np.arccos(cos_i)))


# ---------------------------------------------------------------------------
# Per-parameter samplers  (one call → one float or structured dict)
# ---------------------------------------------------------------------------

def sample_stellar_params(rng: np.random.Generator) -> tuple[float, float]:
    """
    Sample (Teff, Logg) uniformly inside the valid CESAM2K MS grid region.
    """
    teff = round(_uniform(rng, *TEFF_RANGE), 1)
    logg = round(_uniform(rng, *LOGG_RANGE), 3)
    return teff, logg


def sample_rotation_period(activity_class: str, rng: np.random.Generator) -> float:
    """
    Surface rotation period (days).  Drives spot modulation timescale and
    the rotational splitting of p-mode multiplets (seismic only — negligible
    for broadband photometry).  Key: P_rot controls the period of quasi-
    sinusoidal spot modulation, which aliases with P_transit.

    Ranges grounded in McQuillan et al. 2014 (Kepler rotation survey):
      - mild: 20-35 d  (solar-like / magnetically inactive)
      - strong: 5-18 d  (fast rotator, young or active G/K dwarf)

    Mild class upper end overlaps strong lower end [15-18 d] so the
    classifier cannot trivially use P_rot as a class proxy.
    """

    if activity_class == "mild":
        return round(_uniform(rng, 15.0, 30.0), 2)
    else:  # strong
        return round(_uniform(rng, 5.0, 18.0), 2)


def sample_activity_sigma(activity_class: str, rng: np.random.Generator) -> float:
    """
    Stochastic magnetic activity amplitude (ppm).  Log-uniform because the
    quantity spans an order of magnitude.

    Ranges chosen so mild and strong distributions overlap on [60, 80] ppm —
    the classifier cannot threshold on activity amplitude alone.
    """
    if activity_class == "mild":
        return round(_loguniform(rng, 25.0, 100.0), 2)
    else:  # strong
        return round(_loguniform(rng, 60.0, 300.0), 2)


def sample_activity_tau(activity_class: str, rng: np.random.Generator) -> float:
    """
    Stochastic activity correlation timescale (days).
    The key regime: Tau ~ transit duration causes maximum confusion.
    Transit durations for P_orb = 5-50 d around a solar-like star are
    roughly 0.1-0.5 days.  Strong-class Tau extends into this regime.
    """
    if activity_class == "mild":
        return round(_uniform(rng, 0.10, 1.00), 3)
    else:  # strong
        return round(_uniform(rng, 0.30, 3.00), 3)


def sample_spot_params(activity_class: str, rng: np.random.Generator) -> dict | None:
    """
    Return a dict of all PSLS Spot sub-parameters for one spot group

    Parameter notes
    ---------------
    Radius   : angular diameter of spot in degrees.  Flux deficit ∝ sin²(α).
               Solar umbral spots are ~1–3°; active-star spots reach 8–10°.
    Contrast : fs = spot flux / stellar flux.  Solar umbra: fs ≈ 0.52–0.65.
               Mild stars have lighter (warmer) spots: fs ≈ 0.60–0.78.
               Strong stars have darker spots: fs ≈ 0.38–0.62.
    Lifetime : spot survival time in days (spotintime converts to units of
               P_rot internally: taui[i] / prot).  Solar spots last days–months.
    Latitude : signed latitude in degrees.  Solar active belt: ±35°.
    Longitude: initial longitude in [0°, 360°].  Randomised per instance so
               different LCs have different spot–transit phase relationships.
    TimeMax  : time of maximum contrast.  Set to -1 so PSLS draws it
               randomly — this is the correct choice for independent realisations.
    dOmega   : differential rotation Δ Ω / Ω.  Solar value ≈ 0.20.
               Active stars tend toward solid-body rotation (lower dOmega).

    Overlap constraint
    ------------------
    mild Radius   ∈ [1.5, 5.0],  strong Radius   ∈ [3.5, 9.0]  → overlap [3.5, 5.0]
    mild Contrast ∈ [0.60, 0.78], strong Contrast ∈ [0.38, 0.62] → overlap [0.60, 0.62]
    This prevents the classifier from trivially splitting on spot amplitude.
    """

    if activity_class == "mild":
        radius    = round(_uniform(rng, 1.5, 5.0), 2)
        contrast  = round(_uniform(rng, 0.60, 0.78), 3)
        lifetime  = round(_uniform(rng, 20.0, 100.0), 1)
        domega    = round(_uniform(rng, 0.00, 0.25), 3)
        lat_range = 25.0
    else:  # strong
        radius    = round(_uniform(rng, 3.5, 12.0), 2)
        contrast  = round(_uniform(rng, 0.25, 0.55), 3)
        lifetime  = round(_uniform(rng, 30.0, 200.0), 1)
        domega    = round(_uniform(rng, 0.00, 0.15), 3)
        lat_range = 35.0

    latitude  = round(_uniform(rng, -lat_range, lat_range), 2)
    longitude = round(_uniform(rng, 0.0, 360.0), 2)

    return {
        "radius":    radius,
        "contrast":  contrast,
        "lifetime":  lifetime,
        "latitude":  latitude,
        "longitude": longitude,
        "domega":    domega,
    }


FLARE_AMPLITUDE_ALPHA = 2.0   # power-law index for flare amplitude (Davenport 2016)


def sample_flare_params(activity_class: str, rng: np.random.Generator) -> dict:
    """
    Flare parameters.  Grounded in Davenport 2016 (Kepler flare statistics).

    MeanPeriod : mean inter-flare interval in days.
      - mild:  ~3–10 days between flares (low-activity solar-like)
      - strong: ~0.3–2 days (active star; consistent with P_rot < 18 d)

    Amplitude (ppm): power-law distributed p(A) ∝ A^{-α}, α=2.0 — flare
      amplitudes follow a power law (Davenport 2016, Lacy+1976).
      - mild:   200–800 ppm   (detectable but sub-dominant)
      - strong: 800–5000 ppm  (can mimic or swamp short transits)

    UpDown: rise-to-fall time ratio.  Kepler flare morphology: 0.05–0.20.
    MeanDuration: set to -1 so PSLS uses MeanPeriod/5 (its default).
    DurationDispersion: set to -1 so PSLS uses its default scaling.
    """

    if activity_class == "mild":
        mean_period = round(_loguniform(rng, 60.0, 500.0), 2)
        amplitude   = round(_powerlaw(rng, 200.0, 800.0, FLARE_AMPLITUDE_ALPHA), 1)
    else:  # strong
        mean_period = round(_loguniform(rng, 0.3, 2.0), 3)
        amplitude   = round(_powerlaw(rng, 800.0, 5000.0, FLARE_AMPLITUDE_ALPHA), 1)

    updown = round(_uniform(rng, 0.05, 0.20), 3)

    return {
        "enable":               1,
        "mean_period":          mean_period,
        "amplitude":            amplitude,
        "updown":               updown,
        "mean_duration":        -1,   # PSLS default: MeanPeriod / 5
        "duration_dispersion":  -1,   # PSLS default scaling
    }


def sample_inclination(activity_class: str, rng: np.random.Generator) -> float:
    """
    Stellar inclination (degrees from pole-on = 0° to equator-on = 90°).
    Uses the physically correct prior: uniform in cos(i).

    Higher inclination → equatorial spots maximally visible → stronger signal.
    Active classes are biased toward high inclinations to ensure the spot
    signature is prominent enough to challenge the transit classifier.
    """
    if activity_class == "strong":
        return round(_inclination_from_cosi(rng, i_min_deg=45.0, i_max_deg=90.0), 2)
    elif activity_class == "mild":
        return round(_inclination_from_cosi(rng, i_min_deg=25.0, i_max_deg=85.0), 2)


def sample_planet_params(rng: np.random.Generator) -> dict:
    """
    Planet orbital and size parameters.  Both are log-uniform — empirical
    occurrence rates are roughly flat in log-period and log-radius.

    Period range: 3-50 days
      - Lower bound: shorter periods have too few transit durations for PLATO
        noise floor to be meaningful.
      - Upper bound: 50 d gives only ~5 transits in 270 days — hard case.

    Radius range: 1.0-4.0 R_Earth = 0.089-0.356 R_Jup
      - 1 R_Earth → ~80 ppm depth
      - 4 R_Earth → ~1260 ppm depth (well detectable even in active stars)
      - This spans the scientifically interesting detection boundary.

    SemiMajorAxis: derived exactly from Kepler's 3rd law for a 1 M_sun host.
      a [AU] = (P [yr])^(2/3)
      We compute it here to ensure physical self-consistency and write it
      directly into the YAML — PSLS uses this value for transit geometry.

    OrbitalAngle: uniform in [0°, 360°] — randomises orbital phase so
      transit mid-points are uncorrelated across the dataset.
    """
    period_days   = round(_loguniform(rng, 3.0, 50.0), 4)
    radius_rjup   = round(_loguniform(rng, 0.089, 0.356), 5)
    sma_au        = round(kepler_sma(period_days), 5)
    orbital_angle = round(_uniform(rng, 0.0, 360.0), 2)

    return {
        "period_days":    period_days,
        "radius_rjup":    radius_rjup,
        "sma_au":         sma_au,
        "orbital_angle":  orbital_angle,
    }


# ---------------------------------------------------------------------------
# Derived physical quantities
# ---------------------------------------------------------------------------

def kepler_sma(period_days: float, mass_solar: float = 1.0) -> float:
    """Semi-major axis in AU from Kepler's 3rd law: a = (M * P^2)^(1/3)."""
    return (mass_solar * (period_days / 365.25) ** 2) ** (1.0 / 3.0)


def transit_depth_ppm(planet_radius_rjup: float,
                      star_radius_rsun: float = 1.0) -> float:
    """Transit depth in ppm.  1 R_Jup = 0.10045 R_Sun."""
    rp_rsun = planet_radius_rjup * 0.10045
    return (rp_rsun / star_radius_rsun) ** 2 * 1e6


# ---------------------------------------------------------------------------
# YAML template  (static structure — all sampled values filled in build_config)
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "Observation": {
        "QuarterDuration": QUARTER_DURATION,
        "MasterSeed": None,          # filled per LC
        "Gaps": {
            "Enable": 1,
            "Seed": -1,
            "InterQuarterGapDuration": 3.0,
            "RandomGapDuration": 0.0,
            "RandomGapTimeFraction": 0.5,
            "RandomGapStep": 0.0,
            "PeriodicGapCadence": 5.0,
            "PeriodicGapDuration": 20.0,
            "PeriodicGapJitter": 2.0,
            "PeriodicGapStep": 0.0,
        },
    },
    "Instrument": {
        "Sampling": SAMPLING,
        "IntegrationTime": 21,
        "GroupID": [1, 2, 3, 4],
        "NCamera": 6,
        "TimeShift": 6.25,
        "RandomNoise": {
            "Enable": 1,
            "Type": "PLATO_SIMU",
            "NSR": 73.0,
        },
        "Systematics": {
            "Enable": 1,
            "Table": "systematics/PLATO_systematics_BOL_V2.npy",
            "Version": 2,
            "DriftLevel": "low",
            "Seed": -1,
        },
    },
    "Star": {
        "Mag": MAG,
        "ID": None,                      # filled per LC
        "ModelType": "grid",
        "ModelDir": "models",
        "ModelName": "grid_v0.1_ov0-plato.hdf5",
        "ES": "ms",                      # main-sequence only
        "Teff": None,                    # sampled per LC from TEFF_RANGE
        "Logg": None,                    # sampled per LC from LOGG_RANGE
        "SurfaceRotationPeriod": None,   # sampled per LC — controls spot period
        "CoreRotationFreq": 0.0,
        "Inclination": None,             # sampled per LC
    },
    "Oscillations": {
        "Enable": 1,
        # numax and delta_nu are NOT used in grid mode — PSLS reads them
        # from the HDF5 model.  Setting to -1 makes this explicit.
        "numax": -1.0,
        "delta_nu": -1.0,
        "DPI": -1,
        "q": 0.0,
        "SurfaceEffects": 0, # Minimal correction (needed for astroseismology)
        "Seed": -1,
    },
    "Activity": {
        "Enable": 1,
        "Sigma": None,     # sampled per LC
        "Tau": None,       # sampled per LC
        "Seed": -1,
        "Spot": {
            "Enable": 0,
            "Radius":    [2.5],
            "Contrast":  [0.7],
            "Lifetime":  [30],
            "Latitude":  [0.0],
            "Longitude": [0.0],
            "TimeMax":   [-1],
            "Modulation": 0.0,
            "MuSpot": 0.78,
            "MuStar": 0.59,
            "dOmega": 0.0,
            "Seed": -1,
        },
        "Flare": {
            "Enable": 0,
            "MeanPeriod": 30,
            "Amplitude": 500.0,
            "UpDown": 0.1,
            "MeanDuration": -1,
            "DurationDispersion": -1,
            "Seed": -1,
        },
    },
    "Granulation": {
        "Enable": 1,
        "Seed": -1,
        "Type": 1,
    },
    "External": {
        "Enable": 0,
        "FilePath": "examples/external_example.txt",
    },
    "Transit": {
        "Enable": 0,
        "PlanetRadius": 0.1,
        "OrbitalPeriod": 10.0,
        "PlanetSemiMajorAxis": 1.0,
        "OrbitalAngle": 0.0,
        "LimbDarkeningCoefficients": [0.25, 0.75],
    },
}


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(job: dict) -> dict:
    """
    Build a fully-specified PSLS config dict from a pre-sampled job dict.
    All sampling was done in build_job_list — this function only maps the
    sampled values into the YAML structure.
    """
    cfg = yaml.safe_load(yaml.dump(BASE_CONFIG))   # deep copy

    # --- Observation ----------------------------------------------------
    cfg["Observation"]["MasterSeed"] = job["seed"]

    # --- Star / grid selection -----------------------------------------
    cfg["Star"]["ID"]                   = job["star_id"]
    cfg["Star"]["Teff"]                 = job["teff"]
    cfg["Star"]["Logg"]                 = job["logg"]
    cfg["Star"]["SurfaceRotationPeriod"] = job["rotation_period"]
    cfg["Star"]["Inclination"]           = job["inclination"]

    # --- Activity: stochastic component --------------------------------
    cfg["Activity"]["Sigma"] = job["sigma"]
    cfg["Activity"]["Tau"]   = job["tau"]

    # --- Activity: spots -----------------------------------------------
    sp = job["spot"]
    if sp is not None:
        cfg["Activity"]["Spot"]["Enable"]    = 1
        cfg["Activity"]["Spot"]["Radius"]    = [sp["radius"]]
        cfg["Activity"]["Spot"]["Contrast"]  = [sp["contrast"]]
        cfg["Activity"]["Spot"]["Lifetime"]  = [sp["lifetime"]]
        cfg["Activity"]["Spot"]["Latitude"]  = [sp["latitude"]]
        cfg["Activity"]["Spot"]["Longitude"] = [sp["longitude"]]
        cfg["Activity"]["Spot"]["TimeMax"]   = [-1]    # PSLS randomises this
        cfg["Activity"]["Spot"]["dOmega"]    = sp["domega"]
    else:
        cfg["Activity"]["Spot"]["Enable"] = 0

    # --- Activity: flares ----------------------------------------------
    fl = job["flare"]
    cfg["Activity"]["Flare"]["Enable"]              = fl["enable"]
    cfg["Activity"]["Flare"]["MeanPeriod"]          = fl["mean_period"]
    cfg["Activity"]["Flare"]["Amplitude"]           = fl["amplitude"]
    cfg["Activity"]["Flare"]["UpDown"]              = fl["updown"]
    cfg["Activity"]["Flare"]["MeanDuration"]        = fl["mean_duration"]
    cfg["Activity"]["Flare"]["DurationDispersion"]  = fl["duration_dispersion"]

    # --- Planet / transit ----------------------------------------------
    pl = job["planet"]
    if pl is not None:
        cfg["Transit"]["Enable"]              = 1
        cfg["Transit"]["PlanetRadius"]        = pl["radius_rjup"]
        cfg["Transit"]["OrbitalPeriod"]       = pl["period_days"]
        cfg["Transit"]["PlanetSemiMajorAxis"] = pl["sma_au"]
        cfg["Transit"]["OrbitalAngle"]        = pl["orbital_angle"]
    else:
        cfg["Transit"]["Enable"] = 0

    return cfg


# ---------------------------------------------------------------------------
# Job list builder  (all sampling lives here)
# ---------------------------------------------------------------------------

def build_job_list(n_per_class: int, rng_seed: int = 42) -> list[dict]:
    """
    Build the full list of simulation jobs.  All parameter sampling is done
    here so that the complete parameter set is written to metadata.csv and
    configs/, making the pipeline fully reproducible from the master seed.

    Sampling strategy
    -----------------
    - All continuous parameters use appropriate distributions (uniform or
      log-uniform) rather than fixed grids — this forces the classifier to
      learn physical phenomena rather than memorise discrete parameter values.
    - Stellar (Teff, Logg) are drawn uniformly inside the CESAM2K MS grid
      region; PSLS snaps to the nearest model automatically.
    - Mild and strong activity distributions overlap deliberately on Sigma,
      spot Radius, and spot Contrast so the classifier must attend to transit
      morphology rather than activity level.
    """
    rng   = np.random.default_rng(rng_seed)
    jobs  = []
    star_id = STAR_ID_BASE

    def _add(label: int, activity_class: str, planet_cfg: dict | None):
        nonlocal star_id
        seed = int(rng.integers(0, 2**31))

        teff, logg  = sample_stellar_params(rng)
        prot        = sample_rotation_period(activity_class, rng)
        sigma       = sample_activity_sigma(activity_class, rng)
        tau         = sample_activity_tau(activity_class, rng)
        spot        = sample_spot_params(activity_class, rng)
        flare       = sample_flare_params(activity_class, rng)
        inclination = sample_inclination(activity_class, rng)

        jobs.append(dict(
            star_id         = star_id,
            seed            = seed,
            label           = label,
            activity_class  = activity_class,
            teff            = teff,
            logg            = logg,
            rotation_period = prot,
            sigma           = sigma,
            tau             = tau,
            spot            = spot,
            flare           = flare,
            inclination     = inclination,
            planet          = planet_cfg,
        ))
        star_id += 1

    # ---- Class 0: mild activity, no planet --------------------------------
    for _ in range(n_per_class):
        _add(label=0, activity_class="mild", planet_cfg=None)

    # ---- Class 1: strong activity, no planet ---------------------------
    for _ in range(n_per_class):
        _add(label=1, activity_class="strong", planet_cfg=None)

    # ---- Class 2: planet + mild activity --------------------------------
    for _ in range(n_per_class):
        planet = sample_planet_params(rng)
        _add(label=2, activity_class="mild", planet_cfg=planet)

    # ---- Class 3: planet + strong activity ------------------------------
    for _ in range(n_per_class):
        planet = sample_planet_params(rng)
        _add(label=3, activity_class="strong", planet_cfg=planet)

    rng.shuffle(jobs)
    return jobs


# ---------------------------------------------------------------------------
# PSLS runner
# ---------------------------------------------------------------------------

def run_psls(cfg: dict, work_dir: Path) -> Path | None:
    """
    Write a PSLS YAML, run psls.py, and return the path to the output .dat file.
    Returns None on failure.
    """
    cfg_path = work_dir / "sim.yaml"
    with open(cfg_path, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False)

    star_name = f"{cfg['Star']['ID']:010d}"
    out_path  = work_dir / f"{star_name}.dat"

    current_script_dir = Path(__file__).resolve().parent
    project_root = current_script_dir.parent
    psls_dir = project_root / "psls"
    script_path = psls_dir / "psls.py"

    #print(script_path)

    try:
        result = subprocess.run(
            ["python3", str(script_path), "-o", str(work_dir.resolve()), str(cfg_path.resolve())],
            cwd=str(psls_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"\n[PSLS ERROR] returncode={result.returncode}")
            print(f"  stdout: {result.stdout[-500:]}")
            print(f"  stderr: {result.stderr[-500:]}")
            return None
        if not out_path.exists():
            candidates = list(work_dir.glob(f"{star_name}*.dat"))
            if not candidates:
                # Check what files were actually produced
                produced = list(work_dir.glob("*"))
                print(f"\n[PSLS WARNING] Expected {out_path.name} but found: {[f.name for f in produced]}")
                return None
            out_path = candidates[0]
        return out_path
    except subprocess.TimeoutExpired:
        print("\n[PSLS ERROR] subprocess timed out after 300 s")
        return None
    except FileNotFoundError:
        print("\n[PSLS ERROR] psls.py not found.")
        return None


def load_psls_output(dat_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse PSLS .dat output → (time_days, relative flux variation) arrays."""
    try:
        data = np.loadtxt(dat_path, comments="#")
        if data.ndim < 2 or data.shape[0] == 0 or data.shape[1] < 2:
            return None
        time_s   = data[:, 0]
        flux_var = data[:, 1] * 1e-4
        return time_s / 86400.0, flux_var
    except Exception as exc:
        print(f"\n[LOAD ERROR] {dat_path}: {exc}")
        return None

# ---------------------------------------------------------------------------
# SNR proxy  (RMS of folded residuals after naive period search)
# ---------------------------------------------------------------------------

def snr_proxy(time: np.ndarray, flux: np.ndarray, period: float | None) -> float:
    """
    Estimate transit SNR:
      • If period known: use box-fold depth / scatter ratio
      • Otherwise: return ratio of low- to high-frequency power
    """
    if len(flux) < 10:
        return 0.0
    if period is None or period <= 0:
        # low/high power ratio as activity proxy
        fft = np.abs(np.fft.rfft(flux - flux.mean())) ** 2
        n   = len(fft)
        low = fft[: n // 10].mean()
        hi  = fft[n // 2 :].mean() + 1e-9
        return float(low / hi)

    # Simple box-fold depth estimate
    phase = (time % period) / period
    in_transit = phase < 0.05
    if in_transit.sum() < 3:
        return 0.0
    depth   = flux[~in_transit].mean() - flux[in_transit].mean()
    scatter = flux[~in_transit].std() + 1e-9
    return float(depth / scatter)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    out_dir = Path(args.out_dir)
    lc_dir  = out_dir / "lightcurves"
    cfg_dir = out_dir / "configs"
    lc_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_job_list(n_per_class=args.n_per_class, rng_seed=args.seed)
    print(f"Total jobs : {len(jobs)}  ({args.n_per_class} per class x 4 classes)")
    print(f"Stellar grid : Teff {TEFF_RANGE[0]:.0f}-{TEFF_RANGE[1]:.0f} K, "
          f"Logg {LOGG_RANGE[0]:.2f}-{LOGG_RANGE[1]:.2f}  (sampled per LC)")
    print(f"Instrument : Mag={MAG}, cadence={SAMPLING} s, baseline=270 d")

    records = []
    failed  = 0

    for job in tqdm(jobs, desc="Simulating"):
        cfg   = build_config(job)
        lc_id = f"{job['star_id']:010d}"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dat_path = run_psls(cfg, tmp_path)

            if dat_path is None:
                failed += 1
                continue

            result = load_psls_output(dat_path)
            if result is None:
                failed += 1
                continue

            time_days, flux_var = result
            shutil.copy2(dat_path, lc_dir / f"{lc_id}.dat")

        # Save config YAML for reproducibility
        cfg_file = cfg_dir / f"{lc_id}.yaml"
        with open(cfg_file, "w") as fh:
            yaml.dump(cfg, fh)

        # ---------------------------------------------------------------
        # Build metadata row  (all sampled parameters logged)
        # ---------------------------------------------------------------
        pl = job["planet"]
        sp = job["spot"]
        fl = job["flare"]

        records.append(dict(
            file_id              = lc_id,
            label                = job["label"],
            activity_class       = job["activity_class"],
            # stellar
            teff                 = job["teff"],
            logg                 = job["logg"],
            rotation_period_days = job["rotation_period"],
            inclination_deg      = job["inclination"],
            # stochastic activity
            sigma_ppm            = job["sigma"],
            tau_days             = job["tau"],
            # spots
            spot_enable          = int(sp is not None),
            spot_radius_deg      = sp["radius"]    if sp else None,
            spot_contrast        = sp["contrast"]  if sp else None,
            spot_lifetime_days   = sp["lifetime"]  if sp else None,
            spot_latitude_deg    = sp["latitude"]  if sp else None,
            spot_longitude_deg   = sp["longitude"] if sp else None,
            spot_domega          = sp["domega"]    if sp else None,
            # flares
            flare_enable         = fl["enable"],
            flare_mean_period_days = fl["mean_period"] if fl["enable"] else None,
            flare_amplitude_ppm  = fl["amplitude"] if fl["enable"] else None,
            # planet
            planet_present       = int(pl is not None),
            planet_period_days   = pl["period_days"]  if pl else None,
            planet_radius_rjup   = pl["radius_rjup"]  if pl else None,
            planet_radius_earth  = pl["radius_rjup"] * 11.0 if pl else None,
            planet_sma_au        = pl["sma_au"]        if pl else None,
            transit_depth_ppm    = transit_depth_ppm(pl["radius_rjup"]) if pl else None,
            # LC metadata
            seed                 = job["seed"],
            n_points             = len(time_days),
            duration_days        = float(time_days[-1] - time_days[0]) if len(time_days) > 1 else 0.0,
            snr_proxy            = snr_proxy(time_days, flux_var, pl["period_days"] if pl else None),
        ))

    # -------------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------------
    meta_path = out_dir / "metadata.csv"
    df = pd.DataFrame(records)
    df.to_csv(meta_path, index=False)

    print(f"\n✓ Generated {len(records)} light curves → {lc_dir}")
    print(f"✓ Metadata  → {meta_path}")
    if failed:
        print(f"✗ {failed} simulations failed (check PSLS installation / stderr above)")

    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame):
    if df.empty:
        print("\n--- Dataset summary: no light curves generated ---")
        return
    print("\n--- Dataset summary ---")
    print(df.groupby(["label", "activity_class"]).agg(
        count          = ("file_id",        "count"),
        planet_frac    = ("planet_present",  "mean"),
        teff_mean      = ("teff",            "mean"),
        logg_mean      = ("logg",            "mean"),
        prot_mean      = ("rotation_period_days", "mean"),
        sigma_median   = ("sigma_ppm",       "median"),
        depth_median   = ("transit_depth_ppm", "median"),
    ).round(2).to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Generate PSLS light curve dataset for transit ML."
    )
    parser.add_argument(
        "--n_per_class", type=int, default=1500,
        help="Number of light curves per class (default: 1500 → ~6000 total)"
    )
    parser.add_argument(
        "--out_dir", default=Path(__file__).resolve().parent.parent / "data",
        help="Output directory (default: project/data)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed for parameter sampling (default: 42)"
    )
    args = parser.parse_args()
    run_pipeline(args)
