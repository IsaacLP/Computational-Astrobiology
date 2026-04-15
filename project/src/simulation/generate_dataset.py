"""
Phase 2 — PLATO Transit Detection Dataset Generator
=====================================================
Generates ~6000–7000 labeled light curves across 4 classes using PSLS:
  Class 0: quiet star, no planet
  Class 1: strong activity only  (false-positive regime)
  Class 2: planet + mild activity
  Class 3: planet + strong activity

Usage
-----
  python generate_dataset.py                   # full run (~6000 LCs)
  python generate_dataset.py --n_per_class 50  # quick smoke-test
  python generate_dataset.py --out_dir /data/plato_lcs

The script writes:
  <out_dir>/lightcurves/   ← one .npy per LC  (time, flux in ppm)
  <out_dir>/metadata.csv   ← one row per LC with all labels & parameters
  <out_dir>/configs/       ← the PSLS YAML used for each LC (reproducibility)
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

# Stellar rotation periods (days) — split into "mild" and "strong" activity
ROTATION_MILD   = [15, 20, 25, 30]   # slower rotation → less activity
ROTATION_STRONG = [5,  8, 12]        # fast rotation → more activity

# Spot amplitude proxies: encoded as (Radius_deg, Contrast, Lifetime_days)
SPOT_MILD   = dict(radius=2.5, contrast=0.85, lifetime=30)
SPOT_STRONG = dict(radius=4.5, contrast=0.65, lifetime=15)

# Flare settings (MeanPeriod in days, Amplitude in ppm)
FLARE_NONE   = dict(enable=0, mean_period=30, amplitude=500)
FLARE_MILD   = dict(enable=1, mean_period=5,  amplitude=500)
FLARE_STRONG = dict(enable=1, mean_period=1,  amplitude=2500)

# Planet orbital periods (days) and radii (Jupiter radii; 1 R_Jup ≈ 11 R_Earth)
PLANET_PERIODS = [3, 5, 10, 20, 35, 50]           # days
# 0.8–4 R_Earth → 0.073–0.364 R_Jup
PLANET_RADII   = [0.073, 0.12, 0.18, 0.27, 0.364]  # R_Jup
IMPACT_PARAMS  = [0.0, 0.3, 0.6, 0.8]              # b  (controls semi-major axis angle)

# Observation setup (matching PSLS defaults / Phase 1 notes)
QUARTER_DURATION = [90.0, 90.0, 90.0]  # 3 × 90 days = 270 days
SAMPLING         = 25                  # seconds
TEFF             = 5778.               # K  — solar effective temperature
LOGG             = 4.438               # cgs — solar surface gravity; sits in the
                                       #  dense part of the CESAM2K grid (4.2–4.6)
MAG              = 10.0                # V magnitude — bright PLATO-like target
STAR_ID_BASE     = 0000                # arbitrary base for IDs


# ---------------------------------------------------------------------------
# YAML template builder
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
            "DriftLevel": "any",
            "Seed": -1,
        },
    },
    "Star": {
        "Mag": MAG,
        "ID": None,                  # filled per LC
        "ModelType": "grid",
        "ModelDir": "models",            
        "ModelName": "grid_v0.1_ov0-plato.hdf5",         
        "ES": "ms",
        "Teff": TEFF,
        "Logg": LOGG,
        "SurfaceRotationPeriod": 0.0,   # filled per LC
        "CoreRotationFreq": 0.0,
        "Inclination": 60.0,
    },
    "Oscillations": {
        "Enable": 1,
        # Not used when using grid model
        "numax": 3090.0,
        "delta_nu": 135.0,
        "DPI": -1,
        "q": 0.0,
        "SurfaceEffects": 1,         # surface corrections enabled for MS star
        "Seed": -1,
    },
    "Activity": {
        "Enable": 0,
        "Sigma": 40.0,
        "Tau": 0.2,
        "Seed": -1,
        "Spot": {
            "Enable": 0,
            "dOmega": 0.0,
            "MuStar": 0.59,
            "MuSpot": 0.78,
            "Radius": [2.5],
            "Latitude": [20.0],
            "Longitude": [0.0],
            "Lifetime": [30],
            "TimeMax": [-1],
            "Contrast": [0.80],
            "Modulation": 0.0,
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
        "Type": 1,
        "Seed": -1,
    },
    "Transit": {
        "Enable": 0,
        "PlanetRadius": 0.1,         # R_Jup — filled per LC
        "OrbitalPeriod": 10.0,       # days  — filled per LC
        "PlanetSemiMajorAxis": 1.0,  # AU    — filled per LC
        "OrbitalAngle": 0.0,
        "LimbDarkeningCoefficients": [0.25, 0.75],
    },
    "External": {
        "Enable": 0,
        "FilePath": "examples/external_example.txt",
    },
}


def kepler_sma(period_days: float, mass_solar: float = 1.0) -> float:
    """Return semi-major axis in AU from Kepler's 3rd law."""
    return (mass_solar * (period_days / 365.25) ** 2) ** (1.0 / 3.0)


def transit_depth(planet_radius_rjup: float, star_radius_rsun: float = 1.0) -> float:
    """Return transit depth (ppm) given radii."""
    # 1 R_Jup = 0.10045 R_Sun
    rp_rsun = planet_radius_rjup * 0.10045
    return (rp_rsun / star_radius_rsun) ** 2 * 1e6


def build_config(
    star_id: int,
    seed: int,
    activity_class: str,          # "quiet" | "mild" | "strong"
    rotation_period: float,
    planet: dict | None,          # None = no transit
) -> dict:
    """Build a PSLS config dict for one light curve."""
    cfg = yaml.safe_load(yaml.dump(BASE_CONFIG))   # deep copy

    cfg["Observation"]["MasterSeed"] = seed
    cfg["Star"]["ID"] = star_id
    cfg["Star"]["SurfaceRotationPeriod"] = float(rotation_period)

    # ------------------------------------------------------------------
    # Activity components
    # ------------------------------------------------------------------
    if activity_class == "quiet":
        # Minimal stochastic background, no spots, no flares
        cfg["Activity"]["Enable"] = 1
        cfg["Activity"]["Sigma"] = 15.0   # small stochastic component
        cfg["Activity"]["Tau"] = 0.1
        cfg["Activity"]["Spot"]["Enable"] = 0
        cfg["Activity"]["Flare"]["Enable"] = 0

    elif activity_class == "mild":
        cfg["Activity"]["Enable"] = 1
        cfg["Activity"]["Sigma"] = 40.0
        cfg["Activity"]["Tau"] = 0.2
        sp = SPOT_MILD
        cfg["Activity"]["Spot"]["Enable"] = 1
        cfg["Activity"]["Spot"]["Radius"]   = [sp["radius"]]
        cfg["Activity"]["Spot"]["Contrast"] = [sp["contrast"]]
        cfg["Activity"]["Spot"]["Lifetime"] = [int(sp["lifetime"])]
        fl = FLARE_MILD
        cfg["Activity"]["Flare"]["Enable"]      = fl["enable"]
        cfg["Activity"]["Flare"]["MeanPeriod"]  = fl["mean_period"]
        cfg["Activity"]["Flare"]["Amplitude"]   = float(fl["amplitude"])

    elif activity_class == "strong":
        cfg["Activity"]["Enable"] = 1
        cfg["Activity"]["Sigma"] = 80.0
        cfg["Activity"]["Tau"] = 0.5
        sp = SPOT_STRONG
        cfg["Activity"]["Spot"]["Enable"] = 1
        cfg["Activity"]["Spot"]["Radius"]   = [sp["radius"]]
        cfg["Activity"]["Spot"]["Contrast"] = [sp["contrast"]]
        cfg["Activity"]["Spot"]["Lifetime"] = [int(sp["lifetime"])]
        fl = FLARE_STRONG
        cfg["Activity"]["Flare"]["Enable"]      = fl["enable"]
        cfg["Activity"]["Flare"]["MeanPeriod"]  = fl["mean_period"]
        cfg["Activity"]["Flare"]["Amplitude"]   = float(fl["amplitude"])

    # ------------------------------------------------------------------
    # Planet / transit
    # ------------------------------------------------------------------
    if planet is not None:
        cfg["Transit"]["Enable"] = 1
        cfg["Transit"]["PlanetRadius"]        = float(planet["radius_rjup"])
        cfg["Transit"]["OrbitalPeriod"]       = float(planet["period_days"])
        cfg["Transit"]["PlanetSemiMajorAxis"] = float(
            kepler_sma(planet["period_days"])
        )

    return cfg


# ---------------------------------------------------------------------------
# Job list builder
# ---------------------------------------------------------------------------

def build_job_list(n_per_class: int, rng_seed: int = 42) -> list[dict]:
    """Return a list of job dicts that span the parameter grid."""
    rng = np.random.default_rng(rng_seed)
    jobs = []
    star_id = STAR_ID_BASE

    def _add(label, activity_class, rotation, planet_cfg):
        nonlocal star_id
        seed = int(rng.integers(0, 2**30))
        jobs.append(
            dict(
                star_id=star_id,
                seed=seed,
                label=label,
                activity_class=activity_class,
                rotation_period=rotation,
                planet=planet_cfg,
            )
        )
        star_id += 1

    # ---- Class 0: quiet star, no planet --------------------------------
    for _ in range(n_per_class):
        prot = float(rng.choice(ROTATION_MILD + ROTATION_STRONG))
        _add(label=0, activity_class="quiet", rotation=prot, planet_cfg=None)

    # ---- Class 1: strong activity, no planet ---------------------------
    for _ in range(n_per_class):
        prot = float(rng.choice(ROTATION_STRONG))
        _add(label=1, activity_class="strong", rotation=prot, planet_cfg=None)

    # ---- Class 2: planet + mild activity --------------------------------
    periods = list(rng.choice(PLANET_PERIODS, size=n_per_class))
    radii   = list(rng.choice(PLANET_RADII,   size=n_per_class))
    for per, rad in zip(periods, radii):
        prot = float(rng.choice(ROTATION_MILD))
        planet = dict(period_days=float(per), radius_rjup=float(rad))
        _add(label=2, activity_class="mild", rotation=prot, planet_cfg=planet)

    # ---- Class 3: planet + strong activity ------------------------------
    periods = list(rng.choice(PLANET_PERIODS, size=n_per_class))
    radii   = list(rng.choice(PLANET_RADII,   size=n_per_class))
    for per, rad in zip(periods, radii):
        prot = float(rng.choice(ROTATION_STRONG))
        planet = dict(period_days=float(per), radius_rjup=float(rad))
        _add(label=3, activity_class="strong", rotation=prot, planet_cfg=planet)

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
    project_root = current_script_dir.parent.parent
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
            print(f"\n[PSLS ERROR] star_id={cfg['Star']['ID']}")
            print(f"  stdout: {result.stdout[-500:] if result.stdout else '(empty)'}")
            print(f"  stderr: {result.stderr[-500:] if result.stderr else '(empty)'}")
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
        print(f"\n[PSLS TIMEOUT] star_id={cfg['Star']['ID']} exceeded 300s")
        return None
    except FileNotFoundError:
        print("\n[PSLS NOT FOUND] 'psls.py' is not on PATH.")
        print("  Check: which psls.py   or   pip show psls")
        return None


def load_psls_output(dat_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse PSLS .dat output → (time_days, flux_ppm) arrays."""
    try:
        data = np.loadtxt(dat_path, comments="#")
        if data.ndim < 2 or data.shape[1] < 2:
            return None
        time_s   = data[:, 0]
        flux_ppm = data[:, 1]
        return time_s / 86400.0, flux_ppm   # time in days
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    out_dir   = Path(args.out_dir)
    lc_dir    = out_dir / "lightcurves"
    cfg_dir   = out_dir / "configs"
    lc_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_job_list(n_per_class=args.n_per_class, rng_seed=args.seed)
    print(f"Total jobs: {len(jobs)}  ({args.n_per_class} per class × 4 classes)")
    print(f"Star: Mag={MAG}, Teff={TEFF} K, logg={LOGG}")

    records = []
    failed  = 0

    for job in tqdm(jobs, desc="Simulating"):
        cfg = build_config(
            star_id=job["star_id"],
            seed=job["seed"],
            activity_class=job["activity_class"],
            rotation_period=job["rotation_period"],
            planet=job["planet"],
        )

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

            time_days, flux_ppm = result

        # -----------------------------------------------------------
        # Save light curve
        # -----------------------------------------------------------
        lc_id  = f"{job['star_id']:010d}"
        lc_file = lc_dir / f"{lc_id}.npy"
        np.save(lc_file, np.stack([time_days, flux_ppm], axis=0))

        # Save config for reproducibility
        cfg_file = cfg_dir / f"{lc_id}.yaml"
        with open(cfg_file, "w") as fh:
            yaml.dump(cfg, fh)

        # -----------------------------------------------------------
        # Build metadata row
        # -----------------------------------------------------------
        pl = job["planet"]
        period     = pl["period_days"]  if pl else None
        radius_rjup = pl["radius_rjup"] if pl else None
        depth_ppm  = transit_depth(radius_rjup) if radius_rjup else None

        records.append(
            dict(
                file_id=lc_id,
                label=job["label"],
                activity_class=job["activity_class"],
                rotation_period_days=job["rotation_period"],
                spot_amplitude=SPOT_STRONG["radius"] if job["activity_class"] == "strong" else (
                    SPOT_MILD["radius"] if job["activity_class"] == "mild" else 0.0
                ),
                spot_contrast=SPOT_STRONG["contrast"] if job["activity_class"] == "strong" else (
                    SPOT_MILD["contrast"] if job["activity_class"] == "mild" else 1.0
                ),
                flare_rate=FLARE_STRONG["mean_period"] if job["activity_class"] == "strong" else (
                    FLARE_MILD["mean_period"] if job["activity_class"] == "mild" else 0
                ),
                planet_present=int(pl is not None),
                planet_period_days=period,
                planet_radius_rjup=radius_rjup,
                planet_radius_earth=radius_rjup * 11.0 if radius_rjup else None,
                transit_depth_ppm=depth_ppm,
                seed=job["seed"],
                n_points=len(time_days),
                duration_days=float(time_days[-1] - time_days[0]) if len(time_days) > 1 else 0,
            )
        )

    # -------------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------------
    meta_path = out_dir / "metadata.csv"
    df = pd.DataFrame(records)
    df.to_csv(meta_path, index=False)

    print(f"\n✓ Generated {len(records)} light curves → {lc_dir}")
    print(f"✓ Metadata → {meta_path}")
    if failed:
        print(f"✗ {failed} simulations failed (check PSLS installation)")

    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame):
    if df.empty:
        print("\n--- Dataset summary: no light curves generated ---")
        return
    print("\n--- Dataset summary ---")
    print(df.groupby("label").agg(
        count=("file_id", "count"),
        planet_frac=("planet_present", "mean")
    ).to_string())


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
        "--out_dir", default="dataset",
        help="Output directory (default: ./dataset)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed for parameter sampling (default: 42)"
    )
    args = parser.parse_args()
    run_pipeline(args)