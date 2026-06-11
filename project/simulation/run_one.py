"""
Run PSLS for a single star ID and save just its .npz file.

Reuses the directory layout and PSLS-invocation conventions of
generate_dataset.py:
    <out_dir>/configs/<id:010d>.yaml    ← input config (already generated)
    <out_dir>/lightcurves/<id:010d>.npz ← output written here (compressed)

Usage
-----
  python run_one.py 42
  python run_one.py 42 --out_dir /path/to/data
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import yaml


def run_psls(cfg: dict, work_dir: Path) -> Path | None:
    """Write a PSLS YAML, run psls.py, return path to the output .dat (or None)."""
    cfg_path = work_dir / "sim.yaml"
    with open(cfg_path, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False)

    star_name = f"{cfg['Star']['ID']:010d}"
    out_path = work_dir / f"{star_name}.dat"

    # project/simulation/run_one.py  →  project_root = parent of simulation/
    current_script_dir = Path(__file__).resolve().parent
    project_root = current_script_dir.parent
    psls_dir = project_root / "psls"
    script_path = psls_dir / "psls.py"

    try:
        result = subprocess.run(
            ["python3", str(script_path), "-o", str(work_dir.resolve()),
             str(cfg_path.resolve())],
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
                produced = list(work_dir.glob("*"))
                print(f"\n[PSLS WARNING] Expected {out_path.name} but found: "
                      f"{[f.name for f in produced]}")
                return None
            out_path = candidates[0]
        return out_path
    except subprocess.TimeoutExpired:
        print("\n[PSLS ERROR] subprocess timed out after 300 s")
        return None
    except FileNotFoundError:
        print("\n[PSLS ERROR] psls.py not found.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run PSLS for a single star ID and save its .dat file."
    )
    parser.add_argument("star_id", type=int, help="Star ID (integer)")
    parser.add_argument(
        "--out_dir",
        default=Path(__file__).resolve().parent.parent / "data",
        help="Output directory (default: project/data)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    lc_dir = out_dir / "lightcurves"
    cfg_dir = out_dir / "configs"
    lc_dir.mkdir(parents=True, exist_ok=True)

    lc_id = f"{args.star_id:010d}"
    cfg_file = cfg_dir / f"{lc_id}.yaml"

    if not cfg_file.exists():
        print(f"[ERROR] Config not found: {cfg_file}")
        return

    with open(cfg_file) as fh:
        cfg = yaml.safe_load(fh)

    with tempfile.TemporaryDirectory() as tmp:
        dat_path = run_psls(cfg, Path(tmp))
        if dat_path is None:
            print(f"✗ PSLS failed for star {lc_id}")
            return
        dest = lc_dir / f"{lc_id}.npz"
        raw = np.loadtxt(dat_path)
        np.savez_compressed(dest, time=raw[:, 0], flux=raw[:, 1], flag=raw[:, 2].astype(np.int8))

    print(f"✓ {lc_id}.npz → {dest}")


if __name__ == "__main__":
    main()