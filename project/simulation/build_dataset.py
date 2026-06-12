"""
build_dataset.py — Pack compressed .npz light curves into a single HDF5 archive
================================================================================
Reads all light curves listed in metadata.csv, block-averages each to a
user-specified cadence, and writes them into one HDF5 file with embedded
metadata for convenient ML training.

Usage
-----
  python build_dataset.py --downsample 12           # 25 s x 12 = 5 min cadence
  python build_dataset.py --downsample 72           # 25 s x 72 = 30 min cadence
  python build_dataset.py --downsample 12 --out /data/dataset.h5

Output layout
-------------
  dataset.h5
  ├── time          (n_time,)       float32 — seconds from start
  ├── flux          (n_lc, n_time)  float32 — raw ppm, gzip-4 compressed
  ├── flag          (n_lc, n_time)  int8    — PSLS quality flag
  ├── label         (n_lc,)         int8    — class 0-3
  ├── file_id       (n_lc,)         str     — "0000000042"
  └── meta/         — one dataset per metadata.csv column (see below)

Load example
------------
  import h5py, numpy as np
  with h5py.File("dataset.h5", "r") as f:
      X = f["flux"][:]        # (n_lc, n_time) float32
      y = f["label"][:]       # (n_lc,) int8
      t = f["time"][:]        # (n_time,) float32
"""

import argparse
import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


# Columns written to meta/ as int8 (everything else → float32, except seed → int64)
_INT8_COLS  = {"spot_enable", "spot_count", "flare_enable", "planet_present"}
_INT64_COLS = {"seed"}
# Columns handled specially at root level — excluded from meta/
_SKIP_COLS  = {"file_id", "label", "activity_class"}


# ---------------------------------------------------------------------------
# Block-average helpers
# ---------------------------------------------------------------------------

def block_average(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average 1-D array by integer factor; trims the tail."""
    n = (len(arr) // factor) * factor
    return arr[:n].reshape(-1, factor).mean(axis=1)


def block_average_flag(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average flag: take majority value, cast to int8."""
    n = (len(arr) // factor) * factor
    return arr[:n].reshape(-1, factor).mean(axis=1).round().astype(np.int8)


# ---------------------------------------------------------------------------
# Probing helpers
# ---------------------------------------------------------------------------

def determine_output_length(lc_dir: Path, file_ids: list, factor: int) -> tuple[int, np.ndarray]:
    """
    Load the first available .npz and return (n_time_downsampled, time_array).
    Raises RuntimeError if no .npz files are found.
    """
    for fid in file_ids:
        npz = lc_dir / f"{fid}.npz"
        if npz.exists():
            data = np.load(npz)
            t_ds = block_average(data["time"].astype(np.float32), factor)
            return len(t_ds), t_ds
    raise RuntimeError(
        f"No .npz files found in {lc_dir}. "
        "Run generate_dataset.py first to produce the light curves."
    )


# ---------------------------------------------------------------------------
# Metadata writer
# ---------------------------------------------------------------------------

def write_meta_group(hf: h5py.File, df: pd.DataFrame):
    """
    Write all numeric metadata columns into hf['meta/'].
    Encodes activity_class as int8 (0=mild, 1=strong).
    Skips columns in _SKIP_COLS and warns about unrecognised dtypes.
    """
    grp = hf.require_group("meta")

    # activity_class: string → int8
    if "activity_class" in df.columns:
        enc = df["activity_class"].map({"mild": 0, "strong": 1}).astype(np.int8).values
        ds = grp.create_dataset("activity_class", data=enc, dtype=np.int8)
        ds.attrs["encoding"] = "0=mild,1=strong"

    for col in df.columns:
        if col in _SKIP_COLS:
            continue
        if col in _INT8_COLS:
            grp.create_dataset(col, data=df[col].values.astype(np.int8), dtype=np.int8)
        elif col in _INT64_COLS:
            grp.create_dataset(col, data=df[col].values.astype(np.int64), dtype=np.int64)
        else:
            try:
                grp.create_dataset(col, data=df[col].values.astype(np.float32), dtype=np.float32)
            except (ValueError, TypeError):
                print(f"  [WARN] Skipping column '{col}' — cannot convert to float32")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_build(args):
    lc_dir   = Path(args.lc_dir)
    meta_csv = Path(args.meta_csv)
    out_path = Path(args.out)
    factor   = args.downsample

    if not lc_dir.is_dir():
        raise SystemExit(f"[ERROR] lc_dir not found: {lc_dir}")
    if not meta_csv.exists():
        raise SystemExit(f"[ERROR] metadata CSV not found: {meta_csv}")

    df = pd.read_csv(meta_csv, dtype={"file_id": str})
    file_ids = df["file_id"].tolist()
    n_lc = len(file_ids)
    print(f"Metadata rows : {n_lc}")
    print(f"Downsample    : {factor}x (25 s x {factor} = {25 * factor} s cadence)")

    print("Probing first .npz to determine output shape...")
    n_time, time_arr = determine_output_length(lc_dir, file_ids, factor)
    print(f"Output shape  : ({n_lc}, {n_time})  — {n_time} timesteps per LC")

    if out_path.exists():
        print(f"[WARN] Overwriting existing file: {out_path}")
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0
    chunk   = (1, n_time)

    with h5py.File(out_path, "w") as hf:
        # Pre-allocate flux and flag (write row by row in the loop)
        flux_ds = hf.create_dataset(
            "flux", shape=(n_lc, n_time), dtype=np.float32,
            chunks=chunk, compression="gzip", compression_opts=4,
        )
        flag_ds = hf.create_dataset(
            "flag", shape=(n_lc, n_time), dtype=np.int8,
            chunks=chunk, compression="gzip", compression_opts=4,
        )

        labels   = np.empty(n_lc, dtype=np.int8)
        file_ids_out = []

        for i, row in enumerate(tqdm(df.itertuples(index=False), total=n_lc, desc="Building")):
            fid  = row.file_id
            npz  = lc_dir / f"{fid}.npz"

            if not npz.exists():
                print(f"\n  [WARN] Missing: {npz.name} — skipping")
                skipped += 1
                labels[i] = -1
                file_ids_out.append(fid)
                continue

            data = np.load(npz)
            raw_flux = data["flux"].astype(np.float32)
            raw_flag = data["flag"]

            ds_flux = block_average(raw_flux, factor)
            ds_flag = block_average_flag(raw_flag, factor)

            # Length guard: trim if longer, skip if shorter
            if len(ds_flux) > n_time:
                ds_flux = ds_flux[:n_time]
                ds_flag = ds_flag[:n_time]
            elif len(ds_flux) < n_time:
                print(f"\n  [WARN] {fid}: expected {n_time} pts after downsampling, "
                      f"got {len(ds_flux)} — skipping")
                skipped += 1
                labels[i] = -1
                file_ids_out.append(fid)
                continue

            flux_ds[i] = ds_flux
            flag_ds[i] = ds_flag
            labels[i]  = row.label
            file_ids_out.append(fid)
            written += 1

        # Root-level datasets
        hf.create_dataset("time",  data=time_arr,                        dtype=np.float32)
        hf.create_dataset("label", data=labels,                          dtype=np.int8)
        hf.create_dataset(
            "file_id",
            data=np.array(file_ids_out, dtype=h5py.special_dtype(vlen=str)),
        )

        # Metadata group
        write_meta_group(hf, df)

        # Root attributes
        hf.attrs["created"]            = datetime.datetime.utcnow().isoformat() + "Z"
        hf.attrs["downsample_factor"]  = factor
        hf.attrs["original_cadence_s"] = 25
        hf.attrs["cadence_s"]          = 25 * factor
        hf.attrs["n_lightcurves"]      = written
        hf.attrs["n_timesteps"]        = n_time

    size_mb = out_path.stat().st_size / 1024**2
    print(f"\n✓ Wrote {written} light curves  ({skipped} skipped) → {out_path}")
    print(f"  Shape : ({n_lc}, {n_time})   cadence : {25 * factor} s")
    print(f"  Size  : {size_mb:.1f} MB")

    if skipped:
        print(f"  [WARN] {skipped} rows written with label=-1 (missing or short .npz)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _data_default = Path(__file__).resolve().parent.parent / "data"

    parser = argparse.ArgumentParser(
        description="Pack .npz light curves into a single HDF5 dataset."
    )
    parser.add_argument(
        "--downsample", type=int, required=True,
        help="Integer downsample factor (e.g. 12 → 5 min cadence, 72 → 30 min)",
    )
    parser.add_argument(
        "--lc_dir", default=str(_data_default / "lightcurves"),
        help="Directory containing .npz light curve files (default: project/data/lightcurves/)",
    )
    parser.add_argument(
        "--meta_csv", default=str(_data_default / "metadata.csv"),
        help="Path to metadata.csv (default: project/data/metadata.csv)",
    )
    parser.add_argument(
        "--out", default=str(_data_default / "dataset.h5"),
        help="Output HDF5 path (default: project/data/dataset.h5)",
    )
    run_build(parser.parse_args())
