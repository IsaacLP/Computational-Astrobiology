"""
read_dataset.py — Utilities for loading the packed HDF5 dataset
"""
import h5py
import numpy as np
import pandas as pd


def load_dataset(h5_path):
    """
    Load time, flux, labels, and file_ids from the HDF5 dataset.

    Returns
    -------
    time : ndarray (n_time,) — days from start
    flux : ndarray (n_lc, n_time) float32 — flux in ppm
    labels : ndarray (n_lc,) int8 — class 0-3 (-1 for skipped rows)
    file_ids : list of str
    """
    with h5py.File(h5_path, "r") as f:
        time = f["time"][:].astype(np.float64) / 86400.0   # seconds → days
        flux = f["flux"][:]
        labels = f["label"][:]
        raw_ids = f["file_id"][:]
        file_ids = [
            fid.decode() if isinstance(fid, bytes) else str(fid)
            for fid in raw_ids
        ]
    return time, flux, labels, file_ids


def load_metadata(csv_path):
    """Load metadata CSV; returns DataFrame indexed by file_id (string)."""
    df = pd.read_csv(csv_path, dtype={"file_id": str})
    return df.set_index("file_id")
