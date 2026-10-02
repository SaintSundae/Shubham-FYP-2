#!/usr/bin/env python3
"""Verify the processed financial tensors produced by prep_data.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "results" / "processed"


def load_arrays() -> dict[str, np.ndarray | dict]:
    arrays = {}
    names = [
        "X_train",
        "X_val",
        "X_test",
        "C_train",
        "C_val",
        "C_test",
    ]
    for name in names:
        arrays[name] = np.load(PROCESSED_DIR / f"{name}.npy")
    with (PROCESSED_DIR / "scaler_params.json").open("r", encoding="utf-8") as f:
        arrays["scaler_params"] = json.load(f)
    return arrays


def assert_no_invalid(arrays: list[np.ndarray]) -> None:
    for arr in arrays:
        if not np.isfinite(arr).all():
            raise ValueError(f"Array contains NaN or Inf values: shape={arr.shape}, dtype={arr.dtype}")


def verify_shapes(arrays: dict[str, np.ndarray | dict]) -> None:
    x_train = arrays["X_train"]
    x_val = arrays["X_val"]
    x_test = arrays["X_test"]

    for name, arr in {"X_train": x_train, "X_val": x_val, "X_test": x_test}.items():
        assert arr.ndim == 3, f"{name} must be 3D, got shape {arr.shape}"
        assert arr.shape[1:] == (10, 30), f"{name} must have shape (N, 10, 30), got {arr.shape}"

    c_train = arrays["C_train"]
    c_val = arrays["C_val"]
    c_test = arrays["C_test"]

    for name, arr in {"C_train": c_train, "C_val": c_val, "C_test": c_test}.items():
        assert arr.ndim == 1, f"{name} must be 1D, got shape {arr.shape}"

    assert x_train.shape[0] == c_train.shape[0], (
        f"X_train and C_train sample counts mismatch: {x_train.shape[0]} vs {c_train.shape[0]}"
    )
    assert x_val.shape[0] == c_val.shape[0], (
        f"X_val and C_val sample counts mismatch: {x_val.shape[0]} vs {c_val.shape[0]}"
    )
    assert x_test.shape[0] == c_test.shape[0], (
        f"X_test and C_test sample counts mismatch: {x_test.shape[0]} vs {c_test.shape[0]}"
    )

    scaler = arrays["scaler_params"]
    assert "mean" in scaler and "std" in scaler, "scaler_params.json is missing mean/std keys"
    assert len(scaler["mean"]) == 10, f"Expected 10 scaling means, got {len(scaler['mean'])}"
    assert len(scaler["std"]) == 10, f"Expected 10 scaling stds, got {len(scaler['std'])}"


def verify_hmm_regimes(arrays: dict[str, np.ndarray | dict]) -> None:
    c_train = arrays["C_train"]
    unique = np.unique(c_train)
    expected = np.array([0, 1, 2], dtype=int)

    if not np.array_equal(np.sort(unique), np.sort(expected)):
        missing = [int(v) for v in expected if v not in unique]
        unexpected = [int(v) for v in unique if v not in expected]
        raise AssertionError(
            f"C_train does not contain the expected regime labels 0, 1, 2. "
            f"Missing={missing}, Unexpected={unexpected}, Unique={unique.tolist()}"
        )

    counts = np.bincount(c_train.astype(int), minlength=3)
    regime_labels = {0: "Low", 1: "Medium", 2: "High"}
    print("HMM regime value counts:")
    for regime_id, count in enumerate(counts):
        print(f"  {regime_labels[regime_id]} ({regime_id}): {int(count)}")


def verify_normalization(arrays: dict[str, np.ndarray | dict]) -> None:
    x_train = arrays["X_train"]
    mean = float(x_train.mean())
    std = float(x_train.std())

    print(f"Overall X_train mean: {mean:.6f}")
    print(f"Overall X_train std:  {std:.6f}")

    assert abs(mean) < 0.2, f"X_train mean is too far from 0.0: {mean}"
    assert abs(std - 1.0) < 0.25, f"X_train std is too far from 1.0: {std}"


def main() -> None:
    print(f"Loading processed arrays from: {PROCESSED_DIR}")
    arrays = load_arrays()

    verify_shapes(arrays)
    assert_no_invalid([
        arrays["X_train"],
        arrays["X_val"],
        arrays["X_test"],
        arrays["C_train"],
        arrays["C_val"],
        arrays["C_test"],
    ])
    verify_hmm_regimes(arrays)
    verify_normalization(arrays)

    print("\nAll verification checks passed successfully.")


if __name__ == "__main__":
    main()
