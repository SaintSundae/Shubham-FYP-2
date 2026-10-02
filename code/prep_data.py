#!/usr/bin/env python3
"""Transform aligned market CSVs into train, validation, and test model tensors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "index_data"
OUT_DIR = ROOT / "results" / "processed"
WINDOW_LENGTH = 30
BOOTSTRAP_TOTAL = 2000

FILE_MAP = {
    "sp500.csv": ("adj_close", "sp500"),
    "nikkei225.csv": ("adj_close", "nikkei225"),
    "technology_sector.csv": ("adj_close", "tech"),
    "financials_sector.csv": ("adj_close", "financials"),
    "energy_sector.csv": ("adj_close", "energy"),
    "usd_jpy_exchange_rate.csv": ("adj_close", "usd_jpy"),
    "cboe_volatility_index.csv": ("adj_close", "vix"),
    "market_excess_return.csv": ("Mkt-RF", "mkt_rf"),
    "size_factor.csv": ("SMB", "smb"),
    "value_factor.csv": ("HML", "hml"),
}

PRICE_COLUMNS = ["sp500", "nikkei225", "tech", "financials", "energy", "usd_jpy", "vix"]
FACTOR_COLUMNS = ["mkt_rf", "smb", "hml"]
ALL_COLUMNS = PRICE_COLUMNS + FACTOR_COLUMNS


def normalize_name(value: str) -> str:
    """Normalize a CSV header for matching despite punctuation or spacing differences."""
    return "".join(ch.lower() for ch in str(value).strip() if ch.isalnum())


def coerce_date(value) -> pd.Timestamp:
    """Parse supported date values and return NaT when a value is not a date."""
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        s = str(int(value))
        if len(s) == 8 and s.isdigit():
            try:
                return pd.to_datetime(s, format="%Y%m%d")
            except ValueError:
                pass

    s = str(value).strip()
    if not s:
        return pd.NaT

    try:
        return pd.to_datetime(s, errors="raise")
    except (TypeError, ValueError):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
            try:
                return pd.to_datetime(s, format=fmt)
            except ValueError:
                continue
    return pd.NaT


def find_matching_column(columns: Iterable[str], target_aliases: Iterable[str]) -> str:
    """Find a source column using normalized header names and known aliases."""
    aliases = {normalize_name(alias): alias for alias in target_aliases}
    normalized = {normalize_name(col): col for col in columns}

    for alias_key in aliases:
        if alias_key in normalized:
            return normalized[alias_key]

    for col in columns:
        normalized_col = normalize_name(col)
        for alias_key in aliases:
            if normalized_col == alias_key:
                return col
    raise KeyError(f"Could not find any expected column for aliases {list(target_aliases)} in {list(columns)}")


def load_single_series(path: Path, source_col: str, target_name: str) -> pd.DataFrame:
    """Load one CSV series, standardize its date index, and rename its value column."""
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")

    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]

    if "date" not in df.columns:
        date_candidates = [col for col in df.columns if normalize_name(col) in {"date", "datetime", "timestamp"}]
        if not date_candidates:
            raise ValueError(f"File {path.name} does not contain a recognizable date column.")
        date_col = date_candidates[0]
    else:
        date_col = "date"

    if normalize_name(source_col) in {"adjclose", "adjcloseprice"}:
        target_aliases = ["adj_close", "Adj Close", "adj close", "adj-close", "close"]
    elif normalize_name(source_col) in {"mktrf", "mktrf"}:
        target_aliases = [source_col, source_col.replace("_", " "), source_col.replace("_", "-"), "Mkt-RF", "Mkt RF"]
    else:
        target_aliases = [source_col, source_col.replace("_", " "), source_col.replace("_", "-"), source_col.lower()]
    target_col = find_matching_column(df.columns, target_aliases)

    selected = df[[date_col, target_col]].copy()
    selected.rename(columns={date_col: "date", target_col: target_name}, inplace=True)
    selected["date"] = selected["date"].map(coerce_date)
    selected = selected.dropna(subset=["date"]).set_index("date").sort_index()
    return selected


def load_raw_series() -> Dict[str, pd.DataFrame]:
    """Load every available input described by FILE_MAP."""
    series: Dict[str, pd.DataFrame] = {}
    for filename, (source_col, target_name) in FILE_MAP.items():
        path = RAW_DIR / filename
        if not path.exists():
            print(f"Skipping missing file: {path.name}")
            continue
        series[target_name] = load_single_series(path, source_col, target_name)
    if not series:
        raise FileNotFoundError(f"No raw CSV data found in {RAW_DIR}")
    return series


def merge_and_filter(series_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align series by date, discard dates before the latest source start, and forward-fill."""
    merged = pd.concat(series_map.values(), axis=1, join="outer", sort=True).sort_index()
    start_dates = {name: df.index.min() for name, df in series_map.items() if len(df) > 0}
    if not start_dates:
        raise ValueError("No valid series found after loading raw CSV files.")
    common_start = max(start_dates.values())
    merged = merged.loc[merged.index >= common_start].copy()
    merged = merged.ffill()
    return merged


def compute_returns_and_factors(merged: pd.DataFrame) -> pd.DataFrame:
    """Convert prices to log returns and factor percentages to decimal values."""
    returns = merged.copy()

    for name in PRICE_COLUMNS:
        if name not in returns.columns:
            continue
        returns[name] = np.log(returns[name] / returns[name].shift(1))

    for name in FACTOR_COLUMNS:
        if name in returns.columns:
            returns[name] = returns[name] / 100.0

    returns = returns.dropna().copy()
    return returns


def fit_hmm_regimes(raw_vix: pd.Series) -> pd.Series:
    """Fit a three-state VIX HMM and order labels from lowest to highest mean VIX."""
    vix_values = raw_vix.dropna().to_numpy(dtype=float).reshape(-1, 1)
    if len(vix_values) < 3:
        raise ValueError("Not enough valid VIX observations to fit HMM.")

    model = GaussianHMM(n_components=3, covariance_type="diag", random_state=42)
    model.fit(vix_values)
    state_means = model.means_.ravel()
    ordered = np.argsort(state_means)
    remap = np.empty_like(ordered)
    remap[ordered] = np.arange(len(ordered))
    state_ids = model.predict(vix_values)
    remapped = remap[state_ids]

    vix_index = raw_vix.dropna().index
    return pd.Series(remapped, index=vix_index, name="regime")


def build_windows(X: np.ndarray, labels: np.ndarray, window_length: int = WINDOW_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    """Create feature-by-time windows and label each window from its final day."""
    if X.shape[0] < window_length:
        raise ValueError(f"Not enough observations to build windows of length {window_length}.")

    sequences, targets = [], []
    for i in range(0, X.shape[0] - window_length + 1):
        seq = X[i : i + window_length]
        sequences.append(seq.T)
        targets.append(int(labels[i + window_length - 1]))

    return np.stack(sequences), np.asarray(targets, dtype=int)


def augment_train_windows(X_train_windows: np.ndarray, C_train: np.ndarray, total_samples: int = BOOTSTRAP_TOTAL) -> Tuple[np.ndarray, np.ndarray]:
    """Add regime-matched training samples by resampling five-day blocks within windows."""
    if total_samples <= 0:
        return X_train_windows.copy(), C_train.copy()

    groups = [X_train_windows[C_train == regime] for regime in [0, 1, 2]]
    regime_sizes = np.array([g.shape[0] for g in groups], dtype=float)
    regime_sizes = regime_sizes / regime_sizes.sum() if regime_sizes.sum() > 0 else np.ones(3) / 3.0

    counts = np.floor(total_samples * regime_sizes).astype(int)
    counts[0] += total_samples - counts.sum()
    counts = np.clip(counts, 0, total_samples)

    synthetic_X, synthetic_y = [], []
    for regime, n_aug in enumerate(counts):
        regime_group = groups[regime]
        if regime_group.size == 0:
            continue
        for _ in range(n_aug):
            source_seq = regime_group[np.random.randint(0, regime_group.shape[0])]
            block_len = 5
            synthetic_seq = np.empty((source_seq.shape[0], WINDOW_LENGTH), dtype=np.float64)
            for block_idx in range(6):
                start = np.random.randint(0, max(1, source_seq.shape[1] - block_len + 1))
                synthetic_seq[:, block_idx * block_len : (block_idx + 1) * block_len] = source_seq[:, start : start + block_len]
            synthetic_X.append(synthetic_seq)
            synthetic_y.append(regime)

    if synthetic_X:
        X_aug = np.stack(synthetic_X)
        y_aug = np.asarray(synthetic_y, dtype=int)
        return np.concatenate([X_train_windows, X_aug], axis=0), np.concatenate([C_train, y_aug], axis=0)
    return X_train_windows.copy(), C_train.copy()


def save_json(path: Path, payload: dict) -> None:
    """Write JSON with stable indentation for readable scaler metadata."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def assert_no_invalid(arrays: List[np.ndarray]) -> None:
    """Stop preprocessing if any exported array contains NaN or infinite values."""
    for arr in arrays:
        if not np.isfinite(arr).all():
            raise ValueError(f"Array contains NaN/Inf values: shape={arr.shape}, dtype={arr.dtype}")


def print_split_summary(train_dates: pd.DatetimeIndex, val_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, C_train: np.ndarray, C_val: np.ndarray, C_test: np.ndarray) -> None:
    """Print the date coverage and sample shapes for each exported split."""
    print("Train date range:", train_dates.min().date(), "->", train_dates.max().date(), "samples=", X_train.shape[0])
    print("Validation date range:", val_dates.min().date(), "->", val_dates.max().date(), "samples=", X_val.shape[0])
    print("Test date range:", test_dates.min().date(), "->", test_dates.max().date(), "samples=", X_test.shape[0])
    print("Train tensor shape:", X_train.shape)
    print("Val tensor shape:", X_val.shape)
    print("Test tensor shape:", X_test.shape)
    print("Train labels shape:", C_train.shape)
    print("Val labels shape:", C_val.shape)
    print("Test labels shape:", C_test.shape)


def main() -> None:
    """Run the full raw-data-to-tensors preprocessing pipeline."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    series_map = load_raw_series()
    merged = merge_and_filter(series_map)

    raw_vix = merged["vix"].copy()
    returns = compute_returns_and_factors(merged)
    assert "vix" in returns.columns, "The VIX series must be present after merging."
    returns["regime"] = fit_hmm_regimes(raw_vix).reindex(returns.index).to_numpy()

    feature_cols = [col for col in ALL_COLUMNS if col in returns.columns]
    X_daily = returns[feature_cols].to_numpy(dtype=float)
    labels_daily = returns["regime"].to_numpy(dtype=int)

    # Split chronologically so validation and test dates follow training dates.
    n = len(returns)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train_daily = X_daily[:train_end]
    X_val_daily = X_daily[train_end:val_end]
    X_test_daily = X_daily[val_end:]

    y_train_daily = labels_daily[:train_end]
    y_val_daily = labels_daily[train_end:val_end]
    y_test_daily = labels_daily[val_end:]

    mu = X_train_daily.mean(axis=0)
    sigma = X_train_daily.std(axis=0)
    sigma = np.where(sigma == 0.0, 1.0, sigma)

    X_train_z = (X_train_daily - mu) / sigma
    X_val_z = (X_val_daily - mu) / sigma
    X_test_z = (X_test_daily - mu) / sigma

    X_train_w, C_train = build_windows(X_train_z, y_train_daily, WINDOW_LENGTH)
    X_val_w, C_val = build_windows(X_val_z, y_val_daily, WINDOW_LENGTH)
    X_test_w, C_test = build_windows(X_test_z, y_test_daily, WINDOW_LENGTH)

    X_train_aug, C_train_aug = augment_train_windows(X_train_w, C_train, total_samples=BOOTSTRAP_TOTAL)
    X_train = X_train_aug
    C_train = C_train_aug

    save_json(OUT_DIR / "scaler_params.json", {"mean": mu.tolist(), "std": sigma.tolist()})
    np.save(OUT_DIR / "X_train.npy", X_train)
    np.save(OUT_DIR / "X_val.npy", X_val_w)
    np.save(OUT_DIR / "X_test.npy", X_test_w)
    np.save(OUT_DIR / "C_train.npy", C_train)
    np.save(OUT_DIR / "C_val.npy", C_val)
    np.save(OUT_DIR / "C_test.npy", C_test)

    assert_no_invalid([X_train, X_val_w, X_test_w, C_train, C_val, C_test])

    train_dates = returns.index[:train_end]
    val_dates = returns.index[train_end:val_end]
    test_dates = returns.index[val_end:]
    print_split_summary(train_dates, val_dates, test_dates, X_train, X_val_w, X_test_w, C_train, C_val, C_test)
    print(f"Saved processed arrays to: {OUT_DIR}")


if __name__ == "__main__":
    main()
