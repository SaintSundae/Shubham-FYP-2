# Shubham FYP 2

This project downloads and prepares financial time-series data for a conditional diffusion model. The workflow creates 30-trading-day windows from 10 market features and labels each window with a low-, medium-, or high-volatility regime.

## Project Layout

- `code/data_prep`: downloads missing Yahoo Finance series and Ken French factor data into `results/index_data/`. The S&P 500 and Nikkei 225 CSVs are existing inputs and are not downloaded by this script.
- `code/prep_data.py`: aligns raw CSVs, computes returns, assigns HMM regimes, normalizes features, creates chronological splits and windows, augments training windows, and writes processed arrays.
- `code/verify_data.py`: checks the saved arrays for expected shapes, valid numeric values, regime labels, and approximate normalization.
- `results/index_data/`: raw CSV inputs. These are generated or supplied locally and are not required to be committed.
- `results/processed/`: processed NumPy arrays and the feature scaler parameters.
- `configs/`, `logs/`, and `thesis_tables/`: reserved project folders; currently empty.

## Requirements

- Python 3.13 (the current environment was tested with Python 3.13.1)
- Python packages listed in `requirements.txt`
- VS Code extensions: **Python** (`ms-python.python`) and **Pylance** (`ms-python.vscode-pylance`), if using VS Code
- Internet access when downloading missing raw market or factor data

## Setup on Windows

From the project root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select `.venv\Scripts\python.exe` with **Python: Select Interpreter** so the editor and terminal use the same packages.

## Run the Pipeline

Run these commands from the project root after setup:

```powershell
python code/data_prep
python code/prep_data.py
python code/verify_data.py
```

`data_prep` skips raw CSV files that already exist, so running it again does not replace existing downloads. It downloads the technology, financials, and energy sector series, USD/JPY, VIX, and the three Ken French factors. The canonical S&P 500 and Nikkei 225 files must already be present in `results/index_data/`.

`prep_data.py` expects these 10 CSV files in `results/index_data/`:

| File | Feature |
| --- | --- |
| `sp500.csv` | S&P 500 |
| `nikkei225.csv` | Nikkei 225 |
| `technology_sector.csv` | Technology sector |
| `financials_sector.csv` | Financials sector |
| `energy_sector.csv` | Energy sector |
| `usd_jpy_exchange_rate.csv` | USD/JPY exchange rate |
| `cboe_volatility_index.csv` | CBOE Volatility Index (VIX) |
| `market_excess_return.csv` | Market excess return (Mkt-RF) |
| `size_factor.csv` | Size factor (SMB) |
| `value_factor.csv` | Value factor (HML) |

Each file needs a date column and its matching value column. Yahoo Finance inputs use `close` (or `adj_close`); factor files use the corresponding factor column.

## Processing Steps

1. Load each raw series, normalize its date index, and align the series on a shared calendar. Missing observations after a series begins are forward-filled.
2. Convert the seven price series to daily log returns. Divide the three Ken French percentage factors by 100.
3. Fit a three-state Gaussian HMM to the full available VIX series before splitting. States are ordered by their fitted mean VIX level and relabeled 0 (Low), 1 (Medium), and 2 (High). Because the HMM sees the full date range, regime labels are not strictly out-of-sample.
4. Split the observations chronologically into 70% training, 15% validation, and 15% test partitions. Fit feature means and standard deviations on training data only, then apply those values to all partitions to avoid leakage.
5. Create length-30 rolling windows. Each input has shape `(10, 30)` (features, time steps), and its condition is the regime on the final day of that window.
6. Create 2,000 additional regime-conditioned training windows by resampling five-day blocks from training windows. Validation and test data are not augmented.
7. Save all arrays and scaler parameters under `results/processed/`, then run the verifier.

## Generated Files

- `X_train.npy`, `X_val.npy`, `X_test.npy`: feature windows with shape `(samples, 10, 30)`.
- `C_train.npy`, `C_val.npy`, `C_test.npy`: integer regime labels, one per corresponding window.
- `scaler_params.json`: training-set feature means and standard deviations, in the feature order used by the preprocessing pipeline.

The exact number of windows depends on the available source dates. The training set additionally contains the synthetic windows described above.

## Reproducibility and Data Notes

The HMM uses a fixed random state. Training-window augmentation currently uses NumPy's random generator without a fixed seed, so repeated preprocessing runs can produce different augmented training arrays. The raw data sources may also update over time. Keep a copy of source CSVs when exact reproduction is important.

The raw CSVs are ignored by Git and are not committed. Run the downloader for the supported sources and place the canonical S&P 500 and Nikkei 225 CSVs in `results/index_data/` before preprocessing. Generated outputs can be recreated by running the pipeline in order.
