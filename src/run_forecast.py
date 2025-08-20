# src/run_forecast.py
# Robust monthly SARIMAX with auto column detection + duplicate-month handling.
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ---- Config (you can leave these as None to auto-detect) ----
DATA_FILE = Path("data/price_dataset.xlsx")   # main time series (Excel)
FUEL_FILE = Path("data/fuel_price.xlsx")      # optional exogenous (Excel)
SHEET_NAME = 0                                # change if needed
DATE_COL = None                               # e.g., "Month" or "Date"
VALUE_COL = None                              # e.g., "Price" or "Rate"
HORIZON = 12                                  # forecast months
# --------------------------------------------------------------

OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)


def _norm(s) -> str:
    return " ".join(str(s).strip().lower().replace("_", " ").split())


def _pick_date_value(df: pd.DataFrame):
    """Pick a date column and a numeric value column automatically."""
    cols = list(df.columns)
    nmap = {_norm(c): c for c in cols}

    # If explicitly provided, try them first (case/space-insensitive)
    dcol = nmap.get(_norm(DATE_COL), None) if DATE_COL else None
    vcol = nmap.get(_norm(VALUE_COL), None) if VALUE_COL else None

    # Try common date-like names
    if dcol is None:
        for key, orig in nmap.items():
            if key in {"date", "month", "time", "period", "ds"}:
                dcol = orig
                break

    # Pick a numeric column for value
    if vcol is None:
        numeric = [
            c for c in cols if pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.8
        ]
        if dcol in numeric:
            numeric.remove(dcol)
        vcol = numeric[0] if numeric else next((c for c in cols if c != dcol), None)

    if dcol is None or vcol is None:
        raise ValueError(f"Could not detect date/value columns. Available: {cols}")

    return dcol, vcol


def _to_month_start(ts: pd.Series) -> pd.Series:
    """Collapse to month-start and average duplicates."""
    ts = pd.to_datetime(ts, errors="coerce")
    return ts.dt.to_period("M").dt.to_timestamp()  # month start


def load_series() -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    df = df.loc[:, [c for c in df.columns if df[c].notna().any()]]

    dcol, vcol = _pick_date_value(df)
    print(f"[forecast] Using columns -> date: '{dcol}', value: '{vcol}'")

    s = df[[dcol, vcol]].rename(columns={dcol: "ds", vcol: "y"}).copy()
    s["ds"] = _to_month_start(s["ds"])
    s["y"] = pd.to_numeric(s["y"], errors="coerce")
    s = s.dropna(subset=["ds"])

    # Collapse duplicates by month (mean)
    s = s.groupby("ds", as_index=False)["y"].mean()

    # Regular monthly index from first to last month
    s = s.set_index("ds").sort_index()
    s = s.asfreq("MS")
    s["y"] = s["y"].interpolate()

    if s["y"].notna().sum() < 6:
        raise ValueError("Not enough data points after cleaning to fit a model.")

    return s


def load_exog(index: pd.DatetimeIndex) -> pd.DataFrame | None:
    if not FUEL_FILE.exists():
        return None
    ex = pd.read_excel(FUEL_FILE)
    ex = ex.loc[:, [c for c in ex.columns if ex[c].notna().any()]]
    if len(ex.columns) < 2:
        return None
    dcol, vcol = ex.columns[:2]
    ex = ex[[dcol, vcol]].rename(columns={dcol: "ds", vcol: "fuel"})
    ex["ds"] = _to_month_start(ex["ds"])
    ex["fuel"] = pd.to_numeric(ex["fuel"], errors="coerce")
    ex = ex.dropna(subset=["ds"]).groupby("ds", as_index=False)["fuel"].mean()
    ex = ex.set_index("ds").sort_index().asfreq("MS")
    ex = ex.reindex(index).interpolate()
    return ex[["fuel"]]


def main():
    y = load_series()
    exog = load_exog(y.index)

    # Train/test split
    split = max(len(y) - HORIZON, 1)
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    X_tr = exog.iloc[:split] if exog is not None else None
    X_te = exog.iloc[split:] if exog is not None else None

    # Fit SARIMAX
    model = SARIMAX(
        y_tr["y"],
        exog=X_tr,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    # Forecasts
    if len(y_te):
        fc_te = res.get_forecast(steps=len(y_te), exog=X_te).predicted_mean
    else:
        fc_te = pd.Series([], dtype=float)

    fc_fut = res.get_forecast(
        steps=HORIZON,
        exog=X_te if X_te is not None else None,
    ).predicted_mean

    # Save results
    out = pd.DataFrame({"actual": y["y"]})
    out["fitted"] = res.fittedvalues.reindex(out.index)
    if len(y_te):
        out.loc[y_te.index, "forecast_eval"] = fc_te

    fut_index = pd.date_range(
        y.index[-1] + pd.offsets.MonthBegin(1),
        periods=HORIZON,
        freq="MS",
    )
    fut_df = pd.DataFrame(index=fut_index, data={"future_forecast": fc_fut.values})
    pd.concat([out, fut_df], axis=0).to_csv(OUTDIR / "forecast_results.csv")

    # Plot
    plt.figure()
    y["y"].plot(label="actual")
    res.fittedvalues.plot(label="fitted")
    if len(y_te):
        fc_te.plot(label="test-forecast")
    fc_fut.plot(label="future-forecast")
    plt.legend()
    plt.title("Retail price forecast (SARIMAX)")
    plt.savefig(OUTDIR / "forecast_plot.png", bbox_inches="tight")
    print("Saved: outputs/forecast_results.csv and outputs/forecast_plot.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"\nError: {e}\n")
        sys.exit(1)
