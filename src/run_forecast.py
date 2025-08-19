# src/run_forecast.py
# Minimal SARIMAX forecast with optional fuel-price exogenous regressor.
import sys, warnings, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")

# --- CONFIG: edit if your column names differ ---
DATA_FILE = Path("data/price_dataset.xlsx")    # time series data
FUEL_FILE = Path("data/fuel_price.xlsx")       # optional exogenous regressor
DATE_COL  = None   # e.g. "date"; if None, use first column
VALUE_COL = None   # e.g. "price"; if None, use second column
HORIZON   = 12     # months to forecast
# ------------------------------------------------

OUTDIR = Path("outputs"); OUTDIR.mkdir(exist_ok=True)
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

def load_series():
    df = pd.read_excel(DATA_FILE)
    if DATE_COL is None or VALUE_COL is None:
        date_col, val_col = df.columns[:2]
    else:
        date_col, val_col = DATE_COL, VALUE_COL
    s = (df[[date_col, val_col]]
         .rename(columns={date_col:"ds", val_col:"y"})
         .dropna())
    s["ds"] = pd.to_datetime(s["ds"])
    s = s.set_index("ds").asfreq("MS")  # monthly start
    s["y"] = s["y"].interpolate()
    return s

def load_exog(index):
    if not FUEL_FILE.exists(): return None
    ex = pd.read_excel(FUEL_FILE)
    # try to detect columns
    date_col = ex.columns[0]; val_col = ex.columns[1]
    ex = ex[[date_col, val_col]].rename(columns={date_col:"ds", val_col:"fuel"})
    ex["ds"] = pd.to_datetime(ex["ds"])
    ex = ex.set_index("ds").asfreq("MS").reindex(index).interpolate()
    return ex[["fuel"]]

def main():
    y = load_series()
    exog = load_exog(y.index)
    split = len(y) - HORIZON
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    X_tr = exog.iloc[:split] if exog is not None else None
    X_te = exog.iloc[split:] if exog is not None else None

    model = SARIMAX(y_tr["y"], exog=X_tr, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    fc_te = res.get_forecast(steps=len(y_te), exog=X_te).predicted_mean
    fc_fut = res.get_forecast(steps=HORIZON, exog=X_te if X_te is not None else None).predicted_mean

    # Save results
    out = pd.DataFrame({"actual": y["y"]})
    out["fitted"] = res.fittedvalues.reindex(out.index)
    out.loc[y_te.index, "forecast_eval"] = fc_te
    fut_index = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS")
    fut_df = pd.DataFrame(index=fut_index, data={"future_forecast": fc_fut.values})
    out_path = OUTDIR / "forecast_results.csv"
    pd.concat([out, fut_df], axis=0).to_csv(out_path)
    print(f"Saved: {out_path}")

    # Quick plot
    plt.figure()
    y["y"].plot(label="actual")
    res.fittedvalues.plot(label="fitted")
    fc_te.plot(label="test-forecast")
    fc_fut.plot(label="future-forecast")
    plt.legend(); plt.title("Retail price forecast (SARIMAX)")
    fig_path = OUTDIR / "forecast_plot.png"
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"Saved: {fig_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"\nError: {e}\nCheck column names in CONFIG at top of file.\n")
        sys.exit(1)
