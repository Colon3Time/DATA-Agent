import argparse
import contextlib
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_prophet(temp_dir):
    try:
        from prophet import Prophet
        import cmdstanpy.stanfit.runset as runset
        import cmdstanpy.utils.filesystem as filesystem

        filesystem._TMPDIR = str(temp_dir)
        runset._TMPDIR = str(temp_dir)

        return Prophet
    except ImportError as exc:
        raise SystemExit(
            "Prophet is required for Max inventory forecasting. "
            "Install it with: python -m pip install prophet"
        ) from exc


def first_existing_column(df, candidates, required_name):
    lookup = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise KeyError(
        f"Missing required {required_name} column. "
        f"Tried {candidates}; available columns are {list(df.columns)}"
    )


def numeric_column(df, candidates, required_name):
    col = first_existing_column(df, candidates, required_name)
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return col


def project_root_from_script():
    return Path(__file__).resolve().parents[2]


def default_paths(args):
    project_root = Path(args.project_root).resolve() if args.project_root else project_root_from_script()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else project_root / "output" / "max"
    eddie_dir = project_root / "output" / "eddie"
    return {
        "project_root": project_root,
        "output_dir": output_dir,
        "top_products": Path(args.top_products).resolve() if args.top_products else eddie_dir / "top_products_summary.csv",
        "monthly_sales": Path(args.monthly_sales).resolve() if args.monthly_sales else eddie_dir / "monthly_sales_summary.csv",
        "eddie_output": Path(args.input).resolve() if args.input else eddie_dir / "eddie_output.csv",
    }


def load_top_products(path):
    top_df = pd.read_csv(path)
    description_col = first_existing_column(top_df, ["Description", "product", "Product", "item"], "product description")
    stock_col = first_existing_column(top_df, ["StockCode", "stock_code", "stockcode"], "stock code")

    metric_col = None
    for candidates in (
        ["Quantity", "quantity", "TotalQuantity", "total_quantity"],
        ["TotalRevenue", "revenue", "Revenue"],
        ["InvoiceCount", "transactions", "TransactionCount"],
    ):
        try:
            metric_col = numeric_column(top_df, candidates, "ranking metric")
            break
        except KeyError:
            continue

    if metric_col:
        top_df = top_df.sort_values(metric_col, ascending=False)

    top_df = top_df.dropna(subset=[description_col]).copy()
    top_df[description_col] = top_df[description_col].astype(str).str.strip()
    top_df = top_df[top_df[description_col] != ""]
    top_df = top_df.drop_duplicates(subset=[description_col], keep="first").head(10)

    return top_df[[description_col, stock_col]].rename(
        columns={description_col: "Description", stock_col: "StockCode"}
    )


def build_monthly_quantity(eddie_output_path, top_descriptions):
    df = pd.read_csv(eddie_output_path)

    desc_col = first_existing_column(df, ["Description", "product", "Product", "item"], "product description")
    qty_col = numeric_column(df, ["Quantity", "quantity", "qty"], "quantity")

    date_col = None
    for candidates in (["InvoiceDate", "invoice_date", "date"], ["invoice_month", "month"]):
        try:
            date_col = first_existing_column(df, candidates, "date/month")
            break
        except KeyError:
            continue
    if date_col is None:
        raise KeyError(f"No date/month column found in {eddie_output_path}")

    work = df[[desc_col, qty_col, date_col] + [c for c in ["is_valid_sale", "is_return"] if c in df.columns]].copy()
    work[desc_col] = work[desc_col].astype(str).str.strip()
    work = work[work[desc_col].isin(top_descriptions)]
    work = work.dropna(subset=[desc_col, qty_col, date_col])

    if "is_valid_sale" in work.columns:
        work = work[pd.to_numeric(work["is_valid_sale"], errors="coerce").fillna(0).astype(int) == 1]
    if "is_return" in work.columns:
        work = work[pd.to_numeric(work["is_return"], errors="coerce").fillna(0).astype(int) == 0]

    work = work[work[qty_col] > 0]
    work["ds"] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=["ds"])
    work["month"] = work["ds"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        work.groupby([desc_col, "month"], as_index=False)[qty_col]
        .sum()
        .rename(columns={desc_col: "Description", "month": "ds", qty_col: "quantity"})
    )
    return monthly


def complete_monthly_series(monthly, product, global_start, global_end):
    product_monthly = monthly[monthly["Description"] == product][["ds", "quantity"]].copy()
    product_monthly = product_monthly.groupby("ds", as_index=False)["quantity"].sum().sort_values("ds")
    month_index = pd.date_range(global_start, global_end, freq="MS")
    product_monthly = (
        product_monthly.set_index("ds")
        .reindex(month_index, fill_value=0)
        .rename_axis("ds")
        .reset_index()
        .rename(columns={"quantity": "y"})
    )
    product_monthly["y"] = pd.to_numeric(product_monthly["y"], errors="coerce").fillna(0)
    return product_monthly


def prophet_forecast(Prophet, series, output_dir, periods=6):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.8,
    )
    model.fit(series, output_dir=str(output_dir))
    future = model.make_future_dataframe(periods=periods, freq="MS", include_history=False)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def fallback_forecast(series, periods=6):
    last_date = series["ds"].max()
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=periods, freq="MS")
    yhat = float(series["y"].tail(min(6, len(series))).mean()) if len(series) else 0.0
    return pd.DataFrame(
        {
            "ds": future_dates,
            "yhat": yhat,
            "yhat_lower": max(0.0, yhat * 0.8),
            "yhat_upper": yhat * 1.2,
        }
    )


def mape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def backtest_product(Prophet, product, stock_code, series, prophet_run_dir):
    if len(series) < 6:
        return {
            "Description": product,
            "StockCode": stock_code,
            "prophet_mape": np.nan,
            "naive_mape": np.nan,
            "beats_naive": False,
            "backtest_months": 0,
            "model": "insufficient_history",
        }
    train = series.iloc[:-3].copy()
    test = series.iloc[-3:].copy()
    model_name = "prophet"
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            forecast = prophet_forecast(Prophet, train, prophet_run_dir, periods=3)
        pred = forecast["yhat"].clip(lower=0).values
    except Exception:
        model_name = "moving_average_fallback"
        pred = fallback_forecast(train, periods=3)["yhat"].values
    naive = []
    for ds in test["ds"]:
        last_year = ds - pd.DateOffset(years=1)
        matched = series.loc[series["ds"].eq(last_year), "y"]
        naive.append(float(matched.iloc[0]) if not matched.empty else float(train["y"].tail(min(3, len(train))).mean()))
    prophet_mape = mape(test["y"].values, pred)
    naive_mape = mape(test["y"].values, naive)
    return {
        "Description": product,
        "StockCode": stock_code,
        "prophet_mape": round(prophet_mape, 2) if pd.notna(prophet_mape) else np.nan,
        "naive_mape": round(naive_mape, 2) if pd.notna(naive_mape) else np.nan,
        "beats_naive": bool(pd.notna(prophet_mape) and pd.notna(naive_mape) and prophet_mape < naive_mape),
        "backtest_months": int(len(test)),
        "model": model_name,
    }


def generate_report(report_path, forecast_df, backtest_df, top_products, monthly, paths, generated_at):
    first_month = forecast_df[forecast_df["forecast_month_offset"] == 1].copy()
    total_next_demand = first_month["forecast_quantity"].sum()
    total_recommended = first_month["recommended_stock"].sum()

    lines = [
        "# Max Inventory Optimization Report",
        "",
        f"Generated: {generated_at}",
        "",
        "## Inputs",
        f"- Top products: `{paths['top_products']}`",
        f"- Monthly sales summary checked: `{paths['monthly_sales']}`",
        f"- Product-level monthly quantity source: `{paths['eddie_output']}`",
        "",
        "## Method",
        "- Top 10 products are keyed by `Description` from `top_products_summary.csv`.",
        "- Product-month demand is aggregated from `eddie_output.csv` using `Description`, `InvoiceDate` or `invoice_month`, and `Quantity`.",
        "- Forecast horizon is 6 monthly periods per product.",
        "- Forecast model is Prophet when it fits successfully; otherwise the script records a moving-average fallback for that product.",
        "- Safety stock formula: `historical monthly mean + 1.5 * historical monthly std`.",
        "",
        "## Forecast Accuracy",
        "| Description | Prophet MAPE | Naive MAPE | Beats Naive |",
        "|---|---:|---:|---|",
    ]
    for row in backtest_df.head(10).itertuples(index=False):
        lines.append(
            f"| {str(row.Description).replace('|', '\\|')} | "
            f"{row.prophet_mape if pd.notna(row.prophet_mape) else 'N/A'} | "
            f"{row.naive_mape if pd.notna(row.naive_mape) else 'N/A'} | {row.beats_naive} |"
        )
    lines.extend(
        [
            f"- Beats naive count: {int(backtest_df['beats_naive'].sum())} of {len(backtest_df)}",
            "",
        "## Output Summary",
        f"- Products forecasted: {forecast_df['Description'].nunique()}",
        f"- Forecast rows: {len(forecast_df)}",
        f"- Next-month forecast quantity: {total_next_demand:,.0f}",
        f"- Next-month recommended stock: {total_recommended:,.0f}",
        "",
        "## Top Products",
        "| Rank | Description | StockCode | Historical Mean | Historical Std | Safety Stock | Model |",
        "|---:|---|---|---:|---:|---:|---|",
        ]
    )

    product_summary = (
        forecast_df[forecast_df["forecast_month_offset"] == 1]
        .set_index("Description")
        .to_dict(orient="index")
    )
    for rank, row in enumerate(top_products.itertuples(index=False), start=1):
        summary = product_summary.get(row.Description, {})
        lines.append(
            "| {rank} | {desc} | {stock} | {mean:,.2f} | {std:,.2f} | {ss:,.2f} | {model} |".format(
                rank=rank,
                desc=str(row.Description).replace("|", "\\|"),
                stock=row.StockCode,
                mean=summary.get("historical_monthly_mean", 0),
                std=summary.get("historical_monthly_std", 0),
                ss=summary.get("safety_stock", 0),
                model=summary.get("model", "not_forecast"),
            )
        )

    lines.extend(
        [
            "",
            "## Next-Month Inventory Recommendation",
            "| Description | Forecast Quantity | Safety Stock | Recommended Stock |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in first_month.sort_values("recommended_stock", ascending=False).itertuples(index=False):
        lines.append(
            f"| {str(row.Description).replace('|', '\\|')} | "
            f"{row.forecast_quantity:,.0f} | {row.safety_stock:,.0f} | {row.recommended_stock:,.0f} |"
        )

    lines.extend(
        [
            "",
            "## Data Notes",
            f"- Aggregated product-month rows: {len(monthly)}",
            f"- Historical range: {monthly['ds'].min().date()} to {monthly['ds'].max().date()}",
            "- Negative quantities and rows marked as returns are excluded for inventory demand forecasting.",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Max inventory optimization forecast")
    parser.add_argument("--project-root", default="")
    parser.add_argument("--input", default="", help="Path to output/eddie/eddie_output.csv")
    parser.add_argument("--top-products", default="")
    parser.add_argument("--monthly-sales", default="")
    parser.add_argument("--output-dir", default="")
    args, _ = parser.parse_known_args()

    paths = default_paths(args)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    temp_dir = paths["output_dir"] / "_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMPDIR"] = str(temp_dir)

    print(f"[STATUS] Project root: {paths['project_root']}")
    print(f"[STATUS] Output dir: {paths['output_dir']}")
    print(f"[STATUS] Top products: {paths['top_products']}")
    print(f"[STATUS] Eddie output: {paths['eddie_output']}")

    for key in ("top_products", "eddie_output"):
        if not paths[key].exists():
            raise FileNotFoundError(paths[key])

    if paths["monthly_sales"].exists():
        monthly_sales_cols = list(pd.read_csv(paths["monthly_sales"], nrows=0).columns)
        print(f"[STATUS] monthly_sales_summary columns: {monthly_sales_cols}")
        print("[STATUS] monthly_sales_summary is overall monthly data; using eddie_output for product-level aggregation")

    prophet_run_dir = paths["output_dir"] / "prophet_runs"
    prophet_run_dir.mkdir(parents=True, exist_ok=True)
    Prophet = load_prophet(temp_dir)
    top_products = load_top_products(paths["top_products"])
    top_descriptions = top_products["Description"].tolist()
    monthly = build_monthly_quantity(paths["eddie_output"], top_descriptions)

    if monthly.empty:
        raise ValueError("No product-month demand rows were created for the selected top products")

    print(f"[STATUS] Top 10 products: {len(top_products)}")
    print(f"[STATUS] Product-month rows: {len(monthly)}")
    print(f"[STATUS] Historical range: {monthly['ds'].min().date()} to {monthly['ds'].max().date()}")
    global_start = monthly["ds"].min()
    global_end = monthly["ds"].max()

    forecast_rows = []
    backtest_rows = []
    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for rank, product_row in enumerate(top_products.itertuples(index=False), start=1):
        product = product_row.Description
        stock_code = product_row.StockCode
        series = complete_monthly_series(monthly, product, global_start, global_end)
        if series.empty:
            print(f"[WARN] No monthly series for {product}; skipping")
            continue

        historical_mean = float(series["y"].mean())
        historical_std = float(series["y"].std(ddof=0))
        safety_stock = max(0.0, historical_mean + 1.5 * historical_std)
        backtest_rows.append(backtest_product(Prophet, product, stock_code, series, prophet_run_dir))

        model_name = "prophet"
        try:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                future = prophet_forecast(Prophet, series, prophet_run_dir, periods=6)
        except Exception as exc:
            print(f"[WARN] Prophet failed for {product}: {exc}. Using moving_average fallback.")
            model_name = "moving_average_fallback"
            future = fallback_forecast(series, periods=6)

        for offset, forecast in enumerate(future.itertuples(index=False), start=1):
            forecast_qty = max(0.0, float(forecast.yhat))
            yhat_lower = max(0.0, float(forecast.yhat_lower))
            yhat_upper = max(0.0, float(forecast.yhat_upper))
            recommended_stock = forecast_qty + safety_stock
            forecast_rows.append(
                {
                    "rank": rank,
                    "Description": product,
                    "StockCode": stock_code,
                    "forecast_month": pd.Timestamp(forecast.ds).strftime("%Y-%m"),
                    "forecast_month_offset": offset,
                    "forecast_quantity": round(forecast_qty, 2),
                    "forecast_lower": round(yhat_lower, 2),
                    "forecast_upper": round(yhat_upper, 2),
                    "historical_months": int(len(series)),
                    "historical_monthly_mean": round(historical_mean, 2),
                    "historical_monthly_std": round(historical_std, 2),
                    "safety_stock": round(safety_stock, 2),
                    "recommended_stock": round(recommended_stock, 2),
                    "model": model_name,
                    "generated_at": generated_at,
                }
            )
        print(f"[STATUS] Forecasted {rank}/10: {product} ({model_name})")

    forecast_df = pd.DataFrame(forecast_rows)
    if forecast_df.empty:
        raise ValueError("No forecasts were generated")

    forecast_path = paths["output_dir"] / "inventory_forecast.csv"
    backtest_path = paths["output_dir"] / "forecast_backtest.csv"
    report_path = paths["output_dir"] / "max_report.md"
    backtest_df = pd.DataFrame(backtest_rows)
    forecast_df.to_csv(forecast_path, index=False)
    backtest_df.to_csv(backtest_path, index=False)
    generate_report(report_path, forecast_df, backtest_df, top_products, monthly, paths, generated_at)

    print(f"[STATUS] Saved: {forecast_path}")
    print(f"[STATUS] Saved: {backtest_path}")
    print(f"[STATUS] Saved: {report_path}")
    print("[STATUS] === MAX INVENTORY OPTIMIZATION COMPLETE ===")


if __name__ == "__main__":
    main()
