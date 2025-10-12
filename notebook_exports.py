# your_notebook_exports.py
# Minimal, production-minded helpers for REDA — Forecasting

import warnings
warnings.filterwarnings("ignore",
    message="Data Validation extension is not supported and will be removed",
    category=UserWarning,
    module=r"openpyxl\.worksheet\._read_only"
)

from typing import Dict, List
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -----------------------------
# Loading & cleaning
# -----------------------------
def _tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df

def _detect_header_row(df: pd.DataFrame, max_check: int = 12, min_non_null: int = 2) -> int:
    best_i, best_score = 0, -1
    nrows = min(max_check, len(df))
    for i in range(nrows):
        row = df.iloc[i]
        non_null = int(row.notna().sum())
        if non_null < min_non_null:
            continue
        str_like = sum(isinstance(x, str) for x in row)
        as_str = row.dropna().astype(str)
        uniq_ratio = (as_str.nunique() / non_null) if non_null else 0.0
        score = (str_like / max(non_null, 1)) * 2.0 + uniq_ratio
        if score > best_score:
            best_i, best_score = i, score
    return best_i

def load_sheets(file_path) -> Dict[str, pd.DataFrame]:
    # Base (robust header detect)
    base_preview = pd.read_excel(file_path, sheet_name="Base", header=None, nrows=12)
    base_hdr = _detect_header_row(base_preview)
    base = pd.read_excel(file_path, sheet_name="Base", header=base_hdr)
    base = _tidy_columns(base)

    # Customers (promote first row to header; 3rd col becomes Share)
    customers = pd.read_excel(file_path, sheet_name="Customers", header=1, usecols=[0, 1, 2]).reset_index(drop=True)
    cust_cols = customers.iloc[0].tolist()
    if len(cust_cols) >= 3 and (pd.isna(cust_cols[2]) or str(cust_cols[2]).strip().lower() == "nan"):
        cust_cols[2] = "Share"
    customers = customers.iloc[1:].reset_index(drop=True)
    customers.columns = cust_cols
    customers = _tidy_columns(customers)

    # Items (promote first row to header)
    items = pd.read_excel(file_path, sheet_name="Items", header=1, usecols=[0, 1, 2]).reset_index(drop=True)
    item_cols = items.iloc[0].tolist()
    items = items.iloc[1:].reset_index(drop=True)
    items.columns = item_cols
    items = _tidy_columns(items)

    # Working Days
    working_days = pd.read_excel(file_path, sheet_name="Working Days", header=0)
    working_days = _tidy_columns(working_days)

    # Temperature (rename '1' -> 'TempFactor')
    temperature = pd.read_excel(file_path, sheet_name="Temperature", header=0)
    temperature = _tidy_columns(temperature)
    if list(temperature.columns) == [1] or list(temperature.columns) == ["1"]:
        temperature.columns = ["TempFactor"]
    if "TempFactor" in temperature.columns:
        temperature["TempFactor"] = pd.to_numeric(temperature["TempFactor"], errors="coerce")

    # Frecuency
    frecuency = pd.read_excel(file_path, sheet_name="Frecuency", header=0)
    frecuency = _tidy_columns(frecuency)

    return {
        "Base": base,
        "Customers": customers,
        "Items": items,
        "Working Days": working_days,
        "Temperature": temperature,
        "Frecuency": frecuency,
    }

# -----------------------------
# Normalization to FACT
# -----------------------------
def _parse_mixed_excel_date_strict(raw_series: pd.Series) -> pd.Series:
    s = raw_series.copy()
    as_num = pd.to_numeric(s, errors="coerce")
    is_num = as_num.notna()
    is_str = ~is_num
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out.loc[is_str] = pd.to_datetime(s.loc[is_str], errors="coerce")
    if is_num.any():
        out.loc[is_num] = pd.to_datetime(as_num.loc[is_num], unit="D", origin="1899-12-30")
    return out

def normalize_base_to_fact(base_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    b = base_df.copy()
    b["Delivery Date"] = _parse_mixed_excel_date_strict(b["Delivery Date"])
    b["Month_ym"] = b["Delivery Date"].dt.to_period("M").astype(str)

    month_str = b["Month"].astype(str).str.strip()
    is_valid_ym = month_str.str.match(r"^\d{4}-(?:0?[1-9]|1[0-2])$")
    b["is_sop"] = ~is_valid_ym

    b_actuals = b.loc[~b["is_sop"]].copy()
    b_sop = b.loc[b["is_sop"]].copy()

    # Fill missing Group with dominant per Itemcode
    dom_group = (
        b_actuals.dropna(subset=["Group"])
        .groupby("Itemcode")["Group"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA)
    )
    miss_mask = b_actuals["Group"].isna()
    b_actuals.loc[miss_mask, "Group"] = b_actuals.loc[miss_mask, "Itemcode"].map(dom_group)
    b_actuals["Group"] = b_actuals["Group"].fillna("Group Unknown")

    # Standardize keys
    b_actuals["Itemcode"] = b_actuals["Itemcode"].astype(str).str.strip()
    b_actuals["Group"] = b_actuals["Group"].astype(str).str.strip()
    b_actuals["Customer Group"] = b_actuals["Customer Group"].astype(str).str.strip()
    b_actuals["Month_ym"] = pd.PeriodIndex(b_actuals["Month_ym"], freq="M").astype(str)
    b_actuals["Quantity"] = pd.to_numeric(b_actuals["Quantity"], errors="coerce").fillna(0)

    fact = b_actuals[["Month_ym", "Quantity", "Itemcode", "Group", "Customer Group"]].copy()
    return {"FACT": fact, "Base_actuals": b_actuals, "Base_SOP": b_sop}

# -----------------------------
# Core utilities (baseline & ML)
# -----------------------------
def ensure_month_index(wide_df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.PeriodIndex(wide_df.index, freq="M")
    out = wide_df.copy()
    out.index = idx.astype(str)
    return out.sort_index()

def _as_monthly_series(y: pd.Series) -> pd.Series:
    """Return y as a monthly indexed Series with explicit freq='MS' to avoid warnings."""
    y = pd.Series(pd.to_numeric(y, errors="coerce").fillna(0.0).values)
    idx = pd.date_range("2000-01-01", periods=len(y), freq="MS")
    y.index = idx
    y.index.freq = idx.freq
    return y

def seasonal_naive_next(wide_df: pd.DataFrame, next_month: str, season: int = 12) -> pd.Series:
    idx = pd.Index(pd.PeriodIndex(wide_df.index, freq="M").astype(str))
    wide = wide_df.copy()
    wide.index = idx
    prev_same_month = str(pd.Period(next_month, "M") - season)
    if prev_same_month in wide.index:
        return wide.loc[prev_same_month].copy()
    latest = str(pd.Period(wide.index.max(), "M"))
    return wide.loc[latest].copy()

def make_eval_months(wide_df: pd.DataFrame, min_train: int = 12, last_k: int = 3) -> List[str]:
    df = ensure_month_index(wide_df)
    months = pd.PeriodIndex(df.index, freq="M")
    eligible = [str(m) for i, m in enumerate(months) if i >= min_train]
    return eligible[-last_k:]

def per_series_wape(train: pd.DataFrame, eval_months: List[str], forecaster) -> pd.Series:
    train = ensure_month_index(train)
    cols = train.columns
    numer = pd.Series(0.0, index=cols)
    denom = pd.Series(0.0, index=cols)
    for m in eval_months:
        m_prev = str(pd.Period(m, "M") - 1)
        subtrain = train.loc[:m_prev]
        actual_row = train.loc[m].astype(float)
        fc_row = forecaster(subtrain, m).reindex(cols).astype(float)
        numer += (actual_row - fc_row).abs()
        denom += actual_row.abs()
    return numer / denom.replace(0, np.nan)

def forecaster_holt_winters(train: pd.DataFrame, target_month: str,
                            season_length: int = 12, trend: str = "add", seasonal: str = "add") -> pd.Series:
    train = ensure_month_index(train)
    out = {}
    for col in train.columns:
        y_raw = train[col]
        y = _as_monthly_series(y_raw)  # <-- explicit monthly freq to silence warnings
        if y.nunique(dropna=True) < 2 or len(y) < season_length + 2:
            out[col] = float(y.iloc[-1]) if len(y) else 0.0
            continue
        try:
            model = ExponentialSmoothing(
                y, trend=trend, seasonal=seasonal, seasonal_periods=season_length,
                initialization_method="estimated"
            )
            fit = model.fit(optimized=True)
            out[col] = float(fit.forecast(1).iloc[0])
        except Exception:
            out[col] = float(y.iloc[-1]) if len(y) else 0.0
    return pd.Series(out, index=train.columns)


def _opt1_tables_from_fact(fact: pd.DataFrame):
    opt1_long = (
        fact.groupby(["Month_ym", "Group"], as_index=False)["Quantity"]
            .sum().sort_values(["Month_ym", "Group"])
    )
    all_months = pd.period_range(fact["Month_ym"].min(), fact["Month_ym"].max(), freq="M").astype(str)
    all_groups = opt1_long["Group"].unique()
    idx = pd.MultiIndex.from_product([all_months, all_groups], names=["Month_ym", "Group"])
    opt1_full = (
        opt1_long.set_index(["Month_ym", "Group"])
                 .reindex(idx, fill_value=0)
                 .reset_index().sort_values(["Month_ym", "Group"])
    )
    opt1_wide = opt1_full.pivot(index="Month_ym", columns="Group", values="Quantity").fillna(0).sort_index()
    return opt1_wide, opt1_full

def _opt2_tables_from_fact(fact: pd.DataFrame):
    opt2_long = (
        fact.groupby(["Month_ym", "Customer Group"], as_index=False)["Quantity"]
            .sum().sort_values(["Month_ym", "Customer Group"])
    )
    all_months = pd.period_range(fact["Month_ym"].min(), fact["Month_ym"].max(), freq="M").astype(str)
    all_custs = opt2_long["Customer Group"].unique()
    idx = pd.MultiIndex.from_product([all_months, all_custs], names=["Month_ym", "Customer Group"])
    opt2_full = (
        opt2_long.set_index(["Month_ym", "Customer Group"])
                 .reindex(idx, fill_value=0)
                 .reset_index().sort_values(["Month_ym", "Customer Group"])
    )
    opt2_wide = opt2_full.pivot(index="Month_ym", columns="Customer Group", values="Quantity").fillna(0).sort_index()
    return opt2_wide, opt2_full

def _latest_and_next(fact: pd.DataFrame):
    months = pd.PeriodIndex(fact["Month_ym"], freq="M")
    latest = months.max()
    return str(latest), str(latest + 1)

def build_export_bundle_from_fact(fact: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    opt1_wide, _ = _opt1_tables_from_fact(fact)
    opt2_wide, _ = _opt2_tables_from_fact(fact)

    latest_m, next_m = _latest_and_next(fact)

    opt1_train = opt1_wide.loc[:latest_m].copy()
    opt2_train = opt2_wide.loc[:latest_m].copy()

    opt1_next = seasonal_naive_next(opt1_train, next_m, season=12).rename("Forecast_qty").reset_index()
    opt1_next = opt1_next.rename(columns={"index": "Group"})
    opt1_next.insert(0, "Month_ym", next_m)
    opt1_next["Source"] = "baseline_seasonal_naive_12m"

    opt2_next = seasonal_naive_next(opt2_train, next_m, season=12).rename("Forecast_qty").reset_index()
    opt2_next = opt2_next.rename(columns={"index": "Customer Group"})
    opt2_next.insert(0, "Month_ym", next_m)
    opt2_next["Source"] = "baseline_seasonal_naive_12m"

    return {
        "latest_actual_month": latest_m,
        "next_forecast_month": next_m,
        "opt1_train_wide": opt1_train,
        "opt2_train_wide": opt2_train,
        "opt1_next_baseline_long": opt1_next,
        "opt2_next_baseline_long": opt2_next,
    }

# -----------------------------
# ML build & calibration
# -----------------------------
def _apply_total_calibration(forecast_long: pd.DataFrame, target_total: float, source_suffix: str):
    df = forecast_long.copy()
    cur_total = float(df["Forecast_qty"].sum())
    if cur_total <= 0 or target_total <= 0:
        df["Source"] = df["Source"].astype(str) + source_suffix + "_skipped"
        return df, 1.0
    factor = target_total / cur_total
    df["Forecast_qty"] = df["Forecast_qty"] * factor
    df["Source"] = df["Source"].astype(str) + source_suffix
    return df, float(factor)

def _build_opt1_ml(bundle: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    next_m = bundle["next_forecast_month"]
    train = bundle["opt1_train_wide"]

    eval_months = make_eval_months(train, min_train=12, last_k=3)
    w_base = per_series_wape(train, eval_months, seasonal_naive_next) * 100
    w_hw = per_series_wape(train, eval_months, forecaster_holt_winters) * 100

    rows = []
    for g in train.columns:
        wb, wh = w_base.get(g, np.nan), w_hw.get(g, np.nan)
        if np.isfinite(wh) and (wh < wb or not np.isfinite(wb)):
            rows.append(("holt_winters", g, wh, wb))
        else:
            rows.append(("baseline", g, wb, wh))
    sel = pd.DataFrame(rows, columns=["method_selected", "Group", "wape_selected_pct", "wape_other_pct"])

    fc_base_next = seasonal_naive_next(train, next_m, season=12)
    fc_hw_next = forecaster_holt_winters(train, next_m)

    chosen = {}
    for _, row in sel.iterrows():
        g = row["Group"]
        chosen[g] = float(fc_hw_next[g] if row["method_selected"] == "holt_winters" else fc_base_next[g])

    mix = pd.Series(chosen, name="Forecast_qty")
    mix.index.name = "Group"
    mix_long = mix.reset_index()
    mix_long.insert(0, "Month_ym", next_m)
    mix_long["Source"] = mix_long["Group"].map(
        sel.set_index("Group")["method_selected"]
    ).replace({"baseline": "baseline_seasonal_naive_12m", "holt_winters": "ml_holt_winters"})

    return {"opt1_next_mixed_long": mix_long, "opt1_selection": sel}

def _build_opt2_ml(bundle: Dict[str, pd.DataFrame], top_k: int = 30) -> Dict[str, pd.DataFrame]:
    next_m = bundle["next_forecast_month"]
    train = bundle["opt2_train_wide"]

    totals = train.sum(axis=0).sort_values(ascending=False)
    top_cust = list(totals.head(top_k).index)

    eval_months = make_eval_months(train, min_train=12, last_k=3)
    w_base_top = per_series_wape(train[top_cust], eval_months, seasonal_naive_next) * 100
    w_hw_top = per_series_wape(train[top_cust], eval_months, forecaster_holt_winters) * 100

    rows = []
    for c in top_cust:
        wb, wh = w_base_top.get(c, np.nan), w_hw_top.get(c, np.nan)
        if np.isfinite(wh) and (wh < wb or not np.isfinite(wb)):
            rows.append(("holt_winters", c, wh, wb))
        else:
            rows.append(("baseline", c, wb, wh))
    sel_top = pd.DataFrame(rows, columns=["method_selected", "Customer Group", "wape_selected_pct", "wape_other_pct"])

    fc_base_all = seasonal_naive_next(train, next_m, season=12)
    fc_hw_top = forecaster_holt_winters(train[top_cust], next_m)

    chosen = {}
    for cg in train.columns:
        if cg in top_cust:
            method = sel_top.set_index("Customer Group").loc[cg, "method_selected"]
            chosen[cg] = float(fc_hw_top[cg] if method == "holt_winters" else fc_base_all[cg])
        else:
            chosen[cg] = float(fc_base_all[cg])

    mix = pd.Series(chosen, name="Forecast_qty")
    mix.index.name = "Customer Group"
    mix_long = mix.reset_index()
    mix_long.insert(0, "Month_ym", next_m)
    src_map = sel_top.set_index("Customer Group")["method_selected"].replace({
        "baseline": "baseline_seasonal_naive_12m",
        "holt_winters": "ml_holt_winters"
    })
    mix_long["Source"] = mix_long["Customer Group"].map(src_map).fillna("baseline_seasonal_naive_12m")

    return {"opt2_next_mixed_long": mix_long, "opt2_selection": sel_top}

def _attach_ml_and_calibration(bundle: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    # Items
    opt1_ml = _build_opt1_ml(bundle)
    bundle.update(opt1_ml)

    base_total_1 = float(bundle["opt1_next_baseline_long"]["Forecast_qty"].sum())
    items_ml_cal, _ = _apply_total_calibration(opt1_ml["opt1_next_mixed_long"], base_total_1,
                                               source_suffix="+calibrated_to_baseline_total")
    bundle["opt1_next_ml_calibrated_long"] = items_ml_cal

    # Customers
    opt2_ml = _build_opt2_ml(bundle, top_k=30)
    bundle.update(opt2_ml)

    base_total_2 = float(bundle["opt2_next_baseline_long"]["Forecast_qty"].sum())
    cust_ml_cal, _ = _apply_total_calibration(opt2_ml["opt2_next_mixed_long"], base_total_2,
                                              source_suffix="+calibrated_to_baseline_total")
    bundle["opt2_next_ml_calibrated_long"] = cust_ml_cal

    return bundle

# -----------------------------
# Public API for Streamlit
# -----------------------------
def prepare_from_excel(file_path):
    """
    One-call: load → normalize Base → build export bundle (baseline + ML + calibrated).
    Returns dict with: sheets, fact, base_actuals, base_sop, exports
    """
    sheets = load_sheets(file_path)
    norm = normalize_base_to_fact(sheets["Base"])
    fact = norm["FACT"]
    bundle = build_export_bundle_from_fact(fact)
    bundle = _attach_ml_and_calibration(bundle)
    return {
        "sheets": sheets,
        "fact": fact,
        "base_actuals": norm["Base_actuals"],
        "base_sop": norm["Base_SOP"],
        "exports": bundle,
    }

APP_LABELS = {
    "items_title": "Product Demand — Item Groups (Recommended)",
    "customers_title": "Customer Demand — Customer Groups"
}


# -------- Per-series utilities (for charts & drill-down) --------
def list_item_groups(exports):
    return list(exports["opt1_train_wide"].columns)

def list_customer_groups(exports):
    return list(exports["opt2_train_wide"].columns)

def series_history(exports, view: str, name: str) -> pd.DataFrame:
    """
    view: 'items' or 'customers'
    returns tidy df: Month_ym, Quantity
    """
    if view == "items":
        s = exports["opt1_train_wide"][name]
    elif view == "customers":
        s = exports["opt2_train_wide"][name]
    else:
        raise ValueError("view must be 'items' or 'customers'")
    df = s.reset_index().rename(columns={"index": "Month_ym", name: "Quantity"})
    return df

def series_next_forecast(exports, view: str, name: str, source: str = "baseline") -> float:
    """
    source: 'baseline' | 'ml' | 'ml_calibrated'
    returns float Forecast_qty for the selected series in the selected mode
    """
    if view == "items":
        if source == "baseline":
            tbl = exports["opt1_next_baseline_long"]; key = "Group"
        elif source == "ml":
            tbl = exports["opt1_next_mixed_long"]; key = "Group"
        elif source in ("ml_calibrated","ml+calibrated"):
            tbl = exports["opt1_next_ml_calibrated_long"]; key = "Group"
        else:
            raise ValueError("source must be 'baseline','ml','ml_calibrated'")
    elif view == "customers":
        if source == "baseline":
            tbl = exports["opt2_next_baseline_long"]; key = "Customer Group"
        elif source == "ml":
            tbl = exports["opt2_next_mixed_long"]; key = "Customer Group"
        elif source in ("ml_calibrated","ml+calibrated"):
            tbl = exports["opt2_next_ml_calibrated_long"]; key = "Customer Group"
        else:
            raise ValueError("source must be 'baseline','ml','ml_calibrated'")
    else:
        raise ValueError("view must be 'items' or 'customers'")

    row = tbl.loc[tbl[key] == name]
    if row.empty:
        return float("nan")
    return float(row["Forecast_qty"].iloc[0])


# -------- Accuracy snapshot (overall WAPE over last K months) --------
def _overall_wape(train: pd.DataFrame, eval_months, forecaster, cols=None) -> float:
    """
    Sum over series of |Ai - Fi| / sum over series of Ai, aggregated across eval_months.
    cols: optional list of column names to restrict the calculation.
    """
    train = ensure_month_index(train)
    numer = 0.0
    denom = 0.0
    for m in eval_months:
        m_prev = str(pd.Period(m, "M") - 1)
        subtrain = train.loc[:m_prev]
        actual_row = train.loc[m]
        if cols is not None:
            subtrain = subtrain[cols]
            actual_row = actual_row[cols]
        fc_row = forecaster(subtrain, m).reindex(actual_row.index).astype(float)
        numer += float((actual_row.astype(float) - fc_row).abs().sum())
        denom += float(actual_row.abs().sum())
    return (numer / denom) if denom > 0 else float("nan")

def accuracy_snapshot(exports, view: str, top_k_customers: int = 30):
    """
    Returns dict with:
      months (list[str]), baseline_wape_pct (float), ml_wape_pct (float), n_series (int)
    view: 'items' or 'customers'
    """
    if view == "items":
        train = exports["opt1_train_wide"]
        cols = list(train.columns)            # all groups
        eval_months = make_eval_months(train, min_train=12, last_k=3)
        base_wape = _overall_wape(train, eval_months, seasonal_naive_next, cols=cols)
        ml_wape   = _overall_wape(train, eval_months, forecaster_holt_winters, cols=cols)
        return {
            "months": eval_months,
            "baseline_wape_pct": round(base_wape * 100, 2),
            "ml_wape_pct": round(ml_wape * 100, 2),
            "n_series": len(cols),
        }
    elif view == "customers":
        train = exports["opt2_train_wide"]
        # Restrict to Top-K (same policy we used for ML selection)
        totals = train.sum(axis=0).sort_values(ascending=False)
        cols = list(totals.head(top_k_customers).index)
        eval_months = make_eval_months(train, min_train=12, last_k=3)
        base_wape = _overall_wape(train, eval_months, seasonal_naive_next, cols=cols)
        ml_wape   = _overall_wape(train, eval_months, forecaster_holt_winters, cols=cols)
        return {
            "months": eval_months,
            "baseline_wape_pct": round(base_wape * 100, 2),
            "ml_wape_pct": round(ml_wape * 100, 2),
            "n_series": len(cols),
        }
    else:
        raise ValueError("view must be 'items' or 'customers'")


def get_items_view(source: str = "baseline", exports: Dict[str, pd.DataFrame] = None):
    if exports is None:
        raise ValueError("exports bundle is required")
    if source == "baseline":
        next_tbl = exports["opt1_next_baseline_long"]
    elif source == "ml":
        next_tbl = exports["opt1_next_mixed_long"]
    elif source in ("ml_calibrated", "ml+calibrated"):
        next_tbl = exports["opt1_next_ml_calibrated_long"]
    else:
        raise ValueError("source must be 'baseline', 'ml', or 'ml_calibrated'")
    return {
        "title": APP_LABELS["items_title"],
        "latest_actual_month": exports["latest_actual_month"],
        "next_forecast_month": exports["next_forecast_month"],
        "train_wide": exports["opt1_train_wide"],
        "next_forecast_long": next_tbl,
        "selection": exports.get("opt1_selection"),
        "source_used": source
    }

def get_customers_view(source: str = "baseline", exports: Dict[str, pd.DataFrame] = None):
    if exports is None:
        raise ValueError("exports bundle is required")
    if source == "baseline":
        next_tbl = exports["opt2_next_baseline_long"]
    elif source == "ml":
        next_tbl = exports["opt2_next_mixed_long"]
    elif source in ("ml_calibrated", "ml+calibrated"):
        next_tbl = exports["opt2_next_ml_calibrated_long"]
    else:
        raise ValueError("source must be 'baseline', 'ml', or 'ml_calibrated'")
    return {
        "title": APP_LABELS["customers_title"],
        "latest_actual_month": exports["latest_actual_month"],
        "next_forecast_month": exports["next_forecast_month"],
        "train_wide": exports["opt2_train_wide"],
        "next_forecast_long": next_tbl,
        "selection": exports.get("opt2_selection"),
        "source_used": source
    }
