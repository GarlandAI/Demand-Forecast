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

# --- Working Days helpers (sheet-or-auto) ---
import pandas as pd  # usually already imported; harmless if duplicated

def _to_month_ym(s):
    """Normalize ANY month strings like '2025-1' → '2025-01' (YYYY-MM)."""
    return pd.PeriodIndex(pd.Series(s).astype(str), freq="M").astype(str)

def _auto_working_days_from_range(start_ym: str, end_ym: str) -> pd.DataFrame:
    """Mon–Fri working days per month from start_ym..end_ym (inclusive)."""
    months = pd.period_range(start_ym, end_ym, freq="M")
    rows = []
    for m in months:
        p = pd.Period(m, "M")
        # count business days using pandas only (no numpy dependency)
        days = len(pd.date_range(p.start_time, p.end_time, freq="B"))
        rows.append((str(m), days))
    return pd.DataFrame(rows, columns=["Month_ym", "Days"])


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

def _attach_wd_forecasts(bundle, fact, file_path):
    """
    Build WD-adjusted forecasts for Option 1 (Groups):
    - compute Rates (qty / working-days),
    - forecast next-month Rates (seasonal-naive + Holt–Winters),
    - convert back to totals with next-month WD,
    - calibrate HW totals to (a) classic baseline total and (b) WD baseline total,
    - allocate to Itemcodes using existing share logic.
    Returns the augmented bundle.
    """
    import pandas as pd, numpy as np

    series_key = "Group"
    mn = str(fact["Month_ym"].min())
    mx = str(fact["Month_ym"].max())
    next_m = str(pd.Period(mx, "M") + 1)

    # --- Working days map for full range (sheet if present, else auto; fill any gaps)
    wd_map = {}
    try:
        xl = pd.ExcelFile(file_path)
        if "Working Days" in xl.sheet_names:
            wd_raw = pd.read_excel(file_path, sheet_name="Working Days")
            wd = (wd_raw.rename(columns={"Month":"Month_ym","month":"Month_ym",
                                         "Days":"Days","days":"Days"})[["Month_ym","Days"]]
                        .dropna())
            wd["Month_ym"] = _to_month_ym(wd["Month_ym"])
            wd["Days"] = wd["Days"].astype(int)
            wd_map = dict(zip(wd["Month_ym"], wd["Days"]))
    except Exception:
        pass

    auto_full  = _auto_working_days_from_range(mn, mx).set_index("Month_ym")
    sheet_part = (pd.DataFrame(list(wd_map.items()), columns=["Month_ym","Days"])
                    .set_index("Month_ym")) if wd_map else pd.DataFrame(columns=["Days"])
    wd_full = auto_full.combine_first(sheet_part).reset_index()
    wd_map  = dict(zip(wd_full["Month_ym"], wd_full["Days"]))
    next_days = int(wd_map.get(next_m, _auto_working_days_from_range(next_m, next_m).iloc[0]["Days"]))

    # --- Monthly totals & rates per group
    monthly = (fact.groupby(["Month_ym", series_key], as_index=False)["Quantity"].sum()
                    .rename(columns={"Quantity":"Qty"}))
    monthly["Days"] = monthly["Month_ym"].map(wd_map)
    monthly["Rate"] = monthly["Qty"] / monthly["Days"]

    # Wide matrices (Totals and Rates)
    qty_wide  = monthly.pivot(index="Month_ym", columns=series_key, values="Qty").fillna(0.0)
    rate_wide = monthly.pivot(index="Month_ym", columns=series_key, values="Rate").fillna(0.0)
    qty_wide  = ensure_month_index(qty_wide)
    rate_wide = ensure_month_index(rate_wide)

    # --- Forecasts at RATE level
    baseline_rate = seasonal_naive_next(rate_wide, next_m)     # Series indexed by group
    hw_rate       = forecaster_holt_winters(rate_wide, next_m) # Series indexed by group

    # Convert back to totals with next month's WD
    wd_baseline_qty = (baseline_rate * next_days).round().astype(int).rename("Forecast_qty").reset_index()
    wd_baseline_qty.insert(0, "Month_ym", next_m)

    wd_hw_qty_float = hw_rate * next_days

    # --- Calibrate WD HW to the classic (non-WD) baseline total
    target_total = int(bundle["opt1_next_baseline_long"]["Forecast_qty"].sum())
    curr_total   = float(wd_hw_qty_float.sum())
    scale = (target_total / curr_total) if curr_total > 0 else 1.0

    scaled = wd_hw_qty_float.values * scale
    base = np.floor(scaled).astype(int)
    residue = target_total - int(base.sum())
    if residue != 0:
        frac = scaled - base
        order = np.argsort(-frac) if residue > 0 else np.argsort(frac)
        for i in order[:abs(residue)]:
            base[i] += 1 if residue > 0 else -1
    wd_hw_cal = pd.DataFrame({series_key: hw_rate.index, "Forecast_qty": base})
    wd_hw_cal.insert(0, "Month_ym", next_m)

    # --- Alternative calibration: match WD baseline total (used when WD toggle = ON)
    target_total_wd = int(wd_baseline_qty["Forecast_qty"].sum())
    scale_wd = (target_total_wd / curr_total) if curr_total > 0 else 1.0

    scaled_wd  = wd_hw_qty_float.values * scale_wd
    base_wd    = np.floor(scaled_wd).astype(int)
    residue_wd = target_total_wd - int(base_wd.sum())
    if residue_wd != 0:
        frac  = scaled_wd - base_wd
        order = np.argsort(-frac) if residue_wd > 0 else np.argsort(frac)
        for i in order[:abs(residue_wd)]:
            base_wd[i] += 1 if residue_wd > 0 else -1
    wd_hw_cal_wd = pd.DataFrame({series_key: hw_rate.index, "Forecast_qty": base_wd})
    wd_hw_cal_wd.insert(0, "Month_ym", next_m)

    # --- Allocate to Itemcodes using existing logic
    tmp1 = dict(bundle)
    tmp1["opt1_next_baseline_long"]      = wd_baseline_qty.copy()
    tmp1["opt1_next_ml_calibrated_long"] = wd_hw_cal.copy()
    alloc_wd_baseline = get_itemcodes_allocated_view(mode="baseline",      exports=tmp1)
    alloc_wd_cal      = get_itemcodes_allocated_view(mode="ml_calibrated", exports=tmp1)

    tmp2 = dict(bundle)
    tmp2["opt1_next_baseline_long"]      = wd_baseline_qty.copy()
    tmp2["opt1_next_ml_calibrated_long"] = wd_hw_cal_wd.copy()
    alloc_wd_cal_wd  = get_itemcodes_allocated_view(mode="ml_calibrated", exports=tmp2)

    # --- Save outputs into bundle
    bundle["wd_groups_next"] = (
        wd_baseline_qty.rename(columns={"Forecast_qty":"Forecast_qty_WD_baseline"})
        .merge(
            wd_hw_cal.rename(columns={"Forecast_qty":"Forecast_qty_WD_HW_calibrated"}),
            on=["Month_ym", series_key],
            how="left"
        )
        .merge(
            wd_hw_cal_wd.rename(columns={"Forecast_qty":"Forecast_qty_WD_HW_calibrated_to_WD"}),
            on=["Month_ym", series_key],
            how="left"
        )
    )
    bundle["wd_next_baseline_long"]        = alloc_wd_baseline["next_forecast_long"]
    bundle["wd_next_calibrated_long"]      = alloc_wd_cal["next_forecast_long"]
    bundle["wd_next_calibrated_to_wd_long"]= alloc_wd_cal_wd["next_forecast_long"]

    # keep/meta nits
    bundle.setdefault("wd_meta", {})
    bundle["wd_meta"].update({"next_month": next_m, "next_days": int(next_days)})

    return bundle



def _attach_opt2_wd_forecasts(bundle, file_path):
    """
    Build WD-adjusted forecasts for Option 2 (Customer Groups).
    Uses opt2_train_wide (totals), converts to rates via working days (sheet if present, else Mon–Fri),
    forecasts next-month rates (seasonal-naive + Holt–Winters), converts back to totals with next-month WD,
    calibrates HW totals to the classic (non-WD) Option-2 baseline total, and stores results.
    """
    import pandas as pd, numpy as np

    # If Option 2 isn't present, nothing to do
    if "opt2_train_wide" not in bundle or "opt2_next_baseline_long" not in bundle:
        return bundle

    # Detect series key label from existing long format (e.g., "Customer Group")
    series2_key = [c for c in bundle["opt2_next_baseline_long"].columns
                   if c not in ["Month_ym", "Forecast_qty", "Source"]][0]

    # Training totals (months x customers) and month range
    train2 = ensure_month_index(bundle["opt2_train_wide"].copy())
    months = train2.index.astype(str)
    mn, mx = months.min(), months.max()

    # Next month & WD meta
    wd_meta  = bundle.get("wd_meta", {})
    next_m   = wd_meta.get("next_month", str(pd.Period(mx, "M") + 1))
    next_days = int(wd_meta.get("next_days", len(pd.date_range(pd.Period(next_m, "M").start_time,
                                                              pd.Period(next_m, "M").end_time, freq="B"))))

    # --- Working days map for full range (sheet if present, else auto; fill any gaps)
    wd_map = {}
    try:
        xl = pd.ExcelFile(file_path)
        if "Working Days" in xl.sheet_names:
            wd_raw = pd.read_excel(file_path, sheet_name="Working Days")
            wd = (wd_raw.rename(columns={"Month":"Month_ym","month":"Month_ym",
                                         "Days":"Days","days":"Days"})[["Month_ym","Days"]]
                        .dropna())
            wd["Month_ym"] = _to_month_ym(wd["Month_ym"])
            wd["Days"] = wd["Days"].astype(int)
            wd_map = dict(zip(wd["Month_ym"], wd["Days"]))
    except Exception:
        pass

    auto_full  = _auto_working_days_from_range(mn, mx).set_index("Month_ym")
    sheet_part = (pd.DataFrame(list(wd_map.items()), columns=["Month_ym","Days"])
                    .set_index("Month_ym")) if wd_map else pd.DataFrame(columns=["Days"])
    wd_full = auto_full.combine_first(sheet_part).reset_index()
    wd_map  = dict(zip(wd_full["Month_ym"], wd_full["Days"]))

    # Historical rates = totals / WD
    days_s = pd.Series([wd_map[m] for m in months], index=months, name="Days")
    rate_wide_opt2 = train2.div(days_s, axis=0)

    # Forecast rates, then convert back to totals with next month WD
    baseline_rate2 = seasonal_naive_next(rate_wide_opt2, next_m)    # Series indexed by customer
    hw_rate2       = forecaster_holt_winters(rate_wide_opt2, next_m)

    wd2_baseline_long = (baseline_rate2 * next_days).round().astype(int).rename("Forecast_qty").reset_index()
    if "index" in wd2_baseline_long.columns:
        wd2_baseline_long = wd2_baseline_long.rename(columns={"index": series2_key})
    wd2_baseline_long.insert(0, "Month_ym", next_m)
    wd2_baseline_long["Source"] = "wd_baseline_rate_x_days"

    # Calibrate WD HW totals to classic (non-WD) Option-2 baseline total
    wd2_hw_float = hw_rate2 * next_days
    target2 = int(bundle["opt2_next_baseline_long"]["Forecast_qty"].sum())
    curr2   = float(wd2_hw_float.sum())
    scale2  = (target2 / curr2) if curr2 > 0 else 1.0

    scaled  = wd2_hw_float.values * scale2
    base    = np.floor(scaled).astype(int)
    residue = target2 - int(base.sum())
    if residue != 0:
        frac  = scaled - base
        order = np.argsort(-frac) if residue > 0 else np.argsort(frac)
        base[order[:abs(residue)]] += 1 if residue > 0 else -1

    wd2_hw_cal_long = (pd.DataFrame({series2_key: hw_rate2.index, "Forecast_qty": base})
                        .assign(Month_ym=next_m, Source="wd_hw_rate_x_days+calibrated_to_classic_baseline")
                        .loc[:, ["Month_ym", series2_key, "Forecast_qty", "Source"]])

    # Save in bundle (for app toggle)
    bundle["wd_opt2_next_baseline_long"]   = wd2_baseline_long
    bundle["wd_opt2_next_calibrated_long"] = wd2_hw_cal_long

    return bundle




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
    bundle["fact"] = fact[["Month_ym", "Group", "Itemcode", "Quantity"]].copy()
    bundle = _attach_wd_forecasts(bundle, fact, file_path)
    bundle = _attach_opt2_wd_forecasts(bundle, file_path)


    # --- WD meta (sheet-or-auto) for UI caption & toggle readiness ---
    # next month after last actual
    next_m = str(pd.Period(fact["Month_ym"].max(), "M") + 1)

    # try sheet; else auto Mon–Fri
    wd_source = "auto"
    wd_map = {}
    try:
        xl = pd.ExcelFile(file_path)
        if "Working Days" in xl.sheet_names:
            wd_raw = pd.read_excel(file_path, sheet_name="Working Days")
            wd = (
                wd_raw.rename(columns={"Month": "Month_ym", "month": "Month_ym",
                                       "Days": "Days", "days": "Days"})[["Month_ym", "Days"]]
                   .dropna()
            )
            wd["Month_ym"] = _to_month_ym(wd["Month_ym"])
            wd["Days"] = wd["Days"].astype(int)
            wd_map = dict(zip(wd["Month_ym"], wd["Days"]))
            wd_source = "sheet"
    except Exception:
        # keep auto fallback
        pass

    # days for the next forecast month
    if next_m in wd_map:
        next_days = int(wd_map[next_m])
    else:
        next_days = int(_auto_working_days_from_range(next_m, next_m).iloc[0]["Days"])

    bundle["wd_meta"] = {
        "wd_source": wd_source,
        "range_min": str(fact["Month_ym"].min()),
        "range_max": str(fact["Month_ym"].max()),
        "next_month": next_m,
        "next_days": next_days,
    }

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


# ---------- Itemcode allocation (top-down from Group forecast) ----------

def _all_months_from_exports(exports) -> list[str]:
    """Chronological list of YYYY-MM months from the item-group training table."""
    idx = exports["opt1_train_wide"].index
    return [str(p) for p in idx]  # already PeriodIndex('M') in our prep

def _item_desc_map_from_exports(exports):
    """
    Try to build {Itemcode -> Item Description} from any actuals we have.
    Returns a pandas Series or None if not available.
    """
    # Prefer a full actuals table if present
    for key in ["base_actuals", "actuals", "base"]:
        if key in exports:
            df = exports[key]
            if {"Itemcode", "Item Description"}.issubset(df.columns):
                m = (df.dropna(subset=["Itemcode"])
                       .drop_duplicates(subset=["Itemcode"], keep="last")
                       .set_index("Itemcode")["Item Description"])
                return m
    # Fallback: if FACT happens to carry description in future
    if "fact" in exports and "Item Description" in exports["fact"].columns:
        df = exports["fact"]
        m = (df.dropna(subset=["Itemcode"])
               .drop_duplicates(subset=["Itemcode"], keep="last")
               .set_index("Itemcode")["Item Description"])
        return m
    return None

def _compute_window_shares(fact: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """
    Compute shares per (Group, Itemcode) within a month window.
    Returns columns: Group, Itemcode, share
    """
    f = fact.loc[fact["Month_ym"].isin(months), ["Group", "Itemcode", "Quantity"]].copy()
    grp_item = f.groupby(["Group", "Itemcode"], as_index=False)["Quantity"].sum()
    grp_tot = grp_item.groupby("Group", as_index=False)["Quantity"].sum().rename(columns={"Quantity": "GroupTotal"})
    m = grp_item.merge(grp_tot, on="Group", how="left")
    m["share"] = m["Quantity"] / m["GroupTotal"]
    return m[["Group", "Itemcode", "share"]]

def _complete_group_item_grid(fact: pd.DataFrame) -> pd.DataFrame:
    """All (Group, Itemcode) pairs that ever appeared in actuals."""
    pairs = fact[["Group", "Itemcode"]].drop_duplicates().reset_index(drop=True)
    return pairs

def _assemble_shares_with_fallbacks(exports, lookback_months: int = 3) -> pd.DataFrame:
    """
    Build final shares with fallbacks: last K months -> last 12 months -> lifetime -> equal.
    Returns: Group, Itemcode, share, basis
    """
    if "fact" not in exports:
        raise ValueError("exports must include 'fact' with Month_ym, Group, Itemcode, Quantity.")
    fact = exports["fact"].copy()
    all_months = _all_months_from_exports(exports)
    # windows
    win_k   = all_months[-lookback_months:] if lookback_months and len(all_months) >= lookback_months else all_months
    win_12  = all_months[-12:] if len(all_months) >= 12 else all_months
    win_all = all_months

    base_grid = _complete_group_item_grid(fact)

    s_k   = _compute_window_shares(fact, win_k).rename(columns={"share": "share_k"})
    s_12  = _compute_window_shares(fact, win_12).rename(columns={"share": "share_12"})
    s_all = _compute_window_shares(fact, win_all).rename(columns={"share": "share_all"})

    m = (base_grid
         .merge(s_k,   on=["Group", "Itemcode"], how="left")
         .merge(s_12,  on=["Group", "Itemcode"], how="left")
         .merge(s_all, on=["Group", "Itemcode"], how="left"))

    # choose first available share and record basis
    m["share"] = m["share_k"].where(m["share_k"].notna(), m["share_12"].where(m["share_12"].notna(), m["share_all"]))
    m["basis"] = pd.NA
    m.loc[m["share_k"].notna(), "basis"] = f"{len(win_k)}m"
    m.loc[m["basis"].isna() & m["share_12"].notna(), "basis"] = "12m"
    m.loc[m["basis"].isna() & m["share_all"].notna(), "basis"] = "lifetime"

    # equal split for any remaining null shares within a group
    def _fill_equal(group_df: pd.DataFrame) -> pd.DataFrame:
        if group_df["share"].notna().all():
            # normalize tiny numeric drift to 1.0
            s = group_df["share"].sum()
            if s and abs(1.0 - float(s)) > 1e-6:
                group_df["share"] = group_df["share"] / s
            return group_df
        mask_missing = group_df["share"].isna()
        n_missing = int(mask_missing.sum())
        if n_missing == 0:
            return group_df
        remainder = 1.0 - float(group_df.loc[~mask_missing, "share"].sum(skipna=True))
        eq = max(remainder, 0.0) / n_missing if n_missing > 0 else 0.0
        group_df.loc[mask_missing, "share"] = eq
        group_df.loc[mask_missing, "basis"] = group_df.loc[mask_missing, "basis"].fillna("equal")
        # final normalization to sum=1.0 (safety)
        s = float(group_df["share"].sum())
        if s > 0:
            group_df["share"] = group_df["share"] / s
        return group_df

    m = m.groupby("Group", group_keys=False).apply(_fill_equal)
    return m[["Group", "Itemcode", "share", "basis"]]

def get_itemcodes_allocated_view(mode: str, exports, lookback_months: int = 3):
    """
    mode: 'baseline' | 'ml' | 'ml_calibrated'
    Returns dict with:
      - next_forecast_long: Month_ym, Group, Itemcode, Item Description(optional), Forecast_qty, Share_used, Basis, Source
      - shares_table: Group, Itemcode, share, basis
      - source_used: str
      - next_month: str
    """
    mode = mode.lower()
    if mode not in ("baseline", "ml", "ml_calibrated"):
        raise ValueError("mode must be 'baseline', 'ml', or 'ml_calibrated'")

    key_map = {
        "baseline":      "opt1_next_baseline_long",
        "ml":            "opt1_next_mixed_long",
        "ml_calibrated": "opt1_next_ml_calibrated_long",
    }
    next_groups = exports[key_map[mode]].copy()
    shares = _assemble_shares_with_fallbacks(exports, lookback_months=lookback_months)

    # Merge: each group forecast spread to itemcodes via shares
    alloc = shares.merge(next_groups[["Group", "Forecast_qty"]], on="Group", how="left", validate="many_to_one")
    alloc["Forecast_qty"] = alloc["Forecast_qty"].fillna(0) * alloc["share"].fillna(0)

    # Attach Month and optional description
    next_m = exports["next_forecast_month"]
    alloc.insert(0, "Month_ym", next_m)

    desc_map = _item_desc_map_from_exports(exports)
    if desc_map is not None:
        alloc = alloc.merge(desc_map.rename("Item Description"), left_on="Itemcode", right_index=True, how="left")

    # Clean columns
    alloc = alloc.rename(columns={"share": "Share_used", "basis": "Basis"})
    alloc["Source"] = f"{mode}+allocated"

    # Order nicely
    col_order = ["Month_ym", "Group", "Itemcode"]
    if "Item Description" in alloc.columns:
        col_order += ["Item Description"]
    col_order += ["Forecast_qty", "Share_used", "Basis", "Source"]
    alloc = alloc[col_order].sort_values(["Group", "Forecast_qty"], ascending=[True, False]).reset_index(drop=True)

    return {
        "next_forecast_long": alloc,
        "shares_table": shares,
        "source_used": mode,
        "next_month": next_m,
    }




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
