import streamlit as st
import pandas as pd

# === bring in the helpers you validated in Jupyter ===
# EITHER paste their definitions here,
# OR import from a module you save (e.g., from prep import prepare_from_excel, get_items_view, get_customers_view)
from notebook_exports import (
    prepare_from_excel, get_items_view, get_customers_view,
    list_item_groups, list_customer_groups,
    series_history, series_next_forecast,
    accuracy_snapshot,
    get_itemcodes_allocated_view,  
)




REQUIRED_BASE_COLS = [
    "Delivery Date", "Month", "Quantity", "Itemcode", "Group", "Customer Group"
]

@st.cache_data(show_spinner=True, ttl=300)
def _cached_prepare_from_excel(file_obj):
    # Streamlit gives a file-like object; pandas can read it directly
    return prepare_from_excel(file_obj)


st.set_page_config(page_title="Demand Forecasting Portal", layout="wide")
st.title("Demand Forecasting Portal")
st.caption("Monthly forecasts by product or customer from a single Excel file")

with st.expander("About this app", expanded=False):
    st.markdown("""
**What this shows**
- **Product Demand — Item Groups (recommended):** total monthly quantity per item group across all customers.
- **Customer Demand — Customer Groups:** total monthly quantity per customer group across all items.

**Models**
- **Baseline:** Seasonal naïve (same month last year).
- **ML:** Holt–Winters when it beats baseline (1-step ahead); otherwise falls back to baseline.
- **ML (calibrated):** ML scaled so the **next-month total** matches the baseline total (distribution keeps ML’s shape).

**Accuracy guidance (WAPE)**
- **Items (Item Groups):** 15–20% typical.
- **Customers (Customer Groups):** 18–25% typical.

**Data handling**
- Uses a **single Excel file** for all views.
- Rows with **S&OP months** like `P-10`, `P-11`, … are **ignored** in modeling (kept only for comparison if needed).
- Latest actual month and next forecast month are **auto-detected** from the data you upload.
""")


# --- Sidebar ---
st.sidebar.header("Configuration")

excel_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])
view = st.sidebar.radio(
    "Choose a view",
    ["Product Demand — Item Groups (Recommended)", "Customer Demand — Customer Groups"],
    index=0
)
mode = st.sidebar.radio(
    "Forecast mode",
    ["baseline", "ml", "ml_calibrated"],
    index=0,
    help="ML uses Holt-Winters where it beats baseline; calibrated scales ML totals to baseline."
)

use_wd = st.sidebar.toggle("Use Working Days (WD)", value=False, help="Adjust forecasts using working days")


# how many months to look back for item mix when allocating group → itemcodes
ALLOC_LOOKBACK_MONTHS = 3

# show level choice only for the Product Demand page
level = None
if view.startswith("Product Demand"):
    level = st.sidebar.radio(
        "Product level",
        ["Groups", "Itemcodes (allocated)"],
        index=0,
        help="Groups = product families (most accurate). Itemcodes (allocated) = split by recent mix shares."
    )


if st.sidebar.button("Clear cache"):
    st.cache_data.clear()
    st.success("Cache cleared. Re-upload the Excel file.")

if excel_file is None:
    st.info("Upload the Excel file to see forecasts.")
    st.stop()

# --- Prepare data (single call) ---
try:
    prep = _cached_prepare_from_excel(excel_file)
except Exception as e:
    st.error(
        "We couldn’t read this workbook. Please ensure it contains a sheet named "
        "`Base` with these columns: "
        "`Delivery Date`, `Month`, `Quantity`, `Itemcode`, `Group`, `Customer Group`."
    )
    with st.expander("Technical details"):
        st.exception(e)
    st.stop()

base_cols = set(prep["base_actuals"].columns)
missing = [c for c in REQUIRED_BASE_COLS if c not in base_cols]
if missing:
    st.error("Missing required columns in `Base`: " + ", ".join(missing))
    st.stop()

exports = prep["exports"]
latest = exports["latest_actual_month"]
next_m = exports["next_forecast_month"]

# --- Working Days caption (sheet if present, else auto Mon–Fri) ---
wd_meta = exports.get("wd_meta")
if wd_meta:
    wd_days = int(wd_meta.get("next_days", 0))
    wd_source = wd_meta.get("wd_source", "sheet/auto")
else:
    p = pd.Period(next_m, "M")
    wd_days = len(pd.date_range(p.start_time, p.end_time, freq="B"))
    wd_source = "auto (Mon–Fri)"
st.caption(f"Working Days for {next_m}: {wd_days} days • Source: {wd_source}")

# --- Switch data source when WD toggle is ON + show Δ vs classic ---
if use_wd:
    import pandas as pd

    # 0) Snapshot classic (non-WD) NEXT tables the first time WD is turned on
    if "classic_snapshot" not in st.session_state:
        st.session_state["classic_snapshot"] = {
            "opt1_base": exports.get("opt1_next_baseline_long", pd.DataFrame()).copy(),
            "opt1_cal":  exports.get("opt1_next_ml_calibrated_long", pd.DataFrame()).copy(),
            "opt2_base": exports.get("opt2_next_baseline_long", pd.DataFrame()).copy(),
            "opt2_cal":  exports.get("opt2_next_ml_calibrated_long", pd.DataFrame()).copy(),
        }
    snap = st.session_state["classic_snapshot"]

    # 1) Option 1 (Items/Groups): use GROUP-level WD tables to avoid allocator merge issues
    if "wd_groups_next" in exports:
        g = exports["wd_groups_next"].copy()

        # Baseline from WD
        exports["opt1_next_baseline_long"] = (
            g[["Month_ym", "Group", "Forecast_qty_WD_baseline"]]
              .rename(columns={"Forecast_qty_WD_baseline": "Forecast_qty"})
              .assign(Source="wd_baseline_rate_x_days")
        )

        # ML (prefer WD-calibrated totals if available; else fall back to classic-calibrated)
        col = (
            "Forecast_qty_WD_HW_calibrated_to_WD"
            if "Forecast_qty_WD_HW_calibrated_to_WD" in g.columns
            else "Forecast_qty_WD_HW_calibrated"
        )
        label = (
            "wd_hw_rate_x_days+calibrated_to_WD_baseline"
            if col.endswith("_to_WD")
            else "wd_hw_rate_x_days+calibrated_to_classic_baseline"
        )
        exports["opt1_next_ml_calibrated_long"] = (
            g[["Month_ym", "Group", col]]
              .rename(columns={col: "Forecast_qty"})
              .assign(Source=label)
        )



    # 2) Option 2 (Customers): WD versions are already in long format
    if ("wd_opt2_next_baseline_long" in exports) and ("wd_opt2_next_calibrated_long" in exports):
        exports["opt2_next_baseline_long"]    = exports["wd_opt2_next_baseline_long"]
        exports["opt2_next_ml_calibrated_long"] = exports["wd_opt2_next_calibrated_long"]

    # 3) Show totals Δ vs classic (ML variants)
    def _tot(df: pd.DataFrame):
        return int(df["Forecast_qty"].sum()) if isinstance(df, pd.DataFrame) and "Forecast_qty" in df.columns and len(df) else None

    t1_wd = _tot(exports.get("opt1_next_ml_calibrated_long", pd.DataFrame()))
    t1_cl = _tot(snap.get("opt1_cal", pd.DataFrame()))
    if t1_wd is not None and t1_cl is not None:
        d  = t1_wd - t1_cl
        p  = (d / t1_cl * 100.0) if t1_cl else 0.0
        st.caption(f"Δ vs classic (Items/Groups, ML): {d:+,} ({p:+.2f}%)")

    t2_wd = _tot(exports.get("opt2_next_ml_calibrated_long", pd.DataFrame()))
    t2_cl = _tot(snap.get("opt2_cal", pd.DataFrame()))
    if t2_wd is not None and t2_cl is not None:
        d  = t2_wd - t2_cl
        p  = (d / t2_cl * 100.0) if t2_cl else 0.0
        st.caption(f"Δ vs classic (Customers, ML): {d:+,} ({p:+.2f}%)")

    with st.expander("What does ‘Use Working Days (WD)’ change?"):
        st.markdown(
        """
- **When OFF:** forecasts are computed on **monthly totals** (classic).
- **When ON:** we convert history to **rates** (Qty ÷ Working Days), forecast rates (Seasonal-Naive + Holt-Winters), then convert **back to totals** with    **next month’s Working Days**.
- **Calibration rule stays the same:** the ML forecast is calibrated to the **baseline** total. With WD ON, it’s calibrated to the **WD baseline** total so totals reflect fewer/more days.
- **Source of days:** uses the **‘Working Days’** sheet if present; otherwise **Mon–Fri (no holidays)** is auto-computed.
- **Why this helps:** when demand scales with business days, WD reduces calendar effects and can improve accuracy (often a small gain, ~1–3 WAPE points).
        """
        )


    # 4) Per-row Δ / Δ% vs classic for Items/Groups (ML table)
    # We enrich the WD table in-place so your existing table render shows these columns.
    try:
        import pandas as pd
        df_wd = exports.get("opt1_next_ml_calibrated_long")
        df_cl = snap.get("opt1_cal")
        if isinstance(df_wd, pd.DataFrame) and isinstance(df_cl, pd.DataFrame) and not df_wd.empty and not df_cl.empty:
            base = df_cl[["Group", "Forecast_qty"]].rename(columns={"Forecast_qty": "Forecast_qty_classic"})
            df = df_wd.merge(base, on="Group", how="left")
            df["Δ"] = df["Forecast_qty"] - df["Forecast_qty_classic"].fillna(0).astype(int)
            den = df["Forecast_qty_classic"].replace({0: pd.NA})
            df["Δ%"] = ((df["Δ"] / den) * 100).round(2).fillna(0.0)
            exports["opt1_next_ml_calibrated_long"] = df
    except Exception:
        pass

    # 5) WD backtest (baseline only) — WAPE classic vs WD (Mon–Fri days)
    try:
        import pandas as pd, numpy as np
        from notebook_exports import ensure_month_index

        # Training totals per month x group (classic)
        T = ensure_month_index(exports["opt1_train_wide"].copy())

        # Working days per month (Mon–Fri) for the same months
        months = T.index.astype(str)
        days = pd.Series(
            [len(pd.date_range(pd.Period(m, "M").start_time, pd.Period(m, "M").end_time, freq="B")) for m in months],
            index=months, name="Days"
        )

        # Rates = totals / days
        R = T.div(days, axis=0)

        # Seasonal-naive backtest: predict month m using m-12
        T_hat_classic = T.shift(12)            # classic totals
        T_hat_wd      = R.shift(12).mul(days, axis=0)  # rate(m-12) * days(m)

        # Drop the first 12 months (no forecast target)
        Y = T.iloc[12:]
        C = T_hat_classic.iloc[12:].reindex_like(Y).fillna(0.0)
        W = T_hat_wd.iloc[12:].reindex_like(Y).fillna(0.0)

        # Aggregate across all series and months
        err_c = (Y - C).abs().to_numpy().sum()
        err_w = (Y - W).abs().to_numpy().sum()
        denom = Y.to_numpy().sum()

        if denom > 0:
            wape_c = 100.0 * err_c / denom
            wape_w = 100.0 * err_w / denom
            delta  = wape_w - wape_c
            st.caption(f"WAPE backtest (Seasonal-Naive baseline): classic {wape_c:.2f}% → WD {wape_w:.2f}% ({delta:+.2f} pts)")
    except Exception:
        # Keep UI resilient if backtest cannot be computed
        pass



# --- Route to the right view getter ---
if view.startswith("Product Demand"):
    if level == "Itemcodes (allocated)":
        v_alloc = get_itemcodes_allocated_view(
            mode=mode,
            exports=exports,
            lookback_months=ALLOC_LOOKBACK_MONTHS,
        )

        # --- Per-Itemcode Δ / Δ% vs classic when WD is ON ---
        if use_wd and "classic_snapshot" in st.session_state:
            try:
                import pandas as pd

                # Classic (non-WD) snapshot captured earlier in the WD toggle block
                snap = st.session_state["classic_snapshot"]

                # Build classic allocated items for the SAME mode (baseline or ml_calibrated)
                classic_tmp = dict(exports)  # shallow copy
                classic_tmp["opt1_next_baseline_long"]      = snap.get("opt1_base")
                classic_tmp["opt1_next_ml_calibrated_long"] = snap.get("opt1_cal")

                v_classic = get_itemcodes_allocated_view(
                    mode=mode,
                    exports=classic_tmp,
                    lookback_months=ALLOC_LOOKBACK_MONTHS,
                )

                # Merge WD vs classic at itemcode level
                df_wd = v_alloc["next_forecast_long"].copy()
                df_cl = (
                    v_classic["next_forecast_long"][["Group", "Itemcode", "Forecast_qty"]]
                    .rename(columns={"Forecast_qty": "Forecast_qty_classic"})
                )

                df = df_wd.merge(df_cl, on=["Group", "Itemcode"], how="left")
                df["Δ"] = df["Forecast_qty"] - df["Forecast_qty_classic"].fillna(0).astype(int)
                den = df["Forecast_qty_classic"].replace({0: pd.NA})
                df["Δ%"] = ((df["Δ"] / den) * 100).round(2).fillna(0.0)

                # Replace the table the app renders
                v_alloc["next_forecast_long"] = df
            except Exception:
                # Keep UI resilient even if delta calc fails
                st.caption("Δ columns unavailable for Itemcodes (classic comparison failed).")

    else:
        v = get_items_view(mode, exports)
else:
    v = get_customers_view(mode, exports)

 



# --- Header stats ---
lcol, rcol = st.columns([1,1])
with lcol:
    st.metric("Latest actual month", latest)
with rcol:
    st.metric("Next forecast month", next_m)

# --- Accuracy snapshot (last 3 months on training) ---
snap_view = "items" if view.startswith("Product Demand") else "customers"
snap = accuracy_snapshot(exports, view=snap_view, top_k_customers=30)

with st.expander("Accuracy snapshot (last 3 months, training data)", expanded=False):
    c1, c2, c3 = st.columns([1,1,1])
    c1.metric("Series evaluated", f"{snap['n_series']}")
    c2.metric("Baseline WAPE", f"{snap['baseline_wape_pct']}%")
    c3.metric("ML WAPE", f"{snap['ml_wape_pct']}%")
    st.caption(f"Months: {', '.join(snap['months'])} • View: {snap_view}")


# Header works for both levels
if view.startswith("Product Demand") and level == "Itemcodes (allocated)":
    page_title = f"Product Demand — Itemcodes (allocated) | latest/next: {exports['latest_actual_month']} → {exports['next_forecast_month']}"
else:
    page_title = v["title"]
st.subheader(page_title)


# --- Next-month forecast (table) ---
if view.startswith("Product Demand") and level == "Itemcodes (allocated)":
    st.markdown("**Next-month forecast (Itemcodes — allocated)**")

    next_tbl = v_alloc["next_forecast_long"].copy()
    st.dataframe(next_tbl, use_container_width=True)

    total = float(next_tbl["Forecast_qty"].sum())
    st.markdown(f"**Total forecasted quantity (next month):** {total:,.0f}")
    st.caption(
        f"Mode: {v_alloc.get('source_used','baseline')}+allocated • "
        f"Rows: {len(next_tbl)} • Total: {total:,.0f} • "
        f"Lookback: {ALLOC_LOOKBACK_MONTHS}m shares"
    )

    # CSV download for per-Itemcode
    csv = next_tbl.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download per-Itemcode forecast (CSV)",
        data=csv,
        file_name=f"forecast_itemcodes_{v_alloc['next_month']}_{mode}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Optional: show the allocation shares used
    with st.expander("Show allocation shares (by Group → Itemcode)", expanded=False):
        st.dataframe(v_alloc["shares_table"], use_container_width=True)
        st.caption("Share basis: 3m / 12m / lifetime / equal")

    st.stop()  # hide group-only sections below when viewing Itemcodes (allocated)

else:
    # Existing rendering for Groups or Customers
    st.markdown("**Next-month forecast table**")
    next_tbl = v["next_forecast_long"].copy()
    st.dataframe(next_tbl, use_container_width=True)
    total = float(next_tbl["Forecast_qty"].sum())
    st.markdown(f"**Total forecasted quantity (next month):** {total:,.0f}")
    st.caption(f"Mode: {v.get('source_used','baseline')} • Rows: {len(next_tbl)} • Total: {total:,.0f}")

    # (keep your existing CSV download here if you already had one)


# --- Training history (wide) ---
st.markdown("---")
st.markdown("**Training history (monthly totals)**")
train_tbl = v["train_wide"].copy()
st.dataframe(train_tbl, use_container_width=True)

st.caption(f"History coverage: {train_tbl.shape[0]} months × {train_tbl.shape[1]} series")


# --- Drill-down: select a single series and plot its history ---
st.markdown("---")
st.markdown("**Drill-down: time series**")

if view.startswith("Product Demand"):
    names = list_item_groups(exports)
    label = "Item Group"
    chosen = st.selectbox(f"Select {label}", names, index=(names.index("Group 3") if "Group 3" in names else 0))
    hist_df = series_history(exports, view="items", name=chosen)
    next_val = series_next_forecast(exports, view="items", name=chosen, source=v.get("source_used", "baseline"))
else:
    names = list_customer_groups(exports)
    label = "Customer Group"
    chosen = st.selectbox(f"Select {label}", names, index=0)
    hist_df = series_history(exports, view="customers", name=chosen)
    next_val = series_next_forecast(exports, view="customers", name=chosen, source=v.get("source_used", "baseline"))

# Convert Month_ym -> datetime index for plotting
hist_plot = hist_df.copy()
hist_plot["date"] = pd.PeriodIndex(hist_plot["Month_ym"], freq="M").to_timestamp()
hist_plot = hist_plot.set_index("date")[["Quantity"]]

st.line_chart(hist_plot, use_container_width=True)
st.metric(f"Next forecast for {chosen}", f"{next_val:,.0f}", help=f"Mode: {v.get('source_used', 'baseline')}")


# --- (Optional) per-series selection table for ML vs baseline (Items/Customers) ---
if v.get("selection") is not None and mode != "baseline":
    st.markdown("---")
    st.markdown("**Model selection (per series)**")
    st.dataframe(v["selection"], use_container_width=True)

# --- CSV export buttons ---
st.markdown("---")
st.markdown("**Export**")

# Which view are we on?
view_slug = "items" if view.startswith("Product Demand") else "customers"

# Current next-month table (what the user sees)
next_tbl = v["next_forecast_long"].copy()

# Build filenames
fname_forecast = f"forecast_{view_slug}_{v.get('source_used','baseline')}_{next_m}.csv"
fname_selection = f"model_selection_{view_slug}_{v.get('source_used','baseline')}_{next_m}.csv"

c1, c2 = st.columns(2)

with c1:
    st.download_button(
        label="Download next-month forecast (CSV)",
        data=next_tbl.to_csv(index=False).encode("utf-8"),
        file_name=fname_forecast,
        mime="text/csv"
    )

with c2:
    sel = v.get("selection")
    if sel is not None and v.get("source_used") != "baseline":
        st.download_button(
            label="Download per-series model selection (CSV)",
            data=sel.to_csv(index=False).encode("utf-8"),
            file_name=fname_selection,
            mime="text/csv"
        )
    else:
        st.caption("Per-series selection is available in ML modes.")
