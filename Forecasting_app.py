import streamlit as st
import pandas as pd

# === bring in the helpers you validated in Jupyter ===
# EITHER paste their definitions here,
# OR import from a module you save (e.g., from prep import prepare_from_excel, get_items_view, get_customers_view)
from notebook_exports import (
    prepare_from_excel, get_items_view, get_customers_view,
    list_item_groups, list_customer_groups,
    series_history, series_next_forecast,
    accuracy_snapshot
)


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

if st.sidebar.button("Clear cache"):
    st.cache_data.clear()
    st.success("Cache cleared. Re-upload the Excel file.")

if excel_file is None:
    st.info("Upload the Excel file to see forecasts.")
    st.stop()

# --- Prepare data (single call) ---
prep = _cached_prepare_from_excel(excel_file)
exports = prep["exports"]
latest = exports["latest_actual_month"]
next_m = exports["next_forecast_month"]

# --- Route to the right view getter ---
if view.startswith("Product Demand"):
    v = get_items_view(source=mode, exports=exports)
else:
    v = get_customers_view(source=mode, exports=exports)

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


st.subheader(v["title"])

# --- Next-month forecast (tidy long) ---
st.markdown("**Next-month forecast table**")
next_tbl = v["next_forecast_long"].copy()
st.dataframe(next_tbl, use_container_width=True)

total = float(next_tbl["Forecast_qty"].sum())
st.markdown(f"**Total forecasted quantity (next month):** {total:,.0f}")

st.caption(f"Mode: {v.get('source_used','baseline')} • Rows: {len(next_tbl)} • Total: {total:,.0f}")


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
