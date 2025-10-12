# Demand Forecasting Portal

Rolling monthly forecasts from a single Excel file, shown in two views:

- **Product Demand — Item Groups (recommended):** total monthly quantity per item group across all customers.
- **Customer Demand — Customer Groups:** total monthly quantity per customer group across all items.

The app auto-detects the **latest actual month** in your data and forecasts the **next month**.

---

## 1) What you upload

**Filename does not matter.** The workbook **structure** does:

- **Required sheet:** `Base`
- **Required columns in `Base`:**
  - `Delivery Date` — real dates or Excel serial numbers (both supported)
  - `Month` — `YYYY-M` or `YYYY-MM` for actuals. S&OP rows like `P-10`, `P-11`, … are **ignored** in modeling.
  - `Quantity`
  - `Itemcode`
  - `Group`
  - `Customer Group`

**Optional sheets (loaded but not required by the models):**
- `Customers`  
- `Items`  
- `Working Days`  
- `Temperature` (column is `TempFactor`)  
- `Frecuency`

> We model only actuals from `Base`. S&OP months (`P-#`) are kept separate for comparison if needed.

---

## 2) Install & run

**Python**: 3.10+ recommended

```bash
# create and activate a virtual env (optional but recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
