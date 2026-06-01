import os
import time
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import io
import difflib
from datetime import date

# Fetch Credentials securely from Render Environment
CLIENT_ID   = os.getenv("MORNINGSTAR_CLIENT_ID")
ACCESS_CODE = os.getenv("MORNINGSTAR_ACCESS_CODE")
UNIVERSE_ID = os.getenv("MORNINGSTAR_UNIVERSE_ID")
BASE_URL    = "https://api.morningstar.com/v2/service/mf"

# In-Memory Cache to save Render resources
_UNIVERSE_MEM_CACHE = None
_UNIVERSE_CACHE_DATE = None

# ==========================================
# THE ALIAS MAP (FAILSAFE)
# If the Debug column says 'NOT FOUND', map the names manually here.
# Format: "your exact csv name in lowercase": "morningstar exact name in lowercase"
# ==========================================
ALIAS_MAP = {
    # Example: "united-i global balanced fund myr class": "uob united-i global balanced myr",
}

OUTPUT_COLUMNS = [
    ("Fund Name",                                "name"),
    ("Fund Currency",                            "currency"),
    ("Fund Category",                            "category"),
    ("Performance Figures based on Prices as at","nav_date"),
    ("3 Year Sharp Ratio",                       "sharpe_3y"),
    ("YTD (%)",                                  "return_ytd"),
    ("1-wk (%)",                                 "return_1w"),
    ("1-mth (%)",                                "return_1m"),
    ("3-mth (%)",                                "return_3m"),
    ("6-mth (%)",                                "return_6m"),
    ("1-yr (%)",                                 "return_1y"),
    ("2-yr (%)",                                 "return_2y"),
    ("3-yr (%)",                                 "return_3y"),
    ("5-yr (%)",                                 "return_5y"),
    ("10-yr (%)",                                "return_10y"),
    ("3 yr Volatility (%)",                      "stddev_3y"),
    ("Fund Sales Charge (%)",                    "_sales_charge"),
    ("MS Matched Name (Debug)",                  "_debug_name"), # X-RAY COLUMN
]

HEADERS = [h for h, _ in OUTPUT_COLUMNS]

def _fetch_universe_xml(timeout: int = 90) -> ET.Element:
    url = f"{BASE_URL}/{CLIENT_ID}/universeid/{UNIVERSE_ID}"
    resp = requests.get(url, params={"accesscode": ACCESS_CODE}, timeout=timeout)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    return root

def _parse_fund(el: ET.Element) -> dict:
    def g(tag):
        e = el.find(f".//{tag}")
        return e.text.strip() if e is not None and e.text else None
    return {
        "id": el.get("_id"), "isin": g("DP-ISIN"), "name": g("DP-FundName"),
        "currency": g("DP-Currency"), "category": g("DP-CategoryName"),
        "nav_date": g("DP-DayEndDate"), "return_ytd": g("DP-ReturnYTD"),
        "return_1w": g("DP-Return1Week"), "return_1m": g("DP-Return1Mth"),
        "return_3m": g("DP-Return3Mth"), "return_6m": g("DP-Return6Mth"),
        "return_1y": g("DP-Return1Yr"), "return_2y": g("DP-Return2Yr"),
        "return_3y": g("DP-Return3Yr"), "return_5y": g("DP-Return5Yr"),
        "return_10y": g("DP-Return10Yr"),
    }

def load_universe() -> list[dict]:
    global _UNIVERSE_MEM_CACHE, _UNIVERSE_CACHE_DATE
    if _UNIVERSE_CACHE_DATE == date.today() and _UNIVERSE_MEM_CACHE is not None:
        return _UNIVERSE_MEM_CACHE

    root = _fetch_universe_xml()
    funds = [_parse_fund(el) for el in root.iter("data")]
    _UNIVERSE_MEM_CACHE = funds
    _UNIVERSE_CACHE_DATE = date.today()
    return funds

def fetch_risk_measures(isin: str = None, mstar_id: str = None, timeout: int = 15) -> dict:
    if isin and isin.strip():
        id_type, identifier = "isin", isin.strip().upper()
    elif mstar_id and mstar_id.strip():
        id_type, identifier = "mstarid", mstar_id.strip()
    else:
        return {"sharpe_3y": None, "stddev_3y": None}

    url = f"{BASE_URL}/RiskMeasure/{id_type}/{identifier}"
    try:
        resp = requests.get(url, params={"accesscode": ACCESS_CODE}, timeout=timeout)
        root = ET.fromstring(resp.text)
        api_el = root.find(".//api")
        if api_el is None: return {"sharpe_3y": None, "stddev_3y": None}
        def f(tag):
            e = api_el.find(tag)
            try: return float(e.text.strip()) if e is not None and e.text else None
            except: return None
        return {"sharpe_3y": f("SharpeRatio3Yr"), "stddev_3y": f("StdDev3Yr")}
    except Exception:
        return {"sharpe_3y": None, "stddev_3y": None}

# ==========================================
# UPGRADED STRICT MATCHING ENGINE
# ==========================================
def match_fund(query_name: str, query_currency: str, universe: list) -> dict | None:
    q_name = query_name.lower().strip()
    q_curr = query_currency.upper().strip() if query_currency else ""
    if not q_name: return None

    # Apply manual Alias if it exists
    if q_name in ALIAS_MAP:
        q_name = ALIAS_MAP[q_name]

    # Filter by Currency to prevent Share Class collisions
    search_pool = universe
    if q_curr:
        curr_pool = [f for f in universe if str(f.get("currency", "")).upper() == q_curr]
        if curr_pool: 
            search_pool = curr_pool

    name_index = {str(f.get("name", "")).lower().strip(): f for f in search_pool if f.get("name")}

    # 1. Exact Match
    if q_name in name_index:
        return name_index[q_name]

    # 2. Simple Substring Match (e.g. "Fund A" is inside "Fund A MYR Class")
    for fname, fund in name_index.items():
        if q_name in fname or fname in q_name:
            return fund

    # 3. Smart Fuzzy Match (Tightened to 85% to stop duplicates)
    closest = difflib.get_close_matches(q_name, name_index.keys(), n=1, cutoff=0.85)
    if closest:
        return name_index[closest[0]]

    # 4. Strict Token Match
    q_tokens = set(q_name.replace('-', ' ').split())
    best_fund, best_score = None, 0
    for fname, fund in name_index.items():
        f_tokens = set(fname.replace('-', ' ').split())
        score = len(q_tokens & f_tokens)
        # Require 80% of the words to match to avoid merging different share classes
        if score > best_score and score >= len(q_tokens) * 0.8:
            best_fund, best_score = fund, score

    return best_fund

def process_funds_csv(file_bytes: bytes, skip_risk: bool = False) -> bytes:
    def safe_read_csv(skip_rows):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), header=skip_rows, dtype=str, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(file_bytes), header=skip_rows, dtype=str, encoding='latin1')

    # Handle UWEALTH title row format
    df = safe_read_csv(0)
    cols = [str(c).strip() for c in df.columns]
    
    if "Fund Name" not in cols:
        df = safe_read_csv(1)
        cols = [str(c).strip() for c in df.columns]
    
    df.columns = cols
    unnamed = [c for c in df.columns if c.startswith("Unnamed:") or c == ""]
    df.drop(columns=unnamed, inplace=True, errors="ignore")
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = df[df["Fund Name"].notna() & (df["Fund Name"] != "")].reset_index(drop=True)

    universe = load_universe()
    results = []
    
    for _, row in df.iterrows():
        fund_name = str(row.get("Fund Name", "") or "").strip()
        fund_currency = str(row.get("Fund Currency", "") or "").strip()
        sales_charge = str(row.get("Fund Sales Charge (%)", "") or "").strip() or None
        
        fund = match_fund(fund_name, fund_currency, universe)

        if fund is None:
            row_out = {h: None for h in HEADERS}
            row_out["Fund Name"] = fund_name
            row_out["Fund Sales Charge (%)"] = sales_charge
            row_out["MS Matched Name (Debug)"] = "NOT FOUND"
            results.append(row_out)
            continue

        risk = {"sharpe_3y": None, "stddev_3y": None}
        if not skip_risk:
            risk = fetch_risk_measures(isin=fund.get("isin"), mstar_id=fund.get("id"))
            time.sleep(0.25)

        row_out = {}
        for header, key in OUTPUT_COLUMNS:
            if key == "_sales_charge": row_out[header] = sales_charge
            elif key == "_debug_name": row_out[header] = fund.get("name")
            elif key in ("sharpe_3y", "stddev_3y"): row_out[header] = risk.get(key)
            else: row_out[header] = fund.get(key)
        results.append(row_out)

    out_df = pd.DataFrame(results, columns=HEADERS)
    output_stream = io.StringIO()
    out_df.to_csv(output_stream, index=False)
    
    return output_stream.getvalue().encode('utf-8-sig')
