# @title
# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
from matplotlib.ticker import FixedLocator, FixedFormatter
import yfinance as yf
from yahooquery import Ticker as YQTicker
from matplotlib.ticker import FixedLocator, NullFormatter
import textwrap
import string

# ===================== UOB Kay Hian visual theme =====================
UOBKH_NAVY = "#0F2A47" # primary navy for titles/lines
UOBKH_RED = "#C33832" # primary deep red accent
UOBKH_GREEN = "#137D5B" # accent green for positive/KO/up-trend
UOBKH_TEXT = "#243042" # primary text
UOBKH_SLATE = "#2D3947" # dark UI strokes
UOBKH_GRID = "#D7DEE8" # grid lines / table borders
UOBKH_BG = "#FFFFFF" # panels
UOBKH_SOFT_BG = "#F7F9FC" # light panel background
UOBKH_ZEBRA = "#F3F6FA" # table zebra fill
UOBKH_MUTED_GREY= "#6B7280" # small captions / footers
UOBKH_STEEL = "#1A3F66"
UOBKH_NAVY_BRIGHT = "#133861"

# Apply global typography & colors
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "Liberation Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,    # optional, avoids minus sign issues
    "axes.facecolor": UOBKH_BG,
    "figure.facecolor": UOBKH_BG,
    "axes.edgecolor": UOBKH_SLATE,
    "axes.labelcolor": UOBKH_TEXT,
    "text.color": UOBKH_TEXT,
    "xtick.color": UOBKH_TEXT,
    "ytick.color": UOBKH_TEXT,
    "grid.color": UOBKH_GRID,
})

# ----------------------------- tunables -----------------------------
BAR_GAP = 0.8 # distance between month bars
BAR_WIDTH = 0.48 # bar thickness per month
SHOW_SEGMENT_PCT = 0.10 # in-segment number shown if segment/total >= 10%

# ----------------------------- helpers -----------------------------
def _unwrap(v): return v.get("raw", v.get("fmt")) if isinstance(v, dict) else v

def _fmt_num(n, d=2):
    if n is None or (isinstance(n, float) and pd.isna(n)): return ""
    try: return f"{float(n):.{d}f}"
    except: return str(n)

def _fmt_pct(x, d=2):
    if x is None or (isinstance(x, float) and pd.isna(x)): return ""
    try:
        x = float(x)
        if -1.5 <= x <= 1.5: x *= 100.0
        return f"{x:.{d}f}%"
    except: return str(x)

def _fmt_money_compact(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return ""
    try:
        n = float(x); a = abs(n)
        if a >= 1_000_000_000_000: return f"{n/1_000_000_000_000:.2f}T"
        if a >= 1_000_000_000: return f"{n/1_000_000_000:.2f}B"
        if a >= 1_000_000: return f"{n/1_000_000:.2f}M"
        if a >= 1_000: return f"{n/1_000:.2f}K"
        return f"{n:.2f}"
    except: return str(x)

def _fmt_date_only(s):
    if s is None:
        return ""
    try:
        if pd.isna(s):
            return ""
    except Exception:
        pass
    if isinstance(s, str) and s.strip().lower() in ("", "nat", "nan"):
        return ""
    if isinstance(s, (list, tuple)):
        clean = [_fmt_date_only(_unwrap(x)) for x in s if x is not None]
        clean = [c for c in clean if c]
        seen = set(); uniq = []
        for c in clean:
            if c not in seen:
                seen.add(c); uniq.append(c)
        return " – ".join(uniq[:2]) if uniq else ""
    try:
        if isinstance(s, (int, float)) and not pd.isna(s):
            return pd.to_datetime(int(s), unit="s", utc=True).date().isoformat()
        ss = str(s).strip()
        if ss.isdigit() and len(ss) >= 10:
            return pd.to_datetime(int(ss), unit="s", utc=True).date().isoformat()
    except Exception:
        pass
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(dt):
            return ""
        return dt.date().isoformat()
    except Exception:
        return ""

def _pick(df,*names):
    lower={c.lower():c for c in df.columns}
    for n in names:
        c=lower.get(n.lower())
        if c: return c
    return None

def _yq_val(d, *keys):
    if not isinstance(d, dict):
        return None
    for k in keys:
        cur = d
        ok = True
        for part in k.split("."):
            found = None
            if isinstance(cur, dict):
                for kk, vv in cur.items():
                    if kk.lower() == part.lower():
                        found = vv
                        break
            if found is None:
                ok = False
                break
            cur = found
        if ok:
            return cur
    return None

def _fmt_earnings_date(ts) -> str:
    v = _unwrap(ts)
    if v is None or v == "":
        return ""
    if isinstance(v, (list, tuple)) and v:
        v = _unwrap(v[0])
    try:
        if isinstance(v, (int, float)):
            return datetime.utcfromtimestamp(int(float(v))).strftime("%Y-%m-%d")
        if isinstance(v, str) and v.isdigit():
            return datetime.utcfromtimestamp(int(float(v))).strftime("%Y-%m-%d")
    except Exception:
        pass
    if isinstance(v, str):
        m = re.search(r"\d{4}-\d{2}-\d{2}", v)
        return m.group(0) if m else ""
    try:
        dt = pd.to_datetime(v, errors="coerce", utc=True)
        if pd.isna(dt):
            return ""
        return dt.date().isoformat()
    except Exception:
        return ""

def _quote_currency(sym: str) -> str:
    """Best-effort quote currency for a ticker (USD, HKD, etc.)."""
    # 1) Try yfinance.fast_info
    try:
        fi = dict(yf.Ticker(sym).fast_info)
        c = fi.get("currency")
        if c:
            return str(c)
    except:
        pass
    # 2) Try yfinance.info / get_info
    try:
        yt = yf.Ticker(sym)
        try:
            info = yt.get_info()
        except:
            info = getattr(yt, "info", {}) or {}
        c = info.get("currency") or info.get("financialCurrency") or info.get("currencySymbol")
        if c:
            return str(c)
    except:
        pass
    # 3) Try yahooquery
    try:
        yq = YQTicker(sym)
        price = _yq_mod(yq, "price")
        c = _unwrap(price.get("currency")) or _unwrap(price.get("financialCurrency")) or _unwrap(price.get("currencySymbol"))
        if c:
            return str(c)
    except:
        pass
    return ""  # unknown

def _is_us_equity(sym: str) -> bool:
    # Try exchange codes from yfinance
    try:
        fi = dict(yf.Ticker(sym).fast_info)
        ex = str(fi.get("exchange") or fi.get("market") or "").upper()
        if ex in {"NMS","NYQ","NCM","NGM","ASE","NYS","BATS","ARCA","PCX"}:
            return True
    except:
        pass
    # Fallback: info.fullExchangeName
    try:
        info = yf.Ticker(sym).get_info()
        exfull = str(info.get("fullExchangeName") or info.get("exchange") or "").upper()
        if any(tag in exfull for tag in ("NASDAQ","NYSE","AMEX","BATS","ARCA")):
            return True
    except:
        pass
    # Last fallback: no suffix → likely US
    return "." not in sym

def _is_hk_sg_jp(sym: str) -> bool:
    s = sym.upper()
    return s.endswith(".HK") or s.endswith(".SI") or s.endswith(".T")

def _latest_hist_close(sym: str, lookback_days: int = 14):
    """
    Return the most recent non-NaN daily Close from yfinance history.
    Falls back to fast_info previousClose if history unavailable.
    """
    yt = yf.Ticker(sym)
    try:
        hist = yt.history(period=f"{lookback_days}d", interval="1d", auto_adjust=False)
        if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist.columns:
            close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
            if not close.empty:
                return float(close.iloc[-1])
    except Exception:
        pass
    # Fallbacks if history failed/empty
    try:
        fi = dict(yt.fast_info)
        val = fi.get("previousClose") or fi.get("lastPrice") or fi.get("regularMarketPreviousClose")
        return float(val) if val is not None else None
    except Exception:
        return None

# Provide a safe fallback used by get_snapshot/get_widgets when needed
def _hist_prev_last(sym: str, period: str = "60d"):
    """Return (previous_close, last_close) from history for robust fallbacks."""
    try:
        h = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=False)
        s = pd.to_numeric(h.get("Close"), errors="coerce").dropna()
        if len(s) == 0:
            return (None, None)
        last = float(s.iloc[-1])
        prev = float(s.iloc[-2]) if len(s) >= 2 else last
        return (prev, last)
    except Exception:
        return (None, None)

# --- Title case where only "of" stays lowercase (mid-phrase); preserve acronyms like KO/KI/FCN ---
def smart_title_of_only(s: str) -> str:
    if not s:
        return ""
    parts = re.split(r'(\s+|[-/])', s.strip())  # keep spaces, hyphens, slashes
    word_idx = [i for i,p in enumerate(parts) if not re.fullmatch(r'\s+|[-/]', p)]
    if not word_idx:
        return s

    first_i, last_i = word_idx[0], word_idx[-1]
    out = []
    for i, tok in enumerate(parts):
        if re.fullmatch(r'\s+|[-/]', tok):
            out.append(tok); continue
        if tok.isupper():  # KO, KI, FCN, etc.
            out.append(tok); continue

        low = tok.lower()
        if i not in (first_i, last_i) and low == "of":
            out.append("of")
        else:
            out.append(low.capitalize())
    return "".join(out)

# ----------------------------- SGT cutoff helpers -----------------------------
def _target_asia_close_date_sgt():
    """
    SGT target date for .HK/.SI/.T:
      - Mon < 16:00  -> last Fri
      - Mon ≥ 16:00  -> Mon (today)
      - Tue < 16:00  -> Mon
      - General < 16:00 -> previous calendar day
      - General ≥ 16:00 -> today
    """
    now = _now_sgt()
    today = now.date()

    if now.time() >= dtime(16, 0):
        return today

    # Before 16:00 SGT
    if today.weekday() == 0:  # Monday
        return today - timedelta(days=3)  # Friday
    else:
        return today - timedelta(days=1)  # e.g., Tue->Mon

def _close_on_or_before(sym: str, target_date, lookback_days: int = 35):
    """
    Close *on* target_date if present, else the last available close
    ON OR BEFORE target_date. Returns None if not found.
    """
    try:
        h = yf.Ticker(sym).history(period=f"{lookback_days}d", interval="1d", auto_adjust=False)
        s = pd.to_numeric(h.get("Close"), errors="coerce").dropna()
        if s.empty:
            return None
        idx_dates = np.array(pd.to_datetime(s.index).date)
        pos = np.where(idx_dates <= target_date)[0]
        if len(pos) == 0:
            return None
        return float(s.iloc[pos[-1]])
    except Exception:
        return None

from datetime import time as dtime
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

def _now_sgt() -> datetime:
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo("Asia/Singapore"))
    # Fallback: fixed +8 (SGT)
    return datetime.utcnow() + timedelta(hours=8)

def _hk_sg_jp_close_by_sgt(sym: str):
    """
    Exact-date SGT logic for .HK/.SI/.T:
      - Mon < 16:00  -> Friday’s close
      - Mon ≥ 16:00  -> Monday’s close (if posted; fallback handled)
      - Tue < 16:00  -> Monday’s close
      - Otherwise: before 16:00 -> previous trading day; after 16:00 -> today
    Fetches the close for the target date; if missing, falls back to the
    most recent close ON OR BEFORE that date (handles holidays / Yahoo lag).
    """
    target_date = _target_asia_close_date_sgt()

    # Primary: exact (or on/before) target-date close
    px = _close_on_or_before(sym, target_date, lookback_days=60)
    if px is not None:
        return px

    # Safety fallback: use latest history; try to honor after/before 4pm where possible
    try:
        h = yf.Ticker(sym).history(period="60d", interval="1d", auto_adjust=False)
        s = pd.to_numeric(h.get("Close"), errors="coerce").dropna()
        if not s.empty:
            last = float(s.iloc[-1])
            # If it's after 4pm SGT and last is today, return it; else return the most recent anyway
            last_date = pd.to_datetime(s.index[-1]).date()
            now = _now_sgt()
            if now.time() >= dtime(16, 0) and last_date == now.date():
                return last
            # Otherwise, previous trading day's close if available
            return float(s.iloc[-2]) if len(s) >= 2 else last
    except Exception:
        pass

    # Final fallback: give up gracefully
    return None

# ----------------------------- data fetchers -----------------------------
def _yq_mod(tk,name):
    try: obj=getattr(tk,name)
    except: return {}
    if isinstance(obj,dict) and obj:
        sym=next(iter(obj.keys()))
        inner=obj.get(sym,{})
        return inner if isinstance(inner,dict) else {}
    return {}

def _earnings_date(yq):
    cale = _yq_mod(yq, "calendar_events")
    cal_date = _yq_val(cale, "earnings.earningsDate", "earningsDate")
    ds = _fmt_earnings_date(cal_date)
    if ds:
        return ds
    price = _yq_mod(yq, "price")
    single = _fmt_earnings_date(_yq_val(price, "earningsTimestamp"))
    start = _fmt_earnings_date(_yq_val(price, "earningsTimestampStart"))
    end = _fmt_earnings_date(_yq_mod(yq, "price").get("earningsTimestampEnd"))
    if start or end:
        return " – ".join([p for p in [start, end] if p])
    if single:
        return single
    return ""

def _52w_yf_fast(yt):
    try:
        fi=dict(yt.fast_info); return float(fi.get("yearHigh")), float(fi.get("yearLow"))
    except: return (None,None)

def _52w_yq(price, ks):
    hi=_unwrap(price.get("fiftyTwoWeekHigh")) or _unwrap(ks.get("52WeekHigh"))
    lo=_unwrap(price.get("fiftyTwoWeekLow")) or _unwrap(ks.get("52WeekLow"))
    return (float(hi) if hi is not None else None, float(lo) if lo is not None else None)

def _52w_from_hist(yt):
    try:
        hist=yt.history(period="400d",interval="1d")
        if isinstance(hist,pd.DataFrame) and not hist.empty:
            last=hist.tail(252)
            return float(last["High"].max()), float(last["Low"].min())
    except: pass
    return (None,None)

def _eps_growth_yoy(yq, sym):
    """
    Most common YoY:
      - Quarterly YoY: EPS_last_quarter / EPS_same_quarter_prev_year - 1  (needs >= 5 quarters)
      - Fallback: Annual YoY using annual statements (needs >= 2 years)
    """
    # Try quarterly first
    try:
        q = yq.income_statement(frequency="q")
        if isinstance(q, pd.DataFrame) and not q.empty:
            if "symbol" in q.columns:
                q = q[q["symbol"] == sym]
            cE = _pick(q, "dilutedEPS", "basicEPS")
            cD = _pick(q, "asOfDate")
            if cE and cD:
                q = q[[cD, cE]].dropna()
                q[cD] = pd.to_datetime(q[cD], utc=True, errors="coerce")
                q = q.sort_values(cD)
                eps = pd.to_numeric(q[cE], errors="coerce").dropna()
                if len(eps) >= 5:
                    cur = float(eps.iloc[-1])
                    prv = float(eps.iloc[-5])  # same quarter last year
                    if prv != 0:
                        return cur / prv - 1
    except Exception:
        pass

    # Fallback: annual YoY
    try:
        a = yq.income_statement(frequency="a")
        if isinstance(a, pd.DataFrame) and not a.empty:
            if "symbol" in a.columns:
                a = a[a["symbol"] == sym]
            cE = _pick(a, "dilutedEPS", "basicEPS")
            cD = _pick(a, "asOfDate")
            if cE and cD:
                a = a[[cD, cE]].dropna()
                a[cD] = pd.to_datetime(a[cD], utc=True, errors="coerce")
                a = a.sort_values(cD)
                s = pd.to_numeric(a[cE], errors="coerce").dropna()
                if len(s) >= 2:
                    prv = float(s.iloc[-2])
                    if prv != 0:
                        return float(s.iloc[-1]) / prv - 1
    except Exception:
        pass

    return None

def get_snapshot(sym):
    yq = YQTicker(sym); price = _yq_mod(yq, "price"); summ = _yq_mod(yq, "summary_detail"); ks = _yq_mod(yq, "key_stats")
    yt = yf.Ticker(sym)
    try: info = yt.get_info()
    except:
        try: info = yt.info
        except: info = {}

    # --- Previous Close display with SGT cutoff for HK/SG/JP ---
    if _is_hk_sg_jp(sym):
        prev_display = _hk_sg_jp_close_by_sgt(sym)
    else:
        try:
            fi = dict(yt.fast_info)
            cur_price = fi.get("lastPrice") or fi.get("regularMarketPrice")
        except:
            cur_price = None
        prev = (_unwrap(price.get("regularMarketPreviousClose"))
                or _unwrap(summ.get("previousClose"))
                or info.get("previousClose"))
        prev_display = cur_price if (_is_us_equity(sym) and cur_price is not None) else prev
        if prev_display is None:
            # fallback to last historical close
            _, prev_display = _hist_prev_last(sym)

    mcap = _unwrap(price.get("marketCap")) or _unwrap(summ.get("marketCap")) or info.get("marketCap")
    pe   = _unwrap(summ.get("trailingPE")) or _unwrap(ks.get("trailingPE"))
    ps   = _unwrap(summ.get("priceToSalesTrailing12Months"))
    pb   = _unwrap(summ.get("priceToBook")) or _unwrap(ks.get("priceToBook"))
    rate = _unwrap(summ.get("dividendRate")) or _unwrap(summ.get("trailingAnnualDividendRate"))
    yld  = _unwrap(summ.get("dividendYield")) or _unwrap(summ.get("trailingAnnualDividendYield"))
    fwddiv = f"{_fmt_num(rate,2)} ({_fmt_pct(yld)})" if (rate is not None and yld is not None) else (
             _fmt_num(rate,2) if rate is not None else (_fmt_pct(yld) if yld is not None else ""))
    earn = _earnings_date(yq)
    epsg = _eps_growth_yoy(yq, sym)
    epsg_disp = _fmt_pct(epsg) if epsg is not None else ""

    hi, lo = _52w_yf_fast(yt)
    if hi is None or lo is None:
        a_hi, a_lo = _52w_yq(price, ks); hi = hi if hi is not None else a_hi; lo = lo if lo is not None else a_lo
    if hi is None or lo is None:
        c_hi, c_lo = _52w_from_hist(yt); hi = hi if hi is not None else c_hi; lo = lo if lo is not None else c_lo

    return {
        "Previous Close": _fmt_num(prev_display, 2),
        "Market Cap": _fmt_money_compact(mcap),
        "PE (TTM)": _fmt_num(pe, 2),
        "Earnings Date": earn,
        "Fwd Div & Yield": fwddiv,
        "EPS Growth YoY": epsg_disp,
        "P/S (TTM)": _fmt_num(ps, 2),
        "P/B (MRQ)": _fmt_num(pb, 2),
        "52W High": _fmt_num(hi, 2),
        "52W Low": _fmt_num(lo, 2),
    }

def get_widgets(sym):
    yq=YQTicker(sym)
    rec=yq.recommendation_trend
    if isinstance(rec, pd.DataFrame) and not rec.empty:
        if "symbol" in rec.columns:
            rec = rec[rec["symbol"].str.upper()==sym.upper()]
        elif rec.index.names and "symbol" in rec.index.names:
            rec = rec.reset_index()
            rec = rec[rec["symbol"].str.upper()==sym.upper()]
        cols={}
        for k in ["period","strongBuy","buy","hold","sell","strongSell"]:
            c=_pick(rec,k)
            if c: cols[c]=k
        rec=rec.rename(columns=cols)
        keep=[c for c in ["period","strongBuy","buy","hold","sell","strongSell"] if c in rec.columns]
        rec=rec[keep].reset_index(drop=True)
    else:
        rec=pd.DataFrame(columns=["period","strongBuy","buy","hold","sell","strongSell"])

    low=avg=high=num=None
    try: pt=yq.price_target
    except: pt=None
    if isinstance(pt,pd.DataFrame) and not pt.empty:
        if "symbol" in pt.columns:
            pt=pt[pt["symbol"].str.upper()==sym.upper()]
        elif pt.index.names and "symbol" in pt.index.names:
            pt=pt.reset_index()
            pt=pt[pt["symbol"].str.upper()==sym.upper()]
        cL=_pick(pt,"low","targetLowPrice"); cA=_pick(pt,"mean","average","avg","targetMeanPrice")
        cH=_pick(pt,"high","targetHighPrice"); cN=_pick(pt,"numberOfAnalysts","analystCount")
        row=pt.iloc[0]
        low=float(row[cL]) if cL else None; avg=float(row[cA]) if cA else None
        high=float(row[cH]) if cH else None; num=int(row[cN]) if cN else None
    else:
        yt=yf.Ticker(sym)
        try: info=yt.get_info()
        except: info=getattr(yt,"info",{}) or {}
        low=info.get("targetLowPrice"); avg=info.get("targetMeanPrice"); high=info.get("targetHighPrice")
        num=info.get("numberOfAnalystOpinions") or info.get("numberOfAnalysts")
        low=float(low) if low is not None else None; avg=float(avg) if avg is not None else None
        high=float(high) if high is not None else None; num=int(num) if num is not None else None

    # ---- SGT cutoff for HK/SG/JP "Current" (targets dot) ----
    if _is_hk_sg_jp(sym):
        cur = _hk_sg_jp_close_by_sgt(sym)
    else:
        try:
            fi=dict(yf.Ticker(sym).fast_info)
            cur=fi.get("lastPrice") or fi.get("regularMarketPrice")
        except:
            cur=None
        if cur is None:
            _, cur = _hist_prev_last(sym)

    return rec, {"low":low,"average":avg,"high":high,"current":(float(cur) if cur is not None else None),"numberOfAnalysts":num}

def long_name(sym):
    try:
        info=yf.Ticker(sym).info
        return info.get("longName", sym.split('.')[0] if '.' in sym else sym)
    except:
        return sym.split('.')[0] if '.' in sym else sym

# ----------------------------- stock chart -----------------------------
def _draw_year_labels_once(ax, x_dates):
    if len(x_dates) == 0: return
    years = sorted({d.year for d in x_dates})
    for y in years:
        xs = [d for d in x_dates if d.year == y]
        if not xs: continue
        x_at = xs[0]
        blend = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(x_at, -0.15, f"{y}", transform=blend, ha="center", va="top",
                fontsize=11, fontweight="normal", clip_on=False, color=UOBKH_TEXT)

def _side_label(ax, y_val, label, value, color=UOBKH_TEXT, num_size=11):
    blend = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    txt = f"{value:.2f}"
    ax.text(1.01, y_val, txt, transform=blend, va="center", ha="left",
            fontsize=num_size, color=color)

def _compute_plotly_like_month_ticks(start_dt, end_dt):
    start_month = pd.Timestamp(start_dt).to_period("M")
    end_month = pd.Timestamp(end_dt).to_period("M")
    months = pd.date_range(start=start_month.start_time, end=end_month.start_time, freq="MS")
    num_months = len(months)
    if num_months <= 5:
        dtick = "M1"; tick0 = months[0] if len(months) else pd.Timestamp(start_dt).normalize()
    else:
        dtick = "M2"; parity = end_month.month % 2; tick0 = None
        for m in months:
            if m.month % 2 == parity:
                tick0 = m; break
        if tick0 is None:
            tick0 = months[0] if len(months) else pd.Timestamp(start_dt).normalize()
    freq = "MS" if dtick == "M1" else "2MS"
    tickvals = pd.date_range(start=tick0, end=end_dt, freq=freq)
    tickvals = tickvals[(tickvals >= pd.Timestamp(start_dt)) & (tickvals <= pd.Timestamp(end_dt))]
    target = pd.Timestamp(datetime.now()) - pd.DateOffset(months=10)
    target_year = target.year; target_month = target.month
    prev_year_ticks = [d for d in tickvals if d.year == target_year]
    closest_tick = min(prev_year_ticks, key=lambda d: abs(d.month - target_month)) if prev_year_ticks else None
    labels = []
    displayed_years = set()
    for d in tickvals:
        month_label = d.strftime("%b")
        add_year = (d.year not in displayed_years) or (closest_tick is not None and d == closest_tick)
        label = f"{month_label}\n{d.year}" if add_year else month_label
        if add_year: displayed_years.add(d.year)
        labels.append(label)
    return list(tickvals.to_pydatetime()), labels

def draw_stock_chart(ax, symbol, strike, ko, ki, start, end):
    hist = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if hist.empty:
        ax.text(0.02, 0.5, "No price data.", transform=ax.transAxes, va="center", fontsize=12, color=UOBKH_RED)
        ax.axis("off"); return

    s = hist["Close"].dropna()

    # --- NEW: enforce SGT 16:00 cutoff on newest plotted point for HK/SG/JP ---
    if _is_hk_sg_jp(symbol) and not s.empty:
        pn = _hk_sg_jp_close_by_sgt(symbol)  # yesterday's close before 16:00 SGT, today's after
        if pn is not None:
            try:
                s.iloc[-1] = float(pn)  # force last data point to SGT-cutoff close
            except Exception:
                pass

    # rebuild arrays after potential tweak
    x = np.asarray(s.index.to_pydatetime())
    y = np.asarray(s.astype(float).values, dtype=float).ravel()

    # --- Trend line (baseline) ---
    line_c = UOBKH_MUTED_GREY
    ax.plot(x, y, color=line_c, lw=1.7)

    y_min = float(np.nanmin(y)); y_max = float(np.nanmax(y))
    rng   = max(1e-9, y_max - y_min)
    baseline_val = y_min - rng*0.02

    # NEWEST PRICE POINT: always use latest historical close (affects HK/SG/JP request)
    price_now = float(y[-1])

    # Reference levels (guard for None)
    strike_y = price_now * strike / 100.0
    ko_y     = (price_now * ko / 100.0) if (ko is not None) else None
    ki_y     = (price_now * ki / 100.0) if (ki is not None) else None

    # --- Reference lines ---
    ax.axhline(strike_y, color=UOBKH_NAVY,  lw=1.6)
    if ko_y is not None:
        ax.axhline(ko_y,     color=UOBKH_GREEN, lw=1.6)
    if ki_y is not None:
        ax.axhline(ki_y, color=UOBKH_RED,   lw=1.7)

    # --- Side labels ---
    _side_label(ax, strike_y, "Strike", strike_y, color=UOBKH_NAVY)
    if ko_y is not None:
        _side_label(ax, ko_y,     "KO",     ko_y,     color=UOBKH_GREEN)
    if ki_y is not None:
        _side_label(ax, ki_y, "KI", ki_y, color=UOBKH_RED)

    # --- Last strike-cross annotation (kept) ---
    sig = np.sign(y - strike_y)
    cross_idx = np.where(np.diff(sig) != 0)[0]
    if len(cross_idx) > 0:
        i = int(cross_idx[-1])
        ax.annotate(f"{x[i].strftime('%d/%m/%y')}\n{y[i]:.2f}", xy=(x[i], y[i]),
                    xytext=(0, +28), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, color=UOBKH_TEXT,
                    bbox=dict(boxstyle="round,pad=0.35", fc=UOBKH_BG, ec=UOBKH_GRID, lw=1, alpha=0.7),
                    arrowprops=dict(arrowstyle="-|>", color=UOBKH_SLATE, lw=1.2))

    # ----- X-axis ticks -----
    start_dt = x[0]; end_dt = x[-1]
    tick_dt_list, tick_labels = _compute_plotly_like_month_ticks(start_dt, end_dt)
    tick_nums = mdates.date2num(tick_dt_list)
    ax.xaxis.set_major_locator(FixedLocator(tick_nums))
    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="x", which="major", labelsize=11)
    for lbl in ax.get_xticklabels(): lbl.set_visible(False)
    ax.set_xlim(start_dt, end_dt)
    blend = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    month_y = -0.065; year_y = -0.15
    for xd, lbl in zip(tick_dt_list, tick_labels):
        parts = lbl.split("\n", 1)
        ax.text(xd, month_y, parts[0], transform=blend, ha="center", va="top",
                fontsize=11, color=UOBKH_TEXT, clip_on=False)
        if len(parts) == 2 and parts[1]:
            ax.text(xd, year_y, parts[1], transform=blend, ha="center", va="top",
                    fontsize=7, color=UOBKH_MUTED_GREY, clip_on=False)

    # --- y-range ---
    ymins = [baseline_val, strike_y]
    ymaxs = [float(np.nanmax(y)), strike_y]
    if ko_y is not None:
        ymins.append(ko_y); ymaxs.append(ko_y)
    if ki_y is not None:
        ymins.append(ki_y); ymaxs.append(ki_y)
    ymin = min(ymins)
    ymax = max(ymaxs)
    pad = (ymax - ymin) * 0.07 if (ymax - ymin) > 0 else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8)

    # --- legend (hide KO when NA) ---
    legend_elements = [
        Line2D([0], [0], color=UOBKH_MUTED_GREY, lw=2, label='Price'),
        Line2D([0], [0], color=UOBKH_NAVY,       lw=2, label='Strike'),
    ]
    if ko_y is not None:
        legend_elements.append(Line2D([0], [0], color=UOBKH_GREEN,      lw=2, label='KO'))
    if ki_y is not None:
        legend_elements.append(Line2D([0], [0], color=UOBKH_RED,        lw=2, label='KI'))

    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.15), ncol=max(1, len(legend_elements)),
              frameon=False, fontsize=10)

# ----------------------------- kv table -----------------------------
def draw_kv_table(ax, rows, one_col=False,
                  header=None, header_bg=None, header_color=None,
                  row_scale=1.0,
                  label_size=11, value_size=11, header_size=13):
    ax.axis("off")
    left = 0.02; right = 0.98; top = 0.99; bottom = 0.01

    if one_col:
        left = 0.1; right = 0.95; top = 0.995; bottom = 0.005
        n = len(rows)
        row_h = (top - bottom) / n * row_scale
        for i, (lab, val) in enumerate(rows):
            y1 = top - i * row_h; y0 = y1 - row_h
            if i % 2 == 0:
                ax.add_patch(matplotlib.patches.Rectangle(
                    (left, y0), right - left, row_h,
                    transform=ax.transAxes, color=UOBKH_ZEBRA, zorder=0, clip_on=False
                ))
            ax.plot([left, right], [y0, y0], color=UOBKH_GRID, lw=0.8,
                    transform=ax.transAxes, clip_on=False)
            ax.text(left + 0.01, (y0 + y1) / 2, lab, transform=ax.transAxes,
                    va="center", fontsize=label_size, fontweight="bold", color=UOBKH_NAVY)
            ax.text(right - 0.01, (y0 + y1) / 2, val, transform=ax.transAxes,
                    va="center", ha="right", fontsize=value_size, color=UOBKH_TEXT)
        return

    col_w = (right - left) / 2
    n = len(rows) // 2
    total_n = n + 1 if header else n
    row_h = (top - bottom) / total_n * row_scale
    current_y = top

    if header:
        y1 = current_y; y0 = y1 - row_h
        ax.add_patch(matplotlib.patches.Rectangle(
            (left, y0), right - left, row_h,
            transform=ax.transAxes, color=header_bg, zorder=0, clip_on=False
        ))
        ax.plot([left, right], [y0, y0], color=UOBKH_GRID, lw=0.8,
                transform=ax.transAxes, clip_on=False)
        ax.plot([left, right], [y1, y1], color=UOBKH_GRID, lw=0.8,
                transform=ax.transAxes, clip_on=False)
        ax.text((left + right)/2, (y0 + y1)/2, header, transform=ax.transAxes,
                va="center", ha="center", fontsize=header_size, fontweight="bold",
                color=header_color)
        current_y = y0

    for i in range(n):
        y1 = current_y; y0 = y1 - row_h
        current_y = y0
        if i % 2 == 0:
            ax.add_patch(matplotlib.patches.Rectangle(
                (left, y0), right - left, row_h,
                transform=ax.transAxes, color=UOBKH_ZEBRA, zorder=0, clip_on=False
            ))
        for x in [left, left + col_w, right]:
            ax.plot([x, x], [y0, y1], color=UOBKH_GRID, lw=0.8,
                    transform=ax.transAxes, clip_on=False)
        ax.plot([left, right], [y0, y0], color=UOBKH_GRID, lw=0.8,
                transform=ax.transAxes, clip_on=False)
        l1, v1 = rows[2 * i]; l2, v2 = rows[2 * i + 1]
        ax.text(left + 0.012, (y0 + y1) / 2, l1, transform=ax.transAxes,
                va="center", fontsize=label_size, fontweight="bold", color=UOBKH_NAVY)
        ax.text(left + col_w - 0.012, (y0 + y1) / 2, v1, transform=ax.transAxes,
                va="center", ha="right", fontsize=value_size, color=UOBKH_TEXT)
        ax.text(left + col_w + 0.012, (y0 + y1) / 2, l2, transform=ax.transAxes,
                va="center", fontsize=label_size, fontweight="bold", color=UOBKH_NAVY)
        ax.text(right - 0.012, (y0 + y1) / 2, v2, transform=ax.transAxes,
                va="center", ha="right", fontsize=value_size, color=UOBKH_TEXT)

# --------- robust month parsing + force [-3m,-2m,-1m,0m] buckets ---------
def _parse_period_to_date(s):
    if s is None: return None
    s = str(s).strip()
    m_ago = re.fullmatch(r"(-?\d+)\s*[mM]", s)
    if m_ago:
        num = int(m_ago.group(1))
        months_ago = abs(num)
        base = pd.Timestamp(datetime.now()).normalize().replace(day=1)
        dt = (base - pd.DateOffset(months=months_ago)).to_pydatetime()
        return dt
    if re.fullmatch(r"\d{10}", s):
        try: return datetime.utcfromtimestamp(int(s)).replace(day=1)
        except Exception: return None
    if re.fullmatch(r"\d{4}-\d{2}$", s):
        try: return datetime.strptime(s + "-01", "%Y-%m-%d")
        except Exception: pass
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}$", s[:10]):
        try: return datetime.strptime(s[:10], "%Y-%m-%d").replace(day=1)
        except Exception: pass
    try:
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        if pd.isna(dt): return None
        dt = dt.tz_convert(None).to_pydatetime()
        return dt.replace(day=1)
    except Exception:
        return None

def _force_last4_month_rows(rec_df):
    base = pd.Timestamp(datetime.now()).normalize().replace(day=1)
    targets = [(base - pd.DateOffset(months=i)).to_pydatetime() for i in [3,2,1,0]]
    labels = [dt.strftime("%b") for dt in targets]
    rows = []
    if rec_df is not None and not rec_df.empty and "period" in rec_df.columns:
        df = rec_df.copy()
        df["__pdate"] = df["period"].apply(_parse_period_to_date)
        df = df.dropna(subset=["__pdate"])
        df["__year"] = df["__pdate"].dt.year
        df["__month"] = df["__pdate"].dt.month
        for dt in targets:
            yy, mm = dt.year, dt.month
            cand = df[(df["__year"]==yy) & (df["__month"]==mm)]
            if cand.empty:
                rows.append({"strongBuy":0,"buy":0,"hold":0,"sell":0,"strongSell":0})
            else:
                r = cand.iloc[-1]
                rows.append({
                    "strongBuy": float(r.get("strongBuy",0) or 0),
                    "buy": float(r.get("buy",0) or 0),
                    "hold": float(r.get("hold",0) or 0),
                    "sell": float(r.get("sell",0) or 0),
                    "strongSell":float(r.get("strongSell",0) or 0),
                })
    else:
        rows = [{"strongBuy":0,"buy":0,"hold":0,"strongSell":0,"sell":0} for _ in range(4)]
    df4 = pd.DataFrame(rows)
    return df4, labels

# ----------------------------- Analyst widgets -----------------------------
def draw_recs(container_ax, rec_df):
    container_ax.axis("off")
    ss = container_ax.get_subplotspec()
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=ss, width_ratios=[2, 1], wspace=0.23)
    fig = container_ax.figure
    ax1 = fig.add_subplot(gs[0]) # bars
    ax2 = fig.add_subplot(gs[1]) # legend

    fig = ax1.figure
    pos = ax1.get_position()
    TITLE_SHIFT = -0.010
    fig.text(
        pos.x0,
        pos.y1 + 0.00627 + TITLE_SHIFT,
        "Analyst Recommendations & Price Targets",
        ha="left", va="bottom",
        fontsize=13, fontweight="bold", color=UOBKH_NAVY_BRIGHT
    )
    fig.text(pos.x0, pos.y1 - 0.002, "Created by Yutian",
             ha="left", va="top", fontsize=11, color="#FFFFFF", alpha=0.0)
    ax1.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.90])

    ax1.set_facecolor(UOBKH_BG); ax2.set_facecolor(UOBKH_BG)
    df4, months = _force_last4_month_rows(rec_df)
    categories = ['Sell', 'Underperform', 'Hold', 'Buy', 'Strong Buy']
    colors = {
        'Strong Buy': UOBKH_GREEN,
        'Buy': "#7FB77E",
        'Hold': "#C6CCD6",
        'Underperform': "#E17C2D",
        'Sell': UOBKH_RED
    }
    sell = df4.get("strongSell", pd.Series([0,0,0,0])).values
    underperf = df4.get("sell", pd.Series([0,0,0,0])).values
    hold = df4.get("hold", pd.Series([0,0,0,0])).values
    buy = df4.get("buy", pd.Series([0,0,0,0])).values
    strong_buy = df4.get("strongBuy", pd.Series([0,0,0,0])).values
    data = np.vstack([sell, underperf, hold, buy, strong_buy])
    n = len(months)
    x = np.arange(0, n * BAR_GAP, BAR_GAP)
    bottom = np.zeros(n)
    totals = data.sum(axis=0)
    totals = np.where(totals==0, 1.0, totals)
    for series, cat in zip(data, categories):
        ax1.bar(
            x, series, bottom=bottom,
            color=colors[cat], width=BAR_WIDTH, align='center',
            edgecolor=UOBKH_BG, linewidth=0.4, zorder=2
        )
        for j, h in enumerate(series):
            if h > 0 and (h / totals[j]) >= SHOW_SEGMENT_PCT:
                ax1.text(x[j], bottom[j] + h/2.0, f"{int(h)}",
                         ha='center', va='center', fontsize=8, fontweight='bold', color=UOBKH_BG)
        bottom += series
    ymax = max(5.0, bottom.max()) + 5
    for j, tot in enumerate(bottom):
        label_color = UOBKH_NAVY
        if j == 0 and tot == 0:
            label_color = UOBKH_BG
        ax1.text(x[j], tot + ymax*0.03, str(int(tot)),
                 ha='center', va='bottom', fontsize=8, fontweight='bold', color=label_color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(months)
    for t in ax1.get_xticklabels():
        t.set_fontsize(10); t.set_fontweight('bold'); t.set_color(UOBKH_TEXT)
    for j, t in enumerate(ax1.get_xticklabels()):
        if j == 0 and bottom[j] == 0:
            t.set_color(UOBKH_BG)
    ax1.set_ylim(0, ymax)
    ax1.set_xlim(x[0] - BAR_WIDTH*0.7, x[-1] + BAR_WIDTH*0.7)
    for spine in ax1.spines.values(): spine.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.tick_params(axis='x', which='both', length=0)
    legend_order = ['Strong Buy', 'Buy', 'Hold', 'Underperform', 'Sell']
    handles = [Line2D([0],[0], marker='o', color='w', label=lab,
                      markerfacecolor=colors[lab], markersize=9)
               for lab in legend_order]
    ax2.legend(handles=handles, loc='center', frameon=False,
               fontsize=10, handletextpad=0.25, borderpad=0.2, labelspacing=0.8, facecolor=UOBKH_BG)
    ax2.axis('off')

# ---------- helper for bold-number + label inside colored rounded box ----------
def _filled_two_line_badge(ax, xdata, y_ax_frac, color, number, label,
                           trans, num_size=11, lab_size=9):
    txt = rf"$\bf{{{number:.2f}}}$" + f"\n{label}"
    ax.text(xdata, y_ax_frac, txt,
            ha="center", va="center", color=UOBKH_BG,
            fontsize=num_size, linespacing=1.05,
            transform=trans,
            bbox=dict(boxstyle="round,pad=0.25", fc=color, ec=color, lw=1.5, alpha=0.98),
            zorder=4)

def draw_targets(ax, pt):
    ax.set_title("Analyst Price Targets", loc="left", x=0.10, fontsize=13, fontweight="bold", pad=8, color=UOBKH_BG)
    ax.set_facecolor(UOBKH_BG)
    lo, av, hi, cur, n = pt.get("low"), pt.get("average"), pt.get("high"), pt.get("current"), pt.get("numberOfAnalysts")
    if lo is None or hi is None:
        ax.text(0.0, 0.5, "No data", ha="left", va="center", fontsize=11, color=UOBKH_TEXT)
        ax.axis("off")
        return
    ax.set_ylim(0, 1); ax.set_yticks([])
    pad = max(1.0, (hi - lo) * 0.15)
    ax.set_xlim(lo - pad, hi + pad); ax.set_xticks([])
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    pill_y = 0.50; pill_h = 0.12
    pill = FancyBboxPatch((lo, pill_y), hi - lo, pill_h,
        boxstyle="round,pad=0,rounding_size={}".format(pill_h/2),
        transform=trans, facecolor="#D0D6DF", edgecolor="none", alpha=0.95, zorder=1)
    ax.add_patch(pill)
    ax.vlines([lo, hi], pill_y - 0.06, pill_y + pill_h + 0.06, transform=trans, colors=UOBKH_SLATE, lw=0.9, zorder=2)

    # ===== ONLY CHANGE: clamp 'Current' marker inside the bar =====
    cur_plot = cur
    if cur_plot is not None and lo is not None and hi is not None:
        try:
            curf = float(cur_plot)
            span = hi - lo
            if span != 0:
                eps = max(1e-9, 0.001 * abs(span))  # tiny offset so it’s visibly inside
                if curf < lo:
                    curf = lo + eps          # very left inside
                elif curf > hi:
                    curf = hi - eps          # very right inside
            cur_plot = curf
        except Exception:
            pass
    # ===== END ONLY CHANGE =====

    if cur_plot is not None: ax.scatter([cur_plot], [pill_y + pill_h/2], s=40, color=UOBKH_RED, transform=trans, zorder=3)
    if av is not None:  ax.scatter([av],  [pill_y + pill_h/2], s=40, color=UOBKH_NAVY, transform=trans, zorder=3)
    if av is not None:  _filled_two_line_badge(ax, av,       pill_y + pill_h/2 + 0.22, color=UOBKH_NAVY, number=av,  label="Average", trans=trans, num_size=8, lab_size=6)
    if cur_plot is not None: _filled_two_line_badge(ax, cur_plot, pill_y + pill_h/2 - 0.22, color=UOBKH_RED,  number=(cur if cur is not None else cur_plot), label="Current", trans=trans, num_size=8, lab_size=6)

    lowhigh_offset = 0.32
    ax.text(lo, pill_y - lowhigh_offset, f"$\\bf{{{lo:.2f}}}$\nLow", transform=trans, ha="left",  va="top", fontsize=10, color=UOBKH_TEXT, linespacing=1.05)
    ax.text(hi, pill_y - lowhigh_offset, f"$\\bf{{{hi:.2f}}}$\nHigh", transform=trans, ha="right", va="top", fontsize=10, color=UOBKH_TEXT, linespacing=1.05)
    if n is not None:
        ax.text(0.88, 1.00, f"Analysts: {n}", transform=ax.transAxes, ha="right", va="top", fontsize=10, color=UOBKH_BG)
    for s in ["top", "right", "left", "bottom"]: ax.spines[s].set_visible(False)
    # Draw connection lines
    if av is not None:
        ax.vlines(av,      pill_y + pill_h/2,              pill_y + pill_h/2 + 0.22 - 0.04, transform=trans, colors=UOBKH_NAVY, lw=1.2, zorder=2)
    if cur_plot is not None:
        ax.vlines(cur_plot, pill_y + pill_h/2 - 0.22 + 0.04, pill_y + pill_h/2,              transform=trans, colors=UOBKH_RED,  lw=1.2, zorder=2)

from matplotlib.lines import Line2D

def _draw_sandwich_row(fig, ax_left, ax_right,
                       y_above=0.012, y_below=-0.006,
                       inset_left=0.00, inset_right=0.00,
                       lw_top=1.6, lw_bottom=1.6,
                       top_color="#FFFFFF", bottom_color=UOBKH_NAVY,
                       top_alpha=1.0, bottom_alpha=1.0):
    pL = ax_left.get_position(); pR = ax_right.get_position()
    x0 = pL.x0 + inset_left
    x1 = pR.x1 - inset_right
    y  = max(pL.y1, pR.y1)

    line_top = Line2D([x0, x1], [y + y_above, y + y_above],
                      transform=fig.transFigure, lw=lw_top, color=top_color,
                      alpha=top_alpha, solid_capstyle="butt", zorder=20)
    line_bot = Line2D([x0, x1], [y + y_below, y + y_below],
                      transform=fig.transFigure, lw=lw_bottom, color=bottom_color,
                      alpha=bottom_alpha, solid_capstyle="butt", zorder=20)

    line_top.set_clip_on(False); line_bot.set_clip_on(False)
    fig.add_artist(line_top); fig.add_artist(line_bot)

# ----------------------------- NEW: Transposed Underlyings Table (no ticker header row) -----------------------------
def draw_underlying_table_transposed(ax, tickers, matrix, row_scale=1.2):
    """
    Transposed Underlyings table.
    - First row ("Underlying") uses dynamic font sizing with base 13pt (scales down if needed).
    - All other label/value text is fixed at 13pt.
    matrix rows expected:
      [Underlying, Ticker, Indicative Spot, Knock Out, Strike, Knock In]
    """
    import textwrap
    ax.axis("off")

    # --- fixed fonts ---
    LABEL_FS = 13
    VALUE_FS = 13
    UNDER_BASE_FS = 13   # default (max) font for the Underlying row
    UNDER_MIN_FS  = 9    # clamp minimum for readability

    # --- geometry ---
    left, right, top, bottom = 0.02, 0.98, 0.99, 0.01
    LABEL_COL_FRAC = 0.22

    n_rows = len(matrix)
    n_cols = max(1, len(tickers))

    # x boundaries
    label_x0 = left
    label_x1 = left + (right - left) * LABEL_COL_FRAC
    data_x0  = label_x1
    data_x1  = right
    xs = np.linspace(data_x0, data_x1, n_cols + 1)

    # left-side row labels
    row_labels = ["Underlying", "Ticker", "Indicative Spot", "Knock Out", "Strike", "Knock In"]
    if len(row_labels) != n_rows:
        row_labels = (row_labels[:n_rows] + [""] * n_rows)[:n_rows]

    # --- helpers for wrapping & widths ---
    def _wrap_lines(txt, max_chars):
        if txt is None:
            return [""]
        s = str(txt)
        lines = textwrap.wrap(s, width=max_chars, break_long_words=False, break_on_hyphens=False)
        return lines if lines else [s]

    def _cell_wrap_width(nc):
        if nc <= 2: return 24
        if nc == 3: return 18
        return 14  # 4+

    wrap_w = _cell_wrap_width(n_cols)

    # --- row heights (keep total constant; let "Underlying" row grow a bit) ---
    avail_h = (top - bottom) * row_scale
    base_rh = avail_h / max(1, n_rows)

    underlying_vals = matrix[0] if matrix else []
    max_lines_under = max([1] + [len(_wrap_lines(v, wrap_w)) for v in underlying_vals])
    underlying_rh = min(2.0 * base_rh, base_rh * max_lines_under)

    if n_rows > 1:
        other_rh = (avail_h - underlying_rh) / (n_rows - 1)
        other_rh = max(other_rh, 0.6 * base_rh)  # floor
        # ensure total doesn't overflow
        total_h = underlying_rh + other_rh * (n_rows - 1)
        if total_h > avail_h:
            underlying_rh = max(avail_h - other_rh * (n_rows - 1), base_rh)
        row_heights = [underlying_rh] + [other_rh] * (n_rows - 1)
    else:
        row_heights = [base_rh]

    # --- draw ---
    y_cur = top
    for i, (row_vals, rh) in enumerate(zip(matrix, row_heights)):
        y1, y0 = y_cur, y_cur - rh
        y_cur = y0

        # zebra band
        if i % 2 == 0:
            ax.add_patch(matplotlib.patches.Rectangle(
                (left, y0), right - left, rh,
                transform=ax.transAxes, color=UOBKH_ZEBRA, zorder=0, clip_on=False
            ))

        # left label cell (red)
        ax.add_patch(matplotlib.patches.Rectangle(
            (label_x0, y0), label_x1 - label_x0, rh,
            transform=ax.transAxes, color=UOBKH_RED, zorder=1, clip_on=False
        ))
        ax.text(label_x0 + 0.012, (y0 + y1) / 2, row_labels[i],
                transform=ax.transAxes, va="center", ha="left",
                fontsize=LABEL_FS, fontweight="bold", color="white", zorder=2)

        # grid lines
        for x in [label_x0, label_x1, right]:
            ax.plot([x, x], [y0, y1], color=UOBKH_GRID, lw=0.8,
                    transform=ax.transAxes, clip_on=False)
        for x in xs[1:-1]:
            ax.plot([x, x], [y0, y1], color=UOBKH_GRID, lw=0.8,
                    transform=ax.transAxes, clip_on=False)
        ax.plot([left, right], [y0, y0], color=UOBKH_GRID, lw=0.8,
                transform=ax.transAxes, clip_on=False)

        # data cells
        for j in range(n_cols):
            val = "" if j >= len(row_vals) else row_vals[j]
            xR = xs[j+1]

            if i == 0:
                # Underlying row: wrap & dynamic font with base 13pt (scales down if tight)
                lines = _wrap_lines(val, wrap_w)
                L = max(1, len(lines))
                effective_needed   = L * 1.05
                effective_capacity = max(1.0, (rh / base_rh) * 1.9)
                scale = min(1.0, effective_capacity / effective_needed)
                fs = max(UNDER_MIN_FS, UNDER_BASE_FS * scale)  # clamp [UNDER_MIN_FS, 13]
                ax.text(
                    xR - 0.012, (y0 + y1) / 2, "\n".join(lines),
                    transform=ax.transAxes, va="center", ha="right",
                    fontsize=fs, color=UOBKH_TEXT, linespacing=1.05
                )
            else:
                ax.text(
                    xR - 0.012, (y0 + y1) / 2, str(val),
                    transform=ax.transAxes, va="center", ha="right",
                    fontsize=VALUE_FS, color=UOBKH_TEXT
                )
def _draw_warning_box(fig,
                      left=0.06, right=0.94,
                      y_bottom=0.08,
                      body_fontsize=12,
                      # width / centering (keep if you use them)
                      width_frac=0.69, center_x=True, y_center=None,
                      # spacing controls (em = line-heights)
                      pad_y_em=1.60,            # ↑ top/bottom padding → taller box
                      body_leading_em=1.32,      # ↑ line spacing between body lines
                      # NEW underline controls
                      underline_offset_px=1.5,   # distance UNDER the baseline (px). 0 = touching
                      underline_lw=1.0,          # underline thickness (1.0 is thin)
                      # NEW overall height scaler
                      height_scale=1.0,          # multiply computed height (e.g., 0.9 shorter, 1.1 taller)
                      **_):
    import matplotlib.patches as patches
    import matplotlib as mpl
    from matplotlib import font_manager as fm

    header = "WARNING"
    lines = [
        "THE  RETURNS  ON  YOUR  STRUCTURED  PRODUCT  INVESTMENT",
        "WILL BE AFFECTED BY THE PERFORMANCE OF THE UNDERLYING",
        "ASSET/REFERENCE,  AND  THE  RECOVERY  OF  YOUR  PRINCIPAL",
        "INVESTMENT  MAY  BE  JEOPARDISED  IF  YOU  MAKE  AN  EARLY",
        "REDEMPTION."
    ]

    # ---- horizontal sizing / centering ----
    if width_frac is not None:
        w = max(0.30, min(0.98, float(width_frac)))
        if center_x:
            left = (1.0 - w) / 2.0
            right = 1.0 - left
        else:
            right = left + w
    width = right - left

    # ---- type metrics ----
    fig_w, fig_h = fig.get_size_inches()
    line_h_fig = (body_fontsize / 72.0) / fig_h
    gap_fig     = line_h_fig * (body_leading_em - 1.0)
    pad_y_fig   = line_h_fig * pad_y_em
    pad_x_ax    = 0.020  # inner L/R padding (axes fraction)

    total_text_h_fig = (
        line_h_fig +                                           # header line
        (len(lines) * (line_h_fig + gap_fig) - gap_fig)        # body lines
    )
    height_fig = (total_text_h_fig + 2 * pad_y_fig) * float(height_scale)

    # vertical centering if requested
    if y_center is not None:
        y_bottom = float(y_center) - height_fig / 2.0

    # ---- axes (work in AXES fraction) ----
    ax = fig.add_axes([left, y_bottom, width, height_fig])
    ax.axis("off")

    # px ↔ axes conversion
    p0 = ax.transAxes.transform((0, 0)); p1 = ax.transAxes.transform((1, 1))
    ax_w_px, ax_h_px = (p1 - p0)

    def _to_ax(v_fig): return v_fig / height_fig if height_fig else v_fig
    line_h_ax = _to_ax(line_h_fig)
    gap_ax    = _to_ax(gap_fig)
    pad_y_ax  = _to_ax(pad_y_fig)

    # frame
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, lw=2.0,
                                   edgecolor="black", transform=ax.transAxes, clip_on=False))

    # header
    x_text = pad_x_ax
    y_top  = 1.0 - pad_y_ax
    hdr = ax.text(x_text, y_top, header, ha="left", va="top",
                  fontsize=body_fontsize, fontweight="bold",
                  transform=ax.transAxes, color="black")

    # --- underline glued to baseline (thin & a bit lower) ---
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fp = fm.FontProperties(
        family=mpl.rcParams.get("font.family", None),
        weight="bold",
        size=body_fontsize
    )
    # text width/height/descent in pixels
    tw, th, descent = renderer.get_text_width_height_descent(header, fp, ismath=False)
    underline_len = max(0.0, min(1.0 - 2*pad_x_ax, tw / ax_w_px))
    # baseline y = top minus ascent (th is ascent)
    y_baseline = y_top - (th / ax_h_px)
    # place a little below baseline (underline_offset_px)
    y_uline = y_baseline - (float(underline_offset_px) / ax_h_px)
    ax.plot([x_text, x_text + underline_len], [y_uline, y_uline],
            transform=ax.transAxes, color="black", lw=float(underline_lw), solid_capstyle="butt")

    # body
    y = y_uline - (line_h_ax * 0.70)  # gap from underline to first line
    for i, line in enumerate(lines):
        ax.text(x_text, y, line, ha="left", va="top",
                fontsize=body_fontsize, fontweight="bold",
                transform=ax.transAxes, color="black")
        if i < len(lines) - 1:
            y -= (line_h_ax + gap_ax)

# ----------------------------- layout & PDF -----------------------------
def draw_card(fig, subspec, symbol, strike, ko, ki, currency):
    gs = GridSpecFromSubplotSpec(
        5, 1, subplot_spec=subspec,
        height_ratios=[0.05, 0.01, 0.35, 0.16, 0.28],
        hspace=0.0
    )
    ax_title = fig.add_subplot(gs[0]); ax_title.axis("off")
    fig.add_subplot(gs[1]).axis("off") # spacer
    top_pair = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.10, width_ratios=[4.5, 3.5])
    ax_chart = fig.add_subplot(top_pair[0])
    ax_matrix = fig.add_subplot(top_pair[1]); ax_matrix.axis("off")
    fig.add_subplot(gs[3]).axis("off") # spacer
    gs_widgets = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[4], wspace=0.10, width_ratios=[1,1])
    ax_recs_container = fig.add_subplot(gs_widgets[0])
    ax_targets = fig.add_subplot(gs_widgets[1])
    try:
        name = yf.Ticker(symbol).info.get("longName", symbol)
    except Exception:
        name = symbol
    ax_title.add_patch(matplotlib.patches.Rectangle((0, 0), 0.98, 1, transform=ax_title.transAxes, color=UOBKH_NAVY, zorder=1))
    ax_title.text(0.5, 0.5, f"{name} ({symbol})", transform=ax_title.transAxes,
                  va="center", ha="center", fontsize=13, fontweight="bold", color="white")
    end = datetime.now(); start = end - timedelta(days=365)
    draw_stock_chart(ax_chart, symbol, strike, ko, ki, start, end)
    snap = get_snapshot(symbol)
    qc = _quote_currency(symbol) or currency
    snap_rows = [
        ("Previous Close", f"{qc} {snap['Previous Close']}"),
        ("52W High",       snap["52W High"]),
        ("52W Low",        snap["52W Low"]),
        ("PE (TTM)",       snap["PE (TTM)"]),
        ("P/S (TTM)",      snap["P/S (TTM)"]),
        ("P/B (MRQ)",      snap["P/B (MRQ)"]),
        ("Market Cap",     snap["Market Cap"]),
        ("EPS Growth YoY", snap["EPS Growth YoY"]),
        ("Fwd Div & Yield",snap["Fwd Div & Yield"]),
        ("Earnings Date",  snap["Earnings Date"])
    ]
    draw_kv_table(ax_matrix, snap_rows, one_col=True, row_scale=1.3)
    rec_df, pt = get_widgets(symbol)
    draw_recs(ax_recs_container, rec_df)
    draw_targets(ax_targets, pt)
    _draw_sandwich_row(
        fig, ax_recs_container, ax_targets,
        y_above=-0.165, y_below=0.003,
        top_color=UOBKH_GRID, bottom_color=UOBKH_BG,
        inset_left=0.00, inset_right=0.018,
        lw_top=1.2, lw_bottom=1.2,
        bottom_alpha=0.0
    )

def _add_first_page_header(fig, tickers, strike, ko, ki, tenor, currency, coupon, ko_type, ki_type):
    # --- Header bar ---
    fig.patches.append(matplotlib.patches.Rectangle((0, 0.96), 1, 0.04,
                                                    transform=fig.transFigure, color=UOBKH_NAVY, zorder=3))
    fig.text(0.5, 0.98, "Autocallable Equity Linked Structured Investment", ha="center", va="center",
             fontsize=16, fontweight="bold", color="white")
    fig.text(0.06, 0.9525, "FOR REFERENCE ONLY", ha='left', va='top',
             fontsize=10, fontweight='bold', color=UOBKH_RED)

    disclaimer_text = (
        "IMPORTANT: Investment involves risk, including the loss of principal. "
        "Please note that the information provided here are for reference only.\n"
        "You are receiving this because you completed a Financial Needs Analysis and your risk profile is Growth or above. "
        "See disclaimers below."
    )
    fig.text(0.06, 0.9375, disclaimer_text, ha='left', va='top',
             fontsize=8, color=UOBKH_TEXT, linespacing=1.25)

    # --- Normalize display for KO/KI ---
    ko_type_disp = smart_title_of_only(ko_type) if ko_type else ""
    ki_type_cell = "NA" if ki is None else (smart_title_of_only(ki_type) if ki_type else "")
    ki_level_cell = "NA" if ki is None else f"{ki:.2f}%"
    ko_level_cell = "NA" if ko is None else f"{ko:.2f}%"

    # --- Terms table rows (always defined) ---
    terms_rows = [
        ("Tenor (mo)",    str(tenor)),
        ("Currency",      currency),
        ("Coupon (p.a.)", f"{coupon:.2f}%"),
        ("Strike",        f"{strike:.2f}%"),
        ("KO Type",       ko_type_disp),
        ("KO Level",      ko_level_cell),
        ("KI Type",       ki_type_cell),
        ("KI Level",      ki_level_cell),
    ]

    TERMS_POS = [0.04, 0.832, 0.92, 0.083]
    ax_table = fig.add_axes(TERMS_POS)
    draw_kv_table(
        ax_table, terms_rows, one_col=False,
        header="Terms", header_bg=UOBKH_RED, header_color="white",
        row_scale=2.0
    )

    # Make ALL text in this Terms table (header + labels + values) size 15
    for txt in list(ax_table.texts): txt.set_fontsize(15)

    fig.text(0.061, 0.52,
             "*Indicative only. Please refer to Termsheets / Final Termsheets for full terms and conditions",
             ha="left", va="top", fontsize=8, color=UOBKH_MUTED_GREY)

    # --- Underlyings table ---
    UNDER_H = 0.11
    GAP = 0.09
    under_y = TERMS_POS[1] - UNDER_H - GAP
    ax_under = fig.add_axes([TERMS_POS[0], under_y, TERMS_POS[2], UNDER_H])

    underlyings = [long_name(t) for t in tickers]
    tickers_row = list(tickers)

    # Indicative Spot → EXACT same logic/formatting as the 'Previous Close' table cell
    spots = [get_snapshot(t)["Previous Close"] for t in tickers]

    ko_row, strike_row, ki_row = [], [], []
    for spot in spots:
        try:
            spot_val = float(spot)
        except:
            spot_val = None

        if spot_val is not None:
            ko_row.append("NA" if ko is None else _fmt_num(spot_val * (ko / 100.0), 2))
            strike_row.append(_fmt_num(spot_val * (strike / 100.0), 2))
            ki_row.append("NA" if ki is None else _fmt_num(spot_val * (ki / 100.0), 2))
        else:
            ko_row.append("NA" if ko is None else "")
            strike_row.append("")
            ki_row.append("NA" if ki is None else "")

    matrix = [
        underlyings,
        tickers_row,
        spots,
        ko_row,
        strike_row,
        ki_row
    ]
    draw_underlying_table_transposed(ax_under, tickers=tickers, matrix=matrix, row_scale=2.0)

def _add_page_footer(fig, page_num):
    from datetime import date
    ref_date = date.today().strftime("%Y-%m-%d")
    # Left bottom corner: Reference Date
    fig.text(0.06, 0.02, f"Reference Date: {ref_date}",
             ha="left", va="bottom", fontsize=8, color=UOBKH_MUTED_GREY)
    # Right bottom corner: Page number
    fig.text(0.94, 0.02, f"{page_num}",
             ha="right", va="bottom", fontsize=8, color=UOBKH_MUTED_GREY)

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
# ---- Layout-preserving PdfPages shim (no fontTools; no tight bbox) ----
# Tries Matplotlib's PdfPages; if it asserts due to fontTools, falls back to a
# Pillow-based writer that preserves the figure's *exact* page size/margins.
try:
    from matplotlib.backends.backend_pdf import PdfPages as _MplPdfPages
    _PDF_BACKEND_OK = True
except Exception:
    _MplPdfPages = None
    _PDF_BACKEND_OK = False

if _PDF_BACKEND_OK:
    PdfPages = _MplPdfPages
else:
    import io
    from PIL import Image

    class PdfPages:
        """
        Drop-in replacement:
          with PdfPages("out.pdf") as pdf:
              pdf.savefig(fig)
        This rasterizes each page at high DPI but **keeps figure size and margins**
        by disabling tight layout cropping.
        """
        def __init__(self, filename, metadata=None, dpi=400):
            self.filename = filename
            self.metadata = metadata or {}
            self.dpi = int(dpi)
            self._frames = []

        def savefig(self, fig):
            buf = io.BytesIO()
            # IMPORTANT: keep exact canvas (no bbox='tight'), no padding tweaks.
            fig.savefig(
                buf,
                format="png",
                dpi=self.dpi,
                bbox_inches=None,       # preserve full figure canvas
                pad_inches=0,           # no extra padding
                facecolor=fig.get_facecolor(),
                edgecolor="none"
            )
            buf.seek(0)
            # Keep RGB and page dimensions exactly as rendered
            self._frames.append(Image.open(buf).convert("RGB"))

        def close(self):
            if not self._frames:
                return
            first, *rest = self._frames
            # Save as a multi-page PDF. Pillow uses image pixel size as page size,
            # which matches the figure inches * DPI above — so layout stays identical.
            first.save(
                self.filename,
                format="PDF",
                save_all=True,
                append_images=rest,
            )
            self._frames.clear()

        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): self.close()

import textwrap

def build_pdf(tickers, strike=73.54, ko=100.0, ki=None, tenor=8, currency="USD",
              coupon=10.0, ko_type="Daily Memory", ki_type="", out_pdf=None):
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        raise ValueError("No tickers.")
    if out_pdf is None:
        today = date.today().strftime("%Y-%m-%d")
        out_pdf = f"AESI One Pager - {coupon:.2f}% - {today}.pdf"

    if len(tickers) == 1:
        groups = [tickers]
    else:
        groups = [[tickers[0]]] + [tickers[i:i+2] for i in range(1, len(tickers), 2)]

    W, H = 8.27, 11.69  # A4 portrait in inches

    # ---------- helper: draw justified text block ----------
    def _justify_text(fig, ax, text, x0, y0, width,
                      fontsize=9, color="black", line_height=0.018):
        """
        Draw a left+right justified paragraph block (last line left-aligned).
        Preserves paragraph breaks when the input contains '\n\n' by inserting
        a blank line between paragraphs. Coordinates are figure fraction (0..1).
        """
        import textwrap
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        wrap_width = max(20, int(width * fig.get_size_inches()[0] * fig.dpi / 6.2))
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = text.split("\n\n")

        y = y0
        for p_idx, para in enumerate(paragraphs):
            lines = textwrap.wrap(
                para,
                width=wrap_width,
                break_long_words=False,
                break_on_hyphens=False
            ) or [""]

            for i, line in enumerate(lines):
                # last line in a paragraph -> left align
                if i == len(lines) - 1 or not line.strip():
                    fig.text(x0, y, line, ha="left", va="top",
                            fontsize=fontsize, color=color)
                    y -= line_height
                    continue

                tmp = ax.text(0, 0, line, fontsize=fontsize)
                bb = tmp.get_window_extent(renderer=renderer)
                tmp.remove()
                line_w_frac = bb.width / fig.dpi / fig.get_size_inches()[0]

                words = line.split()
                if len(words) <= 1 or line_w_frac >= width:
                    fig.text(x0, y, line, ha="left", va="top",
                            fontsize=fontsize, color=color)
                    y -= line_height
                    continue

                gaps = len(words) - 1
                extra_needed = max(0.0, width - line_w_frac)
                max_extra_per_gap = 0.005
                extra_per_gap = min(extra_needed / gaps, max_extra_per_gap)

                cursor = x0
                for j, w in enumerate(words):
                    fig.text(cursor, y, w, ha="left", va="top",
                            fontsize=fontsize, color=color)
                    if j < gaps:
                        meas = ax.text(0, 0, w + " ", fontsize=fontsize)
                        bbw = meas.get_window_extent(renderer=renderer)
                        meas.remove()
                        step = bbw.width / fig.dpi / fig.get_size_inches()[0]
                        cursor += step + extra_per_gap

                y -= line_height

            if p_idx < len(paragraphs) - 1:
                y -= line_height

    with PdfPages(out_pdf, metadata={"Title": out_pdf, "Author": "Stock Chart Tool"}) as pdf:
        page_num = 1

        # --- your existing chart pages ---
        for idx, group in enumerate(groups):
            fig = plt.figure(figsize=(W, H))
            top_margin = 0.965
            gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1],
                          hspace=0.08, top=top_margin, bottom=0.06, left=0.06, right=0.96)

            if idx == 0:
                _add_first_page_header(fig, tickers, strike, ko, ki, tenor, currency, coupon, ko_type, ki_type)
                draw_card(fig, gs[1], group[0], strike, ko, ki, currency)
            else:
                for r, sym in enumerate(group):
                    draw_card(fig, gs[r], sym, strike, ko, ki, currency)

            _add_page_footer(fig, page_num)
            pdf.savefig(fig)
            plt.close(fig)
            page_num += 1

        # --- Disclaimer page (JUSTIFIED) ---
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        fig.text(0.5, 0.96, "Disclaimer", ha="center", va="top",
                fontsize=14, fontweight="bold", color=UOBKH_NAVY)

        disclaimer_text = (
            "The information provided herein is intended for general circulation and discussion purposes only, and should not be relied upon as financial advice. It does not take into account the specific investment objectives, financial situation, or particular needs of any individual. Without prejudice to the generality of the foregoing, please seek advice from a licensed financial adviser regarding the suitability of any investment product in light of your specific objectives, financial situation, and needs before making any commitment to purchase such a product. If you choose not to seek such advice, you should carefully consider whether the product is suitable for you. This material does not constitute an offer, solicitation, or recommendation to buy, sell, or subscribe for any security or financial instrument, nor to enter into any transaction or adopt any specific trading or investment strategy. Any proposed transaction(s) (if applicable) remain subject to the final terms set forth in the relevant definitive agreement(s) and/or confirmation(s). No representation or warranty whatsoever (including, without limitation, as to accuracy, adequacy, usefulness, timeliness, completeness and fitness for purpose) is given in respect of the information provided, and it should not be relied upon as such. The information may be subject to change without notice, and no obligation is undertaken to update or correct any inaccuracy that may become apparent at a later date. To the fullest extent permitted by law, no responsibility or liability shall be accepted for any loss or damage of any kind (whether or not foreseeable) arising, directly or indirectly, in connection with any person acting or relying on the information contained herein or any errors or omissions in the information contained herein. The information provided may contain projections or other forward-looking statements regarding future events, markets, assets, or companies. Actual outcomes or results may differ materially. Past performance is not necessarily indicative of future or likely performance. Any reference to a specific company, product, or asset class is for illustrative purposes only and does not constitute a recommendation. While the information has been derived from sources believed to be reliable, no independent verification has been undertaken, and no assurance is given as to its reliability."
            "\n\nThis material is confidential. This material may not be published, circulated, reproduced or distributed in whole or in part by any recipient of this material to any other person without our prior written consent. The material has not been reviewed by the Securities Commission Malaysia. The material is intended solely for Sophisticated Investors and not for retail distribution. The material does not constitute investment advice or a recommendation, and Investment risks remain with the investor."
        )

        _justify_text(
            fig, ax, disclaimer_text,
            x0=0.06, y0=0.90, width=0.88,
            fontsize=9, color=UOBKH_TEXT, line_height=0.018
        )
        # Draw the framed WARNING exactly like the reference image (no external image used)
        _draw_warning_box(
            fig,
            left=0.06, right=0.94,   # align to same margins as text
            y_bottom=0.2,           # sits just above the footer
            body_fontsize=12,        # as requested
            fontname="Arial"         # as requested; falls back if Arial unavailable
        )

        try:
            _add_page_footer(fig, page_num)
        except NameError:
            fig.text(0.5, 0.035, f"Page {page_num}", ha="center", va="center",
                    fontsize=101, color=UOBKH_TEXT)

        pdf.savefig(fig)
        plt.close(fig)
        page_num += 1

    print(f"[OK] Saved PDF → {out_pdf}")

# ----------------------------- prompt -----------------------------
def prompt_and_build():
    """
    Multi-panel UI that generates one separate PDF per panel.

    - Duplicate-above checkbox + “Add another FCN”; “Generate PDFs” sits right next to it.
    - Each panel becomes its own PDF (not combined).
    - Suppresses build_pdf() prints; shows one green ✅ line per saved file.
    - Base filename: AESI One Pager - <coupon%> - <TICKERS_COMMA_SEP> - <YYYY-MM-DD>.pdf
      If another panel would produce the same base filename (same coupon + same tickers for today),
      then include tenor to disambiguate:
      AESI One Pager - <TENOR>M <coupon%> - <TICKERS> - <YYYY-MM-DD>.pdf
    """
    import io, contextlib, re
    from datetime import date
    from IPython.display import display, HTML
    import ipywidgets as w
    from collections import Counter

    # ---- Color fallbacks if theme constants aren't in scope ----
    _GRID = globals().get("UOBKH_GRID", "#D7DEE8")
    _NAVY = globals().get("UOBKH_NAVY", "#0F2A47")
    _NAVY_BRIGHT = globals().get("UOBKH_NAVY_BRIGHT", "#133861")
    _TEXT = globals().get("UOBKH_TEXT", "#243042")
    _RED  = globals().get("UOBKH_RED",  "#C33832")
    _GREEN= globals().get("UOBKH_GREEN", "#137D5B")
    _MUTED= globals().get("UOBKH_MUTED_GREY", "#6B7280")

    # -------------------- helpers --------------------
    