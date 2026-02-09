from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import datetime, timedelta
import json

app = FastAPI(title="Fintola Finance API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory cache
cache = {}
CACHE_TTL_QUOTE = 300  # 5 minutes for quotes
CACHE_TTL_CHART = 600  # 10 minutes for chart data


def get_cached(key: str, ttl: int):
    """Get cached data if not expired"""
    if key in cache:
        data, timestamp = cache[key]
        if datetime.now().timestamp() - timestamp < ttl:
            return data
    return None


def set_cache(key: str, data):
    """Set cache with current timestamp"""
    cache[key] = (data, datetime.now().timestamp())


@app.get("/")
async def root():
    return {"message": "Fintola Finance API is running", "status": "ok"}


@app.get("/api/quote")
async def get_quote(symbol: str = "RELIANCE.NS"):
    """Get current quote for a symbol"""
    cache_key = f"quote_{symbol}"
    cached = get_cached(cache_key, CACHE_TTL_QUOTE)
    if cached:
        return cached

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        
        if len(hist) == 0:
            return {"error": f"No data found for {symbol}"}
        
        current_price = float(hist['Close'].iloc[-1])
        previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        
        # Get additional info safely
        try:
            info = ticker.info
        except:
            info = {}
        
        result = {
            "meta": {
                "symbol": symbol,
                "shortName": info.get("shortName", symbol.replace(".NS", "")),
                "regularMarketPrice": current_price,
                "previousClose": previous_close,
                "regularMarketChange": current_price - previous_close,
                "regularMarketChangePercent": ((current_price - previous_close) / previous_close) * 100 if previous_close else 0,
                "regularMarketVolume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
                "currency": info.get("currency", "INR"),
                "exchange": info.get("exchange", "NSE"),
                "regularMarketOpen": float(hist['Open'].iloc[-1]) if 'Open' in hist else current_price,
                "regularMarketDayHigh": float(hist['High'].iloc[-1]) if 'High' in hist else current_price,
                "regularMarketDayLow": float(hist['Low'].iloc[-1]) if 'Low' in hist else current_price,
            }
        }
        
        set_cache(cache_key, result)
        return result
        
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


@app.get("/api/chart")
async def get_chart(symbol: str = "RELIANCE.NS", period: str = "1y", interval: str = "1h", start: str = None, end: str = None):
    """Get historical chart data for a symbol"""
    cache_key = f"chart_{symbol}_{period}_{interval}_{start}_{end}"
    cached = get_cached(cache_key, CACHE_TTL_CHART)
    if cached:
        return cached

    try:
        ticker = yf.Ticker(symbol)
        if start:
            hist = ticker.history(start=start, end=end, interval=interval)
        else:
            hist = ticker.history(period=period, interval=interval)
        
        if len(hist) == 0:
            return {"error": f"No data found for {symbol}"}
        
        # Get basic info safely
        try:
            info = ticker.info
        except:
            info = {}
        
        # Format data similar to yahoo-finance2 chart format
        timestamps = [int(ts.timestamp()) for ts in hist.index]
        
        result = {
            "meta": {
                "symbol": symbol,
                "shortName": info.get("shortName", symbol.replace(".NS", "")),
                "regularMarketPrice": float(hist['Close'].iloc[-1]),
                "previousClose": float(hist['Close'].iloc[-2]) if len(hist) > 1 else float(hist['Close'].iloc[-1]),
                "currency": info.get("currency", "INR"),
                "exchange": info.get("exchange", "NSE"),
            },
            "timestamp": timestamps,
            "indicators": {
                "quote": [{
                    "open": [float(x) for x in hist['Open'].tolist()],
                    "high": [float(x) for x in hist['High'].tolist()],
                    "low": [float(x) for x in hist['Low'].tolist()],
                    "close": [float(x) for x in hist['Close'].tolist()],
                    "volume": [int(x) for x in hist['Volume'].tolist()],
                }]
            }
        }
        
        set_cache(cache_key, result)
        return result
        
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


@app.get("/api/finance")
async def get_finance(symbol: str = "RELIANCE.NS", type: str = "chart", start: str = None, end: str = None):
    """Combined endpoint - use type=quote for current price, type=chart for historical"""
    if type == "quote":
        return await get_quote(symbol)
    else:
        return await get_chart(symbol, start=start, end=end)


CACHE_TTL_NEWS = 600  # 10 minutes for news


@app.get("/api/news")
async def get_news(symbol: str = "RELIANCE.NS"):
    """Get news articles for a symbol. Falls back to ^NSEI if none found."""
    cache_key = f"news_{symbol}"
    cached = get_cached(cache_key, CACHE_TTL_NEWS)
    if cached:
        return cached

    try:
        ticker = yf.Ticker(symbol)
        news_data = ticker.news or []

        articles = []
        for item in news_data[:15]:
            # yfinance 0.2.x+ returns a different structure
            content = item.get("content") or item
            title = content.get("title") or item.get("title", "")
            publisher = content.get("provider", {}).get("displayName") if isinstance(content.get("provider"), dict) else item.get("publisher", "")
            link = content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else item.get("link", "")
            pub_date = content.get("pubDate") or item.get("providerPublishTime")

            # Try to get thumbnail
            thumbnail = ""
            if isinstance(content.get("thumbnail"), dict):
                resolutions = content["thumbnail"].get("resolutions", [])
                if resolutions:
                    thumbnail = resolutions[0].get("url", "")
            elif isinstance(item.get("thumbnail"), dict):
                resolutions = item["thumbnail"].get("resolutions", [])
                if resolutions:
                    thumbnail = resolutions[0].get("url", "")

            # Parse timestamp
            ts = 0
            if isinstance(pub_date, (int, float)):
                ts = int(pub_date)
            elif isinstance(pub_date, str):
                try:
                    from dateutil.parser import parse as dateparse
                    ts = int(dateparse(pub_date).timestamp())
                except Exception:
                    ts = int(datetime.now().timestamp())

            if title:
                articles.append({
                    "title": title,
                    "publisher": publisher or "Unknown",
                    "link": link or "",
                    "timestamp": ts,
                    "thumbnail": thumbnail,
                })

        # Fallback: if symbol-specific news is empty, fetch general market news
        if len(articles) == 0 and symbol not in ("^NSEI", "^BSESN"):
            fallback_ticker = yf.Ticker("^NSEI")
            fallback_news = fallback_ticker.news or []
            for item in fallback_news[:10]:
                content = item.get("content") or item
                title = content.get("title") or item.get("title", "")
                publisher = content.get("provider", {}).get("displayName") if isinstance(content.get("provider"), dict) else item.get("publisher", "")
                link = content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else item.get("link", "")
                pub_date = content.get("pubDate") or item.get("providerPublishTime")
                ts = 0
                if isinstance(pub_date, (int, float)):
                    ts = int(pub_date)
                elif isinstance(pub_date, str):
                    try:
                        from dateutil.parser import parse as dateparse
                        ts = int(dateparse(pub_date).timestamp())
                    except Exception:
                        ts = int(datetime.now().timestamp())
                if title:
                    articles.append({
                        "title": title,
                        "publisher": publisher or "Unknown",
                        "link": link or "",
                        "timestamp": ts,
                        "thumbnail": "",
                    })

        result = {"symbol": symbol, "count": len(articles), "articles": articles}
        set_cache(cache_key, result)
        return result

    except Exception as e:
        return {"error": str(e), "symbol": symbol, "articles": []}

import math
import numpy as np


def safe_val(v):
    """Convert numpy/pandas types to JSON-safe Python primitives."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if math.isnan(v) else round(float(v), 2)
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 2)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def df_to_records(df):
    """Convert a pandas DataFrame to a list of dicts with safe values."""
    if df is None or df.empty:
        return []
    records = []
    for idx, row in df.iterrows():
        rec = {"date": str(idx)}
        for col in df.columns:
            rec[col] = safe_val(row[col])
        records.append(rec)
    return records


# ─── /api/fundamentals ──────────────────────────────────────────
CACHE_TTL_FUNDAMENTALS = 1800  # 30 minutes


@app.get("/api/fundamentals")
async def get_fundamentals(symbol: str = "RELIANCE.NS"):
    """Income statement, balance sheet, cash flow, analyst targets & recommendations."""
    cache_key = f"fundamentals_{symbol}"
    cached = get_cached(cache_key, CACHE_TTL_FUNDAMENTALS)
    if cached:
        return cached

    try:
        ticker = yf.Ticker(symbol)

        # Financial statements
        income = df_to_records(ticker.income_stmt)
        balance = df_to_records(ticker.balance_sheet)
        cashflow = df_to_records(ticker.cash_flow)

        # Quarterly
        q_income = df_to_records(ticker.quarterly_income_stmt)
        q_balance = df_to_records(ticker.quarterly_balance_sheet)
        q_cashflow = df_to_records(ticker.quarterly_cash_flow)

        # Analyst targets
        try:
            targets = ticker.analyst_price_targets
            analyst_targets = {
                "current": safe_val(targets.get("current")),
                "low": safe_val(targets.get("low")),
                "high": safe_val(targets.get("high")),
                "mean": safe_val(targets.get("mean")),
                "median": safe_val(targets.get("median")),
            } if targets else None
        except Exception:
            analyst_targets = None

        # Recommendations
        try:
            recs_df = ticker.recommendations
            recommendations = df_to_records(recs_df) if recs_df is not None else []
        except Exception:
            recommendations = []

        # Key ratios from info (lightweight pick)
        try:
            info = ticker.info or {}
            ratios = {
                "trailingPE": safe_val(info.get("trailingPE")),
                "forwardPE": safe_val(info.get("forwardPE")),
                "priceToBook": safe_val(info.get("priceToBook")),
                "debtToEquity": safe_val(info.get("debtToEquity")),
                "returnOnEquity": safe_val(info.get("returnOnEquity")),
                "profitMargins": safe_val(info.get("profitMargins")),
                "operatingMargins": safe_val(info.get("operatingMargins")),
                "revenueGrowth": safe_val(info.get("revenueGrowth")),
                "earningsGrowth": safe_val(info.get("earningsGrowth")),
                "dividendYield": safe_val(info.get("dividendYield")),
                "marketCap": safe_val(info.get("marketCap")),
                "enterpriseValue": safe_val(info.get("enterpriseValue")),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "shortName": info.get("shortName", symbol),
            }
        except Exception:
            ratios = {}

        result = {
            "symbol": symbol,
            "income_stmt": income,
            "balance_sheet": balance,
            "cash_flow": cashflow,
            "quarterly_income": q_income,
            "quarterly_balance": q_balance,
            "quarterly_cashflow": q_cashflow,
            "analyst_targets": analyst_targets,
            "recommendations": recommendations,
            "ratios": ratios,
        }
        set_cache(cache_key, result)
        return result

    except Exception as e:
        return {"error": str(e), "symbol": symbol}


# ─── /api/holders ────────────────────────────────────────────────
CACHE_TTL_HOLDERS = 1800  # 30 minutes


@app.get("/api/holders")
async def get_holders(symbol: str = "RELIANCE.NS"):
    """Major holders, institutional holders, insider transactions."""
    cache_key = f"holders_{symbol}"
    cached = get_cached(cache_key, CACHE_TTL_HOLDERS)
    if cached:
        return cached

    try:
        ticker = yf.Ticker(symbol)

        # Major holders
        try:
            mh = ticker.major_holders
            if mh is not None and not mh.empty:
                major = []
                for _, row in mh.iterrows():
                    major.append({"value": safe_val(row.iloc[0]), "label": str(row.iloc[1]) if len(row) > 1 else ""})
            else:
                major = []
        except Exception:
            major = []

        # Institutional holders
        try:
            ih = ticker.institutional_holders
            institutional = []
            if ih is not None and not ih.empty:
                for _, row in ih.iterrows():
                    institutional.append({
                        "holder": str(row.get("Holder", "")),
                        "shares": safe_val(row.get("Shares", 0)),
                        "value": safe_val(row.get("Value", 0)),
                        "pctHeld": safe_val(row.get("% Out", row.get("pctHeld", 0))),
                        "dateReported": str(row.get("Date Reported", "")) if row.get("Date Reported") is not None else "",
                    })
        except Exception:
            institutional = []

        # Mutual fund holders
        try:
            mf = ticker.mutualfund_holders
            mutual_funds = []
            if mf is not None and not mf.empty:
                for _, row in mf.iterrows():
                    mutual_funds.append({
                        "holder": str(row.get("Holder", "")),
                        "shares": safe_val(row.get("Shares", 0)),
                        "value": safe_val(row.get("Value", 0)),
                        "pctHeld": safe_val(row.get("% Out", row.get("pctHeld", 0))),
                        "dateReported": str(row.get("Date Reported", "")) if row.get("Date Reported") is not None else "",
                    })
        except Exception:
            mutual_funds = []

        # Insider transactions
        try:
            it = ticker.insider_transactions
            insiders = []
            if it is not None and not it.empty:
                for _, row in it.head(20).iterrows():
                    insiders.append({
                        "insider": str(row.get("Insider", row.get("insider", ""))),
                        "relation": str(row.get("Relation", row.get("relation", ""))),
                        "transaction": str(row.get("Transaction", row.get("text", ""))),
                        "shares": safe_val(row.get("Shares", row.get("shares", 0))),
                        "value": safe_val(row.get("Value", row.get("value", 0))),
                        "date": str(row.get("Start Date", row.get("startDate", ""))) if row.get("Start Date", row.get("startDate")) is not None else "",
                    })
        except Exception:
            insiders = []

        result = {
            "symbol": symbol,
            "major_holders": major,
            "institutional_holders": institutional,
            "mutual_fund_holders": mutual_funds,
            "insider_transactions": insiders,
        }
        set_cache(cache_key, result)
        return result

    except Exception as e:
        return {"error": str(e), "symbol": symbol}


# ─── /api/calendar ───────────────────────────────────────────────
CACHE_TTL_CALENDAR = 1800  # 30 minutes


@app.get("/api/calendar")
async def get_calendar(symbol: str = "RELIANCE.NS"):
    """Upcoming earnings, dividend dates, and earnings history."""
    cache_key = f"calendar_{symbol}"
    cached = get_cached(cache_key, CACHE_TTL_CALENDAR)
    if cached:
        return cached

    try:
        ticker = yf.Ticker(symbol)

        # Calendar (upcoming events)
        try:
            cal = ticker.calendar
            if isinstance(cal, dict):
                calendar_data = {k: safe_val(v) if not isinstance(v, list) else [safe_val(x) for x in v] for k, v in cal.items()}
            else:
                calendar_data = {}
        except Exception:
            calendar_data = {}

        # Earnings dates
        try:
            ed = ticker.get_earnings_dates(limit=12)
            earnings_dates = []
            if ed is not None and not ed.empty:
                for idx, row in ed.iterrows():
                    earnings_dates.append({
                        "date": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                        "epsEstimate": safe_val(row.get("EPS Estimate")),
                        "epsActual": safe_val(row.get("Reported EPS")),
                        "surprise": safe_val(row.get("Surprise(%)")),
                    })
        except Exception:
            earnings_dates = []

        # Dividends (last 10)
        try:
            divs = ticker.dividends
            dividends = []
            if divs is not None and len(divs) > 0:
                for dt, val in divs.tail(10).items():
                    dividends.append({
                        "date": dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
                        "amount": safe_val(val),
                    })
        except Exception:
            dividends = []

        # Splits
        try:
            sp = ticker.splits
            splits = []
            if sp is not None and len(sp) > 0:
                for dt, val in sp.tail(5).items():
                    splits.append({
                        "date": dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
                        "ratio": safe_val(val),
                    })
        except Exception:
            splits = []

        result = {
            "symbol": symbol,
            "calendar": calendar_data,
            "earnings_dates": earnings_dates,
            "dividends": dividends,
            "splits": splits,
        }
        set_cache(cache_key, result)
        return result

    except Exception as e:
        return {"error": str(e), "symbol": symbol}


# ─── /api/screener ───────────────────────────────────────────────
CACHE_TTL_SCREENER = 600  # 10 minutes


@app.get("/api/screener")
async def get_screener(screen: str = "most_actives"):
    """Run a predefined Yahoo Finance screener."""
    cache_key = f"screener_{screen}"
    cached = get_cached(cache_key, CACHE_TTL_SCREENER)
    if cached:
        return cached

    ALLOWED = [
        "aggressive_small_caps", "day_gainers", "day_losers",
        "growth_technology_stocks", "most_actives", "most_shorted_stocks",
        "small_cap_gainers", "undervalued_growth_stocks", "undervalued_large_caps",
        "conservative_foreign_funds", "high_yield_bond", "portfolio_anchors",
        "solid_large_growth_funds", "solid_midcap_growth_funds", "top_mutual_funds",
    ]
    if screen not in ALLOWED:
        return {"error": f"Unknown screen '{screen}'. Allowed: {ALLOWED}"}

    try:
        response = yf.screen(screen)
        quotes = response.get("quotes", []) if isinstance(response, dict) else []

        items = []
        for q in quotes[:30]:
            items.append({
                "symbol": q.get("symbol", ""),
                "shortName": q.get("shortName", q.get("longName", "")),
                "price": safe_val(q.get("regularMarketPrice")),
                "change": safe_val(q.get("regularMarketChange")),
                "changePct": safe_val(q.get("regularMarketChangePercent")),
                "volume": safe_val(q.get("regularMarketVolume")),
                "marketCap": safe_val(q.get("marketCap")),
                "exchange": q.get("exchange", ""),
            })

        result = {"screen": screen, "count": len(items), "items": items}
        set_cache(cache_key, result)
        return result

    except Exception as e:
        return {"error": str(e), "screen": screen, "items": []}


CACHE_TTL_HEATMAP = 60  # 60 seconds for near-live data

# Hardcoded NIFTY sector-index → constituent tickers mapping
# (nsetools get_index_quote is unreliable on servers, so we use yfinance bulk download)
SECTOR_INDICES = {
    "NIFTY 50": {
        "Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "INDUSINDBK.NS", "SHRIRAMFIN.NS"],
        "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIMindtree.NS"],
        "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "ADANIENSOL.NS", "BPCL.NS", "TATAPOWER.NS"],
        "Automobile": ["TATAMOTORS.NS", "M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS"],
        "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "GODREJCP.NS"],
        "Metals & Mining": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "ADANIENT.NS", "COALINDIA.NS"],
        "Pharma & Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS", "APOLLOHOSP.NS"],
        "Construction": ["ULTRACEMCO.NS", "GRASIM.NS", "ADANIPORTS.NS", "LT.NS"],
        "Telecom & Media": ["BHARTIARTL.NS"],
        "Consumer Durables": ["TITAN.NS", "ASIANPAINT.NS"],
    },
    "NIFTY BANK": {
        "Private Banks": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "AUBANK.NS"],
        "PSU Banks": ["SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS"],
    },
    "NIFTY IT": {
        "Large Cap IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
        "Mid Cap IT": ["LTIMindtree.NS", "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS", "LTTS.NS"],
    },
    "NIFTY PHARMA": {
        "Large Cap Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS"],
        "Mid Cap Pharma": ["LUPIN.NS", "BIOCON.NS", "TORNTPHARM.NS", "ALKEM.NS", "IPCALAB.NS"],
    },
    "NIFTY AUTO": {
        "Passenger Vehicles": ["TATAMOTORS.NS", "M&M.NS", "MARUTI.NS"],
        "Two Wheelers": ["BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "TVSMOTOR.NS"],
        "Auto Ancillaries": ["MOTHERSON.NS", "BOSCHLTD.NS", "BHARATFORG.NS", "MRF.NS"],
    },
    "NIFTY FMCG": {
        "Food & Beverages": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS"],
        "Personal Care": ["GODREJCP.NS", "DABUR.NS", "MARICO.NS", "COLPAL.NS"],
        "Others": ["PGHH.NS", "VBL.NS", "UBL.NS"],
    },
    "NIFTY ENERGY": {
        "Oil & Gas": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS"],
        "Power": ["NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "ADANIENSOL.NS", "ADANIGREEN.NS"],
    },
    "NIFTY METAL": {
        "Steel": ["TATASTEEL.NS", "JSWSTEEL.NS", "SAIL.NS", "JINDALSTEL.NS"],
        "Non-Ferrous": ["HINDALCO.NS", "VEDL.NS", "NATIONALUM.NS", "NMDC.NS"],
    },
    "NIFTY REALTY": {
        "Large Cap Realty": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS"],
        "Mid Cap Realty": ["PHOENIXLTD.NS", "BRIGADE.NS", "LODHA.NS", "SOBHA.NS"],
    },
}


@app.get("/api/heatmap")
async def get_heatmap(index: str = "NIFTY 50"):
    """
    Sector heatmap data for recharts Treemap.
    Uses yfinance bulk download for fast multi-ticker fetch.
    Returns nested sector → stock data with R-factor.
    """
    cache_key = f"heatmap_{index}"
    cached = get_cached(cache_key, CACHE_TTL_HEATMAP)
    if cached:
        return cached

    sector_map = SECTOR_INDICES.get(index)
    if not sector_map:
        return {"error": f"Unknown index '{index}'. Available: {list(SECTOR_INDICES.keys())}"}

    try:
        # Collect all tickers for bulk download
        all_tickers = []
        for tickers in sector_map.values():
            all_tickers.extend(tickers)

        # Remove duplicates while preserving order
        seen = set()
        unique_tickers = []
        for t in all_tickers:
            if t not in seen:
                seen.add(t)
                unique_tickers.append(t)

        # ── Bulk download via yfinance (single network call for all tickers) ──
        bulk_data = yf.download(
            tickers=unique_tickers,
            period="2d",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False,
        )

        # ── Fetch market-cap info in parallel via yf.Tickers ──
        tickers_obj = yf.Tickers(" ".join(unique_tickers))

        # Build per-ticker data
        ticker_data = {}
        for symbol in unique_tickers:
            try:
                # Get price data from bulk download
                if len(unique_tickers) == 1:
                    sym_data = bulk_data
                else:
                    sym_data = bulk_data[symbol] if symbol in bulk_data.columns.get_level_values(0) else None

                if sym_data is None or sym_data.empty or len(sym_data) < 1:
                    continue

                # Drop NaN rows
                sym_data = sym_data.dropna(subset=["Close"])
                if len(sym_data) < 1:
                    continue

                current_price = float(sym_data["Close"].iloc[-1])
                prev_close = float(sym_data["Close"].iloc[-2]) if len(sym_data) > 1 else current_price
                change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0

                # Get market cap from info (cached internally by yfinance)
                try:
                    info = tickers_obj.tickers[symbol].info
                    market_cap = info.get("marketCap", 0) or 0
                    short_name = info.get("shortName", symbol.replace(".NS", ""))
                except Exception:
                    market_cap = 1_000_000_000  # Default 1B if unavailable
                    short_name = symbol.replace(".NS", "")

                ticker_data[symbol] = {
                    "symbol": symbol.replace(".NS", ""),
                    "shortName": short_name,
                    "lastPrice": round(current_price, 2),
                    "prevClose": round(prev_close, 2),
                    "pChange": round(change_pct, 2),
                    "marketCap": market_cap,
                    "volume": int(sym_data["Volume"].iloc[-1]) if "Volume" in sym_data else 0,
                }
            except Exception:
                continue

        # ── Compute index-level average change for R-factor baseline ──
        all_changes = [d["pChange"] for d in ticker_data.values()]
        index_change = sum(all_changes) / len(all_changes) if all_changes else 0

        # ── Build nested sector structure for recharts Treemap ──
        sectors = []
        for sector_name, tickers in sector_map.items():
            children = []
            sector_changes = []
            for t in tickers:
                if t in ticker_data:
                    d = ticker_data[t]
                    sector_changes.append(d["pChange"])
                    # R-factor = stock change / index change
                    r_factor = round(d["pChange"] / index_change, 2) if index_change != 0 else 0
                    children.append({
                        "name": d["symbol"],
                        "shortName": d["shortName"],
                        "size": max(d["marketCap"], 1_000_000),  # Ensure minimum for visibility
                        "change": d["pChange"],
                        "lastPrice": d["lastPrice"],
                        "volume": d["volume"],
                        "rFactor": r_factor,
                        "marketCap": d["marketCap"],
                    })
            if children:
                sector_avg = round(sum(sector_changes) / len(sector_changes), 2) if sector_changes else 0
                sectors.append({
                    "name": sector_name,
                    "change": sector_avg,
                    "children": sorted(children, key=lambda x: x["size"], reverse=True),
                })

        result = {
            "index": index,
            "indexChange": round(index_change, 2),
            "sectors": sorted(sectors, key=lambda s: sum(c["size"] for c in s["children"]), reverse=True),
            "totalStocks": len(ticker_data),
            "timestamp": datetime.now().isoformat(),
            "availableIndices": list(SECTOR_INDICES.keys()),
        }
        set_cache(cache_key, result)
        return result

    except Exception as e:
        return {"error": str(e), "index": index}


if __name__ == "__main__":
    import uvicorn
    print("Starting Fintola Finance API on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
