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
async def get_chart(symbol: str = "RELIANCE.NS", period: str = "1y", interval: str = "1h"):
    """Get historical chart data for a symbol"""
    cache_key = f"chart_{symbol}_{period}_{interval}"
    cached = get_cached(cache_key, CACHE_TTL_CHART)
    if cached:
        return cached

    try:
        ticker = yf.Ticker(symbol)
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
async def get_finance(symbol: str = "RELIANCE.NS", type: str = "chart"):
    """Combined endpoint - use type=quote for current price, type=chart for historical"""
    if type == "quote":
        return await get_quote(symbol)
    else:
        return await get_chart(symbol)


if __name__ == "__main__":
    import uvicorn
    print("Starting Fintola Finance API on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
