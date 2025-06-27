import pandas as pd
import numpy as np
from pycoingecko import CoinGeckoAPI
from datetime import datetime
from binance.spot import Spot
from sktime.transformations.series.outlier_detection import HampelFilter
from kats.consts import TimeSeriesData
import talib
import requests

cg = CoinGeckoAPI()
def get_ohlc(coin, days):
    ohlc=cg.get_coin_ohlc_by_id(
        id=coin, vs_currency="usd", days=days
    )
    ohlc_df=pd.DataFrame(ohlc)
    ohlc_df.columns=["date","open","high","low","close"]
    ohlc_df["date"]=pd.to_datetime(ohlc_df["date"], unit="ms")
    ohlc_df["simple_return"]=ohlc_df["close"].pct_change()
    ohlc_df["log_return"]=np.log(ohlc_df["close"]/ohlc_df["close"].shift(1))
    return ohlc_df
#Daily Candles
ohlc_df=get_ohlc("bitcoin", 365)
#Hourly Candles
ohlc_df_intraday = get_ohlc("bitcoin", 7)

def find_outlier(df, window_size, n_sigmas, column):
    df_close = df[[column]].copy()
    df_rolling = df_close.rolling(window=window_size)\
        .agg(["mean", "std"])
    df_rolling.columns=df_rolling.columns.droplevel()
    df_ohlc2 = df.join(df_rolling)

    df_ohlc2["upper"] = df_ohlc2["mean"]+n_sigmas*df_ohlc2["std"]
    df_ohlc2["lower"] = df_ohlc2["mean"]-n_sigmas*df_ohlc2["std"]

    df_ohlc2["outlier"] = ((df_ohlc2["upper"] < df_ohlc2[column]) | (df_ohlc2["lower"] > df_ohlc2[column]))
    return df_ohlc2

df_ohlc2 = find_outlier(ohlc_df, 20, 3, "close")
mask = df_ohlc2["outlier"] == True
df_ohlc2.loc[mask, df_ohlc2.columns != "outlier"] = np.nan
df_ohlc2.drop(columns = ["mean","std"], inplace=True)
def hampel_outlier_detection(df, window_size, column):
    hampel_detector = HampelFilter(window_length=window_size,
                                    return_bool=True)
    df["outliers"] = hampel_detector.fit_transform(df[column])
hampel_outlier_detection(df_ohlc2, 20, "close")
df_ohlc_adjusted = df_ohlc2["close"].interpolate()

#spot_client=Spot(base_url="https://api3.binance.com")
#r = spot_client.trades("BTCUSDT")
url = "https://api.binance.com/api/v3/trades"
params = {"symbol": "BTCUSDT", "limit": 1000}
response = requests.get(url, params=params)
response.raise_for_status()
r = response.json()

df2 = (pd.DataFrame(r)
    .drop(["isBuyerMaker", "isBestMatch"], axis=1)
)
df2["time"] = pd.to_datetime(df2["time"], unit="ms")
for column in ["price", "qty", "quoteQty"]:
    df2[column] = pd.to_numeric(df2[column])

def get_bars(df):
    ohlc = df["price"].agg(open="first", high="max", low="min", close="last")
    vwap = (
    df.apply(lambda x: np.average(x["price"], weights=x["qty"]))
        .to_frame("vwap")
    )
    vol = df["qty"].sum().to_frame("vol")
    cnt = df["qty"].size().to_frame("cnt")
    orig_data = pd.concat([ohlc, vwap, vol, cnt], axis=1)
    return orig_data

#Get Tick Bars
def get_tick_bars(df, bar_size):
    df["tick_bars"] = (
        pd.Series(list(range(len(df))), index=df.index)
        .div(bar_size)
        .astype(int)
        .values
    )
    df_grouped_ticks = df.groupby("tick_bars")
    return get_bars(df_grouped_ticks)

#Get Volume Bars
def get_volume_bars(df, bar_size):
    df["cum_qty"] = df["qty"].cumsum()
    df["volume_bars"] = (
        df["cum_qty"]
        .div(bar_size)
        .astype(int)
        .values
    )
    df_grouped_vol = df.groupby("volume_bars")
    return get_bars(df_grouped_vol)

#Get Dollar Bars
def get_dollar_bars(df, bar_size):
    df["cum_quoteQty"] = df["quoteQty"].cumsum()
    df["dollar_bars"] = (
        df["cum_quoteQty"]
        .div(bar_size)
        .astype(int)
        .values
    )
    df_grouped_dollar_bars = df.groupby("dollar_bars")
    return get_bars(df_grouped_dollar_bars)

# Find Hurst Exponent
# Get Timeseriesdata
def get_timeseries(df):
    ohlc_df3 = df[["close"]].reset_index(drop=False)
    ohlc_df3.columns = ["time", "value"]
    tsd = TimeSeriesData(ohlc_df3)
    return tsd
tsd = get_timeseries(ohlc_df)
tsd_intraday = get_timeseries(ohlc_df_intraday)

def get_hurst_exponent(ts, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst_exp = poly[0]
    return hurst_exp


hurst_exp = get_hurst_exponent(tsd["value"], 20)
hurst_exp_intraday = get_hurst_exponent(tsd_intraday["value"], 20)



def find_sma(df, name, time_period):
    df[name] = talib.SMA(df["close"].to_numpy(), timeperiod=time_period)


for time in 20, 50, 100:
    find_sma(ohlc_df, f"SMA_{time}", time)

def find_bbands(df):
    df["BB_high"], df["BB_mid"], df["BB_low"] = talib.BBANDS(df["close"].to_numpy())
    return df
ohlc_df = find_bbands(ohlc_df)
ohlc_df_intraday = find_bbands(ohlc_df_intraday)

def find_rsi(df):
    df["rsi"] = talib.RSI(df["close"].to_numpy())
    return df
ohlc_df = find_rsi(ohlc_df)
ohlc_df_intraday = find_rsi(ohlc_df_intraday)

def find_macd(df):
    df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(df["close"].to_numpy(), slowperiod=12,
                                                                           fastperiod=26, signalperiod=9)
    return df
ohlc_df = find_macd(ohlc_df)
ohlc_df_intraday = find_macd(ohlc_df_intraday)

created_candles = []
created_candles_intraday = []

def get_candles(df, list_of_candles):
    candle_name = talib.get_function_groups()["pattern recognition"]
    for candle in candle_name:
        df[candle] = getattr(talib, candle)(df["open"], df["high"], df["low"], df["close"])
        list_of_candles.append(candle)
    return df
get_candles(ohlc_df, created_candles)


trade_signals = []
trade_signals_intraday = []
recent_movement = np.mean(ohlc_df["log_return"].head(20))
recent_movement_intraday = np.mean(ohlc_df_intraday["log_return"].head(20))

#find hurst exponent for mean reversion
def hurst_exponent_signal(hurst_exp, recent_movement, trade_signals, mean_reversion_change, momentum_change):
    if hurst_exp < 0.40:
        if recent_movement > mean_reversion_change:
            trade_signals.append(-10)
        elif recent_movement < -mean_reversion_change:
            trade_signals.append(10)
    elif hurst_exp > 0.60:
        if recent_movement > momentum_change:
            trade_signals.append(10)
        elif recent_movement < -momentum_change:
            trade_signals.append(-10)
    else:
        print("no hurst exponent")
hurst_exponent_signal(hurst_exp, recent_movement, trade_signals, 0.075, 0.05)
hurst_exponent_signal(hurst_exp_intraday, recent_movement_intraday, trade_signals_intraday, 0.01, 0.005)

#find RSI signal
def find_rsi_signal(df, trade_signals):
    if df["rsi"].iloc[-1] > 70:
        trade_signals.append(-1)
    elif ohlc_df["rsi"].iloc[-1] < 30:
        trade_signals.append(1)

find_rsi_signal(ohlc_df, trade_signals)
find_rsi_signal(ohlc_df_intraday, trade_signals_intraday)

#find macd signal
def find_macd_signal(df, trade_signals):
    macd = df["macd"]
    macd_signal = df["macd_signal"]

    macd_down = (macd.shift(1)>macd_signal.shift(1)) & (macd<macd_signal)
    macd_up = (macd.shift(1)<macd_signal.shift(1)) & (macd>macd_signal)

    if macd_down.iloc[-1]:
        trade_signals.append(-1)
    elif macd_up.iloc[-1]:
        trade_signals.append(1)
    else:
        print("No MACD Indicator")
find_macd_signal(ohlc_df, trade_signals)
find_macd_signal(ohlc_df_intraday, trade_signals_intraday)

#find sma signal
def find_sma_signal(df, trade_signals):
    sma_df = df[["SMA_20", "SMA_50", "SMA_100"]]
    if df["close"].iloc[-1] == sma_df["SMA_20"][-1]:
        trade_signals.append(1)
    elif df["close"].iloc[-1] == sma_df["SMA_50"][-1]:
        trade_signals.append(1)
    elif df["close"].iloc[-1] == sma_df["SMA_100"][-1]:
        trade_signals.append(1)
    else:
        print("No SMA Indicator")
find_sma_signal(ohlc_df, trade_signals)
find_sma_signal(ohlc_df_intraday, trade_signals_intraday)


#find vwap signal


trade_entry = []   
results = []

for candle in created_candles:
    avg_100 = ohlc_df.loc[ohlc_df[candle] == 100, "log_return"].mean()
    avg_neg100 = ohlc_df.loc[ohlc_df[candle] == -100, "log_return"].mean()
    results.append({
    "candle": candle,
    "avg_log_return_100": avg_100,
    "avg_log_return_neg100": avg_neg100
    })
results_df = pd.DataFrame(results)

get_candles(ohlc_df_intraday, created_candles_intraday)
for candle in created_candles_intraday:
    if candle == 100:
        trade_entry.append(results_df.loc[results_df[candle], "avg_log_return_100:"]) 
    elif candle == -100:
        trade_entry.append(results_df.loc[results_df[candle], "avg_log_return_neg100"])

print(trade_signals)
print(trade_signals_intraday)
print(trade_entry)

#Generate Trade Signals
if sum(trade_signals) >= 12 and sum(trade_signals_intraday) >= 11:
    if sum(trade_entry) >= 0.5:
        print("Buy")
elif sum(trade_signals) <= -12 and sum(trade_signals_intraday) <= -11:
    if sum(trade_entry) <= -0.5:
        print("Sell")
else:
    print("No Trade")
