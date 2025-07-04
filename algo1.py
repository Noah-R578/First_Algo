import pandas as pd
import numpy as np
from pycoingecko import CoinGeckoAPI
from datetime import datetime
from binance.spot import Spot
from sktime.transformations.series.outlier_detection import HampelFilter
from kats.consts import TimeSeriesData
import talib
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import time
import requests

# Load API keys from config file
try:
    from config import CRYPTOCOMPARE_API_KEY
except ImportError:
    print("Warning: config.py not found. Please create config.py with your API keys.")
    CRYPTOCOMPARE_API_KEY = "YOUR_CRYPTOCOMPARE_API_KEY_HERE"

def get_crypto_hourly(fsym, tsym, limit=2000, toTs=None):
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {
        'fsym': fsym,
        'tsym': tsym,
        'limit': limit,
        'api_key': CRYPTOCOMPARE_API_KEY
    }
    if toTs:
        params['toTs'] = toTs
    
    response = requests.get(url, params=params)
    data = response.json()

    if data['Response'] != 'Success':
        raise Exception(f"API Error: {data.get('Message', 'No message')}")

    ohlc_data = data['Data']['Data']
    df = pd.DataFrame(ohlc_data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def get_crypto_daily(fsym, tsym, limit=2000, toTs=None):
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        'fsym': fsym,
        'tsym': tsym,
        'limit': limit,
        'api_key': CRYPTOCOMPARE_API_KEY
    }
    if toTs:
        params['toTs'] = toTs
    
    response = requests.get(url, params=params)
    data = response.json()

    if data['Response'] != 'Success':
        raise Exception(f"API Error: {data.get('Message', 'No message')}")

    ohlc_data = data['Data']['Data']
    df = pd.DataFrame(ohlc_data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def find_outlier(df, window_size, n_sigmas, column):
    df_close = df[[column]].copy()
    df_rolling = df_close.rolling(window=window_size)\
        .agg(["mean", "std"])
    df_rolling.columns=df_rolling.columns.droplevel()
    df_ohlc2 = df.join(df_rolling)
    df_ohlc2["upper"] = df_ohlc2["mean"]+n_sigmas*df_ohlc2["std"]
    df_ohlc2["lower"] = df_ohlc2["mean"]-n_sigmas*df_ohlc2["std"]
    df_ohlc2["outlier"] = ((df_ohlc2[column] > df_ohlc2["upper"]) | (df_ohlc2[column] < df_ohlc2["lower"]))
    return df_ohlc2

def hampel_outlier_detection(df, window_size, column):
    df_for_hampel = df[[column]].copy()
    hampel_detector = HampelFilter(window_length=window_size, return_bool=True)
    df["outliers"] = hampel_detector.fit_transform(df_for_hampel)
    return df

#def get_bars(df):
#    ohlc = df["price"].agg(open="first", high="max", low="min", close="last")
#    vwap = (
#    df.apply(lambda x: np.average(x["price"], weights=x["qty"]))
#        .to_frame("vwap")
#    )
#    vol = df["qty"].sum().to_frame("vol")
#    cnt = df["qty"].size().to_frame("cnt")
#    orig_data = pd.concat([ohlc, vwap, vol, cnt], axis=1)
#    return orig_data

#Get Tick Bars
#def get_tick_bars(df, bar_size):
#    df["tick_bars"] = (
#        pd.Series(list(range(len(df))), index=df.index)
#        .div(bar_size)
#        .astype(int)
#        .values
#    )
#    df_grouped_ticks = df.groupby("tick_bars")
#    return get_bars(df_grouped_ticks)

#Get Volume Bars
#def get_volume_bars(df, bar_size):
#    df["cum_qty"] = df["qty"].cumsum()
#    df["volume_bars"] = (
#        df["cum_qty"]
#        .div(bar_size)
#        .astype(int)
#        .values
#    )
#    df_grouped_vol = df.groupby("volume_bars")
#    return get_bars(df_grouped_vol)

#Get Dollar Bars
#def get_dollar_bars(df, bar_size):
#    df["cum_quoteQty"] = df["quoteQty"].cumsum()
#    df["dollar_bars"] = (
#        df["cum_quoteQty"]
#        .div(bar_size)
#        .astype(int)
#        .values
#    )
#    df_grouped_dollar_bars = df.groupby("dollar_bars")
#    return get_bars(df_grouped_dollar_bars)

def get_timeseries(df):
    ohlc_df3 = df[["close"]].reset_index(drop=False)
    ohlc_df3.columns = ["time", "value"]
    tsd = TimeSeriesData(ohlc_df3)
    return tsd

def get_hurst_exponent(ts, max_lag=20):
    # Convert to numpy array if it's a pandas Series
    if hasattr(ts, 'values'):
        ts = ts.values
    
    # Remove NaN values
    ts = ts[~np.isnan(ts)]
    
    # Check if we have enough data
    if len(ts) < max_lag:
        print(f"Warning: Not enough data points ({len(ts)}) for max_lag ({max_lag})")
        return 0.5
    
    lags = range(2, max_lag)
    tau = []
    
    for lag in lags:
        # Calculate price differences
        price_diff = np.subtract(ts[lag:], ts[:-lag])
        # Remove any NaN values from the difference
        price_diff = price_diff[~np.isnan(price_diff)]
        if len(price_diff) > 0:
            tau.append(np.std(price_diff))
        else:
            tau.append(0)
    
    # Filter out zero values to avoid log(0) error
    valid_indices = [i for i, t in enumerate(tau) if t > 0]
    
    if len(valid_indices) < 2:
        print(f"Warning: Not enough valid tau values ({len(valid_indices)})")
        return 0.5
    
    valid_lags = [lags[i] for i in valid_indices]
    valid_tau = [tau[i] for i in valid_indices]
    
    # Calculate Hurst exponent
    try:
        hurst_exp = np.polyfit(np.log(valid_lags), np.log(valid_tau), 1)[0]
        return hurst_exp
    except Exception as e:
        print(f"Error calculating Hurst exponent: {e}")
        return 0.5

def find_sma(df, name, time_period):
    df[name] = talib.SMA(df["close"].to_numpy(), timeperiod=time_period)

def find_bbands(df):
    df["BB_high"], df["BB_mid"], df["BB_low"] = talib.BBANDS(df["close"].to_numpy())
    return df

def find_rsi(df):
    df["rsi"] = talib.RSI(df["close"].to_numpy())
    return df

def find_macd(df):
    df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(df["close"].to_numpy(), slowperiod=12,
                                                                           fastperiod=26, signalperiod=9)
    return df

#find stationarity with ADF
def adf_test(df):
    indices = ["ADF Test Statistic", "p-value", "Lags Used", "Number of Observations Used"]
    adf_result = adfuller(df["close"], autolag="AIC")
    results=pd.Series(adf_result[0:4], index=indices)
    
    # Safely handle critical values
    if len(adf_result) > 4 and hasattr(adf_result[4], 'items'):
        for key, value in adf_result[4].items():
            results[f"critical value ({key})"] = value
    
    return results["p-value"] < 0.05

#find stationarity with KPSS
def kpss_test(df):
    try:
        indices = ["KPSS Test Statistic", "p-value", "Lags Used"]
        kpss_result = kpss(df["close"], regression="c", nlags="auto")
        results=pd.Series(kpss_result[0:3], index=indices)
        # Handle critical values safely
        if len(kpss_result) > 3 and hasattr(kpss_result[3], 'items'):
            for key, value in kpss_result[3].items():
                results[f"critical value ({key})"] = value
        return results["p-value"] > 0.05
    except Exception as e:
        print(f"KPSS test error: {e}")
        return False

#find hurst exponent for mean reversion
def hurst_exponent_signal(hurst_exp, recent_movement, trade_signals, mean_reversion_change, momentum_change, stationarity):
    if hurst_exp < 0.40 and stationarity == True:
        if recent_movement > mean_reversion_change:
            trade_signals.append(-10)
        elif recent_movement < -mean_reversion_change:
            trade_signals.append(10)
    elif hurst_exp > 0.60 and stationarity == False:
        if recent_movement > momentum_change:
            trade_signals.append(10)
        elif recent_movement < -momentum_change:
            trade_signals.append(-10)
    else:
        print("no hurst exponent")

#find RSI signal
def find_rsi_signal(df, trade_signals):
    if "rsi" not in df.columns:
        print("RSI column not found")
        return
        
    current_rsi = df["rsi"].iloc[-1]
    if pd.isna(current_rsi):
        print("RSI value is NaN")
        return
        
    if current_rsi > 70:
        trade_signals.append(-1)
    elif current_rsi < 30:
        trade_signals.append(1)
    else:
        print("No RSI signal")

#find macd signal
def find_macd_signal(df, trade_signals):
    if "macd" not in df.columns or "macd_signal" not in df.columns:
        print("MACD columns not found")
        return
        
    macd = df["macd"]
    macd_signal = df["macd_signal"]

    # Check if we have enough data
    if len(macd) < 2:
        print("Not enough data for MACD signal")
        return

    macd_down = (macd.shift(1)>macd_signal.shift(1)) & (macd<macd_signal)
    macd_up = (macd.shift(1)<macd_signal.shift(1)) & (macd>macd_signal)

    if macd_down.iloc[-1]:
        trade_signals.append(-1)
    elif macd_up.iloc[-1]:
        trade_signals.append(1)
    else:
        print("No MACD Indicator")

#find sma signal
def find_sma_signal(df, trade_signals):
    # Check if SMA columns exist
    sma_columns = ["SMA_20", "SMA_50", "SMA_100"]
    existing_sma_columns = [col for col in sma_columns if col in df.columns]
    
    if not existing_sma_columns:
        print("No SMA columns found")
        return
        
    sma_df = df[existing_sma_columns]
    current_close = df["close"].iloc[-1]
    previous_close = df["close"].iloc[-2] if len(df) > 1 else current_close
    
    # Check for SMA crossover
    for sma_col in existing_sma_columns:
        sma_value = sma_df[sma_col].iloc[-1]
        previous_sma_value = sma_df[sma_col].iloc[-2] if len(sma_df) > 1 else sma_value
        
        if not pd.isna(sma_value) and not pd.isna(previous_sma_value):
            # bullish sma crossover
            if previous_close <= previous_sma_value and current_close > sma_value:
                trade_signals.append(1)
                return
            # bearish sma crossover
            elif previous_close >= previous_sma_value and current_close < sma_value:
                trade_signals.append(-1)
                return
            else: print(f"No SMA Crossover for {sma_col}")

def get_candles(df, list_of_candles):
    candle_name = talib.get_function_groups()["Pattern Recognition"]
    for candle in candle_name:
        df[candle] = getattr(talib, candle)(df["open"], df["high"], df["low"], df["close"])
        list_of_candles.append(candle)
    return df

def get_results(df, list_of_candles):
    results = []
    for candle in list_of_candles:
        avg_100 = df.loc[df[candle] == 100, "log_return"].mean()
        avg_neg100 = df.loc[df[candle] == -100, "log_return"].mean()
        results.append({
        "candle": candle,
        "avg_log_return_100": avg_100,
        "avg_log_return_neg100": avg_neg100
        })
    results_df = pd.DataFrame(results)
    return results_df

def get_trade_entry_signals(df, results_df):
    trade_entry = []
    last_idx = df.index[-1]
    for _, row in results_df.iterrows():
        pattern = row["candle"]
        pattern_value = df.loc[last_idx, pattern]
        if pattern_value == 100:
            if not np.isnan(row["avg_log_return_100"]):
                trade_entry.append(row["avg_log_return_100"])
        elif pattern_value == -100:
            if not np.isnan(row["avg_log_return_neg100"]):
                trade_entry.append(row["avg_log_return_neg100"])
    return trade_entry

def find_trade(coin):
    #hourly candles
    ohlc_df_intraday = get_crypto_hourly(coin, 'USD', limit=167)
    print(len(ohlc_df_intraday))

    #Daily Candles
    ohlc_df = get_crypto_daily(coin, "USD", limit=365)
    print(len(ohlc_df))

    for col in ["close", "open", "high", "low"]:
        if isinstance(ohlc_df[col], pd.Series):
            ohlc_df[col] = ohlc_df[col].interpolate()
            ohlc_df[col] = ohlc_df[col].fillna(method="ffill").fillna(method="bfill")
        if isinstance(ohlc_df_intraday[col], pd.Series):
            ohlc_df_intraday[col] = ohlc_df_intraday[col].interpolate()
            ohlc_df_intraday[col] = ohlc_df_intraday[col].fillna(method="ffill").fillna(method="bfill")

    ohlc_df["log_return"] = np.log(ohlc_df["close"] / ohlc_df["close"].shift(1))
    ohlc_df_intraday["log_return"] = np.log(ohlc_df_intraday["close"] / ohlc_df_intraday["close"].shift(1))

    print("Length of daily close series:", len(ohlc_df["close"]))
    print("First 10 values:", ohlc_df["close"].head(10))
    print("Last 10 values:", ohlc_df["close"].tail(10))

    df_ohlc2 = find_outlier(ohlc_df, 20, 3, "close")
    mask = df_ohlc2["outlier"] == True
    df_ohlc2.loc[mask, df_ohlc2.columns != "outlier"] = np.nan
    df_ohlc2.drop(columns = ["mean","std"], inplace=True)

    hampel_outlier_detection(df_ohlc2, 20, "close")
    df_ohlc_adjusted = df_ohlc2["close"].interpolate()

    #spot_client=Spot(base_url="https://api3.binance.com")
    #r = spot_client.trades(coin + "USDT")

    #df2 = (pd.DataFrame(r)
    #    .drop(["isBuyerMaker", "isBestMatch"], axis=1)
    #)
    #df2["time"] = pd.to_datetime(df2["time"], unit="ms")
    #for column in ["price", "qty", "quoteQty"]:
    #    df2[column] = pd.to_numeric(df2[column])

    # Find Hurst Exponent
    # Get Timeseriesdata
    tsd = get_timeseries(ohlc_df)
    tsd_intraday = get_timeseries(ohlc_df_intraday)

    hurst_exp = get_hurst_exponent(tsd.to_dataframe()["value"], 20)
    hurst_exp_intraday = get_hurst_exponent(tsd_intraday.to_dataframe()["value"], 20)

    for time in 20, 50, 100:
        find_sma(ohlc_df, f"SMA_{time}", time)
        find_sma(ohlc_df_intraday, f"SMA_{time}", time)

    ohlc_df = find_bbands(ohlc_df)
    ohlc_df_intraday = find_bbands(ohlc_df_intraday)

    ohlc_df = find_rsi(ohlc_df)
    ohlc_df_intraday = find_rsi(ohlc_df_intraday)

    ohlc_df = find_macd(ohlc_df)
    ohlc_df_intraday = find_macd(ohlc_df_intraday)

    trade_signals = []
    trade_signals_intraday = []
    recent_movement = np.mean(ohlc_df["log_return"].head(20))
    recent_movement_intraday = np.mean(ohlc_df_intraday["log_return"].head(20))

    stationary_adf = adf_test(ohlc_df)
    stationary_adf_intraday = adf_test(ohlc_df_intraday)

    stationary_kpss = kpss_test(ohlc_df)
    stationary_kpss_intraday = kpss_test(ohlc_df_intraday)
    if stationary_adf == True and stationary_kpss == True:
        stationarity = True
    elif stationary_adf == False and stationary_kpss == False:
        stationarity = False
    else:
        print("Stationarity not found")

    if stationary_adf_intraday == True and stationary_kpss_intraday == True:
        stationarity_intraday = True
    elif stationary_adf_intraday == False and stationary_kpss_intraday == False:
        stationarity_intraday = False
    else:
        print("Stationarity not found")
        stationarity_intraday = None
    # Print Hurst exponent values
    print(f"Daily Hurst Exponent: {hurst_exp:.3f}")
    print(f"Intraday Hurst Exponent: {hurst_exp_intraday:.3f}")
    print(f"Daily Stationarity: {stationarity}")
    print(f"Intraday Stationarity: {stationarity_intraday}")

    hurst_exponent_signal(hurst_exp, recent_movement, trade_signals, 0.075, 0.05, stationarity)
    hurst_exponent_signal(hurst_exp_intraday, recent_movement_intraday, trade_signals_intraday, 0.01, 0.005, stationarity_intraday)

    find_rsi_signal(ohlc_df, trade_signals)
    find_rsi_signal(ohlc_df_intraday, trade_signals_intraday)

    find_macd_signal(ohlc_df, trade_signals)
    find_macd_signal(ohlc_df_intraday, trade_signals_intraday)

    find_sma_signal(ohlc_df, trade_signals)
    find_sma_signal(ohlc_df_intraday, trade_signals_intraday)

    trade_entry = []
    results = []
    created_candles = []
    created_candles_intraday = []

    get_candles(ohlc_df, created_candles)
    get_candles(ohlc_df_intraday, created_candles_intraday)

    results_df = get_results(ohlc_df, created_candles)
    results_df_intraday = get_results(ohlc_df_intraday, created_candles_intraday)

    trade_entry = get_trade_entry_signals(ohlc_df, results_df)
    trade_entry_intraday = get_trade_entry_signals(ohlc_df_intraday, results_df_intraday)

    print(trade_signals)
    print(trade_signals_intraday)
    print(trade_entry)
    print(trade_entry_intraday)

    #Generate Trade Signals
    if sum(trade_signals) >= 11 and sum(trade_signals_intraday) >= 1:
        if sum(trade_entry) >= 0.1 or sum(trade_entry_intraday) >= 0.1:
            print("Buy")
    elif sum(trade_signals) <= -11 and sum(trade_signals_intraday) <= -1:
        if sum(trade_entry) <= -0.1 or sum(trade_entry_intraday) <= -0.1:
            print("Sell")
    else:
        print("No Trade")
# Top 10 cryptocurrencies by market cap
top_10_coins = [
    'BTC',   
    'ETH',   
    'BNB',   
    'SOL',   
    'XRP',   
    'ADA',   
    'AVAX',  
    'DOGE',  
    'DOT',   
    'MATIC'  
]

for coin in top_10_coins:
    try:
        find_trade(coin)
    except Exception as exception:
        print(f"Error analyzing {coin}: {exception}")
    
