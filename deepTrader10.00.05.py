import os  # file paths, dirs, etc.
import sys  # argv + exit stuff
import json
import logging  # logging everywhere (this gets noisy fast)
import argparse  # read CLI flags like
import importlib  # used for dependency checks
import subprocess
import numpy as np  # core math + indicators
import pandas as pd  # dataframe heavy lifting
import torch  # main ML framework
import torch.nn as nn  # layers + model defs
import torch.optim as optim  # optimizers (Adam mostly)
from torch.utils.data import DataLoader, TensorDataset  # batching
from alpaca.data import StockHistoricalDataClient, TimeFrame, TimeFrameUnit, NewsClient  # market data + news
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.enums import DataFeed  # IEX (free) vs SIP (paid) feed selector
from alpaca.trading.client import TradingClient  # trading interface
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus  # OrderStatus needed for filled order check
from alpaca.common.exceptions import APIError  # API tends to throw these
from transformers import pipeline  # sentiment model 
from sklearn.preprocessing import RobustScaler, StandardScaler  # scaling
import smtplib  # email alerts
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime, timedelta, timezone  # time handling everywhere
import talib  # technical indicators 
import pickle  # caching models/data
from typing import List, Tuple, Dict, Optional, Any  # type hints (not always consistent)
import warnings  # suppress some annoying spam
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type  # retry logic for API calls
from tqdm import tqdm  # progress bars (mainly for training)
from colorama import Fore, Style  # colored console output
import colorama  # init required on Windows
import multiprocessing as mp  # parallel symbol training
import time  # timing + sleeps (also used in caching)
import shutil  # file ops
import tempfile  # temp model checkpoints
from alpaca.trading.requests import GetOrdersRequest  # used in account reset + position tracking
from alpaca.trading.enums import QueryOrderStatus


import requests.exceptions as _req_exc
import urllib3.exceptions as _u3_exc
import matplotlib.pyplot as plt  # plotting results
import statsmodels.tsa.stattools as ts  # cointegration test
from statsmodels.regression.linear_model import OLS  # hedge ratio calc
from hmmlearn.hmm import GaussianHMM  # regime detection
import xgboost as xgb  # ensemble model
from torch.utils.checkpoint import checkpoint  # saves VRAM, slows things a bit
import threading

# CUDA performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Initialize colorama for colored console output
colorama.init()


RETRYABLE_ERRORS = (
    APIError,
    _req_exc.ConnectionError,
    _req_exc.Timeout,
    _req_exc.ReadTimeout,
    _req_exc.ConnectTimeout,
    _u3_exc.ProtocolError,
    _u3_exc.ReadTimeoutError,
    _u3_exc.NewConnectionError,
    ConnectionResetError,
    ConnectionAbortedError,
)

CONFIG = {

    'SYMBOLS': [ 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'AMD', 'SPY', 'QQQ' ],  # List of stock symbols to trade
    'PAIRS': [
        ('AAPL', 'MSFT'), ('GOOGL', 'AMZN'),
        ('NVDA', 'AMD'), ('SPY', 'QQQ')
    ],
    'TIMEFRAME': TimeFrame(15, TimeFrameUnit.Minute),  # Time interval for data fetching
    'INITIAL_CASH': 100000.00,  # Starting cash for trading simulation
    'MIN_HOLDING_PERIOD_MINUTES': 30,  # Minimum holding period before exits are evaluated.
                                        # Lowered from 45 to 30 to allow exits closer to the


    'TRAIN_DATA_START_DATE': '2015-01-01',  # Start date for training data
    'TRAIN_END_DATE': '2025-12-31',
    'VAL_START_DATE': '2026-01-01',
    'VAL_END_DATE': '2026-02-28',
    'BACKTEST_START_DATE': '2026-03-01',
    'BACKTEST_END_DATE': None,


    'WALK_FORWARD_FOLDS': [
        {'tag': '2020_covid',   'train_end': '2019-10-31', 'val_start': '2019-11-01', 'val_end': '2019-12-31', 'test_start': '2020-01-01', 'test_end': '2020-12-31'},
        {'tag': '2021_bull',    'train_end': '2020-10-31', 'val_start': '2020-11-01', 'val_end': '2020-12-31', 'test_start': '2021-01-01', 'test_end': '2021-12-31'},
        {'tag': '2022_bear',    'train_end': '2021-10-31', 'val_start': '2021-11-01', 'val_end': '2021-12-31', 'test_start': '2022-01-01', 'test_end': '2022-12-31'},
        {'tag': '2023_recover', 'train_end': '2022-10-31', 'val_start': '2022-11-01', 'val_end': '2022-12-31', 'test_start': '2023-01-01', 'test_end': '2023-12-31'},
        {'tag': '2024_bull',    'train_end': '2023-10-31', 'val_start': '2023-11-01', 'val_end': '2023-12-31', 'test_start': '2024-01-01', 'test_end': '2024-12-31'},
        {'tag': '2025_recent',  'train_end': '2024-10-31', 'val_start': '2024-11-01', 'val_end': '2024-12-31', 'test_start': '2025-01-01', 'test_end': '2025-12-31'},
    ],

    'SIMULATION_DAYS': 180,  # Number of days for simulation
    'MIN_DATA_POINTS': 100,  # Minimum data points required for processing
    'CACHE_DIR': './cache',  # Directory for caching data
    'MODEL_CACHE_DIR': '/mnt/c/Users/aipla/Desktop/Model Weights',
    'CACHE_EXPIRY_SECONDS': 24 * 60 * 60,  # Expiry time for cached data in seconds
    'LIVE_DATA_BARS': 1200,  # Number of bars to fetch for live data


    'LIVE_FEATURE_MIN_BARS': 600,


    'TRAIN_EPOCHS': 50,  # Number of epochs for training the model
    'BATCH_SIZE': 8192,
    'TIMESTEPS': 30,  # Number of time steps for sequence data
    'EARLY_STOPPING_MONITOR': 'val_loss',  # Metric to monitor for early stopping
    'EARLY_STOPPING_PATIENCE': 6,

    'EARLY_STOPPING_MIN_DELTA': 0.00001,  # Reduced min delta to detect smaller improvements
    'LEARNING_RATE': 0.001,
    'WEIGHT_DECAY': 1e-3,
    'LR_SCHEDULER_PATIENCE': 5,  # Patience for ReduceLROnPlateau
    'LR_REDUCTION_FACTOR': 0.5,  # Factor to multiply LR by
    'LOOK_AHEAD_BARS': 21,
    'NUM_PARALLEL_WORKERS': 8,
    'GB_PER_WORKER_EST': 2.0,


    # NEW: HMM Regime Detection
    'NUM_REGIMES': 6,


    'HMM_N_ITER': 100,


    'HMM_FIT_DEDUP': True,        # Fit the HMM on the de-duplicated, time-ORDERED bar


                                   # Markov transitions at every window boundary).
    'HMM_MAX_FIT_ROWS': 150000,

    'XGBOOST_DEVICE': 'cuda',


    'XGBOOST_N_JOBS': 0,


                                   # how many workers VRAM allows.

    'BLEND_LSTM_WEIGHT': 0.6,


    'MAX_POS_WEIGHT': 8.0,

    'CPU_THREADS_PER_WORKER': 0,


    'ALPACA_API_KEY': 'PKXROHOFWDFC7OFFXQCBRDWVMU',  # API key for Alpaca
    'ALPACA_SECRET_KEY': 'CT25NcFuH7UtkPtut4QVLfBzk8j1juDevVnNpXgLpwgC',  # Secret key for Alpaca


    'EMAIL_SENDER': 'alpaca.ai.tradingbot@gmail.com',  # Email address for sending notifications
    'EMAIL_PASSWORD': 'hjdf sstp pyne rotq',  # Password for the email account
    'EMAIL_RECEIVER': ['aiplane.scientist@gmail.com', 'vmakarov28@students.d125.org', 'tchaikovskiy@hotmail.com'],  # List of email recipients
    'SMTP_SERVER': 'smtp.gmail.com',  # SMTP server for email
    'SMTP_PORT': 587,  # Port for SMTP server


    'LOG_FILE': 'trades.log',  # File for logging trades
    
    # Strategy Thresholds — BEST FROM 10-HOUR OPTIMIZER
    'CONFIDENCE_THRESHOLD': 0.7, #0.7
    'PREDICTION_THRESHOLD_BUY': 0.7, #0.7
    'PREDICTION_THRESHOLD_SELL': 0.3, #0.38
    'RSI_BUY_THRESHOLD': 42,
    'RSI_BUY_THRESHOLD_RELAXED': 52,


                                       # not require oversold RSI to enter.
    'RSI_SELL_THRESHOLD': 72,         # Sell when overbought
    'ADX_TREND_THRESHOLD': 18,
    'MAX_VOLATILITY': 35.0,
    'PREDICTION_TEMPERATURE': 1.0,
                                     # With the triple-barrier target the positive class

                                     # below 0.5; sharpening pushed them further from the


                                     # use the model's calibrated probability directly.


    'ENABLE_TEMPERATURE_CALIBRATION': True,
    'CALIB_TEMP_FLOOR': 0.3,


    'USE_SAMPLE_UNIQUENESS': True,

    'ENABLE_FRACDIFF': True,

                                   # (uses only past bars) so there is no look-ahead.
    'FRACDIFF_D': 0.4,            # differencing order (0=raw level, 1=full return)
    'ENABLE_DAILY_TF': True,      # higher-timeframe daily trend context: close vs the

                                   # look-ahead).


    'MAX_DRAWDOWN_LIMIT': 0.04,  # Maximum allowed drawdown
    'RISK_PERCENTAGE': 0.02,


    'RISK_BY_REGIME': {
        'Calm Bull':     0.020,
        'Moderate Bull': 0.010,
        'Volatile Bull': 0.005,
    },


    'ENABLE_VOLUME_GATE': True,
    'VOLUME_CONFIRMATION_MULTIPLIER': 1.2,
    'VOLUME_MA_PERIOD': 20,


    # Each entry: name → {symbols: [..], max_pct: 0.X}
    'CORRELATION_GROUPS': {
        'Semis (NVDA+AMD)':   {'symbols': ['NVDA', 'AMD'],   'max_pct': 0.25},
        'BroadMkt (SPY+QQQ)': {'symbols': ['SPY', 'QQQ'],    'max_pct': 0.35},
        'BigTech (AAPL+MSFT+GOOGL+AMZN)': {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
            'max_pct': 0.50,
        },
    },


    'DAILY_STOP_COUNT_LIMIT': 3,        # pause new entries after 3 stop-outs today
    'DAILY_LOSS_PCT_LIMIT': 0.02,        # OR after −2% equity drop from session start


    'RISK_MULTIPLIER_BY_SYMBOL': {
        'AMD':  0.5,    # 2 of 3 historical stops; outsized $ loss
        'NVDA': 0.75,   # 1 of 2 stops; moderate skew
        # AAPL, MSFT, GOOGL, AMZN, SPY, QQQ default to 1.0
    },


    #     trailing stop.


    #     the price stop would.

    'ENABLE_REGIME_EXIT': True,
    'REGIME_EXIT_CONFIRM_CYCLES': 2,     # need N consecutive bad-regime cycles
    'REGIME_EXIT_MAX_PRED': 0.65,        # only exit if model also < this
    'REGIME_EXIT_PROFIT_LOCK_ATR': 1.0,  # if +N×ATR in profit, let trailing handle


    'ENABLE_TIME_EXIT': True,
    'TIME_EXIT_MINUTES': 240,            # held longer than this (model horizon ~315min)
    'TIME_EXIT_MAX_PRED': 0.52,          # AND model has turned bearish below this


    'ENABLE_CONVICTION_BYPASS': True,
    'CONVICTION_BYPASS_PRED': 0.97,      # prediction must exceed this ...
    'CONVICTION_BYPASS_CYCLES': 2,       # ... for this many consecutive cycles
    'CONVICTION_BYPASS_MAX_VOL': 6.0,
    'CONVICTION_BYPASS_SIZE_MULT': 0.5,  # half position size for momentum overrides


    #   AND regime in BUY_REGIME_WHITELIST.


    # following robust to a weak directional model).
    'ENABLE_TREND_ENTRY': True,
    'TREND_ADX_MIN': 25.0,            # require a strong, established trend.


    'TREND_PRED_MIN': 0.50,           # model must merely not be bearish
    'TREND_BREAKOUT_LOOKBACK': 20,    # bars defining the recent high
    'TREND_BREAKOUT_PCT': 0.97,       # price within 3% of that high (near/at breakout)
    'TREND_TRAIL_ATR_MULT': 3.5,
    'TREND_TRAIL_ATR_TIGHT': 2.0,
    'TREND_PROFIT_LOCK_ATR': 2.0,
    'TREND_HARD_STOP_ATR': 5.0,       # disaster stop below entry
    'TREND_RISK_PCT': 0.015,          # base risk per trend trade (x B1 symbol mult)
    'TREND_EXIT_PRED': 0.40,          # exit trend if model turns clearly bearish


    'USE_TRIPLE_BARRIER': True,
    'TB_TP_ATR': 3.0,
    'TB_SL_ATR': 2.0,                 # stop barrier (matches STOP_LOSS_ATR_MULTIPLIER)


    'CORE_LONG_PCT': 0.0,
    'CORE_CONF_LOOKBACK': 10,


    'ENABLE_CROSS_ASSET_FEATURES': True,
    'ENABLE_SEASONALITY_FEATURES': True,
    'MARKET_CONTEXT_SYMBOLS': ['SPY', 'QQQ'],


    'SMOKE_TEST': False,
    'SMOKE_TEST_EPOCHS': 3,


    #   conf = clip((pred - thr) / (1 - thr), 0, 1)
    #   mult = 1 + conf * (CONFIDENCE_SIZE_MAX_MULT - 1)


    'ENABLE_CONFIDENCE_SIZING': True,
    'CONFIDENCE_SIZE_MAX_MULT': 2.5,  # risk multiplier at full conviction (pred→1.0)
    'MAX_POSITION_SIZE_PCT': 0.20,
    'MIN_STOP_LOSS_PCT': 0.010,
    'STOP_LOSS_ATR_MULTIPLIER': 2.0,
    'TAKE_PROFIT_ATR_MULTIPLIER': 3.0,  # Multiplier for ATR-based take profit
    'TRAILING_STOP_PERCENTAGE': 0.05,  # Percentage for trailing stop
    'POST_STOP_COOLDOWN_MINUTES': 60,


    'TRANSACTION_COST_PER_TRADE': 0.01,  # Cost per trade


    'SENTIMENT_MODEL': 'distilbert-base-uncased-finetuned-sst-2-english',  # Model for sentiment analysis


    'API_RETRY_ATTEMPTS': 10,  # Number of retry attempts for API calls
    'API_RETRY_DELAY': 1000,  # Delay between retry attempts in milliseconds
    'MODEL_VERSION': '100017',


    '_MODEL_VERSION_PREV_v153': '100016',


    'USE_GRADIENT_CHECKPOINTING': False,


    'DATALOADER_NUM_WORKERS': 0,


    'PREFETCH_FACTOR': 2,
    'PERSISTENT_WORKERS': True,


    # These are ONLY used when USE_TRIPLE_BARRIER=True.
    'TRIPLE_BARRIER_BUY_THRESHOLD': 0.52,

    'TRIPLE_BARRIER_SELL_THRESHOLD': 0.42,

    # Resume support for multiday runs
    'RESUME_FROM_ATTEMPT': None,            # Set to int (e.g. 4) or use --resume CLI
    'SAVE_ATTEMPT_RESULTS': True,

    # Vectorized triple barrier cache key (internal)
    '_TB_CACHE_KEY': None,                  # populated at runtime from TB_* params

    # New: Retraining Cycle Parameters
    'ENABLE_RETRAIN_CYCLE': True,
    'FORCE_FULL_RETRAIN_RUN': True,
    'MIN_FINAL_VALUE': 130000.0,  # Minimum final portfolio value to accept
    'MAX_ALLOWED_DRAWDOWN': 30.0,
    'MAX_RETRAIN_ATTEMPTS': 15,


    'SHARPE_TARGET': 1.0,
    'MIN_TRADES_FOR_SHARPE': 5,
    'SELECTION_DD_PENALTY_PER_PCT': 0.05,
    'GRAD_CLIP_NORM': 1.0,

    #Monte Carlo Probability Simulation
    'NUM_MC_SIMULATIONS': 50000,

    # Account Management
    'RESET_ACCOUNT_ON_START': False,
    'PAPER_TRADING': True,             # Set to False when going live with real money
    'DESIRED_STARTING_CASH': 100000.00,  # Desired cash for reset


    'DATA_FEED': DataFeed.IEX,

    # Pairs-specific params
    'PAIR_CONFIDENCE_THRESHOLD': 0.55,
    'PAIR_REGIME_FILTER': ["Calm Bull", "Moderate Bull", "Calm Bear", "Moderate Bear"],
    'KALMAN_LOOKBACK': 30,
    'ENABLE_FULL_PAIRS_RESOLUTION': False,


    'PREVENT_PYRAMIDING': True,

    # Single-stock regime filter for BUY entries.


    'BUY_REGIME_WHITELIST': ["Calm Bull", "Moderate Bull", "Volatile Bull"],

    # Email summary behaviour:


    'EMAIL_SUMMARY_ALWAYS': True,


    # restart time into the next boundary window.
    'CYCLE_GUARD_MINUTES': 14,
}


if os.environ.get('DT_SMOKE_TEST') == '1':
    CONFIG['SMOKE_TEST'] = True
    CONFIG['MAX_RETRAIN_ATTEMPTS'] = 1
    CONFIG['FORCE_FULL_RETRAIN_RUN'] = False
    CONFIG['TRAIN_EPOCHS'] = int(os.environ.get('DT_SMOKE_EPOCHS', '3'))
    CONFIG['NUM_MC_SIMULATIONS'] = 1000

    CONFIG['USE_GRADIENT_CHECKPOINTING'] = False


    CONFIG['BATCH_SIZE'] = 4096


    CONFIG['BUY_REGIME_WHITELIST'] = None          # trade in any regime during smoke
    CONFIG['ENABLE_VOLUME_GATE'] = False
    CONFIG['DAILY_STOP_COUNT_LIMIT'] = 999
    CONFIG['DAILY_LOSS_PCT_LIMIT'] = 0.99


# ==============================================================================

#pyenv activate pytorch_env


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # add indicators
    df['MA20'] = talib.SMA(df['close'], timeperiod=20)
    df['MA50'] = talib.SMA(df['close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    # Chaikin Money Flow approximation (20-period)
    df['CMF'] = talib.AD(df['high'], df['low'], df['close'], df['volume']) / df['volume'].rolling(20).sum()
    df['Close_ATR'] = df['close'] / df['ATR']
    df['MA20_ATR'] = df['MA20'] / df['ATR']
    df['Return_1d'] = df['close'].pct_change(1)
    df['Return_5d'] = df['close'].pct_change(5)
    df['Volatility'] = df['Return_1d'].rolling(20).std() * np.sqrt(252)  # Annualized
    upper, _, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['BB_upper'] = upper
    df['BB_lower'] = lower
    df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df = df.dropna()  # Keep datetime index for filtering
    return df

def get_sentiment_score(symbol: str) -> float:

    cache_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_news_sentiment_{CONFIG['MODEL_VERSION']}.pkl")
    if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path) < 3600):
        with open(cache_path, 'rb') as f:
            score = pickle.load(f)
        logger.info(f"Loaded REAL sentiment for {symbol} from cache: {score:.3f}")
        return score

    try:
        news_client = NewsClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'])
        request = NewsRequest(
            symbols=symbol,
            start=datetime.now(timezone.utc) - timedelta(days=7),
            limit=30
        )
        news_items = news_client.get_news(request)
        
        if not news_items:
            score = 0.0
        else:
            texts = []
            for item in news_items:
                if isinstance(item, dict):
                    headline = item.get('headline', '')
                    summary = item.get('summary', '') or ''
                elif isinstance(item, (list, tuple)):
                    headline = item[0] if len(item) > 0 else ""
                    summary = item[1] if len(item) > 1 else ""
                else:
                    headline = getattr(item, 'headline', "")
                    summary = getattr(item, 'summary', "") or ""
                texts.append(str(headline) + ". " + str(summary))
            

            texts = [t[:2000] for t in texts]  # safe truncation
            results = get_sentiment_pipeline()(texts, truncation=True, max_length=512, batch_size=8)
            scores = [1.0 if r['label'] == 'POSITIVE' else -1.0 for r in results]
            score = np.mean(scores) if results else 0.0
        
        logger.info(f"REAL sentiment for {symbol}: {score:.3f} from {len(news_items)} articles")
    except Exception as e:
        logger.debug(f"News API failed for {symbol}: {e} → using 0.0 neutral sentiment")
        score = 0.0

    with open(cache_path, 'wb') as f:
        pickle.dump(score, f)
    return score


def is_cointegrated(series1: pd.Series, series2: pd.Series, pvalue_threshold: float = 0.05) -> bool:
    # Align to common index and drop NaNs
    common_idx = series1.index.intersection(series2.index)
    s1 = series1.loc[common_idx].dropna()
    s2 = series2.loc[common_idx].dropna()
    if len(s1) < 100 or len(s2) < 100 or len(s1) != len(s2):
        return False
    result = ts.coint(s1, s2)
    return result[1] < pvalue_threshold

def calculate_spread(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
    common_idx = df1.index.intersection(df2.index)
    price1 = df1['close'].loc[common_idx].dropna()
    price2 = df2['close'].loc[common_idx].dropna()
    if len(price1) != len(price2) or len(price1) < 100:
        return price1 - price2.reindex(price1.index, method='ffill')
    model = OLS(price1, price2).fit()

    hedge_ratio = model.params.iloc[0]
    spread = price1 - hedge_ratio * price2
    return spread

def get_pair_regime(hmm1, hmm2, recent_seq1, recent_seq2) -> str:
    if hmm1 is None or hmm2 is None:
        return "Unknown"                    # ← prevent crash
    try:


        _in1 = recent_seq1[:, -1, :] if recent_seq1.ndim == 3 else recent_seq1.reshape(-1, recent_seq1.shape[-1])
        _in2 = recent_seq2[:, -1, :] if recent_seq2.ndim == 3 else recent_seq2.reshape(-1, recent_seq2.shape[-1])
        n1 = regime_name_for_state(hmm1, hmm1.predict(_in1)[-1])
        n2 = regime_name_for_state(hmm2, hmm2.predict(_in2)[-1])
        if ' ' in n1 and ' ' in n2:
            t1, d1 = n1.split()[0], n1.split()[-1]
            t2, d2 = n2.split()[0], n2.split()[-1]
            if d1 == d2:
                _rank = {"Calm": 0, "Moderate": 1, "Volatile": 2}
                worst_tier = t1 if _rank.get(t1, 0) >= _rank.get(t2, 0) else t2
                return f"{worst_tier} {d1}"
        return "Unknown"
    except Exception as e:
        logger.warning(f"Pair regime failed for {hmm1}/{hmm2}: {e}")
        return "Unknown"


@retry(retry=retry_if_exception_type(Exception), stop=stop_after_attempt(3), wait=wait_fixed(5))
def load_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:

    cache_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_{start_date}_{end_date}.pkl")
    if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path) < CONFIG['CACHE_EXPIRY_SECONDS']):
        with open(cache_path, 'rb') as f:
            df = pickle.load(f)
        logger.info(f"Loaded {len(df)} bars for {symbol} from cache")
    else:
        client = StockHistoricalDataClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'])
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=CONFIG['TIMEFRAME'],
            start=pd.to_datetime(start_date),
            end=pd.to_datetime(end_date),
            feed=CONFIG['DATA_FEED']
        )
        bars = client.get_stock_bars(request_params)
        df = bars.df.reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.index = pd.to_datetime(df.index)  # Force DatetimeIndex for multiprocessing safety
        if not os.path.exists(CONFIG['CACHE_DIR']):
            os.makedirs(CONFIG['CACHE_DIR'])
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)
        logger.info(f"Loaded {len(df)} bars for {symbol} from API")
    if len(df) < CONFIG['MIN_DATA_POINTS']:
        logger.warning(f"Insufficient data for {symbol}: only {len(df)} points")
    return df


def train_wrapper(args):
    worker_id, symbol, expected_features, force_train, barrier, gpu_semaphore, backtest_only, debug = args
    start_time_for_training = time.perf_counter()
    try:
        result_from_train_symbol = train_symbol(
            symbol,
            worker_id,
            expected_features,
            force_train,
            barrier,
            gpu_semaphore,
            backtest_only=backtest_only,
            debug=debug
        )


        _res = list(result_from_train_symbol)
        try:
            if _res[1] is not None and hasattr(_res[1], 'to'):
                _res[1] = _res[1].to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as _ce:
            logger.warning(f"[{symbol}] could not move model to CPU before return: {_ce}")
        result_from_train_symbol = tuple(_res)
        end_time_for_training = time.perf_counter()
        training_time_in_milliseconds = (end_time_for_training - start_time_for_training) * 1000
        return (*result_from_train_symbol, training_time_in_milliseconds)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"WORKER CRASH for {symbol}: {str(e)}\n{error_trace}")
        dummy = (symbol, None, None, False, 0.0, False, False, None, None)
        return (*dummy, 0)


logger = logging.getLogger(__name__)


_sentiment_pipeline = None

def get_sentiment_pipeline():
    """Return the sentiment pipeline, initializing it on first call."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _device = 0 if torch.cuda.is_available() else -1
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=CONFIG['SENTIMENT_MODEL'],
            framework="pt",
            device=_device
        )
    return _sentiment_pipeline


def _get_gpu_info_safe() -> tuple[str, float]:
    """Query GPU name and free VRAM via nvidia-smi (NO CUDA context created in main process)."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=3
        ).decode().strip().split(",")
        name = result[0].strip()
        free_mb = float(result[1].strip())
        free_gb = free_mb / 1024.0
        return name, free_gb
    except Exception:
        return "Unknown GPU", 0.0


def check_dependencies() -> None:
    required_modules = [
        'torch', 'numpy', 'pandas', 'alpaca', 'transformers',
        'sklearn', 'talib', 'tenacity', 'smtplib', 'argparse', 'tqdm', 'colorama',
        'hmmlearn'  # NEW for regime detection
    ]
    for module in required_modules:
        try:
            importlib.import_module(module)
        except ImportError:
            raise ImportError(f"Module '{module}' is required. Install it using: pip install {module}")

def validate_config(config: Dict) -> None:
    if not config['SYMBOLS']:
        raise ValueError("SYMBOLS list cannot be empty")
    if not isinstance(config['TIMEFRAME'], TimeFrame):
        raise ValueError("TIMEFRAME must be a valid TimeFrame object")
    for param in ['SIMULATION_DAYS', 'TRAIN_EPOCHS', 'BATCH_SIZE', 'TIMESTEPS', 'MIN_DATA_POINTS', 'LOOK_AHEAD_BARS']:
        if not isinstance(config[param], int) or config[param] <= 0:
            raise ValueError(f"{param} must be a positive integer")
    for param in ['INITIAL_CASH', 'STOP_LOSS_ATR_MULTIPLIER', 'TAKE_PROFIT_ATR_MULTIPLIER', 'MAX_DRAWDOWN_LIMIT', 'RISK_PERCENTAGE']:
        if not isinstance(config[param], (int, float)) or config[param] <= 0:
            raise ValueError(f"{param} must be a positive number")


def create_cache_directory() -> None:
    os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)
    # Test writability with a dummy file
    test_path = os.path.join(CONFIG['CACHE_DIR'], 'writability_test.txt')
    try:
        with open(test_path, 'w') as f:
            f.write('Test')
        os.remove(test_path)
        logger.info(f"Cache directory {CONFIG['CACHE_DIR']} is writable.")
    except Exception as e:
        logger.warning(f"Cache directory {CONFIG['CACHE_DIR']} writability test failed: {str(e)}. Saves may fail.")

    os.makedirs(CONFIG['MODEL_CACHE_DIR'], exist_ok=True)
    # Test writability with a dummy file for model dir
    test_model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], 'writability_test.txt')
    try:
        with open(test_model_path, 'w') as f:
            f.write('Test')
        os.remove(test_model_path)
        logger.info(f"Model cache directory {CONFIG['MODEL_CACHE_DIR']} is writable.")
    except Exception as e:
        logger.error(f"Model cache directory {CONFIG['MODEL_CACHE_DIR']} writability test failed: {str(e)}. Check Windows permissions for the mapped folder. Halting execution to prevent failed saves.")
        raise PermissionError(f"Cannot write to MODEL_CACHE_DIR: {str(e)}. Use a WSL-native path or fix permissions.")
    logger.info(f"Data caches (e.g., historical bars, news sentiment) will be written to {CONFIG['CACHE_DIR']}.")
    if CONFIG['MODEL_CACHE_DIR'].startswith('/mnt/c/'):
        windows_model_dir = CONFIG['MODEL_CACHE_DIR'].replace('/mnt/c/', 'C:\\').replace('/', '\\')
        logger.info(f"Model files (weights, scalers, training sentiment) will be written to {CONFIG['MODEL_CACHE_DIR']} (Windows: {windows_model_dir}).")
    else:
        logger.info(f"Model files (weights, scalers, training sentiment) will be written to {CONFIG['MODEL_CACHE_DIR']} (WSL-native path; access via \\\\wsl$\\\\<distro>\\\\path\\\\to\\\\dir from Windows).")

@retry(
    stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']),
    wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000),
    retry=retry_if_exception_type(Exception)
)
def cleanup_account_on_start(force_reset: bool = False) -> None:
    # Do nothing unless you explicitly ask for a reset
    if not (force_reset or CONFIG.get('RESET_ACCOUNT_ON_START', False)):
        return

    # Settings
    PAPER_MODE = CONFIG.get('PAPER_TRADING', True)                    # True = paper, False = live
    DESIRED_STARTING_CASH = CONFIG.get('DESIRED_STARTING_CASH', 200000.00)  # Default to 200k if not set
    FAKE_SYMBOL = "SPY"                                               # Cheap liquid ETF for cash injection trick
    CHUNK_SIZE = 100000.00                                            # Smaller $100k chunks to avoid qty locks
    POLL_TIMEOUT = 600                                                # 10min for off-hours
    POLL_INTERVAL = 10                                                # Check every 10s to reduce API load

    trading_client = TradingClient(
        CONFIG['ALPACA_API_KEY'],
        CONFIG['ALPACA_SECRET_KEY'],
        paper=PAPER_MODE
    )

    try:
        print(f"{Fore.YELLOW}=== ACCOUNT RESET REQUESTED ==={Style.RESET_ALL}")
        logger.info("Starting account reset and cash injection")


        trading_client.close_all_positions(cancel_orders=True)
        logger.info("Requested close all positions and cancel orders")

        # Extra cancel for any lingering
        trading_client.cancel_orders()

        # Poll until cleared
        clock = trading_client.get_clock()
        if not clock.is_open:
            warning_msg = f"Market is closed (next open: {clock.next_open}). Position close orders queued and will be filled automatically by Alpaca at market open—no need to rerun for closes. Skipping polling and injection to avoid errors. Positions will clear at open; check dashboard for status. Rerun during open hours if you need to confirm or inject cash after clears."
            logger.warning(warning_msg)
            print(f"{Fore.YELLOW}{warning_msg}{Style.RESET_ALL}")
            raise Exception("Market closed - closes queued but reset incomplete. Rerun during open hours for full reset if needed.")
        else:
            start_time = time.time()
            while time.time() - start_time < POLL_TIMEOUT:
                positions = trading_client.get_all_positions()
                orders = trading_client.get_orders()
                if not positions and not orders:
                    logger.info("All positions closed and orders cancelled successfully")
                    print(f"{Fore.CYAN}All positions closed and orders cancelled{Style.RESET_ALL}")
                    break
                pos_symbols = [pos.symbol for pos in positions]
                order_ids = [str(order.id) for order in orders]
                logger.info(f"Waiting for clears... Positions left: {len(positions)} ({pos_symbols}), Orders left: {len(orders)} ({order_ids})")
                print(f"  → Waiting for clears... ({len(positions)} positions left: {pos_symbols}, {len(orders)} orders left)")
                time.sleep(POLL_INTERVAL)
            else:
                pos_symbols = [pos.symbol for pos in positions]
                order_ids = [str(order.id) for order in orders]
                warning_msg = f"Timed out waiting for clears. Remaining: {len(positions)} positions ({pos_symbols}), {len(orders)} orders ({order_ids}). Continuing to injection anyway - check dashboard."
                logger.warning(warning_msg)
                print(f"{Fore.YELLOW}{warning_msg}{Style.RESET_ALL}")
            # Force cancel any lingering orders post-timeout
            trading_client.cancel_orders()

        # Inject / remove cash in chunks
        account = trading_client.get_account()
        current_cash = float(account.cash)
        difference = DESIRED_STARTING_CASH - current_cash

        if abs(difference) > 50:
            num_chunks = max(1, int(abs(difference) / CHUNK_SIZE))
            chunk_diff = difference / num_chunks
            for i in range(num_chunks):
                # Poll for no holds on FAKE_SYMBOL before submit
                chunk_poll_start = time.time()
                while time.time() - chunk_poll_start < 120:
                    try:
                        pos = trading_client.get_position(FAKE_SYMBOL)
                        if float(pos.qty_available) >= 0 and float(pos.qty_held_for_orders) == 0:
                            break
                    except APIError as e:
                        if 'not found' in str(e):  # No position is fine
                            break
                    print(f"  → Waiting for {FAKE_SYMBOL} to be free from holds...")
                    time.sleep(5)
                else:
                    logger.warning(f"Timeout waiting for {FAKE_SYMBOL} free - skipping chunk {i+1}")
                    continue

                # Recalculate this chunk
                account = trading_client.get_account()
                current_cash = float(account.cash)
                this_diff = min(chunk_diff, DESIRED_STARTING_CASH - current_cash) if difference > 0 else max(chunk_diff, DESIRED_STARTING_CASH - current_cash)
                if abs(this_diff) < 50:
                    break

                price_estimate = 500.0 if FAKE_SYMBOL == "SPY" else 100.0
                fake_qty = abs(this_diff) / price_estimate

                side = OrderSide.SELL if this_diff > 0 else OrderSide.BUY
                fake_order = MarketOrderRequest(
                    symbol=FAKE_SYMBOL,
                    qty=fake_qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                try:
                    trading_client.submit_order(fake_order)
                    time.sleep(2)
                    trading_client.cancel_orders()
                    logger.info(f"Chunk {i+1}/{num_chunks}: adjusted by {this_diff:+,.2f}")
                    print(f"{Fore.CYAN}Chunk {i+1}: Cash adjusted by {this_diff:+,.2f}{Style.RESET_ALL}")
                except Exception as e:
                    logger.warning(f"Chunk {i+1} failed: {str(e)} - skipping")

        # --------------------------------------------------------------

        # --------------------------------------------------------------
        account = trading_client.get_account()
        final_cash = float(account.cash)
        print(f"{Fore.GREEN}RESET COMPLETE!{Style.RESET_ALL}")
        print(f"   Cash          : ${final_cash:,.2f}")
        print(f"   Portfolio     : ${float(account.portfolio_value):,.2f}")
        print(f"   Positions     : 0")
        logger.info(f"Reset complete – cash ≈ ${final_cash:,.2f}")

        # --------------------------------------------------------------

        # --------------------------------------------------------------
        CONFIG['RESET_ACCOUNT_ON_START'] = False
        print(f"{Fore.YELLOW}Reset flag automatically turned OFF – safe for future runs.{Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"Account reset failed: {str(e)}")
        print(f"{Fore.RED}Reset failed: {str(e)}{Style.RESET_ALL}")


# def get_all_positions_with_retry():
#     """Retry wrapper for getting all positions."""
#     return trading_client.get_all_positions()

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(6),
    wait=wait_fixed(2)
)
def fetch_recent_data(symbol: str, num_bars: int = CONFIG['LIVE_DATA_BARS']) -> pd.DataFrame:
    """Fetch recent bars for live inference using the configured data feed.

    Uses CONFIG['DATA_FEED'] (default: DataFeed.IEX) so the free Alpaca tier
    works out of the box.  Window sizes are extended to handle long weekends and
    holiday gaps.  Always returns a DataFrame — empty on total failure.
    The caller MUST check emptiness before passing to calculate_indicators.
    """
    client = StockHistoricalDataClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'])
    end_date = datetime.now(timezone.utc)


    required_bars = max(CONFIG['TIMESTEPS'] + 20, CONFIG.get('LIVE_FEATURE_MIN_BARS', 600))

    logger.info(f"[LIVE FETCH] {symbol} - requesting minimum {required_bars} recent bars (feed={CONFIG['DATA_FEED']})")


    best_df = None
    for days_back in [30, 45, 60, 90]:
        start_date = end_date - timedelta(days=days_back)
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=CONFIG['TIMEFRAME'],
                start=start_date,
                end=end_date,
                limit=max(num_bars * 4, required_bars * 4),
                feed=CONFIG['DATA_FEED']
            )
            bars_response = client.get_stock_bars(request)
            bars = bars_response.df

            if isinstance(bars.index, pd.MultiIndex):
                if symbol in bars.index.get_level_values(0):
                    bars = bars.xs(symbol, level=0)
                else:
                    logger.warning(f"[LIVE] {symbol} not in MultiIndex response for {days_back}d window")
                    continue

            if bars.empty:
                logger.warning(f"[LIVE] {symbol}: empty response for {days_back}d window")
                continue

            df = bars.rename(columns={'vwap': 'VWAP'}).sort_index()

            # Keep the largest result seen so far as fallback
            if best_df is None or len(df) > len(best_df):
                best_df = df

            if len(df) >= required_bars:
                last_price = float(df['close'].iloc[-1])
                logger.info(f"[LIVE FETCH SUCCESS] {symbol}: {len(df)} bars | Last: ${last_price:.2f} ({days_back}d window, {CONFIG['DATA_FEED']})")
                print(f"{Fore.GREEN}[LIVE] {symbol}: {len(df)} bars | Last: ${last_price:.2f}{Style.RESET_ALL}")
                return df

            logger.info(f"[LIVE] {symbol}: {len(df)} bars from {days_back}d window — need {required_bars}, widening")

        except Exception as inner_e:
            logger.warning(f"Live fetch failed for {symbol} ({days_back}d window): {inner_e}")
            continue


    if best_df is not None and not best_df.empty:
        last_price = float(best_df['close'].iloc[-1])
        logger.warning(f"[LIVE PARTIAL] {symbol}: {len(best_df)} bars (need {required_bars}) — using partial data")
        print(f"{Fore.YELLOW}[LIVE PARTIAL] {symbol}: {len(best_df)} bars | Last: ${last_price:.2f}{Style.RESET_ALL}")
        return best_df

    logger.error(f"[LIVE FETCH FAILED] {symbol} - no bars returned from any window. Check feed/subscription.")
    print(f"{Fore.RED}[LIVE FAIL] {symbol}: no data returned — skipping this cycle{Style.RESET_ALL}")

    return pd.DataFrame()


def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    cache_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_full_history_fallback.pkl")
    try:
        client = StockHistoricalDataClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'])
        all_bars = []
        current_start = pd.Timestamp(start_date, tz='UTC')
        end_dt = pd.Timestamp(end_date, tz='UTC')
        increment = pd.DateOffset(years=1)

        while current_start < end_dt:
            current_end = min(current_start + increment, end_dt)
            logger.info(f"Fetching data for {symbol} from {current_start} to {current_end}")
            effective_symbol = 'FB' if symbol == 'META' and current_start < pd.Timestamp('2021-10-28', tz='UTC') else symbol
            request = StockBarsRequest(
                symbol_or_symbols=effective_symbol,
                timeframe=CONFIG['TIMEFRAME'],
                start=current_start,
                end=current_end,
                feed=CONFIG['DATA_FEED']
            )
            bars = client.get_stock_bars(request).df

            if not bars.empty:
                df_bars = bars.reset_index().rename(columns={'vwap': 'VWAP'})
                all_bars.append(df_bars)
                logger.info(f"Fetched {len(df_bars)} bars for {symbol}")
            else:
                logger.info(f"No data for {symbol} from {current_start} to {current_end}, skipping")
            current_start = current_end

        if all_bars:
            df = pd.concat(all_bars).sort_values('timestamp')
            df = df.drop_duplicates(subset='timestamp', keep='first')
            logger.info(f"Total fetched {len(df)} bars for {symbol}")
            # Save fallback cache
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
        else:
            df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'VWAP'])

        if len(df) < CONFIG['MIN_DATA_POINTS']:
            raise ValueError(f"Insufficient data for {symbol}: got {len(df)} bars")

        return df

    except Exception as e:
        logger.warning(f"API error for {symbol}: {str(e)} → trying fallback cache")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                df = pickle.load(f)
            logger.info(f"Used fallback cache for {symbol} ({len(df)} rows)")
            return df
        logger.error(f"No fallback cache for {symbol} — training will skip this symbol")
        raise


def load_or_fetch_data(symbol: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, bool]:
    cache_file = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_train_data_{start_date}_{end_date}.pkl")
    
    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < CONFIG['CACHE_EXPIRY_SECONDS']:
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        logger.info(f"Loaded {len(df)} bars for {symbol} from training cache")
        return df, True
    else:
        df = fetch_data(symbol, start_date, end_date)
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        logger.info(f"Fetched and cached {len(df)} bars for {symbol} as training data")
        return df, False


def load_news_sentiment(symbol: str) -> Tuple[float, bool]:
    cache_file = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_news_sentiment.pkl")
    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < CONFIG['CACHE_EXPIRY_SECONDS']:
        with open(cache_file, 'rb') as f:
            sentiment_score = pickle.load(f)
        return sentiment_score, True
    else:

        sentiment_score = 0.0  # Override to neutral while keeping framework
        with open(cache_file, 'wb') as f:
            pickle.dump(sentiment_score, f)
        return sentiment_score, False


def compute_triple_barrier_label(close: np.ndarray, atr: np.ndarray, lookahead: int,
                                  tp_atr_mult: float, sl_atr_mult: float,
                                  return_touch_ahead: bool = False):
    """v14: FULLY VECTORIZED triple-barrier label (30-100× faster than v13 Python loop).

    Same semantics as before.  Now suitable for 10+ year 15-min histories without
    costing hours per attempt.

    v15: when return_touch_ahead=True, also returns `touch_ahead` — the number of bars
    forward until the first barrier (TP/SL) was touched, or `lookahead` on timeout
    (vertical barrier). NaN where the label is NaN. Used for de Prado sample-uniqueness
    weighting (overlapping label spans share information and are down-weighted).
    """
    close = np.asarray(close, dtype=np.float64)
    atr = np.asarray(atr, dtype=np.float64)
    n = len(close)
    labels = np.full(n, np.nan, dtype=np.float32)
    touch_ahead = np.full(n, np.nan, dtype=np.float32)

    def _ret(lab, ta):
        return (lab, ta) if return_touch_ahead else lab

    valid = np.isfinite(atr) & (atr > 0)
    if not np.any(valid):
        return _ret(labels, touch_ahead)


    max_i = n - lookahead
    if max_i <= 0:
        return _ret(labels, touch_ahead)

    idx = np.where(valid[:max_i])[0]
    if len(idx) == 0:
        return _ret(labels, touch_ahead)

    # Barriers for valid bars
    up_bar = close[idx] + tp_atr_mult * atr[idx]
    dn_bar = close[idx] - sl_atr_mult * atr[idx]


    # for this "first hit" problem without numba.
    windows = np.lib.stride_tricks.as_strided(
        close,
        shape=(len(idx), lookahead),
        strides=(close.strides[0], close.strides[0]),
        writeable=False
    )
    # Offset the windows correctly
    windows = close[idx[:, None] + np.arange(1, lookahead + 1)]

    up_hit = (windows >= up_bar[:, None]).argmax(axis=1)
    up_hit[~np.any(windows >= up_bar[:, None], axis=1)] = lookahead + 1

    dn_hit = (windows <= dn_bar[:, None]).argmax(axis=1)
    dn_hit[~np.any(windows <= dn_bar[:, None], axis=1)] = lookahead + 1

    labels[idx] = np.where(
        (up_hit <= lookahead) & (up_hit < dn_hit),
        1.0,
        0.0
    )
    # timeout (neither hit) already 0.0


    # barrier at `lookahead` bars.
    _mh = np.minimum(up_hit, dn_hit)
    touch_ahead[idx] = np.where(_mh <= lookahead - 1, _mh + 1, lookahead).astype(np.float32)
    return _ret(labels, touch_ahead)


def compute_average_uniqueness(span_lengths: np.ndarray) -> np.ndarray:
    """de Prado average-uniqueness weights from per-label forward span lengths.

    Each label i is treated as 'active' over sequence positions [i, i+span_i]. The
    concurrency c[t] is how many labels are active at t; label i's uniqueness is the
    mean of 1/c[t] over its span. Overlapping (redundant) labels get lower weight.
    Returns weights normalised to mean 1.0 so the overall loss scale is unchanged.
    NaN/invalid spans are treated as 0 (label active only at its own bar).
    """
    m = len(span_lengths)
    if m == 0:
        return np.ones(0, dtype=np.float32)
    spans = np.nan_to_num(np.asarray(span_lengths, dtype=np.float64), nan=0.0)
    spans = np.clip(spans, 0, None).astype(np.int64)
    starts = np.arange(m)
    end = np.minimum(starts + spans, m - 1)
    # concurrency via difference array
    diff = np.zeros(m + 1, dtype=np.float64)
    np.add.at(diff, starts, 1.0)
    np.add.at(diff, end + 1, -1.0)
    conc = np.cumsum(diff[:-1])
    conc[conc < 1.0] = 1.0
    inv = 1.0 / conc
    prefix = np.concatenate([[0.0], np.cumsum(inv)])
    u = (prefix[end + 1] - prefix[starts]) / (end - starts + 1).astype(np.float64)
    mean_u = u.mean()
    if mean_u <= 1e-9:
        return np.ones(m, dtype=np.float32)
    return (u / mean_u).astype(np.float32)


# pipeline never crashes on missing context.
_MARKET_CONTEXT = None
_MARKET_CONTEXT_TS = 0.0

def get_market_context(start_date: Optional[str] = None, end_date: Optional[str] = None,
                       ttl_seconds: int = 3600) -> Dict[str, pd.Series]:
    global _MARKET_CONTEXT, _MARKET_CONTEXT_TS
    if _MARKET_CONTEXT is not None and (time.time() - _MARKET_CONTEXT_TS) < ttl_seconds:
        return _MARKET_CONTEXT
    ctx: Dict[str, pd.Series] = {}
    try:
        sd = start_date or CONFIG['TRAIN_DATA_START_DATE']
        ed = end_date or (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
        for mkt in CONFIG.get('MARKET_CONTEXT_SYMBOLS', ['SPY', 'QQQ']):
            try:
                mdf = load_data(mkt, sd, ed)
                ret = mdf['close'].pct_change().fillna(0.0)
                ret.index = pd.to_datetime(ret.index)
                if ret.index.tz is None:
                    ret.index = ret.index.tz_localize('UTC')
                ctx[mkt] = ret
            except Exception as e:
                logger.warning(f"market context load failed for {mkt}: {e}")
    except Exception as e:
        logger.warning(f"get_market_context failed: {e}")
        ctx = {}
    _MARKET_CONTEXT = ctx
    _MARKET_CONTEXT_TS = time.time()
    return _MARKET_CONTEXT


def _frac_diff_weights(d: float, thres: float = 1e-4, max_width: int = 50) -> np.ndarray:
    """Fixed-width fractional-difference weights (newest-first), de Prado.

    w[0]=1 applies to x[t], w[k] to x[t-k]; series truncated once |w_k| < thres.
    """
    w = [1.0]
    k = 1
    while k < max_width:
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thres:
            break
        w.append(wk)
        k += 1
    return np.array(w, dtype=np.float64)


def calculate_indicators(df: pd.DataFrame, sentiment: float) -> pd.DataFrame:
    """Fixed version for both training AND live data (much more tolerant of short history)."""

    if df is None or df.empty:
        raise ValueError("calculate_indicators received an empty DataFrame — cannot compute indicators")

    df = df.copy()
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")

    # Force proper datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if 'VWAP' not in df.columns:
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()


    df['MA20'] = talib.SMA(df['close'], timeperiod=20)
    df['MA50'] = talib.SMA(df['close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['CMF'] = talib.AD(df['high'], df['low'], df['close'], df['volume']) / df['volume'].rolling(20).sum()
    df['Close_ATR'] = df['close'] / df['ATR']
    df['MA20_ATR'] = df['MA20'] / df['ATR']
    df['Return_1d'] = df['close'].pct_change()
    df['Return_5d'] = df['close'].pct_change(periods=5)


    df['Volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
        df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
    )
    df['Stoch_K'], df['Stoch_D'] = talib.STOCH(
        df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3
    )
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    df['Sentiment'] = sentiment
    df['Trend'] = np.where(df['close'] > df['MA20'], 1, 0)

    # Multi-timeframe (60-min)
    df_60 = df.resample('60min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    df['RSI_60'] = talib.RSI(df_60['close'], timeperiod=14).reindex(df.index, method='ffill')
    df['MA20_60'] = talib.SMA(df_60['close'], timeperiod=20).reindex(df.index, method='ffill')

    # Additional features
    df['VWAP_Dev'] = df['close'] - df['VWAP']
    df['Volume_Delta'] = df['volume'] * (df['close'] - df['open'])
    df['Macro_Stress'] = df['Volatility'].rolling(20).mean() / df['Volatility'].rolling(100, min_periods=20).mean()
    df['Earnings_Proxy'] = df['Sentiment'] * (1 + df['Volume_Delta'].abs() / df['volume'].rolling(20, min_periods=5).mean())


    if CONFIG.get('ENABLE_SEASONALITY_FEATURES', True):
        _mins = df.index.hour * 60 + df.index.minute
        df['TimeOfDay_Sin'] = np.sin(2 * np.pi * _mins / 1440.0)
        df['TimeOfDay_Cos'] = np.cos(2 * np.pi * _mins / 1440.0)
    else:
        df['TimeOfDay_Sin'] = 0.0
        df['TimeOfDay_Cos'] = 0.0


    if CONFIG.get('ENABLE_CROSS_ASSET_FEATURES', True):
        try:
            _mc = get_market_context()
        except Exception:
            _mc = {}
        for _mkt, _col in (('SPY', 'SPY_Ret'), ('QQQ', 'QQQ_Ret')):
            _series = _mc.get(_mkt)
            if _series is not None and len(_series) > 0:
                try:
                    df[_col] = _series.reindex(df.index, method='ffill').fillna(0.0).values
                except Exception:
                    df[_col] = 0.0
            else:
                df[_col] = 0.0
    else:
        df['SPY_Ret'] = 0.0
        df['QQQ_Ret'] = 0.0


    if CONFIG.get('ENABLE_FRACDIFF', True):
        try:
            _w = _frac_diff_weights(float(CONFIG.get('FRACDIFF_D', 0.4)), 1e-4, 50)
            _logc = np.log(df['close'].clip(lower=1e-6).values.astype(np.float64))
            if len(_logc) >= len(_w):
                _fd = np.convolve(_logc, _w)[:len(_logc)]   # causal: sum_k w[k]*x[t-k]
                _fd[:len(_w) - 1] = np.nan                   # partial windows invalid
            else:
                _fd = np.full(len(_logc), np.nan)
            df['FracDiff_Close'] = _fd
        except Exception:
            df['FracDiff_Close'] = 0.0
    else:
        df['FracDiff_Close'] = 0.0


    if CONFIG.get('ENABLE_DAILY_TF', True):
        try:
            df_d = df.resample('1D').agg({'close': 'last'}).dropna()
            _dma = talib.SMA(df_d['close'], timeperiod=20).shift(1)
            _dma_i = _dma.reindex(df.index, method='ffill')
            df['Daily_Trend'] = (df['close'] / _dma_i - 1.0).replace([np.inf, -np.inf], 0.0)
        except Exception:
            df['Daily_Trend'] = 0.0
    else:
        df['Daily_Trend'] = 0.0


    _safe_close = df['close'].clip(lower=1e-6)

    df['Close_MA20_Ratio'] = (_safe_close / df['MA20'].clip(lower=1e-6) - 1.0).replace([np.inf, -np.inf], 0.0)
    # close / MA50 - 1
    df['Close_MA50_Ratio'] = (_safe_close / df['MA50'].clip(lower=1e-6) - 1.0).replace([np.inf, -np.inf], 0.0)
    # close / VWAP - 1
    df['Close_VWAP_Ratio'] = (_safe_close / df['VWAP'].clip(lower=1e-6) - 1.0).replace([np.inf, -np.inf], 0.0)

    _bb_width = (df['BB_upper'] - df['BB_lower']).clip(lower=1e-6)
    df['BB_Position'] = ((_safe_close - df['BB_lower']) / _bb_width).clip(0.0, 2.0)

    df['MA20_MA50_Ratio'] = (df['MA20'].clip(lower=1e-6) / df['MA50'].clip(lower=1e-6) - 1.0).replace([np.inf, -np.inf], 0.0)

    df['ATR_Pct'] = (df['ATR'] / _safe_close).replace([np.inf, -np.inf], 0.0)

    _vol_avg = df['volume'].rolling(20, min_periods=5).mean().clip(lower=1)
    df['Volume_Ratio'] = (df['volume'] / _vol_avg).replace([np.inf, -np.inf], 1.0)


    # Fill NaNs instead of dropping everything
    df = df.ffill().bfill().fillna(0)

    return df

def validate_raw_data(df: pd.DataFrame, symbol: str) -> None:
    if df.empty:
        raise ValueError(f"Empty DataFrame for {symbol}")
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for {symbol}: {missing}")
    if df[required_cols[:-1]].isna().any().any():
        raise ValueError(f"NaN values in OHLCV columns for {symbol}")

def validate_indicators(df: pd.DataFrame, symbol: str) -> None:
    required_cols = [
        'open', 'high', 'low', 'close', 'volume', 'timestamp',
        'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR',
        'CMF', 'Close_ATR', 'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility',
        'BB_upper', 'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment', 'Trend'
    ]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing indicator columns for {symbol}: {missing}")


MODEL_FEATURES = [
    'Return_1d', 'Return_5d',
    'Close_MA20_Ratio', 'Close_MA50_Ratio', 'Close_VWAP_Ratio',
    'BB_Position', 'MA20_MA50_Ratio', 'ATR_Pct',
    'high', 'low', 'volume',
    'RSI', 'MACD', 'MACD_signal', 'OBV', 'CMF', 'Close_ATR',
    'MA20_ATR', 'Volatility', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment', 'Trend',
    'RSI_60', 'MA20_60', 'VWAP_Dev', 'Volume_Delta', 'Macro_Stress', 'Earnings_Proxy',
    'SPY_Ret', 'QQQ_Ret', 'TimeOfDay_Sin', 'TimeOfDay_Cos',
    'FracDiff_Close', 'Daily_Trend', 'Volume_Ratio',
]  # 37 features


def adaptive_buy_threshold(predictions, smoke: bool = False) -> float:
    """v15.4: per-symbol adaptive buy threshold — IDENTICAL formula to the backtest's
    inline logic (see backtest ~'Adaptive threshold for this backtest'). The live loop
    previously gated on a FIXED threshold (max(PREDICTION_THRESHOLD_BUY=0.7, ...)) that the
    calibrated model almost never reached (preds max ~0.57) → it NEVER traded, while the
    backtest/walk-forward (10.35 Sharpe) adaptively dropped to the 82nd percentile of the
    prediction distribution and traded ~18% of bars. This makes live match that validated
    behaviour. Returns the buy threshold for the given `predictions` distribution.
    """
    preds = np.asarray(predictions, dtype=np.float64).ravel()
    buy_th = float(CONFIG.get('TRIPLE_BARRIER_BUY_THRESHOLD',
                              CONFIG.get('PREDICTION_THRESHOLD_BUY', 0.52)))
    if preds.size == 0:
        return buy_th
    n_bars = len(preds)
    n_above = int(np.sum(preds >= buy_th))
    if n_above < max(5, int(n_bars * 0.03)):
        percentile = 65 if smoke else 82
        eff = float(np.percentile(preds, percentile))
        floor = 0.05 if smoke else 0.22
        buy_th = max(floor, min(buy_th, eff))
    return buy_th


def preprocess_data(df: pd.DataFrame, timesteps: int, add_noise: bool = False, inference_scaler: Optional[RobustScaler] = None, inference_mode: bool = False, fit_scaler: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[RobustScaler]]:
    df = df.copy()
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
   


    features = MODEL_FEATURES
    
    if 'Future_Direction' not in df.columns and not inference_mode:
        raise ValueError("Future_Direction column missing; required for training.")
    X_raw = df[features].values
   
    if fit_scaler:
        scaler = RobustScaler()
        X = scaler.fit_transform(X_raw)
    else:
        if inference_scaler is None:
            raise ValueError("inference_scaler must be provided when fit_scaler=False.")
        X = inference_scaler.transform(X_raw)
        scaler = None
   
    if not inference_mode:
        y = df['Future_Direction'].values
        y_seq = y
    else:
        y_seq = None
   
    if add_noise:
        X += np.random.normal(0, 0.005, X.shape)
   
    N = X.shape[0]
    num_sequences = N - timesteps
    if num_sequences <= 0:
        raise ValueError(f"Not enough data for {timesteps} timesteps: only {N} rows available")
   
    window = np.lib.stride_tricks.sliding_window_view(X, (timesteps, X.shape[1]))
    X_seq = window[:num_sequences].reshape(num_sequences, timesteps, X.shape[1])
   
    if not inference_mode:
        y_seq = y[timesteps - 1: timesteps - 1 + num_sequences]
        logger.info(f"Preprocessed {len(X_seq)} sequences; y balance: {np.mean(y_seq):.3f} (up fraction)")
    else:
        logger.info(f"Preprocessed {len(X_seq)} inference sequences")
   
    return X_seq, y_seq, scaler

def monte_carlo_simulation(returns: List[float], initial_cash: float, num_simulations: int = CONFIG['NUM_MC_SIMULATIONS']) -> Dict[str, float]:
    # bootstrap resample to get distribution of outcomes
    # returns mc_mean, mc_median, var_95, prob_profit
    if not returns:
        return {'mc_mean_final_value': initial_cash, 'mc_median_final_value': initial_cash, 'mc_var_95': 0.0, 'mc_prob_profit': 0.0}
    
    returns = np.array(returns)
    simulation_results = []
    for _ in range(num_simulations):
        # Bootstrap: resample returns with replacement
        sim_returns = np.random.choice(returns, size=len(returns), replace=True)
        sim_cumulative = np.cumprod(1 + sim_returns)
        sim_final_value = initial_cash * sim_cumulative[-1]
        simulation_results.append(sim_final_value)
    
    simulation_results = np.array(simulation_results)
    mc_mean_final_value = np.mean(simulation_results)
    mc_median_final_value = np.median(simulation_results)
    mc_var_95 = -np.percentile(simulation_results - initial_cash, 5) / initial_cash * 100  # 95% VaR as positive % loss
    mc_prob_profit = np.mean(simulation_results > initial_cash) * 100  # % simulations profitable
    
    return {
        'mc_mean_final_value': mc_mean_final_value,
        'mc_median_final_value': mc_median_final_value,
        'mc_var_95': mc_var_95,
        'mc_prob_profit': mc_prob_profit
    }

class TradingModel(nn.Module):
    def __init__(self, timesteps: int, features: int):
        super(TradingModel, self).__init__()
        self.timesteps = timesteps
        self.features = features
        self.hidden_size = 128


        # GPU footprint stays the same.
        self.lstm = nn.LSTM(features, self.hidden_size, num_layers=2,
                            batch_first=True, dropout=0.45)
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size,
                                               num_heads=8, dropout=0.25,
                                               batch_first=True)
        self.ln_lstm = nn.LayerNorm(self.hidden_size)
        self.ln_attn = nn.LayerNorm(self.hidden_size)
        self.dense1 = nn.Linear(self.hidden_size, 64)
        self.dense2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.40)
        
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)

    def forward(self, x):

        use_ckpt = False
        try:
            use_ckpt = CONFIG.get('USE_GRADIENT_CHECKPOINTING', False)
        except Exception:
            pass

        if use_ckpt and torch.is_grad_enabled():
            lstm_out, _ = checkpoint(self._lstm_forward, x, use_reentrant=False)
            attn_out, _ = checkpoint(self._attention_forward, lstm_out, lstm_out, lstm_out)
        else:
            lstm_out, _ = self._lstm_forward(x)
            attn_out, _ = self._attention_forward(lstm_out, lstm_out, lstm_out)

        x = attn_out[:, -1, :]  # last timestep
        x = self.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        return x

    def _lstm_forward(self, x):
        return self.lstm(x)

    def _attention_forward(self, q, k, v):
        return self.attention(q, k, v)

def train_model(symbol: str, worker_id: int, df: pd.DataFrame, epochs: int, batch_size: int, timesteps: int, expected_features: int, barrier=None, gpu_semaphore=None, preprocessed_train=None, preprocessed_val=None) -> Tuple[nn.Module, Any, GaussianHMM, Any]:
    if gpu_semaphore is not None:
        gpu_semaphore.acquire()
        logger.info(f"[{symbol}] Acquired gpu_semaphore (worker starting heavy GPU ops)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingModel(timesteps, expected_features).to(device)
    


    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'],
                           weight_decay=CONFIG.get('WEIGHT_DECAY', 1e-3))


    criterion = None  # deferred until y_train is known
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     patience=CONFIG['LR_SCHEDULER_PATIENCE'], 
                                                     factor=CONFIG['LR_REDUCTION_FACTOR'])

    def get_convergence_note(train_loss, val_loss, best_val_loss, current_lr):
        if val_loss < best_val_loss * 0.997:   # slightly easier to trigger green
            return f"{Fore.GREEN}✓ {Fore.LIGHTBLACK_EX}Both losses dropping — model is learning to guess stock direction better{Style.RESET_ALL}"
        elif val_loss > best_val_loss * 1.01 and train_loss < best_val_loss * 0.95:
            return f"{Fore.LIGHTBLACK_EX}⚠ Val loss rising while train drops — model is memorizing old data instead of learning new patterns{Style.RESET_ALL}"
        elif current_lr < CONFIG['LEARNING_RATE'] * 0.6:
            return f"{Fore.GREEN}✓ {Fore.LIGHTBLACK_EX}LR lowered — model now making smaller, more careful adjustments to avoid mistakes{Style.RESET_ALL}"
        else:
            return f"{Fore.GREEN}✓ {Fore.LIGHTBLACK_EX}Losses mostly stable — training continuing normally"
    
    # === Exact same data prep as before (no change) ===
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()

    train_end = pd.Timestamp(CONFIG['TRAIN_END_DATE'], tz='UTC').normalize()
    val_end   = pd.Timestamp(CONFIG['VAL_END_DATE'], tz='UTC').normalize()
    df_train = df[df.index <= train_end].copy()
    df_val   = df[(df.index > train_end) & (df.index <= val_end)].copy()


    if CONFIG.get('USE_TRIPLE_BARRIER', True):


        _tr_lab, _tr_ta = compute_triple_barrier_label(
            df_train['close'].values, df_train['ATR'].values,
            CONFIG['LOOK_AHEAD_BARS'], CONFIG['TB_TP_ATR'], CONFIG['TB_SL_ATR'],
            return_touch_ahead=True)
        df_train['Future_Direction'] = _tr_lab
        df_train['_tb_touch_ahead'] = _tr_ta
        df_val['Future_Direction'] = compute_triple_barrier_label(
            df_val['close'].values, df_val['ATR'].values,
            CONFIG['LOOK_AHEAD_BARS'], CONFIG['TB_TP_ATR'], CONFIG['TB_SL_ATR'])
    else:
        df_train['Future_Direction'] = np.where(df_train['close'].shift(-CONFIG['LOOK_AHEAD_BARS']) > df_train['close'], 1, 0)
        df_val['Future_Direction'] = np.where(df_val['close'].shift(-CONFIG['LOOK_AHEAD_BARS']) > df_val['close'], 1, 0)
    df_train = df_train.dropna(subset=['Future_Direction'])
    df_val = df_val.dropna(subset=['Future_Direction'])

    X_train, y_train, scaler = preprocess_data(df_train, CONFIG['TIMESTEPS'], add_noise=True)
    X_val,   y_val,   _      = preprocess_data(df_val,   CONFIG['TIMESTEPS'], inference_scaler=scaler, inference_mode=False, fit_scaler=False)


    _n_pos = float(np.sum(y_train == 1))
    _n_neg = float(np.sum(y_train == 0))
    if _n_pos > 0:
        _pw = _n_neg / _n_pos
    else:
        _pw = 1.0
    _pw = float(np.clip(_pw, 1.0, CONFIG.get('MAX_POS_WEIGHT', 8.0)))
    pos_weight = torch.tensor([_pw], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    criterion_per = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction='none')
    logger.info(f"[{symbol}] Dynamic pos_weight={_pw:.3f} (N_pos={int(_n_pos)}, N_neg={int(_n_neg)})")


    train_weights = np.ones(len(y_train), dtype=np.float32)
    if CONFIG.get('USE_SAMPLE_UNIQUENESS', True) and '_tb_touch_ahead' in df_train.columns:
        try:
            _ta_full = df_train['_tb_touch_ahead'].values
            _ts = CONFIG['TIMESTEPS']
            _span = _ta_full[_ts - 1: _ts - 1 + len(y_train)]
            if len(_span) == len(y_train):
                train_weights = compute_average_uniqueness(_span)
                logger.info(f"[{symbol}] Sample-uniqueness weights: mean={train_weights.mean():.3f} "
                            f"min={train_weights.min():.3f} max={train_weights.max():.3f}")
            else:
                logger.warning(f"[{symbol}] uniqueness span/label length mismatch "
                               f"({len(_span)} vs {len(y_train)}); using uniform weights")
        except Exception as _ue:
            logger.warning(f"[{symbol}] sample-uniqueness weighting failed ({_ue}); uniform weights")

    X_train_pinned = torch.from_numpy(X_train.astype(np.float32)).pin_memory()
    y_train_pinned = torch.from_numpy(y_train.astype(np.float32)).pin_memory()
    w_train_pinned = torch.from_numpy(train_weights.astype(np.float32)).pin_memory()
    X_val_pinned   = torch.from_numpy(X_val.astype(np.float32)).pin_memory()
    y_val_pinned   = torch.from_numpy(y_val.astype(np.float32)).pin_memory()

    train_dataset = TensorDataset(X_train_pinned, y_train_pinned, w_train_pinned)
    val_dataset   = TensorDataset(X_val_pinned, y_val_pinned)


    import multiprocessing as _mp
    _current = _mp.current_process()
    _is_inside_worker = (_current.name != 'MainProcess') or getattr(_current, 'daemon', False)

    if _is_inside_worker:
        dl_num_workers = 0
        dl_prefetch = None
        dl_persistent = False
    else:
        dl_num_workers = CONFIG.get('DATALOADER_NUM_WORKERS', 0)
        dl_prefetch    = CONFIG.get('PREFETCH_FACTOR', 2)
        dl_persistent  = CONFIG.get('PERSISTENT_WORKERS', True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=dl_num_workers,
        pin_memory=True,
        persistent_workers=dl_persistent and dl_num_workers > 0,
        prefetch_factor=dl_prefetch if dl_num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None
    )

    scaler_amp = torch.amp.GradScaler('cuda')

    print(f"{Fore.CYAN}[{symbol}]{Fore.LIGHTCYAN_EX} Slave {worker_id}{Fore.CYAN} started GPU training with compile + streams + prefetch{Style.RESET_ALL}")


    best_val_loss = float('inf')
    patience_counter = 0
    best_state_dict = None          # <-- the actual fix
    best_epoch = 0

    use_checkpoint = CONFIG.get('USE_GRADIENT_CHECKPOINTING', False)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y, batch_w in train_loader:
            batch_X = batch_X.to(device, non_blocking=True).float()
            batch_y = batch_y.to(device, non_blocking=True).float()
            batch_w = batch_w.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                if use_checkpoint:
                    outputs = model(batch_X)  # checkpoint inside forward
                else:
                    outputs = model(batch_X)

                per_sample = criterion_per(outputs.squeeze(1), batch_y)
                loss = (per_sample * batch_w).sum() / batch_w.sum().clamp_min(1e-8)
            
            scaler_amp.scale(loss).backward()


            _clip = CONFIG.get('GRAD_CLIP_NORM', 0.0)
            if _clip and _clip > 0:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), _clip)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device).float()
                batch_y = batch_y.to(device).float()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(1), batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        note = get_convergence_note(train_loss, val_loss, best_val_loss, current_lr)

        print(f"[{symbol}] Epoch {epoch+1:02d}/{epochs} |{Fore.LIGHTBLACK_EX} "
                f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                f"LR: {current_lr:.2e} | {note}{Style.RESET_ALL}")

        scheduler.step(val_loss)


        improved = val_loss < (best_val_loss - CONFIG['EARLY_STOPPING_MIN_DELTA'])
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[{symbol}]   ↳ New best val_loss @ epoch {best_epoch} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['EARLY_STOPPING_PATIENCE']:
                print(f"[{symbol}] Early stopping at epoch {epoch+1} (best was {best_epoch})")
                break


    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"{Fore.GREEN}[{symbol}] Restored BEST weights from epoch {best_epoch} (val_loss={best_val_loss:.6f}){Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}[{symbol}] No improvement recorded — returning final weights{Style.RESET_ALL}")


    model._calib_temp = 1.0
    if CONFIG.get('ENABLE_TEMPERATURE_CALIBRATION', True):
        try:
            model.eval()
            with torch.no_grad():
                _val_logits = []
                for _xb, _ in DataLoader(val_dataset, batch_size=batch_size, shuffle=False):
                    _val_logits.append(model(_xb.to(device)).squeeze(-1).detach().cpu())
            _vl = torch.cat(_val_logits).view(-1).float()
            _vy = y_val_pinned.cpu().view(-1).float()
            _n = min(_vl.shape[0], _vy.shape[0])
            _vl, _vy = _vl[:_n], _vy[:_n]
            best_T, best_nll = 1.0, float('inf')


            _calib_floor = float(CONFIG.get('CALIB_TEMP_FLOOR', 0.5))
            for _T in np.linspace(_calib_floor, 3.0, 51):
                _loss = nn.functional.binary_cross_entropy_with_logits(_vl / float(_T), _vy).item()
                if _loss < best_nll:
                    best_nll, best_T = _loss, float(_T)
            model._calib_temp = best_T
            print(f"{Fore.GREEN}[{symbol}] Calibration temperature T={best_T:.3f} (val NLL {best_nll:.4f}){Style.RESET_ALL}")
        except Exception as _ce:
            logger.warning(f"[{symbol}] temperature calibration failed ({_ce}); using T=1.0")
            model._calib_temp = 1.0

    print(f"{Fore.CYAN}[{symbol}] LSTM + Attention training COMPLETE — switching to CPU for HMM + XGBoost ensemble{Style.RESET_ALL}")


    print(f"{Fore.CYAN}[{symbol}] Worker {worker_id} → Starting HMM regime detection on CPU...{Style.RESET_ALL}")
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    finally:


        if gpu_semaphore is not None:
            try:
                gpu_semaphore.release()
                print(f"{Fore.BLUE}[{symbol}]{Fore.LIGHTCYAN_EX} Slave {worker_id}{Fore.BLUE} → RELEASED GPU {Style.RESET_ALL}")
                logger.debug(f"[{symbol}] Released gpu semaphore after training")
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to release gpu_semaphore: {str(e)}")


    X_train = X_train_pinned.cpu().numpy()
    y_train = y_train_pinned.cpu().numpy()
    X_val   = X_val_pinned.cpu().numpy()
    y_val   = y_val_pinned.cpu().numpy()


    # transition matrix garbled at each window boundary.
    print(f"{Fore.CYAN}[{symbol}] {Fore.LIGHTCYAN_EX}Slave {worker_id}{Fore.CYAN} → Training HMM regime model on CPU...{Style.RESET_ALL}")
    if CONFIG.get('HMM_FIT_DEDUP', True):
        hmm_input = X_train[:, -1, :]                       # (n_bars, features), ordered
    else:
        hmm_input = X_train.reshape(-1, X_train.shape[2])   # legacy flattened windows
    hmm = train_hmm(hmm_input)


    try:
        hmm.regime_label_map = assign_regime_labels(hmm, scaler)
        logger.info(f"[{symbol}] HMM regime labels: {hmm.regime_label_map}")
    except Exception as _le:
        logger.warning(f"[{symbol}] regime label assignment failed ({_le}); will use legacy positional names")
        hmm.regime_label_map = {}
    print(f"{Fore.BLUE}[{symbol}] {Fore.LIGHTCYAN_EX}Slave {worker_id}{Fore.BLUE} → HMM regime detection COMPLETE "
          f"({len(getattr(hmm, 'regime_label_map', {}) or {})} regimes labelled by return/volatility){Style.RESET_ALL}")


    _xgb_dev = 'GPU' if (CONFIG.get('XGBOOST_DEVICE', 'cuda') == 'cuda' and torch.cuda.is_available()) else 'CPU'
    print(f"{Fore.CYAN}[{symbol}] {Fore.LIGHTCYAN_EX}Slave {worker_id}{Fore.CYAN} → Training XGBoost ensemble model on {_xgb_dev}...{Style.RESET_ALL}")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat   = X_val.reshape(X_val.shape[0], -1)
    _xgb_sw = train_weights if (CONFIG.get('USE_SAMPLE_UNIQUENESS', True)
                                and len(train_weights) == len(y_train)) else None
    xgb_model = train_xgboost(X_train_flat, y_train, X_val_flat, y_val, sample_weight=_xgb_sw)
    print(f"{Fore.BLUE}[{symbol}]{Fore.LIGHTCYAN_EX} Slave {worker_id}{Fore.BLUE} → CPU phase finished{Style.RESET_ALL}")


    return model, scaler, hmm, xgb_model

def _sanitize_hmm(hmm: GaussianHMM) -> GaussianHMM:
    """v15: repair a degenerate HMM fit so .predict() can't raise later.

    EM occasionally converges to NaN parameters on thin/odd data; the symptom was the
    recurring 'startprob_ must sum to 1 (got nan)' warning in get_pair_regime(). We
    replace NaN/non-normalized start & transition probabilities with uniform rows and
    scrub NaN emission params, so the HMM is always usable (worst case: uninformative).
    """
    try:
        n = hmm.n_components
        sp = np.asarray(getattr(hmm, 'startprob_', np.full(n, 1.0 / n)), dtype=np.float64)
        if sp.shape != (n,) or not np.all(np.isfinite(sp)) or abs(sp.sum() - 1.0) > 1e-6 or np.any(sp < 0):
            hmm.startprob_ = np.full(n, 1.0 / n)
        tm = np.asarray(getattr(hmm, 'transmat_', np.full((n, n), 1.0 / n)), dtype=np.float64)
        if tm.shape != (n, n) or not np.all(np.isfinite(tm)):
            hmm.transmat_ = np.full((n, n), 1.0 / n)
        else:
            bad = (np.abs(tm.sum(axis=1) - 1.0) > 1e-6) | np.any(tm < 0, axis=1)
            if np.any(bad):
                tm[bad] = 1.0 / n
                hmm.transmat_ = tm
        if hasattr(hmm, 'means_'):
            hmm.means_ = np.nan_to_num(np.asarray(hmm.means_, dtype=np.float64), nan=0.0)
        if hasattr(hmm, 'covars_'):
            _cv = np.asarray(hmm._covars_, dtype=np.float64)
            if not np.all(np.isfinite(_cv)) or np.any(_cv <= 0):
                _cv = np.nan_to_num(_cv, nan=1.0)
                _cv[_cv <= 0] = 1e-6
                hmm._covars_ = _cv
    except Exception as _se:
        logger.warning(f"_sanitize_hmm failed ({_se}); leaving HMM as-is")
    return hmm


def train_hmm(X_scaled: np.ndarray, num_regimes: int = CONFIG['NUM_REGIMES']) -> GaussianHMM:


    n_iter = CONFIG.get('HMM_N_ITER', 100)
    max_rows = CONFIG.get('HMM_MAX_FIT_ROWS', 150000)
    if X_scaled.shape[0] > max_rows:
        X_scaled = X_scaled[:max_rows]
    try:
        hmm = GaussianHMM(
            n_components=num_regimes,
            covariance_type="diag",      #Much more stable
            n_iter=n_iter,
            tol=1e-3,                    # explicit convergence tol (early-stops the EM loop)
            random_state=42,
            min_covar=1e-6,              #Prevents singular matrices
            params="stmc",               #Only learn safe parameters
            init_params="stmc"
        )
        hmm.fit(X_scaled)
        _sanitize_hmm(hmm)
        logger.info(f"Trained HMM with {num_regimes} regimes (stable diagonal covariance)")
        return hmm
    except Exception as e:
        logger.warning(f"6-regime HMM failed ({e}). Falling back to 4 regimes.")
        hmm = GaussianHMM(               #Safe fallback
            n_components=4,
            covariance_type="diag",
            n_iter=n_iter,
            tol=1e-3,
            random_state=42,
            min_covar=1e-6
        )
        hmm.fit(X_scaled)
        _sanitize_hmm(hmm)
        logger.info("Fallback to 4 regimes successful")
        return hmm


# computed once at import.
_HMM_RETURN_FEATURE_IDX = MODEL_FEATURES.index('Return_1d')      # bull/bear axis
_HMM_VOLATILITY_FEATURE_IDX = MODEL_FEATURES.index('Volatility')  # calm/volatile axis

def assign_regime_labels(hmm, scaler=None,
                         ret_idx: int = _HMM_RETURN_FEATURE_IDX,
                         vol_idx: int = _HMM_VOLATILITY_FEATURE_IDX) -> dict:
    """Map each HMM hidden-state index → a SEMANTIC regime name.

    v14 CRITICAL FIX: HMM state indices are arbitrary — state 0 is NOT inherently
    "Calm Bull".  The old code mapped state→name with `names[idx % 6]`, so the name a
    state received was pure luck of EM initialisation.  Because BUY_REGIME_WHITELIST only
    admits {Calm Bull, Moderate Bull}, that randomness meant most symbols got pinned to a
    non-whitelisted name and NEVER traded (smoke: 7/8 symbols stuck on one constant
    non-bull regime).

    Fix: name each state from what it actually represents, using the HMM's learned
    per-state feature means.  Direction = sign of mean 1-bar return (Bull/Bear); intensity
    = volatility rank within that direction (Calm < Moderate < Volatile).  RobustScaler is
    monotonic, so we recover the raw (unscaled) means via center_/scale_ to get a true
    sign for the Bull/Bear split.

    Returns {state_index: "Tier Direction"} and is robust to any n_states (incl. the
    4-regime fallback) and to lopsided Bull/Bear splits.
    """
    try:
        means = np.asarray(hmm.means_)               # (n_states, n_features)
    except Exception:
        return {}
    n_states, n_features = means.shape
    ri = ret_idx if ret_idx < n_features else min(n_features - 1, 0)
    vi = vol_idx if vol_idx < n_features else min(n_features - 1, 0)


    if scaler is not None and hasattr(scaler, 'scale_') and hasattr(scaler, 'center_'):
        raw_ret = means[:, ri] * scaler.scale_[ri] + scaler.center_[ri]
        raw_vol = means[:, vi] * scaler.scale_[vi] + scaler.center_[vi]
    else:
        raw_ret = means[:, ri]
        raw_vol = means[:, vi]

    label_map = {}
    for direction in ("Bull", "Bear"):
        idxs = [s for s in range(n_states)
                if (raw_ret[s] >= 0) == (direction == "Bull")]
        if not idxs:
            continue
        ordered = sorted(idxs, key=lambda s: raw_vol[s])   # low vol → high vol
        m = len(ordered)
        for rank, s in enumerate(ordered):
            frac = rank / (m - 1) if m > 1 else 0.5
            tier = "Calm" if frac < 1.0 / 3 else ("Moderate" if frac < 2.0 / 3 else "Volatile")
            label_map[int(s)] = f"{tier} {direction}"
    return label_map

def regime_name_for_state(hmm, state: int) -> str:
    """Look up a state's semantic regime name via the map attached at train time.
    Falls back to the legacy positional naming for old HMMs that predate the map."""
    _map = getattr(hmm, 'regime_label_map', None)
    if _map:
        return _map.get(int(state), "Unknown")
    _legacy = ["Calm Bull", "Moderate Bull", "Volatile Bull",
               "Calm Bear", "Moderate Bear", "Volatile Bear"]
    return _legacy[int(state) % len(_legacy)]

def confidence_size_mult(pred: float, threshold: float) -> float:
    """v15: scale the risk fraction by how far `pred` clears `threshold`.

    Returns 1.0 for a marginal signal (pred≈threshold) and up to
    CONFIG['CONFIDENCE_SIZE_MAX_MULT'] at full conviction (pred→1.0). Bounded so a
    pred below threshold (shouldn't happen at a buy site) never shrinks below 1.0.
    No-op (returns 1.0) when ENABLE_CONFIDENCE_SIZING is False.
    """
    if not CONFIG.get('ENABLE_CONFIDENCE_SIZING', True):
        return 1.0
    max_mult = float(CONFIG.get('CONFIDENCE_SIZE_MAX_MULT', 2.5))
    denom = max(1e-6, 1.0 - float(threshold))
    conf = (float(pred) - float(threshold)) / denom
    conf = min(1.0, max(0.0, conf))
    return 1.0 + conf * (max_mult - 1.0)


def train_xgboost(X_train, y_train, X_val, y_val, sample_weight=None):


    want_device = CONFIG.get('XGBOOST_DEVICE', 'cuda')
    use_gpu = (want_device == 'cuda' and torch.cuda.is_available())
    device = 'cuda' if use_gpu else 'cpu'


    if sample_weight is not None:
        sw = np.ascontiguousarray(np.asarray(sample_weight, dtype=np.float32)).ravel()
        if sw.shape[0] != np.asarray(X_train).shape[0] or not np.all(np.isfinite(sw)) or np.all(sw <= 0):
            logger.warning("train_xgboost: invalid sample_weight (shape/NaN/zero) — ignoring it")
            sample_weight = None
        else:
            sample_weight = sw
            if device == 'cuda':
                device = 'cpu'  # weighted fit on CPU to avoid the GPU+weights crash


    _xn_pos = float(np.sum(np.asarray(y_train) == 1))
    _xn_neg = float(np.sum(np.asarray(y_train) == 0))
    _spw = (_xn_neg / _xn_pos) if _xn_pos > 0 else 1.0
    _spw = float(np.clip(_spw, 1.0, CONFIG.get('MAX_POS_WEIGHT', 8.0)))
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist',
        device=device,
        scale_pos_weight=_spw,


        n_jobs=(CONFIG.get('XGBOOST_N_JOBS', 0) or int(os.environ.get('DT_THREADS_PER_WORKER', '0')) or 4),
    )

    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)


    try:
        model.get_booster().set_param({'device': 'cpu'})
    except Exception:
        pass
    return model

def load_model_and_scaler(symbol: str, expected_features: int, force_retrain: bool = False) -> Tuple[Optional[nn.Module], Optional[RobustScaler], Optional[float], Optional[GaussianHMM], Optional[Any]]:
    logger.info(f"Entering load_model_and_scaler for {symbol} (force_retrain={force_retrain}).")
    if force_retrain:
        return None, None, None, None, None

    model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
    scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
    sentiment_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_news_sentiment_{CONFIG['MODEL_VERSION']}.pkl")
    hmm_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_hmm_{CONFIG['MODEL_VERSION']}.pkl")
    xgb_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_xgb_{CONFIG['MODEL_VERSION']}.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(hmm_path):
        logger.info(f"Found model, scaler, and HMM for {symbol}.")
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(hmm_path, 'rb') as f:
                hmm = pickle.load(f)

            checkpoint = torch.load(model_path, map_location='cpu')
            model = TradingModel(CONFIG['TIMESTEPS'], expected_features)
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)

                model._calib_temp = float(checkpoint.get('calib_temp', 1.0)) if isinstance(checkpoint, dict) else 1.0
            else:
                model._calib_temp = 1.0

            training_sentiment = None
            if os.path.exists(sentiment_path):
                with open(sentiment_path, 'rb') as f:
                    training_sentiment = pickle.load(f)


            xgb_model = None
            if os.path.exists(xgb_path):
                try:
                    with open(xgb_path, 'rb') as f:
                        xgb_model = pickle.load(f)
                    logger.info(f"Loaded XGBoost model for {symbol}")
                except Exception as xe:
                    logger.warning(f"Failed to load XGBoost for {symbol}: {xe} — will run LSTM-only")

            logger.info(f"Successfully loaded cached model, scaler, sentiment, and HMM for {symbol}.")
            return model, scaler, training_sentiment, hmm, xgb_model
        except Exception as e:
            logger.error(f"Failed to load for {symbol}: {str(e)}. Retraining.")
            return None, None, None, None, None
    else:
        logger.info(f"No cached model/scaler/HMM for {symbol}. Will train.")
        return None, None, None, None, None

def save_model_and_scaler(symbol: str, model: nn.Module, scaler: RobustScaler, sentiment: float, hmm: GaussianHMM, xgb_model=None) -> None:
    try:
        model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
        scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
        sentiment_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_news_sentiment_{CONFIG['MODEL_VERSION']}.pkl")
        hmm_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_hmm_{CONFIG['MODEL_VERSION']}.pkl")
        xgb_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_xgb_{CONFIG['MODEL_VERSION']}.pkl")

        torch.save({'model_state_dict': model.state_dict(), 'class_name': 'TradingModel',
                    'calib_temp': float(getattr(model, '_calib_temp', 1.0))}, model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(sentiment_path, 'wb') as f:
            pickle.dump(sentiment, f)
        with open(hmm_path, 'wb') as f:
            pickle.dump(hmm, f)

        if xgb_model is not None:
            with open(xgb_path, 'wb') as f:
                pickle.dump(xgb_model, f)

        windows_model_path = model_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
        logger.info(f"Saved model, scaler, sentiment, HMM, and XGBoost for {symbol} (Windows: {windows_model_path})")
    except Exception as e:
        logger.error(f"Failed to save for {symbol}: {str(e)}")
        raise


def train_symbol(symbol: str, worker_id: int, expected_features: int, force_train: bool, barrier=None, gpu_semaphore=None, preprocessed_train=None, preprocessed_val=None, backtest_only: bool = False, debug: bool = False) -> Tuple[str, nn.Module, Any, bool, float, bool, bool, GaussianHMM, Any]:


    try:
        _tpw = int(os.environ.get('DT_THREADS_PER_WORKER', '0'))
        if _tpw > 0:
            torch.set_num_threads(_tpw)
    except Exception:
        pass


    model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
    will_train = force_train or not os.path.exists(model_path)

    if will_train and not backtest_only:
        print(f"{Fore.CYAN}[{symbol}] === STARTING Training with {Fore.LIGHTCYAN_EX}Slave {worker_id}{Fore.CYAN} ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[{symbol}] Fetching FULL training data from {CONFIG['TRAIN_DATA_START_DATE']} to {CONFIG['VAL_END_DATE']}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[{symbol}] Worker {worker_id} → Waiting for GPU slot{Style.RESET_ALL}")


    if force_train:
        logger.debug(f"[{symbol}] --force-train: Fetching fresh from API (no cache)")
        df = fetch_data(symbol, CONFIG['TRAIN_DATA_START_DATE'], CONFIG['VAL_END_DATE'])
        data_loaded = False
    else:
        df, data_loaded = load_or_fetch_data(symbol, CONFIG['TRAIN_DATA_START_DATE'], CONFIG['VAL_END_DATE'])

    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    logger.debug(f"[{symbol}] After fetch + index fix: {len(df)} rows, min={df.index.min()}, max={df.index.max()}")

    sentiment, sentiment_loaded = load_news_sentiment(symbol)
    df = calculate_indicators(df, sentiment)
    logger.debug(f"[{symbol}] After calculate_indicators: {len(df)} rows")
    model_file = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
    scaler_file = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")

    if not force_train and os.path.exists(model_file) and os.path.exists(scaler_file):
        model, scaler, training_sentiment, hmm, xgb_model_cached = load_model_and_scaler(symbol, expected_features, force_retrain=False)
        if model is not None:
            model_loaded = True
            xgb_model = xgb_model_cached  # may be None for old cache files — that is fine
            logger.info(f"Loaded cached model, scaler, and HMM for {symbol}")
        else:
            force_train = True
            model_loaded = False
    else:
        model_loaded = False

    if force_train or not model_loaded:
        model, scaler, hmm, xgb_model = train_model(
            symbol, 
            worker_id,
            df, 
            CONFIG['TRAIN_EPOCHS'], 
            CONFIG['BATCH_SIZE'], 
            CONFIG['TIMESTEPS'], 
            expected_features, 
            barrier, 
            gpu_semaphore, 
            preprocessed_train, 
            preprocessed_val
        )
        save_model_and_scaler(symbol, model, scaler, sentiment, hmm, xgb_model)
        model_loaded = False
        logger.info(f"Trained and saved new model for {symbol}")
    else:
        xgb_model = None


    return symbol, model, scaler, data_loaded, sentiment, sentiment_loaded, model_loaded, hmm, xgb_model

def backtest(symbol: str, model: nn.Module, scaler: RobustScaler, df: pd.DataFrame, initial_cash: float,
             stop_loss_atr_multiplier: float, take_profit_atr_multiplier: float, timesteps: int,
             buy_threshold: float, sell_threshold: float, min_holding_period_minutes: int,
             transaction_cost_per_trade: float, xgb_model=None, hmm=None,
             dfs_backtest: Dict[str, pd.DataFrame] = None, hmms: Dict[str, GaussianHMM] = None,
             scalers: Dict[str, RobustScaler] = None, debug: bool = False) -> Tuple[float, List[float], int, float, float, pd.Series]:

    if os.getenv("OPTIMIZER_MODE") == "true":
        CONFIG['NUM_MC_SIMULATIONS'] = 5000   # instead of 50,000
        print("🚀 OPTIMIZER MODE: Reduced Monte Carlo to 5k for 10-hour run")

    if debug:
        print(f"{Fore.GREEN}=== NEW BACKTEST FUNCTION LOADED FOR {symbol} — DEBUG FORCE ENABLED ==={Style.RESET_ALL}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running backtest for {symbol} on device: {device}")
    
    confidence_threshold = CONFIG['CONFIDENCE_THRESHOLD']
    rsi_buy_threshold = CONFIG['RSI_BUY_THRESHOLD']
    rsi_sell_threshold = CONFIG['RSI_SELL_THRESHOLD']
    adx_trend_threshold = CONFIG['ADX_TREND_THRESHOLD']
    max_volatility = CONFIG['MAX_VOLATILITY']
    trailing_stop_percentage = CONFIG['TRAILING_STOP_PERCENTAGE']
    risk_percentage = CONFIG['RISK_PERCENTAGE']


    if CONFIG.get('USE_TRIPLE_BARRIER', True):
        buy_threshold = CONFIG.get('TRIPLE_BARRIER_BUY_THRESHOLD', 0.58)
        sell_threshold = CONFIG.get('TRIPLE_BARRIER_SELL_THRESHOLD', 0.42)
        confidence_threshold = buy_threshold


        if CONFIG.get('SMOKE_TEST', False):
            buy_threshold = min(buy_threshold, 0.35)
            sell_threshold = max(sell_threshold, 0.65)
            confidence_threshold = buy_threshold
            print(f"{Fore.YELLOW}[SMOKE] Using extra-relaxed thresholds for smoke test (buy>={buy_threshold:.2f}) "
                  f"so we can actually see trades and validate the full decision pipeline (stops, exits, risk, etc.).{Style.RESET_ALL}")
    

    backtest_start = pd.Timestamp(CONFIG['BACKTEST_START_DATE'], tz='UTC')
    _bt_end_cfg = CONFIG.get('BACKTEST_END_DATE')
    if _bt_end_cfg:
        _bt_end_ts = pd.Timestamp(_bt_end_cfg, tz='UTC')
        df_backtest = df[(df.index >= backtest_start) & (df.index <= _bt_end_ts)].copy()
    else:
        df_backtest = df[df.index >= backtest_start].copy()
    if len(df_backtest) < CONFIG['MIN_DATA_POINTS']:
        raise ValueError(f"Insufficient backtest data for {symbol}: {len(df_backtest)} bars")
    


    X_seq, _, _ = preprocess_data(df_backtest, timesteps, inference_mode=True,
                                  inference_scaler=scaler, fit_scaler=False)
    
    model.eval()
    model = model.to(device)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)

    predictions = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), CONFIG['BATCH_SIZE']):
            batch = X_tensor[i:i + CONFIG['BATCH_SIZE']]
            raw_logits = model(batch)


            _temp = getattr(model, '_calib_temp', None) or CONFIG['PREDICTION_TEMPERATURE']
            scaled_logits = raw_logits / _temp
            outputs = torch.sigmoid(scaled_logits)
            predictions.extend(outputs.cpu().numpy().flatten())
            del raw_logits, scaled_logits, outputs

    predictions = np.array(predictions)


    #     pred = (lstm_sigmoid + xgb_predict_proba) / 2


    # loop gates on.
    if xgb_model is not None and hasattr(xgb_model, 'predict_proba'):
        try:
            xgb_probs = xgb_model.predict_proba(X_seq.reshape(X_seq.shape[0], -1))[:, 1]
            n_blend = min(len(predictions), len(xgb_probs))
            _w = float(CONFIG.get('BLEND_LSTM_WEIGHT', 0.6))
            predictions[:n_blend] = _w * predictions[:n_blend] + (1.0 - _w) * xgb_probs[:n_blend]
            print(f"[{symbol}] Diagnostics/threshold use LSTM+XGB blend (w_lstm={_w:.2f}, parity with trading loop).")
        except Exception as _blend_e:
            logger.warning(f"[{symbol}] XGB blend for diagnostics failed ({_blend_e}); "
                           f"adaptive threshold falls back to LSTM-only distribution")


    try:
        tb = CONFIG.get('USE_TRIPLE_BARRIER', True)
        buy_th = CONFIG.get('TRIPLE_BARRIER_BUY_THRESHOLD', CONFIG['PREDICTION_THRESHOLD_BUY']) if tb else CONFIG['PREDICTION_THRESHOLD_BUY']
        sell_th = CONFIG.get('TRIPLE_BARRIER_SELL_THRESHOLD', CONFIG['PREDICTION_THRESHOLD_SELL']) if tb else CONFIG['PREDICTION_THRESHOLD_SELL']
        n_bars = len(predictions)
        n_above_buy = int(np.sum(predictions >= buy_th))
        n_below_sell = int(np.sum(predictions <= sell_th))
        print(f"[{symbol}] Pred dist: min={predictions.min():.3f} p05={np.percentile(predictions,5):.3f} "
              f"mean={predictions.mean():.3f} p95={np.percentile(predictions,95):.3f} max={predictions.max():.3f}")
        print(f"[{symbol}] Would-trigger: {n_above_buy}/{n_bars} bars >= {buy_th:.2f} (buy), "
              f"{n_below_sell}/{n_bars} <= {sell_th:.2f} (sell)  [triple_barrier={tb}]")


        if n_above_buy < max(5, int(n_bars * 0.03)):  # less than ~3% of bars or <5 signals
            original_buy_th = buy_th


            percentile = 65 if CONFIG.get('SMOKE_TEST', False) else 82
            effective_buy_th = float(np.percentile(predictions, percentile))


            floor = 0.05 if CONFIG.get('SMOKE_TEST', False) else 0.22
            buy_th = max(floor, min(buy_th, effective_buy_th))
            confidence_threshold = buy_th
            print(f"{Fore.YELLOW}[{symbol}] Adaptive threshold for this backtest: lowered from {original_buy_th:.3f} → {buy_th:.3f} "
                  f"({percentile}th percentile) so we can validate the full strategy logic.{Style.RESET_ALL}")


        # so the full exit pipeline is exercised.
        if CONFIG.get('SMOKE_TEST', False) and tb:
            sell_th = max(sell_th, 0.65)
        buy_threshold = buy_th
        sell_threshold = sell_th
        confidence_threshold = buy_th
    except Exception:
        buy_threshold = CONFIG.get('TRIPLE_BARRIER_BUY_THRESHOLD', 0.58)
        sell_threshold = CONFIG.get('TRIPLE_BARRIER_SELL_THRESHOLD', 0.42)
        confidence_threshold = buy_threshold


    if CONFIG.get('USE_TRIPLE_BARRIER', True):
        df_backtest['Future_Direction'] = compute_triple_barrier_label(
            df_backtest['close'].values, df_backtest['ATR'].values,
            CONFIG['LOOK_AHEAD_BARS'], CONFIG['TB_TP_ATR'], CONFIG['TB_SL_ATR'])
    else:
        df_backtest['Future_Direction'] = np.where(
            df_backtest['close'].shift(-CONFIG['LOOK_AHEAD_BARS']) > df_backtest['close'], 1, 0
        )
    df_backtest = df_backtest.dropna(subset=['Future_Direction'])

    true_y_for_accuracy = df_backtest['Future_Direction'].iloc[CONFIG['TIMESTEPS']: CONFIG['TIMESTEPS'] + len(predictions)].values
    min_len = min(len(predictions), len(true_y_for_accuracy))
    predictions_acc = predictions[:min_len]
    true_y_acc = true_y_for_accuracy[:min_len]

    valid_mask = ~np.isnan(true_y_acc)
    accuracy_percentage = np.mean((predictions_acc[valid_mask] > 0.5) == true_y_acc[valid_mask]) * 100 if np.any(valid_mask) else 0.0

    logger.info(f"Predictions for {symbol}: min={predictions.min():.3f}, max={predictions.max():.3f}, mean={predictions.mean():.3f}")
    logger.info(f"Backtest accuracy for {symbol}: {accuracy_percentage:.2f}%")

    del X_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cash = initial_cash
    returns = []
    trade_count = 0
    win_rate = 0.0
    position = 0
    entry_price = 0.0
    entry_time = None
    max_price = 0.0
    winning_trades = 0
    pair_positions = {}


    sim_start = pd.Timestamp(CONFIG['BACKTEST_START_DATE'], tz='UTC')
    backtest_mask = df.index >= sim_start
    if CONFIG.get('BACKTEST_END_DATE'):
        backtest_mask &= df.index <= pd.Timestamp(CONFIG['BACKTEST_END_DATE'], tz='UTC')
    backtest_df = df[backtest_mask].iloc[timesteps:]  # align with prediction length
    num_backtest_steps = len(predictions)
    if num_backtest_steps == 0:
        logger.warning(f"No backtest steps for {symbol} — skipping")
        return initial_cash, [], 0, 0.0, accuracy_percentage, pd.Series()

    atr = backtest_df['ATR'].values[:num_backtest_steps]
    prices = backtest_df['close'].values[:num_backtest_steps]
    rsi = backtest_df['RSI'].values[:num_backtest_steps]
    adx = backtest_df['ADX'].values[:num_backtest_steps]
    volatility = backtest_df['Volatility'].values[:num_backtest_steps]
    volume = backtest_df['volume'].values[:num_backtest_steps]
    ma20 = backtest_df['MA20'].values[:num_backtest_steps]
    ma50 = backtest_df['MA50'].values[:num_backtest_steps]
    macd_arr = backtest_df['MACD'].values[:num_backtest_steps]
    macd_sig = backtest_df['MACD_signal'].values[:num_backtest_steps]
    sim_timestamps = backtest_df.index.values[:num_backtest_steps]


    # below runs on the REMAINING (tactical) cash.


    _core_pct = float(CONFIG.get('CORE_LONG_PCT', 0.0))
    _core_active = False
    _core_qty = 0
    _core_entry_i = 0
    _core_budget = 0.0
    _core_remainder = 0.0
    if _core_pct > 0.0 and num_backtest_steps > 2 and prices[0] > 0:
        _K = int(CONFIG.get('CORE_CONF_LOOKBACK', 10))
        _K = max(1, min(_K, num_backtest_steps - 1))
        _core_entry_i = _K - 1
        _pw = np.asarray(predictions[:_K], dtype=np.float64)
        _pred_mean = float(np.nanmean(_pw)) if _pw.size else 0.0
        _denom = max(1e-6, 1.0 - float(buy_threshold))
        _conf_frac = float(np.clip((_pred_mean - float(buy_threshold)) / _denom, 0.0, 1.0))
        _core_budget = initial_cash * _core_pct
        _entry_price = prices[_core_entry_i]
        if _entry_price > 0:
            _core_qty = int((_core_budget * _conf_frac) / _entry_price)   # confidence-scaled volume
        _core_remainder = _core_budget - _core_qty * _entry_price          # uninvested core stays as cash
        _core_active = True
        cash = initial_cash - _core_budget                                 # tactical sleeve trades on the rest
        logger.info(f"[{symbol}] CORE-LONG: pct={_core_pct:.2f} conf={_conf_frac:.2f} qty={_core_qty} "
                    f"entry@bar{_core_entry_i} (${_entry_price:.2f}) tactical_cash=${cash:.2f}")

    def _core_value(i, price):

        if not _core_active:
            return 0.0
        return _core_budget if i < _core_entry_i else (_core_qty * price + _core_remainder)


    portfolio_series = pd.Series(index=sim_timestamps, dtype=float)
    portfolio_series.iloc[0] = initial_cash

    # === OPTIMIZED PAIRS LOGIC WITH TOGGLE ===
    if CONFIG.get('ENABLE_FULL_PAIRS_RESOLUTION', False):
        check_interval = 1
    else:
        check_interval = 8


    valid_pairs = []
    pair_positions = {}
    for pair in CONFIG['PAIRS']:
        sym1, sym2 = pair

        if (sym1 in hmms and sym2 in hmms and
            hmms.get(sym1) is not None and
            hmms.get(sym2) is not None and
            sym1 in dfs_backtest and sym2 in dfs_backtest):
            if is_cointegrated(dfs_backtest[sym1]['close'], dfs_backtest[sym2]['close']):
                valid_pairs.append(pair)
                pair_positions[pair] = None


    _sym_hmm = hmms.get(symbol) if hmms else None


    # semantic label map attached at train time.
    if _sym_hmm is not None:
        try:
            _hmm_series = X_seq[:num_backtest_steps, -1, :]      # (n_bars, features), ordered
            _state_seq = _sym_hmm.predict(_hmm_series)
            bar_regimes = [regime_name_for_state(_sym_hmm, _s) for _s in _state_seq]
            if len(bar_regimes) < num_backtest_steps:            # safety pad
                bar_regimes += ["Unknown"] * (num_backtest_steps - len(bar_regimes))
        except Exception as _re:
            logger.warning(f"[{symbol}] regime Viterbi pass failed ({_re}); marking all Unknown")
            bar_regimes = ["Unknown"] * num_backtest_steps
    else:
        bar_regimes = ["Unknown"] * num_backtest_steps


    _last_stop_time = None        # post-stop cooldown
    _regime_hist = []             # B2 regime-flip confirmation buffer
    _conviction_streak = 0        # IDEA4 conviction streak
    _daily_stop_count = 0
    _current_day = None
    _bull_regimes = ["Calm Bull", "Moderate Bull"]
    position_type = "meanrev"


    _bk = int(CONFIG.get('TREND_BREAKOUT_LOOKBACK', 20))
    _roll_high = pd.Series(prices).rolling(_bk, min_periods=1).max().shift(1).values


    _gate = {'flat_bars': 0, 'vol_ok': 0, 'adx_ok': 0, 'conf_ok': 0, 'gateA': 0,
             'predbuy_ok': 0, 'rsi_ok': 0, 'reach_inner': 0, 'block_regime': 0,
             'block_cooldown': 0, 'block_volume': 0, 'block_breaker': 0, 'buys': 0}
    from collections import Counter as _Counter
    _regime_counter = _Counter()


    for local_i in range(num_backtest_steps):
        pred = make_prediction(model, X_seq[local_i:local_i+1], xgb_model)

        # === EXTRACT ALL VARIABLES ===
        price = prices[local_i]
        atr_val = atr[local_i]
        current_rsi = rsi[local_i]
        current_adx = adx[local_i]
        current_volatility = volatility[local_i]
        ts = pd.Timestamp(sim_timestamps[local_i])
        regime = bar_regimes[local_i]


        _regime_hist.append(regime)
        _max_hist = CONFIG.get('REGIME_EXIT_CONFIRM_CYCLES', 2) + 2
        if len(_regime_hist) > _max_hist:
            del _regime_hist[:-_max_hist]
        if pred > CONFIG.get('CONVICTION_BYPASS_PRED', 0.97):
            _conviction_streak += 1
        else:
            _conviction_streak = 0
        _day = ts.date()
        if _day != _current_day:
            _current_day = _day
            _daily_stop_count = 0

        # volume ratio vs N-bar MA (for volume gate)
        _vol_ratio = 1.0
        _vp = CONFIG.get('VOLUME_MA_PERIOD', 20)
        if local_i >= _vp:
            _vma = volume[local_i - _vp:local_i].mean()
            if _vma > 0:
                _vol_ratio = volume[local_i] / _vma


        if position > 0:
            if price > max_price:
                max_price = price
            time_held = (ts - entry_time).total_seconds() / 60 if entry_time else 0
            if time_held >= min_holding_period_minutes:

                if position_type == "trend":


                    _ma50_now = ma50[local_i]
                    chandelier = max_price - CONFIG.get('TREND_TRAIL_ATR_MULT', 3.5) * atr_val
                    hard_floor = entry_price - CONFIG.get('TREND_HARD_STOP_ATR', 5.0) * atr_val
                    _ma_break = (not np.isnan(_ma50_now)) and price < _ma50_now
                    sell_triggered = (price <= chandelier or price <= hard_floor or
                                      _ma_break or pred < CONFIG.get('TREND_EXIT_PRED', 0.40))
                    if sell_triggered:
                        reason = ("trend_trail" if price <= chandelier else
                                  "trend_hardstop" if price <= hard_floor else
                                  "trend_break" if _ma_break else "trend_signal")
                        cash += position * price - transaction_cost_per_trade
                        ret = (price - entry_price) / entry_price
                        returns.append(ret)
                        trade_count += 1
                        if ret > 0:
                            winning_trades += 1
                        if reason in ("trend_trail", "trend_hardstop"):
                            _last_stop_time = ts
                        if debug:
                            print(f"{Fore.CYAN}[{symbol}] Bar {local_i:5d} TREND-SOLD ({reason}) Pred={pred:.3f} Ret={ret:.3f}{Style.RESET_ALL}")
                        logger.info(f"{ts}: TrendSold {position} {symbol} @ ${price:.2f} ({reason}) ret={ret:.3f} cash=${cash:.2f}")
                        position = 0
                        entry_time = None
                        max_price = 0.0
                        position_type = "meanrev"
                else:

                    trailing_stop = max_price * (1 - trailing_stop_percentage)
                    min_stop = entry_price * CONFIG['MIN_STOP_LOSS_PCT']
                    stop_loss = entry_price - max(stop_loss_atr_multiplier * atr_val, min_stop)
                    take_profit = entry_price + take_profit_atr_multiplier * atr_val

                    # B2 regime-flip exit (same guardrails as live)
                    regime_flip_exit = False
                    if CONFIG.get('ENABLE_REGIME_EXIT', False):
                        _cn = int(CONFIG.get('REGIME_EXIT_CONFIRM_CYCLES', 2))
                        _rv = [r for r in _regime_hist[-(_cn * 2):] if r != "Unknown"][-_cn:]
                        if len(_rv) >= _cn:
                            _wl_b2 = CONFIG.get('BUY_REGIME_WHITELIST') or []
                            _all_out = all(r not in _wl_b2 for r in _rv)
                            _weak = pred < CONFIG.get('REGIME_EXIT_MAX_PRED', 0.65)
                            _strong_profit = ((price - entry_price) >= CONFIG.get('REGIME_EXIT_PROFIT_LOCK_ATR', 1.0) * atr_val) if atr_val > 0 else False
                            if _all_out and _weak and not _strong_profit:
                                regime_flip_exit = True

                    # time-weak-signal exit (the live bugfix)
                    time_exit = (CONFIG.get('ENABLE_TIME_EXIT', False) and
                                 time_held > CONFIG.get('TIME_EXIT_MINUTES', 240) and
                                 pred < CONFIG.get('TIME_EXIT_MAX_PRED', 0.52))

                    sell_triggered = (price <= trailing_stop or price <= stop_loss or
                                      price >= take_profit or
                                      (pred < sell_threshold and current_rsi > rsi_sell_threshold) or
                                      (pred < 0.50 and current_rsi > 65) or
                                      regime_flip_exit or time_exit)
                    if sell_triggered:
                        reason = ("trail" if price <= trailing_stop else
                                  "stop" if price <= stop_loss else
                                  "tp" if price >= take_profit else
                                  "regime_flip" if regime_flip_exit else
                                  "time" if time_exit else "signal")
                        cash += position * price - transaction_cost_per_trade
                        ret = (price - entry_price) / entry_price
                        returns.append(ret)
                        trade_count += 1
                        if ret > 0:
                            winning_trades += 1
                        if reason in ("stop", "trail"):
                            _daily_stop_count += 1
                            _last_stop_time = ts
                        elif reason == "regime_flip":
                            _last_stop_time = ts   # cooldown, but not a daily-stop count
                        if debug:
                            print(f"{Fore.GREEN}[{symbol}] Bar {local_i:5d} SOLD ({reason}) Pred={pred:.3f} RSI={current_rsi:.1f} Ret={ret:.3f}{Style.RESET_ALL}")
                        logger.info(f"{ts}: Sold {position} {symbol} @ ${price:.2f} ({reason}) ret={ret:.3f} cash=${cash:.2f}")
                        position = 0
                        entry_time = None
                        max_price = 0.0
                        position_type = "meanrev"


        elif position == 0:


            _gate['flat_bars'] += 1
            _regime_counter[regime] += 1
            if current_volatility <= max_volatility: _gate['vol_ok'] += 1
            if current_adx >= adx_trend_threshold: _gate['adx_ok'] += 1
            if pred >= confidence_threshold: _gate['conf_ok'] += 1
            if (current_volatility <= max_volatility and current_adx >= adx_trend_threshold
                    and pred >= confidence_threshold):
                _gate['gateA'] += 1
                # regime-dependent RSI threshold (relaxed in bull)
                _rsi_thr = (CONFIG.get('RSI_BUY_THRESHOLD_RELAXED', rsi_buy_threshold)
                            if regime in _bull_regimes else rsi_buy_threshold)
                # IDEA4 conviction bypass
                _cb_active = (CONFIG.get('ENABLE_CONVICTION_BYPASS', False) and
                              _conviction_streak >= CONFIG.get('CONVICTION_BYPASS_CYCLES', 2) and
                              current_volatility < CONFIG.get('CONVICTION_BYPASS_MAX_VOL', 6.0))
                _is_bypass = current_rsi >= _rsi_thr and _cb_active

                if pred > buy_threshold:
                    _gate['predbuy_ok'] += 1
                if pred > buy_threshold and (current_rsi < _rsi_thr or _cb_active):
                    _gate['rsi_ok'] += 1
                    _gate['reach_inner'] += 1
                    _wl = CONFIG.get('BUY_REGIME_WHITELIST')
                    _in_cooldown = (_last_stop_time is not None and
                                    (ts - _last_stop_time).total_seconds() / 60 < CONFIG['POST_STOP_COOLDOWN_MINUTES'])
                    if _wl and regime not in _wl:
                        _gate['block_regime'] += 1
                        pass  # regime whitelist block
                    elif _in_cooldown:
                        _gate['block_cooldown'] += 1
                        pass  # post-stop cooldown block
                    elif CONFIG.get('ENABLE_VOLUME_GATE', False) and _vol_ratio < CONFIG.get('VOLUME_CONFIRMATION_MULTIPLIER', 1.2):
                        _gate['block_volume'] += 1
                        pass  # volume gate block
                    elif CONFIG.get('DAILY_STOP_COUNT_LIMIT', 0) > 0 and _daily_stop_count >= CONFIG['DAILY_STOP_COUNT_LIMIT']:
                        _gate['block_breaker'] += 1
                        pass  # per-symbol daily breaker block
                    elif atr_val > 0:

                        _rr = CONFIG.get('RISK_BY_REGIME', {}).get(regime, risk_percentage)
                        _sm = float(CONFIG.get('RISK_MULTIPLIER_BY_SYMBOL', {}).get(symbol, 1.0))
                        _bm = float(CONFIG.get('CONVICTION_BYPASS_SIZE_MULT', 0.5)) if _is_bypass else 1.0
                        _cm = confidence_size_mult(pred, buy_threshold)
                        _eff = _rr * _sm * _bm * _cm
                        risk_per_trade = cash * _eff
                        stop_dist = max(atr_val * stop_loss_atr_multiplier, price * CONFIG['MIN_STOP_LOSS_PCT'])
                        if stop_dist > 0 and price > 0:
                            qty = max(1, int(risk_per_trade / stop_dist))
                            cost = qty * price + transaction_cost_per_trade
                            if cost > cash:
                                qty = max(0, int((cash - transaction_cost_per_trade) / price))
                                cost = qty * price + transaction_cost_per_trade
                            if qty > 0 and cost <= cash:
                                position = qty
                                entry_price = price
                                max_price = price
                                entry_time = ts
                                cash -= cost
                                position_type = "meanrev"
                                _gate['buys'] += 1
                                if debug:
                                    _lbl = "BUY-BYPASS" if _is_bypass else "BUY"
                                    print(f"{Fore.GREEN}[{symbol}] Bar {local_i:5d} {_lbl} ({regime}) Pred={pred:.3f} RSI={current_rsi:.1f} eff_risk={_eff*100:.2f}%{Style.RESET_ALL}")
                                logger.info(f"{ts}: Bought {qty} {symbol} @ ${price:.2f} ({regime}, eff_risk={_eff*100:.2f}%) cash=${cash:.2f}")


            if position == 0 and CONFIG.get('ENABLE_TREND_ENTRY', False) and atr_val > 0:
                _ma20_now = ma20[local_i]
                _ma50_now = ma50[local_i]
                _macd_now = macd_arr[local_i]
                _macd_sig_now = macd_sig[local_i]
                _hi = _roll_high[local_i]
                _struct_ok = (
                    (not np.isnan(_ma20_now)) and (not np.isnan(_ma50_now)) and
                    price > _ma20_now and _ma20_now > _ma50_now and                      # uptrend structure
                    current_adx >= CONFIG.get('TREND_ADX_MIN', 25.0) and                  # strong trend
                    current_volatility <= max_volatility and                              # sanity vol cap
                    (not np.isnan(_macd_now)) and (not np.isnan(_macd_sig_now)) and
                    _macd_now > _macd_sig_now and                                          # momentum confirm
                    (not np.isnan(_hi)) and price >= CONFIG.get('TREND_BREAKOUT_PCT', 0.97) * _hi and  # near breakout high
                    pred > CONFIG.get('TREND_PRED_MIN', 0.50)                              # model not bearish
                )
                _wl_t = CONFIG.get('BUY_REGIME_WHITELIST')
                _in_cd_t = (_last_stop_time is not None and
                            (ts - _last_stop_time).total_seconds() / 60 < CONFIG['POST_STOP_COOLDOWN_MINUTES'])
                _vol_block = (CONFIG.get('ENABLE_VOLUME_GATE', False) and
                              _vol_ratio < CONFIG.get('VOLUME_CONFIRMATION_MULTIPLIER', 1.2))
                _breaker_block = (CONFIG.get('DAILY_STOP_COUNT_LIMIT', 0) > 0 and
                                  _daily_stop_count >= CONFIG['DAILY_STOP_COUNT_LIMIT'])
                if (_struct_ok and not (_wl_t and regime not in _wl_t)
                        and not _in_cd_t and not _vol_block and not _breaker_block):
                    _sm_t = float(CONFIG.get('RISK_MULTIPLIER_BY_SYMBOL', {}).get(symbol, 1.0))
                    _cm_t = confidence_size_mult(pred, CONFIG.get('TREND_PRED_MIN', 0.50))
                    _eff_t = CONFIG.get('TREND_RISK_PCT', 0.015) * _sm_t * _cm_t
                    risk_per_trade = cash * _eff_t
                    stop_dist = CONFIG.get('TREND_TRAIL_ATR_MULT', 3.5) * atr_val
                    if stop_dist > 0 and price > 0:
                        qty = max(1, int(risk_per_trade / stop_dist))
                        cost = qty * price + transaction_cost_per_trade
                        if cost > cash:
                            qty = max(0, int((cash - transaction_cost_per_trade) / price))
                            cost = qty * price + transaction_cost_per_trade
                        if qty > 0 and cost <= cash:
                            position = qty
                            entry_price = price
                            max_price = price
                            entry_time = ts
                            cash -= cost
                            position_type = "trend"
                            if debug:
                                print(f"{Fore.CYAN}[{symbol}] Bar {local_i:5d} TREND-BUY ({regime}) Pred={pred:.3f} ADX={current_adx:.1f} RSI={current_rsi:.1f} eff_risk={_eff_t*100:.2f}%{Style.RESET_ALL}")
                            logger.info(f"{ts}: TrendBought {qty} {symbol} @ ${price:.2f} (ADX={current_adx:.1f}, eff_risk={_eff_t*100:.2f}%) cash=${cash:.2f}")


        if local_i % check_interval == 0:
            for pair in valid_pairs:
                sym1, sym2 = pair
                hmm1 = hmms.get(sym1)
                hmm2 = hmms.get(sym2)
                
                if hmm1 is None or hmm2 is None:
                    continue


                _sc1 = scalers.get(sym1) if scalers else None
                _sc2 = scalers.get(sym2) if scalers else None
                if _sc1 is None or _sc2 is None:
                    continue


                recent_seq1 = preprocess_data(
                    dfs_backtest[sym1].iloc[-CONFIG['TIMESTEPS']-5:],
                    CONFIG['TIMESTEPS'], inference_mode=True,
                    inference_scaler=_sc1, fit_scaler=False
                )[0][-1:]

                recent_seq2 = preprocess_data(
                    dfs_backtest[sym2].iloc[-CONFIG['TIMESTEPS']-5:],
                    CONFIG['TIMESTEPS'], inference_mode=True,
                    inference_scaler=_sc2, fit_scaler=False
                )[0][-1:]
                
                regime = get_pair_regime(hmm1, hmm2, recent_seq1, recent_seq2)
                
                if regime not in CONFIG['PAIR_REGIME_FILTER']:
                    continue
                    
                df1 = dfs_backtest[sym1]
                df2 = dfs_backtest[sym2]
                spread = calculate_spread(df1, df2)
                zscore = (spread.iloc[-1] - spread.mean()) / spread.std()
                
                if abs(zscore) > 2.0 and pair_positions.get(pair) is None:
                    hedge = calculate_spread(df1, df2).iloc[-1]
                    qty1 = int(initial_cash * 0.01 / df1['close'].iloc[-1])
                    qty2 = int(qty1 * hedge)
                    side = 'long' if zscore > 0 else 'short'
                    pair_positions[pair] = (side, qty1, qty2, spread.iloc[-1])
                    logger.info(f"[{symbol}] ENTERED PAIR {pair} ({regime}) zscore={zscore:.2f} side={side}")

        current_value = cash + (position * price if position > 0 else 0) + _core_value(local_i, price)
        portfolio_series.iloc[local_i] = current_value


    _fb = max(1, _gate['flat_bars'])
    print(f"{Fore.MAGENTA}[{symbol}] ENTRY FUNNEL ({_gate['flat_bars']} flat bars): "
          f"vol_ok={_gate['vol_ok']} adx_ok={_gate['adx_ok']} conf_ok={_gate['conf_ok']} "
          f"→ gateA(all3)={_gate['gateA']} → pred>buy={_gate['predbuy_ok']} → +rsi={_gate['rsi_ok']} "
          f"→ blocks[regime={_gate['block_regime']} cooldown={_gate['block_cooldown']} "
          f"volume={_gate['block_volume']} breaker={_gate['block_breaker']}] → BUYS={_gate['buys']}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}[{symbol}] Regime distribution (flat bars): "
          f"{dict(_regime_counter.most_common())}  | whitelist={CONFIG.get('BUY_REGIME_WHITELIST')}{Style.RESET_ALL}")

    # Close any remaining single-stock position
    if position > 0:
        last_price = prices[-1]
        cash += position * last_price - transaction_cost_per_trade
        ret = (last_price - entry_price) / entry_price
        returns.append(ret)
        trade_count += 1
        if ret > 0:
            winning_trades += 1


    if _core_active:
        cash += _core_qty * prices[-1] + _core_remainder

    # Final portfolio value
    portfolio_series.iloc[-1] = cash

    win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0.0
    portfolio_series = portfolio_series.ffill().fillna(initial_cash)

    return cash, returns, trade_count, win_rate, accuracy_percentage, portfolio_series

def buy_and_hold_backtest(dfs_backtest: Dict[str, pd.DataFrame], initial_cash: float) -> Tuple[float, Dict[str, pd.Series]]:
    backtest_start = pd.Timestamp(CONFIG['BACKTEST_START_DATE'], tz='UTC')
    initial_per_symbol = initial_cash / len(CONFIG['SYMBOLS'])
    bh_final_value = 0.0
    bh_series_per_symbol = {}


    _ts = CONFIG['TIMESTEPS']
    _end_cfg = CONFIG.get('BACKTEST_END_DATE')
    _end_ts = pd.Timestamp(_end_cfg, tz='UTC') if _end_cfg else None

    for symbol, df in dfs_backtest.items():

        df_win = df[df.index >= backtest_start]
        if _end_ts is not None:
            df_win = df_win[df_win.index <= _end_ts]
        df_bh = df_win.iloc[_ts:].copy()
        if df_bh.empty or len(df_bh) < 2:
            logger.warning(f"Insufficient data for buy-and-hold on {symbol}; skipping.")
            continue
            
        first_close = df_bh['close'].iloc[0]
        if first_close <= 0:
            continue
            
        qty = int((initial_per_symbol - CONFIG['TRANSACTION_COST_PER_TRADE']) / first_close)
        if qty <= 0:
            continue
            
        # Build series for graphing (only 2025 onward)
        bh_values = qty * df_bh['close']
        bh_series = pd.Series(bh_values.values, index=df_bh.index, name=symbol)
        bh_series_per_symbol[symbol] = bh_series
        last_value = bh_series.iloc[-1]
        bh_final_value += last_value
    
    logger.info(f"Buy-and-hold final value (same period as neural): ${bh_final_value:.2f}")
    return bh_final_value, bh_series_per_symbol

def calculate_performance_metrics(returns: List[float], cash: float, initial_per_symbol: float) -> Dict[str, float]:

    if not returns or len(returns) == 0:
        total_return = (cash - initial_per_symbol) / initial_per_symbol * 100 if initial_per_symbol > 0 else 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
    else:
        returns = np.array(returns)
        total_return = (cash - initial_per_symbol) / initial_per_symbol * 100 if initial_per_symbol > 0 else 0.0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else 0.0  # Annualized, risk-free=0
        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = -np.min(drawdown) * 100 if len(drawdown) > 0 else 0.0  # Positive percentage
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }


def compute_portfolio_metrics(portfolio_series_per_symbol: Dict[str, pd.Series]) -> Dict[str, float]:
    """v15: THE honest objective metric.

    Per-symbol Sharpe (in calculate_performance_metrics) is computed on the
    per-TRADE return list — with only 2-9 trades/symbol that number is pure
    sampling noise (e.g. Sharpe 44.9 on 3 trades). The real risk-adjusted
    performance lives in the combined bar-by-bar EQUITY CURVE.

    This sums all per-symbol equity series into one portfolio curve, resamples
    to daily, and computes annualized Sharpe + Sortino + max drawdown on the
    daily returns. This is what the Sharpe>1 objective is judged against.
    """
    series_list = [s for s in portfolio_series_per_symbol.values()
                   if s is not None and len(s) > 0]
    if not series_list:
        return {'portfolio_sharpe': 0.0, 'portfolio_sortino': 0.0,
                'portfolio_max_drawdown': 0.0, 'portfolio_total_return': 0.0,
                'portfolio_n_days': 0}
    all_series = pd.concat(series_list, axis=1, join='outer')
    all_series.index = pd.to_datetime(all_series.index)
    all_series = all_series.sort_index().ffill().bfill()
    total = all_series.sum(axis=1)


    daily = total.resample('B').last().ffill().dropna()
    rets = daily.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0:
        sharpe = 0.0
        sortino = 0.0
    else:
        sharpe = float((rets.mean() / rets.std()) * np.sqrt(252))   # annualized, rf=0
        downside_std = rets[rets < 0].std()
        sortino = float((rets.mean() / downside_std) * np.sqrt(252)) if downside_std and downside_std != 0 else 0.0
    if len(rets) >= 1:
        cum = (1 + rets).cumprod()
        peak = cum.cummax()
        max_dd = float(-((cum - peak) / peak).min() * 100)
    else:
        max_dd = 0.0
    total_return = float((daily.iloc[-1] - daily.iloc[0]) / daily.iloc[0] * 100) if len(daily) > 1 else 0.0
    return {'portfolio_sharpe': sharpe, 'portfolio_sortino': sortino,
            'portfolio_max_drawdown': max_dd, 'portfolio_total_return': total_return,
            'portfolio_n_days': int(len(daily))}


def selection_score(metrics: Dict[str, float], trade_count: int) -> Optional[float]:
    """v15: composite per-symbol selection score = Sharpe − λ·max_drawdown(%).

    Trade-guarded: returns None (disqualified) when trades < MIN_TRADES_FOR_SHARPE,
    so a fluke low-trade Sharpe can never win. Drawdown-penalized so a smooth
    equity curve is preferred over a jagged one with the same Sharpe.
    """
    min_tr = CONFIG.get('MIN_TRADES_FOR_SHARPE', 5)
    if trade_count < min_tr:
        return None
    sharpe = metrics.get('sharpe_ratio', 0.0)
    dd = metrics.get('max_drawdown', 0.0)
    return sharpe - CONFIG.get('SELECTION_DD_PENALTY_PER_PCT', 0.05) * dd


def format_email_body(
    initial_cash: float,
    final_value: float,
    symbol_results: Dict[str, Dict[str, float]],
    trade_counts: Dict[str, int],
    win_rates: Dict[str, float]
) -> str:
    body = [
        f"Backtest Results",
        f"Initial Cash: ${initial_cash:.2f}",
        f"Final Value: ${final_value:.2f}",
        f"Total Return: {(final_value - initial_cash) / initial_cash * 100:.2f}%",
        f"",
        f"Per-Symbol Performance:"
    ]
    for symbol, metrics in symbol_results.items():
        body.append(f"\n{symbol}:")
        for metric, value in metrics.items():
            metric_lower = metric.lower()
            if 'final_value' in metric_lower:
                value_str = f"${value:.2f}"
                unit = ''
            else:
                value_str = f"{value:.3f}"
                if any(k in metric_lower for k in ['return', 'drawdown', 'var', 'prob']):
                    unit = '%'
                else:
                    unit = ''
            body.append(f"  {metric.replace('_', ' ').title()}: {value_str}{unit}")
        body.append(f"  Trades: {trade_counts.get(symbol, 0)}")
        body.append(f"  Win Rate: {win_rates.get(symbol, 0.0):.3f}%")
    return "\n".join(body)


def send_email(subject: str, body: str, attachment_path: Optional[str] = None) -> None:
    

    if os.getenv("OPTIMIZER_MODE") == "true":
        logger.info(f"[EMAIL SKIPPED] {subject} (backtest or optimizer mode)")
        return
    


    _sender = CONFIG['EMAIL_SENDER'].lower()
    _recipients = list(dict.fromkeys(   # preserve order, remove duplicates
        r for r in CONFIG['EMAIL_RECEIVER']
        if r.lower() != _sender
    ))
    if not _recipients:
        logger.warning("send_email: no recipients after dedup — skipping")
        return
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = CONFIG['EMAIL_SENDER']
    msg['To'] = ', '.join(_recipients)
    msg.attach(MIMEText(body, 'plain'))
    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
            msg.attach(img)
    with smtplib.SMTP(CONFIG['SMTP_SERVER'], CONFIG['SMTP_PORT']) as server:
        server.starttls()
        server.login(CONFIG['EMAIL_SENDER'], CONFIG['EMAIL_PASSWORD'])
        server.sendmail(CONFIG['EMAIL_SENDER'], _recipients, msg.as_string())

def send_email_async(subject: str, body: str, attachment_path: Optional[str] = None) -> None:
    """Fire send_email in a daemon thread so the live trading loop is never blocked by SMTP.

    SMTP handshake + TLS + send typically takes 30–40 seconds.  When send_email() was called
    synchronously inside the scan loop, the loop restarted 35s later — landing within 30s of
    the next 15-min boundary — which caused the 14-min cycle guard to be skipped and triggered
    a second scan firing on the same candle.  Making email async eliminates that lag entirely.
    """
    t = threading.Thread(target=send_email, args=(subject, body, attachment_path), daemon=True)
    t.start()


def make_prediction(model: nn.Module, X: np.ndarray, xgb_model=None) -> float:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        raw_logit = model(X_tensor).squeeze(-1)           # get raw logit
        _temp = getattr(model, '_calib_temp', None) or CONFIG['PREDICTION_TEMPERATURE']
        scaled_logit = raw_logit / _temp
        prob = torch.sigmoid(scaled_logit).cpu().item()
    
    # XGBoost ensemble still works on top
    if xgb_model is not None and hasattr(xgb_model, 'predict_proba'):
        try:
            xgb_prob = xgb_model.predict_proba(X.reshape(1, -1))[0][1]
            _w = float(CONFIG.get('BLEND_LSTM_WEIGHT', 0.6))
            prob = _w * prob + (1.0 - _w) * xgb_prob
        except Exception:
            pass
    
    return float(prob)

def clean_weights_directory(weights_dir: str) -> None:
    """Organise the Model Weights directory.

    Moves files belonging to legacy versions (v228, v302, v303, v31026) into
    per-version sub-directories, keeping only deepTrader10 files in the root.
    Removes empty junk folders ("sort this shit", "beta", "New folder").
    """
    if not os.path.isdir(weights_dir):
        print(f"Model Weights directory not found: {weights_dir}")
        return


    import re
    legacy_pattern = re.compile(r'_(v228|v302|v303|v31026)(_attempt\d+)?\.(pth|pkl)$', re.IGNORECASE)

    moved = 0
    for fname in os.listdir(weights_dir):
        fpath = os.path.join(weights_dir, fname)
        if not os.path.isfile(fpath):
            continue
        m = legacy_pattern.search(fname)
        if m:
            version_tag = m.group(1)
            dest_dir = os.path.join(weights_dir, version_tag)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, fname)
            if not os.path.exists(dest):
                shutil.move(fpath, dest)
                moved += 1
                print(f"  Moved {fname} → {version_tag}/")
            else:
                os.remove(fpath)
                print(f"  Removed duplicate {fname}")


    junk_folders = ["sort this shit", "beta", "New folder"]
    for folder in junk_folders:
        junk_path = os.path.join(weights_dir, folder)
        if os.path.isdir(junk_path):
            contents = os.listdir(junk_path)
            if not contents:
                os.rmdir(junk_path)
                print(f"  Removed empty folder: '{folder}'")
            else:

                for item in contents:
                    item_path = os.path.join(junk_path, item)
                    if os.path.isfile(item_path):
                        m2 = legacy_pattern.search(item)
                        if m2:
                            version_tag = m2.group(1)
                            dest_dir = os.path.join(weights_dir, version_tag)
                            os.makedirs(dest_dir, exist_ok=True)
                            dest = os.path.join(dest_dir, item)
                            if not os.path.exists(dest):
                                shutil.move(item_path, dest)
                                moved += 1
                                print(f"  Moved '{folder}/{item}' → {version_tag}/")
                            else:
                                os.remove(item_path)
                        else:
                            # Unknown file — move to a misc archive folder
                            misc_dir = os.path.join(weights_dir, "_misc_archive")
                            os.makedirs(misc_dir, exist_ok=True)
                            dest = os.path.join(misc_dir, item)
                            if not os.path.exists(dest):
                                shutil.move(item_path, dest)
                                print(f"  Archived unknown file '{folder}/{item}' → _misc_archive/")
                            else:
                                os.remove(item_path)
                remaining = os.listdir(junk_path)
                if not remaining:
                    os.rmdir(junk_path)
                    print(f"  Removed now-empty folder: '{folder}'")

    print(f"\nDone. Moved {moved} legacy files. Root now contains only deepTrader10 weights.")


def get_api_keys(config: Dict) -> None:
    if config['ALPACA_API_KEY'] in [None, '', 'REPLACE ME'] or config['ALPACA_SECRET_KEY'] in [None, '', 'REPLACE ME']:
        logger.info("Alpaca API keys missing or invalid. Prompting for input.")
        config['ALPACA_API_KEY'] = input("Enter Alpaca API Key: ").strip()
        config['ALPACA_SECRET_KEY'] = input("Enter Alpaca Secret Key: ").strip()
        if not config['ALPACA_API_KEY'] or not config['ALPACA_SECRET_KEY']:
            raise ValueError("Alpaca API keys cannot be empty.")
    else:
        logger.info("Using hardcoded Alpaca API keys from CONFIG.")

def main(backtest_only: bool = False, force_train: bool = False, debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s,%(msecs)03d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Safe GPU info – NO CUDA context in main process
    gpu_name, free_gb = _get_gpu_info_safe()
    device_str = f"cuda ({free_gb:.1f} GB free)" if torch.cuda.is_available() else "cpu"
    print(f"{Fore.LIGHTGREEN_EX}Device set to use {Fore.GREEN}{device_str}{Fore.LIGHTGREEN_EX} with {Fore.GREEN}{gpu_name}{Style.RESET_ALL}")
    if not backtest_only:
        logger.info("Bot started")


        if not debug:
            stream_handler.setLevel(logging.CRITICAL)
    else:
        logger.info("Bot started in backtest mode")
    get_api_keys(CONFIG)
    
    check_dependencies()
    validate_config(CONFIG)
    create_cache_directory()
    trading_client = TradingClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'], paper=CONFIG['PAPER_TRADING'])
    expected_features = len(MODEL_FEATURES)
    models = {}
    scalers = {}
    dfs = {}
    xgb_models = {}
    stock_info = []
    total_epochs = len(CONFIG['SYMBOLS']) * CONFIG['TRAIN_EPOCHS']
    training_sentiments = {}
    sentiments = {}  # ADD THIS LINE
    hmms = {}  # NEW: for live use
    need_training = False
    for symbol in CONFIG['SYMBOLS']:
        model, scaler, sentiment, hmm, xgb_model = load_model_and_scaler(symbol, expected_features, force_train)
        models[symbol] = model
        scalers[symbol] = scaler
        training_sentiments[symbol] = sentiment
        hmms[symbol] = hmm
        xgb_models[symbol] = xgb_model
        if model is None:
            need_training = True
    progress_bar = tqdm(total=total_epochs, desc="Training Progress", bar_format="{l_bar}\033[32m{bar}\033[0m{r_bar}") if need_training else None

    if not backtest_only:
        # Live trading block
        start_total_training_time = time.perf_counter()


        all_models_cached = all(
            os.path.exists(os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")) and
            os.path.exists(os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl"))
            for symbol in CONFIG['SYMBOLS']
        )

        if all_models_cached and not force_train:
            logger.info("All models cached – skipping multiprocessing for live mode (fast path)")
            outputs = []
            for symbol in CONFIG['SYMBOLS']:
                model, scaler, sentiment, hmm, xgb_model = load_model_and_scaler(symbol, expected_features, force_retrain=False)
                if sentiment is None:
                    sentiment = 0.0
                outputs.append((
                    symbol, model, scaler,
                    True,       # data_loaded
                    sentiment,
                    True,       # sentiment_loaded
                    True,       # model_loaded
                    hmm,
                    xgb_model,
                    0           # training_time_ms
                ))


            # "free(): invalid next size (fast)" → SIGABRT.


            if torch.cuda.is_available():
                _warmup = torch.zeros(1, device='cuda')
                del _warmup
                torch.cuda.empty_cache()
                logger.info("CUDA context pre-initialised (fast path)")
        else:


            mp.set_start_method('spawn', force=True)
            with mp.Manager() as manager:
                gpu_semaphore = manager.Semaphore(CONFIG['NUM_PARALLEL_WORKERS'])
                with mp.Pool(processes=CONFIG['NUM_PARALLEL_WORKERS']) as pool:
                    worker_tasks = [(i+1, sym, expected_features, force_train, None, gpu_semaphore, False, debug)
                                  for i, sym in enumerate(CONFIG['SYMBOLS'])]
                    outputs = list(tqdm(pool.imap(train_wrapper, worker_tasks),
                        total=len(CONFIG['SYMBOLS']), desc="Processing symbols"))

            if torch.cuda.is_available():
                _warmup = torch.zeros(1, device='cuda')
                del _warmup
                torch.cuda.empty_cache()
                logger.info("CUDA context pre-initialised (training path)")

        logger.info("Parallel processing completed; CUDA memory cleared.")

        end_total_training_time = time.perf_counter()
        total_training_time_in_milliseconds = (end_total_training_time - start_total_training_time) * 1000
        training_times_dictionary = {}  # Must be initialized in live path too

        sentiments = {}  # Collect sentiments for live consistency
        hmms = {}
        xgb_models = {}
        for output_tuple in outputs:
            symbol = output_tuple[0]
            model = output_tuple[1]
            scaler = output_tuple[2]
            data_loaded = output_tuple[3]
            sentiment = output_tuple[4]
            sentiment_loaded = output_tuple[5]
            model_loaded = output_tuple[6]
            hmm = output_tuple[7]
            xgb_model = output_tuple[8]
            training_time_in_milliseconds = output_tuple[9]

            training_times_dictionary[symbol] = training_time_in_milliseconds if training_time_in_milliseconds is not None else 0
            models[symbol] = model
            scalers[symbol] = scaler
            sentiments[symbol] = sentiment
            hmms[symbol] = hmm
            xgb_models[symbol] = xgb_model

            info = []
            info.append(f"{Fore.LIGHTBLUE_EX}{symbol}:{Style.RESET_ALL}")
            info.append(f"  {'Loaded cached model and scaler' if model_loaded else 'Trained model'} for {symbol}.")
            info.append(f"  {'Loaded' if data_loaded else 'Fetched'} bars for {symbol} {'from cache' if data_loaded else ''}.")
            info.append(f"  {'Loaded news data' if sentiment_loaded else 'Computed news sentiment'} for {symbol} {'from cache' if sentiment_loaded else ''}.")
            info.append(f"  Calculated sentiment score: {sentiment:.3f}")
            info.append(f"  Calculated stop-loss ATR multiplier: {CONFIG['STOP_LOSS_ATR_MULTIPLIER']:.2f}")
            try:
                position = trading_client.get_open_position(symbol)
                qty_owned = int(float(position.qty))
                info.append(f"  Current amount of stocks owned: {qty_owned}")
            except:
                qty_owned = 0
                info.append(f"  Current amount of stocks owned: {qty_owned}")
            if not model_loaded:
                info.append(f"  Saved model and scaler for {symbol}.")
            stock_info.append(info)
            if progress_bar and not model_loaded:
                progress_bar.update(CONFIG['TRAIN_EPOCHS'])

        if progress_bar:
            progress_bar.close()

        if debug:
            for info in stock_info:
                for line in info:
                    print(line)
                print()


        if not backtest_only:
            try:
                _all_pos = trading_client.get_all_positions()
                if _all_pos:
                    _total_mv = sum(float(p.market_value) for p in _all_pos)
                    print(f"\n{Fore.YELLOW}{'='*60}")
                    print(f"EXISTING POSITIONS AT STARTUP ({len(_all_pos)} open):")
                    for _p in _all_pos:
                        _sym  = _p.symbol
                        _qty  = int(float(_p.qty))
                        _mv   = float(_p.market_value)
                        _pnl  = float(_p.unrealized_pl)
                        _ep   = float(_p.avg_entry_price)
                        _cp   = float(_p.current_price)
                        _sign = "+" if _pnl >= 0 else ""
                        print(f"  {_sym:6s} {_qty:5d} shares  entry=${_ep:.2f}  now=${_cp:.2f}  "
                              f"MV=${_mv:,.2f}  P&L={_sign}${_pnl:,.2f}")
                    print(f"  Total market value: ${_total_mv:,.2f}")
                    print(f"{'='*60}{Style.RESET_ALL}\n")
                else:
                    print(f"\n{Fore.GREEN}No existing positions at startup — starting flat.{Style.RESET_ALL}\n")
            except Exception as _pe:
                logger.warning(f"Could not fetch existing positions at startup: {_pe}")

        portfolio_value = CONFIG['INITIAL_CASH']
        peak_value = portfolio_value


        _live_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        @retry(stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']),
               wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000),
               retry=retry_if_exception_type(RETRYABLE_ERRORS))
        def get_clock_with_retry():
            return trading_client.get_clock()

        @retry(stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']),
               wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000),
               retry=retry_if_exception_type(RETRYABLE_ERRORS))
        def get_account_with_retry():
            return trading_client.get_account()

        @retry(stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']),
               wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000),
               retry=retry_if_exception_type(RETRYABLE_ERRORS))
        def get_all_positions_with_retry():
            return trading_client.get_all_positions()


        _stop_cooldown: dict = {}


        _regime_history: dict = {}


        _conviction_streak: dict = {}


        # never left running on a loose trend trail.
        _entry_type: dict = {}


        _last_cycle_start: Optional[datetime] = None


        _session_start_equity: float = 0.0
        _daily_stop_count: int = 0
        _session_date: Optional[Any] = None

        while True:  # Infinite loop for continuous live trading
            try:
                clock = get_clock_with_retry()
            except RETRYABLE_ERRORS as e:
                logger.error(f"Failed to get clock after {CONFIG['API_RETRY_ATTEMPTS']} retries: {type(e).__name__}: {str(e)}")
                send_email_async("API Error", f"Failed to get clock after retries:\n{type(e).__name__}: {str(e)}")
                time.sleep(60)
                continue
            except Exception as e:
                logger.error(f"Unexpected error getting clock: {type(e).__name__}: {str(e)}")
                time.sleep(60)
                continue

            if clock.is_open:


                now = datetime.now(timezone.utc)
                mins_past = now.minute % 15
                secs_past = mins_past * 60 + now.second + now.microsecond / 1e6
                secs_to_next = (15 * 60) - secs_past


                # re-fire.
                if secs_to_next < 30:
                    secs_to_next += 15 * 60
                time.sleep(secs_to_next)

                now = datetime.now(timezone.utc)


                cycle_guard_secs = CONFIG.get('CYCLE_GUARD_MINUTES', 14) * 60
                if _last_cycle_start is not None:
                    elapsed = (now - _last_cycle_start).total_seconds()
                    if elapsed < cycle_guard_secs:
                        wait_for = cycle_guard_secs - elapsed
                        logger.info(f"Cycle guard: last cycle was {elapsed:.0f}s ago, sleeping {wait_for:.0f}s more")
                        time.sleep(wait_for)
                        now = datetime.now(timezone.utc)
                _last_cycle_start = now

                try:
                    account = get_account_with_retry()
                    cash = float(account.cash)
                    portfolio_value = float(account.equity)
                except RETRYABLE_ERRORS as e:
                    logger.error(f"Failed to get account after retries: {type(e).__name__}: {str(e)}")
                    send_email_async("API Error", f"Failed to get account:\n{type(e).__name__}: {str(e)}")
                    time.sleep(60)
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error getting account: {type(e).__name__}: {str(e)}")
                    time.sleep(60)
                    continue


                _today = now.date()
                if _session_date != _today:
                    _session_date = _today
                    _session_start_equity = portfolio_value
                    _daily_stop_count = 0
                    logger.info(f"New trading session {_today.isoformat()} — start equity ${portfolio_value:,.2f}")

                peak_value = max(peak_value, portfolio_value)
                drawdown = (peak_value - portfolio_value) / peak_value
                if drawdown > CONFIG['MAX_DRAWDOWN_LIMIT']:
                    logger.warning(f"Portfolio drawdown exceeded {CONFIG['MAX_DRAWDOWN_LIMIT'] * 100}%. Pausing trading.")
                    send_email_async("Portfolio Drawdown Alert", f"Portfolio drawdown exceeded {CONFIG['MAX_DRAWDOWN_LIMIT'] * 100}%. Trading paused.")
                    break

                decisions = []
                try:
                    open_positions = get_all_positions_with_retry()
                except RETRYABLE_ERRORS as e:
                    logger.error(f"Failed to get positions after retries: {type(e).__name__}: {str(e)}")
                    time.sleep(60)
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error getting positions: {type(e).__name__}: {str(e)}")
                    time.sleep(60)
                    continue


                remaining_cash = cash
                max_per_symbol = portfolio_value * CONFIG['MAX_POSITION_SIZE_PCT']

                for symbol in CONFIG['SYMBOLS']:
                    if symbol not in models or models[symbol] is None:
                        continue


                    prediction = 0.5
                    regime = "Unknown"
                    price = 0.0
                    current_rsi = 50.0
                    current_adx = 25.0
                    current_volatility = 15.0
                    atr_val = 1.0
                    ma20_val = 0.0
                    ma50_val = 0.0
                    macd_val = 0.0
                    macd_sig_val = 0.0
                    breakout_high = 0.0
                    decision = "Hold (No Data)"

                    try:
                        df = fetch_recent_data(symbol, CONFIG['LIVE_DATA_BARS'])

                        min_required = CONFIG['TIMESTEPS'] + 20
                        if df is None or df.empty or len(df) < min_required:
                            logger.warning(f"Insufficient live bars for {symbol} ({0 if df is None or df.empty else len(df)} bars, need {min_required}) — holding")
                            decisions.append({'symbol': symbol, 'decision': "Hold (No Data)",
                                'confidence': prediction, 'rsi': current_rsi, 'adx': current_adx,
                                'volatility': current_volatility, 'price': price, 'owned': 0, 'regime': regime})
                            continue

                        if CONFIG.get('_LIVE_SENTIMENT_ENABLED', False):
                            sentiment = get_sentiment_score(symbol)
                        else:
                            sentiment = sentiments.get(symbol, 0.0)
                        df = calculate_indicators(df, sentiment)

                        if len(df) < min_required or df['close'].iloc[-1] <= 0:
                            logger.warning(f"Post-indicator data invalid for {symbol} — holding")
                            decisions.append({'symbol': symbol, 'decision': "Hold (Bad Indicators)",
                                'confidence': prediction, 'rsi': current_rsi, 'adx': current_adx,
                                'volatility': current_volatility, 'price': price, 'owned': 0, 'regime': regime})
                            continue

                        X_seq, _, _ = preprocess_data(df, CONFIG['TIMESTEPS'], inference_mode=True,
                                                       inference_scaler=scalers[symbol], fit_scaler=False)
                        recent_seq = X_seq[-1:].astype(np.float32)

                        model = models[symbol].to(_live_device)
                        model.eval()


                        _temp = getattr(model, '_calib_temp', None) or CONFIG['PREDICTION_TEMPERATURE']
                        with torch.no_grad():
                            _Xt = torch.tensor(X_seq.astype(np.float32)).to(_live_device)
                            _logits = model(_Xt).squeeze(-1)
                            _lstm_dist = torch.sigmoid(_logits / _temp).cpu().numpy().ravel()
                        lstm_prob = float(_lstm_dist[-1])


                        xgb_model = xgb_models.get(symbol)
                        if xgb_model is not None and hasattr(xgb_model, 'predict_proba'):
                            try:
                                _xgb_dist = xgb_model.predict_proba(X_seq.reshape(X_seq.shape[0], -1))[:, 1]
                                _w = float(CONFIG.get('BLEND_LSTM_WEIGHT', 0.6))
                                _pred_dist = _w * _lstm_dist + (1.0 - _w) * np.asarray(_xgb_dist).ravel()
                            except Exception as xgb_err:
                                logger.warning(f"XGBoost predict failed for {symbol}: {xgb_err} — using LSTM only")
                                _pred_dist = _lstm_dist
                        else:
                            _pred_dist = _lstm_dist
                        prediction = float(_pred_dist[-1])


                        _adaptive_buy_th = adaptive_buy_threshold(_pred_dist, smoke=CONFIG.get('SMOKE_TEST', False))

                        price = float(df['close'].iloc[-1])
                        current_rsi = float(df['RSI'].iloc[-1])
                        current_adx = float(df['ADX'].iloc[-1])
                        current_volatility = float(df['Volatility'].iloc[-1])
                        atr_val = float(df['ATR'].iloc[-1])

                        ma20_val = float(df['MA20'].iloc[-1])
                        ma50_val = float(df['MA50'].iloc[-1])
                        macd_val = float(df['MACD'].iloc[-1])
                        macd_sig_val = float(df['MACD_signal'].iloc[-1])
                        _bk_live = int(CONFIG.get('TREND_BREAKOUT_LOOKBACK', 20))
                        breakout_high = float(df['close'].iloc[-(_bk_live + 1):-1].max()) if len(df) > _bk_live else price

                        hmm_model = hmms.get(symbol)
                        if hmm_model is not None:
                            try:


                                hmm_input = recent_seq[:, -1, :] if recent_seq.ndim == 3 else recent_seq.reshape(-1, recent_seq.shape[-1])
                                regime_idx = hmm_model.predict(hmm_input)[-1]   # latest bar's regime
                                regime = regime_name_for_state(hmm_model, regime_idx)
                            except Exception as e:
                                logger.warning(f"HMM regime prediction failed for {symbol}: {e}")
                                regime = "Unknown"

                        logger.info(f"LIVE → {symbol} | Pred={prediction:.3f} | Price=${price:.2f} | "
                                   f"RSI={current_rsi:.1f} | ADX={current_adx:.1f} | Vol={current_volatility:.2f} | ATR=${atr_val:.2f}")


                        # force-exit a healthy position).
                        _rh = _regime_history.setdefault(symbol, [])
                        _rh.append(regime)
                        _max_hist = CONFIG.get('REGIME_EXIT_CONFIRM_CYCLES', 2) + 2
                        if len(_rh) > _max_hist:
                            del _rh[:-_max_hist]


                        if prediction > CONFIG.get('CONVICTION_BYPASS_PRED', 0.97):
                            _conviction_streak[symbol] = _conviction_streak.get(symbol, 0) + 1
                        else:
                            _conviction_streak[symbol] = 0


                        # overwrite this with a more specific reason.
                        decision = "Hold"

                    except Exception as e:
                        logger.error(f"LIVE CYCLE SKIPPED for {symbol}: {str(e)}")
                        decisions.append({'symbol': symbol, 'decision': "Hold (No Data)",
                            'confidence': prediction, 'rsi': current_rsi, 'adx': current_adx,
                            'volatility': current_volatility, 'price': price, 'owned': 0, 'regime': regime})
                        continue


                    qty_owned = 0
                    entry_time = None
                    entry_price = 0.0
                    time_held = 0
                    position_obj = next((pos for pos in open_positions if pos.symbol == symbol), None)
                    if position_obj:
                        qty_owned = int(float(position_obj.qty))
                        order_req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol],
                                                     side=OrderSide.BUY, limit=50)
                        try:
                            orders = trading_client.get_orders(order_req)
                            filled_buy_orders = [o for o in orders if o.status == OrderStatus.FILLED
                                                 and o.side == OrderSide.BUY]
                            if filled_buy_orders:
                                latest_order = max(filled_buy_orders,
                                    key=lambda o: o.filled_at if o.filled_at
                                    else datetime.min.replace(tzinfo=timezone.utc))
                                entry_time = latest_order.filled_at if latest_order.filled_at else now
                            else:
                                entry_time = now
                        except Exception as e:
                            logger.warning(f"Failed to fetch orders for {symbol}: {str(e)}")
                            entry_time = now
                        entry_price = float(position_obj.avg_entry_price)
                        if entry_time:
                            entry_time = entry_time.astimezone(timezone.utc) if entry_time.tzinfo \
                                else entry_time.replace(tzinfo=timezone.utc)
                        time_held = (now - entry_time).total_seconds() / 60 if entry_time else 0


                    _bull_regimes = ["Calm Bull", "Moderate Bull"]
                    _rsi_threshold = (
                        CONFIG.get('RSI_BUY_THRESHOLD_RELAXED', CONFIG['RSI_BUY_THRESHOLD'])
                        if regime in _bull_regimes
                        else CONFIG['RSI_BUY_THRESHOLD']
                    )


                    _cb_active = (
                        CONFIG.get('ENABLE_CONVICTION_BYPASS', False) and
                        _conviction_streak.get(symbol, 0) >= CONFIG.get('CONVICTION_BYPASS_CYCLES', 2) and
                        current_volatility < CONFIG.get('CONVICTION_BYPASS_MAX_VOL', 6.0)
                    )
                    _is_bypass_entry = current_rsi >= _rsi_threshold and _cb_active


                    _volume_ratio = 1.0
                    try:
                        _vol_period = CONFIG.get('VOLUME_MA_PERIOD', 20)
                        if len(df) >= _vol_period + 1:
                            _recent_vol = float(df['volume'].iloc[-1])
                            _vol_ma = float(df['volume'].iloc[-_vol_period-1:-1].mean())
                            if _vol_ma > 0:
                                _volume_ratio = _recent_vol / _vol_ma
                    except Exception:
                        _volume_ratio = 1.0


                    # the group past its limit (% of portfolio), block.
                    _correlation_group_block: Optional[str] = None
                    _corr_groups = CONFIG.get('CORRELATION_GROUPS', {})
                    for _grp_name, _grp_cfg in _corr_groups.items():
                        _grp_symbols = _grp_cfg.get('symbols', [])
                        _grp_cap_pct = _grp_cfg.get('max_pct', 1.0)
                        if symbol not in _grp_symbols:
                            continue
                        _grp_dollar = 0.0
                        for _pos in open_positions:
                            if _pos.symbol in _grp_symbols and _pos.symbol != symbol:
                                try:
                                    _grp_dollar += float(_pos.market_value)
                                except Exception:
                                    pass

                        _projected_pct = (_grp_dollar + max_per_symbol) / portfolio_value if portfolio_value > 0 else 0.0
                        if _projected_pct > _grp_cap_pct:
                            _correlation_group_block = (
                                f"CorrCap {_grp_name} "
                                f"{(_grp_dollar/portfolio_value)*100:.1f}%+new>{_grp_cap_pct*100:.0f}%"
                            )
                            break


                    _daily_breaker_reason: Optional[str] = None
                    _stop_limit = CONFIG.get('DAILY_STOP_COUNT_LIMIT', 0)
                    _loss_limit_pct = CONFIG.get('DAILY_LOSS_PCT_LIMIT', 0.0)
                    if _stop_limit > 0 and _daily_stop_count >= _stop_limit:
                        _daily_breaker_reason = f"{_daily_stop_count} stops today"
                    elif _loss_limit_pct > 0 and _session_start_equity > 0:
                        _daily_dd = (_session_start_equity - portfolio_value) / _session_start_equity
                        if _daily_dd >= _loss_limit_pct:
                            _daily_breaker_reason = f"DD {_daily_dd*100:.1f}% >= {_loss_limit_pct*100:.1f}%"


                    # IMPORTANT: Sell evaluation runs FIRST.


                    if qty_owned > 0 and time_held >= CONFIG['MIN_HOLDING_PERIOD_MINUTES']:
                        max_price = max(float(position_obj.current_price) if position_obj else price, price)
                        _ptype = _entry_type.get(symbol, 'meanrev')

                        if _ptype == 'trend':


                            _profit_atr = CONFIG.get('TREND_PROFIT_LOCK_ATR', 2.0)
                            _tight_mult = CONFIG.get('TREND_TRAIL_ATR_TIGHT', 2.0)
                            _wide_mult  = CONFIG.get('TREND_TRAIL_ATR_MULT', 3.5)
                            _in_profit = (price - entry_price >= _profit_atr * atr_val) if atr_val > 0 else False
                            _trail_mult = _tight_mult if _in_profit else _wide_mult
                            chandelier = max_price - _trail_mult * atr_val
                            hard_floor = entry_price - CONFIG.get('TREND_HARD_STOP_ATR', 5.0) * atr_val
                            _ma_break = (ma50_val == ma50_val) and price < ma50_val   # NaN-safe
                            sell_triggered = (price <= chandelier or price <= hard_floor or
                                              _ma_break or prediction < CONFIG.get('TREND_EXIT_PRED', 0.40))
                            sell_reason = ("trend_trail" if price <= chandelier else
                                           "trend_hardstop" if price <= hard_floor else
                                           "trend_break" if _ma_break else "trend_signal")
                            stop_loss = hard_floor
                            take_profit = float('nan')
                            trailing_stop = chandelier
                            regime_flip_reason = ""
                        else:

                            trailing_stop = max_price * (1 - CONFIG['TRAILING_STOP_PERCENTAGE'])
                            min_stop = entry_price * CONFIG['MIN_STOP_LOSS_PCT']
                            raw_stop = CONFIG['STOP_LOSS_ATR_MULTIPLIER'] * atr_val
                            stop_loss = entry_price - max(raw_stop, min_stop)
                            take_profit = entry_price + CONFIG['TAKE_PROFIT_ATR_MULTIPLIER'] * atr_val


                            regime_flip_exit = False
                            regime_flip_reason = ""
                            if CONFIG.get('ENABLE_REGIME_EXIT', False):
                                _confirm_n = int(CONFIG.get('REGIME_EXIT_CONFIRM_CYCLES', 2))
                                _whitelist = CONFIG.get('BUY_REGIME_WHITELIST', []) or []
                                _max_pred_exit = float(CONFIG.get('REGIME_EXIT_MAX_PRED', 0.65))
                                _profit_lock_atr = float(CONFIG.get('REGIME_EXIT_PROFIT_LOCK_ATR', 1.0))
                                _rh_sym = _regime_history.get(symbol, [])
                                _recent_valid = [r for r in _rh_sym[-(_confirm_n * 2):] if r != "Unknown"][-_confirm_n:]
                                if len(_recent_valid) >= _confirm_n:
                                    _all_out = all(r not in _whitelist for r in _recent_valid)
                                    _model_weak = prediction < _max_pred_exit
                                    _gain_per_share = price - entry_price
                                    _in_strong_profit = (_gain_per_share >= _profit_lock_atr * atr_val) if atr_val > 0 else False
                                    if _all_out and _model_weak and not _in_strong_profit:
                                        regime_flip_exit = True
                                        regime_flip_reason = (
                                            f"regime_flip ({','.join(_recent_valid)}) "
                                            f"pred={prediction:.2f}<{_max_pred_exit:.2f} "
                                            f"gain=${_gain_per_share:.2f}<{_profit_lock_atr:.1f}xATR"
                                        )


                            time_exit = (CONFIG.get('ENABLE_TIME_EXIT', False) and
                                         time_held > CONFIG.get('TIME_EXIT_MINUTES', 240) and
                                         prediction < CONFIG.get('TIME_EXIT_MAX_PRED', 0.52))

                            sell_triggered = (
                                price <= trailing_stop or
                                price <= stop_loss or
                                price >= take_profit or
                                (prediction < CONFIG['PREDICTION_THRESHOLD_SELL'] and current_rsi > CONFIG['RSI_SELL_THRESHOLD']) or
                                (prediction < 0.50 and current_rsi > 65) or
                                regime_flip_exit or time_exit
                            )
                            sell_reason = ("trailing_stop" if price <= trailing_stop else
                                           "stop_loss" if price <= stop_loss else
                                           "take_profit" if price >= take_profit else
                                           "regime_flip" if regime_flip_exit else
                                           "time_weak_signal" if time_exit else "signal")

                        if sell_triggered:
                            decision = f"Sell ({sell_reason})"
                            logger.info(f"Submitting SELL order for {qty_owned} shares of {symbol} "
                                        f"at ${price:.2f} (reason={sell_reason}, type={_ptype})")
                            sell_order = MarketOrderRequest(symbol=symbol, qty=qty_owned,
                                                            side=OrderSide.SELL,
                                                            time_in_force=TimeInForce.DAY)
                            try:
                                trading_client.submit_order(sell_order)
                                # ── Cooldown + daily-stop accounting ──
                                if sell_reason in ("stop_loss", "trailing_stop", "trend_trail", "trend_hardstop"):
                                    _stop_cooldown[symbol] = now
                                    _daily_stop_count += 1
                                    logger.info(f"{symbol} stop-out cooldown set for "
                                                f"{CONFIG['POST_STOP_COOLDOWN_MINUTES']}min "
                                                f"(daily stops: {_daily_stop_count})")
                                elif sell_reason in ("regime_flip", "trend_break"):

                                    _stop_cooldown[symbol] = now
                                    logger.info(f"{symbol} defensive exit ({sell_reason}) — cooldown set "
                                                f"for {CONFIG['POST_STOP_COOLDOWN_MINUTES']}min")
                                _entry_type.pop(symbol, None)
                                proceeds = qty_owned * price - CONFIG['TRANSACTION_COST_PER_TRADE']
                                remaining_cash += proceeds
                                email_body = (
                                    f"Sold {qty_owned} shares of {symbol} at ${price:.2f}\n"
                                    f"Reason: {sell_reason} (entry={_ptype})\n"
                                    + (f"Detail: {regime_flip_reason}\n" if sell_reason == "regime_flip" else "")
                                    + f"Prediction: {prediction:.3f} | Regime: {regime}\n"
                                    f"RSI: {current_rsi:.2f} | Stop: ${stop_loss:.2f} | "
                                    f"Trail/Chandelier: ${trailing_stop:.2f}\n"
                                    f"Cash returning: ${proceeds:,.2f}"
                                )
                                send_email_async(f"Trade: Sold {symbol}", email_body)
                            except Exception as e:
                                logger.error(f"Failed to submit sell order for {symbol}: {str(e)}")
                                decision = "Sell (Failed)"

                    elif current_volatility > CONFIG['MAX_VOLATILITY'] or current_adx < CONFIG['ADX_TREND_THRESHOLD']:
                        decision = "Hold (Filters)"

                    elif CONFIG['PREDICTION_THRESHOLD_SELL'] < prediction < _adaptive_buy_th:
                        decision = "Hold (Low Confidence)"

                    elif (prediction >= _adaptive_buy_th
                          and (current_rsi < _rsi_threshold or _cb_active)):


                        if CONFIG.get('PREVENT_PYRAMIDING', True) and qty_owned > 0:
                            decision = "Hold (Already Held)"


                        elif CONFIG.get('BUY_REGIME_WHITELIST') and regime not in CONFIG['BUY_REGIME_WHITELIST']:
                            decision = f"Hold (Regime={regime})"


                        elif symbol in _stop_cooldown and \
                             (now - _stop_cooldown[symbol]).total_seconds() / 60 < CONFIG['POST_STOP_COOLDOWN_MINUTES']:
                            remaining_cd = CONFIG['POST_STOP_COOLDOWN_MINUTES'] - \
                                           int((now - _stop_cooldown[symbol]).total_seconds() / 60)
                            decision = f"Hold (Cooldown {remaining_cd}m)"


                        elif CONFIG.get('ENABLE_VOLUME_GATE', False) and _volume_ratio < CONFIG.get('VOLUME_CONFIRMATION_MULTIPLIER', 1.2):
                            decision = f"Hold (Low Volume {_volume_ratio:.2f}x)"


                        elif _correlation_group_block:
                            decision = f"Hold ({_correlation_group_block})"


                        # Pause new entries (not exits) once today's losses

                        # market open.
                        elif _daily_breaker_reason is not None:
                            decision = f"Hold (Daily Breaker: {_daily_breaker_reason})"

                        else:
                            decision = "Buy"
                            if atr_val > 0:
                                try:


                                    _regime_risk_pct = CONFIG.get('RISK_BY_REGIME', {}).get(
                                        regime, CONFIG['RISK_PERCENTAGE']
                                    )


                                    _symbol_risk_mult = float(
                                        CONFIG.get('RISK_MULTIPLIER_BY_SYMBOL', {}).get(symbol, 1.0)
                                    )

                                    _bypass_mult = (float(CONFIG.get('CONVICTION_BYPASS_SIZE_MULT', 0.5))
                                                    if _is_bypass_entry else 1.0)


                                    _conf_mult = confidence_size_mult(prediction, _adaptive_buy_th)
                                    _effective_risk_pct = _regime_risk_pct * _symbol_risk_mult * _bypass_mult * _conf_mult
                                    risk_per_trade = remaining_cash * _effective_risk_pct


                                    min_stop = price * CONFIG['MIN_STOP_LOSS_PCT']
                                    stop_loss_distance = max(atr_val * CONFIG['STOP_LOSS_ATR_MULTIPLIER'], min_stop)
                                    if stop_loss_distance <= 0:
                                        raise ValueError("Stop loss distance <= 0")
                                    qty = max(1, int(risk_per_trade / stop_loss_distance))

                                    max_qty_by_cap = int(max_per_symbol / price)
                                    qty = min(qty, max_qty_by_cap)
                                    cost = qty * price + CONFIG['TRANSACTION_COST_PER_TRADE']
                                    if cost > remaining_cash:
                                        qty = max(0, int((remaining_cash - CONFIG['TRANSACTION_COST_PER_TRADE']) / price))
                                        qty = min(qty, max_qty_by_cap)
                                        cost = qty * price + CONFIG['TRANSACTION_COST_PER_TRADE']
                                    if qty > 0 and cost <= remaining_cash:
                                        logger.info(f"Submitting buy order for {qty} shares of {symbol} at ${price:.2f} "
                                                    f"(remaining_cash=${remaining_cash:,.2f}, cap=${max_per_symbol:,.2f})")

                                        # GTC would carry over to tomorrow's open.
                                        order = MarketOrderRequest(symbol=symbol, qty=qty,
                                                                   side=OrderSide.BUY,
                                                                   time_in_force=TimeInForce.DAY)
                                        try:
                                            trading_client.submit_order(order)
                                            # ── Deduct from local cash tracker immediately ──
                                            remaining_cash -= cost
                                            _entry_type[symbol] = 'meanrev'
                                            decision = "Buy (Conviction Bypass)" if _is_bypass_entry else "Buy"
                                            email_body = (
                                                f"Bought {qty} shares of {symbol} at ${price:.2f}\n"
                                                + ("** CONVICTION RSI-BYPASS (half size, "
                                                   f"streak={_conviction_streak.get(symbol, 0)}) **\n" if _is_bypass_entry else "")
                                                + f"Prediction: {prediction:.3f} | Regime: {regime} "
                                                f"(regime_risk={_regime_risk_pct*100:.2f}%, "
                                                f"sym_mult={_symbol_risk_mult:.2f}, "
                                                f"eff_risk={_effective_risk_pct*100:.2f}%)\n"
                                                f"RSI: {current_rsi:.2f} (threshold={_rsi_threshold}) | ADX: {current_adx:.2f}\n"
                                                f"Volatility: {current_volatility:.2f} | ATR: ${atr_val:.2f}\n"
                                                f"Stop distance: ${stop_loss_distance:.2f} "
                                                f"(ATR×{CONFIG['STOP_LOSS_ATR_MULTIPLIER']:.1f} floored at {CONFIG['MIN_STOP_LOSS_PCT']*100:.1f}%)\n"
                                                f"Cash remaining: ${remaining_cash:,.2f}\n"
                                                f"Portfolio Value: ${portfolio_value:,.2f}"
                                            )
                                            send_email_async(f"Trade: Bought {symbol}", email_body)
                                        except Exception as e:
                                            logger.error(f"Failed to submit buy order for {symbol}: {str(e)}")
                                            decision = "Buy (Failed)"
                                            remaining_cash += cost  # roll back the deduction
                                    else:
                                        decision = "Hold (Qty=0 or Insufficient Cash)"
                                except (ValueError, ZeroDivisionError) as e:
                                    decision = "Hold (Calculation Error)"
                            else:
                                decision = "Hold (Invalid ATR)"


                    if (qty_owned == 0 and not decision.startswith(("Buy", "Sell"))
                            and CONFIG.get('ENABLE_TREND_ENTRY', False) and atr_val > 0 and price > 0):
                        _hi = breakout_high
                        _struct_ok = (
                            ma20_val == ma20_val and ma50_val == ma50_val and          # NaN-safe
                            price > ma20_val and ma20_val > ma50_val and               # uptrend structure
                            current_adx >= CONFIG.get('TREND_ADX_MIN', 25.0) and       # strong trend
                            current_volatility <= CONFIG['MAX_VOLATILITY'] and         # sanity vol cap
                            macd_val == macd_val and macd_sig_val == macd_sig_val and
                            macd_val > macd_sig_val and                                # momentum confirm
                            _hi > 0 and price >= CONFIG.get('TREND_BREAKOUT_PCT', 0.97) * _hi and  # near breakout
                            prediction > CONFIG.get('TREND_PRED_MIN', 0.50)            # model not bearish
                        )
                        _wl_t = CONFIG.get('BUY_REGIME_WHITELIST')
                        _in_cd_t = (symbol in _stop_cooldown and
                                    (now - _stop_cooldown[symbol]).total_seconds() / 60 < CONFIG['POST_STOP_COOLDOWN_MINUTES'])
                        if not _struct_ok:
                            pass
                        elif _wl_t and regime not in _wl_t:
                            decision = f"Hold (Trend Regime={regime})"
                        elif _in_cd_t:
                            decision = "Hold (Trend Cooldown)"
                        elif CONFIG.get('ENABLE_VOLUME_GATE', False) and _volume_ratio < CONFIG.get('VOLUME_CONFIRMATION_MULTIPLIER', 1.2):
                            decision = f"Hold (Trend Low Volume {_volume_ratio:.2f}x)"
                        elif _correlation_group_block:
                            decision = f"Hold (Trend {_correlation_group_block})"
                        elif _daily_breaker_reason is not None:
                            decision = f"Hold (Trend Daily Breaker)"
                        else:
                            try:
                                _sm_t = float(CONFIG.get('RISK_MULTIPLIER_BY_SYMBOL', {}).get(symbol, 1.0))
                                _cm_t = confidence_size_mult(prediction, CONFIG.get('TREND_PRED_MIN', 0.50))
                                _eff_t = CONFIG.get('TREND_RISK_PCT', 0.015) * _sm_t * _cm_t
                                risk_per_trade = remaining_cash * _eff_t
                                stop_dist = CONFIG.get('TREND_TRAIL_ATR_MULT', 3.5) * atr_val
                                if stop_dist <= 0:
                                    raise ValueError("trend stop_dist <= 0")
                                qty = max(1, int(risk_per_trade / stop_dist))
                                max_qty_by_cap = int(max_per_symbol / price)
                                qty = min(qty, max_qty_by_cap)
                                cost = qty * price + CONFIG['TRANSACTION_COST_PER_TRADE']
                                if cost > remaining_cash:
                                    qty = max(0, int((remaining_cash - CONFIG['TRANSACTION_COST_PER_TRADE']) / price))
                                    qty = min(qty, max_qty_by_cap)
                                    cost = qty * price + CONFIG['TRANSACTION_COST_PER_TRADE']
                                if qty > 0 and cost <= remaining_cash:
                                    order = MarketOrderRequest(symbol=symbol, qty=qty,
                                                               side=OrderSide.BUY,
                                                               time_in_force=TimeInForce.DAY)
                                    trading_client.submit_order(order)
                                    remaining_cash -= cost
                                    _entry_type[symbol] = 'trend'
                                    decision = "Buy (Trend)"
                                    logger.info(f"Submitting TREND buy order for {qty} shares of {symbol} at ${price:.2f} "
                                                f"(ADX={current_adx:.1f}, eff_risk={_eff_t*100:.2f}%, cap=${max_per_symbol:,.2f})")
                                    email_body = (
                                        f"TREND BUY {qty} shares of {symbol} at ${price:.2f}\n"
                                        f"Structure: close > MA20(${ma20_val:.2f}) > MA50(${ma50_val:.2f}), "
                                        f"ADX={current_adx:.1f}, MACD>signal, "
                                        f"price >= {CONFIG.get('TREND_BREAKOUT_PCT',0.97)*100:.0f}% of {_hi:.2f} high\n"
                                        f"Prediction: {prediction:.3f} | Regime: {regime} | RSI: {current_rsi:.2f} (overbought OK in trend)\n"
                                        f"Exit: chandelier (max - {CONFIG.get('TREND_TRAIL_ATR_MULT',3.5)}xATR) / MA50 break / "
                                        f"hard -{CONFIG.get('TREND_HARD_STOP_ATR',5.0)}xATR / pred<{CONFIG.get('TREND_EXIT_PRED',0.40)}\n"
                                        f"eff_risk={_eff_t*100:.2f}% | ATR=${atr_val:.2f} | Cash remaining: ${remaining_cash:,.2f}"
                                    )
                                    send_email_async(f"Trade: Trend Bought {symbol}", email_body)
                                else:
                                    decision = "Hold (Trend Qty=0 or Cash)"
                            except Exception as e:
                                logger.error(f"Trend buy failed for {symbol}: {str(e)}")
                                decision = "Hold (Trend Calc Error)"

                    decisions.append({
                        'symbol': symbol,
                        'decision': decision,
                        'confidence': prediction,
                        'rsi': current_rsi,
                        'adx': current_adx,
                        'volatility': current_volatility,
                        'price': price,
                        'owned': qty_owned,
                        'regime': regime
                    })

                # Refresh account and positions after trades
                try:
                    account = get_account_with_retry()
                    portfolio_value = float(account.equity)
                except RETRYABLE_ERRORS as e:
                    logger.warning(f"Post-trade account refresh failed ({type(e).__name__}) — using pre-trade value")
                except Exception as e:
                    logger.warning(f"Post-trade account refresh unexpected error — using pre-trade value")
                post_trade_owned = {}
                for symbol in CONFIG['SYMBOLS']:
                    try:
                        position = trading_client.get_open_position(symbol)
                        post_trade_owned[symbol] = int(float(position.qty))
                    except (APIError, *RETRYABLE_ERRORS):
                        post_trade_owned[symbol] = 0


                any_trade = any(d['decision'].startswith(('Buy', 'Sell')) for d in decisions)
                if CONFIG['EMAIL_SUMMARY_ALWAYS'] or any_trade:
                    summary_body = "Trading Summary:\n"
                    for dec in decisions:
                        owned_display = post_trade_owned.get(dec['symbol'], dec['owned'])


                        pnl_str = ""
                        if owned_display > 0:
                            pos_obj = next((p for p in open_positions if p.symbol == dec['symbol']), None)
                            if pos_obj is not None:
                                try:
                                    unrealized_pl = float(pos_obj.unrealized_pl)
                                    pnl_sign = "+" if unrealized_pl >= 0 else ""
                                    pnl_str = f", P&L: {pnl_sign}${unrealized_pl:,.2f}"
                                except Exception:
                                    pnl_str = ""
                        summary_body += (
                            f"{dec['symbol']}: {dec['decision']}, Regime: {dec.get('regime', 'Unknown')}, "
                            f"Confidence: {dec['confidence']:.3f}, RSI: {dec['rsi']:.2f}, "
                            f"ADX: {dec['adx']:.2f}, Volatility: {dec['volatility']:.2f}, "
                            f"Price: ${dec['price']:.2f}, Owned: {owned_display}{pnl_str}\n"
                        )
                    summary_body += f"\nPortfolio Value: ${portfolio_value:.2f}"
                    send_email_async("Trading Summary", summary_body)

            else:
                # Market is closed — count down to next open.


                # next_open from the clock is already UTC-aware.
                next_open = clock.next_open
                logger.info(f"Market closed. Next open: {next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                while True:
                    now_utc = datetime.now(timezone.utc)
                    if now_utc >= next_open:
                        break
                    time_left = next_open - now_utc
                    total_secs = max(0.0, time_left.total_seconds())
                    hours, remainder = divmod(total_secs, 3600)
                    minutes_left, seconds_left = divmod(remainder, 60)
                    timer_str = f"{int(hours):02}:{int(minutes_left):02}:{int(seconds_left):02}"
                    print(f"\r{Fore.RED}Time until market opens: {timer_str}{Style.RESET_ALL}", end='', flush=True)


                    sleep_chunk = min(30, max(1, int(total_secs)))
                    time.sleep(sleep_chunk)
                    if sleep_chunk == 30:
                        try:
                            refreshed = get_clock_with_retry()
                            next_open = refreshed.next_open
                        except Exception:
                            pass  # keep using the last known next_open
                print()  # newline after countdown clears

    else:


        all_models_cached = all(
            os.path.exists(os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")) and
            os.path.exists(os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")) and
            os.path.exists(os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_hmm_{CONFIG['MODEL_VERSION']}.pkl"))
            for symbol in CONFIG['SYMBOLS']
        )

        dfs_backtest = {}
        now = datetime.now(timezone.utc)
        end_date = (now - timedelta(days=1)).strftime('%Y-%m-%d')

        print(f"{Fore.CYAN}Loading and preparing backtest data with technical indicators...{Style.RESET_ALL}")
        for symbol in tqdm(CONFIG['SYMBOLS'], desc="Fetching + Processing backtest data"):
            df = load_data(symbol, CONFIG['TRAIN_DATA_START_DATE'], end_date)

            df = calculate_indicators(df, sentiment=0.0)
            dfs_backtest[symbol] = df

        if all_models_cached and not force_train:
            print(f"{Fore.GREEN}✓ All models loaded from cache — running pure backtest{Style.RESET_ALL}")
            logger.info("Pure backtest with cached models")

            for symbol in CONFIG['SYMBOLS']:
                model, scaler, sentiment, hmm, xgb_model = load_model_and_scaler(symbol, expected_features, force_retrain=False)
                models[symbol] = model
                scalers[symbol] = scaler
                hmms[symbol] = hmm
                xgb_models[symbol] = xgb_model

            attempt_results = []
            effective_max = 1
        else:
            print(f"{Fore.YELLOW}⚠ Models missing or --force-train — running full training{Style.RESET_ALL}")
            attempt_results = []
            if CONFIG.get('FORCE_FULL_RETRAIN_RUN', False):
                effective_max = CONFIG['MAX_RETRAIN_ATTEMPTS']
            elif CONFIG['ENABLE_RETRAIN_CYCLE'] and force_train:
                effective_max = CONFIG['MAX_RETRAIN_ATTEMPTS']
            else:
                effective_max = 1

        if effective_max > 0:
            for symbol in CONFIG['SYMBOLS']:

                if CONFIG.get('USE_TRIPLE_BARRIER', True):
                    dfs_backtest[symbol]['Future_Direction'] = compute_triple_barrier_label(
                        dfs_backtest[symbol]['close'].values, dfs_backtest[symbol]['ATR'].values,
                        CONFIG['LOOK_AHEAD_BARS'], CONFIG['TB_TP_ATR'], CONFIG['TB_SL_ATR'])
                else:
                    dfs_backtest[symbol]['Future_Direction'] = (dfs_backtest[symbol]['close'].shift(-CONFIG['LOOK_AHEAD_BARS']) > dfs_backtest[symbol]['close']).astype(int)

            start_attempt = CONFIG.get('RESUME_FROM_ATTEMPT') or 1
            if start_attempt > 1:
                logger.info(f"RESUME: skipping to attempt {start_attempt}")

            for retrain_attempts in range(start_attempt, effective_max + 1):
                logger.info(f"Retraining attempt {retrain_attempts}/{effective_max}")

                if force_train or retrain_attempts > 1:
                    for symbol in CONFIG['SYMBOLS']:
                        train_cache = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_train_data_{CONFIG['TRAIN_DATA_START_DATE']}_{CONFIG['VAL_END_DATE']}.pkl")
                        if os.path.exists(train_cache):
                            os.remove(train_cache)
                            logger.info(f"Deleted stale training cache for {symbol} (force retrain)")

                models = {}
                scalers = {}
                sentiments = {}
                hmms = {}
                xgb_models = {}
                training_times_dictionary = {}
                if 'progress_bar' in locals() and progress_bar is not None:
                    progress_bar.close()
                progress_bar = None

                if force_train or retrain_attempts > 1:
                    for symbol in CONFIG['SYMBOLS']:
                        model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
                        scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
                        if os.path.exists(model_path):
                            os.remove(model_path)
                            logger.info(f"Deleted existing model for {symbol} to force retrain.")
                        if os.path.exists(scaler_path):
                            os.remove(scaler_path)
                            logger.info(f"Deleted existing scaler for {symbol} to force retrain.")

                need_training = any(load_model_and_scaler(symbol, expected_features, force_train or retrain_attempts > 1)[0] is None for symbol in CONFIG['SYMBOLS'])
                progress_bar = tqdm(total=total_epochs, desc="Training Progress", bar_format="{l_bar}\033[32m{bar}\033[0m{r_bar}") if need_training else None

                if torch.cuda.is_available():
                    gpu_name, free_gb = _get_gpu_info_safe()


                    estimated_gb_per_worker = CONFIG.get('GB_PER_WORKER_EST', 2.0)
                    _n_symbols = len(CONFIG['SYMBOLS'])
                    group_size = max(1, min(_n_symbols, CONFIG['NUM_PARALLEL_WORKERS'],
                                            int(free_gb // estimated_gb_per_worker)))
                    if free_gb < 6.0:
                        group_size = 1
                        print(f"{Fore.YELLOW}Very low VRAM ({free_gb:.1f} GB) → 1 worker. "
                              f"Free GPU memory (stop the live bot) for full parallelism.{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.LIGHTGREEN_EX}{free_gb:.1f} GB available on {Fore.LIGHTGREEN_EX}{gpu_name}{Fore.GREEN} → using {Fore.LIGHTGREEN_EX}{group_size} parallel workers{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}CUDA not available → No GPU detected, defaulting to CPU{Style.RESET_ALL}")
                    group_size = CONFIG['NUM_PARALLEL_WORKERS']
                    free_gb = 0.0


                _total_cores = os.cpu_count() or 8
                _cfg_tpw = int(CONFIG.get('CPU_THREADS_PER_WORKER', 0) or 0)
                _threads_per_worker = _cfg_tpw if _cfg_tpw > 0 else max(1, _total_cores // max(1, group_size))
                for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
                    os.environ[_v] = str(_threads_per_worker)
                os.environ['DT_THREADS_PER_WORKER'] = str(_threads_per_worker)
                print(f"{Fore.LIGHTGREEN_EX}CPU thread budget: {_threads_per_worker} threads/worker "
                      f"× {group_size} workers = {_threads_per_worker * group_size} / {_total_cores} cores "
                      f"(no oversubscription){Style.RESET_ALL}")


                mp.set_start_method('spawn', force=True)
                start_total_training_time = time.perf_counter()

                with mp.Manager() as manager:
                    gpu_semaphore = manager.Semaphore(group_size)
                    with mp.Pool(processes=group_size) as pool:
                        worker_tasks = [(i+1, sym, expected_features, force_train or retrain_attempts > 1, None, gpu_semaphore, backtest_only, debug)
                                      for i, sym in enumerate(CONFIG['SYMBOLS'])]
                        outputs = list(tqdm(pool.imap(train_wrapper, worker_tasks),
                            total=len(CONFIG['SYMBOLS']), desc="Processing symbols"))

                if torch.cuda.is_available():
                    _warmup = torch.zeros(1, device='cuda')
                    del _warmup
                    torch.cuda.empty_cache()

                logger.info("Parallel processing completed; CUDA memory cleared.")

                end_total_training_time = time.perf_counter()
                total_training_time_in_milliseconds = (end_total_training_time - start_total_training_time) * 1000

                for output_tuple in outputs:
                    symbol = output_tuple[0]
                    model = output_tuple[1]
                    scaler = output_tuple[2]
                    data_loaded = output_tuple[3]
                    sentiment = output_tuple[4]
                    sentiment_loaded = output_tuple[5]
                    model_loaded = output_tuple[6]
                    hmm = output_tuple[7]
                    xgb_model = output_tuple[8]
                    training_time_in_milliseconds = output_tuple[9]

                    training_times_dictionary[symbol] = training_time_in_milliseconds if training_time_in_milliseconds is not None else 0
                    models[symbol] = model
                    scalers[symbol] = scaler
                    sentiments[symbol] = sentiment


                    hmms[symbol] = hmm
                    xgb_models[symbol] = xgb_model

                if progress_bar:
                    progress_bar.close()

                for symbol in CONFIG['SYMBOLS']:
                    model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
                    scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
                    attempt_model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}_attempt{retrain_attempts}.pth")
                    attempt_scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}_attempt{retrain_attempts}.pkl")
                    if os.path.exists(model_path):
                        shutil.copyfile(model_path, attempt_model_path)
                    if os.path.exists(scaler_path):
                        shutil.copyfile(scaler_path, attempt_scaler_path)

                backtest_times_dictionary = {}
                accuracies_dictionary = {}
                start_total_backtest_time = time.perf_counter()
                initial_cash = CONFIG['INITIAL_CASH']
                final_value = 0
                symbol_results = {}
                trade_counts = {}
                win_rates = {}
                portfolio_series_per_symbol = {}
                initial_per_symbol = CONFIG['INITIAL_CASH'] / len(CONFIG['SYMBOLS'])

                for symbol in CONFIG['SYMBOLS']:
                    if symbol in models and models[symbol] is not None:
                        start_backtest_time_for_symbol = time.perf_counter()
                        cash, returns, trade_count, win_rate, accuracy_percentage, portfolio_series = backtest(
                            symbol, models[symbol], scalers[symbol], dfs_backtest[symbol], initial_per_symbol,
                            CONFIG['STOP_LOSS_ATR_MULTIPLIER'], CONFIG['TAKE_PROFIT_ATR_MULTIPLIER'],
                            CONFIG['TIMESTEPS'], CONFIG['PREDICTION_THRESHOLD_BUY'], CONFIG['PREDICTION_THRESHOLD_SELL'],
                            CONFIG['MIN_HOLDING_PERIOD_MINUTES'], CONFIG['TRANSACTION_COST_PER_TRADE'],
                            xgb_models.get(symbol), None, dfs_backtest, hmms, scalers, debug=debug
                        )
                    else:
                        cash = initial_per_symbol
                        returns = []
                        trade_count = 0
                        win_rate = 0.0
                        accuracy_percentage = 0.0
                        portfolio_series = pd.Series(dtype=float)

                    trade_counts[symbol] = trade_count
                    win_rates[symbol] = win_rate
                    end_backtest_time_for_symbol = time.perf_counter()
                    backtest_times_dictionary[symbol] = (end_backtest_time_for_symbol - start_backtest_time_for_symbol) * 1000
                    accuracies_dictionary[symbol] = accuracy_percentage
                    final_value += cash
                    portfolio_series_per_symbol[symbol] = portfolio_series

                    symbol_results[symbol] = calculate_performance_metrics(returns, cash, initial_per_symbol)
                    mc_metrics = monte_carlo_simulation(returns, initial_per_symbol)
                    symbol_results[symbol].update(mc_metrics)

                end_total_backtest_time = time.perf_counter()
                total_backtest_time_in_milliseconds = (end_total_backtest_time - start_total_backtest_time) * 1000

                bh_final_value, _ = buy_and_hold_backtest(dfs_backtest, initial_cash)

                max_drawdown_across_symbols = max([res['max_drawdown'] for res in symbol_results.values()]) if symbol_results else 0.0

                portfolio_metrics = compute_portfolio_metrics(portfolio_series_per_symbol)
                portfolio_sharpe = portfolio_metrics['portfolio_sharpe']


                criteria_met = (
                    portfolio_sharpe >= CONFIG.get('SHARPE_TARGET', 1.0) and
                    portfolio_metrics['portfolio_max_drawdown'] <= CONFIG['MAX_ALLOWED_DRAWDOWN']
                )

                email_body = format_email_body(initial_cash, final_value, symbol_results, trade_counts, win_rates)
                email_body += "\n\nMonte Carlo Simulation Summary (per symbol):\n"
                for symbol in CONFIG['SYMBOLS']:
                    if symbol in symbol_results:
                        mc = symbol_results[symbol]
                        email_body += f"{symbol}: MC Mean Final: ${mc['mc_mean_final_value']:.2f}, MC Median Final: ${mc['mc_median_final_value']:.2f}, MC 95% VaR: {mc['mc_var_95']:.3f}%, MC Prob Profit: {mc['mc_prob_profit']:.3f}%\n"
                email_body += f"\nBuy-and-Hold Final Value: ${bh_final_value:.2f}\nDay Trading {'beats' if final_value > bh_final_value else 'does not beat'} Buy-and-Hold."
                email_body += f"\nAttempt: {retrain_attempts}"
                email_body += f"\nCriteria Met: {criteria_met}"
                send_email(f"Backtest Attempt {retrain_attempts} Results", email_body)

                print(f"\n{Fore.CYAN}=== Backtest Attempt {retrain_attempts}/{effective_max} Performance Summary ==={Style.RESET_ALL}")
                print(f"{Fore.GREEN}Backtest Performance Summary:{Style.RESET_ALL}")
                print(f"{'Symbol':<8} {'Total Return (%)':<18} {'Sharpe Ratio':<14} {'Max Drawdown (%)':<20} {'Trades':<8} {'Win Rate (%)':<14} {'Accuracy (%)':<14} {'MC Mean Final ($)':<18} {'MC Median Final ($)':<20} {'MC 95% VaR (%)':<15} {'MC Prob Profit (%)':<18}")
                for symbol in CONFIG['SYMBOLS']:
                    if symbol in symbol_results:
                        metrics_for_symbol = symbol_results[symbol]
                        return_color = Fore.GREEN if metrics_for_symbol['total_return'] > 0 else Fore.RED
                        drawdown_color = Fore.RED if metrics_for_symbol['max_drawdown'] > 0 else Fore.GREEN
                        win_rate_color = Fore.GREEN if win_rates.get(symbol, 0) > 50 else Fore.RED
                        accuracy = accuracies_dictionary.get(symbol, 0.0)
                        accuracy_color = Fore.GREEN if accuracy > 50 else Fore.RED
                        mc_mean_color = Fore.GREEN if metrics_for_symbol['mc_mean_final_value'] > initial_per_symbol else Fore.RED
                        mc_median_color = Fore.GREEN if metrics_for_symbol['mc_median_final_value'] > initial_per_symbol else Fore.RED
                        mc_var_color = Fore.RED if metrics_for_symbol['mc_var_95'] > 0 else Fore.GREEN
                        mc_prob_color = Fore.GREEN if metrics_for_symbol['mc_prob_profit'] > 50 else Fore.RED
                        print(f"{symbol:<8} {return_color}{metrics_for_symbol['total_return']:<18.3f}{Style.RESET_ALL} {metrics_for_symbol['sharpe_ratio']:<14.3f} {drawdown_color}{metrics_for_symbol['max_drawdown']:<20.3f}{Style.RESET_ALL} {trade_counts.get(symbol, 0):<8} {win_rate_color}{win_rates.get(symbol, 0):<14.3f}{Style.RESET_ALL} {accuracy_color}{accuracy:<14.3f}{Style.RESET_ALL} {mc_mean_color}{metrics_for_symbol['mc_mean_final_value']:<18.2f}{Style.RESET_ALL} {mc_median_color}{metrics_for_symbol['mc_median_final_value']:<20.2f}{Style.RESET_ALL} {mc_var_color}{metrics_for_symbol['mc_var_95']:<15.3f}{Style.RESET_ALL} {mc_prob_color}{metrics_for_symbol['mc_prob_profit']:<18.3f}{Style.RESET_ALL}")
                print(f"\nBuy-and-Hold Final Value: ${bh_final_value:.2f}")
                attempt_color = Fore.GREEN if final_value > CONFIG['INITIAL_CASH'] else Fore.RED
                print(f"{Fore.YELLOW}→ Attempt {retrain_attempts} Final Portfolio Value: {attempt_color}${final_value:,.2f}{Style.RESET_ALL}")
                _sharpe_color = Fore.GREEN if portfolio_sharpe >= CONFIG.get('SHARPE_TARGET', 1.0) else Fore.RED
                print(f"{Fore.CYAN}→ Portfolio (equity-curve) metrics: "
                      f"Sharpe={_sharpe_color}{portfolio_sharpe:.3f}{Fore.CYAN} "
                      f"(target {CONFIG.get('SHARPE_TARGET', 1.0):.2f}) | "
                      f"Sortino={portfolio_metrics['portfolio_sortino']:.3f} | "
                      f"MaxDD={portfolio_metrics['portfolio_max_drawdown']:.2f}% | "
                      f"Return={portfolio_metrics['portfolio_total_return']:.2f}% | "
                      f"days={portfolio_metrics['portfolio_n_days']} | "
                      f"criteria_met={criteria_met}{Style.RESET_ALL}")


                total_trades = sum(trade_counts.values())
                if total_trades == 0:
                    print(f"{Fore.RED}⚠ WARNING: 0 trades were taken across all symbols in this backtest.{Style.RESET_ALL}")
                    print(f"{Fore.RED}  Even after relaxing the prediction threshold and (in smoke test) several risk filters,{Style.RESET_ALL}")
                    print(f"{Fore.RED}  no entries passed all remaining gates (regime, RSI, cooldowns, volume, etc.).{Style.RESET_ALL}")
                    print(f"{Fore.RED}  In smoke test: this is often due to the backtest period having almost no favorable regimes.{Style.RESET_ALL}")
                    print(f"{Fore.RED}  In full runs: consider lowering TRIPLE_BARRIER_BUY_THRESHOLD further or training much longer.{Style.RESET_ALL}")

                attempt_results.append({
                    'attempt': retrain_attempts,
                    'final_value': final_value,
                    'symbol_results': symbol_results,
                    'trade_counts': trade_counts,
                    'win_rates': win_rates,
                    'bh_final_value': bh_final_value,
                    'max_drawdown_across_symbols': max_drawdown_across_symbols,
                    'criteria_met': criteria_met,
                    'portfolio_metrics': portfolio_metrics,
                    'portfolio_sharpe': portfolio_sharpe,
                    'training_times_dictionary': training_times_dictionary,
                    'backtest_times_dictionary': backtest_times_dictionary,
                    'total_training_time_in_milliseconds': total_training_time_in_milliseconds,
                    'total_backtest_time_in_milliseconds': total_backtest_time_in_milliseconds,
                    'accuracies_dictionary': accuracies_dictionary,
                    'portfolio_series_per_symbol': portfolio_series_per_symbol
                })


                if CONFIG.get('SAVE_ATTEMPT_RESULTS', True):
                    try:
                        results_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], 'attempt_results.json')

                        serial = []
                        for a in attempt_results:
                            s = {k: v for k, v in a.items() if k != 'portfolio_series_per_symbol'}
                            serial.append(s)
                        with open(results_path, 'w') as f:
                            json.dump(serial, f, indent=2, default=str)

                        summary_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f'attempt_{retrain_attempts}_summary.txt')
                        with open(summary_path, 'w') as f:
                            f.write(f"Attempt {retrain_attempts} Final Value: ${final_value:,.2f}\n")
                            f.write(f"Criteria met: {criteria_met}\n")
                            f.write(f"Max DD across symbols: {max_drawdown_across_symbols:.2f}%\n")
                        logger.info(f"Saved attempt {retrain_attempts} results for resume (attempt_results.json)")
                    except Exception as e:
                        logger.warning(f"Could not persist attempt results: {e}")

        if attempt_results:
            best_attempt_per_symbol = {}
            best_symbol_results = {}
            best_trade_counts = {}
            best_win_rates = {}
            best_accuracies = {}
            best_portfolio_series_per_symbol = {}
            bh_final_value = attempt_results[-1]['bh_final_value']
            final_value = 0.0
            initial_per_symbol = CONFIG['INITIAL_CASH'] / len(CONFIG['SYMBOLS'])
            for symbol in CONFIG['SYMBOLS']:


                _scored = []
                for a in attempt_results:
                    _m = a['symbol_results'].get(symbol, {})
                    _tc = a['trade_counts'].get(symbol, 0)
                    _s = selection_score(_m, _tc)
                    if _s is not None:
                        _scored.append((_s, a['attempt']))
                if _scored:
                    best_att_for_sym = max(_scored, key=lambda t: t[0])[1]
                    logger.info(f"[{symbol}] selected attempt {best_att_for_sym} by composite "
                                f"Sharpe-DD score {max(_scored, key=lambda t: t[0])[0]:.3f}")
                else:
                    best_att_for_sym = max(attempt_results, key=lambda x: x['symbol_results'].get(symbol, {}).get('total_return', -float('inf')))['attempt']
                    logger.info(f"[{symbol}] no attempt cleared {CONFIG.get('MIN_TRADES_FOR_SHARPE', 5)}-trade "
                                f"floor; fell back to best raw return (attempt {best_att_for_sym})")
                best_attempt_per_symbol[symbol] = best_att_for_sym
                best_att_results = next(a for a in attempt_results if a['attempt'] == best_att_for_sym)
                best_symbol_results[symbol] = best_att_results['symbol_results'][symbol]
                best_trade_counts[symbol] = best_att_results['trade_counts'][symbol]
                best_win_rates[symbol] = best_att_results['win_rates'][symbol]
                best_accuracies[symbol] = best_att_results['accuracies_dictionary'][symbol]
                best_portfolio_series_per_symbol[symbol] = best_att_results['portfolio_series_per_symbol'][symbol]
                sym_cash = initial_per_symbol * (1 + best_symbol_results[symbol]['total_return'] / 100)
                final_value += sym_cash
                attempt_model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}_attempt{best_att_for_sym}.pth")
                attempt_scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}_attempt{best_att_for_sym}.pkl")
                model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
                scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
                if os.path.exists(attempt_model_path):
                    shutil.copyfile(attempt_model_path, model_path)
                if os.path.exists(attempt_scaler_path):
                    shutil.copyfile(attempt_scaler_path, scaler_path)

            logger.info(f"Selected best models per symbol: {best_attempt_per_symbol}")

            training_times_dictionary = attempt_results[-1]['training_times_dictionary']
            backtest_times_dictionary = attempt_results[-1]['backtest_times_dictionary']
            total_training_time_in_milliseconds = attempt_results[-1]['total_training_time_in_milliseconds']
            total_backtest_time_in_milliseconds = attempt_results[-1]['total_backtest_time_in_milliseconds']

            email_body = format_email_body(CONFIG['INITIAL_CASH'], final_value, best_symbol_results, best_trade_counts, best_win_rates)
            email_body += f"\nBest Models Per Symbol: {', '.join(f'{sym} from attempt {best_attempt_per_symbol[sym]}' for sym in CONFIG['SYMBOLS'])}"
            email_body += "\n\nMonte Carlo Simulation Summary (per symbol):\n"
            for symbol in CONFIG['SYMBOLS']:
                if symbol in best_symbol_results:
                    mc = best_symbol_results[symbol]
                    email_body += f"{symbol}: MC Mean Final: ${mc['mc_mean_final_value']:.2f}, MC Median Final: ${mc['mc_median_final_value']:.2f}, MC 95% VaR: {mc['mc_var_95']:.3f}%, MC Prob Profit: {mc['mc_prob_profit']:.3f}%\n"
            email_body += f"\nBuy-and-Hold Final Value: ${bh_final_value:.2f}\nDay Trading {'beats' if final_value > bh_final_value else 'does not beat'} Buy-and-Hold."
            send_email("Backtest Completed - Best Results", email_body)

            def format_time(ms):
                if ms is None or not isinstance(ms, (int, float)):
                    return "00:00.000"
                minutes = int(ms // 60000)
                seconds = int((ms % 60000) // 1000)
                milliseconds = int(ms % 1000)
                return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

            print(f"{Fore.CYAN}Training Times (mm:ss.mmm):{Style.RESET_ALL}")
            for symbol_in_training_times, time_in_ms in training_times_dictionary.items():
                print(f"  {symbol_in_training_times}: {format_time(time_in_ms)}")
            print(f"Total Training Time: {format_time(total_training_time_in_milliseconds)}")

            print(f"{Fore.CYAN}Backtest Times (mm:ss.mmm):{Style.RESET_ALL}")
            for symbol_in_backtest_times, time_in_ms in backtest_times_dictionary.items():
                print(f"  {symbol_in_backtest_times}: {format_time(time_in_ms)}")
            print(f"Total Backtest Time: {format_time(total_backtest_time_in_milliseconds)}")

            print(f"Total Time (Loading/Training/Backtest): {format_time(total_training_time_in_milliseconds + total_backtest_time_in_milliseconds)}")

            print(f"\n{Fore.CYAN}=== FINAL BEST BACKTEST RESULTS (Best of {len(attempt_results)} attempts) ==={Style.RESET_ALL}")
            print(f"{Fore.GREEN}Final Performance Summary (Selected Best Models):{Style.RESET_ALL}")
            print(f"{'Symbol':<8} {'Attempt':<8} {'Total Return (%)':<18} {'Sharpe Ratio':<14} {'Max Drawdown (%)':<20} {'Trades':<8} {'Win Rate (%)':<14} {'Accuracy (%)':<14} {'MC Mean Final ($)':<18} {'MC Median Final ($)':<20} {'MC 95% VaR (%)':<15} {'MC Prob Profit (%)':<18}")
            for symbol in CONFIG['SYMBOLS']:
                if symbol in best_symbol_results:
                    metrics_for_symbol = best_symbol_results[symbol]
                    attempt = best_attempt_per_symbol[symbol]
                    return_color = Fore.GREEN if metrics_for_symbol['total_return'] > 0 else Fore.RED
                    drawdown_color = Fore.RED if metrics_for_symbol['max_drawdown'] > 0 else Fore.GREEN
                    win_rate_color = Fore.GREEN if best_win_rates[symbol] > 50 else Fore.RED
                    accuracy = best_accuracies[symbol] if best_trade_counts[symbol] > 0 else 0.0
                    accuracy_color = Fore.GREEN if accuracy > 50 else Fore.RED
                    mc_mean_color = Fore.GREEN if metrics_for_symbol['mc_mean_final_value'] > initial_per_symbol else Fore.RED
                    mc_median_color = Fore.GREEN if metrics_for_symbol['mc_median_final_value'] > initial_per_symbol else Fore.RED
                    mc_var_color = Fore.RED if metrics_for_symbol['mc_var_95'] > 0 else Fore.GREEN
                    mc_prob_color = Fore.GREEN if metrics_for_symbol['mc_prob_profit'] > 50 else Fore.RED
                    print(f"{symbol:<8} {attempt:<8} {return_color}{metrics_for_symbol['total_return']:<18.3f}{Style.RESET_ALL} {metrics_for_symbol['sharpe_ratio']:<14.3f} {drawdown_color}{metrics_for_symbol['max_drawdown']:<20.3f}{Style.RESET_ALL} {best_trade_counts.get(symbol, 0):<8} {win_rate_color}{best_win_rates.get(symbol, 0):<14.3f}{Style.RESET_ALL} {accuracy_color}{accuracy:<14.3f}{Style.RESET_ALL} {mc_mean_color}{metrics_for_symbol['mc_mean_final_value']:<18.2f}{Style.RESET_ALL} {mc_median_color}{metrics_for_symbol['mc_median_final_value']:<20.2f}{Style.RESET_ALL} {mc_var_color}{metrics_for_symbol['mc_var_95']:<15.3f}{Style.RESET_ALL} {mc_prob_color}{metrics_for_symbol['mc_prob_profit']:<18.3f}{Style.RESET_ALL}")

            bh_color = Fore.GREEN if CONFIG['INITIAL_CASH'] < bh_final_value else Fore.RED
            print(f"\nBuy-and-Hold Final Value: {bh_color}${bh_final_value:.2f}{Style.RESET_ALL}")
            print(f"Day Trading {'beats' if final_value > bh_final_value else 'does not beat'} Buy-and-Hold.")
            color = Fore.RED if final_value <= CONFIG['INITIAL_CASH'] else Fore.GREEN
            print(f"\nFull Backtest completed. Final value: {color}${final_value:.2f}{Style.RESET_ALL}")


            best_portfolio_metrics = compute_portfolio_metrics(best_portfolio_series_per_symbol)
            _bm_color = Fore.GREEN if best_portfolio_metrics['portfolio_sharpe'] >= CONFIG.get('SHARPE_TARGET', 1.0) else Fore.RED
            print(f"\n{Fore.CYAN}=== PORTFOLIO RISK-ADJUSTED METRICS (selected best models, equity curve) ==={Style.RESET_ALL}")
            print(f"  Portfolio Sharpe  : {_bm_color}{best_portfolio_metrics['portfolio_sharpe']:.3f}{Style.RESET_ALL}  (target {CONFIG.get('SHARPE_TARGET', 1.0):.2f})")
            print(f"  Portfolio Sortino : {best_portfolio_metrics['portfolio_sortino']:.3f}")
            print(f"  Portfolio Max DD  : {best_portfolio_metrics['portfolio_max_drawdown']:.2f}%")
            print(f"  Portfolio Return  : {best_portfolio_metrics['portfolio_total_return']:.2f}% over {best_portfolio_metrics['portfolio_n_days']} trading days")

            print(f"WF_FOLD_METRICS tag={os.environ.get('DT_FOLD_TAG','single')} "
                  f"sharpe={best_portfolio_metrics['portfolio_sharpe']:.6f} "
                  f"sortino={best_portfolio_metrics['portfolio_sortino']:.6f} "
                  f"maxdd={best_portfolio_metrics['portfolio_max_drawdown']:.6f} "
                  f"ret={best_portfolio_metrics['portfolio_total_return']:.6f} "
                  f"days={best_portfolio_metrics['portfolio_n_days']}")

            print(f"\n{Fore.CYAN}=== TOTAL RUN TIME SUMMARY FOR ALL {len(attempt_results)} ATTEMPTS ==={Style.RESET_ALL}")
            total_run_ms = total_training_time_in_milliseconds + total_backtest_time_in_milliseconds
            print(f"{Fore.GREEN}Overall Timing:{Style.RESET_ALL}")
            print(f"  Training Time (best attempt) : {format_time(total_training_time_in_milliseconds)}")
            print(f"  Backtesting Time             : {format_time(total_backtest_time_in_milliseconds)}")
            print(f"  Grand Total Run Time         : {format_time(total_run_ms)}")
            print(f" {Fore.YELLOW} Final Best Portfolio Value{Style.RESET_ALL}   : {color}${final_value:,.2f}")

            # Recompute BH with series for graphing
            bh_final_value, bh_series_per_symbol = buy_and_hold_backtest(dfs_backtest, initial_cash)


            all_series = pd.concat(best_portfolio_series_per_symbol.values(), axis=1, join='outer')
            all_series.index = pd.to_datetime(all_series.index)
            all_series = all_series.ffill().bfill().fillna(CONFIG['INITIAL_CASH'])
            total_portfolio = all_series.sum(axis=1)

            daily_portfolio = total_portfolio.resample('D').last().ffill()

            all_bh_series = pd.concat(bh_series_per_symbol.values(), axis=1, join='outer')
            all_bh_series.index = pd.to_datetime(all_bh_series.index)


            all_bh_series = all_bh_series.ffill().bfill()
            total_bh_portfolio = all_bh_series.sum(axis=1)
            daily_bh_portfolio = total_bh_portfolio.resample('D').last().ffill()

            plt.figure(figsize=(12, 6))
            plt.plot(daily_portfolio.index, daily_portfolio.values, label='Day Trading Portfolio', color='blue', linewidth=2)
            plt.plot(daily_bh_portfolio.index, daily_bh_portfolio.values, label='Buy-and-Hold Portfolio', color='green', linewidth=2)
            # === Cash Breakeven Line ===
            plt.axhline(y=CONFIG['INITIAL_CASH'], color='darkred', linestyle='--', linewidth=1.5, 
                        label='Initial Cash')
            plt.title('Daily Portfolio Value: Day Trading vs Buy-and-Hold Over Backtest Period')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.legend()
            plot_file = os.path.join(CONFIG['MODEL_CACHE_DIR'], 'portfolio_value_graph.png')
            plt.savefig(plot_file)
            plt.close()
            logger.info(f"Portfolio value graph saved to {plot_file}")
            print(f"{Fore.LIGHTBLACK_EX}Portfolio value graph saved to {plot_file}{Style.RESET_ALL}")
            email_body += f"\n\nPortfolio Value Graph (Day Trading vs Buy-and-Hold): Attached as portfolio_value_graph.png."
            send_email("Backtest Completed - Best Results with Graph", email_body, plot_file)
        
def run_walk_forward(args) -> None:
    """v15: orchestrate walk-forward validation as isolated per-fold SUBPROCESSES.

    Why subprocesses: each fold needs a clean CUDA + multiprocessing-spawn state.
    Re-running train+backtest inline in one process risks CUDA-context / resource-
    tracker corruption (the exact crash class documented in __main__). A fresh
    process per fold is the safe, simple isolation boundary.

    Each fold trains on data up to fold['train_end'] and tests on the fold's
    out-of-sample year (fold['test_start']→fold['test_end']) — a window the fold's
    model never trained on. We parse each fold's WF_FOLD_METRICS marker and report
    aggregate OOS Sharpe across bull AND bear regimes.

    NOTE: this is a VALIDATION harness — it overwrites the cached model files per
    fold, so it does not produce the deployment model. Run the normal (non-WF)
    full run to train the model you actually trade with.
    """
    import subprocess
    import re
    import statistics

    folds = list(CONFIG.get('WALK_FORWARD_FOLDS', []))
    if getattr(args, 'wf_max_folds', 0):
        folds = folds[:args.wf_max_folds]
    if not folds:
        print(f"{Fore.RED}No WALK_FORWARD_FOLDS configured — nothing to do.{Style.RESET_ALL}")
        return

    cache_dir = CONFIG['CACHE_DIR']
    os.makedirs(cache_dir, exist_ok=True)
    results = []
    print(f"{Fore.CYAN}{'='*72}")
    print(f"WALK-FORWARD VALIDATION — {len(folds)} folds (each an isolated subprocess)")
    print(f"{'='*72}{Style.RESET_ALL}")

    for i, fold in enumerate(folds, 1):
        tag = fold['tag']
        child_env = os.environ.copy()
        child_env['DT_FOLD_TAG'] = tag
        child_env['DT_TRAIN_END_DATE'] = fold['train_end']
        child_env['DT_VAL_START_DATE'] = fold['val_start']
        child_env['DT_VAL_END_DATE'] = fold['val_end']
        child_env['DT_BACKTEST_START_DATE'] = fold['test_start']
        child_env['DT_BACKTEST_END_DATE'] = fold['test_end']

        cmd = [sys.executable, os.path.abspath(__file__), '--backtest', '--force-train']
        if getattr(args, 'smoke_test', False):
            cmd.append('--smoke-test')
        if getattr(args, 'DEBUG', False):
            cmd.append('--DEBUG')

        log_path = os.path.join(cache_dir, f"wf_fold_{i}_{tag}.txt")
        print(f"{Fore.YELLOW}[Fold {i}/{len(folds)}] {tag}: train≤{fold['train_end']} | "
              f"test {fold['test_start']}→{fold['test_end']}  (log: {log_path}){Style.RESET_ALL}")

        with open(log_path, 'w') as lf:
            proc = subprocess.run(cmd, env=child_env, stdout=lf, stderr=subprocess.STDOUT)

        sharpe = sortino = maxdd = ret = None
        days = 0
        try:
            with open(log_path) as lf:
                for line in lf:
                    if line.startswith('WF_FOLD_METRICS'):
                        m = dict(re.findall(r'(\w+)=(-?\d+\.?\d*)', line))
                        sharpe = float(m['sharpe']);  sortino = float(m['sortino'])
                        maxdd = float(m['maxdd']);     ret = float(m['ret'])
                        days = int(float(m.get('days', '0')))
        except Exception as e:
            print(f"{Fore.RED}  could not parse metrics for fold {tag}: {e}{Style.RESET_ALL}")

        ok = (proc.returncode == 0 and sharpe is not None)
        status = 'OK' if ok else f'FAIL(rc={proc.returncode})'
        results.append({'tag': tag, 'sharpe': sharpe, 'sortino': sortino,
                        'maxdd': maxdd, 'ret': ret, 'days': days, 'status': status})
        _c = Fore.GREEN if (sharpe is not None and sharpe >= CONFIG.get('SHARPE_TARGET', 1.0)) else Fore.RED
        print(f"  → {tag}: {_c}Sharpe={sharpe}{Style.RESET_ALL} Sortino={sortino} "
              f"MaxDD={maxdd}% Ret={ret}% days={days} [{status}]")

    print(f"\n{Fore.CYAN}{'='*72}\nWALK-FORWARD SUMMARY (out-of-sample, no leakage)\n{'='*72}{Style.RESET_ALL}")
    print(f"{'Fold':<16}{'Sharpe':>10}{'Sortino':>10}{'MaxDD%':>10}{'Ret%':>10}{'Status':>14}")
    for r in results:
        _f = lambda v: (v if v is not None else float('nan'))
        print(f"{r['tag']:<16}{_f(r['sharpe']):>10.3f}{_f(r['sortino']):>10.3f}"
              f"{_f(r['maxdd']):>10.2f}{_f(r['ret']):>10.2f}{r['status']:>14}")

    valid = [r['sharpe'] for r in results if r['sharpe'] is not None and r['sharpe'] == r['sharpe']]
    if valid:
        target = CONFIG.get('SHARPE_TARGET', 1.0)
        mean_sh, med_sh, min_sh = statistics.mean(valid), statistics.median(valid), min(valid)
        print(f"\n  Mean Sharpe   : {mean_sh:.3f}")
        print(f"  Median Sharpe : {med_sh:.3f}")
        print(f"  Min  Sharpe   : {min_sh:.3f}  ← worst fold (the generalization stress point)")
        print(f"  Folds ≥ target {target:.2f}: {sum(1 for s in valid if s >= target)}/{len(valid)}")
        try:
            with open(os.path.join(cache_dir, 'walk_forward_results.json'), 'w') as f:
                json.dump({'folds': results, 'mean_sharpe': mean_sh,
                           'median_sharpe': med_sh, 'min_sharpe': min_sh}, f, indent=2)
            print(f"  (saved cache/walk_forward_results.json)")
        except Exception as e:
            print(f"{Fore.YELLOW}  could not save walk_forward_results.json: {e}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}  No valid fold metrics parsed — check the per-fold logs in {cache_dir}.{Style.RESET_ALL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading bot with backtest mode")
    parser.add_argument('--backtest', action='store_true', help="Run in backtest-only mode")
    parser.add_argument('--force-train', action='store_true', help="Force retraining of models")
    parser.add_argument('--DEBUG', action='store_true', help="Enable debug printing for detailed informative outputs")
    parser.add_argument('--reset', action='store_true', help="Force account reset and cash injection (overrides CONFIG flag)")
    parser.add_argument('--horizon', type=int, default=None, help="Override LOOK_AHEAD_BARS (e.g. 21)")
    parser.add_argument('--live-sentiment', action='store_true',
                        help="Enable real-time DistilBERT news sentiment during live trading "
                             "(disabled by default — adds ~5–15s per symbol per cycle)")
    parser.add_argument('--clean-weights', action='store_true',
                        help="Organise the Model Weights directory: archive legacy version files "
                             "and remove empty junk folders, then exit")
    parser.add_argument('--smoke-test', action='store_true',
                        help="v13: dry-run the training pipeline (1 attempt, few epochs) to confirm "
                             "train>val accuracy and no crashes BEFORE committing to a multi-day run")
    parser.add_argument('--resume', action='store_true',
                        help="v14: resume a previous multiday training run from the last completed attempt "
                             "(reads attempt_results.json in MODEL_CACHE_DIR)")
    parser.add_argument('--walk-forward', action='store_true',
                        help="v15: run walk-forward validation — retrain+test on each fold in "
                             "CONFIG['WALK_FORWARD_FOLDS'] as isolated subprocesses, then report "
                             "aggregate out-of-sample Sharpe across bull AND bear regimes.")
    parser.add_argument('--wf-max-folds', type=int, default=0,
                        help="v15: limit walk-forward to the first N folds (0 = all). Use 2 for a quick smoke.")
    args = parser.parse_args()

    if args.clean_weights:
        print(f"{Fore.CYAN}Cleaning Model Weights directory: {CONFIG['MODEL_CACHE_DIR']}{Style.RESET_ALL}")
        clean_weights_directory(CONFIG['MODEL_CACHE_DIR'])
        sys.exit(0)


    # minutes before you spend days on the real retrain.
    if args.smoke_test or CONFIG.get('SMOKE_TEST', False):
        _smoke_epochs = CONFIG.get('SMOKE_TEST_EPOCHS', 3)
        os.environ['DT_SMOKE_TEST'] = '1'
        os.environ['DT_SMOKE_EPOCHS'] = str(_smoke_epochs)
        CONFIG['SMOKE_TEST'] = True
        CONFIG['MAX_RETRAIN_ATTEMPTS'] = 1
        CONFIG['FORCE_FULL_RETRAIN_RUN'] = False
        CONFIG['TRAIN_EPOCHS'] = _smoke_epochs
        CONFIG['NUM_MC_SIMULATIONS'] = 1000
        print(f"{Fore.YELLOW}*** SMOKE TEST MODE *** 1 attempt, {CONFIG['TRAIN_EPOCHS']} epochs, "
              f"MC=1000 — validate the pipeline before the real run.{Style.RESET_ALL}")


    _wf_env_map = {
        'DT_TRAIN_END_DATE': 'TRAIN_END_DATE',
        'DT_VAL_START_DATE': 'VAL_START_DATE',
        'DT_VAL_END_DATE': 'VAL_END_DATE',
        'DT_BACKTEST_START_DATE': 'BACKTEST_START_DATE',
        'DT_BACKTEST_END_DATE': 'BACKTEST_END_DATE',
    }
    _wf_applied = []
    for _env_k, _cfg_k in _wf_env_map.items():
        _v = os.environ.get(_env_k)
        if _v:
            CONFIG[_cfg_k] = _v
            _wf_applied.append(f"{_cfg_k}={_v}")
    if _wf_applied:
        print(f"{Fore.MAGENTA}WALK-FORWARD FOLD [{os.environ.get('DT_FOLD_TAG','?')}] date overrides: "
              f"{', '.join(_wf_applied)}{Style.RESET_ALL}")


    _wl_env = os.environ.get('DT_BUY_REGIME_WHITELIST')
    if _wl_env is not None:
        if _wl_env.strip().lower() == 'none':
            CONFIG['BUY_REGIME_WHITELIST'] = None
        else:
            CONFIG['BUY_REGIME_WHITELIST'] = [s.strip() for s in _wl_env.split(',') if s.strip()]
        print(f"{Fore.MAGENTA}OVERRIDE: BUY_REGIME_WHITELIST set to "
              f"{CONFIG['BUY_REGIME_WHITELIST']}{Style.RESET_ALL}")


    _mra_env = os.environ.get('DT_MAX_RETRAIN_ATTEMPTS')
    if _mra_env:
        try:
            CONFIG['MAX_RETRAIN_ATTEMPTS'] = int(_mra_env)
            print(f"{Fore.MAGENTA}OVERRIDE: MAX_RETRAIN_ATTEMPTS set to "
                  f"{CONFIG['MAX_RETRAIN_ATTEMPTS']}{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.YELLOW}Ignoring invalid DT_MAX_RETRAIN_ATTEMPTS={_mra_env}{Style.RESET_ALL}")


    _csm_env = os.environ.get('DT_CONFIDENCE_SIZE_MAX_MULT')
    if _csm_env:
        try:
            CONFIG['CONFIDENCE_SIZE_MAX_MULT'] = float(_csm_env)
            print(f"{Fore.MAGENTA}OVERRIDE: CONFIDENCE_SIZE_MAX_MULT set to "
                  f"{CONFIG['CONFIDENCE_SIZE_MAX_MULT']}{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.YELLOW}Ignoring invalid DT_CONFIDENCE_SIZE_MAX_MULT={_csm_env}{Style.RESET_ALL}")
    _rsk_env = os.environ.get('DT_RISK_PERCENTAGE')
    if _rsk_env:
        try:
            CONFIG['RISK_PERCENTAGE'] = float(_rsk_env)
            print(f"{Fore.MAGENTA}OVERRIDE: RISK_PERCENTAGE set to {CONFIG['RISK_PERCENTAGE']}{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.YELLOW}Ignoring invalid DT_RISK_PERCENTAGE={_rsk_env}{Style.RESET_ALL}")
    _mps_env = os.environ.get('DT_MAX_POSITION_SIZE_PCT')
    if _mps_env:
        try:
            CONFIG['MAX_POSITION_SIZE_PCT'] = float(_mps_env)
            print(f"{Fore.MAGENTA}OVERRIDE: MAX_POSITION_SIZE_PCT set to {CONFIG['MAX_POSITION_SIZE_PCT']}{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.YELLOW}Ignoring invalid DT_MAX_POSITION_SIZE_PCT={_mps_env}{Style.RESET_ALL}")


    def _env_bool(name):
        v = os.environ.get(name)
        if v is None:
            return None
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    _evg = _env_bool('DT_ENABLE_VOLUME_GATE')
    if _evg is not None:
        CONFIG['ENABLE_VOLUME_GATE'] = _evg
        print(f"{Fore.MAGENTA}OVERRIDE: ENABLE_VOLUME_GATE set to {_evg}{Style.RESET_ALL}")
    _ete = _env_bool('DT_ENABLE_TREND_ENTRY')
    if _ete is not None:
        CONFIG['ENABLE_TREND_ENTRY'] = _ete
        print(f"{Fore.MAGENTA}OVERRIDE: ENABLE_TREND_ENTRY set to {_ete}{Style.RESET_ALL}")
    for _rk in ('DT_RSI_BUY_THRESHOLD', 'DT_RSI_BUY_THRESHOLD_RELAXED'):
        _rv = os.environ.get(_rk)
        if _rv:
            try:
                CONFIG[_rk[3:]] = int(_rv)
                print(f"{Fore.MAGENTA}OVERRIDE: {_rk[3:]} set to {CONFIG[_rk[3:]]}{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.YELLOW}Ignoring invalid {_rk}={_rv}{Style.RESET_ALL}")


    _rf_env = os.environ.get('DT_RISK_FLAT')
    if _rf_env:
        try:
            _rf = float(_rf_env)
            CONFIG['RISK_BY_REGIME'] = {}
            CONFIG['RISK_PERCENTAGE'] = _rf
            print(f"{Fore.MAGENTA}OVERRIDE: RISK_FLAT set to {_rf} (RISK_BY_REGIME cleared){Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.YELLOW}Ignoring invalid DT_RISK_FLAT={_rf_env}{Style.RESET_ALL}")
    for _fk in ('DT_STOP_LOSS_ATR_MULTIPLIER', 'DT_TAKE_PROFIT_ATR_MULTIPLIER'):
        _fv = os.environ.get(_fk)
        if _fv:
            try:
                CONFIG[_fk[3:]] = float(_fv)
                print(f"{Fore.MAGENTA}OVERRIDE: {_fk[3:]} set to {CONFIG[_fk[3:]]}{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.YELLOW}Ignoring invalid {_fk}={_fv}{Style.RESET_ALL}")
    _te = _env_bool('DT_ENABLE_TIME_EXIT')
    if _te is not None:
        CONFIG['ENABLE_TIME_EXIT'] = _te
        print(f"{Fore.MAGENTA}OVERRIDE: ENABLE_TIME_EXIT set to {_te}{Style.RESET_ALL}")
    _pp = _env_bool('DT_PREVENT_PYRAMIDING')
    if _pp is not None:
        CONFIG['PREVENT_PYRAMIDING'] = _pp
        print(f"{Fore.MAGENTA}OVERRIDE: PREVENT_PYRAMIDING set to {_pp}{Style.RESET_ALL}")
    _ctf = os.environ.get('DT_CALIB_TEMP_FLOOR')
    if _ctf:
        try:
            CONFIG['CALIB_TEMP_FLOOR'] = float(_ctf)
            print(f"{Fore.MAGENTA}OVERRIDE: CALIB_TEMP_FLOOR set to {CONFIG['CALIB_TEMP_FLOOR']}{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.YELLOW}Ignoring invalid DT_CALIB_TEMP_FLOOR={_ctf}{Style.RESET_ALL}")
    _clp = os.environ.get('DT_CORE_LONG_PCT')
    if _clp:
        try:
            CONFIG['CORE_LONG_PCT'] = float(_clp)
            print(f"{Fore.MAGENTA}OVERRIDE: CORE_LONG_PCT set to {CONFIG['CORE_LONG_PCT']}{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.YELLOW}Ignoring invalid DT_CORE_LONG_PCT={_clp}{Style.RESET_ALL}")
    _ccl = os.environ.get('DT_CORE_CONF_LOOKBACK')
    if _ccl:
        try:
            CONFIG['CORE_CONF_LOOKBACK'] = int(_ccl)
            print(f"{Fore.MAGENTA}OVERRIDE: CORE_CONF_LOOKBACK set to {CONFIG['CORE_CONF_LOOKBACK']}{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.YELLOW}Ignoring invalid DT_CORE_CONF_LOOKBACK={_ccl}{Style.RESET_ALL}")


    if args.resume:
        resume_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], 'attempt_results.json')
        if os.path.exists(resume_path):
            try:
                with open(resume_path) as f:
                    prev = json.load(f)
                if prev:
                    last_done = max(a.get('attempt', 0) for a in prev)
                    CONFIG['RESUME_FROM_ATTEMPT'] = last_done + 1
                    print(f"{Fore.CYAN}RESUME MODE: will start from attempt {CONFIG['RESUME_FROM_ATTEMPT']} "
                          f"(found {last_done} previous attempts){Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.YELLOW}Could not parse previous attempt_results.json: {e}{Style.RESET_ALL}")

    if args.live_sentiment:
        CONFIG['_LIVE_SENTIMENT_ENABLED'] = True
        print(f"{Fore.CYAN}Live sentiment enabled — DistilBERT will score news each cycle{Style.RESET_ALL}")
    else:
        CONFIG['_LIVE_SENTIMENT_ENABLED'] = False

    if args.horizon is not None:
        CONFIG['LOOK_AHEAD_BARS'] = args.horizon
        print(f"{Fore.CYAN}OVERRIDE: LOOK_AHEAD_BARS set to {args.horizon}{Style.RESET_ALL}")


    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(CONFIG['LOG_FILE'])
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.WARNING if not args.DEBUG else logging.INFO)  # Hide INFO on console unless --DEBUG
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logger = logging.getLogger(__name__)


    cleanup_account_on_start(force_reset=args.reset)


    if getattr(args, 'walk_forward', False):
        run_walk_forward(args)
        sys.exit(0)


    # anywhere (SSL, GC, talib) then raises
    #   "free(): invalid next size (fast)" → SIGABRT.


    # initialise cleanly.
    main(backtest_only=args.backtest, force_train=args.force_train, debug=args.DEBUG)
