# 🤖 Autonomous NSE Algorithmic Trading Bot & Ensemble AI System

An institutional-grade, event-driven algorithmic trading engine engineered for high-frequency 5-minute candles on the **Indian Stock Market (NSE)**. 

The bot pairs **Smart Money Concepts (SMC)** and institutional technical indicators with a **4-Model Machine Learning Ensemble (Deep LSTM + XGBoost + LightGBM + Logistic Regression Meta-Classifier)**, dynamic Indian Rupee (₹) position sizing, and strict risk guardrails.

---

## 🏗️ System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │   Market Feed (WebSocket / REST / Polling)   │
                       │   - NSE: NIFTY 50 (^NSEI), Reliance, TCS     │
                       └──────────────────────┬───────────────────────┘
                                              │ TICK_RECEIVED
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │        Candle Buffer (5-Minute / 300s)       │
                       └──────────────────────┬───────────────────────┘
                                              │ CANDLE_CLOSED
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
       ┌──────────────────────────────┐                ┌──────────────────────────────┐
       │   Technical & SMC Suite      │                │   AI Ensemble Forecaster     │
       │   - Smart Money Concepts     │                │   - Deep LSTM Network        │
       │     (BOS, CHoCH, OrderBlocks)│                │   - XGBoost Classifier       │
       │   - AlphaTrend & AlphaTrend+ │                │   - LightGBM Classifier      │
       │   - Hidden Supply/Demand     │                │   - Stacking Meta-Classifier │
       │   - Predictive MACD & HW-RSI │                └──────────────┬───────────────┘
       │   - Fibonacci & Harmonics    │                               │
       └──────────────┬───────────────┘                               │
                      │ INDICATORS_READY                              │ FORECAST_READY
                      └───────────────────────┬───────────────────────┘
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │    Decision Engine & Confluence Scorer       │
                       │    - Weighted voting (-1.0 to +1.0)          │
                       │    - Requires ≥ 3 independent confirmations  │
                       └──────────────────────┬───────────────────────┘
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │   Institutional Risk & Safety Filter         │
                       │   - Spread filter (max ₹5.00)                │
                       │   - Volatility spike rejection (3x ATR)      │
                       │   - Daily loss ceiling (₹2,000 max risk)     │
                       └──────────────────────┬───────────────────────┘
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │       Dynamic Rupee (₹) Position Sizing      │
                       │       - Stop-loss & Fibonacci TP targets     │
                       │       - Risk-to-Reward ratio ≥ 1.5           │
                       │       - 2% equity risk sizing                │
                       └──────────────────────┬───────────────────────┘
                                              │ SIGNAL_GENERATED
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │    Execution Engine & Reporting Audit Trail  │
                       │    - Structured JSON trade tickets           │
                       │    - Real-time FastAPI WebSocket broadcast   │
                       │    - React + Vite Dashboard Monitoring       │
                       └──────────────────────────────────────────────┘
```

---

## ⚙️ Core Capabilities

- **Indian Stock Market (NSE / ₹ INR):** Calibrated for Indian market price structures, tick sizes, point values, and brokerage rules.
- **5-Minute Candle Intervals:** Operates on intraday 5-minute candles (`timeframe_seconds: 300`) with zero lookahead bias.
- **Multi-Model AI Stacking:**
  - **PyTorch LSTM Forecaster:** Captures sequential price memory across rolling 40-candle windows.
  - **XGBoost & LightGBM:** Captures non-linear indicator interactions and decision trees.
  - **Logistic Regression Meta-Model:** Stacks predictions using out-of-fold calibration for high-conviction directional bias ($P(\text{UP}) \ge 0.65$ or $P(\text{DOWN}) \ge 0.65$).
- **Smart Money Concepts (SMC):** Tracks institutional footprint via Break of Structure (BOS), Change of Character (CHoCH), and supply/demand imbalances.
- **Strict Confluence Enforcement:** No trade is triggered unless technical analysis, SMC structure, and machine learning agree simultaneously ($\ge 3$ confluences).
- **Asymmetric Risk-to-Reward:** Automatically enforces a minimum 1.5:1 Risk-to-Reward ratio, ATR-cushioned stop-losses, and Fibonacci extension targets (1.272 / 1.618).
- **Full Test Coverage:** Comprehensive test suite with **67/67 passing tests** verifying feeds, indicators, models, and execution sanity.

---

## 📁 Repository Structure

```
Trading/
├── config.yaml                 # Master configuration (timeframe, symbols, risk, capital)
├── main.py                     # Primary trading engine entrypoint (Live / Dry-Run / Backtest)
├── server.py                   # FastAPI backend server with WebSocket streaming
├── pytest.ini                  # PyTest test runner configuration
├── conftest.py                 # PyTest test fixtures and environment hooks
├── requirements.txt            # Python dependencies (PyTorch, Pandas, Scikit-Learn, etc.)
│
├── core/                       # Low-latency asynchronous infrastructure
│   ├── event_bus.py            # Typed asynchronous pub/sub event bus
│   ├── candle_buffer.py        # Real-time tick aggregator & rolling 500-candle buffer
│   ├── events.py               # Event schemas (Tick, CandleClosed, SignalGenerated, etc.)
│   └── types.py                # Strongly-typed dataclasses and enums
│
├── indicators/                 # Technical indicator calculations (<10ms parallel)
│   ├── smc_engine.py           # Smart Money Concepts (BOS, CHoCH, Order Blocks)
│   ├── alpha_trend.py          # Adaptive ATR dynamic trendline
│   ├── alpha_trend_plus.py     # Volume surge & volatility confirmation
│   ├── hidden_zone.py          # Institutional supply & demand zones
│   ├── fibonacci.py            # Dynamic Fibonacci retracements & extensions
│   ├── harmonic.py             # XABCD Harmonic reversal pattern detector
│   ├── holt_winters_rsi.py     # Exponentially-smoothed predictive RSI
│   └── predictive_macd.py      # Momentum & histogram trajectory
│
├── forecast/                   # Machine learning forecasting pipeline
│   ├── model.py                # PyTorch LSTM & Transformer network definitions
│   ├── train_lstm_v2.py        # Walk-forward ensemble training script
│   ├── data_pipeline.py        # Feature extraction and z-score normalization
│   ├── feature_builder.py      # Real-time streaming 40-candle tensor generator
│   ├── inference.py            # Low-latency JIT-compiled inference (<5ms)
│   └── evaluator.py            # Accuracy, Sharpe ratio, and profit factor evaluations
│
├── decision/                   # Trade decision & risk governance
│   ├── signal_generator.py     # Signal fusion engine (BUY / SELL / HOLD)
│   ├── scorer.py               # Multi-factor composite weight scorer
│   ├── confluence.py           # Confluence matrix (minimum 3 systems required)
│   ├── risk_filter.py          # Guardrails (spread, ATR spike, daily loss limit)
│   └── position_sizing.py      # Dynamic rupee-based stop-loss, take-profit, & lots
│
├── execution/                  # Execution & reporting
│   ├── engine.py               # Order router and trade lifecycle manager
│   └── reporter.py             # Session P&L and downloadable performance reports
│
├── feed/                       # Market data feed adapters
│   ├── websocket_feed.py       # Live WebSocket client (TwelveData / Finnhub)
│   └── college_feed.py         # Dual-mode adapter (Testing vs Competition API)
│
├── scripts/                    # Automation utilities
│   ├── download_nifty.py       # 5-minute historical data downloader (Yahoo Finance)
│   └── setup_env.ps1           # Environment setup & package installation script
│
├── tests/                      # Automated test suite (67 passed tests)
│   ├── test_all.py             # Unit tests for event bus, buffer, indicators, and risk
│   └── test_nse_dataset.py     # NSE 5m dataset, schema, latency, and model tests
│
└── ui/                         # React + Vite frontend dashboard
    ├── src/
    │   ├── pages/Dashboard.jsx # Real-time charts, signal confidence, and order log
    │   ├── pages/Setup.jsx     # Session configuration & API key injector
    │   └── pages/Report.jsx    # Post-session performance breakdown & export
    └── package.json
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.10+** (Python 3.13 recommended)
- **Node.js 18+** (for the UI dashboard)
- Git

### 2. Environment Setup
Create a virtual environment and install all dependencies:

```powershell
# Using the automated PowerShell setup script:
.\scripts\setup_env.ps1

# Or manually:
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

### 3. Download 5-Minute NSE Data
Fetch the latest 60 days of 5-minute candle data for NIFTY 50 and constituents:
```powershell
.\venv\Scripts\python.exe scripts\download_nifty.py
```
This generates:
- `data/nifty50_5m.csv` (~4,500 candles)
- `data/reliance_5m.csv`
- `data/tcs_5m.csv`

### 4. Train the Ensemble Models
Train the LSTM, XGBoost, LightGBM, and Stacking Meta-Model on the 5-minute data:
```powershell
.\venv\Scripts\python.exe -m forecast.train_lstm_v2
```
Trained artifacts are exported to `models/`:
- `models/lstm_eurusd_v2.pt`
- `models/xgb_eurusd_v2.json`
- `models/lgbm_eurusd_v2.txt`
- `models/meta_model_v2.pkl`
- `models/scaler_v2.json`
- `models/feature_schema_v2.json`

### 5. Run the Test Suite
Validate that all components and models pass verification:
```powershell
.\venv\Scripts\pytest tests/ -v
```

---

## 💻 Operating Modes

### Mode 1: Dry-Run Simulation (Synthetic Stream)
Simulates realistic 5-minute NSE price action around ₹24,000 without requiring live internet feeds:
```powershell
.\venv\Scripts\python.exe main.py --dry-run
```

### Mode 2: Historical Backtesting
Runs backtest simulations on historical 5-minute candles:
```powershell
.\venv\Scripts\python.exe main.py --backtest
```

### Mode 3: Live Trading Engine
Connects to the configured live feed (TwelveData / Finnhub / College API) and streams real-time trading signals:
```powershell
.\venv\Scripts\python.exe main.py
```

### Mode 4: Full-Stack Web Dashboard (FastAPI + React)
Start the backend server:
```powershell
.\venv\Scripts\python.exe server.py
```
*(Backend runs at `http://localhost:8000`)*

In another terminal, start the frontend UI:
```powershell
cd ui
npm install
npm run dev
```
*(Dashboard opens at `http://localhost:5173`)*

---

## 🛡️ Risk Management Parameters

All parameters are centrally configurable in [`config.yaml`](file:///c:/Users/KRRISH/.vscode/codes/Techverse/Trading/config.yaml):

| Parameter | Value | Description |
|---|---|---|
| `symbol` | `NIFTY50` | Primary traded instrument |
| `timeframe_seconds` | `300` | 5-minute candle interval |
| `initial_equity` | `₹5,000` | Starting virtual portfolio equity |
| `max_spread_pips` | `₹5.00` | Maximum acceptable market spread |
| `daily_loss_limit_pips` | `₹2,000` | Emergency stop if cumulative loss exceeds threshold |
| `max_open_positions` | `1` | Prevents overexposure in fast markets |
| `min_risk_reward` | `1.5` | Minimum risk-reward ratio to allow trade entry |
| `risk_per_trade_pct` | `2.0%` | Capital percentage risked per position |
| `commission_per_lot` | `₹20.00` | Flat brokerage cost per executed order |

---

## 📄 License
This project is developed for educational, quantitative research, and competition purposes.
