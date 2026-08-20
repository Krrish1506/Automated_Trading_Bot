# 🤖 AI Sniper Trading Bot — Competition Platform

An autonomous, multi-asset trading engine built for a 3-hour live trading competition. The system uses a triple-gate sniper strategy (Fibonacci Golden Zone + ML Forecast + Candlestick Pattern Detection) to execute high-conviction trades on ₹10,00,000 starting capital.

---

## 🏗️ Architecture

```
Live Price Feed (Finnhub / College API)
         ↓
   Candle Buffer (OHLCV)
         ↓
  Indicator Registry (RSI, MACD, EMA, ATR, Fibonacci)
         ↓
  ML Forecast Engine (LSTM/XGBoost Inference)
         ↓
  Signal Generator — Triple Sniper Gate:
     ✅ Gate 1: Fibonacci Golden Zone (0.382–0.618)
     ✅ Gate 2: ML Confidence > 60%
     ✅ Gate 3: Candlestick Pattern (Hammer / W-Pattern / Engulfing / Pin Bar)
         ↓
  Execution Engine
     → Auto Stop-Loss / Take-Profit
     → Session Timer (auto-stops at end)
     → Emergency Drawdown Guard (>15%)
         ↓
  Trade Reporter → CSV / JSON / HTML Reports
         ↓
  FastAPI Backend (server.py) → React UI Dashboard
```

---

## ⚙️ Features

- **Sniper Strategy** — Only enters trades when ALL 3 gates align simultaneously
- **₹10,00,000 Capital** — 2% risk per trade (₹20,000 max risk)
- **15% Max Drawdown Guard** — Emergency stop if equity drops below ₹8,50,000
- **Session Timer** — Configurable from the UI (60–240 mins); positions force-closed 30 mins before end
- **Dual Feed Mode** — Test with Finnhub/TwelveData; switch to College API on competition day
- **Live Dashboard** — Real-time price, equity curve, signal confidence, and trade log
- **Auto Reports** — CSV, JSON, and HTML trade report generated at session end

---

## 📁 Project Structure

```
Trading/
├── server.py               # FastAPI backend (API + WebSocket server)
├── main.py                 # Standalone trading loop entry point
├── config.yaml             # All configuration (capital, risk, session, etc.)
├── requirements.txt        # Python dependencies
│
├── core/                   # Event bus, types, candle buffer
├── feed/                   # Price feed adapters (Finnhub, TwelveData, College API)
├── indicators/             # Technical indicators + candlestick pattern engine
├── decision/               # Signal generator, scorer, confluence detector
├── execution/              # Execution engine, trade reporter
├── forecast/               # ML inference engine
├── monitoring/             # Logging setup
│
└── ui/                     # React + Vite frontend dashboard
    ├── src/
    │   ├── App.jsx
    │   ├── pages/
    │   │   ├── Setup.jsx       # Session config (API key, duration, provider)
    │   │   ├── Dashboard.jsx   # Live trading dashboard
    │   │   └── Report.jsx      # End-of-session report + downloads
    │   └── index.css
    └── package.json
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Install Python Dependencies
```bash
cd "d:\Devam\Microsoft VS Code\Codes\Trading"
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies
```bash
cd "d:\Devam\Microsoft VS Code\Codes\Trading\ui"
npm install
```

---

### ▶️ Start the Backend (FastAPI Server)
```bash
cd "d:\Devam\Microsoft VS Code\Codes\Trading"
python server.py
```
Backend runs at: `http://localhost:8000`

---

### ▶️ Start the Frontend (React Dashboard)
```bash
cd "d:\Devam\Microsoft VS Code\Codes\Trading\ui"
npm run dev
```
Dashboard opens at: `http://localhost:5173`

---

## 🖥️ Using the Dashboard

1. Open `http://localhost:5173` in your browser
2. **Setup Page:**
   - Select **Finnhub** (for pre-competition testing) or **College API** (competition day)
   - Paste your API Key
   - If using College API, paste the WebSocket URL they provide
   - Set your session duration using the slider
   - Click **START SESSION**
3. **Live Dashboard** — Monitor real-time price, AI signal, equity curve, and trade log
4. **Session End** — Full report is auto-generated; download as HTML or CSV

---

## 📋 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/session/start` | Start trading session |
| `POST` | `/session/stop` | Stop session manually |
| `GET` | `/session/status` | Get live stats + trade log |
| `GET` | `/report/download?format=html` | Download HTML report |
| `GET` | `/report/download?format=csv` | Download CSV report |
| `WS` | `/ws/live` | Real-time price/signal/status stream |

---

## ⚠️ Competition Day Checklist

- [ ] College provides API Key → paste into Setup page
- [ ] College provides WebSocket URL → paste into Setup page
- [ ] Select **College API** mode in the dropdown
- [ ] Set session duration to match competition time
- [ ] Click START — the bot handles everything automatically

---

## 📄 License

Built for educational competition purposes.
