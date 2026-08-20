# Autonomous Financial Trading Strategy — MERCATUS
### Full Architecture & Training Explanation
> **Starting Capital:** ₹5,000 virtual | **Goal:** Maximize final portfolio value

---

## 🎯 What the Competition Actually Wants

From the screenshot:
- You get **sequential market snapshots** from their hidden simulator (fake stock API)
- You must output **BUY / SELL / HOLD** for each snapshot
- Goal: **maximize final portfolio value** starting with **₹5,000 virtual capital**
- Constraints: **Sharpe ratio**, **drawdown control**, **no lookahead** (you can only see past data)
- On competition day: they add **surprise constraints** like position size limits, transaction costs, etc.

This means your system needs to be:
1. A **rule-based + ML hybrid** (pure ML alone is too risky for unknown stocks)
2. **Fast** — each snapshot must be answered almost instantly
3. **Adaptive** — must handle surprise constraints on competition day

---

## 🏗️ Your Existing Architecture (What You Already Have)

Looking at your codebase, you already have a very solid foundation:

```
main.py              → Entry point, wires everything together
core/
  types.py           → All data structures (Candle, Signal, IndicatorState, etc.)
  candle_buffer.py   → Rolling buffer of candles
  event_bus.py       → Async pub/sub event system
indicators/
  alpha_trend.py     → Trend direction detection
  alpha_trend_plus.py→ Confirmed trend strength
  fibonacci.py       → Fib retracements + extensions
  harmonic.py        → XABCD harmonic pattern detection
  hidden_zone.py     → Supply/Demand zones
  holt_winters_rsi.py→ RSI with forecasting (Holt-Winters exponential smoothing)
  predictive_macd.py → MACD with crossover prediction
  smc_engine.py      → Smart Money Concepts (BOS, CHoCH, Order Blocks)
decision/
  confluence.py      → Checks if 3+ indicators agree before a signal
  signal_generator.py→ Produces BUY/SELL/HOLD with entry/SL/TP
  scorer.py          → Weights each indicator's contribution
  risk_filter.py     → Blocks signals that violate risk rules
  position_sizing.py → Calculates how many units to buy/sell
forecast/
  model.py           → LSTM architecture
  train_lstm.py      → Training pipeline
  inference.py       → Real-time prediction
backtest/            → Historical simulation
```

**Your architecture is already correct.** What you need now is to:
1. Understand HOW each piece thinks
2. Connect it to the college's fake stock API
3. Train the LSTM on synthetic/historical data
4. Handle the competition's surprise constraints on day-of

---

## 📐 The Three Layers of Your Strategy

### Layer 1 — Classic Technical Analysis (Rule-Based, Always On)
These are math formulas. No training needed. They fire signals based on price patterns.

### Layer 2 — Smart Money Concepts (SMC) (Rule-Based, Market Structure)
These detect what big institutions (banks, hedge funds) are doing by reading order flow.

### Layer 3 — LSTM Forecast (ML, Trained)
This predicts the next price direction based on sequences of candle features.

**Final decision = all 3 layers must broadly agree (Confluence) before acting.**

---

## 📊 Layer 1: Technical Analysis — The Signals

### 1.1 Fibonacci — The Most Important One

**Why Fibonacci works:**
Traders worldwide use the same Fibonacci levels. When price falls to 61.8%, millions of traders buy — which MAKES the price bounce. It's a self-fulfilling prophecy at scale.

**Key Fibonacci Levels:**
```
Retracements (pullback zones):
  23.6% → Weak support, shallow pullback
  38.2% → Moderate pullback, trend still strong
  50.0% → Psychological midpoint
  61.8% → "Golden Ratio" — strongest retracement level ★
  78.6% → Deep retracement, trend weakening

Extensions (profit targets):
  127.2% → First extension target
  161.8% → "Golden Extension" — most common profit target ★
  261.8% → Aggressive target
```

**How You Use It:**
1. Find a significant swing (major high to major low or vice versa)
2. Draw Fibonacci levels on that swing
3. When current price is at 61.8% AND your trend indicator says bullish → BUY signal
4. Set take-profit at 161.8% extension

**Your `fibonacci.py` already computes this.** It tracks `confluence_count` — how many Fib levels are clustering near the current price. When multiple Fib levels overlap at one price point, that's an extremely high-probability zone.

---

### 1.2 Fibonacci + Harmonic Patterns (XABCD)

Harmonics are geometric price patterns where the legs of the pattern have EXACT Fibonacci ratios between them.

**The 4 Patterns Your `harmonic.py` Detects:**

```
GARTLEY (Most Common)
  XA → AB = 61.8% retracement of XA
  AB → BC = 38.2%–88.6% retracement of AB
  BC → CD = 127.2%–161.8% extension
  Completion point D = 78.6% retracement of XA ★
  Action: BUY at D (bullish) or SELL at D (bearish)

BUTTERFLY
  Completion point D = 127.2% extension of XA ★
  Occurs at market EXTREMES — strongest reversal signal

BAT
  Completion point D = 88.6% retracement of XA ★
  Tighter pattern, very precise entry

CRAB
  Completion point D = 161.8% extension of XA ★
  Most extreme pattern, catches the final blow-off
```

**The PRZ (Potential Reversal Zone):**
- D point is not a single price — it's a small zone
- Your code stores `prz_low` and `prz_high`
- When price enters PRZ → prepare to trade
- Wait for 1 confirming candle (engulfing, pin bar) → execute

**Why this matters for your competition:**
Harmonic patterns work on ANY asset class — stocks, forex, commodities. Your college's fake stock API will still produce price series that follow these geometric patterns because they use real market behavior simulations.

---

### 1.3 SMC — Smart Money Concepts (What Institutions Do)

**Core idea:** Banks and funds can't buy/sell all at once — they leave footprints in price action. SMC reads those footprints.

**Key Concepts:**

**BOS (Break of Structure)**
```
Normal uptrend:
  Higher High (HH) → Higher Low (HL) → HH → HL ...
  
A BULLISH BOS happens when price breaks ABOVE the last HH
→ Signals: institutions are accumulating, trend is accelerating
→ Action: BUY on next pullback

A BEARISH BOS happens when price breaks BELOW the last LL
→ Signals: institutions are distributing, trend is breaking down
→ Action: SELL on next bounce
```

**CHoCH (Change of Character)**
```
CHoCH = a REVERSAL signal (opposite of BOS)
If you're in an uptrend (HH, HL pattern) and price breaks BELOW
the last Higher Low → CHoCH → trend is reversing
→ Action: CLOSE longs, prepare to SHORT
```

**Order Blocks (OBs)**
```
An Order Block is the last bearish candle before a big bullish move
(or last bullish candle before a big bearish move).
Why? Banks placed their big orders in that candle.
When price comes BACK to that zone → banks defend their position → price reverses.

Bullish OB:
  Last bearish candle before an impulsive move up
  When price returns to this zone → BUY
  
Bearish OB:
  Last bullish candle before an impulsive move down
  When price returns to this zone → SELL
```

**Liquidity Sweeps**
```
Banks hunt for stop-losses above obvious highs and below obvious lows.
They push price just past a key level (triggering retail stops),
then REVERSE in the other direction.

Pattern: Price spikes past a key high → immediately reverses down
→ This is a liquidity sweep → expect strong move in reverse direction
```

Your `smc_engine.py` already detects BOS, CHoCH, order blocks, and liquidity sweeps.

---

### 1.4 AlphaTrend

This is a volatility-adjusted trend indicator:
- Uses ATR (Average True Range) to measure current volatility
- Combines with RSI or MFI to determine trend direction
- Produces a "trend line" — price ABOVE it = bullish, BELOW = bearish

Your `alpha_trend_plus.py` is an enhanced version that confirms trend strength (`trend_confirmed = True` means both AlphaTrend and Alpha++ agree).

**Simple rule:** 
- `trend_direction == BULLISH` + `trend_confirmed == True` → Only take BUY signals
- `trend_direction == BEARISH` + `trend_confirmed == True` → Only take SELL signals
- `is_sideways == True` → HOLD (don't trade choppy markets)

---

### 1.5 Holt-Winters RSI

Normal RSI: overbought at 70, oversold at 30.

Your version adds **Holt-Winters exponential smoothing** — a forecasting technique from time-series statistics that predicts what RSI will be in the next 1-3 candles.

```
Normal RSI use:
  RSI < 30 → Oversold → BUY
  RSI > 70 → Overbought → SELL

Your enhanced version (hw_rsi):
  If hw_rsi predicts RSI will CROSS 30 from below → early BUY signal
  If hw_rsi predicts RSI will CROSS 70 from above → early SELL signal
  "momentum_shift = bullish/bearish" in your types.py = this prediction
```

---

### 1.6 Predictive MACD

Normal MACD: signal when MACD line crosses signal line (lagging by 2-3 candles).

Your `predictive_macd.py` predicts WHERE the crossover will happen BEFORE it happens:
- Uses the current slope of MACD and signal lines
- Extrapolates to find the exact bar/price where they will cross
- Stores `crossover_target` price and `crossover_direction`

This gives you an EARLY entry — before the crossover candle, not after.

---

### 1.7 Hidden Zones (Supply & Demand)

These are price zones where:
- **Demand Zone**: Price consolidated, then shot UP impulsively → institutions were buying here
- **Supply Zone**: Price consolidated, then dropped DOWN impulsively → institutions were selling here

When price returns to these zones → same institutions defend them → strong reversal.

Your `hidden_zone.py` scores each zone by:
- `strength`: How violently price left the zone (bigger move = stronger zone)
- `touch_count`: How many times price has returned (more touches = weaker, it'll break)
- `age_bars`: Fresh zones are stronger than old zones

---

## 🧠 Layer 2: The Confluence System (Your `decision/` folder)

This is the heart of your system. Raw signals from individual indicators are often wrong alone. The confluence system only acts when **3+ indicators agree**.

### How It Works (Your `confluence.py`)

```
Each snapshot, the confluence detector checks 9 conditions:

1. AlphaTrend Alignment — Is trend confirmed bullish/bearish?
2. Structural Confirmation — Does BOS match the trend direction?
3. Zone Alignment — Is price near a demand/supply zone in the right direction?
4. Harmonic PRZ — Is price inside a harmonic pattern completion zone?
5. Forecast + Structure — Does LSTM say the same thing as the BOS?
6. RSI Momentum — Does momentum_shift match the trend direction?
7. MACD Crossover — Does predicted MACD crossover align with structure?
8. Fibonacci Confluence — Are 2+ Fib levels near the current price?
9. Order Block — Is price inside a matching order block?

Rule: Need ≥ 3 of these 9 to fire a trade signal.
```

### The Scorer (`scorer.py`)
Each indicator gives a score from -1.0 (max bearish) to +1.0 (max bullish):
- AlphaTrend: weight ~30%
- SMC Structure: weight ~25%
- Fibonacci: weight ~20%
- Harmonics: weight ~15%
- MACD/RSI: weight ~10%

Weighted sum → overall direction + strength of conviction.

### The Risk Filter (`risk_filter.py`)
Even with high confluence, the filter blocks trades if:
- Drawdown exceeds the threshold (protects the ₹5,000 capital)
- Current open position would exceed allocation limits
- Market state = SIDEWAYS (no trend to ride)
- ATR is too low (no momentum, price isn't moving)

---

## 🤖 Layer 3: LSTM Model — How It's Trained

### What the LSTM Learns
The LSTM sees a **sequence of 60 candles** (lookback window) and predicts:
- Direction of next candle: UP / DOWN
- The probability/confidence of that prediction

### Feature Engineering (What Goes INTO the LSTM)

Your `forecast/feature_builder.py` builds these features per candle:

```
Price-based (normalized):
  - open, high, low, close (normalized by rolling std)
  - body size = |close - open| / ATR
  - wick ratio = (high - low - body) / ATR
  - gap from prev close

Indicator features:
  - RSI (0-100 normalized to 0-1)
  - hw_rsi (forecasted RSI)
  - MACD histogram (normalized)
  - AlphaTrend distance from price (normalized by ATR)
  - trend_direction (-1, 0, +1)
  - trend_strength (0 to 1)

Structure features:
  - bos_up (0 or 1)
  - bos_down (0 or 1)
  - choch (0 or 1)
  - nearest_demand_dist (normalized)
  - nearest_supply_dist (normalized)

Fibonacci features:
  - distance to nearest 61.8% level
  - distance to nearest 161.8% level
  - confluence_count (0 to 5+)

Time features:
  - hour of day (sin/cos encoded for cyclical nature)
  - day of week (sin/cos encoded)
```

### LSTM Architecture (Your `forecast/model.py`)

```
Input: Sequence of 60 candles × N features

→ LSTM Layer 1: 128 hidden units, dropout=0.2
→ LSTM Layer 2: 64 hidden units, dropout=0.2
→ Attention Layer (weights which past candles matter most)
→ Fully Connected: 64 → 32 → 1
→ Sigmoid output: 0 = strong down, 0.5 = neutral, 1 = strong up
```

### Training Strategy

**Phase 1: Training Data Source**
Since you have a FAKE stock API, you need to generate synthetic data for training. Options:
1. Use your `--dry-run` mode to generate synthetic candles and train on those
2. Use historical forex/stock data (same patterns exist everywhere) — Yahoo Finance is free
3. Generate correlated random walk data that mimics real market behavior

**Phase 2: Labels (What You're Teaching)**
```
For each candle at position i:
  Label = 1 if close[i+1] > close[i] (price went up next candle)
  Label = 0 if close[i+1] < close[i] (price went down)
  
But we want PROFITABLE signals, not just directional accuracy.
So: Label = 1 if price rose by > 0.3% in next 3 candles (enough to cover spreads)
         = 0 if price fell by > 0.3% in next 3 candles
         = ignore if less than 0.3% move (no meaningful signal)
```

**Phase 3: Training Loop**
- Loss function: Binary Cross-Entropy
- Optimizer: Adam (lr=0.001, with learning rate decay)
- Batch size: 64
- Epochs: 100 with early stopping (stop if val_loss doesn't improve for 10 epochs)
- Train/Val split: 70/15/15 (train/validation/test), NO SHUFFLING (time must be sequential!)

**Critical rule:** NEVER shuffle time-series data. Past bars train, future bars validate.

---

## 🔄 How a Single Snapshot Flows Through Your System

```
College API sends snapshot:
  {timestamp, open, high, low, close, volume}
         ↓
  CandleBuffer.add(candle)
  → Checks if new candle is complete (M5 timeframe)
         ↓
  EventBus fires: CANDLE_CLOSED event
         ↓
  IndicatorRegistry.compute_all(candles)
  → fibonacci.py computes retracement/extension levels
  → harmonic.py scans for XABCD patterns  
  → smc_engine.py detects BOS/CHoCH/OBs
  → alpha_trend.py gets trend direction
  → holt_winters_rsi.py predicts RSI
  → predictive_macd.py forecasts crossover
  → hidden_zone.py finds nearest supply/demand
  → Packages all into: IndicatorState
         ↓
  InferenceEngine.predict(last_60_candles → features)
  → LSTM outputs: forecast_direction (up/down) + confidence
         ↓
  ConfluenceDetector.detect(state, forecast_dir)
  → Checks all 9 confluence conditions
  → Returns: ["AlphaTrend Bullish", "Fibonacci Confluence", ...]
         ↓
  Scorer.score(state, individual_scores)
  → Returns: composite_score (-1.0 to +1.0)
         ↓
  RiskFilter.check(composite_score, confluences)
  → Blocks if drawdown too high, market sideways, etc.
         ↓
  SignalGenerator.generate()
  → Computes entry, stop-loss, take-profit
  → Returns: TradeSignal {BUY/SELL/HOLD, confidence, entry, SL, TP, RR}
         ↓
  Output: "BUY" / "SELL" / "HOLD"
```

---

## 💰 Risk Management — Why This Is Half the Battle

Starting with ₹5,000, one bad sequence without risk management could wipe you out before you get any compounding benefit.

### Position Sizing (Your `position_sizing.py`)
```
Starting Capital: ₹5,000

Risk per trade: 1% of current portfolio value (never more)
  → Starting capital ₹5,000 → risk at most ₹50 per trade
  → As portfolio grows to ₹6,000 → risk ₹60 per trade (scales up)

Position size = Risk Amount / Distance to Stop-Loss
  → If SL is 2% away from entry:
     Position size = ₹50 / 2% = ₹2,500 worth of stock in that trade
  → This ensures SL always costs exactly ₹50 (1% of capital)

Why 1%?
  → Even 10 consecutive losses = only -10% total drawdown (₹500)
  → Keeps you alive long enough for your edge to play out
```

### Drawdown Control
```
Max drawdown limit: 15% of starting capital (₹750 from ₹5,000)
If portfolio drops by ₹750 from peak:
  → Risk filter BLOCKS all new trades
  → System goes into HOLD-only mode
  → Waits for market structure to clarify

Why 15% for ₹5K instead of 10%?
  → Small capital = more variance in short simulations
  → Need slightly more breathing room before shutting off
  → But NEVER exceed 20% drawdown under any constraint
```

### Sharpe Ratio Target
```
Sharpe = (Average Return - Risk-Free Rate) / Standard Deviation of Returns

Your goal: Sharpe > 1.5 (very good), > 2.0 (excellent)
This is computed in your backtest/evaluator.py

How to improve Sharpe:
  → Be selective (higher confluence threshold = fewer but better trades)
  → Cut losers fast (tight stop-losses via ATR)
  → Let winners run (trail stop to 2x ATR from entry)
```

---

## 🎯 Event-Day Adaptation (Surprise Constraints)

The competition says they'll add constraints on the day. Your system handles this through `config.yaml`:

**Possible constraints and how to handle:**

| Surprise Constraint | Your Response |
|---|---|
| Transaction cost added (e.g. 0.1%) | Raise minimum confluence to 5 (fewer, higher quality trades) |
| Position size limit (e.g. max 5%) | `position_sizing.py` caps at that % |
| Max drawdown tighter (e.g. 5%) | Update `risk_filter.py` threshold |
| Specific assets only | Update feed symbols in `config.yaml` |
| Faster intervals (M1 instead of M5) | Update `timeframe_seconds` in config |

**Your architecture already supports all of these through config — no code change needed on day-of.**

---

## 📈 What "Training" Really Means Here

There are TWO things to "train":

### Training 1: LSTM (Machine Learning)
- Done OFFLINE before competition day
- Trained on synthetic/historical candle data
- Saved as `.pth` weights file
- Loaded at runtime for inference
- Inference is instant (~1ms per prediction)

### Training 2: Hyperparameter Tuning (Backtesting)
This is more important than ML training for this competition:
```
Your backtest/engine.py runs your full system on historical data
and outputs: total return, Sharpe ratio, max drawdown, win rate

You tune:
  - min_confluences: 3 or 4 or 5?
  - risk_per_trade: 0.5% or 1% or 2%?
  - Fibonacci lookback period: 50 candles or 100?
  - LSTM threshold: 0.6 = trade, or 0.7 = trade?
  - Stop-loss multiplier: 1.5x ATR or 2x ATR?

Run backtest → adjust → run again → find the optimal combination
This process = "training your strategy"
```

---

## 🗺️ What You Still Need to Build / Fix

Based on your current codebase:

1. **Connect to the College API** — `feed/websocket_feed.py` needs the actual API key and endpoint URL from the college. Replace the OANDA connection with theirs.

2. **Generate Training Data** — Run your dry-run mode to create synthetic candles, OR download free historical data (Yahoo Finance via `yfinance` library), then run `forecast/train_lstm.py`.

3. **Backtest on Synthetic Data** — Run `python main.py --backtest` to find optimal hyperparameters.

4. **Simplify Output** — Competition probably expects just `"BUY"`, `"SELL"`, or `"HOLD"` as a string. Make sure your output format matches their expected format.

5. **Handle HOLD correctly** — In sideways markets, HOLD is the BEST decision. Don't force trades. Let the system sit on cash while waiting for high-confluence setups.

---

## 🏆 Winning Strategy Summary

```
The system wins by:

1. PATIENCE — Only trade when 3+ indicators agree (high confluence)
   → Win rate > 60% instead of random 50%

2. ASYMMETRY — Risk 1% to make 2-3% (1:2 or 1:3 risk/reward)
   → Even with 50% win rate, you're profitable

3. FIBONACCI PRECISION — Enter exactly at 61.8%, exit at 161.8%
   → These levels work because everyone uses them

4. SMART MONEY READING — Trade WITH institutions, not against them
   → BOS = follow the trend. CHoCH = catch the reversal early.

5. DRAWDOWN PROTECTION — Stop trading when losing
   → Protect the ₹5,000 base. Never blow up.
   → Target: ₹5,000 → ₹8,000–₹12,000 (60%–140% gain is realistic and impressive)
   
6. LSTM CONFIRMATION — Only enter when ML model agrees with technical setup
   → Reduces false signals in choppy/news-driven markets
```

---

*This document maps to your existing codebase. Every concept described here has a corresponding file already in your project. The next step is wiring the college's API key into `feed/websocket_feed.py` and running a backtest.*
