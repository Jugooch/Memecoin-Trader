# Memecoin Trading Bot - Detailed Trading Plan & Strategy

## Overview
This document provides a comprehensive breakdown of the memecoin trading bot's strategy, implementation, and operational details based on the current codebase analysis. The bot follows a smart money "alpha wallet" tracking strategy to identify and trade promising memecoin launches on Solana.

## Core Trading Philosophy
The bot operates on the principle of **following smart money wallets** rather than trying to compete with high-frequency trading bots. It identifies wallets that consistently buy successful tokens early and follows their trading patterns.

## System Architecture

### 1. Main Components

#### Trading Bot (`main.py`)
- **Purpose**: Core trading engine that monitors new token launches and executes trades
- **Initial Capital**: $500 (configurable)
- **Mode**: Paper trading (simulation) by default, live trading optional
- **Max Trades Per Day**: 75
- **Trade Size**: 5% of capital per trade (base)
- **Hold Strategy**: Tiered exit with time limits

#### Alpha Discovery System
Two complementary systems for finding profitable wallets:

1. **Alpha Discovery V2** (`src/discovery/alpha_discovery_v2.py`)
   - One-off analysis tool
   - Analyzes recent 2-hour window
   - Finds wallets appearing on 2+ successful tokens
   - Success tiers: High (2x), Medium (1.5x), Low (1.2x)

2. **Alpha Accumulator** (`src/discovery/alpha_accumulator.py`)
   - Continuous discovery system
   - Runs every 2 minutes
   - Builds historical overlap data
   - Better for production use

## Data Sources & APIs

### 1. Bitquery (Token Launch Monitoring)
- **Purpose**: Real-time token launch detection via GraphQL subscriptions
- **Endpoint**: `wss://streaming.bitquery.io/eap`
- **Focus**: Pump.fun DEX trades
- **Limitations**: Returns ~3-4 minutes of data per query
- **Token Rotation**: Supports multiple API tokens for failover

### 2. Moralis (Token Metadata & Pricing)
- **Purpose**: Token metadata, liquidity, pricing, holder data
- **Endpoint**: `https://solana-gateway.moralis.io`
- **Rate Limit**: 5 requests/second (conservative)
- **Key Rotation**: Supports 8 API keys with automatic rotation
- **Caching**: 
  - Metadata: 2 hours
  - Liquidity: 10 minutes
  - Price: 8 seconds
  - Swaps: 30 seconds

### 3. Solana RPC
- **Default**: Public endpoint (`https://api.mainnet-beta.solana.com`)
- **QuickNode**: Optional (not implemented yet, we are too broke) for better performance

## Trading Decision Logic

### Token Detection Flow
1. **New Token Launch Detection** (via Bitquery WebSocket)
   - Monitor Pump.fun program trades
   - Group trades by mint address
   - Calculate launch timestamp

2. **Initial Filtering**
   - Minimum liquidity: $5,000 USD
   - Spam filter: Exclude test/fake/scam tokens
   - Activity check: Minimum swap count and unique traders

3. **Alpha Wallet Check**
   - Check if watched wallets bought within 2-minute window
   - Threshold: 3 alpha wallets minimum
   - Calculate confidence score (0-100)
   - Determine investment multiplier based on wallet tiers

4. **Safety Validation**
   - Rug pull score calculation (0-100, <70 is safe)
   - Liquidity check (<$1k = +30 score)
   - Trader diversity check (<20 unique = +20 score)
   - Honeypot detection (no sells = +50 score)

### Alpha Wallet Tiering System

**Wallet Performance Tiers:**
- **S-Tier**: Win rate ≥70%, Avg profit ≥100% → 2.0x investment
- **A-Tier**: Win rate ≥60%, Avg profit ≥50% → 1.4x investment
- **B-Tier**: Win rate ≥50%, Avg profit ≥20% → 1.0x investment
- **C-Tier**: Below B-tier thresholds → 0.6x investment

### Entry Criteria
```
Required Conditions:
- At least 3 alpha wallets bought the token
- Confidence score ≥ 50
- Rug score < 70
- Liquidity ≥ $1,000
- Distinct wallets ≥ 3
- Time window: Within 120 seconds of detection
```

## Position Management

### Entry Execution

**Base Trade Amount Calculation:**
```
base_amount = current_capital × 0.05 (5%)
final_amount = base_amount × investment_multiplier
```

**Fee Structure (Paper Trading):**
- DEX Fee: 0.30% (30 bps)
- Network Fee: ~$0.28 (0.002 SOL priority + 0.000005 SOL base)
- Buy Slippage: 0.50% (50 bps)
- Sell Slippage: 0.75% (75 bps)

### Exit Strategy (Tiered)

**Take Profit Levels:**
1. **TP1** (1.5x): Sell 30% of position
2. **TP2** (1.75x): Sell another 30% of position
3. **Trailing Stop**: After 50% gain, exit if price drops to 85% of peak

**Stop Loss:**
- Hard stop: 85% of entry price (15% loss)
- Time stop: Exit after 300 seconds (5 minutes)
- Break-even stop: Armed after +10%, exits at entry price

**Partial Exit Logic:**
```python
remaining_position = 1.0
if price >= tp1_price and not tp1_executed:
    sell 30% → remaining = 70%
if price >= tp2_price and not tp2_executed:
    sell 30% → remaining = 40%
if stop_loss or time_stop:
    sell 100% of remaining
```

## Risk Management

### Position Limits
- **Max trades per day**: 75
- **Max concurrent positions**: 3
- **Min time between trades**: 120 seconds
- **Cool-down after stop loss**: 180 seconds

### Capital Management
- **Initial capital**: $500
- **Max trade size**: 5% of capital
- **Max with S-tier signal**: 5% of capital (5% × 1.0x)
- **Paper mode**: Track P&L in JSON file

## Alpha Wallet Discovery Process

### Continuous Discovery (Production)
1. **Frequency**: Every 2 minutes
2. **Data Collection**: Latest 3-4 minutes of trades from Bitquery
3. **Token Ranking**: Sort by activity (trade count)
4. **Early Buyer Detection**: Find wallets buying in first 5 minutes
5. **Overlap Analysis**: Identify wallets on 2+ successful tokens
6. **Auto-update**: Updates config.yml with top 100 wallets

### Discovery Metrics
```
Success Token Criteria:
- Price increase ≥ 1.2x (low tier)
- OR: 20+ unique traders AND 40+ swaps (activity fallback)

Wallet Scoring:
- Base score from tier (S=300, A=200, B=150, C=100)
- Performance bonus: avg_multiplier × 25
- Success bonus: success_rate × 40
- Volume bonus: token_count × 10
- Recency decay: 12-hour half-life
```

## Performance Tracking

### P&L Management (`src/utils/pnl_store.py`)
- **Storage**: JSON file (`data/pnl_state.json`)
- **Metrics Tracked**:
  - Current capital
  - Realized/Unrealized P&L
  - Win rate
  - Trade history (last 100)
  - Active positions
  - Daily P&L

### Trade Recording
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "action": "BUY",
  "symbol": "MEME",
  "mint": "abc123...",
  "amount": 1000000,
  "price": 0.00000123,
  "usd_value": 5.00,
  "realized_pnl": null,
  "paper_mode": true
}
```

## Monitoring & Notifications

### Discord Integration
- **Webhook**: Sends trade notifications and summaries
- **Trade Alerts**: Buy/Sell with confidence scores
- **Error Notifications**: API failures, rate limits
- **5-min Summaries**: Activity and P&L updates

### Logging System
- **Trading Log**: `logs/trading.log`
- **Alpha Discovery Log**: `logs/alpha_discovery.log`
- **Heartbeat**: Every 30 seconds
- **Summary**: Every 5 minutes with metrics

## Operational Metrics

### Key Performance Indicators
1. **Token Processing Rate**: ~50-200 tokens/5min
2. **Alpha Check Rate**: ~10-30 checks/5min
3. **API Efficiency**: Deduplication saves ~20-40% calls
4. **Execution Latency**: Target <3 seconds total
5. **Active Wallet Ratio**: Target >25/100 active

### Resource Optimization
- **Cache Strategy**: Multi-tier TTL based on data type
- **Request Coalescing**: Prevents duplicate in-flight requests
- **Wallet Deduplication**: 15-minute cache for alpha detections
- **Memory Management**: 
  - Trade history: Last 100 trades
  - Position cache: Active only
  - Token processed cache: 10-minute TTL

## Configuration Parameters

### Core Settings (`config/config.yml`)
```yaml
# Trading Parameters
threshold_alpha_buys: 3        # Min alpha wallets required
min_liquidity_usd: 5000        # Min token liquidity
time_window_sec: 120           # Alpha detection window
max_trade_pct: 0.05            # 5% of capital per trade
tp_multiplier: 1.5             # Take profit at 1.5x
stop_loss_pct: 0.85            # Stop at 15% loss

# Operational Settings
trading_mode: simulation       # or "live"
initial_capital: 500.0
max_trades_per_day: 75
min_time_between_trades: 30    # seconds

# Alpha Wallet Management
watched_wallets: [...]         # List of 50-100 addresses
inactive_threshold_hours: 6    # Mark inactive after 6h
```

## Current Implementation Status

### ✅ Implemented Features
- WebSocket token monitoring
- Alpha wallet tracking with tiers
- Safety scoring system
- Paper trading with realistic fees
- P&L tracking
- Multi-key API rotation
- Request caching and coalescing
- Discord notifications
- Tiered exit strategy
- Wallet performance tracking

### 🔄 Partially Implemented
- Live trading (QuickNode integration ready but not tested)
- Automated wallet recycling (triggers but needs testing)

### ❌ Not Implemented
- MEV protection
- Multi-chain support
- Advanced ML predictions
- Liquidity pool monitoring
- Automated wallet quality scoring updates

## Risk Warnings
1. **Paper Trading**: Current implementation is paper trading only
2. **API Dependencies**: Relies heavily on Moralis/Bitquery availability
3. **Latency**: Not suitable for high-frequency trading
4. **Alpha Decay**: Wallet effectiveness may decrease over time
5. **Rug Pull Risk**: Safety checks are heuristic-based, not guaranteed

## Optimization Opportunities
1. **Reduce API Calls**: 
   - Implement more aggressive caching
   - Batch Moralis requests where possible

2. **Improve Alpha Discovery**:
   - Add machine learning for wallet scoring
   - Track wallet fund sources

3. **Enhanced Safety**:
   - Check LP lock status
   - Verify mint authority revocation
   - Monitor dev wallet movements

4. **Performance**:
   - Implement WebSocket for price updates
   - Use dedicated RPC for lower latency
   - Parallel processing for alpha checks

## Summary
The bot implements a conservative, smart-money-following strategy optimized for small capital growth. It prioritizes safety over speed, using multiple validation layers before entering positions. The tiered exit strategy and strict risk management aim to preserve capital while capturing upside from successful memecoin launches. The system is designed to run continuously with minimal supervision, automatically discovering and adapting to new alpha wallets over time.