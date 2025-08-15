# Memecoin Trading Bot - Detailed Trading Plan & Strategy

## Overview
This document provides a comprehensive breakdown of the memecoin trading bot's strategy, implementation, and operational details based on the current codebase analysis. The bot follows a smart money "alpha wallet" tracking strategy to identify and trade promising memecoin launches on Solana using sophisticated Bayesian wallet scoring and adaptive risk management.

## Core Trading Philosophy
The bot operates on the principle of **following smart money wallets** rather than trying to compete with high-frequency trading bots. It identifies wallets that consistently buy successful tokens early and follows their trading patterns using a conservative, alpha-following approach optimized for consistency over explosive gains.

**Key Principles:**
- **Alpha Following, Not Sniping**: Enter 30-60 seconds after alpha wallets, not competing for first position
- **Consistency Over Moonshots**: Target 15-25% gains with controlled downside
- **Bayesian Confidence**: Use statistical models to weight wallet reliability
- **Adaptive Risk**: Automatically reduce activity during drawdowns

## System Architecture

### 1. Main Components

#### Trading Bot (`main.py`)
- **Purpose**: Core trading engine that monitors new token launches and executes trades
- **Initial Capital**: $500 (configurable)
- **Mode**: Paper trading (simulation) with pessimistic fee structure
- **Max Trades Per Day**: 20-30 (adaptive based on performance)
- **Trade Size**: 5% of capital per trade (fixed position sizing)
- **Hold Strategy**: Dynamic exit with intelligent trailing stops

#### Alpha Discovery System
Sophisticated dual-system approach for finding and maintaining profitable wallets:

1. **Alpha Discovery V2** (`src/discovery/alpha_discovery_v2.py`)
   - Analyzes 2-hour historical window for proven performance
   - Multi-tier success criteria: High (2x), Medium (1.5x), Low (1.2x)
   - Finds wallets appearing on 2+ successful tokens
   - Validates with comprehensive metrics and wash trading detection

2. **Wallet Rotation Manager** (`src/core/wallet_rotation_manager.py`)
   - Runs every 2 hours automatically
   - **Smart Retention**: Keeps high-performing wallets (top 6-20)
   - **Gradual Rotation**: Replaces underperformers with new discoveries
   - **PumpPortal Sync**: Updates real-time subscriptions during rotation

#### Bayesian Wallet Scorer (`src/core/wallet_scorer.py`)
- **Smart Confidence System**: Handles fresh vs established wallets
- **Fresh Wallets**: Start with 65% confidence (above 55% threshold)
- **Developing Wallets** (1-3 trades): Use max(Bayesian, 55%) floor
- **Established Wallets** (4+ trades): Pure Bayesian scoring
- **Prior**: Beta(3,5) representing ~37.5% baseline win rate

## Data Sources & APIs

### 1. PumpPortal (Primary Real-time Source)
- **Purpose**: Real-time wallet activity monitoring
- **Endpoint**: `wss://pumpportal.fun/api/data`
- **Focus**: Direct wallet buy/sell notifications
- **Advantages**: Lower latency, direct alpha signals
- **Rotation Sync**: Updates subscriptions during wallet rotation

### 2. Bitquery (Secondary/Discovery)
- **Purpose**: Historical analysis for alpha discovery
- **Endpoint**: `wss://streaming.bitquery.io/eap`
- **Focus**: Pump.fun DEX trades for discovery
- **Limitations**: Returns ~3-4 minutes of data per query
- **Token Rotation**: Supports multiple API tokens for failover

### 3. Moralis (Token Metadata & Pricing)
- **Purpose**: Token metadata, liquidity, pricing, holder data
- **Endpoint**: `https://solana-gateway.moralis.io`
- **Rate Limit**: 5 requests/second (conservative)
- **Key Rotation**: Supports 25+ API keys with automatic rotation
- **Caching**: 
  - Metadata: 2 hours
  - Liquidity: 10 minutes (600s)
  - Price: 8 seconds
  - Swaps: 30 seconds (alpha), 10 seconds (alpha early abort)

### 4. Solana RPC
- **Default**: Public endpoint (`https://api.mainnet-beta.solana.com`)
- **Purpose**: Backup data source and transaction broadcasting (when live trading implemented)

## Trading Decision Logic

### Token Detection Flow
1. **Real-time Alpha Signal** (via PumpPortal WebSocket)
   - Monitor watched wallet buy transactions
   - Group by mint address with time-based deduplication
   - Calculate confidence scores using Bayesian weights

2. **Alpha Analysis** (`src/core/wallet_tracker.py`)
   - **Weighted Voting**: Each wallet contributes based on confidence score
   - **Smart Confidence Check**: Ensure at least one wallet ≥55% confidence
   - **Tier Mapping**: Calculate investment multipliers based on wallet performance
   - **Deduplication**: 15-minute cache prevents repeated signals for same token

3. **Safety Validation** (`src/core/safety_checks.py`)
   - Rug pull score calculation (0-100, <70 is safe)
   - Liquidity check: Minimum $5,000 USD
   - Trader diversity: Minimum unique wallet count
   - Honeypot detection: Verify sellability

4. **Risk Management Gating** (`src/core/risk_manager.py`)
   - **Adaptive Thresholds**: Stricter requirements during drawdowns
   - **Trade Limits**: Daily caps based on performance
   - **Cooldowns**: Automatic pauses after consecutive losses
   - **Time Spacing**: Minimum intervals between trades

### Bayesian Wallet Confidence System

The core innovation of the system is the smart confidence scoring that handles both fresh and established wallets:

```python
def get_smart_confidence_score(wallet):
    if wallet not in wallet_stats:
        # Fresh wallet from discovery - trust the discovery process
        return 0.65  # 65% confidence (above 55% threshold)
    
    trade_count = wallet_stats[wallet].wins + wallet_stats[wallet].losses
    
    if trade_count <= 3:
        # Developing wallet - blend Bayesian with floor
        bayesian_score = get_wallet_score(wallet)
        return max(bayesian_score, 0.55)  # Floor at 55%
    else:
        # Established wallet - pure Bayesian scoring
        return get_wallet_score(wallet)  # No floor - must earn confidence
```

**Key Benefits:**
- **Cold Start Solution**: New wallets can trade immediately after discovery
- **Statistical Rigor**: Established wallets must prove performance over time
- **Automatic Transition**: Seamless shift from discovery trust to earned trust

### Entry Criteria

**Required Conditions:**
```
Base Requirements:
- At least 2-6 alpha wallets (dynamic based on risk level)
- Weighted alpha score ≥ 2.0-4.0 (dynamic based on performance)
- At least one wallet with ≥55% confidence (smart confidence)
- Safety score < 70 (rug pull protection)
- Minimum liquidity ≥ $5,000 USD
- Time window: Within 300 seconds of alpha detection

Risk-Adjusted Requirements (Adaptive):
Normal Mode: 3 wallets, weight ≥2.5
Cautious Mode (-2% daily): 4 wallets, weight ≥3.0  
Conservative Mode (-4% daily): 5 wallets, weight ≥3.5
Defensive Mode (-6%+ daily): 6 wallets, weight ≥4.0
```

**Investment Multiplier Calculation:**
Based on highest-tier wallet in the signal:
- **Multiple high-tier wallets**: 1.4x base investment
- **Single high-tier wallet**: 1.2x base investment  
- **Mixed/medium wallets**: 1.0x base investment
- **Low confidence signal**: 0.6x base investment

## Position Management

### Entry Execution

**Base Trade Amount Calculation:**
```
base_amount = current_capital × 0.05 (5% fixed position size)
final_amount = base_amount × investment_multiplier
```

**Pessimistic Fee Structure (Paper Trading):**
- **DEX Fee**: 0.30% (30 bps) - Pump.fun standard
- **Network Fee**: ~$0.28 (0.002 SOL priority + 0.000005 SOL base)
- **Buy Slippage**: 0.75% (75 bps) - Conservative following estimate
- **Sell Slippage**: 1.00% (100 bps) - Higher due to market impact
- **Max Slippage Gate**: 1.50% (150 bps) - Reject if exceeded

*Note: Fees are intentionally pessimistic to ensure live trading profitability*

### Dynamic Exit Strategy

The bot uses an intelligent, multi-phase exit system optimized for alpha-following:

**Phase 1: Dynamic TP1 Sizing**
```python
# TP1 at 1.25x (25% gain) with timing-based sizing
if time_to_tp1 < 60 seconds:
    sell_percentage = 12%  # Very fast move - let it run
elif time_to_tp1 < 180 seconds:
    sell_percentage = 20%  # Fast move - moderate taking
else:
    sell_percentage = 30%  # Normal move - standard taking
```

**Phase 2: Intelligent Trailing Stops**
- **High Gain Trail** (≥60% peak): Trail at 82% of peak price
- **Fast Gain Trail** (≥30% in <2min): Trail at 85% of peak price
- **Standard Trail** (≥15% gain): Trail at 85% of peak price

**Phase 3: Break-Even Protection**
- **Armed at +8% gain**: Provides 60-second break-even protection
- **1% Buffer**: Allows for minor volatility (exit at 101% of entry)
- **Automatic Transition**: Switches to trailing after 60 seconds

**Stop Loss Strategy:**
- **Hard Stop**: 92% of entry price (8% loss tolerance)
- **Time Stop**: 900 seconds (15 minutes) only if <5% gain
- **Emergency Stop**: 1800 seconds (30 minutes) regardless of gain

### Position Tracking
```python
@dataclass
class Position:
    mint: str
    entry_price: float
    amount: float  # Tokens held
    sol_invested: float
    entry_time: datetime
    tp_price: float  # 1.25x entry
    sl_price: float  # 0.92x entry
    peak_price: float  # Tracks highest price seen
    tp1_hit_time: Optional[datetime]  # When TP1 was triggered
    tp1_percentage_sold: float  # How much sold at TP1
    break_even_armed: bool  # Protection status
    trailing_stop_active: bool  # Trailing status
    high_gain_peak: float  # Highest gain % achieved
```

## Adaptive Risk Management

### Dynamic Parameter Adjustment

The system automatically adjusts trading parameters based on daily P&L performance:

**Risk Levels:**
```yaml
Normal (0% to -2% daily):
  min_wallets: 3
  min_weight: 2.5
  max_daily_trades: 20
  min_time_between: 120 seconds

Cautious (-2% to -4% daily):
  min_wallets: 4
  min_weight: 3.0
  max_daily_trades: 15
  min_time_between: 180 seconds

Conservative (-4% to -6% daily):
  min_wallets: 5
  min_weight: 3.5
  max_daily_trades: 10
  min_time_between: 300 seconds

Defensive (-6%+ daily):
  min_wallets: 6
  min_weight: 4.0
  max_daily_trades: 5
  min_time_between: 600 seconds
```

### Circuit Breakers
- **3 Consecutive Losses**: 30-minute cooldown
- **5 Consecutive Losses**: 60-minute cooldown
- **Daily Trade Limits**: Hard caps prevent overtrading
- **Minimum Time Spacing**: Prevents emotional rapid-fire trading

### Capital Preservation
- **Fixed Position Size**: 5% of capital (never changes to avoid fee impact)
- **No Leverage**: Pure spot trading only
- **P&L Tracking**: Real-time equity updates with persistent storage

## Alpha Wallet Discovery & Management

### Discovery Process (Every 2 Hours)

**Phase 1: Historical Analysis**
1. **Time Window**: Analyze tokens from 2 hours to 30 minutes ago
2. **Paginated Retrieval**: Get up to 60,000 trades via BitQuery
3. **Token Metrics**: Calculate swap counts, unique traders, price progression
4. **Success Classification**: Multi-tier success based on performance multipliers

**Phase 2: Alpha Identification**
1. **Early Buyer Detection**: Find wallets buying within first 10 minutes
2. **Position Size Filtering**: $50-$5,000 range (avoid noise and whales)
3. **Overlap Analysis**: Identify wallets appearing on 2+ successful tokens
4. **Wash Trading Detection**: Flag suspicious coordination patterns

**Phase 3: Scoring & Ranking**
```python
# Multi-tier qualification system
Tier 1 (Premium): 2+ high success tokens (2x gain)
Tier 2 (Good): 3+ medium success tokens (1.5x gain)  
Tier 3 (Emerging): 4+ low success tokens (1.2x gain)

# Final score calculation
tier_base_score = tier_multiplier × 100
performance_bonus = avg_performance × 25
success_bonus = success_rate × 40
volume_bonus = token_count × 10
risk_adjusted_bonus = skill_vs_capital × 15
recency_decay = exp(-time_diff / 12_hours)

final_score = (tier_base_score + bonuses) × recency_decay
```

### Wallet Rotation Strategy

**Smart Retention Logic:**
- **High Performers**: Keep top 6-20 wallets indefinitely
- **Medium Performers**: Keep for at least one rotation cycle
- **New Discoveries**: Add up to 94 wallets per rotation
- **Underperformers**: Replace with fresh discoveries

**Rotation Triggers:**
- **Scheduled**: Every 2 hours automatically
- **Performance-Based**: Remove consistently failing wallets
- **Discovery-Based**: Integrate newly discovered alpha wallets

**PumpPortal Integration:**
- **Subscription Updates**: Automatically update WebSocket subscriptions
- **Seamless Transition**: No data gaps during rotation
- **Error Handling**: Fallback to BitQuery if PumpPortal fails

### Wallet Performance Tracking

**Bayesian Statistics:**
- **Prior**: Beta(3,5) representing 37.5% baseline success rate
- **Updates**: Add wins/losses to posterior distribution
- **Confidence Intervals**: Calculate uncertainty bounds
- **Recency Weighting**: Recent performance weighted higher

**Performance Metrics:**
```python
@dataclass
class WalletStats:
    wins: int = 0
    losses: int = 0
    recent_trades: deque(maxlen=100)  # Rolling window
    last_activity: float = 0  # Unix timestamp
    total_volume: float = 0  # USD volume
    trades_count: int = 0
    
    # Calculated metrics
    win_rate: float  # wins / (wins + losses)
    confidence_score: float  # Bayesian posterior mean
    recency_boost: float  # 1.0-1.3x based on recent activity
    size_factor: float  # 1.0-1.5x based on average trade size
```

## Performance Tracking & Analytics

### P&L Management (`src/utils/pnl_store.py`)

**Persistent Storage**: JSON file (`data/pnl_state.json`) with atomic updates

**Tracked Metrics:**
```json
{
  "current_equity": 551.55,
  "total_trades": 88,
  "winning_trades": 42,
  "total_realized_pnl": 51.55,
  "daily_pnl": -5.09,
  "win_rate": 47.7,
  "active_positions": {
    "mint_address": {
      "symbol": "TOKEN",
      "entry_price": 0.00001234,
      "amount": 1000000,
      "timestamp": "2024-01-15T16:00:00Z",
      "confidence_score": 65,
      "wallet_tiers": {"wallet1": "tier_1"}
    }
  },
  "trade_history": [...] // Last 100 trades
}
```

### Trade Recording
```json
{
  "timestamp": "2024-01-15T16:00:00Z",
  "action": "BUY",
  "symbol": "MEME",
  "mint": "abc123...",
  "amount": 1000000,
  "price": 0.00001234,
  "usd_value": 15.00,
  "realized_pnl": null,
  "paper_mode": true,
  "confidence_score": 65,
  "investment_multiplier": 1.2,
  "wallet_tiers": {
    "wallet1": "tier_1",
    "wallet2": "tier_2"
  },
  "metadata": {
    "safety_score": 45,
    "liquidity_usd": 8500,
    "alpha_latency": 2.3
  }
}
```

### Key Performance Indicators
1. **Daily P&L**: Drives adaptive risk management
2. **Win Rate**: Target >55% for sustainable growth
3. **Average Hold Time**: ~5-10 minutes per position
4. **Alpha Latency**: <3 seconds from signal to execution
5. **Position Fill Rate**: % of alpha signals that result in trades
6. **Wallet Effectiveness**: Individual wallet performance tracking

## Monitoring & Notifications

### Discord Integration (`src/utils/discord_notifier.py`)
- **Trade Alerts**: Real-time buy/sell notifications with confidence scores
- **Daily Summaries**: P&L, trade count, win rate, top performing wallets
- **Error Notifications**: API failures, rate limits, system errors
- **5-Minute Heartbeats**: Activity summary with key metrics

**Trade Notification Format:**
```
🟢 BUY: MEME
Token: 5NmzY6Rz...
Quantity: 516,330
Price: $0.00002828
USD Value: $15.00
Confidence: 65%
Equity: 💰 $509.63
📝 Paper Trading
```

### Logging System
- **Main Log**: `logs/trading.log` - All trading activity
- **Alpha Discovery Log**: `logs/alpha_discovery.log` - Discovery process
- **Structured Logging**: JSON format for easy parsing
- **Log Rotation**: Daily rotation with compression
- **Debug Levels**: Configurable verbosity

### Health Monitoring
- **Heartbeat**: Every 30 seconds with system status
- **API Health**: Track response times and error rates
- **Memory Usage**: Monitor for memory leaks
- **WebSocket Status**: Connection health for real-time feeds
- **Trade Execution Success Rate**: End-to-end performance

## Configuration Management

### Core Settings (`config/config.yml`)
```yaml
# Trading Parameters
threshold_alpha_buys: 3-6              # Dynamic based on risk level
min_liquidity_usd: 5000                # Minimum token liquidity
time_window_sec: 300                   # Alpha detection window (5min)
max_trade_pct: 0.05                    # Fixed 5% position size
tp_multiplier: 1.25                    # Take profit at 25% gain
stop_loss_pct: 0.92                    # Stop at 8% loss
max_hold_seconds: 900                  # 15 minute time limit

# Alpha Wallet Management  
alpha_weight_min: 2.0                  # Minimum weighted threshold
require_one_wallet_pge_55: true        # Confidence requirement
watched_wallets: [...]                 # 50-100 wallet addresses

# Risk Management
risk_management:
  enabled: true
  position_size_pct: 0.05              # Fixed position sizing
  levels:
    normal:
      min_wallets: 3
      min_weight: 2.5
      max_daily_trades: 20
      min_time_between: 120
    # ... other risk levels

# Bayesian Scoring
beta_prior_alpha: 3                    # Beta distribution alpha
beta_prior_beta: 5                     # Beta distribution beta

# API Optimization
api_optimization:
  alpha_early_abort_buy_sell_ratio: 1.0
  alpha_early_abort_min_buyers: 10
  discovery_early_window_minutes: 10
  liquidity_ttl: 600                   # 10 minute cache
  price_ttl: 8                         # 8 second cache
  swaps_alpha_ttl: 10                  # 10 second alpha cache

# Paper Trading Fees (Pessimistic)
paper_trading:
  buy_slippage_bps: 75                 # 0.75% slippage
  sell_slippage_bps: 100               # 1.00% slippage  
  fee_bps: 30                          # 0.30% DEX fee
  max_slippage_bps: 150                # 1.50% rejection threshold

# Operational Settings
trading_mode: simulation               # "simulation" or "live"
initial_capital: 500.0
max_concurrent_positions: 5
min_time_between_trades: 30           # Base cooldown
realtime_source: pumpportal           # "pumpportal" or "bitquery"
```

## Current Implementation Status

### ✅ Fully Implemented Features
- **PumpPortal Real-time Monitoring**: WebSocket integration with wallet subscriptions
- **Bayesian Wallet Scoring**: Smart confidence system with fresh/established handling
- **Adaptive Risk Management**: Dynamic parameter adjustment based on performance
- **Intelligent Exit Strategy**: Dynamic TP1 sizing and trailing stops
- **Wallet Rotation Management**: Smart retention with PumpPortal sync
- **Multi-tier Alpha Discovery**: Comprehensive historical analysis
- **Safety Scoring System**: Rug pull protection with multiple checks
- **Paper Trading**: Pessimistic fee structure for realistic backtesting
- **P&L Tracking**: Persistent storage with detailed trade history
- **Discord Notifications**: Real-time alerts and performance summaries
- **API Management**: Multi-key rotation with intelligent caching

### 🔄 Partially Implemented
- **Live Trading Infrastructure**: Framework ready, needs wallet integration
- **Advanced Analytics**: Basic metrics tracked, could expand reporting
- **Wash Trading Detection**: Implemented but could be enhanced

### ❌ Not Implemented  
- **MEV Protection**: Not needed for alpha-following strategy
- **Multi-chain Support**: Solana-focused approach
- **Machine Learning Predictions**: Statistical approach preferred
- **Automated Rebalancing**: Fixed position sizing by design

## Risk Warnings & Limitations

### Trading Risks
1. **Paper Trading Only**: Current implementation simulates trading with pessimistic fees
2. **Alpha Decay**: Wallet effectiveness may decrease as strategies become known
3. **Market Conditions**: Strategy optimized for bull market memecoin activity
4. **Latency Dependency**: Not suitable for high-frequency or MEV-competitive strategies

### Technical Risks  
1. **API Dependencies**: Heavy reliance on PumpPortal, Moralis, and BitQuery availability
2. **WebSocket Reliability**: Real-time feeds can disconnect and require reconnection
3. **Rate Limiting**: Conservative limits may miss opportunities during high activity
4. **Data Quality**: Dependent on accurate price feeds and transaction data

### Operational Risks
1. **Configuration Drift**: Manual config changes can impact performance
2. **Wallet Discovery Quality**: New alpha wallets may not maintain historical performance
3. **Market Regime Changes**: Strategy may need adaptation for different market conditions
4. **Capital Scale Limitations**: Strategy designed for small to medium capital sizes

## Optimization Opportunities

### Near-term Improvements
1. **Enhanced Discovery Metrics**: Add funding source analysis for wallet quality
2. **Dynamic Time Windows**: Adjust alpha detection windows based on market activity
3. **Advanced Caching**: Implement smarter cache invalidation strategies
4. **Performance Analytics**: Add more detailed attribution analysis

### Medium-term Enhancements
1. **Multi-timeframe Analysis**: Combine short-term and medium-term alpha signals
2. **Sentiment Integration**: Factor in social signals and community metrics
3. **Cross-DEX Monitoring**: Expand beyond Pump.fun to other Solana DEXes
4. **Portfolio Correlation**: Track correlation between held positions

### Long-term Evolution
1. **Live Trading Integration**: Full implementation with proper wallet management
2. **Strategy Diversification**: Add complementary trading strategies
3. **Risk Modeling**: Advanced risk metrics and stress testing
4. **Community Features**: Social trading and strategy sharing capabilities

## Summary

The bot implements a sophisticated alpha-following strategy that prioritizes consistency and capital preservation over explosive gains. The system uses advanced Bayesian statistics to weight wallet reliability, adaptive risk management to adjust to market conditions, and intelligent exit strategies optimized for following rather than leading market movements.

**Key Innovations:**
- **Smart Confidence System**: Seamlessly handles fresh and established wallets
- **Adaptive Risk Management**: Automatically becomes more selective during drawdowns  
- **Dynamic Exit Strategy**: Timing-based profit taking with intelligent trailing
- **Sophisticated Discovery**: Multi-tier historical analysis with wash trading detection

The strategy is designed for systematic, repeatable performance with minimal supervision, automatically discovering new alpha wallets while preserving the best historical performers. The conservative approach, pessimistic fee modeling, and multiple safety layers make it well-suited for consistent growth with controlled downside risk.