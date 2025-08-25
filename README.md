# 🚀 Memecoin Trading Bot

Sophisticated Solana memecoin trading bot that follows proven "alpha wallets" (smart money) using advanced Bayesian statistics and adaptive risk management. Monitors pump.fun token launches in real-time and executes trades when multiple high-performing wallets show buying activity.

## ✨ Key Features (Recently Enhanced - Dec 2024)

- **🧠 Bayesian Wallet Scoring**: Smart confidence system handling fresh vs established wallets
- **📊 Adaptive Risk Management**: Dynamic parameter adjustment based on daily P&L performance  
- **⏱️ Temporal Clustering**: NEW - Ensures alpha buys are coordinated within 90 seconds
- **📈 Price Extension Guard**: NEW - Blocks trades at price peaks (90th percentile)
- **🎯 Entry Timing Control**: NEW - Blocks late entries after momentum dissipates
- **🔥 Moonshot Detection**: NEW - Skip profit-taking on exceptional 60%+ runs
- **🛡️ Volatility-Based Stops**: NEW - Dynamic stop losses based on token volatility
- **⚡ Scratch Exit System**: NEW - Early exit at -2% to prevent -8% stops
- **🎯 PumpPortal Integration**: Real-time wallet activity monitoring with WebSocket subscriptions
- **🔄 Intelligent Wallet Rotation**: Smart retention logic keeping best performers while discovering new alpha
- **💰 Paper Trading Mode**: Pessimistic fee simulation ensuring live trading profitability
- **📱 Enhanced Logging**: NEW - Comprehensive trade decision tracking
- **🔔 Discord Integration**: Real-time notifications with confidence scores and performance summaries

### 🎯 Performance Targets (Post-Update)
- **Win Rate**: 35-45% (improved from 20%)
- **Average Loss**: -2.5% (improved from -6%)
- **Moonshot Capture**: Occasional 100%+ gains with optimized exits

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the configuration template
cp config/config.yml.example config.yml

# Edit config.yml with your API credentials
nano config.yml
```

Add your API keys:
```yaml
# PumpPortal API (Primary real-time source)
pumpportal:
  api_key: "your_pumpportal_api_key"
  ws_endpoint: "wss://pumpportal.fun/api/data"

# Bitquery (Discovery and backup)
bitquery_tokens:
  - "your_bitquery_token_1"
  - "your_bitquery_token_2"  # Multiple tokens for rotation

# Multiple Moralis keys for 24/7 operation
moralis_keys:
  - your_first_moralis_key_here
  - your_second_moralis_key_here  # Recommended for rotation
  - your_third_moralis_key_here   # For high availability

# Discord notifications (optional)
notifications:
  discord_webhook_url: "your_discord_webhook_url"
  enabled: true
```

### 3. Start the Bot

```bash
python start_bot.py
```

**Note:** Alpha wallet discovery and rotation are built into the bot. No separate processes needed!

### 4. Optional: Manual Alpha Discovery (Testing)

```bash
# Run one-off analysis for testing
python -m src.discovery.alpha_discovery_v2
```

## Configuration System

The bot uses a sophisticated configuration system with smart path resolution and validation.

### Trading Modes
Choose your execution mode in `config.yml`:

```yaml
trading_mode: "simulation"  # Options:
# "simulation" = Paper trading with pessimistic fees (recommended)
# "live" = Real trading (requires proper wallet setup)
```

### Key Configuration Options:

**Core Trading Parameters:**
- `initial_capital: 500.0` - Starting capital amount
- `max_trade_pct: 0.05` - Fixed 5% position sizing per trade
- `threshold_alpha_buys: 3` - Minimum alpha wallet buys to trigger trade
- `tp_multiplier: 1.25` - Take profit at 25% gain with dynamic sizing
- `stop_loss_pct: 0.92` - Stop loss at 8% loss (realistic for volatility)
- `max_hold_seconds: 900` - 15-minute maximum hold time

**Alpha Wallet Management:**
- `alpha_weight_min: 2.0` - Minimum weighted threshold for signals
- `require_one_wallet_pge_55: true` - Require at least one wallet with ≥55% confidence
- `watched_wallets: [...]` - 50-100 alpha wallet addresses (auto-managed)

**Adaptive Risk Management:**
```yaml
risk_management:
  enabled: true
  levels:
    normal:      # 0% to -2% daily P&L
      min_wallets: 3
      min_weight: 2.5
      max_daily_trades: 20
    cautious:    # -2% to -4% daily P&L
      min_wallets: 4
      min_weight: 3.0
      max_daily_trades: 15
    conservative: # -4% to -6% daily P&L
      min_wallets: 5
      min_weight: 3.5
      max_daily_trades: 10
    defensive:   # -6%+ daily P&L
      min_wallets: 6
      min_weight: 4.0
      max_daily_trades: 5
```

## API Setup Required

### PumpPortal (Primary Real-time Source)
1. Sign up at [PumpPortal.fun](https://pumpportal.fun)
2. Get API key for WebSocket access
3. Used for real-time wallet buy/sell notifications
4. **Advantages**: Lower latency, direct alpha signals

### Moralis (Token Data & Analytics)
1. Create account at [Moralis.io](https://moralis.io)
2. Get Solana API key (multiple recommended)
3. Used for token metadata, liquidity, pricing, and safety checks

### Bitquery (Discovery & Backup)
1. Sign up at [Bitquery.io](https://bitquery.io)
2. Get API key for GraphQL access
3. Used for historical analysis and alpha discovery
4. Fallback source if PumpPortal unavailable

### QuickNode (Optional - Only for Live Trading)
1. **NOT REQUIRED** for paper trading mode
2. Only needed for automated live trading execution
3. Sign up at [QuickNode.com](https://quicknode.com) when ready for live trading

## Why This Alpha-Following Strategy Works

### Intelligence Over Speed
Instead of competing on latency (expensive), we compete on intelligence:
- **Smart Money Following**: Wait for proven alpha wallets to validate tokens
- **Bayesian Confidence**: Statistical models weight wallet reliability over time
- **Risk-Adjusted Entries**: Position sizing based on signal quality and recent performance
- **Conservative Exits**: Consistent 15-25% gains with controlled downside

### Adaptive Risk Management
The system automatically becomes more selective during drawdowns:
- **Dynamic Thresholds**: Stricter requirements when losing money
- **Circuit Breakers**: Automatic cooldowns after consecutive losses
- **Performance-Based**: Parameters adjust based on daily P&L
- **Capital Preservation**: Fixed 5% position sizing prevents overexposure

## Trading Strategy Overview

The bot implements a sophisticated alpha-following strategy with Bayesian wallet scoring:

### 1. Real-time Alpha Detection (PumpPortal)
- Monitor 50-100 alpha wallets via WebSocket subscriptions
- Detect buy transactions in real-time (sub-second latency)
- Group signals by token with time-based deduplication

### 2. Bayesian Confidence Scoring
```python
# Smart confidence system handles all wallet types
Fresh Wallets (from discovery): 65% confidence
Developing Wallets (1-3 trades): max(Bayesian, 55%)
Established Wallets (4+ trades): Pure Bayesian scoring
```

### 3. Weighted Alpha Analysis
- Each wallet contributes weight based on confidence score
- Require minimum weighted threshold (2.0-4.0, adaptive)
- Investment multiplier based on highest-tier wallets in signal

### 4. Safety Validation
- Rug pull score calculation (<70 required)
- Minimum $5,000 USD liquidity requirement
- Wash trading detection for wallet coordination
- Honeypot detection and sellability verification

### 5. Dynamic Position Entry
- **Base Size**: 5% of current capital (fixed)
- **Multipliers**: 0.6x-1.4x based on signal quality
- **Realistic Fees**: 0.75% buy + 1.0% sell slippage, 0.3% DEX fee
- **Max Slippage Gate**: Reject if >1.5%

### 6. Intelligent Exit Management
- **Dynamic TP1**: 12-30% of position based on timing to 25% gain
- **Trailing Stops**: Multiple tiers based on peak gain percentages
- **Break-Even Protection**: Armed at +8%, provides 60-second safety
- **Time Limits**: 15 minutes max hold, earlier if <5% gain

## 🎯 Alpha Wallet Discovery & Management

### Sophisticated Discovery System
The bot includes advanced alpha wallet discovery with wash trading detection:

#### Continuous Discovery (Every 2 Hours)
- **Historical Analysis**: 2-hour window with up to 60,000 trades
- **Multi-tier Success**: High (2x), Medium (1.5x), Low (1.2x) performance tiers
- **Position Filtering**: $50-$5,000 range (avoid noise and whales)
- **Wash Trading Detection**: Flag suspicious coordination patterns

#### Smart Wallet Rotation
- **Retention Logic**: Keep top 6-20 performing wallets
- **Gradual Replacement**: Replace underperformers with new discoveries
- **PumpPortal Sync**: Update WebSocket subscriptions seamlessly
- **Performance Tracking**: Bayesian statistics with recency weighting

### Wallet Confidence Evolution
```python
# Wallet progression through confidence tiers
Discovery → 65% confidence (immediate trading ability)
    ↓
1-3 trades → max(Bayesian, 55%) (learning phase)
    ↓
4+ trades → Pure Bayesian (earned confidence)
```

### Alpha Trading Logic
1. PumpPortal detects wallet buy transactions
2. System calculates weighted confidence score
3. Checks adaptive risk management thresholds
4. Validates safety and liquidity requirements
5. Executes trade with appropriate position sizing

## Risk Management & Safety

### Adaptive Parameter Adjustment
The system automatically adjusts based on daily P&L:
- **Normal** (0% to -2%): Standard thresholds, 20 trades/day
- **Cautious** (-2% to -4%): Stricter requirements, 15 trades/day
- **Conservative** (-4% to -6%): High selectivity, 10 trades/day
- **Defensive** (-6%+): Maximum selectivity, 5 trades/day

### Circuit Breakers & Cooldowns
- **3 Consecutive Losses**: 30-minute cooldown
- **5 Consecutive Losses**: 60-minute cooldown
- **Trade Spacing**: Minimum 60-600 seconds between trades (adaptive)
- **Daily Limits**: Hard caps prevent overtrading

### Capital Preservation
- **Fixed Position Size**: Always 5% of capital (no leverage)
- **Stop Loss**: 8% maximum loss per trade
- **Diversification**: Maximum 5 concurrent positions
- **Emergency Stops**: Multiple fallback exit conditions

## Paper Trading Mode

The bot uses pessimistic fee modeling to ensure live trading profitability:

### Realistic Fee Structure
- **DEX Fee**: 0.30% (Pump.fun standard)
- **Network Fee**: ~$0.28 per transaction
- **Buy Slippage**: 0.75% (conservative alpha-following estimate)
- **Sell Slippage**: 1.00% (higher due to market impact)

### Benefits
1. **Conservative Backtesting**: If profitable in paper mode, live trading should be profitable
2. **Risk-Free Testing**: Validate strategy without capital risk
3. **Performance Validation**: Build confidence before live deployment
4. **Strategy Refinement**: Optimize parameters with real market data

## Monitoring & Analytics

### Real-time Monitoring
- **Heartbeat**: Every 30 seconds with system status
- **Trade Execution**: Real-time logging with confidence scores
- **API Health**: Response times, error rates, cache hit rates
- **Wallet Performance**: Individual wallet effectiveness tracking

### Discord Integration
**Trade Notifications:**
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

**Daily Summaries:**
- P&L performance and win rate
- Top performing alpha wallets
- Risk level adjustments
- Error notifications and system alerts

### Performance Tracking
- **P&L Store**: `data/pnl_state.json` with atomic updates
- **Trade History**: Detailed records with wallet attribution
- **Equity Curve**: Real-time capital tracking
- **Attribution Analysis**: Performance by wallet tier and confidence

## Multiple Entry Points

### Production Use
```bash
python start_bot.py    # Full production mode with monitoring
```

### Development/Testing
```bash
python main.py         # Core bot only, minimal overhead
```

### Discovery & Analysis
```bash
python -m src.discovery.alpha_discovery_v2   # Manual alpha discovery
```

## Data Storage

### Primary Storage (JSON)
- **P&L State**: `data/pnl_state.json` - Equity, trades, positions
- **Wallet Performance**: `data/wallet_performance.json` - Bayesian statistics

### Configuration
- **Main Config**: `config/config.yml` - All settings and parameters
- **Watched Wallets**: Auto-updated during rotation process

### SQLite Databases (Optional)
- **Main DB**: `trades.db` - Comprehensive trade and position history
- **Analytics**: Extended performance metrics and research data

## Error Handling & Resilience

### Built-in Reliability
- **Automatic Retry**: Exponential backoff for API failures
- **Fallback Sources**: PumpPortal → BitQuery → Moralis
- **Rate Limiting**: Intelligent compliance with API limits
- **Connection Monitoring**: WebSocket health checks and reconnection
- **Graceful Degradation**: Continues operation with reduced functionality

### Safety Features
- **Paper Mode Default**: No accidental live trading
- **Configuration Validation**: Comprehensive parameter checking
- **Position Limits**: Multiple layers of risk control
- **Emergency Stops**: Failsafe mechanisms for all scenarios

## Development

Project structure:
```
├── main.py                           # Core bot implementation
├── start_bot.py                     # Production launcher
├── config.yml                       # Your configuration
├── src/
│   ├── clients/                     # API integrations
│   │   ├── pumpportal_client.py    # PumpPortal WebSocket
│   │   ├── bitquery_client.py      # Historical data
│   │   ├── moralis_client.py       # Token metadata
│   │   └── realtime_client.py      # Unified real-time interface
│   ├── core/                        # Core trading logic
│   │   ├── trading_engine.py       # Position management
│   │   ├── wallet_tracker.py       # Alpha signal analysis
│   │   ├── wallet_scorer.py        # Bayesian scoring
│   │   ├── risk_manager.py         # Adaptive risk management
│   │   ├── safety_checks.py        # Rug pull detection
│   │   └── wallet_rotation_manager.py # Smart rotation
│   ├── discovery/                   # Alpha discovery
│   │   └── alpha_discovery_v2.py   # Historical analysis
│   └── utils/                       # Utilities
│       ├── config_loader.py        # Configuration management
│       ├── discord_notifier.py     # Notifications
│       ├── pnl_store.py           # P&L tracking
│       └── wallet_performance.py   # Performance analytics
├── scripts/                         # Automation
│   └── alpha_discovery_scheduler.py # Auto-discovery
└── config/
    └── config.yml.example          # Configuration template
```

## Performance Expectations

### Realistic Targets (Paper Trading Validated)
- **Win Rate**: 50-60% (conservative alpha-following)
- **Average Gain**: 15-25% per winning trade
- **Average Loss**: 5-8% per losing trade (early stop loss)
- **Daily Trades**: 5-20 (adaptive based on market conditions)
- **Hold Time**: 2-15 minutes average

### Key Success Metrics
1. **Consistent Profitability**: Positive expectancy over 100+ trades
2. **Risk Control**: Maximum 8% loss per trade, controlled drawdowns
3. **Signal Quality**: High confidence alpha wallet signals
4. **Execution Speed**: <3 seconds from signal to execution

## 📚 Documentation

### Recent Updates (December 2024)
- **[CURRENT_STATE.md](docs/CURRENT_STATE.md)** - Complete overview of current bot state and recent optimizations
- **[Improvements.md](Improvements.md)** - Implementation status of win rate improvements  
- **[FEATURE_REFERENCE.md](docs/FEATURE_REFERENCE.md)** - Quick reference for all active features

### Strategy Documentation  
- **[TRADING_PLAN_DETAILED.md](docs/TRADING_PLAN_DETAILED.md)** - Comprehensive strategy breakdown
- **[DISCORD_SETUP.md](docs/DISCORD_SETUP.md)** - Discord notification setup guide

### Configuration Examples
- **[config.yml.example](config/config.yml.example)** - Complete configuration template with new features

**📊 Performance Note**: Recent optimizations target 35-45% win rate (up from 20%) with reduced average losses through enhanced entry filtering and intelligent exit strategies.

## Disclaimer

This bot is for educational and research purposes. Cryptocurrency trading carries significant risk of loss. Past performance does not guarantee future results. The system uses sophisticated risk management but cannot eliminate all trading risks. Only trade with money you can afford to lose.

## License

MIT License - see LICENSE file for details.