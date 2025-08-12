# 🚀 Memecoin Trading Bot

Intelligent Solana memecoin trading bot that follows proven "alpha wallets" (smart money) to identify profitable trades early. Monitors pump.fun token launches in real-time and executes trades when multiple high-performing wallets show buying activity.

## ✨ Key Features

- **🎯 Dual Alpha Discovery Systems**: Continuous accumulator + one-off analysis tools
- **🔄 Multi-Key Rotation**: 8 Moralis keys + multiple Bitquery tokens for 24/7 operation
- **📊 Intelligent Caching**: Request coalescing + tiered TTLs reduce API calls by 70-90%
- **🛡️ Tiered Risk Management**: Wallet performance tiers (S/A/B/C) adjust position sizes
- **🔍 Safety Scoring**: Rug pull detection, honeypot checks, liquidity validation
- **📈 Paper Trading Mode**: Realistic fee simulation (DEX fees, slippage, network costs)
- **⚡ Real-time Monitoring**: WebSocket subscriptions for instant token detection
- **📱 P&L Tracking**: JSON-based persistence with atomic writes
- **🛠️ Production Ready**: Deduplication, inactive wallet detection, error recovery
- **🔔 Discord Integration**: Trade notifications, summaries, and error alerts

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
bitquery_token: "your_bitquery_api_key"

# Multiple Moralis keys for rotation (recommended)
moralis_keys:
  - your_first_moralis_key_here
  - your_second_moralis_key_here  # Optional but recommended
  - your_third_moralis_key_here   # For 24/7 operation

# Optional: QuickNode for automated trading
quicknode_endpoint: ""  # Leave empty for simulation mode
quicknode_api_key: ""
```

### 3. Start Alpha Wallet Discovery (Recommended)

```bash
# PRODUCTION: Run continuous alpha wallet discovery (recommended)
python scripts/run_alpha_accumulator.py --loop

# TESTING: Run one-off analysis
python -m src.discovery.alpha_discovery_v2
```

The accumulator runs continuously and automatically finds proven alpha wallets over time, updating your config.

### 4. Start the Bot

```bash
python start_bot.py
```

## Configuration System

The bot uses a centralized configuration system with smart path resolution and validation.

### Configuration File Location
The bot automatically searches for `config.yml` in multiple locations:
- Current directory: `./config.yml`
- Config directory: `./config/config.yml` 
- Project root variations

### Configuration Template
Copy the template and customize:
```bash
cp config/config.yml.example config.yml
```

### Trading Modes
Choose your execution mode in `config.yml`:

```yaml
trading_mode: "simulation"  # Options:
# "simulation" = Paper trading only (100% free)
# "alerts" = Generate buy/sell alerts for manual execution (free)  
# "auto" = Fully automated trading (requires QuickNode $49/mo)
```

### Key Configuration Options:

- `initial_capital: 100.0` - Starting capital amount
- `max_trade_pct: 0.05` - Maximum 5% of capital per trade (base amount)
- `threshold_alpha_buys: 3` - Minimum alpha wallet buys to trigger trade
- `tp_multiplier: 1.5` - First take profit at 1.5x (sells 30%)
- `stop_loss_pct: 0.85` - Stop loss at 85% of entry price (15% max loss)
- `time_window_sec: 120` - 2-minute window for alpha wallet detection
- `min_liquidity_usd: 1000` - Minimum token liquidity requirement
- `max_trades_per_day: 10` - Daily trade limit (capped at 20 max)
- `min_time_between_trades: 30` - Minimum seconds between trades

## API Setup Required

### Bitquery (Token Launch Monitoring)
1. Sign up at [Bitquery.io](https://bitquery.io)
2. Get API key for GraphQL access
3. Set up Solana Pump.fun program monitoring

### Moralis (Token Data & Analytics)
1. Create account at [Moralis.io](https://moralis.io)
2. Get Solana API key
3. Used for token metadata, liquidity, and price data

### QuickNode (Optional - Only for Automated Trading)
1. **NOT REQUIRED** for paper trading or alerts mode
2. Only needed if you want fully automated live trading
3. Sign up at [QuickNode.com](https://quicknode.com) when ready to scale
4. Enable Pump.fun marketplace add-on ($49/month)

## Why This Approach Works on $100 Budget

### No Expensive Infrastructure Required
- **Free RPC**: Uses public Solana mainnet RPC (no $500+/month subscriptions)
- **Free APIs**: Bitquery & Moralis have generous free tiers
- **Smart Timing**: Waits for alpha confirmation rather than racing milliseconds
- **Quality Focus**: Makes 5-10 high-conviction trades vs. 100s of rushed trades

### Strategy Over Speed
Instead of competing on latency (expensive), we compete on intelligence:
- **Alpha Wallet Signals**: Wait for proven smart money to validate tokens
- **Liquidity Filtering**: Only trade tokens with sufficient backing
- **Risk Management**: Position sizing and stop losses protect capital
- **Patience**: 2-minute analysis windows vs. instant reactions

## Trading Strategy Overview

The bot implements a conservative smart-money-following strategy:

1. **Token Detection**: WebSocket monitoring of Pump.fun via Bitquery GraphQL
2. **Initial Filtering**: 
   - Minimum $1,000 liquidity
   - Spam/scam token filtering
   - Activity validation (swaps, unique traders)
3. **Alpha Validation**:
   - Check if 3+ watched wallets bought within 120 seconds
   - Calculate confidence score (0-100)
   - Determine investment multiplier based on wallet tiers
4. **Safety Checks**:
   - Rug score calculation (<70 required)
   - Honeypot detection (check for sells)
   - Trader diversity analysis
5. **Position Entry**:
   - Base: 5% of capital
   - Adjusted by wallet tier multiplier (0.6x-2.0x)
   - Realistic fees: 0.3% DEX + ~$0.28 network + 0.5% slippage
6. **Exit Management**:
   - TP1 @ 1.5x: Sell 30%
   - TP2 @ 1.75x: Sell 30%
   - Trailing stop after 50% gain
   - Time stop after 5 minutes
   - Break-even stop armed at +10%

## 🎯 Alpha Wallet Discovery System

The bot includes an advanced alpha wallet discovery system that automatically finds proven successful wallets:

### Two Discovery Systems:

#### 1. Alpha Discovery V2 (One-off Analysis)
- Analyzes recent 2-hour window of trades
- Identifies successful tokens (1.2x-2x price increase)
- Finds wallets buying within first 10 minutes
- Requires 2+ appearances on successful tokens
- Best for: Quick manual discovery

#### 2. Alpha Accumulator (Continuous - RECOMMENDED)
- Runs every 2 minutes continuously
- Builds overlap data over time
- Handles Bitquery's 3-4 minute data limitation
- Stores in SQLite database
- Best for: Production use

### Wallet Tiering System:
- **S-Tier**: 70%+ win rate, 100%+ avg profit → 2.0x investment
- **A-Tier**: 60%+ win rate, 50%+ avg profit → 1.4x investment  
- **B-Tier**: 50%+ win rate, 20%+ avg profit → 1.0x investment
- **C-Tier**: Below B-tier thresholds → 0.6x investment

### Run Discovery:
```bash
python src/discovery/alpha_discovery_v2.py
```

### Manual Alpha Wallets:
You can also manually add proven wallets to `config.yml`:
```yaml
watched_wallets:
  - "wallet_address_1"
  - "wallet_address_2"
  # Automatically populated by alpha discovery
```

### Alpha Trading Logic:
- Bot monitors all alpha wallets in real-time
- When 3+ alpha wallets buy the same token → Trade signal triggered
- Executes trade automatically (in simulation mode by default)

## Risk Management

### Position Sizing
- **Base Size**: 5% of capital per trade
- **Tier Adjustment**: Multiplied by wallet tier (0.6x-2.0x)
- **Max Concurrent**: 3 positions
- **Daily Limit**: 10 trades (hard cap at 20)

### Exit Strategy (Tiered)
- **TP1 @ 1.5x**: Sell 30% of position
- **TP2 @ 1.75x**: Sell 30% of position  
- **Stop Loss**: Exit at 85% of entry (15% max loss)
- **Trailing Stop**: After 50% gain, exit if drops to 85% of peak
- **Time Stop**: Force exit after 5 minutes
- **Break-Even**: Armed at +10%, exits at entry price

### Safety Features
- **Cool-down**: 3 minutes after stop loss
- **Min Interval**: 2 minutes between trades
- **Inactive Wallet Detection**: Marks wallets inactive after 6 hours
- **Rug Score**: Must be <70 to trade

## Paper Trading Mode

Start with paper trading to test the strategy:

1. Set `trading_mode: "simulation"` in config.yml
2. Monitor performance for several days using `python dashboard.py`
3. Analyze win rate and profit metrics
4. Switch to live trading when confident

## Multiple Entry Points

The bot provides several ways to run depending on your needs:

### Production Use
```bash
python start_bot.py    # Full production mode with monitoring
```

### Development/Testing
```bash
python main.py         # Core bot only, minimal overhead
```

### Monitoring
```bash
python dashboard.py    # Performance dashboard and analytics
```

### Discovery
```bash
python src/discovery/alpha_discovery_v2.py   # Find new alpha wallets
```

## Monitoring & Analytics

### Real-time Monitoring
- **Heartbeat**: Every 30 seconds WebSocket status
- **5-min Summary**: Tokens scanned, alpha checks, trades, P&L
- **Trade Logs**: Entry/exit with confidence scores
- **API Stats**: Deduplication savings, cache hits
- **Wallet Status**: Active/inactive tracking

### Discord Notifications
- Trade alerts with confidence scores
- 5-minute activity summaries
- Error notifications (API failures, rate limits)
- Wallet recycling alerts

### Performance Tracking
- **P&L Store**: JSON file (`data/pnl_state.json`)
- **Metrics**: Win rate, daily P&L, equity curve
- **Trade History**: Last 100 trades cached
- **Position Tracking**: Real-time unrealized P&L

View performance reports:
```bash
# Check daily summary
tail -f logs/trading.log

# View live dashboard
python dashboard.py

# Generate performance report  
python -c "
import asyncio
from src.core.database import Database
from src.utils.monitoring import PerformanceMonitor
from src.utils.config_loader import load_config, get_database_path

async def report():
    config = load_config()
    db = Database(get_database_path(config))
    await db.initialize()
    monitor = PerformanceMonitor(db)
    report = await monitor.generate_performance_report(7)
    print(report)

asyncio.run(report())
"
```

## Data Storage

### Primary Storage (JSON)
- **P&L State**: `data/pnl_state.json` - Capital, trades, positions
- **Config**: `config/config.yml` - All settings and wallets

### SQLite Databases
- **Main DB**: `trades.db`
  - `trades` - All trade executions
  - `positions` - Active positions  
  - `alpha_wallets` - Wallet performance
  - `token_analysis` - Research data
  
- **Accumulator DB**: `data/alpha_accumulator.db`
  - `tokens` - Tracked token launches
  - `early_buys` - Wallet-token mappings

## Error Handling

Built-in resilience features:
- Automatic retry with exponential backoff
- Fallback API endpoints
- Rate limiting compliance
- 530 error handling for Pump.fun API
- Network connectivity monitoring

## Safety Features

- Paper mode for risk-free testing
- Position size limits (5% max per trade)
- Daily trade limits (configurable)
- Stop loss protection (30% max loss)
- Comprehensive configuration validation
- Smart error handling and fallbacks
- No automatic mainnet transactions without explicit configuration

## Development

Project structure:
```
├── main.py                    # Core bot implementation
├── start_bot.py              # Production launcher with monitoring
├── dashboard.py              # Performance monitoring dashboard
├── config.yml                # Your configuration (create from template)
├── src/
│   ├── clients/              # API clients
│   │   ├── bitquery_client.py
│   │   ├── moralis_client.py 
│   │   └── pumpfun_client.py
│   ├── core/                 # Core trading logic
│   │   ├── database.py
│   │   ├── trading_engine.py
│   │   └── wallet_tracker.py
│   ├── discovery/            # Alpha wallet discovery
│   │   └── alpha_discovery_v2.py
│   └── utils/                # Utilities and helpers
│       ├── config_loader.py  # Shared configuration loading
│       ├── logger_setup.py
│       └── monitoring.py
├── scripts/                  # Deployment and automation
│   ├── start_all.sh         # System service startup
│   ├── alpha_discovery_scheduler.py  # Auto-discovery every 6h
│   ├── health_check.py      # System health monitoring
│   └── install_service.sh   # Service installation
└── config/
    └── config.yml.example   # Configuration template
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test thoroughly with paper trading
4. Submit pull request

## Disclaimer

This bot is for educational purposes. Cryptocurrency trading carries significant risk. Past performance does not guarantee future results. Only trade with money you can afford to lose.

## License

MIT License - see LICENSE file for details.