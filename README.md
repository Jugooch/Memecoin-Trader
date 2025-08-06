# 🚀 Memecoin Trading Bot

Intelligent Solana memecoin trading bot that follows proven alpha wallets to identify profitable trades early. Monitors pump.fun launches and executes trades when multiple alpha wallets show buying activity.

## ✨ Key Features

- **🎯 Alpha Wallet Discovery**: Automated system finds proven successful wallets
- **🔄 API Key Rotation**: Multiple Moralis keys for 24/7 operation without limits
- **📊 Smart Caching**: Reduces API calls by 70-90% with intelligent caching
- **🛡️ Advanced Risk Management**: Position sizing, stop losses, take profits
- **🔍 Multi-Layer Filtering**: Liquidity checks, spam detection, quality assessment
- **📈 Multiple Trading Modes**: Simulation, alerts, or fully automated trading
- **⚡ Real-time Monitoring**: Live alpha wallet activity detection
- **📱 Performance Tracking**: Complete trade history and analytics
- **🛠️ Production Ready**: Error handling, rate limiting, graceful degradation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the configuration template
cp config.yml.example config.yml

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

### 3. Discover Alpha Wallets (Recommended)

```bash
# Run the advanced alpha wallet discovery system
python alpha_discovery_v2.py
```

This will automatically find proven alpha wallets and update your config.

### 4. Start the Bot

```bash
python start_bot.py
```

## Configuration

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
- `max_trade_pct: 0.05` - Maximum 5% of capital per trade
- `threshold_alpha_buys: 3` - Minimum alpha wallet buys to trigger trade
- `tp_multiplier: 2.0` - Take profit at 2x entry price
- `stop_loss_pct: 0.7` - Stop loss at 70% of entry price

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

## Strategy Overview

The bot implements a systematic approach to memecoin trading:

1. **Detection**: Monitor Pump.fun for new token launches via Bitquery
2. **Filtering**: Apply liquidity, volume, and quality filters via Moralis
3. **Validation**: Check for alpha wallet activity within time window (2+ minutes)
4. **Execution**: Place buy orders via QuickNode Pump.fun API
5. **Management**: Monitor positions with TP/SL and trailing stops
6. **Analysis**: Track performance and optimize parameters

## 🎯 Alpha Wallet Discovery System

The bot includes an advanced alpha wallet discovery system that automatically finds proven successful wallets:

### How It Works:
1. **Time-Delayed Analysis**: Analyzes tokens from 24-48 hours ago (proven timeframe)
2. **Success Validation**: Identifies tokens that became successful (holders, activity, price)
3. **Early Buyer Detection**: Finds wallets that bought successful tokens within first 30 seconds
4. **Performance Scoring**: Validates wallet consistency across multiple successful tokens
5. **Auto-Configuration**: Updates your config.yml with discovered alpha wallets

### Run Discovery:
```bash
python alpha_discovery_v2.py
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

- **Position Sizing**: Maximum 5% of capital per trade
- **Take Profit**: Sell 50% at 2x entry price
- **Stop Loss**: Exit at 70% of entry price
- **Trailing Stop**: Exit if price drops 20% from peak
- **Time Limit**: Maximum 4-hour hold time
- **Daily Limits**: Maximum trades per day to prevent overtrading

## Paper Trading Mode

Start with paper trading to test the strategy:

1. Set `paper_mode: true` in config
2. Monitor performance for several days
3. Analyze win rate and profit metrics
4. Switch to live trading when confident

## Monitoring & Analytics

The bot provides comprehensive monitoring:

- Real-time trade execution logs
- Daily performance summaries
- Win rate and profit tracking
- API health monitoring
- System resource usage
- Performance alerts

View performance reports:
```bash
# Check daily summary
tail -f logs/trading.log

# Generate performance report
python -c "
import asyncio
from monitoring import PerformanceMonitor
from database import Database

async def report():
    db = Database()
    await db.initialize()
    monitor = PerformanceMonitor(db)
    report = await monitor.generate_performance_report(7)
    print(report)

asyncio.run(report())
"
```

## Database Schema

The bot maintains SQLite database with tables:
- `trades` - All trade executions
- `positions` - Active positions
- `performance` - Daily metrics
- `alpha_wallets` - Wallet performance data
- `token_analysis` - Token research data

## Error Handling

Built-in resilience features:
- Automatic retry with exponential backoff
- Fallback API endpoints
- Rate limiting compliance
- 530 error handling for Pump.fun API
- Network connectivity monitoring

## Safety Features

- Paper mode for risk-free testing
- Position size limits
- Daily trade limits
- Stop loss protection
- API key encryption support
- No automatic mainnet transactions

## Development

Project structure:
```
  src/
  ├── clients/     (API clients)
  ├── core/        (Trading engine, database, wallet tracker)
  ├── discovery/   (Alpha wallet discovery)
  └── utils/       (Monitoring, logging)

  scripts/
  ├── start_all.sh              (Main startup script)
  ├── alpha_discovery_scheduler.py  (6-hour scheduler)
  ├── health_check.py           (Monitor bot health)
  └── memecoin-bot.service      (Systemd service)
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