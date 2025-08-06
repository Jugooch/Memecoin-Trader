# Memecoin Trading Bot - Oracle Cloud VM Deployment Guide

## Project Status Analysis

### ✅ What's Ready for 24/7 Operation

1. **Main Trading Bot** (`start_bot.py`)
   - Paper trading mode fully implemented
   - Monitors pump.fun for new token launches
   - Tracks alpha wallet activity
   - Automatic buy/sell with stop-loss and take-profit
   - Performance monitoring and reporting

2. **Alpha Wallet Discovery** (`src/discovery/alpha_discovery_v2.py`)
   - Time-delayed analysis to find proven performers
   - Runs every 6 hours via scheduler
   - Automatically updates config with new wallets
   - Database persistence for tracking

3. **Key Rotation for Moralis**
   - Multiple API keys configured
   - Automatic rotation when rate limited
   - Prevents service interruption

4. **Monitoring & Health Checks**
   - Performance monitoring dashboard
   - System health checks
   - Trade logging and metrics

### ⚠️ Current Gaps & Solutions

1. **Alpha Discovery Not Auto-Scheduled**
   - **Gap**: Alpha discovery needs manual triggering
   - **Solution**: Created `scripts/alpha_discovery_scheduler.py` to run every 6 hours

2. **No Automatic VM Startup**
   - **Gap**: Bot doesn't start automatically on VM boot
   - **Solution**: Created systemd service file for automatic startup

## Deployment Instructions for Oracle Cloud VM

### 1. Initial Setup on VM

```bash
# Clone the repository
git clone <your-repo-url> memecoin-trader
cd memecoin-trader

# Install Python 3.8+ if not present
sudo apt update
sudo apt install python3.8 python3.8-venv python3-pip

# Create virtual environment
python3 -m venv bot_env
source bot_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure the Bot

```bash
# Copy and edit config
cp config/config.yml.example config/config.yml
nano config/config.yml

# Make sure to set:
# - Your API keys (Bitquery, Moralis, QuickNode if available)
# - trading_mode: simulation (for paper trading)
# - initial_capital: 100.0
# - watched_wallets: (already populated with discovered wallets)
```

### 3. Test the Bot First

```bash
# Test alpha discovery
python3 src/discovery/alpha_discovery_v2.py

# Test main bot (run for a few minutes)
python3 start_bot.py
```

### 4. Install as System Service (for 24/7 operation)

```bash
# Install the service
sudo ./scripts/install_service.sh

# Start the bot
sudo systemctl start memecoin-bot

# Check status
sudo systemctl status memecoin-bot

# View logs
sudo journalctl -u memecoin-bot -f
```

### 5. Alternative: Run with Screen (simpler method)

```bash
# Install screen
sudo apt install screen

# Start a new screen session
screen -S memecoin

# Run the bot
./scripts/start_all.sh

# Detach from screen (Ctrl+A, then D)
# To reattach: screen -r memecoin
```

## Project Structure (Reorganized)

```
memecoin-trader/
├── config/
│   ├── config.yml           # Your actual config (API keys)
│   └── config.yml.example   # Template config
├── src/
│   ├── clients/             # API clients
│   │   ├── bitquery_client.py
│   │   ├── moralis_client.py
│   │   └── pumpfun_client.py
│   ├── core/                # Core bot logic
│   │   ├── database.py
│   │   ├── trading_engine.py
│   │   └── wallet_tracker.py
│   ├── discovery/           # Alpha wallet discovery
│   │   ├── alpha_discovery_v2.py
│   │   └── alpha_finder.py
│   └── utils/               # Utilities
│       ├── logger_setup.py
│       └── monitoring.py
├── scripts/                 # Deployment scripts
│   ├── start_all.sh        # Main startup script
│   ├── alpha_discovery_scheduler.py
│   ├── health_check.py
│   ├── install_service.sh
│   └── memecoin-bot.service
├── logs/                    # Log files (created automatically)
├── main.py                  # Main bot logic
├── start_bot.py            # Bot entry point
└── dashboard.py            # Performance dashboard
```

## How It Works (24/7 Operation)

1. **Main Trading Bot** runs continuously:
   - Monitors WebSocket for new token launches
   - Checks if alpha wallets are buying
   - Executes paper trades automatically
   - Manages positions with TP/SL

2. **Alpha Discovery Scheduler** runs every 6 hours:
   - Analyzes tokens from 24-48 hours ago
   - Finds wallets that consistently buy winners
   - Updates config with new alpha wallets
   - Bot automatically uses updated wallet list

3. **Paper Trading** tracks everything:
   - All trades saved to database
   - Performance metrics calculated
   - No real money used
   - Full simulation of trading strategy

## Monitoring the Bot

### Check Health Status
```bash
python3 scripts/health_check.py
```

### View Dashboard
```bash
python3 dashboard.py
```

### Check Logs
```bash
# Main bot logs
tail -f logs/trading.log

# Alpha discovery logs
tail -f logs/alpha_discovery_scheduler.log

# System service logs
sudo journalctl -u memecoin-bot -f
```

## Expected Behavior

When running 24/7, you should see:

1. **Every 5 minutes**: Summary of tokens scanned and trades
2. **Every 30 seconds**: Heartbeat if no other activity
3. **When alpha signal detected**: Trade execution logs
4. **Every 6 hours**: New alpha wallets discovered

## Troubleshooting

### Bot not finding trades?
- Normal - strict filters mean fewer but higher quality trades
- Check that alpha wallets are active
- Verify Moralis/Bitquery API keys are working

### Rate limiting issues?
- Multiple Moralis keys are rotating automatically
- If still limited, add more keys to config

### No WebSocket data?
- Check Bitquery token is valid
- Ensure network connectivity
- Bot will auto-reconnect on disconnection

## Safety Features

1. **Paper Mode Only**: No real money at risk
2. **Daily Trade Limits**: Max 10 trades per day
3. **Position Sizing**: Max 5% per trade
4. **Stop Loss**: 15% protection on all trades
5. **API Key Rotation**: Prevents service interruption

## Next Steps for Production

1. **Monitor for 1-2 weeks** in paper mode
2. **Analyze performance** via dashboard
3. **Tune parameters** based on results:
   - Adjust `threshold_alpha_buys` if too strict/loose
   - Modify `tp_multiplier` and `stop_loss_pct`
   - Update `min_liquidity_usd` threshold
4. **Only then** consider live trading with small amounts

## Important Notes

- The bot is designed for 24/7 operation but monitor it regularly
- Alpha wallet discovery improves over time
- Paper trading gives realistic results without risk
- Keep your API keys secure and never commit them to git

## Support

- Check logs first for any errors
- Health check script shows system status
- Dashboard provides performance metrics
- All trades are logged in the database for analysis