# Live Trading Configuration Guide

## ⚠️ SECURITY WARNING
- NEVER commit your config.yml with real keys to git
- Add config/config.yml to .gitignore
- Store backups of keys securely offline

## Step 1: Copy this to your config.yml

Update these sections in your config.yml:

```yaml
# QuickNode Configuration (REQUIRED for live trading)
quicknode_endpoint: "https://YOUR-ENDPOINT-NAME.solana-mainnet.quiknode.pro/YOUR-KEY-HERE/"
quicknode_api_key: "YOUR-QUICKNODE-API-KEY-HERE"

# PumpPortal - Add wallet keys for trading features
pumpportal:
  api_key: "your_pumpportal_api_key_here"
  wallet_public_key: "YOUR-WALLET-PUBLIC-KEY-HERE"    # Your Solana address
  wallet_private_key: "YOUR-WALLET-PRIVATE-KEY-HERE"  # Base58 private key - KEEP SECURE!
  ws_endpoint: "wss://pumpportal.fun/api/data"
  trade_endpoint: "https://pumpportal.fun/api/trade"

# CRITICAL: Change trading mode for live trading
trading_mode: "auto"  # Change from "simulation" to "auto" for live trading
# Options:
# - "simulation": Paper trading only (safe for testing)
# - "alerts": Generate signals but don't execute (recommended first step)
# - "auto": Fully automated live trading

# Adjust trading limits for live trading (start conservative!)
trading:
  initial_capital: 100.0      # Start small! (was 500.0)
  max_trades_per_day: 5       # Reduce from 20 for safety
  min_time_between_trades: 300 # Increase to 5 min between trades
  max_concurrent_positions: 1  # Start with 1 position at a time
  
# Tighter risk management for live trading
max_trade_pct: 0.02           # Reduce to 2% per trade (was 5%)
tp_multiplier: 1.15           # Lower target: 15% profit (was 20%)
stop_loss_pct: 0.97           # Tighter stop: 3% loss (was 5%)
max_hold_seconds: 300         # Keep same: 5 minutes

# MORE CONSERVATIVE entry requirements for live trading
threshold_alpha_buys: 6       # Increase from 5 (more confirmation)
alpha_weight_min: 4.0         # Increase from 3.5 (stronger signals)
```

## Step 2: Test Configuration

Run this command to verify your config loads:
```bash
python -c "from src.utils.config_loader import load_config; c = load_config('config/config.yml'); print('Config loaded successfully!')"
```

## Step 3: Start in ALERTS mode first

Before going fully live, run in alerts mode:
1. Set `trading_mode: "alerts"` 
2. Run the bot and verify you're getting good signals
3. Only switch to `"auto"` after confirming signal quality

## Step 4: Funding Guidelines

For initial testing:
- Wallet balance: 0.5-1.0 SOL ($70-140)
- This allows for:
  - 5 trades at $10-15 each
  - Transaction fees (~0.01 SOL per trade)
  - Buffer for slippage

## Step 5: Monitor Your First Trades

When you switch to live:
1. Watch the first 2-3 trades manually
2. Verify entries and exits are executing
3. Check transaction on Solscan.io
4. Monitor your P&L tracking

## Emergency Stop

To stop the bot immediately:
- Press Ctrl+C twice
- Or run: `bash scripts/stop_bots.sh`

## Checklist Before Going Live

- [ ] Wallet has sufficient SOL balance
- [ ] QuickNode endpoint is working
- [ ] Config loads without errors
- [ ] Tested in "alerts" mode first
- [ ] Reduced position sizes for safety
- [ ] Have emergency stop plan ready
- [ ] Discord notifications configured (optional but recommended)