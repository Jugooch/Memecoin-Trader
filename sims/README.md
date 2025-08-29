# Strategy Simulation Engine - Implementation Guide

## Overview

This simulation engine monitors real-time alpha wallet trades and tests three different trading strategies **without risking real money**. It collects actual market data to determine which strategy performs best, allowing us to optimize our trading approach based on hard data rather than assumptions.

## What This Proves

### Core Questions We're Answering:
1. **Are we exiting too early?** - Do tokens pump after we hit stop loss?
2. **Is volatility buffer worth it?** - Do more tokens recover after initial dips?
3. **Are we missing late entries?** - Do tokens with 4+ alpha wallets still pump?
4. **What's our real win rate?** - Which strategy has the best risk/reward?

### Expected Insights:
- **If Buffer Strategy wins**: Our stop losses are too tight, we're panic selling during normal volatility
- **If Late Alpha Strategy wins**: We should wait for more confirmation before entering
- **If Current Strategy wins**: Our current approach is already optimal

## Three Strategies Being Tested

### Strategy 1: "Current Implementation"
**Simulates our existing bot behavior**
- ✅ **Entry**: 1+ alpha wallet, within 15 seconds
- ⚠️ **Stop Loss**: -15% immediately (aggressive)
- 🎯 **Take Profit**: 20% (full exit)
- 🕐 **Issues**: Panic sells during volatility, can't execute due to indexing delays

### Strategy 2: "Buffer Strategy" 
**Proposed improvement with volatility buffer**
- ✅ **Entry**: Same as current (1+ alpha, 15s)
- 🛡️ **Buffer**: No stop losses for first 60 seconds
- 📈 **Take Profits**: Still allowed during buffer (catch pumps!)
- 💰 **Multi-tier**: TP1(20%/45%), TP2(50%/35%), TP3(100%/15%)
- ⚠️ **Stop Loss**: -35% after buffer period (wider)

### Strategy 3: "Late Alpha Strategy"
**Tests waiting for more confirmation**
- ⏰ **Entry**: 4+ alpha wallets, within 5 minutes
- 🛡️ **Buffer**: Same as Strategy 2 (already late entry)
- 💡 **Hypothesis**: More alphas = higher success rate

## Technical Architecture

### Data Flow:
```
Real-time Alpha Trades → Strategy Simulation → Price Monitoring → Exit Simulation → Results Storage
```

### Key Components:
- **Alpha Monitor**: Watches PumpPortal WSS for wallet trades
- **Position Simulator**: Creates virtual positions with real prices
- **Exit Engine**: Tests strategy-specific exit conditions
- **Results Tracker**: Stores all trades in JSON files
- **Progress Reporter**: Sends Discord updates every 30 minutes

### File Structure:
```
sims/
├── strategy_simulator.py    # Main simulation engine
├── data/                   # JSON results storage
│   ├── strategy_sim_current.json
│   ├── strategy_sim_buffer.json
│   └── strategy_sim_alphas.json
├── logs/                   # Simulation logs
│   └── simulation_YYYYMMDD_HHMMSS.log
├── configs/                # Strategy configurations
└── README.md              # This file
```

## Implementation Checklist

### Prerequisites ✅
- [ ] **Config file exists**: `config/config.yml` with PumpPortal API key
- [ ] **Alpha wallets loaded**: `watched_wallets` list populated in config
- [ ] **Dependencies installed**: All bot dependencies already available
- [ ] **Discord webhook** (optional): For progress reports

### Core Implementation Tasks

#### Phase 1: Setup & Structure ✅
- [x] **Create sims directory structure**: `/sims/data/`, `/sims/logs/`, `/sims/configs/`
- [x] **Main simulator file**: `strategy_simulator.py` with class structure
- [ ] **Test basic imports**: Ensure all parent directory imports work
- [ ] **Config loading test**: Verify config.yml loads correctly

#### Phase 2: Real-time Monitoring 🔄
- [ ] **PumpPortal integration**: Connect to WSS and subscribe to alpha wallets
- [ ] **Alpha event processing**: Parse trade events and track by mint address
- [ ] **Price data integration**: Connect Moralis for current price fetching
- [ ] **Symbol resolution**: Get token names/symbols for logging

#### Phase 3: Strategy Logic 🧠
- [ ] **Entry condition logic**: Implement 3 different entry criteria
  - Current: 1+ alpha, ≤15s delay
  - Buffer: Same as current
  - Late Alpha: 4+ alphas, ≤300s delay
- [ ] **Exit condition engines**: 
  - Current: -15% stop, +20% TP
  - Buffer: 60s buffer, -35% stop, multi-tier TPs
  - Late Alpha: Same as buffer
- [ ] **Position tracking**: Track virtual positions with real price updates
- [ ] **Partial exit simulation**: Handle multi-tier profit taking

#### Phase 4: Data Management 💾
- [ ] **JSON storage system**: Save results to strategy-specific files
- [ ] **Real-time updates**: Update positions every 5 seconds
- [ ] **Results calculation**: Win rate, P&L, hold times, exit breakdowns
- [ ] **Data persistence**: Handle restarts and resume functionality

#### Phase 5: Monitoring & Reporting 📊
- [ ] **Progress reports**: Generate comparison reports every 30 minutes
- [ ] **Discord integration**: Send automated updates (optional)
- [ ] **Logging system**: Comprehensive logging for debugging
- [ ] **Error handling**: Robust error handling for production use

#### Phase 6: Testing & Validation 🧪
- [ ] **Dry run test**: Run simulator for 1 hour with logging
- [ ] **Price accuracy**: Verify price data matches reality
- [ ] **Exit timing**: Confirm exit conditions trigger correctly  
- [ ] **Results validation**: Manual check of P&L calculations

### Implementation Dependencies

#### Required Files to Modify/Create:
1. **`sims/strategy_simulator.py`** - Main engine (created)
2. **Test script** - `sims/test_simulator.py` for validation
3. **Requirements** - Ensure all parent imports work
4. **Config validation** - Test PumpPortal connectivity

#### Required APIs:
- ✅ **PumpPortal**: Real-time alpha wallet monitoring
- ✅ **Moralis**: Token price data and metadata
- ✅ **Discord** (optional): Progress notifications

## Expected Results Format

### JSON Output Structure:
```json
{
  "last_updated": "2025-08-29T15:30:00",
  "results": {
    "strategy_name": "buffer",
    "total_trades": 45,
    "winning_trades": 28,
    "losing_trades": 17,
    "win_rate": 62.2,
    "total_profit_usd": 234.50,
    "avg_profit_per_trade": 5.21,
    "max_profit_trade": 87.30,
    "max_loss_trade": -23.40
  },
  "active_positions": 3,
  "completed_positions": [...trade_details...]
}
```

### Discord Report Example:
```
📊 Strategy Simulation Update

Current Implementation:
• Total Trades: 42 (Active: 2)
• Win Rate: 31.0%
• Total P&L: $-127.30
• Avg P&L/Trade: $-3.03

Buffer Strategy (60s):
• Total Trades: 45 (Active: 3) 
• Win Rate: 62.2%
• Total P&L: $+234.50
• Avg P&L/Trade: $+5.21

Late Alpha Strategy (4+ alphas):
• Total Trades: 12 (Active: 1)
• Win Rate: 75.0%
• Total P&L: $+89.40
• Avg P&L/Trade: $+7.45
```

## Usage Instructions

### Starting the Simulation:
```bash
cd /path/to/Memecoin-Trader/sims
python strategy_simulator.py
```

### Interactive Setup:
- Enter desired simulation duration (hours)
- Monitor logs for real-time activity
- Check JSON files for results
- Receive Discord progress updates

### Stopping the Simulation:
- `Ctrl+C` for graceful shutdown
- Results automatically saved
- Resume capability (future enhancement)

## Success Criteria

### Validation Checklist:
- [ ] **Real trades detected**: Alpha wallet trades captured in real-time
- [ ] **Price accuracy**: Current prices match actual market data
- [ ] **Strategy differentiation**: Each strategy shows different results
- [ ] **Exit logic works**: Stop losses and take profits trigger correctly
- [ ] **Data persistence**: Results saved and readable in JSON format
- [ ] **Error resilience**: Handles API failures gracefully

### Expected Timeline:
- **Setup & Integration**: 2-4 hours
- **Testing & Debugging**: 1-2 hours  
- **Initial Data Collection**: 24 hours minimum
- **Results Analysis**: 1 hour

### Key Metrics to Watch:
1. **Trade Volume**: Are we capturing enough alpha signals?
2. **Strategy Divergence**: Do strategies show significantly different results?
3. **Price Data Quality**: Are we getting accurate, timely price feeds?
4. **System Stability**: Does it run for 24+ hours without errors?

## Troubleshooting Guide

### Common Issues:
- **Import errors**: Check parent directory path in sys.path
- **API failures**: Verify PumpPortal/Moralis keys in config
- **No alpha signals**: Confirm watched_wallets list is populated  
- **Price data missing**: Check Moralis rate limits and API status
- **JSON errors**: Verify write permissions in sims/data/ directory

### Debug Mode:
- Set logging level to DEBUG for verbose output
- Monitor sims/logs/ for detailed error messages
- Check active_positions count in JSON files
- Verify alpha_signals dictionary population

This simulation engine will provide concrete data to optimize our trading strategy and maximize profitability based on real market behavior rather than theoretical assumptions.