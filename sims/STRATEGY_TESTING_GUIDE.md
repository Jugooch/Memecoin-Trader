# 🧪 Advanced Strategy Testing Framework

## Quick Start

Get the **exact data GPT5 demanded** to settle the strategy debate:

```bash
# 1. Validate framework
cd sims
python test_framework.py

# 2. Run comprehensive test (6 hours)  
python run_comprehensive_test.py --live-hours 6

# 3. Get results in sims/results/
```

## What This Framework Does

This framework tests **your current 30% win rate strategy** vs **GPT5's recommendations** with:

✅ **Realistic execution modeling** (latency, slippage, failures)  
✅ **Historical backtesting** on your actual trading logs  
✅ **Live A/B testing** with real market data  
✅ **GPT5's demanded metrics** (fill rate, leader delta, TP rates, etc.)  
✅ **Mathematical break-even analysis** 
✅ **Discord progress reports**

## The Strategies Being Tested

### 🔴 Current Aggressive (Your Bleeding Strategy)
- **Single wallet trigger** (1 alpha = trade)
- **3% slippage** (low, causes missed fills)
- **-35% stop loss** with 60s buffer
- **Multi-tier exits**: 40% at +20%, 35% at +45%, 20% at +90%
- **Safety bypasses** for S-tier wallets
- **Result**: 30% win rate, bleeding money

### 🟢 GPT5 Recommended (The "Fix")  
- **Graduated entry**: 40% probe → 100% on 2nd wallet confirmation
- **Dynamic slippage**: 4-6-10% ladder based on timing
- **-15% stop loss** (immediate, no buffer)
- **Tighter exits**: 25% at +20%, derisk at +100%
- **Safety checks enabled** for all wallets
- **Prediction**: 40-45% win rate, profitable

### 🟡 Hybrid Compromise
- **Balanced approach** between current and GPT5
- **Tests what works** from both strategies

## Key Metrics Measured

**GPT5's Demanded Metrics:**
1. **Fill Rate** - % of signals that execute successfully  
2. **Leader Price Delta** - How much higher you pay vs alpha wallet
3. **TP Hit Rates** - What % of trades reach profit targets
4. **Loss Distribution** - How many trades lose >20%, >25%, >35%
5. **Early Recovery Rate** - % of -20% trades that recover
6. **Fee Burden** - Fees as % of gross profit

**Mathematical Viability:**
- **Winner/Loser Ratio** - Average winner vs average loser
- **Required Win Rate** - Minimum win rate needed to break even
- **Actual Win Rate** vs Required (the key number!)

## Usage Options

### 🚀 Quick Test (2 hours, good for initial validation)
```bash
python run_comprehensive_test.py --quick-test
```

### 📊 Standard Test (6 hours, reliable results)
```bash
python run_comprehensive_test.py --live-hours 6
```

### 🔬 Comprehensive Test (12+ hours, high confidence)
```bash
python run_comprehensive_test.py --live-hours 12 --historical-days 30
```

### 🎯 Historical Only (5 minutes, limited accuracy)
```bash
python historical_backtester.py
```

## Expected Results Timeline

**After 2 Hours:**
- 5-10 alpha signals detected
- Initial trend visible
- Low confidence recommendations

**After 6 Hours:**  
- 15-25 alpha signals detected
- Clear performance differences
- Medium confidence recommendations

**After 12+ Hours:**
- 30+ alpha signals detected  
- Statistical significance achieved
- High confidence recommendations

## Interpreting Results

### 🎉 Success Indicators
- **Win rate > 40%**: Strategy is mathematically profitable
- **Winner/loser ratio > 2.0**: Good risk/reward balance
- **Fill rate > 80%**: Getting executions on winners
- **Leader delta < 8%**: Not overpaying vs alpha wallets

### ⚠️ Warning Signs  
- **Win rate 35-40%**: Close but not quite profitable
- **High leader delta**: Paying too much, missing alpha advantage
- **Low TP hit rate**: Not capturing profits effectively
- **High loss concentration**: Too many big losers

### ❌ Failure Indicators
- **Win rate < 35%**: Strategy is bleeding money
- **Winner/loser ratio < 1.5**: Poor risk management
- **Low fill rate**: Missing good opportunities
- **High fee burden**: Edge being eaten by costs

## The Verdict Framework

The framework will give you one of these conclusions:

### 🏆 **Clear Winner**
- One strategy significantly outperforms (>10% win rate difference)
- Mathematical profitability achieved (>40% win rate)
- **Action**: Switch to winning configuration immediately

### 🤔 **Marginal Improvement**  
- GPT5 strategy slightly better (5-10% win rate improvement)
- Still below profitability threshold
- **Action**: Implement changes, continue testing

### 😐 **Inconclusive**
- Similar performance between strategies  
- Neither clearly profitable
- **Action**: Longer testing period needed, or major redesign

### 🚫 **Both Failing**
- Neither strategy achieves >35% win rate
- Mathematical impossibility to profit
- **Action**: Fundamental strategy overhaul needed

## Files Generated

**Results:**
- `sims/results/comprehensive_test_TIMESTAMP.json` - Complete results
- `sims/logs/comprehensive_test.log` - Detailed logs

**Discord Reports:**
- Progress updates every 2 hours
- Final summary with clear winner
- Actionable recommendations

## Framework Architecture  

```
📦 Advanced Testing Framework
├── 🔬 Historical Backtester
│   ├── Extract alpha signals from logs/database
│   ├── Test strategies on past data  
│   └── Calculate basic performance metrics
│
├── ⚡ Live A/B Tester  
│   ├── Real-time alpha signal monitoring
│   ├── Parallel strategy simulation
│   ├── Realistic execution modeling
│   └── Live position management
│
├── 📊 Comprehensive Analysis
│   ├── Statistical significance testing
│   ├── GPT5's demanded metrics
│   ├── Mathematical viability analysis  
│   └── Actionable recommendations
│
└── 🚨 Validation & Testing
    ├── Framework health checks
    ├── Config validation
    └── Quick test modes
```

## Troubleshooting

**"No historical data found"**
- Check logs directory exists and has .log files
- Ensure database path is correct in config
- Try shorter time period (7 days vs 30)

**"PumpPortal connection failed"**  
- Verify API key in config.yml
- Check network connectivity
- Confirm watched_wallets list is populated

**"Insufficient alpha signals"**
- Run longer test (12+ hours)
- Check if your alpha wallets are active
- Verify alpha wallet addresses are correct

**"Low confidence results"**
- Increase test duration
- Check data quality metrics  
- Ensure minimum 30 trades for statistical significance

## The Bottom Line

This framework will definitively answer:

1. **Is GPT5 right about your 30% win rate being fixable?**
2. **Which specific changes actually improve performance?**  
3. **What's the mathematical break-even point for your strategy?**
4. **Should you implement GPT5's recommendations or stick with current?**

**No more guessing. Just data-driven decisions.**

---

*Built using your existing bot infrastructure for maximum accuracy and minimal code changes.*