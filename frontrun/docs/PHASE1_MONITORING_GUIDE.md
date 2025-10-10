# Phase 1: Monitoring Mode Guide

**Status**: Ready to Deploy
**Duration**: 1 Week (168 hours)
**Capital Required**: $0 (no trading)
**Infrastructure Cost**: $999/month (Helius Professional)

## Overview

Phase 1 is pure data collection. **No trading happens**. The bot monitors Pump.fun transactions in real-time to understand:

1. **Opportunity Volume**: How many initial dev buys happen per day?
2. **Competition**: What priority fees are competitors paying?
3. **Latency**: Can we detect and react within 100ms?
4. **Simulated PnL**: What would we have made if we traded?

## Prerequisites

### 1. Helius Professional Subscription
- ✅ Sign up at https://helius.dev
- ✅ Select Professional plan ($999/month with LaserStream)
- ✅ Get your API key and LaserStream endpoint

### 2. Configuration
Update `frontrun/config/config.yml` with your Helius credentials:

```yaml
rpc:
  endpoints:
    - url: "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY"
      websocket_url: "wss://mainnet.helius-rpc.com/?api-key=YOUR_KEY"
      laserstream_url: "https://laserstream-mainnet-ewr.helius-rpc.com"
      laserstream_api_key: "YOUR_KEY"
      secure_rpc_url: "https://YOUR_CUSTOM_URL.helius-rpc.com"
      priority: 0
      label: "helius_mainnet"
```

### 3. Install Dependencies
```bash
cd frontrun
pip install -r requirements.txt
```

## Running Phase 1

### Standard 1-Week Run
```bash
cd frontrun
python scripts/phase1_monitor.py --duration-hours 168
```

### Test Run (4 hours)
```bash
python scripts/phase1_monitor.py --duration-hours 4
```

### Custom Output Directory
```bash
python scripts/phase1_monitor.py --duration-hours 168 --output-dir data/week1
```

## What Gets Collected

### Real-Time Data
- **Timestamp**: When transaction was seen
- **Signature**: Transaction signature
- **Buy Amount**: SOL amount in the buy
- **Priority Fee**: Competitor's priority fee
- **Would Frontrun**: Binary decision (yes/no)
- **Reason**: Why we would/wouldn't frontrun

### Exported Files

After monitoring completes, two JSON files are exported:

#### 1. `monitoring_stats_YYYYMMDD_HHMMSS.json`
Summary statistics:
```json
{
  "start_time": "2025-10-09T10:00:00",
  "end_time": "2025-10-16T10:00:00",
  "total_pump_fun_txs": 45230,
  "total_opportunities": 1250,
  "opportunities_by_hour": {
    "0": 45,
    "1": 38,
    ...
  },
  "avg_buy_size_sol": 0.75,
  "max_buy_size_sol": 5.2,
  "avg_competitor_fee": 25000,
  "max_competitor_fee": 500000
}
```

#### 2. `opportunities_YYYYMMDD_HHMMSS.json`
Full list of every opportunity:
```json
[
  {
    "timestamp": "2025-10-09T10:15:23",
    "signature": "abc123...",
    "buy_amount_sol": 1.2,
    "detected_fee": 50000,
    "would_frontrun": true,
    "reason": "Large buy detected (1.20 SOL)"
  },
  ...
]
```

## Monitoring Progress

### Hourly Reports
The script logs a progress report every hour:

```
[INFO] periodic_report
  elapsed_hours: 24.0
  total_pump_fun_txs: 2500
  total_opportunities: 85
  frontrun_opportunities: 42
  avg_buy_size_sol: 0.72
  max_buy_size_sol: 3.5
  avg_competitor_fee: 28000
  reconnections: 0
  errors: 0
```

### Key Metrics to Watch

| Metric | Good | Concerning | Action |
|--------|------|------------|--------|
| **Opportunities/Day** | >10 | <5 | Review detector config |
| **Avg Competitor Fee** | <50k lamports | >200k | May need higher fees |
| **Reconnections** | 0-2 | >10 | Check network stability |
| **Errors** | 0-5 | >20 | Review logs, contact Helius |

## Stopping Early

Press `Ctrl+C` to stop monitoring gracefully. The script will:
1. Stop mempool monitoring
2. Generate final report
3. Export all collected data

## Analyzing Results

### After 1 Week, Review:

#### 1. **Opportunity Volume**
- How many opportunities per day?
- **Decision Point**:
  - If >10/day: Proceed to Phase 2
  - If 5-10/day: Marginally viable, consider tweaking detector
  - If <5/day: Re-evaluate strategy or increase monitoring duration

#### 2. **Competition Analysis**
- What's the average competitor fee?
- What's the maximum competitor fee?
- **Decision Point**:
  - If avg <50k lamports: Easy to compete
  - If avg 50-200k: Moderate competition
  - If avg >200k: Fierce competition, need aggressive fee strategy

#### 3. **Buy Size Distribution**
- What's the typical buy size?
- Are there enough large buys (>0.5 SOL)?
- **Decision Point**: Adjust `min_buy_amount_sol` in detector config

#### 4. **Time Distribution**
- What hours have most opportunities?
- **Decision Point**: Consider running bot only during peak hours

## Success Criteria for Phase 1

✅ **Proceed to Phase 2 if**:
- Opportunities per day: >10
- Average buy size: >0.5 SOL
- System stability: <5 errors, <3 reconnections per day
- Data collected: At least 100 opportunities

⚠️ **Adjust Strategy if**:
- Opportunities per day: 5-10 (marginal)
- High error rate (>20 errors/day)

🛑 **Abort if**:
- Opportunities per day: <5 (not enough volume)
- Helius connection unstable (constant reconnections/errors)
- Infrastructure costs don't justify opportunity volume

## Troubleshooting

### Connection Errors
```
[ERROR] websocket_connection_error
```
**Solution**:
- Check Helius dashboard for service status
- Verify API key is correct
- Check network connectivity

### No Opportunities Detected
```
[WARNING] no_opportunities_detected
```
**Solution**:
- Lower `min_buy_amount_sol` in `scripts/phase1_monitor.py` (line 60)
- Verify Pump.fun is active (check dexscreener.com)
- Increase monitoring duration (maybe it's a slow week)

### High Error Rate
```
[ERROR] message_processing_error
```
**Solution**:
- Check logs in `logs/phase1_monitor.log`
- Verify transaction parsing logic
- Report to Helius if LaserStream format changed

## Next Steps

After successful Phase 1 completion:

1. **Analyze exported data** - Review JSON files
2. **Calculate expected PnL** - Estimate profitability
3. **Tune detector config** - Adjust `min_buy_amount_sol` based on findings
4. **Proceed to Phase 2** - Micro-Trading with 10-20 tiny trades

See: `TEMP_CURRENT_STATE.md` for full deployment roadmap

## Files Created by Phase 1

```
data/monitoring/
├── monitoring_stats_20251009_100000.json   # Summary statistics
├── opportunities_20251009_100000.json      # Full opportunity list
└── ...

logs/
└── phase1_monitor.log                      # Full execution logs
```

## Cost Breakdown for Phase 1

| Item | Cost |
|------|------|
| Helius Professional (1 month) | $999 |
| Capital for trading | $0 (no trading yet) |
| **Total Phase 1 Cost** | **$999** |

## Questions?

Phase 1 is the safest phase - zero trading risk. Take your time analyzing the data before moving to Phase 2.

If opportunity volume looks good (>10/day) and system is stable, you're ready for Phase 2: Micro-Trading.
