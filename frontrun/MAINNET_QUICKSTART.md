# Mainnet Deployment - Quick Start

**You're ready to go live!** Your Helius Professional subscription is active. Here's how to start Phase 1.

## Step 1: Install Dependencies (5 minutes)

```bash
cd frontrun
pip install -r requirements.txt
```

This installs:
- `grpcio` and `grpcio-tools` (Geyser/LaserStream)
- `websockets` (WebSocket streaming)
- `solders` (Solana transaction parsing)
- All other Phase 1-4 dependencies

## Step 2: Test Geyser Connection (30 seconds)

Verify your Helius LaserStream is working:

```bash
python scripts/test_geyser_connection.py --duration 30
```

**Expected output**:
```
[INFO] monitoring_started - Waiting for transactions...
[INFO] transaction_received - signature=abc123... buy_amount_sol=1.2
[INFO] transaction_received - signature=def456... buy_amount_sol=0.8
...
[INFO] test_passed - ✓ Successfully received 15 transactions
```

If you see errors, check:
- Helius API key is correct in `config/config.yml`
- Helius dashboard shows service is active
- Network connectivity is good

## Step 3: Start Phase 1 Monitoring (1 week)

Launch monitoring mode - **no trading, just data collection**:

```bash
python scripts/phase1_monitor.py --duration-hours 168
```

This will run for **1 week** (168 hours) and collect data on:
- How many frontrun opportunities per day?
- What fees are competitors paying?
- What's the typical buy size?

**Output location**: `data/monitoring/`

### Alternative: Test Run (4 hours)

Want to see if it works before committing to a full week?

```bash
python scripts/phase1_monitor.py --duration-hours 4
```

## Step 4: Monitor Progress

### Check Logs
```bash
tail -f logs/phase1_monitor.log
```

### Hourly Reports
Every hour, the script logs:
```json
{
  "elapsed_hours": 24.0,
  "total_pump_fun_txs": 2500,
  "frontrun_opportunities": 42,
  "avg_buy_size_sol": 0.72,
  "avg_competitor_fee": 28000
}
```

### Stop Early
Press `Ctrl+C` to stop gracefully. Data will be exported automatically.

## Step 5: Analyze Results (After 1 week)

Two JSON files will be exported to `data/monitoring/`:

1. **`monitoring_stats_*.json`** - Summary statistics
2. **`opportunities_*.json`** - Full list of opportunities

### Decision Matrix

| Opportunities/Day | Action |
|-------------------|--------|
| **>10** | ✅ Proceed to Phase 2 (Micro-Trading) |
| **5-10** | ⚠️ Marginal - consider tweaking detector |
| **<5** | 🛑 Low volume - re-evaluate strategy |

## What's Next?

### Phase 2: Frontrun Validation (after successful Phase 1)

**CRITICAL TEST**: Can we actually land transactions BEFORE dev buys?

#### Setup
```bash
# Create dedicated wallet for testing
solana-keygen new --outfile frontrun/data/phase2-wallet.json

# Fund wallet with 1 SOL
# Send 1 SOL to the address shown above
```

#### Step 1: Simulation Test (no capital risk)
```bash
cd frontrun
python scripts/phase2_microtrading.py --wallet data/phase2-wallet.json --max-trades 10
```

This will:
- Detect 10 dev buy opportunities via LaserStream
- **SIMULATE** trades (no real transactions)
- Track slot comparison (simulated: assumes same slot)
- Generate report with frontrun metrics

**Expected output**:
```
[INFO] trade_completed - frontrun_success=True slot_delta=0 (simulated)
```

#### Step 2: Real Trades Test (ACTUAL FRONTRUN TEST)
```bash
# WARNING: This uses real capital!
python scripts/phase2_microtrading.py --wallet data/phase2-wallet.json --real-trades --max-trades 5
```

This will:
- Execute **REAL** Pump.fun buys (0.01-0.05 SOL each)
- Submit transactions to blockchain
- Get actual slot numbers from confirmations
- Compare: our_slot vs dev_slot
- Report real frontrun success rate

**Success criteria**:
- ✅ **70%+ frontrun rate** (our_slot <= dev_slot)
- ✅ **5+ successful trades** out of 5 attempts
- ✅ **Average slot_delta ≤ 0**

**Example output**:
```json
{
  "frontrun_analysis": {
    "frontrun_wins": 4,
    "frontrun_losses": 1,
    "frontrun_rate": 0.8,
    "avg_slot_delta": -0.2,
    "verdict": "SUCCESS"
  }
}
```

#### Decision Matrix

| Frontrun Rate | Action |
|---------------|--------|
| **≥70%** | ✅ Proceed to Phase 3 (Production) |
| **50-70%** | ⚠️ Optimize latency, retry with adjusted fees |
| **<50%** | 🛑 Cannot frontrun reliably - needs architecture changes |

**Cost**: ~$0.25-0.50 in trades + fees

---

### Phase 3: Single-Wallet Production (after successful Phase 2)
- **Duration**: 2 weeks
- **Capital**: $500 (2 SOL)
- **Position Size**: 0.2 SOL (~$50/trade)
- **Goal**: Prove profitability with realistic position sizes
- **Entry Criteria**: Phase 2 frontrun rate ≥70%

### Phase 4: Horizontal Scaling (after profitable Phase 3)
- **Timeline**: Weeks 4+
- **Capital**: Scale from profits
- **Goal**: Add 1 wallet per week up to 10 wallets

## Files Created

```
frontrun/
├── services/
│   └── mempool_monitor_geyser.py    # Real Geyser client
├── scripts/
│   ├── test_geyser_connection.py    # Connection test
│   └── phase1_monitor.py            # Phase 1 monitoring
├── docs/
│   └── PHASE1_MONITORING_GUIDE.md   # Detailed Phase 1 guide
├── data/
│   └── monitoring/                  # Exported data
└── logs/
    └── phase1_monitor.log           # Execution logs
```

## Troubleshooting

### "LaserStream URL and API key required"
**Fix**: Update `config/config.yml` with your Helius credentials

### "No transactions received"
**Possible causes**:
1. Pump.fun is slow (normal on weekends)
2. Test duration too short (try 60 seconds)
3. Helius service issue (check dashboard)

### "websocket_connection_error"
**Fix**:
1. Check Helius dashboard - service active?
2. Verify API key is correct
3. Check network connectivity

### High error rate
**Fix**:
1. Check `logs/phase1_monitor.log` for details
2. Report to Helius support if LaserStream issue
3. Consider reconnection settings in code

## Cost Summary

| Phase | Duration | Capital | Infrastructure | Total |
|-------|----------|---------|----------------|-------|
| **Phase 1** | 1 week | $0 | $999/month | $999 |
| **Phase 2** | 3 days | $250 | (included) | $250 |
| **Phase 3** | 2 weeks | $500 | (included) | $500 |
| **Phase 4** | Ongoing | From profits | $999/month | Varies |

**First month total**: ~$1,750 ($999 Helius + $750 capital)

## Support

- **Documentation**: See `docs/PHASE1_MONITORING_GUIDE.md`
- **Full roadmap**: See `docs/TEMP_CURRENT_STATE.md`
- **Helius support**: https://helius.dev/support

## Ready to Launch?

```bash
# Test connection (30 seconds)
python scripts/test_geyser_connection.py --duration 30

# Start Phase 1 monitoring (1 week)
python scripts/phase1_monitor.py --duration-hours 168
```

**Good luck!** Phase 1 has zero trading risk. Take your time analyzing the data.
