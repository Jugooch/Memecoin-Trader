# Phase 3: Complete Copy Trading (BUY + SELL) ✅

## What's New

Phase 2/3 script now executes **COMPLETE trades** (buy → hold → sell) to measure actual profitability!

## How It Works

### The Flow
```
1. Dev buys 5 SOL → Price starts moving
2. LaserStream detects (10-50ms) → WE BUY 0.1 SOL
3. Free websockets detect (100-500ms) → RETAIL BUYS (price spikes!)
4. We hold 1-3 seconds → Retail wave builds
5. WE SELL into retail buying pressure → PROFIT
```

### Key Features Added

✅ **Sell transactions** using Pump Portal API
✅ **Token balance fetching** from associated token account
✅ **Fast exit strategy** (configurable 1-3 second hold)
✅ **P&L calculation** (buy vs sell, profit/loss tracking)
✅ **Win rate analysis** (% of profitable trades)
✅ **Profitability verdict** (is strategy viable?)

## Test Commands

### 1. Ultra-Conservative Test (RECOMMENDED FIRST)
```bash
cd frontrun

# 10 trades, max 0.1 SOL, 2s hold (needs ~1.5 SOL)
python scripts/phase2_microtrading.py \
    --wallet data/phase2-wallet.json \
    --real-trades \
    --max-trades 10 \
    --max-position-sol 0.1 \
    --hold-time 2.0 \
    --min-buy-sol 1.0
```

**Capital needed**: 1.5 SOL (~$375)
**Duration**: 5-30 minutes (depends on Pump.fun activity)

### 2. Test Different Hold Times

**Fast Exit (1 second - aggressive scalping)**
```bash
python scripts/phase2_microtrading.py \
    --wallet data/phase2-wallet.json \
    --real-trades \
    --max-trades 10 \
    --max-position-sol 0.1 \
    --hold-time 1.0
```

**Medium Exit (2 seconds - balanced)**
```bash
python scripts/phase2_microtrading.py \
    --wallet data/phase2-wallet.json \
    --real-trades \
    --max-trades 10 \
    --max-position-sol 0.1 \
    --hold-time 2.0
```

**Slow Exit (3 seconds - let retail wave build)**
```bash
python scripts/phase2_microtrading.py \
    --wallet data/phase2-wallet.json \
    --real-trades \
    --max-trades 10 \
    --max-position-sol 0.1 \
    --hold-time 3.0
```

### 3. Target Higher Quality Opportunities
```bash
# Target 2+ SOL dev buys (fewer but better quality)
python scripts/phase2_microtrading.py \
    --wallet data/phase2-wallet.json \
    --real-trades \
    --max-trades 20 \
    --max-position-sol 0.1 \
    --hold-time 2.0 \
    --min-buy-sol 2.0
```

## What to Watch For

### During Execution

You'll see logs like:
```
=== TRADE #1 ===
our_buy_amount_sol: 0.1
dev_buy_amount_sol: 1.5

executing_buy
buy_complete: signature=5kH7... buy_slot=285623403 frontrun_success=False slot_delta=+2

holding_for_retail_wave: hold_seconds=2.0

executing_sell
selling_tokens: token_balance=1234567
sell_complete: signature=3mK9... sell_slot=285623408 sol_received=0.105

trade_pnl: profit_sol=0.005 profit_pct=5.0% verdict=✅ PROFIT

trade_completed: profit_sol=0.005 profit_pct=5.00
```

### Final Results

After all trades complete:
```
=== PHASE 2 MICRO-TRADING COMPLETE ===

phase2_results:
  total_trades: 10
  successful_trades: 10
  net_pnl_sol: 0.0342

profitability_analysis:
  winning_trades: 6
  losing_trades: 4
  win_rate: 0.6  (60%)
  avg_profit_sol: 0.012
  avg_loss_sol: -0.008
  verdict: ✅ PROFITABLE

frontrun_analysis:
  frontrun_rate: 0.2  (20% - we landed before dev)
  avg_slot_delta: +2.3  (average 2.3 slots behind dev)

=== STRATEGY VERDICT ===
✅ STRATEGY IS PROFITABLE!
Win rate: 60.0% (need 40%+ to be viable)
Net P&L: 0.034200 SOL
Average profit on wins: 0.012000 SOL
Average loss on losses: -0.008000 SOL

✅ READY TO SCALE: Test with more capital and longer time period
```

## Success Criteria

### ✅ Strategy is Viable If:
- **Win rate ≥ 40%** (at least 40% of trades are profitable)
- **Net P&L > 0** (overall profit after all trades)
- **Avg profit > Avg loss** (wins are bigger than losses)

### ⚠️ Needs Tuning If:
- Win rate 30-40% → Try different hold times
- Net positive but low win rate → Good strategy but needs optimization
- High win rate but net negative → Losses are too large, reduce slippage

### ❌ Strategy Fails If:
- Win rate < 30% → Not enough profitable trades
- Net P&L < 0 → Losing money overall
- Consider: Different hold time, different dev buy filter, or abandon strategy

## Key Metrics to Analyze

### 1. Win Rate
- **What**: % of trades that are profitable
- **Target**: 40%+ to be viable
- **Calculation**: winning_trades / total_trades

### 2. Average Profit on Wins
- **What**: How much we make when we win
- **Target**: Should be > 0.01 SOL (on 0.1 SOL position = 10%+ gain)
- **Calculation**: sum(profits) / winning_trades

### 3. Average Loss on Losses
- **What**: How much we lose when we lose
- **Target**: Should be < avg profit (smaller losses than wins)
- **Calculation**: sum(losses) / losing_trades

### 4. Net P&L
- **What**: Total profit/loss across all trades
- **Target**: Positive (any amount)
- **Calculation**: total_profit - total_fees

### 5. Slot Delta
- **What**: How many slots behind/ahead of dev we land
- **Target**: Doesn't matter much - we're beating retail, not devs
- **Note**: Negative = we frontran dev (rare), Positive = we're behind (normal)

## Cost Analysis

### Per Trade
| Cost | Amount |
|------|--------|
| Pump Portal buy fee (0.5%) | ~$0.125 |
| Pump Portal sell fee (0.5%) | ~$0.125 |
| Priority fees | ~$0.20 |
| **Total per trade** | **~$0.45** |

### For 10 Trades @ 0.1 SOL
- **Total costs**: ~$4.50
- **Break-even**: Need 0.018 SOL profit total (~$4.50)
- **Required avg gain**: 1.8% per trade

**If you achieve 5% avg gain on 60% win rate**:
- Gross profit: 0.03 SOL ($7.50)
- Net profit: 0.012 SOL ($3.00)
- ROI: 20% on 1.5 SOL capital in ~30 minutes

## Next Steps Based on Results

### If Win Rate ≥ 40% and Net Positive ✅
1. **Run larger test**: 50-100 trades to confirm consistency
2. **Test different hold times**: Find optimal (1s vs 2s vs 3s)
3. **Scale capital**: If profitable, increase to 5-10 SOL
4. **Calculate monthly projections**: Extrapolate to full-time running

### If Win Rate 30-40% ⚠️
1. **Tune hold time**: Try 1s (faster exit) or 3s (slower exit)
2. **Filter better opportunities**: Increase `--min-buy-sol` to 2.0 or 5.0
3. **Reduce slippage**: Lower `slippage_bps` in code (currently 1000 = 10%)

### If Win Rate < 30% ❌
1. **Test LaserStream vs free websockets**: Is $999/month worth it?
2. **Abandon copy trading**: Consider different strategy
3. **Focus on other alpha sources**: Wallet tracking, social signals, etc.

## Data Output

All trades are saved to: `data/phase2/trades_YYYYMMDD_HHMMSS.json`

Example:
```json
[
  {
    "timestamp": "2025-10-09T10:15:23",
    "mint": "GASneQtATBSWVNVCixbmAHuRRTGrbF3XBhTMKBtypump",
    "signature_buy": "5kH7...",
    "signature_sell": "3mK9...",
    "buy_amount_sol": 0.1,
    "sell_amount_sol": 0.105,
    "profit_sol": 0.005,
    "profit_pct": 5.0,
    "dev_tx_slot": 285623401,
    "our_tx_slot": 285623403,
    "slot_delta": 2,
    "frontrun_success": false
  }
]
```

## Important Notes

### Pump Portal Sell Behavior
- Sells ALL tokens in associated token account
- No partial sells (would need manual instruction building)
- Slippage: 10% (1000 bps) - may be too high, tune if needed

### Token Balance Fetching
- Uses RPC `getTokenAccountsByOwner` to get balance
- If balance is 0, uses estimated amount from buy response
- May have slight delay (~100ms) for RPC call

### Hold Time Optimization
The key variable to test:
- **1 second**: Fast exit, may miss retail wave peak
- **2 seconds**: Balanced, catch most of retail wave
- **3 seconds**: Slow exit, risk price dumping before we sell

**Test all three and compare results!**

## Troubleshooting

### "No tokens to sell"
- Buy may have failed silently
- Check buy transaction on Solscan
- May be slippage issue (price moved too much)

### "Sell failed: Pump Portal error"
- Token balance might be 0
- Bonding curve may have graduated (can't sell on Pump.fun)
- RPC lag in fetching balance

### All trades losing money
- Hold time may be wrong (too fast or too slow)
- Slippage too high (price dumps before we sell)
- Dev buys too small (not enough retail interest)

## Ready to Test?

1. ✅ Create wallet: `solana-keygen new --outfile frontrun/data/phase2-wallet.json`
2. ✅ Fund with 1.5 SOL
3. ✅ Run ultra-conservative test (10 trades @ 0.1 SOL, 2s hold)
4. ✅ Analyze results
5. ✅ Tune hold time based on data
6. ✅ Scale if profitable!

**Good luck!** 🚀
