# Test Execution Checklist - Pre-Phase 4

**Purpose**: Complete test validation before starting Phase 4 development
**Time Required**: ~30-45 minutes (including wallet funding wait time)
**Last Updated**: Phase 3 Complete (309 unit tests passing)

---

## Pre-Flight Checklist

### 1. Environment Setup
- [ ] Virtual environment activated: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
- [ ] All dependencies installed: `pip install -r requirements.txt && pip install -r requirements-dev.txt`
- [ ] Working directory: `cd /path/to/frontrun`

### 2. Test Wallet Funding (CRITICAL - Do This First)

**You need 2 SOL in your test wallet before running integration tests.**

#### Check Current Balance
```bash
# If you have solana CLI installed:
solana balance tests/integration/devnet-test-wallet.json --url devnet

# Otherwise, check wallet address and use block explorer
```

#### Fund Wallet from Faucet
1. **Get wallet address**:
   ```bash
   python tests/integration/generate_test_wallet.py
   # This will show your wallet address if it exists
   ```

2. **Request SOL from faucet** (choose one method):

   **Method A: Solana CLI** (recommended)
   ```bash
   solana airdrop 2 <YOUR_WALLET_ADDRESS> --url devnet
   ```

   **Method B: Web Faucet**
   - Go to: https://faucet.solana.com/
   - Select "Devnet"
   - Enter your wallet address
   - Request 2 SOL
   - Wait ~30 seconds for confirmation

3. **Verify funding**:
   ```bash
   solana balance tests/integration/devnet-test-wallet.json --url devnet
   # Should show: ~2 SOL
   ```

4. **Set Wallet ENV**
  $env:DEVNET_TEST_WALLET_PRIVATE_KEY = (Get-Content ./devnet-test-wallet.json -Raw).Trim()

**⚠️ IMPORTANT**: Faucet has rate limits. If you hit the limit, wait 24 hours or use a different IP.

---

## Test Execution Order

### Phase 1: Unit Tests (No Wallet Required)

**Expected Duration**: 5-10 seconds
**Expected Result**: 309 tests passed

```bash
# Run all unit tests with verbose output
python -m pytest tests/unit -v --tb=short

# Expected output:
# ===================== 309 passed in X.XXs =====================
```

#### Unit Test Breakdown:
- Config & Metrics: 6 tests
- Logger: 8 tests
- RPC Manager: 12 tests
- Health Monitor: 19 tests
- **Transaction Infrastructure (Phase 2)**: 45 tests
  - Transaction Builder: 9 tests
  - Transaction Signer: 9 tests
  - Transaction Submitter: 9 tests
  - Priority Fee Calculator: 9 tests
  - Wallet Manager: 9 tests
- **Trading Primitives (Phase 3)**: 154 tests
  - Pump.fun Client: 21 tests
  - Bonding Curve: 41 tests
  - Slippage Manager: 19 tests
  - PnL Calculator: 49 tests
  - Position Tracker: 24 tests

**If unit tests fail**:
- ❌ **STOP**: Do not proceed to integration tests
- Check error messages for details
- Common issues:
  - Missing dependencies → `pip install -r requirements-dev.txt`
  - Import errors → Verify you're in `/frontrun` directory
  - Database errors → Delete `tests/unit/.test_data/*.db` and retry

---

### Phase 2: Integration Tests (Requires 2 SOL)

**Expected Duration**: 30-60 seconds
**Expected Result**: 7 tests passed
**Cost**: ~0.6 SOL on first run, ~0.01 SOL on subsequent runs (wallet caching)

#### Pre-Execution Check
- [ ] Test wallet funded with 2 SOL (verified above)
- [ x ] Connected to internet (needs devnet RPC)
- [ x ] RPC endpoint responsive: https://api.devnet.solana.com

#### Run Integration Tests
```bash
# Run all integration tests
python -m pytest tests/integration -v --tb=short -s

# Expected output:
# ===================== 7 passed in XX.XXs =====================
```

#### Integration Test Breakdown:
1. **test_devnet_rpc_connection** - Verify RPC connectivity
2. **test_devnet_wallet_operations** - Create/fund test wallets
3. **test_devnet_bonding_curve_read** - Read on-chain bonding curve data
4. **test_devnet_transaction_submission** - Submit simple transfer transaction
5. **test_devnet_full_trade_flow** - Complete buy → hold → sell flow
6. **test_phase2_integration** - Transaction pipeline integration
7. **test_phase3_complete_trade_flow** - End-to-end trading integration

**If integration tests fail**:

| Error | Cause | Solution |
|-------|-------|----------|
| `Insufficient funds` | Wallet not funded | Fund wallet with 2 SOL (see above) |
| `Connection refused` | RPC endpoint down | Wait and retry, or change endpoint in config |
| `Transaction simulation failed` | Invalid instruction | Check Pump.fun program is active on devnet |
| `Timeout waiting for confirmation` | Network congestion | Increase timeout or retry |
| `Wallet cache not found` | First run | Normal - wallets will be created and cached |

---

## Post-Test Validation

### 1. Verify All Tests Passed
- [ ] **Unit tests**: 309/309 passed ✅
- [ ] **Integration tests**: 7/7 passed ✅
- [ ] **Total**: 316/316 tests passing ✅

### 2. Check Wallet Balance
```bash
# Should have ~1.4 SOL remaining after all tests
solana balance tests/integration/devnet-test-wallet.json --url devnet
```

### 3. Review Test Artifacts
- [ ] Test wallets cached: `.test_wallets_cache.json` exists
- [ ] SQLite test DBs cleaned up: `tests/unit/.test_data/` is empty
- [ ] No error logs in console output

---

## Quick Troubleshooting Guide

### "ModuleNotFoundError: No module named 'pytest'"
```bash
pip install -r requirements-dev.txt
```

### "FileNotFoundError: devnet-test-wallet.json"
```bash
python tests/integration/generate_test_wallet.py
# This creates the wallet file
```

### "InsufficientFundsError"
```bash
# Check balance
solana balance tests/integration/devnet-test-wallet.json --url devnet

# If < 2 SOL, request from faucet
solana airdrop 2 <wallet_address> --url devnet
```

### "RPC connection failed"
- Check internet connection
- Verify devnet is operational: https://status.solana.com/
- Try alternative RPC: `https://rpc.ankr.com/solana_devnet`

### "Tests hang/timeout"
- Network congestion on devnet
- Increase pytest timeout: `pytest --timeout=300`
- Try during off-peak hours

### Unit tests pass but integration tests fail
- This is expected if wallet not funded
- Fund wallet and retry
- Integration tests require on-chain interaction

---

## Success Criteria

**You are ready for Phase 4 development when**:
- ✅ All 309 unit tests passing
- ✅ All 7 integration tests passing
- ✅ No critical errors in test output
- ✅ Test wallet has sufficient balance (~1.4 SOL remaining)
- ✅ Wallet cache file exists (`.test_wallets_cache.json`)

**Total Tests**: 316 ✅
**Phase 1**: Complete (61 tests) ✅
**Phase 2**: Complete (45 tests) ✅
**Phase 3**: Complete (154 tests) ✅

---

## What's Next: Phase 4 Preview

Phase 4 will implement **Live Trading & Risk Management**:
- Real-time market data streaming
- Position sizing with Kelly Criterion
- Stop-loss / take-profit automation
- Multi-wallet rotation under fire
- Live performance monitoring

**Before starting Phase 4**:
1. Ensure all 316 tests pass ✅
2. Review Phase 1-3 documentation
3. Understand bonding curve mechanics
4. Familiarize yourself with Pump.fun protocol

---

## Quick Reference Commands

```bash
# Complete test suite (run in order)
python -m pytest tests/unit -v --tb=short           # Unit tests (309)
python -m pytest tests/integration -v --tb=short    # Integration tests (7)

# Specific test files
python -m pytest tests/unit/test_bonding_curve.py -v
python -m pytest tests/integration/test_devnet_full_trade_flow.py -v

# Run with detailed output
python -m pytest tests/unit -v --tb=long -s

# Run and stop on first failure
python -m pytest tests/unit -v -x

# Check wallet balance
solana balance tests/integration/devnet-test-wallet.json --url devnet

# Request devnet SOL
solana airdrop 2 <wallet_address> --url devnet
```

---

## Time Estimate

| Task | Duration | Notes |
|------|----------|-------|
| Environment setup | 2 min | If already configured |
| Wallet funding | 1-5 min | Depends on faucet response |
| Unit tests | 10 sec | 309 tests |
| Integration tests | 30-60 sec | 7 tests |
| Validation | 2 min | Review results |
| **Total** | **5-10 min** | Excluding faucet wait time |

**⏰ Pro Tip**: Request SOL from faucet first, then run unit tests while waiting for faucet confirmation.

---

## Support

- **Documentation**: See `docs/phases/Phase1.md`, `Phase2.md`, `Phase3.md`
- **Test Details**: See `tests/README.md`
- **Common Issues**: See individual phase docs under "Common Issues and Solutions"

Good luck with Phase 4! 🚀
