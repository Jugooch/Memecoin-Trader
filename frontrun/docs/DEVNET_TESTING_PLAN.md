# Devnet Testing Plan: Maximizing Validation Before Phase 4

**Purpose**: Prove Phase 1-3 infrastructure works correctly before paying for premium RPC and deploying to mainnet.

**Timeline**: Weeks 1-3 (before Phase 4 Geyser integration)

**Key Distinction**:
- ✅ **What we CAN test on devnet**: Technical execution (code correctness, transaction building, wallet management)
- ❌ **What we CANNOT test on devnet**: Frontrunning profitability, mempool monitoring, real market conditions

---

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [What Devnet CAN Validate](#what-devnet-can-validate)
3. [What Devnet CANNOT Validate](#what-devnet-cannot-validate)
4. [Week-by-Week Testing Plan](#week-by-week-testing-plan)
5. [Success Criteria](#success-criteria)
6. [Test Execution Checklist](#test-execution-checklist)
7. [Known Limitations](#known-limitations)

---

## Testing Philosophy

**Goal**: Maximize confidence in our Phase 1-3 implementation before spending money on:
- Premium RPC with Geyser ($400-500/month starting Week 4)
- Mainnet trading capital ($300-1000 starting Week 7)

**Strategy**:
1. **Unit tests**: Validate individual components in isolation
2. **Integration tests on devnet**: Validate components work together end-to-end
3. **Stress tests on devnet**: Validate system handles edge cases and concurrent operations
4. **Manual testing on devnet**: Validate user workflows and operational procedures

**What success looks like**:
- All unit tests pass (40+ tests)
- All integration tests pass on devnet
- Complete buy→sell flow executes successfully on devnet
- Multi-wallet rotation works under load
- Position tracking persists correctly across restarts
- RPC failover works when primary endpoint fails

---

## What Devnet CAN Validate

### ✅ Phase 1: RPC Infrastructure
- **HTTP RPC calls**: `getAccountInfo`, `getBalance`, `getLatestBlockhash`, etc.
- **WebSocket subscriptions**: Account updates, slot updates (devnet has WebSocket support)
- **RPC failover**: Primary endpoint failure triggers fallback to secondary
- **Connection pooling**: Multiple concurrent RPC requests don't cause errors
- **Error handling**: Network errors, timeouts, invalid responses handled gracefully

**Test Coverage**: `tests/integration/test_devnet_rpc_connection.py`

### ✅ Phase 2: Transaction Pipeline
- **Transaction building**: Solana transactions with multiple instructions constructed correctly
- **Transaction signing**: Keypairs sign transactions without errors
- **Transaction submission**: Transactions accepted by devnet validator (may fail due to simulation, but submission works)
- **Priority fees**: Compute budget instructions added correctly
- **Wallet manager**: Multi-wallet acquisition, locking, cooldowns, balance caching

**Test Coverage**:
- `tests/integration/test_devnet_wallet_operations.py`
- `tests/integration/test_devnet_transaction_submission.py`

### ✅ Phase 3: Trading Primitives
- **Pump.fun instruction encoding**: Buy/sell instructions match expected format
- **Bonding curve calculations**: Price quotes match constant product formula exactly
- **Slippage validation**: Slippage checks accept valid trades, reject excessive slippage
- **PnL calculation**: Profit/loss calculated correctly from entry/exit prices
- **Position tracking**: Positions persist to database, survive process restarts

**Test Coverage**:
- `tests/integration/test_devnet_bonding_curve_read.py`
- `tests/integration/test_phase3_complete_trade_flow.py`

### ✅ What We Can Prove
1. **Code is correct**: Logic errors, math errors, API misuse caught and fixed
2. **Transactions execute**: Devnet accepts our transactions (proves format is correct)
3. **Error handling works**: Network failures, insufficient balance, etc. handled gracefully
4. **Concurrency is safe**: Multiple wallets can trade simultaneously without race conditions
5. **System is stable**: Can run for hours without crashes or memory leaks

---

## What Devnet CANNOT Validate

### ❌ Frontrunning Strategy
- **Mempool monitoring**: Devnet doesn't have Geyser support (requires mainnet premium RPC)
- **Transaction ordering**: Devnet doesn't have MEV bots competing for transaction ordering
- **Profitability**: Can't validate that frontrunning actually makes money
- **Competition**: No real traders to frontrun on devnet

### ❌ Real Market Conditions
- **Pump.fun tokens**: Pump.fun contracts may not exist on devnet (mainnet only)
- **Liquidity**: Devnet tokens have no real liquidity or price discovery
- **Slippage**: Real-world slippage depends on actual order book depth
- **Gas wars**: Devnet has no priority fee competition

### ❌ Production Performance
- **Latency**: Devnet RPC latency doesn't match production (usually slower)
- **Throughput**: Devnet validators may have different performance characteristics
- **Reliability**: Devnet uptime/stability is not production-grade

### ❌ What We Cannot Prove Yet
1. **Frontrunning works**: Need Geyser + mainnet to validate (Phase 4+)
2. **Strategy is profitable**: Need real capital on mainnet (Week 7+)
3. **System is fast enough**: Need production RPC latency (Phase 4+)
4. **Wallet rotation prevents detection**: Need real market observation (Week 8+)

---

## Week-by-Week Testing Plan

### Week 1: Complete Phase 3 Implementation
**Goal**: Finish any missing Phase 3 functionality and get all integration tests passing.

**Tasks**:
1. ✅ Fix `PumpFunClient.get_bonding_curve_state()` naming inconsistency
2. ✅ Add integration test for bonding curve fetching
3. Run full test suite: `pytest tests/integration -v`
4. Fix any failing tests
5. Add missing test coverage for edge cases

**Success Criteria**:
- All unit tests pass (40+ tests)
- All integration tests pass on devnet (excluding skipped tests)
- No warnings or deprecation notices in test output

**Deliverable**: Clean test suite with 100% passing rate on devnet

---

### Week 2: End-to-End Flow Testing
**Goal**: Validate complete buy→sell→PnL tracking flow on devnet with real transactions.

**Tasks**:
1. **Setup funded devnet wallets**:
   ```bash
   # Generate test wallet
   cd frontrun/tests/integration
   python generate_test_wallet.py

   # Fund at https://faucet.solana.com
   # Set environment variable
   export DEVNET_TEST_WALLET_PRIVATE_KEY=$(cat devnet-test-wallet.json)
   ```

2. **Test complete trade flow**:
   - Run: `pytest tests/integration/test_phase3_complete_trade_flow.py -v`
   - Verify buy instruction builds correctly
   - Verify sell instruction builds correctly
   - Verify position tracking persists

3. **Test wallet operations**:
   - Run: `pytest tests/integration/test_devnet_wallet_operations.py -v`
   - Verify SOL transfers execute on devnet
   - Verify balance checks work
   - Verify wallet manager handles cooldowns

4. **Test RPC resilience**:
   - Manually kill primary RPC endpoint
   - Verify failover to backup works
   - Verify transactions still submit

**Success Criteria**:
- Buy instruction successfully submitted to devnet (even if it fails simulation - that's ok, we're testing submission)
- SOL transfer between wallets completes within 30 seconds
- Position tracker correctly records open/close events
- RPC failover works when primary dies

**Deliverable**: Video recording of complete e2e flow executing on devnet

---

### Week 3: Multi-Wallet Stress Testing
**Goal**: Validate system handles concurrent operations and edge cases.

**Tasks**:
1. **Setup multiple funded wallets**:
   - Generate 5 test wallets
   - Fund each with 1 SOL from faucet
   - Configure wallet manager with all 5

2. **Concurrent wallet acquisition test**:
   ```python
   # Simulate 10 concurrent trade attempts
   tasks = [manager.get_wallet(min_sol=0.1) for _ in range(10)]
   results = await asyncio.gather(*tasks)
   # Verify: All acquisitions succeed, no deadlocks, wallets properly locked
   ```

3. **Wallet rotation validation**:
   - Execute 20 trades sequentially
   - Verify: Different wallets used, cooldowns respected, no wallet used twice in a row

4. **Position tracking stress test**:
   - Open 50 positions rapidly
   - Close 25 positions
   - Restart application
   - Verify: All positions correctly loaded from database

5. **RPC connection pool test**:
   - Make 100 concurrent RPC calls
   - Verify: No connection pool exhaustion, no errors

6. **Error injection tests**:
   - Test insufficient balance error handling
   - Test network timeout error handling
   - Test invalid transaction error handling

**Success Criteria**:
- 20+ concurrent trades with no deadlocks or race conditions
- Wallet cooldowns correctly enforced (no wallet used before cooldown expires)
- Position database never corrupted (all positions recoverable after crash)
- RPC connection pool handles 100+ concurrent requests

**Deliverable**: Stress test report showing system handles edge cases

---

## Success Criteria

### Phase 1-3 Devnet Readiness Checklist

#### Code Quality
- [ ] All unit tests pass (40+ tests)
- [ ] All integration tests pass on devnet
- [ ] No linter warnings or errors
- [ ] No deprecation warnings in dependencies
- [ ] Code coverage > 80% for core modules

#### Functional Validation
- [ ] Buy instruction builds and submits successfully
- [ ] Sell instruction builds and submits successfully
- [ ] Bonding curve calculations match expected values (within 1 lamport)
- [ ] Slippage validation correctly accepts/rejects trades
- [ ] PnL calculator produces correct profit/loss values
- [ ] Position tracker persists positions across restarts

#### System Reliability
- [ ] RPC failover works when primary endpoint fails
- [ ] Wallet manager never double-locks same wallet
- [ ] Wallet cooldowns correctly enforced
- [ ] No memory leaks after 1 hour of operation
- [ ] No crashes under stress testing

#### Operational Readiness
- [ ] Documentation is complete and accurate
- [ ] Environment variables documented and tested
- [ ] Error messages are clear and actionable
- [ ] Logging provides sufficient debugging information
- [ ] Monitoring/health checks implemented

---

## Test Execution Checklist

### Daily Development Testing
```bash
# Run unit tests (fast - completes in seconds)
cd frontrun
pytest tests/unit -v

# Run integration tests (slower - requires RPC calls)
pytest tests/integration -v

# Run specific test file
pytest tests/integration/test_devnet_wallet_operations.py -v

# Run with detailed logging
pytest tests/integration -v -s --log-cli-level=INFO
```

### Weekly Validation Testing
```bash
# Full test suite with coverage
pytest tests/ -v --cov=core --cov=clients --cov-report=html

# Open coverage report
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Pre-Phase 4 Final Validation
```bash
# Run ALL tests including skipped tests that require manual setup
pytest tests/ -v --run-skipped

# Run stress tests
pytest tests/integration -v -k "stress"

# Run end-to-end flow multiple times
for i in {1..10}; do
  echo "Run $i of 10"
  pytest tests/integration/test_phase3_complete_trade_flow.py -v
done
```

---

## Known Limitations

### Devnet Limitations We Accept
1. **No pump.fun tokens on devnet**: We test with simulated bonding curve state, will validate with real tokens on mainnet
2. **Devnet may be slower**: RPC latency and block times may not match mainnet
3. **Devnet may be unstable**: Occasional downtime or resets are normal
4. **No Geyser on devnet**: Mempool monitoring requires Phase 4 premium RPC

### What We'll Learn in Phase 4+ (Not Before)
1. **Geyser integration works**: Can't test without premium RPC ($400-500/month)
2. **Frontrunning is profitable**: Can't test without real market data
3. **Wallet rotation prevents detection**: Can't test without sustained mainnet operation
4. **System handles mainnet throughput**: Can't test devnet performance

### Acceptable Test Gaps
- Bonding curve state fetching may return `None` on devnet (no pump.fun tokens exist)
- Transaction simulation may fail (simulated tokens don't have real liquidity)
- WebSocket subscriptions may not receive updates (devnet has low activity)

**These gaps are EXPECTED and do not block Phase 4 progress.**

---

## Transition to Phase 4

### When to Start Phase 4
Start Phase 4 when ALL of these are true:
- ✅ All Phase 1-3 integration tests pass on devnet
- ✅ End-to-end flow tested successfully (Week 2)
- ✅ Multi-wallet stress tests pass (Week 3)
- ✅ No known bugs or crashes in Phase 1-3 code

### What Phase 4 Adds
- **Geyser integration**: Real-time mempool monitoring via gRPC
- **Premium RPC**: Yellowstone/Helius for production-grade performance
- **Cost**: $400-500/month for RPC services

### Phase 4 Devnet Testing
Even in Phase 4, we'll continue testing on devnet:
- Geyser integration can be tested on devnet (if provider supports it)
- Transaction submission still validated on devnet first
- Only move to mainnet after Geyser integration proven on devnet

---

## Summary

**Devnet testing proves our CODE works. It does NOT prove our STRATEGY works.**

**What we prove by end of Week 3**:
- ✅ Transactions build and submit correctly
- ✅ Bonding curve math is correct
- ✅ Wallet management works under load
- ✅ Position tracking is persistent and accurate
- ✅ System is stable and resilient

**What we still need to prove (Phase 4+)**:
- ❌ Geyser integration works
- ❌ Frontrunning is profitable
- ❌ System is fast enough for real competition
- ❌ Wallet rotation prevents detection

**Bottom line**: After Week 3, we'll have high confidence that our infrastructure is solid. We'll be ready to pay for premium RPC and start Phase 4 integration.

---

## Appendix: Running Tests

### Environment Setup
```bash
# Install dependencies
cd frontrun
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up devnet test wallet
cd tests/integration
python generate_test_wallet.py
# Fund at https://faucet.solana.com
export DEVNET_TEST_WALLET_PRIVATE_KEY=$(cat devnet-test-wallet.json)

# Verify setup
pytest tests/integration/test_devnet_rpc_connection.py -v
```

### Continuous Integration
```bash
# Run tests on every commit (GitHub Actions, GitLab CI, etc.)
pytest tests/ -v --cov=core --cov=clients --cov-report=xml

# Fail build if coverage drops below 80%
pytest tests/ --cov=core --cov=clients --cov-fail-under=80
```

### Debugging Failed Tests
```bash
# Run with verbose output and drop into debugger on failure
pytest tests/integration/test_devnet_wallet_operations.py -v -s --pdb

# Run only failed tests from last run
pytest --lf -v

# Run with increased logging
pytest tests/integration -v --log-cli-level=DEBUG
```
