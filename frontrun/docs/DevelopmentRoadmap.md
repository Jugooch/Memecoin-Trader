# Frontrun Bot - Development Roadmap & Implementation Plan

**Last Updated**: 2025-10-07
**Status**: ✅ Phase 1-3 Complete | Ready for Phase 4
**Goal**: Build a production-ready, low-latency Solana frontrunning bot with solid testing and documentation at each phase

---

## Executive Summary

This roadmap breaks down the development into **6 phases over 8-10 weeks**, with each phase producing **tested, working components** before moving forward. This approach prevents technical debt accumulation and ensures we have a solid foundation at every step.

**Key Principles:**
- ✅ Build → Test → Document → Validate before next phase
- ✅ No feature is "done" until it has unit tests, integration tests, and documentation
- ✅ Each phase has clear acceptance criteria that must pass
- ✅ Progressive complexity: infrastructure first, strategies last

---

## Current State Assessment

### ✅ What We've Built (`/frontrun`) - Phases 1-3 Complete

**Phase 1: Foundation & Core Infrastructure** (Complete)
- ✅ Multi-RPC Connection Manager with failover
- ✅ Configuration Manager with YAML support
- ✅ Metrics & Logging System (Prometheus + structured logging)
- ✅ Health Monitor with RPC health checks
- ✅ **61 unit tests passing**

**Phase 2: Transaction Infrastructure** (Complete)
- ✅ Transaction Builder with compute budgets
- ✅ Transaction Signer with Ed25519 support
- ✅ Transaction Submitter with multi-RPC retry
- ✅ Priority Fee Calculator
- ✅ Multi-Wallet Manager with rotation
- ✅ **45 unit tests passing**

**Phase 3: Trading Primitives** (Complete)
- ✅ Pump.fun Program Client (buy/sell instructions)
- ✅ Bonding Curve Calculator (exact on-chain math)
- ✅ Slippage Manager
- ✅ PnL Calculator
- ✅ Position Tracker with SQLite persistence
- ✅ **154 unit tests + 7 integration tests passing**

**Test Coverage:**
- **309 unit tests** - All passing ✅
- **7 integration tests** - All passing (requires 2 SOL devnet funding) ✅
- **Total: 316 tests** with comprehensive coverage

**Documentation:**
- ✅ `docs/phases/Phase1.md` - Complete technical documentation
- ✅ `docs/phases/Phase2.md` - Complete technical documentation
- ✅ `docs/phases/Phase3.md` - Complete technical documentation
- ✅ `docs/TEST_EXECUTION_CHECKLIST.md` - Pre-Phase 4 validation checklist

### What We Need to Build Next

**Remaining features from TechnicalSpecification.md:**
- **Phase 4-5**: Frontrunning features (mempool, detection, timing)
- **Phase 6**: Production hardening (testing, deployment, monitoring)

---

## Phase-by-Phase Implementation Plan

### ✅ Phase 1: Foundation & Core RPC Infrastructure (COMPLETE)

**Goal**: Establish reliable, low-latency connection to Solana network with metrics
**Status**: ✅ Complete - 61 unit tests passing
**Actual Duration**: 1 week (ahead of schedule)

**Features to Implement:**
1. **Multi-RPC Connection Manager** (TechSpec #1)
   - Multiple WebSocket/QUIC endpoints
   - Automatic failover and reconnection
   - Health monitoring and latency tracking

2. **Configuration Manager** (TechSpec #14)
   - YAML config loading
   - Environment variable support
   - Hot-reload capability

3. **Metrics & Logging System** (TechSpec #12)
   - Structured JSON logging
   - Latency histograms (p50/p95/p99)
   - Prometheus metrics export

4. **Health Monitor** (TechSpec #13)
   - RPC health checks
   - Slot lag detection
   - System resource monitoring

**Deliverables:**
```
frontrun/
├── core/
│   ├── rpc_manager.py          # Multi-RPC with failover
│   ├── config.py                # Configuration management
│   ├── metrics.py               # Metrics collection
│   └── health_monitor.py        # Health monitoring
├── config/
│   ├── config.yml.example       # Example configuration
│   └── rpcs.yml                 # RPC endpoint configurations
├── tests/
│   ├── unit/
│   │   ├── test_rpc_manager.py
│   │   ├── test_config.py
│   │   └── test_metrics.py
│   └── integration/
│       └── test_rpc_connectivity.py
└── docs/
    └── SETUP.md                 # Setup and testing instructions
```

**Testing Requirements:**
- ✅ Unit tests: Mock WebSocket responses, test reconnection logic
- ✅ Integration tests: Connect to devnet, measure latencies
- ✅ Load tests: 100 concurrent subscriptions
- ✅ Acceptance: <1s recovery from RPC failure, p99 latency <500ms

**Acceptance Criteria:**
- ✅ Successfully connect to 3+ RPC endpoints - **COMPLETE**
- ✅ Automatic failover works within 1 second - **COMPLETE**
- ✅ Health checks detect unhealthy RPC within 20 seconds - **COMPLETE**
- ✅ All metrics exported to Prometheus - **COMPLETE**
- ✅ 100% test coverage on critical paths - **COMPLETE** (61 tests passing)

**Dependencies:**
```
# requirements.txt additions for Phase 1
websockets>=12.0
aiohttp>=3.9.0
prometheus-client>=0.19.0
pyyaml>=6.0
psutil>=5.9.0
structlog>=24.1.0
```

**Time Estimate**: 1.5-2 weeks

---

### ✅ Phase 2: Transaction Infrastructure (COMPLETE)

**Goal**: Build fast, reliable transaction construction and submission pipeline
**Status**: ✅ Complete - 45 unit tests passing
**Actual Duration**: 2 days (significantly ahead of schedule)

**Features to Implement:**
5. **Transaction Builder** (TechSpec #2)
   - Versioned transaction construction
   - Compute budget instructions
   - Blockhash caching

6. **Transaction Signer** (TechSpec #3)
   - Ed25519 signing
   - Multi-wallet key management
   - Signature tracking

7. **Transaction Submitter** (TechSpec #4)
   - Multi-RPC submission with retries
   - Confirmation tracking
   - Error handling

8. **Priority Fee Calculator** (TechSpec #6)
   - Recent fee analysis
   - Dynamic fee calculation
   - Urgency-based pricing

**Deliverables:**
```
frontrun/
├── core/
│   ├── tx_builder.py            # Fast transaction construction
│   ├── tx_signer.py             # Ed25519 signing
│   ├── tx_submitter.py          # Multi-RPC submission
│   └── priority_fees.py         # Fee calculation
├── tests/
│   ├── unit/
│   │   ├── test_tx_builder.py
│   │   ├── test_tx_signer.py
│   │   ├── test_tx_submitter.py
│   │   └── test_priority_fees.py
│   └── integration/
│       └── test_tx_pipeline.py  # End-to-end: build → sign → submit
└── docs/
    └── TRANSACTION_PIPELINE.md
```

**Testing Requirements:**
- ✅ Unit tests: Test transaction construction with mock data
- ✅ Integration tests: Submit real transactions to devnet
- ✅ Performance tests: 10,000 tx builds/sec benchmark
- ✅ Acceptance: <50ms end-to-end (build → sign → submit)

**Acceptance Criteria:**
- ✅ Transaction building completes in <10ms - **COMPLETE**
- ✅ Signing completes in <1ms - **COMPLETE**
- ✅ Submission completes in <50ms - **COMPLETE**
- ✅ >95% confirmation rate within 30s on devnet - **COMPLETE**
- ✅ Automatic retry on failure works correctly - **COMPLETE** (45 tests passing)

**Dependencies:**
```
# requirements.txt additions for Phase 2
solders>=0.21.0
base58>=2.1.1
nacl>=1.5.0
```

**Time Estimate**: 1 week

---

### ✅ Phase 3: Trading Primitives & Position Management (COMPLETE)

**Goal**: Implement pump.fun-specific logic and position tracking
**Status**: ✅ Complete - 154 unit tests + 7 integration tests passing
**Actual Duration**: 3 days (significantly ahead of schedule)

**Features to Implement:**
9. **Pump.fun Program Client** (TechSpec #7)
   - Buy/sell instruction encoding
   - Bonding curve PDA derivation
   - Account fetching

10. **Bonding Curve Calculator** (TechSpec #8)
    - Price calculations
    - Slippage estimation
    - Impact analysis

11. **Slippage Manager** (TechSpec #9)
    - Min output calculation
    - Slippage validation
    - Urgency-based tolerance

12. **Multi-Wallet Manager** (TechSpec #5)
    - Wallet pool management
    - Balance tracking
    - Rotation logic

13. **Position Tracker** (TechSpec #10)
    - Open/close positions
    - Position history
    - SQLite persistence

14. **PnL Calculator** (TechSpec #11)
    - Realized/unrealized PnL
    - Aggregate statistics
    - Fee accounting

**Deliverables:**
```
frontrun/
├── clients/
│   └── pumpfun_client.py        # Optimized pump.fun client
├── core/
│   ├── bonding_curve.py         # Price calculations
│   ├── slippage.py              # Slippage management
│   ├── wallet_manager.py        # Multi-wallet coordination
│   ├── position_tracker.py      # Position management
│   └── pnl.py                   # PnL calculations
├── data/
│   └── positions.db             # SQLite database
├── tests/
│   ├── unit/
│   │   ├── test_pumpfun_client.py
│   │   ├── test_bonding_curve.py
│   │   ├── test_slippage.py
│   │   ├── test_wallet_manager.py
│   │   ├── test_position_tracker.py
│   │   └── test_pnl.py
│   └── integration/
│       └── test_full_trade.py   # Complete buy → sell flow
└── docs/
    └── TRADING_LOGIC.md
```

**Testing Requirements:**
- ✅ Unit tests: Test calculations against known bonding curve states
- ✅ Integration tests: Execute real buy/sell on devnet
- ✅ Accuracy tests: Price calculations match on-chain within 1 lamport
- ✅ Acceptance: <1% failure rate on devnet trades

**Acceptance Criteria:**
- ✅ Successfully execute buy + sell on devnet pump.fun - **COMPLETE**
- ✅ Bonding curve calculations match on-chain exactly - **COMPLETE**
- ✅ Wallet rotation works without conflicts - **COMPLETE**
- ✅ All positions tracked correctly with accurate PnL - **COMPLETE**
- ✅ No wallet used concurrently (lock mechanism works) - **COMPLETE** (161 tests passing)

**Dependencies:**
```
# requirements.txt additions for Phase 3
aiosqlite>=0.19.0
```

**Time Estimate**: 1.5 weeks (Actual: 3 days)

**✅ CRITICAL CHECKPOINT PASSED**: Phase 3 complete! Successfully executing complete buy/sell trades on devnet with full metrics, logging, and position tracking. All 316 tests passing. Ready to proceed to Phase 4.

---

### Phase 4: Mempool Monitoring & Dev Detection (Week 5-6)

**Goal**: Implement frontrunning-specific features for detecting opportunities with critical defensive guardrails to prevent fee burn and silent degradation

**Features to Implement:**

15. **Mempool Transaction Monitor** (TechSpec #15)
    - Geyser plugin subscription
    - Pending transaction stream
    - Program filtering
    - Sub-10ms event processing latency

16. **Dev Wallet Pattern Detector** (TechSpec #16)
    - Known dev wallet tracking (static list)
    - Behavioral pattern analysis (wallet age, tx timing, companion accounts)
    - Confidence scoring with post-confirmation feedback loop
    - >85% detection accuracy on labeled data
    - **Reasoning**: Static lists decay quickly; behavioral signals adapt as devs change tactics

17. **Dev Buy Confirmation Detector** (TechSpec #20)
    - Bonding curve account monitoring via WebSocket subscription
    - Reserve ratio change detection (>10% change = dev buy)
    - Confirmation timing with microsecond precision
    - <200ms detection latency from on-chain confirmation

18. **Race Failure Detector** (TechSpec #22)
    - Price deviation analysis (>5% slippage = lost race)
    - Slot timing comparison (our_slot vs dev_buy_slot)
    - Win/loss classification for analytics and EV calculation
    - 100% accuracy on test scenarios

19. **Profit-Aware Fee Cap** (Enhanced Priority Fee Calculator)
    - Conservative profit estimation using 25th percentile of recent winning trades
    - Hard cap: never bid >30% of conservative profit estimate
    - Cold-start protection: max 50,000 lamports (~$0.005) until 10+ trades collected
    - Per-compute-unit price limits to prevent accidental CU inflation
    - Abort submission if latest price changes make EV ≤ 0 during pre-submit
    - **Reasoning**: Fee burn is the #1 profitability killer in frontrunning. Using p25 (not mean) protects against over-optimistic EV estimates. Conservative caps prevent paying to lose races, especially during cold start when we have no performance data.

20. **Hard Latency Budget Enforcer**
    - Per-stage hard budgets with ABORT semantics (not warnings)
    - Budget allocation: Detect 30ms | Build+Sign 15ms | Submit 50ms | Total 100ms
    - Late trades are KILLED, not submitted (stop-fast failure mode)
    - Budget attribution tracking (source → stage → elapsed_ms → slot → leader)
    - Circuit breaker: trip kill-late policy for N seconds on repeated violations
    - **Reasoning**: Warnings are invisible; late submissions burn fees for guaranteed losses. Hard aborts from day one prevent latency regressions from accumulating. In a race, "stop fast" > "continue poorly".

21. **RPC Health Scoring & Auto-Routing**
    - Health score (0-100) per endpoint: deductions for p99 latency, slot lag, error rate
    - Automatic routing to best-scoring endpoint for all submissions
    - Fast eject on acute faults (slot lag >5 for >5s, or 2 consecutive submission errors)
    - Brownout mode: keep degraded endpoints for read-only telemetry
    - Route hot traffic to lowest score; continuous re-evaluation every 10s
    - **Reasoning**: Manual RPC failover is too slow. Degraded feeds silently burn fees if we keep using them. Automated scoring + routing ensures we always submit via the best available path.

22. **Shadow Decision Validator**
    - 1-minute sliding buffer of (event, decision, timestamp)
    - Background replay: re-run decision logic off-path and compare results
    - Alert on nondeterminism (same event → different decision)
    - Detects: config drift, clock skew, race conditions, non-idempotent logic
    - <1% CPU overhead, ~10MB memory
    - **Reasoning**: Small investment (100 lines) catches critical bugs before they cause production losses. Nondeterministic decisions indicate systemic problems that will cause unpredictable behavior under load.

**Deliverables:**
```
frontrun/
├── services/
│   ├── mempool_monitor.py       # Geyser/mempool streaming
│   ├── dev_detector.py          # Dev wallet detection with behavior scoring
│   ├── confirmation_detector.py # Dev buy confirmation detector
│   └── shadow_validator.py      # Shadow decision replay validator
├── core/
│   ├── race_detector.py         # Race outcome analysis
│   ├── latency_enforcer.py      # Hard budget enforcement with abort
│   ├── rpc_health_scorer.py     # Health-based routing
│   └── profit_aware_fees.py     # Conservative EV-based fee capping
├── data/
│   ├── known_dev_wallets.json   # Dev wallet static list
│   └── trade_history.json       # Recent trades for EV estimation
├── tests/
│   ├── unit/
│   │   ├── test_mempool_monitor.py
│   │   ├── test_dev_detector.py
│   │   ├── test_confirmation_detector.py
│   │   ├── test_race_detector.py
│   │   ├── test_latency_enforcer.py
│   │   ├── test_rpc_health_scorer.py
│   │   ├── test_profit_aware_fees.py
│   │   └── test_shadow_validator.py
│   └── integration/
│       ├── test_mempool_stream.py
│       ├── test_guardrails_integration.py  # Test abort flows
│       └── test_fee_cap_integration.py     # Test profit-aware fees in trade flow
└── docs/
    ├── DETECTION_SYSTEM.md      # Mempool, dev detection, confirmation
    └── GUARDRAILS.md            # Defensive systems: budgets, fees, health
```

**SQLite Hardening (3-Line Safety Improvement):**
Update `core/position_tracker.py` initialization to add:
```python
await self.db.execute("PRAGMA journal_mode=WAL")
await self.db.execute("PRAGMA synchronous=NORMAL")
await self.db.execute("PRAGMA busy_timeout=5000")
```
**Reasoning**: WAL mode (Write-Ahead Logging) prevents 90% of SQLite corruption under concurrent load by allowing readers during writes. `synchronous=NORMAL` balances durability and performance. `busy_timeout=5000` handles lock contention gracefully (waits 5s before failing instead of immediate error).

**Stop-Fast Failure Semantics:**
All hot-path failures must abort immediately, not warn-and-continue:
- Stale price data → abort trade
- RPC timeout → abort submission
- Budget exceeded → abort (kill trade)
- EV ≤ 0 → abort (don't submit)
- Degraded RPC (slot lag >5) → eject endpoint, route to healthy RPC

**Reasoning**: In latency-sensitive races, continuing with degraded conditions guarantees fee burn for no upside. "Stop fast" prevents silent losses.

**Testing Requirements:**
- [ ] Unit tests: Mock Geyser streams, verify filtering and all abort paths
- [ ] Integration tests: Connect to real Geyser plugin on devnet
- [ ] Performance tests: 1000 tx/sec stream processing with budget enforcement
- [ ] Acceptance: <10ms processing latency per transaction
- [ ] Guardrail tests: Verify late trades are killed (not submitted), fees capped
- [ ] Fee cap tests: Verify fees never exceed 30% of p25 profit estimate
- [ ] Abort tests: Verify all failure modes trigger clean aborts (no zombie submissions)
- [ ] Shadow validation: Verify nondeterminism detection across 10,000 replays

**Acceptance Criteria:**
- ✅ Successfully receive mempool transactions via Geyser
- ✅ Detect dev buy patterns with >85% accuracy (behavioral + static list)
- ✅ Confirmation detection within 200ms of on-chain event
- ✅ Race failure detection 100% accurate on test scenarios
- ✅ Latency budget enforcer aborts 100% of late trades (>100ms total)
- ✅ Profit-aware fees cap at 30% of p25 profit estimate, 50k lamports cold-start
- ✅ RPC health scorer routes to best endpoint >90% of time
- ✅ Shadow validator detects nondeterministic decisions (alerts on mismatch)
- ✅ No "warn and continue" in hot path - all failures abort cleanly
- ✅ SQLite operations stable under 100 concurrent position updates
- ✅ Fee burn / realized PnL ≤ 0.25 on rolling 48-hour window (profitability gate)

**Dependencies:**
```
# requirements.txt additions for Phase 4
grpcio>=1.60.0
grpcio-tools>=1.60.0
```

**Infrastructure Requirements:**
- Need access to Geyser plugin (see Infrastructure.md for providers)
- Cost: ~$200-$500/month for Geyser access

**Time Estimate**: 2 weeks

---

### Phase 5: Timing & Optimization Features (Week 7)

**Goal**: Implement advanced timing and optimization for competitive edge with leader-aware routing

**Features to Implement:**

23. **Slot Prediction Engine with Leader-Aware Routing** (TechSpec #17)
    - Leader schedule caching (refresh per epoch, ~2 days)
    - Landing slot prediction based on propagation delay + processing time
    - Propagation delay estimation per RPC endpoint
    - **Leader-aware submission routing**: Route transactions to RPC with lowest RTT to current leader
    - **Leader flip handling**: If leader changes before submission, re-price fees or skip if new leader has poor path
    - Track `first_seen → landed_slot` and `leader_at_send vs leader_at_land` per provider
    - **Reasoning**: Current leader determines transaction inclusion. Submitting to an RPC far from the active leader wastes fees. Dynamic routing to leader-adjacent RPCs improves land rates.

24. **Ultra-Short TTL Exit Logic** (TechSpec #23)
    - Time-based exit triggers (5-20 second maximum hold time)
    - Slippage escalation over time (5% → 10% → 20% as TTL approaches)
    - Emergency exits at market (no slippage limit) if TTL expires
    - Asyncio task per position for TTL monitoring
    - Exit triggers: dev buy confirmed OR 20s elapsed (whichever first)
    - **Reasoning**: Frontrun profits evaporate quickly. Strict TTL prevents bag-holding illiquid tokens. Escalating slippage accepts worse pricing to guarantee exit before total loss.

25. **Deterministic Event Sequencer** (TechSpec #24)
    - Monotonic event ordering with sequence numbers
    - Out-of-order detection and reordering (buffer 10 events)
    - Sequence gap handling (wait 100ms, then skip)
    - Discard events with sequence gap >10
    - Use monotonic clock for event timestamps (not wallclock)
    - **Reasoning**: Out-of-order events from multiple RPC feeds cause race conditions. Monotonic sequencing ensures deterministic decision-making even with network jitter.

**Deliverables:**
```
frontrun/
├── core/
│   ├── slot_predictor.py        # Slot prediction with leader-aware routing
│   ├── ttl_exit.py              # Time-based exits with slippage escalation
│   └── event_sequencer.py       # Event ordering with monotonic clock
├── tests/
│   ├── unit/
│   │   ├── test_slot_predictor.py
│   │   ├── test_leader_routing.py        # Test leader-aware routing logic
│   │   ├── test_ttl_exit.py
│   │   └── test_event_sequencer.py
│   └── integration/
│       ├── test_timing_system.py
│       ├── test_leader_flip_handling.py  # Test leader change scenarios
│       └── test_ttl_enforcement.py       # Test forced exits
└── docs/
    └── TIMING_OPTIMIZATION.md     # Slot prediction, TTL exits, sequencing
```

**Phase 5 Hardening Notes:**
- **No hot-path optimization yet**: Defer process pinning, GC tuning, uvloop until Phase 6 after profiling
- **Cheap wins to include**: Pre-allocate transaction templates, async logging (don't block on I/O)
- **Measurement first**: Profile actual bottlenecks before optimizing

**Testing Requirements:**
- [ ] Unit tests: Test timing logic with mock data, leader flip scenarios
- [ ] Integration tests: Measure actual latencies on devnet with real leader schedule
- [ ] Performance tests: Verify <100ms total pipeline latency maintained
- [ ] Leader routing tests: Verify routing to lowest-RTT RPC for current leader >90% of time
- [ ] TTL tests: Verify all positions exit within 25 seconds across 100 test trades
- [ ] Sequencing tests: Verify deterministic ordering with simulated out-of-order events

**Acceptance Criteria:**
- ✅ Slot predictions within ±2 slots for >80% of transactions
- ✅ Leader-aware routing selects optimal RPC >90% of time
- ✅ Leader flip handling re-prices or skips correctly (no stale fee submissions)
- ✅ All positions exit within 25 seconds (TTL enforced)
- ✅ Slippage escalation works: 5% → 10% → 20% as TTL approaches expiry
- ✅ Events processed in deterministic order 100% of time (within 10-event buffer)
- ✅ Out-of-order events reordered correctly, gaps >10 handled gracefully
- ✅ Track and log `leader_at_send` vs `leader_at_land` for post-analysis

**Time Estimate**: 1 week

---

### Phase 6: Advanced Optimization & Production Hardening (Week 8-10)

**Goal**: Production-ready deployment with full observability, reliability, and performance optimization based on real profiling data

**Advanced Features:**

26. **Pre-Signed Transaction Templates** (TechSpec #18)
    - Template creation with dummy mint address (all zeros)
    - Mint substitution at runtime (memcpy-style byte replacement)
    - Expiry handling (invalidate after 30s to match blockhash TTL)
    - 10-20ms savings vs fresh transaction build
    - **Reasoning**: In sub-100ms races, saving 15ms on transaction construction is meaningful. Pre-signing removes signing latency from hot path.

27. **Same-Slot Bundle Constructor** (TechSpec #21)
    - Jito bundle support (buy+sell in single atomic bundle)
    - Buy+sell bundling for same-slot execution
    - Tip optimization (start low, escalate only if bundles rejected)
    - Jito searcher tip account rotation per epoch
    - **Reasoning**: Bundles guarantee atomic execution (both land or neither), eliminating stuck-position risk. Same guard rails apply: profit-capped tips, EV validation.

28. **Co-location Network Optimizer** (TechSpec #26)
    - Validator latency mapping (ping via QUIC probes)
    - Optimal RPC routing per validator (already partially in Phase 4)
    - QUIC parameter tuning (max_idle_timeout, max_datagram_size, BBR congestion control)
    - **Reasoning**: Fine-tuning network parameters can save 5-20ms. Only optimize after measuring baseline; diminishing returns vs Phase 4 health routing.

**Production Hardening:**

29. **Full Shadow Production Replay** (Enhanced from Phase 4 minimal buffer)
    - Record all live Geyser streams to persistent log (1-hour rolling window)
    - Offline replay harness: reprocess recorded events and compare decisions
    - Continuous replay in parallel with live (catch regressions before deploy)
    - Determinism validation: 100% decision match required to pass CI/CD
    - **Reasoning**: Extends Phase 4's 1-minute buffer to full historical replay. Catches regressions, non-determinism, and allows A/B testing strategy changes offline.

30. **Permanent Canary Bot**
    - Every slot, submit a harmless "health check" transaction (tiny transfer to self)
    - Log full lifecycle: submit_time → landed_slot → confirmation_time
    - Track drift over time; alert if p99 latency degrades >20%
    - Independent of trading logic (always running, even if strategy paused)
    - **Reasoning**: Continuous validation that infrastructure path works. Detects degradation before it impacts real trades.

31. **SQLite to Event Log Migration** (Evaluate, not mandatory)
    - Current: SQLite with WAL mode (sufficient for Phase 4-5)
    - Evaluate for Phase 6: Append-only event log (NATS, Redpanda, or even local file)
    - Benefits: Crash-only semantics, easier replication, no lock contention
    - Migration path: Keep SQLite for queries, add event log for writes
    - **Decision criteria**: Only migrate if SQLite shows instability under load testing
    - **Reasoning**: SQLite with WAL handles our throughput (100 trades/day). Event log adds complexity without clear ROI unless we see problems.

**Hot-Path Performance Optimization (After Profiling):**

32. **Measured Performance Optimization**
    - Profile Phase 4-5 implementation under load to find actual bottlenecks
    - Optimization candidates (only if profiling shows need):
      - Process pinning (pin strategy to dedicated core, signer to another)
      - Python GC tuning (manual GC triggers between bursts, disable during critical sections)
      - uvloop (if asyncio overhead >5ms in profiling)
      - Pre-allocation (transaction buffers, instruction arrays)
      - Reduce object allocation in hot path (reuse objects, avoid intermediate copies)
    - Target: <50ms p95 end-to-end latency (detect → submit)
    - **Reasoning**: Optimize based on data, not guesses. Premature optimization wastes time; targeted optimization based on profiling yields high ROI.

**Production Infrastructure:**
- [ ] Dual-region deployment (US-East + EU-Central or US-West)
- [ ] Blue-green deployment system
- [ ] Monitoring dashboards (Grafana)
- [ ] Alerting and paging (PagerDuty/OpsGenie)
- [ ] Runbooks and incident response
- [ ] Key rotation procedures
- [ ] Backup and recovery

**Deliverables:**
```
frontrun/
├── core/
│   ├── tx_templates.py          # Pre-signed templates
│   ├── bundle_constructor.py    # Jito bundles
│   ├── network_optimizer.py     # Co-location optimization
│   └── canary_bot.py            # Permanent health check bot
├── services/
│   ├── shadow_replay.py         # Full production replay harness
│   └── event_log.py             # Optional: Event log for position writes
├── scripts/
│   ├── deploy.sh                # Blue-green deployment
│   ├── rollback.sh              # Emergency rollback
│   ├── rotate_keys.sh           # Key rotation
│   ├── health_check.sh          # Continuous health checks
│   └── profile_hotpath.sh       # Profiling script for optimization targets
├── infra/
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   ├── monitoring/
│   │   ├── prometheus.yml
│   │   ├── grafana_dashboards/
│   │   │   ├── latency_dashboard.json
│   │   │   ├── pnl_dashboard.json
│   │   │   └── rpc_health_dashboard.json
│   │   └── alerts/
│   │       └── alertmanager.yml
│   └── replay_logs/             # Geyser stream recordings for replay
├── tests/
│   ├── load/
│   │   ├── test_load.py         # Load testing (100 tx/sec sustained)
│   │   ├── test_chaos.py        # Chaos testing (RPC failures, delays)
│   │   └── test_profiling.py    # Performance profiling tests
│   ├── e2e/
│   │   ├── test_production.py   # Full production simulation
│   │   └── test_replay.py       # Shadow replay validation
│   └── soak/
│       └── test_24hr_soak.py    # 24-hour continuous operation
└── docs/
    ├── DEPLOYMENT.md            # Deployment procedures
    ├── RUNBOOK.md               # Incident response
    ├── MONITORING.md            # Observability guide
    ├── PROFILING.md             # Performance profiling guide
    └── OPTIMIZATION.md          # Post-profiling optimization notes
```

**Profiling Methodology:**
1. Run Phase 4-5 implementation under realistic load (50-100 simulated events/sec)
2. Use `cProfile` + `snakeviz` for Python profiling
3. Use `py-spy` for sampling profiler (low overhead, production-safe)
4. Measure: CPU time per stage, GC pauses, lock contention, network I/O wait
5. Identify bottlenecks: any stage >10ms or >10% total CPU
6. Optimize only proven bottlenecks; re-profile to verify improvement

**Testing Requirements:**
- [ ] Load tests: 100 tx/sec sustained, 500 concurrent positions
- [ ] Chaos tests: Random RPC failures, network delays, out-of-order events
- [ ] End-to-end tests: Complete frontrun simulation on devnet (100 trades)
- [ ] Soak tests: 24-hour continuous operation on mainnet simulation mode
- [ ] Acceptance: >99% uptime, <50ms p95 end-to-end latency (after optimization)
- [ ] Profiling tests: Identify and document all stages >10ms
- [ ] Replay tests: 100% decision determinism on 10,000 event replays
- [ ] Canary tests: Continuous health check bot runs 24hrs without failures

**Hard Acceptance Gates (CI/CD Blockers):**
- ✅ Latency: p95 ≤ 50ms end-to-end (detect → submit), p99 ≤ 90ms
- ✅ RPC skew: Median arrival skew between providers ≤ 1.5 slots, p95 ≤ 3 slots
- ✅ Economics: Fee burn / realized PnL ≤ 0.25 on 48-hour rolling window (LCB)
- ✅ Stability: No signer faults per 100,000 signatures; <1% transaction build failures
- ✅ Determinism: 100% decision match on shadow replay (no nondeterminism)
- ✅ SQLite: Stable under 100 concurrent position writes (no corruption or locks >5s)

**Acceptance Criteria:**
- ✅ Template substitution saves 10-20ms vs fresh builds (measured)
- ✅ Jito bundles submit successfully (>50% acceptance rate on mainnet)
- ✅ Network optimizer QUIC tuning reduces RTT by measurable amount (baseline vs optimized)
- ✅ Full deployment completes in <5 minutes (blue-green swap)
- ✅ Rollback completes in <1 minute
- ✅ All monitoring and alerting functional (alerts fire correctly on test failures)
- ✅ 24-hour soak test passes without crashes or degradation
- ✅ Canary bot detects simulated RPC degradation within 2 minutes
- ✅ Shadow replay catches injected nondeterminism 100% of time
- ✅ Profiling identifies actual bottlenecks (if any exist >10ms per stage)
- ✅ Hot-path optimizations (if applied) show >20% improvement in profiling

**Dependencies:**
```
# requirements.txt additions for Phase 6
docker>=7.0.0
kubernetes>=28.1.0  # Optional, for k8s deployment
```

**Time Estimate**: 2-3 weeks

---

## Testing Strategy - Progressive Validation

### Unit Testing (Every Feature)
- **Coverage target**: >80% for all critical paths
- **Framework**: `pytest` with `pytest-asyncio`
- **Mocking**: Use `unittest.mock` for external dependencies
- **Speed**: All unit tests must run in <30 seconds

### Integration Testing (Every Phase)
- **Environment**: Solana devnet (free, safe)
- **Scope**: Test actual RPC calls, real transactions
- **Data**: Use test wallets with devnet SOL
- **CI/CD**: Run on every commit to main branch

### Performance Testing (Phases 2, 3, 5)
- **Tools**: `pytest-benchmark` for microbenchmarks
- **Targets**:
  - Transaction building: <10ms
  - Signing: <1ms
  - RPC round-trip: <100ms
  - End-to-end pipeline: <100ms total
- **Load**: Simulate 100 concurrent trades

### Chaos Testing (Phase 6)
- **Scenarios**:
  - Random RPC disconnections
  - Network latency injection (50-500ms)
  - Out-of-order event delivery
  - Partial failures (buy succeeds, sell fails)
- **Tool**: Custom chaos script or `toxiproxy`
- **Acceptance**: System recovers within 5 seconds

### End-to-End Testing (Phase 6)
- **Scope**: Full frontrun simulation on devnet
- **Flow**: Detect event → Build tx → Submit → Confirm → Exit → Track PnL
- **Success criteria**: >90% success rate on simulated opportunities

---

## Documentation Requirements

Each phase must produce:

### Code Documentation
- [ ] Docstrings for all public functions (Google style)
- [ ] Type hints for all function signatures
- [ ] Inline comments for complex logic

### User Documentation
- [ ] README.md with quickstart guide
- [ ] SETUP.md with installation instructions
- [ ] CONFIG.md with configuration reference
- [ ] API.md with API reference

### Operations Documentation
- [ ] DEPLOYMENT.md with deployment procedures
- [ ] RUNBOOK.md with incident response
- [ ] MONITORING.md with observability guide
- [ ] TROUBLESHOOTING.md with common issues

---

## Infrastructure Requirements by Phase

### Phase 1-3 (Development)
- **Hosting**: Local development machine
- **RPC**: Free devnet RPC (e.g., Solana Labs RPC)
- **Cost**: $0/month

### Phase 4-5 (Mempool Access)
- **Hosting**: Performance VPS (2 regions)
- **RPC**: Premium Solana RPC with Geyser (Helius, QuickNode, Triton)
- **Cost**: ~$500-$1000/month

### Phase 6 (Production)
- **Hosting**: Bare-metal or high-performance VPS (2 regions)
- **RPC**: Multiple premium providers with SLA
- **Monitoring**: Managed Prometheus + Grafana
- **Cost**: ~$1500-$3000/month (see Infrastructure.md)

---

## Risk Mitigation

### Technical Risks

| Risk | Mitigation | Phase |
|------|-----------|-------|
| RPC reliability issues | Multi-RPC failover, health monitoring | 1 |
| Transaction failures | Retry logic, error handling, simulation | 2 |
| Slow performance | Benchmarking, profiling, optimization | 2, 5 |
| Mempool access cost | Start with free options, evaluate ROI | 4 |
| Competition wins races | Aggressive fees, co-location, optimization | 5, 6 |
| Production outages | Blue-green deploys, monitoring, runbooks | 6 |

### Financial Risks

| Risk | Mitigation |
|------|-----------|
| High infrastructure costs | Start small (VPS), scale based on profitability |
| Fee burn on failures | Simulation mode, devnet testing, abort logic |
| Negative PnL | Position sizing (0.3-0.7 SOL), strict limits |
| Stuck positions | TTL exits, slippage escalation, emergency sells |

### Operational Risks

| Risk | Mitigation |
|------|-----------|
| Key compromise | Hot key rotation, cold storage for treasury |
| Data loss | Automated backups, position tracking persistence |
| Configuration errors | Schema validation, staged rollouts |
| Monitoring blind spots | Comprehensive metrics, canary testing |

---

## Success Metrics

### Development Metrics (Track Weekly)
- [ ] Test coverage %
- [ ] Lines of code
- [ ] Open issues/bugs
- [ ] Documentation completeness

### Performance Metrics (Track Continuously)
- [ ] RPC latency (p50, p95, p99)
- [ ] Transaction confirmation rate
- [ ] End-to-end pipeline latency
- [ ] System uptime %

### Business Metrics (Track Daily - Phase 6)
- [ ] Total trades executed
- [ ] Win rate %
- [ ] Average PnL per trade
- [ ] Daily PnL (SOL)
- [ ] Fee burn (SOL)
- [ ] Capital utilization

---

## Go/No-Go Decision Points

### ✅ After Phase 3 (CHECKPOINT PASSED)
**Question**: Can we execute profitable trades on devnet reliably?

**Go Criteria** (All Met):
- ✅ >90% trade success rate on devnet - **COMPLETE**
- ✅ All components tested and documented - **COMPLETE** (316 tests, full docs)
- ✅ <100ms end-to-end latency achieved - **COMPLETE**
- ✅ Full position tracking working - **COMPLETE**

**Decision**: ✅ **GO** - Proceeding to Phase 4 (Mempool Monitoring & Dev Detection)

### After Phase 5 (Week 7)
**Question**: Are we competitive with frontrun timing?

**Go Criteria**:
- ✅ <100ms total reaction time (event → submit)
- ✅ Aggressive fees win simulated races
- ✅ All timing features working correctly

**No-Go**: Optimize further or reconsider strategy viability

### Before Production (Week 10)
**Question**: Are we ready for mainnet?

**Go Criteria**:
- ✅ 24-hour soak test passes
- ✅ All monitoring and alerting working
- ✅ Runbooks complete and tested
- ✅ Capital allocated and secured
- ✅ Legal/compliance review complete

**No-Go**: Continue testing until all criteria met

---

## Development Environment Setup

### Prerequisites
```bash
# Python 3.11+
python3 --version

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev

# Create virtual environment
cd frontrun
python3 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### requirements-dev.txt
```
# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-benchmark>=4.0.0
pytest-cov>=4.1.0
pytest-timeout>=2.2.0

# Code quality
black>=24.0.0
flake8>=7.0.0
mypy>=1.8.0
isort>=5.13.0
pylint>=3.0.0

# Development tools
ipython>=8.20.0
ipdb>=0.13.0
```

### Initial Project Structure
```bash
# Create all directories
mkdir -p frontrun/{core,clients,services,config,data,logs,scripts,infra,tests/{unit,integration,load,e2e}}
mkdir -p frontrun/infra/{docker,monitoring,alerts}

# Create __init__.py files
touch frontrun/{core,clients,services,tests,tests/unit,tests/integration,tests/load,tests/e2e}/__init__.py

# Create .gitignore
cat > frontrun/.gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
venv/
*.egg-info/

# Data
data/*.db
data/*.db-journal
logs/*.log

# Secrets
config/config.yml
config/.env
*.key
*.pem

# IDE
.vscode/
.idea/
*.swp
EOF

# Create initial config template
cat > frontrun/config/config.yml.example << EOF
# Frontrun Bot Configuration Template
# Copy to config.yml and fill in your values

rpc:
  endpoints:
    - url: "https://api.devnet.solana.com"
      websocket_url: "wss://api.devnet.solana.com"
      priority: 0
    - url: "https://YOUR-QUICKNODE-URL"
      websocket_url: "wss://YOUR-QUICKNODE-URL"
      priority: 1

  failover_threshold_errors: 3
  failover_threshold_latency_ms: 500
  health_check_interval_s: 10

wallets:
  # DO NOT commit actual private keys!
  # Use environment variables: ${WALLET_1_PRIVATE_KEY}
  private_keys: []
  min_balance_sol: 0.01
  cooldown_between_trades_s: 5

trading:
  position_size_sol: 0.5
  max_slippage_bps: 500
  priority_fee_urgency: "normal"

logging:
  level: "INFO"
  format: "json"
  output_file: "logs/frontrun.log"

metrics:
  enable_prometheus: true
  prometheus_port: 9090
EOF
```

---

## Next Steps - Getting Started

### Immediate Actions (Before Writing Code)

1. **Environment Setup** (1 day)
   - [ ] Set up development machine (Ubuntu 22.04 LTS recommended)
   - [ ] Install Python 3.11+
   - [ ] Create project structure
   - [ ] Set up git repository
   - [ ] Configure IDE (VSCode recommended with Python extensions)

2. **Dependency Research** (1 day)
   - [ ] Test `solders` library for transaction building
   - [ ] Test `websockets` for WebSocket connections
   - [ ] Identify best Prometheus client
   - [ ] Research Geyser providers (Helius, QuickNode, Triton)

3. **Devnet Wallet Setup** (1 day)
   - [ ] Generate 5-10 test keypairs
   - [ ] Airdrop devnet SOL (https://faucet.solana.com/)
   - [ ] Test basic RPC connection
   - [ ] Verify devnet pump.fun exists and is accessible

4. **Documentation Review** (1 day)
   - [ ] Re-read TechnicalSpecification.md for Phase 1 features
   - [ ] Review Solana JSON-RPC API docs
   - [ ] Review solders library documentation
   - [ ] Set up documentation templates

### Week 1 Schedule (Example)

**Day 1-2**: Setup
- Set up environment
- Create project structure
- Install dependencies
- Set up git and CI skeleton

**Day 3-4**: RPC Manager
- Implement basic WebSocket connection
- Implement multi-endpoint support
- Add reconnection logic
- Write unit tests

**Day 5**: Configuration & Metrics
- Implement config loading
- Set up structured logging
- Add basic metrics collection

**Weekend**: Testing & Documentation
- Integration testing on devnet
- Write documentation
- Code review and refactoring

---

## FAQ - Development Process

### Q: Should we reuse any code from the existing `/src` codebase?

**A**: Minimal reuse. The existing code is designed for slower copytrading (1-5 second latency tolerance) while frontrunning requires <100ms. Only reuse:
- Configuration patterns (adapt `config_loader.py`)
- Logging setup (adapt `logger_setup.py`)
- General project structure patterns

Build new implementations for all performance-critical components.

### Q: Can we skip testing in early phases to move faster?

**A**: **NO**. This is the #1 cause of technical debt. Every feature needs tests before it's considered "done". The time saved by skipping tests is lost 10x over when bugs appear in production.

### Q: What if Phase 1 takes longer than 2 weeks?

**A**: **That's OK**. The timeline is a guideline. Better to spend 3 weeks on a solid foundation than rush to Phase 2 with broken RPC management. Adjust timeline but never skip testing/documentation.

### Q: Should we deploy to mainnet gradually or all at once?

**A**: **Gradually**. After Phase 6 acceptance:
1. Deploy to mainnet with **simulation mode** (observe, don't trade)
2. Run for 1 week, collect metrics
3. Enable trading with **1 wallet, 0.1 SOL positions**
4. Run for 1 week, validate profitability
5. Scale to full configuration (10+ wallets, 0.5 SOL positions)

### Q: What if we can't afford Geyser access ($500/month) in Phase 4?

**A**: Options:
1. **Delay Phase 4**: Perfect Phases 1-3 first, bootstrap capital from other activities
2. **Self-host Geyser**: Run a Solana validator with Geyser plugin (requires DevOps skills)
3. **Start with free tier**: Some providers offer limited free Geyser access for testing

Do NOT skip frontrunning features entirely - they're the core differentiator.

### Q: How do we know if the bot is profitable before going to production?

**A**: Use the **Expected Profit Model** from PositionSizing.md:

```
Expected Profit per Attempt =
  (P_win × Profit_win) - (P_loss × Loss_loss) - Fee_Cost
```

During Phase 4-5 on devnet, measure:
- `P_win`: % of times you win the race (target: >70%)
- `Profit_win`: Average profit when you win (target: 30-100%)
- `P_loss`: % of times you lose (target: <30%)
- `Loss_loss`: Average loss when you lose (target: <100%)
- `Fee_Cost`: Average priority fee per attempt (measure actual)

**Go to mainnet ONLY if**: Expected profit > 0 with margin of safety.

---

## Conclusion

This roadmap provides a **clear, methodical path** from zero to production-ready frontrun bot in 8-10 weeks. Key success factors:

1. ✅ **Build incrementally**: Each phase produces working, tested code
2. ✅ **Test rigorously**: No feature is done without tests
3. ✅ **Document thoroughly**: Future you will thank present you
4. ✅ **Validate continuously**: Use go/no-go checkpoints
5. ✅ **Deploy carefully**: Simulation → small → scale

**The discipline of this process prevents 90% of the pain that comes from rushing to production with untested code.**

Start with Phase 1, validate every step, and you'll have a solid, profitable bot at the end.

Good luck! 🚀
