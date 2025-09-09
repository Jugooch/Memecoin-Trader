Sniper Bot Add-On — Implementation Playbook

Target files:

start_sniper.py — entrypoint + orchestration

config_sniper.yml — runtime configuration

sniper/ package (new)

discovery.py (Bitquery/Moralis dev discovery)

filters.py (safety checks, blacklist/whitelist logic)

execution.py (buy/sell, node selection, slippage, fee control)

exits.py (state machine for “no TP” dynamic exits sized for ~$30 positions)

risk.py (per-dev risk scoring + throttle)

storage.py (SQLite or JSONL persistence)

metrics.py (prometheus/logging)

backtest.py (optional: simulated fills for dry runs)

Assumptions:

You already have pump.fun trade plumbing (wallet, signer, AMM router, price feed, RPC).

Python 3.11+, asyncio.

Solana (values in SOL unless noted).

1) config_sniper.yml (schema + example)
# config_sniper.yml
network:
  cluster: "mainnet-beta"
  rpc_endpoints:
    - "https://<your-private-rpc-1>"
    - "https://<your-private-rpc-2>"
  node_region: "us-east"   # us-east | us-west | eu | asia
  retry_backoff_ms: 150

discovery:
  poll_interval_sec: 5
  sources:
    bitquery:
      enabled: true
      api_key: "${BITQUERY_KEY}"
    moralis:
      enabled: true
      api_key: "${MORALIS_KEY}"
  min_prev_peak_mc_usd: 1_000_000      # dev must have launched ≥ this MC on at least one token
  max_dev_launches_24h: 3              # anti-spam
  lookback_days: 60
  watchlist_bootstrap: ["<dev_pubkey1>", "<dev_pubkey2>"]  # optional seeds

trade:
  enable_live: true                     # false => shadow/dry-run
  quote_ccy: "SOL"
  position_size_sol: 0.2                # ~ $30 (adjust to current SOLUSD)
  max_open_positions: 1
  max_trades_per_hour: 6
  priority_fee_sol:
    base: 0.002
    max: 0.01
  slippage_bps: 250                     # 2.5%
  max_buy_impact_bps: 150               # abort if estimated impact > 1.5%

safety:
  dev_min_hold_pct: 1.0                 # dev must retain at least this % at/after launch
  dev_max_hold_pct: 20.0                # dev must hold less than this % (avoid choke)
  dev_min_liq_sol: 5.0                  # min initial LP
  dev_max_initial_buy_sol: 2.0          # avoid farmy self-buys
  dev_no_rug_history_days: 90           # disqualify devs who rugged in window
  max_tax_bps: 200                      # 2% max combined buy+sell tax
  max_tokens_launched_7d: 5             # too many launches → farm flag
  min_unique_buyers_first_5min: 12      # market breadth
  blocklist:
    wallets: []
    tickers: []
  allowlist:
    wallets: []                         # overrides most checks (still runs structural checks)

exit:
  # Dynamic “no TP” exit stack tuned for small size
  loss_cap_pct: 20
  trailing_floor_pct_min: 18
  trailing_floor_pct_max: 30
  whale_sell_threshold_pct_supply: 1.5
  whale_single_sale_pct: 35
  whale_multi_topN: 10
  whale_multi_sale_pct: 25
  whale_multi_window_sec: 120
  liq_cliff:
    vol_drop_pct: 70
    depth_drop_pct: 40
    spread_bps: 40
    window_min: 10
  time_decay_min_without_new_ath: 20
  partials:
    enabled: true
    steps:
      - trigger_mult: 3.0
        sell_pct: 30
      - trigger_mult: 6.0
        sell_pct: 20

storage:
  path: "data/sniper.sqlite"            # if empty, fallback JSONL files in data/
  rotate_jsonl_mb: 128

alerts:
  discord_webhook: "${DISCORD_SNIPER_WEBHOOK}"
  send_on:
    - "ENTRY_OK"
    - "ENTRY_ABORTED"
    - "EXIT_FILLED"
    - "RUG_ALERT"
    - "WHITELIST_UPDATE"
    - "BLACKLIST_UPDATE"

metrics:
  prometheus_port: 9109

2) Entrypoint start_sniper.py (orchestration)
# start_sniper.py
import asyncio
from sniper.storage import Store
from sniper.discovery import DevDiscovery
from sniper.filters import SafetyFilters
from sniper.execution import Executor
from sniper.exits import ExitManager
from sniper.risk import RiskManager
from sniper.metrics import Metrics
from utils.config import load_yaml

async def main():
    cfg = load_yaml("config_sniper.yml")
    store = Store(cfg["storage"])
    metrics = Metrics(cfg["metrics"])
    risk = RiskManager(cfg, store)
    exits = ExitManager(cfg, store, metrics)
    execu = Executor(cfg, store, exits, metrics)
    discover = DevDiscovery(cfg, store, metrics)

    safety = SafetyFilters(cfg, store, metrics)

    # background tasks
    tasks = [
        asyncio.create_task(discover.loop()),
        asyncio.create_task(execu.healthcheck_loop()),
        asyncio.create_task(metrics.serve_http()),
    ]

    # main event loop: react to candidate launches
    async for event in discover.stream_candidates():
        # event = {dev_wallet, token_mint, ticker, lp_init, taxes, dev_hold_pct, ...}
        if not safety.passes_all(event):
            metrics.inc("entry_rejected")
            store.log_event("ENTRY_ABORTED", event, reason=safety.last_reason)
            continue

        score = risk.score_dev(event["dev_wallet"], event)
        if not risk.allowed(score):
            store.log_event("ENTRY_ABORTED", event, reason="RISK_GATE")
            continue

        ok = await execu.try_enter(event)
        if not ok:
            store.log_event("ENTRY_ABORTED", event, reason=execu.last_reason)
            continue

        store.log_event("ENTRY_OK", event)
        # executor attaches ExitManager to token to manage lifecycle

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())

3) Dev Discovery (sniper/discovery.py)

Goal: discover/maintain a whitelist of dev wallets with positive history, then stream new token launches by these devs.

Inputs:

Bitquery: historical tokens created by wallet, price/MC paths, LP events.

Moralis: token metadata, top holders, transfers.

Your internal pump.fun event feed (if you already have it).

Logic:

Build dev_profile with:

num_tokens_launched, num_rugs_90d, best_peak_mc_usd, median_peak_mc_usd, avg_holder_count_30m, lp_lock_events, tax_history.

Eligibility:

best_peak_mc_usd >= min_prev_peak_mc_usd

num_rugs_90d == 0

tokens_launched_last_7d <= max_tokens_launched_7d

Streaming candidates:

Subscribe to new pump.fun token creations.

For each creation, map creator → dev_wallet, then:

If in whitelist: emit CandidateEvent downstream.

Else, if not blocklisted and dev_profile meets thresholds: emit with lower priority (optional).

Skeleton:

class DevDiscovery:
    def __init__(self, cfg, store, metrics):
        ...

    async def loop(self):
        # periodic rebuild of whitelist based on history (every 10 min)
        while True:
            await self.refresh_dev_scores()
            await asyncio.sleep(self.cfg["discovery"]["poll_interval_sec"])

    async def stream_candidates(self):
        # async generator yielding CandidateEvent dicts
        async for launch in self._stream_pumpfun_creations():
            dev = await self._resolve_dev_wallet(launch)
            profile = await self._get_or_build_profile(dev)
            if self._eligible(profile) and not self.store.is_blacklisted_dev(dev):
                yield self._to_candidate_event(launch, dev, profile)


Bitquery/Moralis notes:

Create adapters: bitquery_adapter.py, moralis_adapter.py with typed responses.

Cache results in Store to bound API usage.

Rate limit + backoff.

4) Whitelist / Blacklist (sniper/filters.py)

Whitelist seeding:

Merge static allowlist.wallets with dynamic discovery set.

Store maintains tables:

dev_whitelist(dev_wallet, score, last_reviewed_at)

dev_blacklist(dev_wallet, reason, inserted_at)

ticker_blacklist(ticker, reason)

Blacklist rules (automatic):

Add dev to blacklist if:

LP removed within X minutes after launch.

Dev sells >Y% of supply within Z minutes of launch.

Honeypot/tax flip detected (buy allowed, sell fails/tax > max_tax_bps).

Manual admin override via CLI (below).

Safety checks (must all pass before entry):

Dev not in blacklist.

Dev prior history clean for dev_no_rug_history_days.

Dev initial buy ≤ dev_max_initial_buy_sol.

Dev LP ≥ dev_min_liq_sol.

Dev hold % ∈ [dev_min_hold_pct, dev_max_hold_pct].

Combined buy+sell tax ≤ max_tax_bps.

Token breadth early: min_unique_buyers_first_5min (defer check for delayed entries or use last 60s if we’re sniping block 1—configure to bypass when entering instantly, but enforce if entry delayed).

Ticker/wallet not on blocklist.

class SafetyFilters:
    def passes_all(self, evt) -> bool:
        checks = [
            self._not_blacklisted(evt),
            self._dev_history_clean(evt),
            self._dev_initials_ok(evt),
            self._lp_min_ok(evt),
            self._dev_hold_bounds(evt),
            self._tax_bounds(evt),
            self._breadth_ok(evt),
        ]
        for ok, reason in checks:
            if not ok:
                self.last_reason = reason
                return False
        return True

5) Risk Manager (sniper/risk.py)

Scoring (0–100):

Components (weights):

Peak MC achieved historically (30%)

Rug count inverse (20%)

Median holder count at 15m (15%)

Launch frequency penalty (15%)

LP lock / credible LP sources (10%)

Social footprint (optional, 10%)

Gates:

allowed(score) with threshold, e.g., ≥60.

Throttle: per-dev cooldown (e.g., only 1 attempt per dev per 24h).

6) Execution (sniper/execution.py)

Entry:

Estimate price impact using pool reserves; compare against max_buy_impact_bps.

Priority fee = clamp(base → max) based on local mempool congestion metric (or simple tiering by dev score).

Slippage from config.

Order types:

IOC market buy for speed (your existing router).

On success, register position with ExitManager.

On failure (slippage/price moved), abort and log reason.

Healthcheck:

Rotate RPCs on error spikes.

Validate clock drift, slot lag, and node region routing.

class Executor:
    async def try_enter(self, evt) -> bool:
        # sanity estimates
        if self._est_impact_bps(evt) > self.cfg["trade"]["max_buy_impact_bps"]:
            self.last_reason = "IMPACT_TOO_HIGH"
            return False
        # place buy
        txsig = await self.router.buy(
            mint=evt["token_mint"],
            amount_sol=self.cfg["trade"]["position_size_sol"],
            slippage_bps=self.cfg["trade"]["slippage_bps"],
            priority_fee_sol=self._priority_fee(evt),
        )
        if not txsig:
            self.last_reason = "TX_FAILED"
            return False
        self.exits.register_position(evt, txsig)
        return True

7) Exit Strategy (sniper/exits.py)

State machine (evaluated every slot):

Safety:

Rug/structural break → sell 100% immediately.

Loss cap: price ≤ entry × (1 − 0.20) → sell 100%.

Whale distribution:

Any top-10 holder (≥1.5% supply) sells ≥35% inside 120s → sell 100%.

Or ≥2 top-10 holders sell ≥25% each inside 120s → sell 100%.

Partials (if enabled):

Hit 3× → sell 30%.

Hit 6× → sell 20%.

Trailing stop (adaptive):

Track ATH; exit all if price ≤ ATH × (1 − τ),

τ = clip(3 × ATR_pct, 0.18, 0.30).

Liquidity/volume cliff:

10-min vol ↓ ≥70% and depth ↓ ≥40% and spread > 40 bps → sell 50% and tighten τ to 0.18.

Time decay:

No new ATH for 20 min and EMA20 < EMA50 → exit remainder.

Sized for ~$30:

Because fills are small, prefer single-shot exits (IOC market) except when liq is thin → micro-TWAP over 5–10s.

Keep priority fee modest (0.002–0.01 SOL) even on emergency exits.

8) Storage (sniper/storage.py)

Tables:

positions(id, token, dev_wallet, entry_price, qty, entry_time, status)

dev_profiles(dev_wallet, score, stats_json, updated_at)

blacklist(dev_wallet, reason, ts)

events(ts, type, payload_json)

fills(position_id, side, qty, price, fee_sol, ts)

Stick with JSON for this storage because it works best on the VM

9) Metrics & Alerts

Prometheus:

sniper_entry_attempts_total{result}

sniper_exit_events_total{reason}

pnl_cumulative_sol

latency_ms{phase=discovery|enter|exit}

rpc_errors_total{endpoint}

Discord alerts (concise JSON):

ENTRY_OK with tx, mint, price, est impact.

ENTRY_ABORTED with reason.

EXIT_FILLED with reason, realized multiple, PnL.

BLACKLIST_UPDATE/WHITELIST_UPDATE.

10) CLI (ops)

Add cli_sniper.py with:

python cli_sniper.py add-blacklist <dev_wallet> <reason>

python cli_sniper.py remove-blacklist <dev_wallet>

python cli_sniper.py add-whitelist <dev_wallet> [score]

python cli_sniper.py show-dev <dev_wallet>

python cli_sniper.py dump-positions --open/--closed

11) Backtesting Hooks (sniper/backtest.py) — optional but recommended

Input: JSONL of historical launches with per-slot OHLCV, holder changes, LP stats.

Simulate execution + exits with given config.

Output: CSV of trade outcomes + event reason histogram.

Unit test: ensure exit stack triggers in the right order.

12) Unit Tests (minimum)

Filters: each safety gate rejects malformed/risky events and accepts good ones.

Blacklist auto-add when rug conditions met.

Exit manager: deterministic triggers when fed synthetic price/holder streams.

Executor: impact calc aborts on thin pools.

Risk manager: scoring and gating thresholds.

13) Engineer Task Checklist (cut/paste)

Project setup

Create sniper/ package with modules listed above.

Wire start_sniper.py orchestrator.

Config

Implement utils/config.load_yaml with env var expansion.

Validate config at startup.

Discovery

Build Bitquery & Moralis adapters.

Implement DevDiscovery.refresh_dev_scores() and stream_candidates().

Cache profiles in Store.

Filters & Risk

Implement safety gates per config_sniper.yml.

Implement RiskManager.score_dev() and gating.

Execution

Implement Executor.try_enter() with impact calc, slippage, priority fee.

Connect to existing pump.fun router.

Exits

Implement ExitManager loop; attach to positions.

Implement whale detection from holder deltas.

Implement trailing/partials/liquidity/time rules.

Storage & Metrics

SQLite schema or JSONL logs.

Prometheus server + counters.

Discord webhooks.

Ops/CLI

Blacklist/whitelist management.

Position introspection.

Testing

Unit tests for filters/exits.

Dry-run mode (enable_live: false) with shadow fills.

Rollout

Start in dry-run 48–72h.

Turn on live with position_size_sol=0.2, max_trades_per_hour=3.

Review event histograms; adjust thresholds (especially taxes, whale %, τ bounds).

14) Notes on Parameterization

$30 position (~0.2 SOL @ $150/SOL): keep slippage_bps ≤ 250, otherwise small fills get degraded.

Aim for mid-tier launches only: set dev_max_initial_buy_sol=2.0, max_tokens_launched_7d=5, and require min_prev_peak_mc_usd ≥ 1M.

Keep priority fees modest; let node latency carry you. Rotate RPCs on slot lag.