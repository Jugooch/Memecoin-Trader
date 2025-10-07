# Solana Low-Latency Trading Bot - Technical Specification

This document provides implementation-ready specifications for all features outlined in FeatureSet.md.

---

## Shared Infrastructure (Features 1-14)

### 1. Multi-RPC Connection Manager

**Purpose**: Maintains persistent connections to 2-4 Solana RPC providers with automatic failover.

**API Specification**
```python
class RPCManager:
    def __init__(self, endpoints: List[RPCEndpoint], config: RPCConfig):
        """Initialize RPC manager with multiple endpoints"""

    async def get_connection(self) -> RPCConnection:
        """Get healthy connection with lowest latency"""

    async def subscribe_program(
        self,
        program_id: str,
        filters: List[RpcFilter],
        commitment: str = "confirmed"
    ) -> AsyncIterator[ProgramAccountNotification]:
        """Subscribe to program account changes"""

    async def send_transaction(
        self,
        tx: VersionedTransaction,
        skip_preflight: bool = True,
        commitment: str = "confirmed"
    ) -> str:
        """Send transaction with automatic failover"""

    async def get_recent_blockhash(self) -> Tuple[str, int]:
        """Get recent blockhash with slot number"""

    async def get_account_info(
        self,
        pubkey: str,
        commitment: str = "confirmed"
    ) -> Optional[AccountInfo]:
        """Fetch account data"""

    def get_health_stats(self) -> Dict[str, RPCHealthStats]:
        """Get latency and availability metrics per endpoint"""
```

**Data Structures**
```python
@dataclass
class RPCEndpoint:
    url: str
    websocket_url: str
    priority: int  # 0=primary, 1=fallback, etc.
    max_connections: int = 10
    timeout_ms: int = 5000

@dataclass
class RPCConfig:
    failover_threshold_errors: int = 3
    failover_threshold_latency_ms: int = 500
    health_check_interval_s: int = 10
    reconnect_backoff_base_ms: int = 100
    reconnect_backoff_max_ms: int = 5000

@dataclass
class RPCHealthStats:
    endpoint_url: str
    is_healthy: bool
    p50_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    last_successful_call: datetime
    consecutive_failures: int

@dataclass
class ProgramAccountNotification:
    pubkey: str
    account: AccountInfo
    slot: int
```

**Implementation Notes**
- Use `websockets` library for WebSocket connections
- Use `aiohttp` for HTTP RPC calls
- Implement exponential backoff: 100ms → 200ms → 400ms → 800ms → max 5s
- Health check via `getHealth` RPC method every 10s
- Track latency using `time.perf_counter()` with microsecond precision
- Failover triggers: 3 consecutive failures OR p99 latency >500ms
- Maintain connection pool per endpoint (max 10 concurrent)

**Performance Requirements**
- Connection establishment: <200ms
- Reconnection after failure: <1s
- WebSocket message processing: <5ms per message
- Health check overhead: <1% of total RPC capacity

**Error Handling**
- **Connection refused**: Try next endpoint immediately
- **Timeout**: 5s for initial connect, 10s for active connections
- **Rate limits**: Implement token bucket (100 req/s per endpoint)
- **Invalid response**: Log error, retry with exponential backoff
- **WebSocket disconnect**: Auto-reconnect with backoff, resubscribe to all subscriptions

**Testing**
- Unit: Mock WebSocket/HTTP responses, verify reconnection logic
- Integration: Test against devnet with simulated network failures (use `tc` to add latency/drops)
- Load: 1000 concurrent subscriptions across endpoints
- Acceptance: <1s recovery time from endpoint failure

---

### 2. Transaction Builder (Base)

**Purpose**: Constructs Solana transactions from instructions with proper account ordering and compute budget.

**API Specification**
```python
class TransactionBuilder:
    def __init__(self, rpc_manager: RPCManager):
        """Initialize with RPC manager for blockhash fetching"""

    async def build_transaction(
        self,
        instructions: List[Instruction],
        payer: Pubkey,
        signers: List[Keypair],
        compute_unit_limit: Optional[int] = None,
        compute_unit_price: Optional[int] = None,
        use_versioned: bool = True
    ) -> Union[Transaction, VersionedTransaction]:
        """Build transaction with optional compute budget"""

    def add_compute_budget_instructions(
        self,
        instructions: List[Instruction],
        compute_unit_limit: int,
        compute_unit_price: int
    ) -> List[Instruction]:
        """Prepend compute budget instructions"""

    async def get_transaction_size(self, tx: VersionedTransaction) -> int:
        """Calculate serialized transaction size in bytes"""

    def optimize_account_metas(self, instructions: List[Instruction]) -> List[Instruction]:
        """Deduplicate and optimize account meta ordering"""
```

**Data Structures**
```python
@dataclass
class Instruction:
    program_id: Pubkey
    accounts: List[AccountMeta]
    data: bytes

@dataclass
class AccountMeta:
    pubkey: Pubkey
    is_signer: bool
    is_writable: bool

@dataclass
class TransactionBuildConfig:
    max_tx_size_bytes: int = 1232  # Solana limit
    default_compute_units: int = 200_000
    signature_verification_compute: int = 5000  # per signature
```

**Solana-Specific Details**
- Use `VersionedTransaction` with v0 message format for address lookup tables support
- Compute budget program ID: `ComputeBudget111111111111111111111111111111`
- Compute budget instructions must be first in instruction list
- Recent blockhash valid for ~60 seconds (120 slots at 500ms/slot)
- Transaction size limit: 1232 bytes

**Implementation Notes**
- Use `solders` library for transaction construction (faster than `solana-py`)
- Cache recent blockhash for 30 seconds to reduce RPC calls
- Automatically add compute budget instructions if `compute_unit_limit` or `compute_unit_price` provided
- Validate transaction size before signing (reject if >1232 bytes)

**Performance Requirements**
- Transaction building: <10ms per transaction
- Blockhash caching hit rate: >90%
- Memory: <1MB per 1000 cached blockhashes

**Error Handling**
- **Missing blockhash**: Fetch new blockhash, retry
- **Transaction too large**: Log error with size breakdown, abort
- **Invalid instruction data**: Validate before serialization, raise `InvalidInstructionError`
- **Stale blockhash**: Auto-refresh if >30s old

**Testing**
- Unit: Test with mock instructions, verify compute budget prepending
- Integration: Build real pump.fun transactions, validate on devnet
- Performance: Benchmark 10,000 transaction builds/sec
- Acceptance: Successfully build and submit transaction <50ms end-to-end

---

### 3. Transaction Signer

**Purpose**: Signs transactions with hot wallet keys using Ed25519.

**API Specification**
```python
class TransactionSigner:
    def __init__(self, keypairs: List[Keypair], config: SignerConfig):
        """Initialize with hot wallet keypairs"""

    def sign_transaction(
        self,
        tx: Union[Transaction, VersionedTransaction],
        signers: List[Pubkey]
    ) -> Union[Transaction, VersionedTransaction]:
        """Sign transaction with specified keypairs"""

    def sign_message(self, message: bytes, signer: Pubkey) -> bytes:
        """Sign arbitrary message with keypair"""

    def get_keypair(self, pubkey: Pubkey) -> Optional[Keypair]:
        """Retrieve keypair for public key"""

    def add_keypair(self, keypair: Keypair) -> None:
        """Add new keypair to signer (for wallet rotation)"""

    def remove_keypair(self, pubkey: Pubkey) -> None:
        """Remove keypair from memory"""
```

**Data Structures**
```python
@dataclass
class SignerConfig:
    enable_key_rotation: bool = True
    max_signatures_per_key: int = 10_000
    key_rotation_interval_minutes: int = 60

@dataclass
class SignatureStats:
    pubkey: Pubkey
    total_signatures: int
    last_used: datetime
    created_at: datetime
```

**Solana-Specific Details**
- Ed25519 signatures are 64 bytes
- Each signature adds ~5000 compute units to transaction
- Multi-signature transactions require all signers to sign the same message hash

**Implementation Notes**
- Use `nacl.signing` for Ed25519 operations
- Store keypairs in memory as `Keypair` objects (never disk in production)
- Implement signature counting for key rotation triggers
- Use `secrets` module for secure key generation

**Performance Requirements**
- Signing latency: <1ms per signature
- Key lookup: O(1) via hashmap
- Memory: ~100 bytes per keypair

**Error Handling**
- **Missing keypair**: Raise `KeypairNotFoundError`
- **Invalid transaction**: Validate message hash exists before signing
- **Signature verification failure**: Log and re-sign once, else abort

**Testing**
- Unit: Sign mock transactions, verify signature validity
- Integration: Sign and submit real transactions on devnet
- Security: Ensure keypairs never logged or persisted
- Acceptance: 1000 signatures/sec throughput

---

### 4. Transaction Submitter

**Purpose**: Broadcasts signed transactions to Solana network with retry logic.

**API Specification**
```python
class TransactionSubmitter:
    def __init__(self, rpc_manager: RPCManager, config: SubmitterConfig):
        """Initialize with RPC manager"""

    async def submit_transaction(
        self,
        signed_tx: VersionedTransaction,
        commitment: str = "confirmed",
        max_retries: int = 3
    ) -> TransactionResult:
        """Submit transaction with retries"""

    async def submit_and_confirm(
        self,
        signed_tx: VersionedTransaction,
        timeout_seconds: int = 30,
        commitment: str = "confirmed"
    ) -> ConfirmedTransaction:
        """Submit and wait for confirmation"""

    async def get_transaction_status(
        self,
        signature: str,
        commitment: str = "confirmed"
    ) -> Optional[TransactionStatus]:
        """Check transaction status"""
```

**Data Structures**
```python
@dataclass
class SubmitterConfig:
    skip_preflight: bool = True  # Skip simulation for speed
    max_retries: int = 3
    retry_delay_ms: int = 200
    confirmation_timeout_s: int = 30

@dataclass
class TransactionResult:
    signature: str
    submitted_at: datetime
    slot: Optional[int]
    error: Optional[str]

@dataclass
class ConfirmedTransaction:
    signature: str
    slot: int
    block_time: int
    confirmation_status: str  # "processed" | "confirmed" | "finalized"
    compute_units_consumed: int
    error: Optional[TransactionError]

@dataclass
class TransactionStatus:
    signature: str
    confirmation_status: Optional[str]
    slot: Optional[int]
    err: Optional[dict]
```

**Solana-Specific Details**
- Use `sendTransaction` RPC method with `skipPreflight: true` for speed
- Confirmation levels: processed (1 slot) → confirmed (32 slots) → finalized (32+ slots)
- Transaction signatures are base58-encoded strings (87-88 chars)

**Implementation Notes**
- Use `rpc_manager.send_transaction()` for automatic RPC failover
- Implement exponential backoff: 200ms → 400ms → 800ms
- Poll for confirmation every 500ms using `getSignatureStatuses`
- Log all submissions with signature, slot, and latency

**Performance Requirements**
- Submission latency: <50ms to RPC
- Confirmation check overhead: <10ms per poll
- Throughput: 100 concurrent submissions

**Error Handling**
- **BlockhashNotFound**: Transaction expired, rebuild with new blockhash
- **InsufficientFundsForFee**: Log error, abort (no retry)
- **AlreadyProcessed**: Check status, return existing result
- **Network timeout**: Retry with next RPC endpoint
- **Unknown error**: Log full error, retry up to max_retries

**Testing**
- Unit: Mock RPC responses for success/failure scenarios
- Integration: Submit to devnet, verify confirmation
- Load: 1000 concurrent submissions
- Acceptance: >95% confirmation rate within 30s

---

### 5. Multi-Wallet Manager

**Purpose**: Manages pool of trading wallets with rotation and balance tracking.

**API Specification**
```python
class MultiWalletManager:
    def __init__(
        self,
        keypairs: List[Keypair],
        rpc_manager: RPCManager,
        config: WalletConfig
    ):
        """Initialize wallet pool"""

    async def get_available_wallet(
        self,
        min_sol_balance: float = 0.01
    ) -> Optional[WalletContext]:
        """Get wallet with sufficient balance and not in cooldown"""

    async def refresh_balances(self) -> Dict[Pubkey, float]:
        """Refresh SOL balances for all wallets"""

    def mark_wallet_used(self, pubkey: Pubkey, cooldown_seconds: int = 5):
        """Mark wallet as used, put in cooldown"""

    def release_wallet(self, pubkey: Pubkey):
        """Release wallet from cooldown"""

    async def fund_wallet(self, target: Pubkey, amount_sol: float) -> str:
        """Fund wallet from master wallet"""
```

**Data Structures**
```python
@dataclass
class WalletConfig:
    min_balance_sol: float = 0.01
    balance_refresh_interval_s: int = 30
    cooldown_between_trades_s: int = 5
    max_concurrent_wallets: int = 10

@dataclass
class WalletContext:
    keypair: Keypair
    pubkey: Pubkey
    sol_balance: float
    in_use: bool
    last_used: Optional[datetime]
    total_trades: int

@dataclass
class WalletStats:
    total_wallets: int
    available_wallets: int
    wallets_in_use: int
    wallets_in_cooldown: int
    total_sol_balance: float
```

**Implementation Notes**
- Use `asyncio.Lock()` per wallet to prevent concurrent usage
- Cache balances, refresh every 30s or after transaction
- Implement round-robin selection among available wallets
- Track nonce per wallet for transaction ordering (not needed for standard txs)

**Performance Requirements**
- Wallet selection: <1ms
- Balance refresh: <100ms for 10 wallets
- Lock acquisition: Non-blocking with timeout

**Error Handling**
- **No available wallets**: Return None, caller should wait/retry
- **Insufficient balance**: Skip wallet in selection
- **Balance fetch failure**: Use cached balance, log warning
- **Lock timeout**: Return None after 100ms wait

**Testing**
- Unit: Mock balance fetches, verify rotation logic
- Integration: Test with real wallets on devnet
- Concurrency: 100 concurrent wallet requests
- Acceptance: No wallet used concurrently, all wallets rotated evenly

---

### 6. Priority Fee Calculator

**Purpose**: Computes optimal compute unit price for fast transaction inclusion.

**API Specification**
```python
class PriorityFeeCalculator:
    def __init__(self, rpc_manager: RPCManager, config: FeeConfig):
        """Initialize fee calculator"""

    async def calculate_priority_fee(
        self,
        compute_units: int = 200_000,
        urgency: FeeUrgency = FeeUrgency.NORMAL
    ) -> int:
        """Calculate compute unit price in micro-lamports"""

    async def get_recent_fees(self) -> List[PriorityFeeStats]:
        """Get recent prioritization fees from network"""

    def get_fee_estimate(self, percentile: int = 50) -> int:
        """Get fee estimate from cached data at percentile"""
```

**Data Structures**
```python
@dataclass
class FeeConfig:
    update_interval_s: int = 10
    lookback_slots: int = 150
    min_priority_fee: int = 1000  # micro-lamports
    max_priority_fee: int = 1_000_000

class FeeUrgency(Enum):
    LOW = "low"  # p50 fees
    NORMAL = "normal"  # p75 fees
    HIGH = "high"  # p90 fees
    CRITICAL = "critical"  # p99 fees

@dataclass
class PriorityFeeStats:
    slot: int
    prioritization_fee: int
    timestamp: datetime
```

**Solana-Specific Details**
- Use `getRecentPrioritizationFees` RPC method with account list or lookback
- Fees measured in micro-lamports per compute unit (1 lamport = 1_000_000 micro-lamports)
- Compute budget instruction: `ComputeBudget::SetComputeUnitPrice(micro_lamports)`

**Implementation Notes**
- Cache fee data, refresh every 10 seconds
- Calculate percentiles: p50, p75, p90, p99
- Map urgency to percentile: LOW→p50, NORMAL→p75, HIGH→p90, CRITICAL→p99
- Clamp fees between min (1000) and max (1_000_000) micro-lamports

**Performance Requirements**
- Fee calculation: <5ms
- RPC fee fetch: <100ms
- Cache hit rate: >90%

**Error Handling**
- **RPC failure**: Use cached fees, log warning
- **No fee data**: Return `min_priority_fee`
- **Outlier fees**: Cap at `max_priority_fee`

**Testing**
- Unit: Mock RPC responses, verify percentile calculations
- Integration: Fetch real fees from mainnet, validate ranges
- Acceptance: Fees calculated within 5ms, >80% transaction inclusion rate

---

### 7. Pump.fun Program Client (Base)

**Purpose**: Encodes/decodes pump.fun program instructions for buy/sell trades.

**API Specification**
```python
class PumpFunClient:
    PROGRAM_ID: ClassVar[Pubkey] = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")

    def __init__(self, rpc_manager: RPCManager):
        """Initialize pump.fun client"""

    async def build_buy_instruction(
        self,
        mint: Pubkey,
        bonding_curve: Pubkey,
        associated_bonding_curve: Pubkey,
        user: Pubkey,
        user_token_account: Pubkey,
        amount_sol: int,
        max_slippage_bps: int = 500
    ) -> Instruction:
        """Build buy instruction"""

    async def build_sell_instruction(
        self,
        mint: Pubkey,
        bonding_curve: Pubkey,
        associated_bonding_curve: Pubkey,
        user: Pubkey,
        user_token_account: Pubkey,
        amount_tokens: int,
        min_sol_output: int
    ) -> Instruction:
        """Build sell instruction"""

    async def get_bonding_curve_account(self, mint: Pubkey) -> BondingCurveAccount:
        """Fetch bonding curve state"""

    def derive_bonding_curve_pda(self, mint: Pubkey) -> Pubkey:
        """Derive bonding curve PDA from mint"""

    def derive_associated_bonding_curve(self, bonding_curve: Pubkey, mint: Pubkey) -> Pubkey:
        """Derive associated bonding curve token account"""
```

**Data Structures**
```python
@dataclass
class BondingCurveAccount:
    virtual_token_reserves: int
    virtual_sol_reserves: int
    real_token_reserves: int
    real_sol_reserves: int
    token_total_supply: int
    complete: bool

@dataclass
class PumpFunConfig:
    fee_bps: int = 100  # 1% fee
    bonding_curve_seed: bytes = b"bonding-curve"
```

**Solana-Specific Details**
- Program ID: `6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P`
- Buy instruction discriminator: `0x66063d1201daebea` (first 8 bytes of SHA256("global:buy"))
- Sell instruction discriminator: `0x33e685a4017f83ad` (first 8 bytes of SHA256("global:sell"))
- Bonding curve PDA: `[b"bonding-curve", mint.key()]`
- Accounts order for buy: `[bonding_curve, associated_bonding_curve, user, user_token_account, system_program, token_program, rent, event_authority, program]`

**Implementation Notes**
- Use `solders.pubkey.Pubkey.find_program_address()` for PDA derivation
- Serialize instruction data with `struct.pack("<Q", amount_lamports)` after discriminator
- Cache bonding curve PDAs per mint to avoid recomputation

**Performance Requirements**
- Instruction building: <5ms
- PDA derivation: <1ms
- Bonding curve account fetch: <50ms

**Error Handling**
- **Invalid mint**: Validate pubkey format, raise `InvalidMintError`
- **Bonding curve not found**: Raise `BondingCurveNotFoundError`
- **Account deserialization failure**: Log raw data, raise parsing error

**Testing**
- Unit: Test PDA derivation against known mint/curve pairs
- Integration: Build and submit real buy/sell on devnet
- Acceptance: Successfully execute trades with <1% failure rate

---

### 8. Bonding Curve Calculator

**Purpose**: Computes price and slippage from bonding curve state.

**API Specification**
```python
class BondingCurveCalculator:
    def calculate_buy_price(
        self,
        bonding_curve: BondingCurveAccount,
        amount_sol: int
    ) -> BuyQuote:
        """Calculate tokens received for SOL amount"""

    def calculate_sell_price(
        self,
        bonding_curve: BondingCurveAccount,
        amount_tokens: int
    ) -> SellQuote:
        """Calculate SOL received for token amount"""

    def calculate_price_impact(
        self,
        bonding_curve: BondingCurveAccount,
        amount_sol: int
    ) -> float:
        """Calculate price impact as percentage"""

    def get_current_price(self, bonding_curve: BondingCurveAccount) -> float:
        """Get current price in SOL per token"""
```

**Data Structures**
```python
@dataclass
class BuyQuote:
    tokens_out: int
    price_per_token: float
    price_impact_pct: float
    fee_lamports: int

@dataclass
class SellQuote:
    sol_out: int
    price_per_token: float
    price_impact_pct: float
    fee_lamports: int
```

**Solana-Specific Details**
- Bonding curve formula: constant product AMM `x * y = k`
- Virtual reserves used for pricing (includes fees)
- Fee: 1% of trade amount (100 bps)

**Implementation Notes**
- Formula for buy: `tokens_out = (reserves_token * amount_sol) / (reserves_sol + amount_sol)`
- Formula for sell: `sol_out = (reserves_sol * amount_tokens) / (reserves_token + amount_tokens)`
- Apply 1% fee before calculation
- Use integer math to match on-chain behavior exactly

**Performance Requirements**
- Calculation latency: <1ms
- Precision: Match on-chain calculation within 1 lamport

**Error Handling**
- **Division by zero**: Check reserves > 0, raise `InvalidBondingCurveError`
- **Overflow**: Use Python's arbitrary precision integers
- **Negative result**: Raise `InsufficientLiquidityError`

**Testing**
- Unit: Test against known bonding curve states
- Integration: Compare calculated vs actual fills on-chain
- Acceptance: <1 lamport deviation from on-chain results

---

### 9. Slippage Manager

**Purpose**: Enforces max slippage tolerance on trades.

**API Specification**
```python
class SlippageManager:
    def __init__(self, config: SlippageConfig):
        """Initialize slippage manager"""

    def calculate_minimum_output(
        self,
        expected_output: int,
        slippage_bps: int
    ) -> int:
        """Calculate minimum acceptable output with slippage"""

    def validate_slippage(
        self,
        expected: int,
        actual: int,
        max_slippage_bps: int
    ) -> bool:
        """Check if actual output within slippage tolerance"""

    def get_max_slippage_for_urgency(self, urgency: TradeUrgency) -> int:
        """Get slippage tolerance based on trade urgency"""
```

**Data Structures**
```python
@dataclass
class SlippageConfig:
    default_slippage_bps: int = 500  # 5%
    max_slippage_bps: int = 1000  # 10%
    min_slippage_bps: int = 50  # 0.5%

class TradeUrgency(Enum):
    LOW = "low"  # 1% slippage
    NORMAL = "normal"  # 5% slippage
    HIGH = "high"  # 10% slippage
    CRITICAL = "critical"  # 20% slippage (frontrun exits)
```

**Implementation Notes**
- 1 bps = 0.01%
- Formula: `min_output = expected * (10000 - slippage_bps) / 10000`
- Always round down for minimum output (conservative)

**Performance Requirements**
- Calculation latency: <100 microseconds
- Memory: Negligible

**Error Handling**
- **Invalid bps**: Clamp between 0-10000
- **Negative values**: Raise `ValueError`

**Testing**
- Unit: Test calculations with various slippage values
- Acceptance: All calculations match expected values exactly

---

### 10. Position Tracker

**Purpose**: Tracks open positions per wallet.

**API Specification**
```python
class PositionTracker:
    def __init__(self, storage: PositionStorage):
        """Initialize position tracker"""

    async def open_position(
        self,
        wallet: Pubkey,
        mint: Pubkey,
        amount_tokens: int,
        entry_price_sol: float,
        entry_slot: int,
        strategy: str
    ) -> Position:
        """Record new position"""

    async def close_position(
        self,
        position_id: str,
        exit_price_sol: float,
        exit_slot: int
    ) -> ClosedPosition:
        """Close position and calculate PnL"""

    async def get_open_positions(
        self,
        wallet: Optional[Pubkey] = None,
        mint: Optional[Pubkey] = None
    ) -> List[Position]:
        """Get open positions with optional filters"""

    async def update_position_size(
        self,
        position_id: str,
        new_amount: int
    ) -> Position:
        """Update position size (partial exits)"""
```

**Data Structures**
```python
@dataclass
class Position:
    position_id: str  # UUID
    wallet: Pubkey
    mint: Pubkey
    amount_tokens: int
    entry_price_sol: float
    entry_slot: int
    entry_timestamp: datetime
    strategy: str  # "frontrun" or "copytrading"
    status: PositionStatus

@dataclass
class ClosedPosition(Position):
    exit_price_sol: float
    exit_slot: int
    exit_timestamp: datetime
    pnl_sol: float
    pnl_pct: float
    holding_time_seconds: int

class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    FAILED = "failed"
```

**Implementation Notes**
- Use SQLite for persistence with in-memory cache
- Index on `(wallet, mint, status)` for fast lookups
- Generate position_id as UUID v4
- Store timestamps in UTC

**Performance Requirements**
- Position lookup: <5ms
- Position insert: <10ms
- Cache hit rate: >95%

**Error Handling**
- **Duplicate position**: Log warning, return existing
- **Position not found**: Raise `PositionNotFoundError`
- **Invalid update**: Validate amount > 0

**Testing**
- Unit: Test CRUD operations with mock storage
- Integration: Test with SQLite database
- Acceptance: 1000 positions/sec throughput

---

### 11. PnL Calculator

**Purpose**: Calculates profit/loss for positions.

**API Specification**
```python
class PnLCalculator:
    def __init__(self, config: PnLConfig):
        """Initialize PnL calculator"""

    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        amount: int,
        fees_paid: float = 0
    ) -> PnLResult:
        """Calculate PnL in SOL and percentage"""

    def calculate_unrealized_pnl(
        self,
        position: Position,
        current_price: float
    ) -> UnrealizedPnL:
        """Calculate unrealized PnL for open position"""

    def get_aggregated_pnl(
        self,
        positions: List[ClosedPosition]
    ) -> AggregatedPnL:
        """Aggregate PnL across multiple positions"""
```

**Data Structures**
```python
@dataclass
class PnLConfig:
    include_fees: bool = True

@dataclass
class PnLResult:
    pnl_sol: float
    pnl_pct: float
    gross_pnl_sol: float
    fees_paid_sol: float

@dataclass
class UnrealizedPnL:
    position_id: str
    current_value_sol: float
    unrealized_pnl_sol: float
    unrealized_pnl_pct: float

@dataclass
class AggregatedPnL:
    total_pnl_sol: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_pnl_sol: float
    largest_win_sol: float
    largest_loss_sol: float
```

**Implementation Notes**
- Formula: `pnl_sol = (exit_price - entry_price) * amount - fees`
- Formula: `pnl_pct = ((exit_price - entry_price) / entry_price) * 100`
- Always express prices in SOL (not lamports) for readability

**Performance Requirements**
- Calculation latency: <100 microseconds per position
- Aggregation: <10ms for 1000 positions

**Error Handling**
- **Zero entry price**: Raise `ValueError`
- **Negative amount**: Raise `ValueError`

**Testing**
- Unit: Test calculations with known inputs/outputs
- Acceptance: Match manual calculations exactly

---

### 12. Metrics & Logging System

**Purpose**: Structured logging and performance metrics.

**API Specification**
```python
class MetricsCollector:
    def __init__(self, config: MetricsConfig):
        """Initialize metrics collector"""

    def record_latency(
        self,
        operation: str,
        latency_ms: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record operation latency"""

    def increment_counter(
        self,
        metric_name: str,
        value: int = 1,
        labels: Optional[Dict[str, str]] = None
    ):
        """Increment counter metric"""

    def set_gauge(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Set gauge value"""

    def get_histogram_stats(
        self,
        operation: str
    ) -> HistogramStats:
        """Get p50/p95/p99 latency for operation"""

    def export_metrics(self) -> Dict[str, any]:
        """Export all metrics as JSON"""

class StructuredLogger:
    def __init__(self, name: str, config: LogConfig):
        """Initialize structured logger"""

    def info(self, message: str, **kwargs):
        """Log info with structured fields"""

    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error with exception details"""

    def debug(self, message: str, **kwargs):
        """Log debug message"""
```

**Data Structures**
```python
@dataclass
class MetricsConfig:
    enable_histogram: bool = True
    histogram_buckets: List[float] = field(default_factory=lambda: [1, 5, 10, 50, 100, 500, 1000])
    export_interval_s: int = 60

@dataclass
class LogConfig:
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    format: str = "json"  # json or text
    output_file: Optional[str] = None

@dataclass
class HistogramStats:
    operation: str
    count: int
    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float
```

**Implementation Notes**
- Use `structlog` for JSON logging
- Use `prometheus_client` for metrics (histograms, counters, gauges)
- Log all transactions with: signature, slot, latency, success/failure
- Track latencies: tx_build, tx_sign, tx_submit, tx_confirm, end-to-end

**Performance Requirements**
- Logging overhead: <1ms per log line
- Metrics recording: <100 microseconds
- Memory: <50MB for 1M data points

**Error Handling**
- **Log write failure**: Buffer logs, retry async
- **Metric export failure**: Log error, continue operation

**Testing**
- Unit: Verify JSON log format, metric calculations
- Integration: Test with real workload, verify percentiles
- Acceptance: <1% overhead on transaction throughput

---

### 13. Health Monitor

**Purpose**: Monitors system health and RPC availability.

**API Specification**
```python
class HealthMonitor:
    def __init__(
        self,
        rpc_manager: RPCManager,
        config: HealthConfig
    ):
        """Initialize health monitor"""

    async def start_monitoring(self):
        """Start health check loop"""

    async def stop_monitoring(self):
        """Stop health check loop"""

    async def get_system_health(self) -> SystemHealth:
        """Get overall system health status"""

    async def check_rpc_health(self) -> Dict[str, RPCHealth]:
        """Check health of all RPC endpoints"""

    def get_resource_usage(self) -> ResourceUsage:
        """Get CPU/memory usage"""
```

**Data Structures**
```python
@dataclass
class HealthConfig:
    check_interval_s: int = 10
    rpc_timeout_ms: int = 5000
    max_slot_lag: int = 50
    cpu_threshold_pct: float = 80.0
    memory_threshold_pct: float = 80.0

@dataclass
class SystemHealth:
    is_healthy: bool
    issues: List[str]
    rpcs_healthy: int
    rpcs_total: int
    cpu_usage_pct: float
    memory_usage_pct: float
    uptime_seconds: int

@dataclass
class RPCHealth:
    endpoint: str
    is_healthy: bool
    latency_ms: float
    slot: int
    slot_lag: int
    last_check: datetime

@dataclass
class ResourceUsage:
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    thread_count: int
```

**Implementation Notes**
- Use `psutil` for system resource monitoring
- Ping RPCs via `getHealth` and `getSlot` methods
- Compare slots across RPCs to detect lag
- Run health checks in background asyncio task

**Performance Requirements**
- Health check overhead: <100ms
- CPU monitoring: <1% overhead
- Check interval: Every 10 seconds

**Error Handling**
- **RPC unreachable**: Mark unhealthy, continue checks
- **Monitoring loop crash**: Auto-restart with backoff

**Testing**
- Unit: Mock RPC responses, verify health detection
- Integration: Test with real RPCs
- Acceptance: Detect unhealthy RPC within 20 seconds

---

### 14. Configuration Manager

**Purpose**: Loads runtime configuration from files and environment.

**API Specification**
```python
class ConfigurationManager:
    def __init__(self, config_path: str):
        """Initialize with path to config file"""

    def load_config(self) -> BotConfig:
        """Load and validate configuration"""

    def reload_config(self) -> BotConfig:
        """Hot-reload configuration"""

    def get(self, key: str, default: any = None) -> any:
        """Get config value by dot-notation key"""

    def set(self, key: str, value: any):
        """Set config value (runtime only)"""
```

**Data Structures**
```python
@dataclass
class BotConfig:
    # RPC settings
    rpc_endpoints: List[RPCEndpoint]
    rpc_config: RPCConfig

    # Wallet settings
    wallet_keypairs: List[str]  # Base58 private keys
    wallet_config: WalletConfig

    # Trading settings
    priority_fee_config: FeeConfig
    slippage_config: SlippageConfig

    # Strategy settings
    frontrun_config: Optional[FrontrunConfig]
    copytrade_config: Optional[CopytradeConfig]

    # System settings
    metrics_config: MetricsConfig
    log_config: LogConfig
    health_config: HealthConfig
```

**Implementation Notes**
- Use `pyyaml` for YAML parsing
- Support environment variable overrides: `${ENV_VAR}`
- Validate config against schema on load
- Watch config file for changes (optional hot-reload)

**Performance Requirements**
- Config load: <100ms
- Hot-reload: <500ms
- Memory: <10MB

**Error Handling**
- **File not found**: Raise `ConfigFileNotFoundError`
- **Invalid YAML**: Raise `ConfigParseError` with line number
- **Missing required field**: Raise `ConfigValidationError`

**Testing**
- Unit: Test with valid/invalid config files
- Integration: Test environment variable substitution
- Acceptance: Successfully load production config

---

## Frontrunning-Specific Features (15-26)

### 15. Mempool Transaction Monitor

**Purpose**: Listens to pending/unconfirmed transactions before block inclusion.

**API Specification**
```python
class MempoolMonitor:
    def __init__(
        self,
        rpc_manager: RPCManager,
        config: MempoolConfig
    ):
        """Initialize mempool monitor"""

    async def start_monitoring(
        self,
        callback: Callable[[PendingTransaction], Awaitable[None]]
    ):
        """Start monitoring with callback for each pending tx"""

    async def stop_monitoring(self):
        """Stop monitoring"""

    async def subscribe_to_address(self, address: Pubkey):
        """Subscribe to transactions involving specific address"""

    def get_mempool_stats(self) -> MempoolStats:
        """Get mempool monitoring statistics"""
```

**Data Structures**
```python
@dataclass
class MempoolConfig:
    geyser_endpoint: str
    program_filter: Optional[Pubkey] = None
    account_filters: List[Pubkey] = field(default_factory=list)

@dataclass
class PendingTransaction:
    signature: str
    transaction: VersionedTransaction
    received_at: datetime
    fee: int
    accounts: List[Pubkey]
    program_id: Pubkey

@dataclass
class MempoolStats:
    transactions_seen: int
    avg_time_to_inclusion_ms: float
    current_mempool_size: int
```

**Solana-Specific Details**
- Requires Geyser plugin or specialized RPC (Jito, Triton, etc.)
- Subscribe via gRPC stream: `SubscribeUpdate` with transaction filters
- Transactions may not be included (can be dropped)

**Implementation Notes**
- Use `grpcio` for Geyser plugin connection
- Filter transactions by program ID (pump.fun)
- Parse transaction to extract accounts and instructions
- Measure time from mempool seen to on-chain confirmation

**Performance Requirements**
- Transaction processing: <5ms per transaction
- Latency from RPC: <10ms
- Throughput: 1000 tx/sec

**Error Handling**
- **Connection lost**: Auto-reconnect with backoff
- **Invalid transaction**: Log and skip
- **Callback error**: Log error, continue monitoring

**Testing**
- Integration: Test with Geyser plugin on devnet
- Load: Simulate 1000 tx/sec stream
- Acceptance: No transactions dropped, <10ms processing latency

---

### 16. Dev Wallet Pattern Detector

**Purpose**: Identifies developer wallet behavior patterns.

**API Specification**
```python
class DevWalletDetector:
    def __init__(
        self,
        known_dev_wallets: List[Pubkey],
        config: DetectorConfig
    ):
        """Initialize with known dev wallet list"""

    def is_dev_buy_transaction(
        self,
        tx: VersionedTransaction,
        sender: Pubkey
    ) -> bool:
        """Check if transaction is a dev buy"""

    def add_known_dev_wallet(self, wallet: Pubkey):
        """Add wallet to known dev list"""

    def analyze_transaction_pattern(
        self,
        tx: VersionedTransaction
    ) -> TransactionPattern:
        """Analyze transaction for dev buy characteristics"""
```

**Data Structures**
```python
@dataclass
class DetectorConfig:
    min_buy_amount_sol: float = 0.5  # Devs typically buy 0.5+ SOL
    max_time_since_token_creation_s: int = 300  # Within 5 min of creation

@dataclass
class TransactionPattern:
    is_likely_dev_buy: bool
    confidence: float  # 0-1
    reasons: List[str]
    buy_amount_sol: Optional[float]
    mint: Optional[Pubkey]
```

**Implementation Notes**
- Maintain in-memory set of known dev wallets (loaded from file/API)
- Check transaction sender against known dev list
- Heuristics: large buy amount, early in token lifecycle, single instruction
- Update known dev wallets from community sources

**Performance Requirements**
- Detection latency: <1ms per transaction
- Memory: <10MB for 100k known wallets

**Error Handling**
- **Unknown wallet**: Mark confidence as low
- **Parse failure**: Return negative result

**Testing**
- Unit: Test with known dev buy transactions
- Acceptance: >90% accuracy on labeled dataset

---

### 17. Slot Prediction Engine

**Purpose**: Predicts which slot a transaction will land in.

**API Specification**
```python
class SlotPredictor:
    def __init__(
        self,
        rpc_manager: RPCManager,
        config: PredictorConfig
    ):
        """Initialize slot predictor"""

    async def predict_landing_slot(
        self,
        tx: VersionedTransaction,
        current_slot: int
    ) -> SlotPrediction:
        """Predict which slot transaction will land in"""

    async def get_leader_schedule(self) -> LeaderSchedule:
        """Get current epoch leader schedule"""

    def estimate_propagation_delay_ms(
        self,
        rpc_endpoint: str
    ) -> float:
        """Estimate network propagation delay"""
```

**Data Structures**
```python
@dataclass
class PredictorConfig:
    avg_slot_time_ms: int = 400
    network_propagation_ms: int = 50

@dataclass
class SlotPrediction:
    predicted_slot: int
    confidence: float
    earliest_slot: int
    latest_slot: int
    current_leader: str
    next_leader: str

@dataclass
class LeaderSchedule:
    epoch: int
    slot_leaders: Dict[int, str]  # slot -> validator pubkey
```

**Solana-Specific Details**
- Use `getLeaderSchedule` RPC method
- Slot time: ~400ms (target)
- Transactions land in next 1-4 slots typically

**Implementation Notes**
- Cache leader schedule per epoch (refresh every 2 days)
- Formula: `predicted_slot = current_slot + ceil((propagation_ms + processing_ms) / slot_time_ms)`
- Account for priority fee impact on inclusion probability

**Performance Requirements**
- Prediction latency: <5ms
- Leader schedule cache hit: >99%

**Error Handling**
- **Schedule not found**: Fetch from RPC
- **Stale schedule**: Auto-refresh on epoch boundary

**Testing**
- Integration: Compare predictions vs actual landing slots
- Acceptance: Within ±2 slots for >80% of transactions

---

### 18. Pre-Signed Transaction Templates

**Purpose**: Pre-signs skeleton transactions with placeholders for speed.

**API Specification**
```python
class TransactionTemplateManager:
    def __init__(
        self,
        signer: TransactionSigner,
        config: TemplateConfig
    ):
        """Initialize template manager"""

    async def create_buy_template(
        self,
        wallet: Pubkey,
        amount_sol: int,
        compute_units: int,
        priority_fee: int
    ) -> TransactionTemplate:
        """Create pre-signed buy transaction template"""

    async def instantiate_template(
        self,
        template: TransactionTemplate,
        mint: Pubkey
    ) -> VersionedTransaction:
        """Swap mint address into template"""

    def invalidate_templates(self, wallet: Pubkey):
        """Invalidate all templates for wallet (on blockhash expiry)"""
```

**Data Structures**
```python
@dataclass
class TemplateConfig:
    template_ttl_seconds: int = 30  # Match blockhash validity
    max_templates_per_wallet: int = 10

@dataclass
class TransactionTemplate:
    template_id: str
    wallet: Pubkey
    signed_tx_bytes: bytes
    mint_placeholder_offset: int
    created_at: datetime
    blockhash: str
    expires_at: datetime
```

**Implementation Notes**
- Use dummy mint address (all zeros) in template
- Store byte offset of mint address in serialized transaction
- Replace mint bytes at runtime with `memcpy`-like operation
- Invalidate templates after 30 seconds (blockhash expiry)

**Performance Requirements**
- Template creation: <50ms
- Mint substitution: <1ms
- Memory: ~1KB per template

**Error Handling**
- **Expired template**: Recreate with fresh blockhash
- **Invalid substitution**: Log error, rebuild transaction normally

**Testing**
- Unit: Verify mint substitution correctness
- Integration: Submit instantiated templates on devnet
- Acceptance: 10-20ms savings vs fresh transaction build

---

### 19. Aggressive Priority Fee Bidder

**Purpose**: Dynamically sets very high priority fees to guarantee first position.

**API Specification**
```python
class AggressivePriorityFeeBidder:
    def __init__(
        self,
        fee_calculator: PriorityFeeCalculator,
        config: BidderConfig
    ):
        """Initialize aggressive bidder"""

    async def calculate_aggressive_fee(
        self,
        competing_tx: Optional[PendingTransaction] = None,
        compute_units: int = 200_000
    ) -> int:
        """Calculate fee to outbid competition"""

    def get_max_fee_for_profit(
        self,
        expected_profit_sol: float,
        max_fee_pct: float = 0.5
    ) -> int:
        """Calculate max fee as percentage of expected profit"""
```

**Data Structures**
```python
@dataclass
class BidderConfig:
    base_multiplier: float = 10.0  # 10x average fee
    competition_multiplier: float = 1.5  # 1.5x competing tx fee
    max_fee_lamports: int = 10_000_000  # 0.01 SOL max
    min_fee_lamports: int = 100_000  # 0.0001 SOL min
```

**Implementation Notes**
- If competing tx detected: bid = competing_fee * 1.5
- Otherwise: bid = avg_fee * 10
- Clamp between min and max fee limits
- Consider profit: never bid >50% of expected profit

**Performance Requirements**
- Fee calculation: <1ms
- Memory: Negligible

**Error Handling**
- **No fee data**: Use max_fee_lamports
- **Negative profit**: Return 0 (abort trade)

**Testing**
- Unit: Test multiplier logic with various inputs
- Acceptance: Win >90% of frontrun races against competition

---

### 20. Dev Buy Confirmation Detector

**Purpose**: Detects the exact moment dev buy transaction confirms.

**API Specification**
```python
class DevBuyConfirmationDetector:
    def __init__(
        self,
        rpc_manager: RPCManager,
        pump_client: PumpFunClient
    ):
        """Initialize confirmation detector"""

    async def wait_for_dev_buy_confirmation(
        self,
        mint: Pubkey,
        timeout_seconds: int = 30
    ) -> Optional[DevBuyConfirmation]:
        """Wait for dev buy to confirm, return confirmation details"""

    async def subscribe_bonding_curve_changes(
        self,
        mint: Pubkey,
        callback: Callable[[BondingCurveChange], Awaitable[None]]
    ):
        """Subscribe to bonding curve account changes"""
```

**Data Structures**
```python
@dataclass
class DevBuyConfirmation:
    mint: Pubkey
    dev_buy_signature: str
    confirmed_slot: int
    confirmed_at: datetime
    bonding_curve_state: BondingCurveAccount
    dev_buy_amount_sol: float

@dataclass
class BondingCurveChange:
    mint: Pubkey
    old_state: BondingCurveAccount
    new_state: BondingCurveAccount
    slot: int
    reserve_ratio_change: float
```

**Solana-Specific Details**
- Subscribe via `programSubscribe` to bonding curve account
- Detect reserve ratio change: `sol_reserves / token_reserves`
- Large ratio change indicates dev buy

**Implementation Notes**
- Use RPC account subscription on bonding curve PDA
- Compare old vs new virtual reserves
- Threshold: >10% reserve ratio change = dev buy
- Track timestamp with microsecond precision

**Performance Requirements**
- Detection latency: <100ms from confirmation
- Subscription overhead: <10ms

**Error Handling**
- **Timeout**: Return None
- **Subscription lost**: Reconnect, resume
- **False positive**: Verify with transaction signature lookup

**Testing**
- Integration: Test with simulated dev buys on devnet
- Acceptance: Detect within 200ms of on-chain confirmation

---

### 21. Same-Slot Bundle Constructor

**Purpose**: Bundles buy + sell in same slot/block.

**API Specification**
```python
class SameSlotBundleConstructor:
    def __init__(
        self,
        tx_builder: TransactionBuilder,
        rpc_manager: RPCManager
    ):
        """Initialize bundle constructor"""

    async def build_buy_sell_bundle(
        self,
        mint: Pubkey,
        wallet: Keypair,
        buy_amount_sol: int,
        sell_slippage_bps: int
    ) -> TransactionBundle:
        """Build buy+sell bundle for same slot execution"""

    async def submit_bundle(
        self,
        bundle: TransactionBundle,
        tip_lamports: int = 10_000
    ) -> BundleResult:
        """Submit bundle to Jito or other bundler"""
```

**Data Structures**
```python
@dataclass
class TransactionBundle:
    buy_tx: VersionedTransaction
    sell_tx: VersionedTransaction
    bundle_id: str

@dataclass
class BundleResult:
    bundle_id: str
    accepted: bool
    buy_signature: Optional[str]
    sell_signature: Optional[str]
    landing_slot: Optional[int]
```

**Solana-Specific Details**
- Requires Jito block engine or similar MEV infrastructure
- Use Jito JSON-RPC: `sendBundle` method
- Bundle format: array of base64-encoded signed transactions
- Add tip instruction to searcher tip account

**Implementation Notes**
- Use same blockhash for both transactions
- Buy must execute before sell (order matters)
- Add tip instruction to last transaction (sell)
- Jito tip accounts rotate per epoch

**Performance Requirements**
- Bundle construction: <20ms
- Submission: <100ms

**Error Handling**
- **Bundle rejected**: Retry with higher tip
- **Partial execution**: Handle buy-only scenario
- **Timeout**: Abort after 10 seconds

**Testing**
- Integration: Test on Jito devnet
- Acceptance: >50% bundle acceptance rate

---

### 22. Race Failure Detector

**Purpose**: Identifies when bot lost the race to dev buy.

**API Specification**
```python
class RaceFailureDetector:
    def __init__(self, config: RaceConfig):
        """Initialize race detector"""

    def detect_race_loss(
        self,
        expected_price: float,
        actual_fill_price: float,
        threshold_pct: float = 5.0
    ) -> RaceResult:
        """Detect if entry price indicates lost race"""

    def analyze_fill_timing(
        self,
        dev_buy_slot: int,
        our_fill_slot: int
    ) -> RaceResult:
        """Analyze slot timing to determine race outcome"""
```

**Data Structures**
```python
@dataclass
class RaceConfig:
    price_deviation_threshold_pct: float = 5.0
    max_acceptable_slot_delay: int = 2

@dataclass
class RaceResult:
    won_race: bool
    reason: str
    price_deviation_pct: float
    slot_delay: int
```

**Implementation Notes**
- Compare fill price to expected pre-dev price
- If fill price >5% higher → lost race (entered after dev)
- If our fill slot > dev buy slot + 2 → lost race
- Use for abort decisions and post-trade analysis

**Performance Requirements**
- Detection: <1ms
- Memory: Negligible

**Error Handling**
- **Missing data**: Assume race lost (conservative)

**Testing**
- Unit: Test with known race scenarios
- Acceptance: 100% accuracy on labeled race outcomes

---

### 23. Ultra-Short TTL Exit Logic

**Purpose**: Forces exit within 5-20 seconds regardless of price.

**API Specification**
```python
class UltraShortTTLExitManager:
    def __init__(self, config: ExitConfig):
        """Initialize exit manager"""

    async def monitor_position_ttl(
        self,
        position: Position,
        exit_callback: Callable[[Position], Awaitable[None]]
    ):
        """Monitor position and trigger exit at TTL"""

    def should_exit(
        self,
        position: Position,
        current_time: datetime,
        dev_buy_detected: bool
    ) -> bool:
        """Determine if position should be exited"""

    def get_exit_slippage(
        self,
        time_remaining_seconds: float
    ) -> int:
        """Get slippage tolerance based on remaining TTL"""
```

**Data Structures**
```python
@dataclass
class ExitConfig:
    max_ttl_seconds: int = 20
    target_exit_window_seconds: int = 5  # After dev buy
    emergency_slippage_bps: int = 2000  # 20% if near expiry

@dataclass
class ExitDecision:
    should_exit: bool
    reason: str
    slippage_bps: int
    urgency: TradeUrgency
```

**Implementation Notes**
- Start timer on position entry
- Exit triggers: dev buy confirmed OR 20s elapsed
- Slippage increases with time: 5% → 10% → 20%
- Use asyncio.create_task for TTL monitoring per position

**Performance Requirements**
- Timer precision: ±100ms
- Exit execution: <200ms from trigger

**Error Handling**
- **Exit failure**: Retry with increased slippage (up to 20%)
- **Timeout**: Force exit at market (no slippage limit)

**Testing**
- Unit: Test TTL triggers at various intervals
- Integration: Test forced exits on devnet
- Acceptance: All positions exited within 25 seconds

---

### 24. Deterministic Event Sequencer

**Purpose**: Ensures events are processed in exact order received.

**API Specification**
```python
class DeterministicEventSequencer:
    def __init__(self, config: SequencerConfig):
        """Initialize event sequencer"""

    async def enqueue_event(
        self,
        event: Event,
        sequence_number: int
    ):
        """Add event to sequence queue"""

    async def process_events(
        self,
        handler: Callable[[Event], Awaitable[None]]
    ):
        """Process events in strict order"""

    def detect_out_of_order(
        self,
        event: Event,
        expected_sequence: int
    ) -> bool:
        """Detect if event is out of sequence"""
```

**Data Structures**
```python
@dataclass
class SequencerConfig:
    max_queue_size: int = 10_000
    out_of_order_tolerance: int = 10  # Buffer 10 events

@dataclass
class Event:
    event_id: str
    event_type: str
    sequence_number: int
    timestamp: datetime
    data: dict
```

**Implementation Notes**
- Use asyncio.Queue with sequence number tracking
- Maintain buffer for out-of-order events (reorder window)
- Use monotonic clock for event timestamps
- Discard events with sequence gap >10

**Performance Requirements**
- Enqueue latency: <100 microseconds
- Processing latency: <1ms per event
- Max queue size: 10k events (~10MB)

**Error Handling**
- **Queue full**: Drop oldest events, log warning
- **Sequence gap**: Wait 100ms for missing event, then skip
- **Duplicate sequence**: Ignore duplicate

**Testing**
- Unit: Test with out-of-order event streams
- Acceptance: 100% ordering guarantee within tolerance window

---

### 25. Latency Budget Enforcer

**Purpose**: Aborts attempt if any pipeline stage exceeds latency budget.

**API Specification**
```python
class LatencyBudgetEnforcer:
    def __init__(self, config: LatencyConfig):
        """Initialize latency enforcer"""

    def start_operation(self, operation_id: str) -> OperationTimer:
        """Start timing an operation"""

    def check_budget(
        self,
        timer: OperationTimer,
        stage: str
    ) -> BudgetCheckResult:
        """Check if stage exceeded budget"""

    def abort_if_exceeded(
        self,
        timer: OperationTimer
    ) -> bool:
        """Check if total budget exceeded, return should_abort"""
```

**Data Structures**
```python
@dataclass
class LatencyConfig:
    detect_stage_budget_ms: float = 10
    decide_stage_budget_ms: float = 5
    build_stage_budget_ms: float = 10
    sign_stage_budget_ms: float = 5
    submit_stage_budget_ms: float = 50
    total_budget_ms: float = 100

@dataclass
class OperationTimer:
    operation_id: str
    start_time: float  # perf_counter
    stage_times: Dict[str, float]

@dataclass
class BudgetCheckResult:
    stage: str
    elapsed_ms: float
    budget_ms: float
    exceeded: bool
    should_abort: bool
```

**Implementation Notes**
- Use `time.perf_counter()` for microsecond precision
- Track cumulative time across stages
- Abort if any stage exceeds budget OR total >100ms
- Log all aborts with timing breakdown

**Performance Requirements**
- Timer overhead: <10 microseconds
- Check overhead: <1 microsecond

**Error Handling**
- **Budget exceeded**: Log timing details, abort operation
- **Timer not found**: Create new timer, log warning

**Testing**
- Unit: Test with simulated delays at each stage
- Acceptance: 100% abort rate when budget exceeded

---

### 26. Co-location Network Optimizer

**Purpose**: Routes traffic through lowest-latency paths to validators.

**API Specification**
```python
class CoLocationOptimizer:
    def __init__(self, config: CoLocationConfig):
        """Initialize network optimizer"""

    async def measure_validator_latencies(
        self
    ) -> Dict[str, float]:
        """Measure RTT to each validator"""

    async def get_optimal_rpc_for_leader(
        self,
        leader_pubkey: str
    ) -> str:
        """Get RPC endpoint closest to current leader"""

    def configure_quic_params(self) -> QUICConfig:
        """Get optimized QUIC protocol parameters"""
```

**Data Structures**
```python
@dataclass
class CoLocationConfig:
    enable_direct_validator_connections: bool = False
    preferred_regions: List[str] = field(default_factory=lambda: ["us-east", "us-west"])

@dataclass
class QUICConfig:
    max_idle_timeout_ms: int = 10_000
    max_datagram_size: int = 1350
    congestion_control: str = "bbr"
```

**Solana-Specific Details**
- Validators run TPU on UDP port 1024-1123 (QUIC)
- Solana uses QUIC for transaction forwarding
- RPCs may be co-located with validators (check IP ranges)

**Implementation Notes**
- Ping validators via QUIC probe packets
- Maintain latency map: validator_pubkey → RTT
- Route transactions via RPC nearest to current slot leader
- Advanced: Direct QUIC connection to validator TPU (requires `quic` library)

**Performance Requirements**
- Latency measurement: <50ms per validator
- Routing decision: <1ms
- Latency improvement: 10-50ms vs random RPC

**Error Handling**
- **Validator unreachable**: Fall back to default RPC
- **Measurement timeout**: Use stale latency data

**Testing**
- Integration: Measure real validator latencies on mainnet
- Acceptance: Route to optimal endpoint >90% of time

---

## Cross-Cutting Concerns

### Error Recovery Strategy
- **RPC failures**: Automatic failover to backup endpoints
- **Transaction failures**: Retry with fresh blockhash (max 3 attempts)
- **Mempool miss**: Abort frontrun attempt
- **Partial fills**: Track actual vs expected, adjust position size
- **System overload**: Circuit breaker pattern (pause new trades)

### Security Considerations
- **Private keys**: Never log or persist to disk (memory only)
- **RPC endpoints**: Use authenticated endpoints with API keys
- **Rate limiting**: Implement per-endpoint rate limits
- **Input validation**: Validate all pubkeys, amounts, percentages
- **Replay attacks**: Use recent blockhash (60s validity)

### Observability Requirements
- **Metrics to track**: Latency (p50/p95/p99), throughput, error rate, PnL, win rate
- **Logs to emit**: All transactions (signature, slot, amount, PnL), errors with stack traces, performance warnings
- **Dashboards**: Real-time PnL, latency heatmap, RPC health, position overview

### Testing Strategy
- **Unit tests**: All calculators, builders, validators (>80% coverage)
- **Integration tests**: Real RPC calls on devnet, full trade flow
- **Load tests**: 1000 concurrent positions, 100 tx/sec submission rate
- **Chaos tests**: Random RPC failures, network delays, out-of-order events

---

## Implementation Priority

**Phase 1: Core Infrastructure (Weeks 1-2)**
1. Multi-RPC Connection Manager
2. Transaction Builder
3. Transaction Signer
4. Transaction Submitter
5. Configuration Manager
6. Metrics & Logging System

**Phase 2: Trading Primitives (Week 3)**
7. Pump.fun Program Client
8. Bonding Curve Calculator
9. Slippage Manager
10. Priority Fee Calculator
11. Multi-Wallet Manager
12. Position Tracker
13. PnL Calculator

**Phase 3: System Reliability (Week 4)**
14. Health Monitor
15. Error recovery patterns
16. Integration testing

**Phase 4: Frontrun Features (Weeks 5-6)**
17. Mempool Transaction Monitor
18. Dev Wallet Pattern Detector
19. Aggressive Priority Fee Bidder
20. Dev Buy Confirmation Detector
21. Latency Budget Enforcer
22. Race Failure Detector
23. Ultra-Short TTL Exit Logic

**Phase 5: Advanced Optimization (Week 7)**
24. Slot Prediction Engine
25. Pre-Signed Transaction Templates
26. Same-Slot Bundle Constructor
27. Deterministic Event Sequencer
28. Co-location Network Optimizer

**Phase 6: Production Hardening (Week 8)**
29. Load testing
30. Chaos engineering
31. Production deployment
32. Monitoring and alerting setup

---

## Acceptance Criteria per Feature

Each feature must meet:
1. **Performance**: All latency/throughput requirements met
2. **Testing**: Unit + integration tests passing
3. **Documentation**: API documented with examples
4. **Observability**: Metrics and logs implemented
5. **Error handling**: All error cases handled gracefully
6. **Integration**: Works with dependent features

---

This specification provides actionable implementation details for all 26 features. Each section includes concrete APIs, data structures, performance targets, and testing requirements needed for development.
