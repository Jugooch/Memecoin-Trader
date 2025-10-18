"""
Data types for anti-bot wallet classification system
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime
from enum import Enum


class WalletLabel(str, Enum):
    """Wallet classification labels"""
    UNKNOWN = "unknown"
    DEPLOYER = "deployer"
    SYBIL = "sybil"
    METRIC_BOT = "metric_bot"
    COPY_BOT = "copy_bot"
    PRIORITY_RACER = "priority_racer"
    MOMENTUM_SCALPER = "momentum_scalper"
    BAIT_CLUSTER = "bait_cluster"
    ORGANIC = "organic"
    EARLY_EXIT = "early_exit"


@dataclass
class TxFeatures:
    """Extracted features from a single transaction"""
    slot: int
    entry_index: int  # Position within slot (if available)
    signature: str
    signer: str
    timestamp: datetime

    # Program activity
    programs: List[str]
    ix_types: List[str]  # Decoded instruction types (Create, Buy, Sell, etc.)

    # Priority & fees
    cu_price: int  # Lamports per CU (priority fee)
    cu_consumed: int
    cu_requested: int
    base_fee_lamports: int
    jito_tip_lamports: int = 0

    # Token deltas
    mint: Optional[str] = None
    token_delta: float = 0.0  # Tokens bought/sold
    sol_delta: float = 0.0  # SOL spent/received

    # Trade specifics
    is_buy: bool = True
    sol_amount: float = 0.0
    token_amount: float = 0.0

    # Curve state (from TradeEvent)
    virtual_sol_reserves: int = 0
    virtual_token_reserves: int = 0

    # Logs sample
    logs_sample: List[str] = field(default_factory=list)

    # Metadata
    tx_size_bytes: int = 0


@dataclass
class WalletProfile:
    """Rolling profile for a wallet"""
    address: str
    label: WalletLabel = WalletLabel.UNKNOWN
    confidence: float = 0.0  # 0.0 to 1.0
    observations: int = 0

    # Funding analysis
    funding_parents: List[str] = field(default_factory=list)  # Who funded this wallet
    funding_age_seconds: float = 0.0  # Time since last funding

    # Sibling detection
    sibling_wallets: Set[str] = field(default_factory=set)  # Wallets funded by same parent

    # Transaction patterns
    bundle_rate: float = 0.0  # % of txs with Jito tip
    mean_entry_index: float = 0.0  # Average position in slot
    mean_cu_price: float = 0.0

    # Trading behavior
    trades_count: int = 0
    buys_count: int = 0
    sells_count: int = 0
    median_hold_sec: float = 0.0
    mean_buy_size_sol: float = 0.0

    # Performance
    realized_pnl_sol: float = 0.0
    median_pnl_pct: float = 0.0
    win_rate: float = 0.0

    # Pattern signatures
    mints_traded: Set[str] = field(default_factory=set)
    mints_created: Set[str] = field(default_factory=set)

    # Time tracking
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

    # Evidence tracking
    label_reasons: List[str] = field(default_factory=list)


@dataclass
class MintState:
    """Rolling state for a token mint"""
    mint: str
    created_at: datetime
    created_at_slot: int = 0
    creator: str = ""

    # Buyer tracking
    unique_buyers: Set[str] = field(default_factory=set)
    unique_sellers: Set[str] = field(default_factory=set)

    # Volume windows (1s, 3s, 10s rolling)
    buy_count_1s: int = 0
    buy_count_3s: int = 0
    buy_count_10s: int = 0
    sol_volume_1s: float = 0.0
    sol_volume_3s: float = 0.0
    sol_volume_10s: float = 0.0

    # Distribution metrics
    buy_sizes: List[float] = field(default_factory=list)
    gini_buy_size: float = 0.0

    # Sybil detection
    sybil_buyers: Set[str] = field(default_factory=set)
    sybil_ratio: float = 0.0  # sybil buyers / total buyers

    # Priority metrics
    cu_prices: List[int] = field(default_factory=list)
    median_cu_price: int = 0
    p90_cu_price: int = 0

    # Momentum
    has_second_burst: bool = False
    burst_count: int = 0

    # Curve state
    current_sol_reserves: int = 30_000_000_000  # 30 SOL initial
    peak_sol_reserves: int = 30_000_000_000

    # Cohort classification
    is_organic: bool = False
    is_wash_trade: bool = False

    # Time tracking
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class BurstEvent:
    """A detected burst of activity"""
    mint: str
    start_time: datetime
    end_time: datetime
    buy_count: int
    sol_volume: float
    unique_wallets: int
    triggered_bots: List[str] = field(default_factory=list)  # Wallets that bought 1-4s after


@dataclass
class ClusterProfile:
    """A cluster of related wallets (deployer + sybils)"""
    cluster_id: str
    deployer: str
    sybil_wallets: Set[str] = field(default_factory=set)
    mints_deployed: Set[str] = field(default_factory=set)

    # Pattern
    avg_time_to_rug_seconds: float = 0.0
    success_rate: float = 0.0

    # Identification
    first_seen: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0


@dataclass
class ClassificationMetrics:
    """Overall system metrics"""
    total_wallets_tracked: int = 0
    total_mints_tracked: int = 0

    # Label distribution
    label_counts: Dict[WalletLabel, int] = field(default_factory=dict)

    # Detection stats
    sybil_clusters_found: int = 0
    bait_clusters_found: int = 0
    metric_bots_found: int = 0

    # Performance
    organic_cohort_accuracy: float = 0.0
    wash_trade_detection_rate: float = 0.0
