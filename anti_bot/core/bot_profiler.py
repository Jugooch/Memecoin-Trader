"""
Bot behavior profiler and strategy analyzer

Generates detailed profiles of detected bots and strategies for exploiting/avoiding them
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics

from anti_bot.core.types import WalletLabel, WalletProfile


@dataclass
class BotBehaviorProfile:
    """Detailed behavior profile for a specific bot"""
    wallet: str
    label: WalletLabel
    confidence: float

    # Trading characteristics
    total_trades: int = 0
    buys_count: int = 0
    sells_count: int = 0
    mints_traded: int = 0

    # Timing patterns
    median_hold_time_sec: float = 0.0
    mean_buy_size_sol: float = 0.0
    mean_cu_price: int = 0
    burst_follow_rate: float = 0.0  # For metric bots

    # Advanced indicators
    cluster_size: int = 0  # For sybils
    early_lander_rate: float = 0.0  # For priority racers
    first_seller_count: int = 0  # For early exit specialists

    # Profitability indicators (estimated)
    estimated_win_rate: float = 0.0
    typical_exit_multiplier: float = 0.0

    # Behavioral signature
    signature: str = ""  # Human-readable description


@dataclass
class BotTypeStrategy:
    """Strategy recommendations for a specific bot type"""
    bot_type: WalletLabel

    # How the bot operates
    operation_description: str = ""
    typical_timing: str = ""
    common_characteristics: List[str] = field(default_factory=list)

    # How to avoid getting exploited
    avoidance_strategies: List[str] = field(default_factory=list)

    # How to exploit their predictable behavior
    exploitation_strategies: List[str] = field(default_factory=list)

    # Detection signals
    detection_signals: List[str] = field(default_factory=list)

    # Risk assessment
    threat_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL


# Pre-defined strategy templates
STRATEGY_TEMPLATES: Dict[WalletLabel, BotTypeStrategy] = {
    WalletLabel.SYBIL: BotTypeStrategy(
        bot_type=WalletLabel.SYBIL,
        operation_description=(
            "Sybil clusters use multiple wallets funded by same source to simulate organic demand. "
            "They coordinate micro-buys (≤0.02 SOL) with tight timing (<250ms gaps) to pump volume metrics."
        ),
        typical_timing="Synchronized buys within 0-5 seconds of each other",
        common_characteristics=[
            "Funded by same parent wallet within 90s window",
            "Micro-buys ≤0.02 SOL per wallet",
            "Very tight timing coordination (<250ms median gap)",
            "6+ wallets in cluster",
            "Similar CU prices across cluster"
        ],
        avoidance_strategies=[
            "Exit immediately if you see coordinated micro-buys flooding in",
            "Check if recent buyers share funding parents (on-chain analysis)",
            "Avoid tokens with >50% micro-buys in first 30s",
            "Set sell triggers if volume suddenly drops (sybils exiting)"
        ],
        exploitation_strategies=[
            "Front-run the sybil dump: Sell when you detect synchronized sell signals",
            "Wait for sybil exit before entering (better entry after fake pump clears)",
            "Short opportunities: Enter after sybil pump peaks, exit when real buyers arrive",
            "Copy their deployer's next token (if sybils profitable, deployer may repeat)"
        ],
        detection_signals=[
            "Sudden burst of 6+ micro-buys within 3 seconds",
            "Low Gini coefficient (<0.3) indicating uniform buy sizes",
            "Multiple wallets with identical CU prices",
            "Funding graph shows common parent"
        ],
        threat_level="HIGH"
    ),

    WalletLabel.METRIC_BOT: BotTypeStrategy(
        bot_type=WalletLabel.METRIC_BOT,
        operation_description=(
            "Metric bots monitor on-chain activity and auto-buy when they detect volume/momentum bursts. "
            "They follow a template: detect burst → wait 0.5-5s → execute buy → flip within 25s."
        ),
        typical_timing="Buys 0.5-5 seconds after detecting a burst (15+ buys, 2.5+ SOL in 3s)",
        common_characteristics=[
            "50-70% of buys occur 0.5-5s after bursts",
            "Low inter-arrival CV (<0.4) - template timing",
            "Consistent delay after bursts (std <0.3s)",
            "Template CU prices (variance <100k)",
            "Fast flips: median hold <25 seconds"
        ],
        avoidance_strategies=[
            "Don't market buy during/after bursts - you're competing with bots",
            "Wait 5-10s after burst peak to see if metric bots dump",
            "If you see fast flips (25s holds) after every burst, avoid the token",
            "Use limit orders below market to avoid bot-inflated prices"
        ],
        exploitation_strategies=[
            "FRONT-RUN BURST: Buy 1-2s into a forming burst, sell when bots arrive at 3-5s",
            "FADE THE BOTS: Short after bot buying wave completes (5-10s after burst)",
            "PREDICT DUMPS: Metric bots hold <25s, so sell 20-23s after burst",
            "BE FASTER: Use lower latency setup to beat bots to burst detection"
        ],
        detection_signals=[
            "Sudden buy volume spike 2-5s after initial burst",
            "Multiple wallets buying with identical CU prices",
            "Sells occurring exactly 20-30s after buys (algorithmic timing)",
            "Inter-arrival times showing template pattern"
        ],
        threat_level="CRITICAL"
    ),

    WalletLabel.PRIORITY_RACER: BotTypeStrategy(
        bot_type=WalletLabel.PRIORITY_RACER,
        operation_description=(
            "Priority racers pay premium fees to land early in blocks. They use high CU prices (>p90+30%), "
            "Jito bundles, and validator connections to consistently win transaction ordering races."
        ),
        typical_timing="First 10% of transactions in each block/slot",
        common_characteristics=[
            "High CU rate: 60%+ trades above adaptive p90 threshold",
            "Early lander rate: 30%+ in first 10% of slot",
            "High Jito usage: 50%+ bundled transactions",
            "Mean entry_index ≤2 (very early in slot)",
            "Leader concentration (sticky to specific validators)"
        ],
        avoidance_strategies=[
            "Don't compete on speed - you'll lose to priority racers",
            "Wait for racer exits (they flip fast for small gains)",
            "Avoid buying immediately after token creation (racers front-run)",
            "Use different strategy: patience, not speed"
        ],
        exploitation_strategies=[
            "FOLLOW THEIR EXITS: Priority racers sell at predictable points, buy their dips",
            "SLOWER TOKENS: Trade tokens racers ignore (low initial volume)",
            "COPY SETUPS: Analyze which validators they target, use same for your trades",
            "BACKEND VALUE: Enter after racers exit, ride organic wave they miss"
        ],
        detection_signals=[
            "Consistently landing in first 3 transactions of each new token",
            "CU prices 2-5x higher than median",
            "Heavy Jito bundle usage",
            "Same wallet winning race repeatedly across multiple mints"
        ],
        threat_level="MEDIUM"
    ),

    WalletLabel.EARLY_EXIT: BotTypeStrategy(
        bot_type=WalletLabel.EARLY_EXIT,
        operation_description=(
            "Early exit specialists predict failed launches and dump before momentum dies. "
            "They track second burst timing - if no momentum within 30-60s, they're first sellers."
        ),
        typical_timing="First to sell within 30-60s of token creation if no second burst",
        common_characteristics=[
            "First seller 40-60% of the time",
            "High prediction accuracy (60%+ on failed launches)",
            "Median hold <30s on sells",
            "Only sells on tokens without second burst",
            "Often profitable despite fast exits"
        ],
        avoidance_strategies=[
            "If you see early exit specialist enter → watch for their sell as signal",
            "Their sell = strong signal token will fail (60%+ accuracy)",
            "Don't hold through their exit - follow their lead or exit before",
            "Tokens they avoid = better momentum potential"
        ],
        exploitation_strategies=[
            "INVERSE SIGNAL: If they DON'T sell after 60s, token has momentum - hold",
            "COPY EXITS: Follow their sells within 1-2s for safe exits on failed launches",
            "FADE FALSE ALARMS: Rare times they're wrong (40%), momentum is explosive",
            "ENTRY TIMING: If they sold and were wrong, re-enter at panic bottom"
        ],
        detection_signals=[
            "Wallet consistently first seller on failed launches",
            "Sells occurring 30-60s after entry if no second burst",
            "High win rate despite fast exits",
            "Never holds through failed launches"
        ],
        threat_level="LOW"
    ),

    WalletLabel.DEPLOYER: BotTypeStrategy(
        bot_type=WalletLabel.DEPLOYER,
        operation_description=(
            "Token deployers create memecoins. Track their history to identify: "
            "serial scammers (rug pulls), quality deployers (successful tokens), farming bots."
        ),
        typical_timing="Creates token, sometimes buys immediately after",
        common_characteristics=[
            "Created 1+ tokens",
            "May hold large supply",
            "Often buys own token to seed liquidity",
            "Behavior varies: some rug, some legit"
        ],
        avoidance_strategies=[
            "Check deployer's token history before entering",
            "Avoid deployers with >5 failed tokens (rug pull pattern)",
            "Avoid if deployer holds >20% supply unfairly",
            "Exit if deployer dumps large position"
        ],
        exploitation_strategies=[
            "COPY GOOD DEPLOYERS: Track successful deployers, snipe their new tokens",
            "FADE BAD DEPLOYERS: Short tokens from known rug pullers",
            "TRACK SYBIL DEPLOYERS: If deployer uses sybils successfully, copy their next launch",
            "EARLY WARNING: Deployer sells = exit signal for everyone"
        ],
        detection_signals=[
            "Wallet that created the token contract",
            "Often holds founder allocation",
            "May have created multiple tokens (check history)"
        ],
        threat_level="VARIES"
    ),
}


class BotProfiler:
    """Generates detailed bot behavior profiles and strategy analysis"""

    def __init__(self):
        self.profiles: List[BotBehaviorProfile] = []

    def generate_profile(self, wallet_profile: WalletProfile,
                        classifier_data: Dict) -> BotBehaviorProfile:
        """
        Generate detailed behavior profile for a wallet

        Args:
            wallet_profile: WalletProfile from classifier
            classifier_data: Additional data from classifier (hold times, etc.)

        Returns:
            BotBehaviorProfile with detailed analysis
        """
        profile = BotBehaviorProfile(
            wallet=wallet_profile.address,
            label=wallet_profile.label,
            confidence=wallet_profile.confidence,
            total_trades=wallet_profile.trades_count,
            buys_count=wallet_profile.buys_count,
            sells_count=wallet_profile.sells_count,
            mints_traded=len(wallet_profile.mints_traded),
            mean_buy_size_sol=wallet_profile.mean_buy_size_sol,
            mean_cu_price=int(wallet_profile.mean_cu_price)
        )

        # Extract behavior-specific features from classifier_data
        if 'hold_times' in classifier_data and classifier_data['hold_times']:
            profile.median_hold_time_sec = statistics.median(classifier_data['hold_times'])

        if 'cluster_size' in classifier_data:
            profile.cluster_size = classifier_data['cluster_size']

        if 'burst_follows' in classifier_data and classifier_data.get('total_buys', 0) > 0:
            profile.burst_follow_rate = classifier_data['burst_follows'] / classifier_data['total_buys']

        if 'early_trades' in classifier_data and profile.total_trades > 0:
            profile.early_lander_rate = classifier_data['early_trades'] / profile.total_trades

        if 'first_seller_count' in classifier_data:
            profile.first_seller_count = classifier_data['first_seller_count']

        # Generate signature
        profile.signature = self._generate_signature(profile)

        # Estimate profitability (simplified)
        profile.estimated_win_rate = self._estimate_win_rate(profile)
        profile.typical_exit_multiplier = self._estimate_exit_multiplier(profile)

        self.profiles.append(profile)
        return profile

    def _generate_signature(self, profile: BotBehaviorProfile) -> str:
        """Generate human-readable behavioral signature"""
        parts = []

        if profile.label == WalletLabel.SYBIL:
            parts.append(f"Sybil cluster ({profile.cluster_size} wallets)")
            parts.append(f"micro-buys avg {profile.mean_buy_size_sol:.4f} SOL")
        elif profile.label == WalletLabel.METRIC_BOT:
            parts.append(f"Metric bot ({profile.burst_follow_rate:.0%} burst-triggered)")
            parts.append(f"flips in {profile.median_hold_time_sec:.0f}s")
        elif profile.label == WalletLabel.PRIORITY_RACER:
            parts.append(f"Priority racer ({profile.early_lander_rate:.0%} early lands)")
            parts.append(f"avg CU {profile.mean_cu_price:,}")
        elif profile.label == WalletLabel.EARLY_EXIT:
            parts.append(f"Early exit specialist ({profile.first_seller_count}x first seller)")
            parts.append(f"exits in {profile.median_hold_time_sec:.0f}s")
        elif profile.label == WalletLabel.DEPLOYER:
            # mints_traded is already an int, not a set
            parts.append(f"Deployer ({profile.mints_traded} tokens created)")

        return " | ".join(parts) if parts else "Unknown pattern"

    def _estimate_win_rate(self, profile: BotBehaviorProfile) -> float:
        """Estimate win rate based on behavior type"""
        # Simplified heuristic
        if profile.label == WalletLabel.EARLY_EXIT:
            return 0.65  # Early exit specialists have good accuracy
        elif profile.label == WalletLabel.METRIC_BOT:
            return 0.55  # Metric bots win on speed
        elif profile.label == WalletLabel.PRIORITY_RACER:
            return 0.60  # Priority racers get good entries
        elif profile.label == WalletLabel.SYBIL:
            return 0.45  # Sybils often rug or fail
        return 0.50

    def _estimate_exit_multiplier(self, profile: BotBehaviorProfile) -> float:
        """Estimate typical exit multiplier (1.0 = break even)"""
        # Simplified heuristic
        if profile.label == WalletLabel.PRIORITY_RACER:
            return 1.15  # Small but consistent gains
        elif profile.label == WalletLabel.METRIC_BOT:
            return 1.10  # Fast flips, small margins
        elif profile.label == WalletLabel.EARLY_EXIT:
            return 1.05  # Exit before big gains/losses
        elif profile.label == WalletLabel.SYBIL:
            return 1.20  # High variance (rug or moon)
        return 1.0

    def get_strategy_guide(self, bot_type: WalletLabel) -> Optional[BotTypeStrategy]:
        """Get strategy recommendations for a bot type"""
        return STRATEGY_TEMPLATES.get(bot_type)

    def generate_summary_report(self) -> Dict:
        """Generate summary report of all detected bots"""
        report = {
            'total_bots_profiled': len(self.profiles),
            'by_type': {},
            'top_threats': [],
            'profiles': []
        }

        # Count by type
        type_counts = {}
        for profile in self.profiles:
            label = profile.label.name
            type_counts[label] = type_counts.get(label, 0) + 1

        report['by_type'] = type_counts

        # Identify top threats (high confidence METRIC_BOT and SYBIL)
        threats = [
            p for p in self.profiles
            if p.label in [WalletLabel.SYBIL, WalletLabel.METRIC_BOT]
            and p.confidence >= 0.65
        ]
        threats.sort(key=lambda p: p.confidence, reverse=True)
        report['top_threats'] = threats[:20]  # Top 20

        # Add all profiles
        report['profiles'] = self.profiles

        return report

    def export_blacklist(self, min_confidence: float = 0.60,
                        exclude_types: Optional[Set[WalletLabel]] = None) -> List[str]:
        """
        Export wallet addresses to blacklist

        Args:
            min_confidence: Minimum confidence to include
            exclude_types: Bot types to exclude from blacklist

        Returns:
            List of wallet addresses to avoid
        """
        if exclude_types is None:
            # Default: blacklist harmful bots, not deployers
            exclude_types = {WalletLabel.DEPLOYER, WalletLabel.UNKNOWN, WalletLabel.ORGANIC}

        blacklist = [
            profile.wallet
            for profile in self.profiles
            if profile.confidence >= min_confidence
            and profile.label not in exclude_types
        ]

        return blacklist
