"""
Adaptive thresholds that adjust to market conditions

Replaces hardcoded values with rolling percentiles per mint cohort
"""

from typing import List, Dict
from collections import deque
from dataclasses import dataclass, field
import statistics


@dataclass
class MintCohort:
    """Rolling statistics for a cohort of mints (e.g., all mints 0-60s old)"""
    name: str
    max_age_seconds: float

    # Rolling samples (last N mints in this cohort)
    buy_counts_3s: deque = field(default_factory=lambda: deque(maxlen=200))
    sol_volumes_3s: deque = field(default_factory=lambda: deque(maxlen=200))
    gini_coefficients: deque = field(default_factory=lambda: deque(maxlen=200))
    median_hold_times: deque = field(default_factory=lambda: deque(maxlen=200))

    # Per-mint CU prices (for racer detection)
    mint_cu_prices: Dict[str, deque] = field(default_factory=dict)

    def add_sample(self, buy_count_3s: int, sol_volume_3s: float,
                   gini: float, median_hold: float):
        """Add a sample to rolling window"""
        self.buy_counts_3s.append(buy_count_3s)
        self.sol_volumes_3s.append(sol_volume_3s)
        self.gini_coefficients.append(gini)
        self.median_hold_times.append(median_hold)

    def get_burst_thresholds(self) -> tuple:
        """Get adaptive burst thresholds (p75 - top 25% of activity)"""
        if len(self.buy_counts_3s) < 20:
            # Fallback to hardcoded (lowered from 25 to 15)
            return (15, 2.5)

        sorted_counts = sorted(self.buy_counts_3s)
        sorted_volumes = sorted(self.sol_volumes_3s)

        # Changed from P95 to P75 - detect more bursts (top 25% of activity)
        p75_idx = int(len(sorted_counts) * 0.75)

        buy_threshold = sorted_counts[p75_idx]
        volume_threshold = sorted_volumes[p75_idx]

        return (buy_threshold, volume_threshold)

    def get_organic_gini_range(self) -> tuple:
        """Get expected Gini range for organic cohort"""
        if len(self.gini_coefficients) < 20:
            return (0.35, 0.65)

        sorted_gini = sorted(self.gini_coefficients)

        # Organic is between p25 and p75
        p25_idx = int(len(sorted_gini) * 0.25)
        p75_idx = int(len(sorted_gini) * 0.75)

        return (sorted_gini[p25_idx], sorted_gini[p75_idx])

    def add_mint_cu_price(self, mint: str, cu_price: int):
        """Track CU prices for a specific mint"""
        if mint not in self.mint_cu_prices:
            self.mint_cu_prices[mint] = deque(maxlen=50)

        self.mint_cu_prices[mint].append(cu_price)

    def get_priority_threshold(self, mint: str) -> int:
        """Get adaptive priority fee threshold for this mint (p90 + 0.3*IQR)"""
        if mint not in self.mint_cu_prices or len(self.mint_cu_prices[mint]) < 10:
            # Fallback: global estimate
            return 1_000_000  # 1M microlamports per CU

        cu_prices = list(self.mint_cu_prices[mint])
        sorted_prices = sorted(cu_prices)

        p90_idx = int(len(sorted_prices) * 0.9)
        p75_idx = int(len(sorted_prices) * 0.75)
        p25_idx = int(len(sorted_prices) * 0.25)

        p90 = sorted_prices[p90_idx]
        iqr = sorted_prices[p75_idx] - sorted_prices[p25_idx]

        return int(p90 + 0.3 * iqr)


class AdaptiveThresholdManager:
    """Manages adaptive thresholds across different mint age cohorts"""

    def __init__(self):
        # Cohorts by age
        self.cohorts = {
            'very_new': MintCohort('very_new', max_age_seconds=60),
            'new': MintCohort('new', max_age_seconds=300),
            'established': MintCohort('established', max_age_seconds=1800),
        }

    def get_cohort(self, token_age_seconds: float) -> MintCohort:
        """Get appropriate cohort for a mint's age"""
        if token_age_seconds <= 60:
            return self.cohorts['very_new']
        elif token_age_seconds <= 300:
            return self.cohorts['new']
        else:
            return self.cohorts['established']

    def update_mint_stats(self, mint: str, token_age_seconds: float,
                         buy_count_3s: int, sol_volume_3s: float,
                         gini: float, median_hold: float):
        """Update cohort statistics for a mint"""
        cohort = self.get_cohort(token_age_seconds)
        cohort.add_sample(buy_count_3s, sol_volume_3s, gini, median_hold)

    def add_cu_price(self, mint: str, token_age_seconds: float, cu_price: int):
        """Track CU price for a mint"""
        cohort = self.get_cohort(token_age_seconds)
        cohort.add_mint_cu_price(mint, cu_price)

    def get_burst_thresholds(self, token_age_seconds: float) -> tuple:
        """Get adaptive burst thresholds for a token"""
        cohort = self.get_cohort(token_age_seconds)
        return cohort.get_burst_thresholds()

    def get_priority_threshold(self, mint: str, token_age_seconds: float) -> int:
        """Get adaptive priority fee threshold"""
        cohort = self.get_cohort(token_age_seconds)
        return cohort.get_priority_threshold(mint)

    def is_burst(self, mint: str, token_age_seconds: float,
                 buy_count_3s: int, sol_volume_3s: float) -> bool:
        """Check if current activity constitutes a burst"""
        buy_threshold, volume_threshold = self.get_burst_thresholds(token_age_seconds)

        return (buy_count_3s >= buy_threshold or
                sol_volume_3s >= volume_threshold)


def calculate_inter_arrival_cv(timestamps: List[float]) -> float:
    """
    Calculate coefficient of variation for inter-arrival times

    Low CV (<0.4) = metric bot (template timing)
    High CV (>0.8) = organic/human (bursty)

    Args:
        timestamps: List of timestamps in seconds

    Returns:
        CV (coefficient of variation)
    """
    if len(timestamps) < 3:
        return 0.0

    # Calculate inter-arrival times
    diffs = [timestamps[i+1] - timestamps[i]
            for i in range(len(timestamps)-1)]

    if not diffs or min(diffs) < 0:
        return 0.0

    mean_diff = statistics.mean(diffs)
    if mean_diff == 0:
        return 0.0

    std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0

    cv = std_diff / mean_diff

    return cv


def calculate_smallest_k_mass(buy_sizes: List[float], k: int = 8) -> float:
    """
    Calculate share of volume in smallest k buys

    Sybils show high smallest-k mass (many tiny coordinated buys)
    Organic shows low smallest-k mass (varied sizes)

    Args:
        buy_sizes: List of buy sizes in SOL
        k: Number of smallest buys to consider

    Returns:
        Fraction of total volume in smallest k buys
    """
    if len(buy_sizes) < k:
        k = len(buy_sizes)

    if k == 0 or sum(buy_sizes) == 0:
        return 0.0

    sorted_sizes = sorted(buy_sizes)
    smallest_k = sorted_sizes[:k]

    return sum(smallest_k) / sum(buy_sizes)


def calculate_preferred_delay_std(delays: List[float]) -> float:
    """
    Calculate standard deviation of delays after burst events

    Metric bots have low std (consistent delay)
    Organic traders have high std (random reactions)

    Args:
        delays: List of delays in seconds

    Returns:
        Standard deviation of delays
    """
    if len(delays) < 2:
        return 0.0

    return statistics.stdev(delays)
