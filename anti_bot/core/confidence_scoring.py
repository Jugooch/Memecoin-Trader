"""
Feature-based confidence scoring for wallet classification

Replaces naive `len(reasons) * constant` with weighted feature scoring
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class FeatureWeights:
    """Weights for a specific label's features"""
    features: Dict[str, float]  # feature_name -> weight
    bias: float = 0.0


# Pre-calibrated weights (bootstrap from 4-6h observation run)
# These would ideally be tuned per environment, but these are reasonable defaults

SYBIL_WEIGHTS = FeatureWeights(
    features={
        'parent_strength': 0.50,      # Has same-parent siblings
        'timing_tightness': 0.30,     # Synchronized timing
        'micro_buy_score': 0.15,      # Micro-buy pattern
        'smallest_k_mass': 0.05,      # High smallest-k mass
    },
    bias=-0.5  # Require strong evidence
)

METRIC_BOT_WEIGHTS = FeatureWeights(
    features={
        'burst_follow_rate': 0.40,    # % of buys after bursts
        'inter_arrival_cv': 0.25,     # Low CV = template timing
        'preferred_delay_consistency': 0.20,  # Consistent delay
        'template_cu_variance': 0.10,  # Low CU variance
        'fast_flip_rate': 0.05,       # Quick exits
    },
    bias=-0.3
)

PRIORITY_RACER_WEIGHTS = FeatureWeights(
    features={
        'early_lander_rate': 0.35,    # Lands in first 10% of slot
        'high_cu_rate': 0.30,         # High CU vs peers
        'jito_rate': 0.25,            # Jito bundle usage
        'leader_stickiness': 0.10,    # Concentration under specific leaders
    },
    bias=0.0
)

EARLY_EXIT_WEIGHTS = FeatureWeights(
    features={
        'first_seller_rate': 0.60,    # % of times first seller
        'failed_launch_accuracy': 0.25,  # Accuracy in detecting fails
        'pnl_positive': 0.15,         # Actually profitable
    },
    bias=0.2  # Easier to detect
)


def calculate_feature_score(features: Dict[str, float], weights: FeatureWeights) -> float:
    """
    Calculate weighted feature score

    Args:
        features: Dict of feature_name -> value (0-1)
        weights: FeatureWeights for this label

    Returns:
        Confidence score (0-1) after sigmoid
    """
    # Linear combination
    score = weights.bias

    for feature_name, feature_value in features.items():
        weight = weights.features.get(feature_name, 0.0)
        score += weight * feature_value

    # Sigmoid to [0, 1]
    confidence = 1.0 / (1.0 + (2.718281828 ** (-score)))

    return confidence


def calculate_sybil_features(cluster_size: int, timing_gaps_ms: List[float],
                            median_buy_sol: float, smallest_k_mass: float) -> Dict[str, float]:
    """
    Calculate features for SYBIL classification

    Args:
        cluster_size: Number of wallets funded by same parent
        timing_gaps_ms: Gaps between coordinated buys (ms)
        median_buy_sol: Median buy size
        smallest_k_mass: Share of volume in smallest 8 buys

    Returns:
        Dict of normalized features (0-1)
    """
    # Parent strength: sigmoid((cluster_size - 2) / 6)
    parent_strength = min(1.0, max(0.0, (cluster_size - 2) / 6.0))

    # Timing tightness: exp(-median_gap / 250ms)
    if timing_gaps_ms:
        median_gap = sorted(timing_gaps_ms)[len(timing_gaps_ms) // 2]
        timing_tightness = min(1.0, 2.718281828 ** (-median_gap / 250.0))
    else:
        timing_tightness = 0.0

    # Micro-buy score: how far below 0.02 SOL threshold
    micro_buy_score = min(1.0, max(0.0, (0.02 - median_buy_sol) / 0.02))

    # Smallest-k mass: higher = more suspicious
    smallest_k_score = min(1.0, smallest_k_mass * 2.0)  # 0.5 = 1.0 score

    return {
        'parent_strength': parent_strength,
        'timing_tightness': timing_tightness,
        'micro_buy_score': micro_buy_score,
        'smallest_k_mass': smallest_k_score,
    }


def calculate_metric_bot_features(burst_follows: int, total_buys: int,
                                  distinct_mints: int, inter_arrival_cv: float,
                                  preferred_delay_std: float, cu_variance: float,
                                  median_hold_sec: float) -> Dict[str, float]:
    """
    Calculate features for METRIC_BOT classification

    Args:
        burst_follows: Number of buys 1-4s after burst
        total_buys: Total buy count
        distinct_mints: Number of distinct mints traded
        inter_arrival_cv: Coefficient of variation for inter-arrival times
        preferred_delay_std: Std dev of delays after bursts
        cu_variance: Variance of CU prices
        median_hold_sec: Median hold time

    Returns:
        Dict of normalized features (0-1)
    """
    # Burst follow rate (sigmoid around 0.55 target)
    burst_follow_rate = burst_follows / max(1, total_buys)
    burst_score = 1.0 / (1.0 + 2.718281828 ** (-(burst_follow_rate - 0.55) * 10))

    # Inter-arrival CV (low CV = bot)
    # Target: <0.4 = bot, >0.8 = organic
    # Score: 1.0 at CV=0, 0.0 at CV=0.8
    cv_score = max(0.0, 1.0 - inter_arrival_cv / 0.8)

    # Preferred delay consistency (low std = consistent)
    # Score: 1.0 at std=0, 0.0 at std=0.6s
    delay_score = max(0.0, 1.0 - preferred_delay_std / 0.6)

    # CU variance (low variance = template)
    # Score: 1.0 at variance=0, 0.0 at variance=100k
    cu_score = max(0.0, 1.0 - cu_variance / 100000.0)

    # Fast flip rate (median hold < 25s)
    # Score: 1.0 at 0s, 0.0 at 25s
    flip_score = max(0.0, 1.0 - median_hold_sec / 25.0)

    return {
        'burst_follow_rate': burst_score,
        'inter_arrival_cv': cv_score,
        'preferred_delay_consistency': delay_score,
        'template_cu_variance': cu_score,
        'fast_flip_rate': flip_score,
    }


def calculate_priority_racer_features(early_lander_rate: float, high_cu_rate: float,
                                     jito_rate: float, leader_concentration: float) -> Dict[str, float]:
    """
    Calculate features for PRIORITY_RACER classification

    Args:
        early_lander_rate: % of trades landing in earliest 10% of slot
        high_cu_rate: % of trades with CU >= p90 + 30%
        jito_rate: % of trades with Jito tip
        leader_concentration: Herfindahl index (0-1) for leader concentration

    Returns:
        Dict of normalized features (0-1)
    """
    # All rates are already 0-1
    return {
        'early_lander_rate': early_lander_rate,
        'high_cu_rate': high_cu_rate,
        'jito_rate': jito_rate,
        'leader_stickiness': leader_concentration,
    }


def calculate_early_exit_features(first_seller_count: int, total_sells: int,
                                  correct_predictions: int, total_predictions: int,
                                  median_pnl_pct: float) -> Dict[str, float]:
    """
    Calculate features for EARLY_EXIT classification

    Args:
        first_seller_count: Times wallet was first seller
        total_sells: Total sell count
        correct_predictions: Failed launches correctly exited
        total_predictions: Total early exits attempted
        median_pnl_pct: Median P&L %

    Returns:
        Dict of normalized features (0-1)
    """
    # First seller rate
    first_seller_rate = first_seller_count / max(1, total_sells)

    # Prediction accuracy
    accuracy = correct_predictions / max(1, total_predictions)

    # P&L positive (sigmoid around 3%)
    pnl_score = 1.0 / (1.0 + 2.718281828 ** (-(median_pnl_pct - 3.0) * 0.5))

    return {
        'first_seller_rate': first_seller_rate,
        'failed_launch_accuracy': accuracy,
        'pnl_positive': pnl_score,
    }


# Convenience functions for each label

def score_sybil(cluster_size: int, timing_gaps_ms: List[float],
               median_buy_sol: float, smallest_k_mass: float) -> float:
    """Calculate SYBIL confidence score"""
    features = calculate_sybil_features(cluster_size, timing_gaps_ms,
                                       median_buy_sol, smallest_k_mass)
    return calculate_feature_score(features, SYBIL_WEIGHTS)


def score_metric_bot(burst_follows: int, total_buys: int, distinct_mints: int,
                     inter_arrival_cv: float, preferred_delay_std: float,
                     cu_variance: float, median_hold_sec: float) -> float:
    """Calculate METRIC_BOT confidence score"""
    features = calculate_metric_bot_features(burst_follows, total_buys, distinct_mints,
                                            inter_arrival_cv, preferred_delay_std,
                                            cu_variance, median_hold_sec)
    return calculate_feature_score(features, METRIC_BOT_WEIGHTS)


def score_priority_racer(early_lander_rate: float, high_cu_rate: float,
                        jito_rate: float, leader_concentration: float) -> float:
    """Calculate PRIORITY_RACER confidence score"""
    features = calculate_priority_racer_features(early_lander_rate, high_cu_rate,
                                                 jito_rate, leader_concentration)
    return calculate_feature_score(features, PRIORITY_RACER_WEIGHTS)


def score_early_exit(first_seller_count: int, total_sells: int,
                    correct_predictions: int, total_predictions: int,
                    median_pnl_pct: float) -> float:
    """Calculate EARLY_EXIT confidence score"""
    features = calculate_early_exit_features(first_seller_count, total_sells,
                                            correct_predictions, total_predictions,
                                            median_pnl_pct)
    return calculate_feature_score(features, EARLY_EXIT_WEIGHTS)
