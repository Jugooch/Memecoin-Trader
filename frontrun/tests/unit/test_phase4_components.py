"""
Unit Tests for Phase 4 Components
Tests all defensive guardrails and frontrun-specific features
"""

import pytest
import time
import asyncio
from datetime import datetime, timedelta

from core.latency_enforcer import (
    LatencyBudgetEnforcer,
    LatencyConfig,
    TradeStage
)
from core.profit_aware_fees import (
    ProfitAwareFeeBidder,
    BidderConfig,
    Trade
)
from core.rpc_health_scorer import (
    RPCHealthScorer,
    HealthScorerConfig,
    EndpointHealth
)
from core.race_detector import (
    RaceFailureDetector,
    RaceConfig,
    RaceResult
)


class TestLatencyBudgetEnforcer:
    """Tests for Hard Latency Budget Enforcer"""

    def test_latency_enforcer_initialization(self):
        """Test enforcer initializes with correct config"""
        config = LatencyConfig(
            detect_stage_budget_ms=30.0,
            build_sign_budget_ms=15.0,
            submit_stage_budget_ms=50.0,
            total_budget_ms=100.0
        )
        enforcer = LatencyBudgetEnforcer(config)

        assert enforcer.config.detect_stage_budget_ms == 30.0
        assert enforcer.config.total_budget_ms == 100.0
        assert enforcer.consecutive_violations == 0
        assert enforcer.circuit_open_until is None

    def test_start_operation_creates_timer(self):
        """Test starting operation creates timer with correct ID"""
        enforcer = LatencyBudgetEnforcer()
        timer = enforcer.start_operation("test_trade_001")

        assert timer.operation_id == "test_trade_001"
        assert timer.start_time > 0
        assert len(timer.stage_times) == 0

    def test_check_and_abort_within_budget(self):
        """Test check_and_abort returns False when within budget"""
        enforcer = LatencyBudgetEnforcer()
        timer = enforcer.start_operation("test_trade_002")

        # Immediately check (should be within budget)
        should_abort = enforcer.check_and_abort(timer, TradeStage.DETECT)

        assert should_abort is False
        assert enforcer.consecutive_violations == 0

    def test_check_and_abort_exceeds_budget(self):
        """Test check_and_abort returns True when budget exceeded"""
        config = LatencyConfig(detect_stage_budget_ms=10.0)  # Very tight budget
        enforcer = LatencyBudgetEnforcer(config)
        timer = enforcer.start_operation("test_trade_003")

        # Sleep to exceed budget
        time.sleep(0.02)  # 20ms

        should_abort = enforcer.check_and_abort(timer, TradeStage.DETECT)

        assert should_abort is True
        assert enforcer.consecutive_violations == 1

    def test_circuit_breaker_trips_after_violations(self):
        """Test circuit breaker trips after consecutive violations"""
        config = LatencyConfig(
            detect_stage_budget_ms=1.0,  # 1ms budget (impossible)
            violations_before_trip=3
        )
        enforcer = LatencyBudgetEnforcer(config)

        # Trigger 3 violations
        for i in range(3):
            timer = enforcer.start_operation(f"test_trade_{i}")
            time.sleep(0.002)  # 2ms
            enforcer.check_and_abort(timer, TradeStage.DETECT)

        assert enforcer.is_circuit_open() is True

    def test_circuit_breaker_resets_after_timeout(self):
        """Test circuit breaker resets after timeout"""
        config = LatencyConfig(
            detect_stage_budget_ms=1.0,
            violations_before_trip=1,
            circuit_reset_after_s=0  # Immediate reset for testing
        )
        enforcer = LatencyBudgetEnforcer(config)

        # Trip circuit
        timer = enforcer.start_operation("test_trade")
        time.sleep(0.002)
        enforcer.check_and_abort(timer, TradeStage.DETECT)

        assert enforcer.is_circuit_open() is True

        # Wait for reset
        time.sleep(0.1)

        assert enforcer.is_circuit_open() is False

    def test_record_stage_tracks_timing(self):
        """Test record_stage tracks stage completion times"""
        enforcer = LatencyBudgetEnforcer()
        timer = enforcer.start_operation("test_trade")

        time.sleep(0.01)  # 10ms
        enforcer.record_stage(timer, TradeStage.DETECT)

        assert TradeStage.DETECT.value in timer.stage_times
        assert timer.stage_times[TradeStage.DETECT.value] >= 10.0  # At least 10ms

    def test_attribution_stats_tracking(self):
        """Test attribution statistics are tracked correctly"""
        enforcer = LatencyBudgetEnforcer()

        # Create some operations
        for i in range(5):
            timer = enforcer.start_operation(f"trade_{i}")
            enforcer.check_and_abort(timer, TradeStage.DETECT)

        stats = enforcer.get_attribution_stats()

        assert stats["total_operations"] == 5
        assert "violation_rate" in stats


class TestProfitAwareFeeBidder:
    """Tests for Profit-Aware Fee Cap"""

    @pytest.mark.asyncio
    async def test_bidder_initialization(self):
        """Test bidder initializes with correct config"""
        from core.priority_fees import PriorityFeeCalculator
        from unittest.mock import AsyncMock, MagicMock

        # Mock RPC manager
        mock_rpc = MagicMock()
        mock_rpc.get_healthy_connection = AsyncMock(return_value=None)

        fee_calc = PriorityFeeCalculator(mock_rpc)

        config = BidderConfig(
            cold_start_max_lamports=50_000,
            profit_cap_pct=0.3
        )
        bidder = ProfitAwareFeeBidder(fee_calc, config)

        assert bidder.config.cold_start_max_lamports == 50_000
        assert bidder.config.profit_cap_pct == 0.3
        assert len(bidder._trade_history) == 0

    def test_add_trade_to_history(self):
        """Test adding trade to history"""
        from core.priority_fees import PriorityFeeCalculator
        from unittest.mock import MagicMock

        mock_rpc = MagicMock()
        fee_calc = PriorityFeeCalculator(mock_rpc)
        bidder = ProfitAwareFeeBidder(fee_calc)

        trade = Trade(
            won_race=True,
            pnl_sol=0.05,
            fee_paid_lamports=100_000,
            timestamp=datetime.now()
        )

        bidder.add_trade(trade)

        assert len(bidder._trade_history) == 1
        assert bidder._trade_history[0] == trade

    def test_estimate_conservative_ev_with_wins(self):
        """Test EV estimation uses p25 of wins"""
        from core.priority_fees import PriorityFeeCalculator
        from unittest.mock import MagicMock

        mock_rpc = MagicMock()
        fee_calc = PriorityFeeCalculator(mock_rpc)
        bidder = ProfitAwareFeeBidder(fee_calc)

        # Add 20 winning trades with varying profits
        for i in range(20):
            bidder.add_trade(Trade(
                won_race=True,
                pnl_sol=0.01 * (i + 1),  # 0.01 to 0.20 SOL
                fee_paid_lamports=100_000,
                timestamp=datetime.now()
            ))

        ev = bidder._estimate_conservative_ev(bidder._trade_history)

        # Should be p25 of 0.01-0.20 range, which is around 0.05
        assert 0.04 <= ev <= 0.06

    def test_estimate_conservative_ev_no_wins(self):
        """Test EV estimation returns 0 with no wins"""
        from core.priority_fees import PriorityFeeCalculator
        from unittest.mock import MagicMock

        mock_rpc = MagicMock()
        fee_calc = PriorityFeeCalculator(mock_rpc)
        bidder = ProfitAwareFeeBidder(fee_calc)

        # Add only losing trades
        for i in range(10):
            bidder.add_trade(Trade(
                won_race=False,
                pnl_sol=-0.01,
                fee_paid_lamports=100_000,
                timestamp=datetime.now()
            ))

        ev = bidder._estimate_conservative_ev(bidder._trade_history)

        assert ev == 0.0

    def test_validate_ev_before_submit_positive(self):
        """Test EV validation passes with positive EV"""
        from core.priority_fees import PriorityFeeCalculator
        from unittest.mock import MagicMock

        mock_rpc = MagicMock()
        fee_calc = PriorityFeeCalculator(mock_rpc)
        bidder = ProfitAwareFeeBidder(fee_calc)

        # Profit > Fee
        valid = bidder.validate_ev_before_submit(
            expected_profit_sol=0.05,
            fee_lamports=100_000  # 0.0001 SOL
        )

        assert valid is True

    def test_validate_ev_before_submit_negative(self):
        """Test EV validation fails with negative EV"""
        from core.priority_fees import PriorityFeeCalculator
        from unittest.mock import MagicMock

        mock_rpc = MagicMock()
        fee_calc = PriorityFeeCalculator(mock_rpc)
        bidder = ProfitAwareFeeBidder(fee_calc)

        # Profit < Fee
        valid = bidder.validate_ev_before_submit(
            expected_profit_sol=0.0001,
            fee_lamports=10_000_000  # 0.01 SOL
        )

        assert valid is False

    def test_get_stats_empty_history(self):
        """Test get_stats returns correct values with empty history"""
        from core.priority_fees import PriorityFeeCalculator
        from unittest.mock import MagicMock

        mock_rpc = MagicMock()
        fee_calc = PriorityFeeCalculator(mock_rpc)
        bidder = ProfitAwareFeeBidder(fee_calc)

        stats = bidder.get_stats()

        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["using_cold_start"] is True


class TestRPCHealthScorer:
    """Tests for RPC Health Scoring & Auto-Routing"""

    def test_scorer_initialization(self):
        """Test scorer initializes correctly"""
        from unittest.mock import MagicMock

        mock_rpc_manager = MagicMock()
        config = HealthScorerConfig(
            base_score=100,
            latency_threshold_p99_ms=500.0
        )
        scorer = RPCHealthScorer(mock_rpc_manager, config)

        assert scorer.config.base_score == 100
        assert len(scorer._endpoint_health) == 0

    def test_calculate_score_low_latency(self):
        """Test score calculation with low latency"""
        from unittest.mock import MagicMock

        mock_rpc_manager = MagicMock()
        scorer = RPCHealthScorer(mock_rpc_manager)

        score = scorer._calculate_score(
            p99_latency_ms=100.0,  # Well under threshold
            slot_lag=0,
            error_rate=0.0
        )

        assert score == 100  # Perfect score

    def test_calculate_score_high_latency(self):
        """Test score calculation with high latency"""
        from unittest.mock import MagicMock

        mock_rpc_manager = MagicMock()
        config = HealthScorerConfig(latency_threshold_p99_ms=500.0)
        scorer = RPCHealthScorer(mock_rpc_manager, config)

        score = scorer._calculate_score(
            p99_latency_ms=700.0,  # 200ms over threshold
            slot_lag=0,
            error_rate=0.0
        )

        # Should lose 10 points per 100ms over threshold = -20
        assert score == 80

    def test_calculate_score_with_errors(self):
        """Test score calculation with error rate"""
        from unittest.mock import MagicMock

        mock_rpc_manager = MagicMock()
        scorer = RPCHealthScorer(mock_rpc_manager)

        score = scorer._calculate_score(
            p99_latency_ms=100.0,
            slot_lag=0,
            error_rate=0.1  # 10% error rate
        )

        # Should lose 5 points per 1% error rate = -50
        assert score == 50

    def test_record_request_success_resets_violations(self):
        """Test successful request resets consecutive errors"""
        from unittest.mock import MagicMock

        mock_rpc_manager = MagicMock()
        scorer = RPCHealthScorer(mock_rpc_manager)

        # Add an endpoint
        scorer._endpoint_health["test_endpoint"] = EndpointHealth(
            label="test_endpoint",
            consecutive_errors=2
        )

        # Record successful request
        scorer.record_request(
            endpoint_label="test_endpoint",
            success=True,
            latency_ms=50.0
        )

        assert scorer._endpoint_health["test_endpoint"].consecutive_errors == 0

    def test_record_request_failure_increments_violations(self):
        """Test failed request increments consecutive errors"""
        from unittest.mock import MagicMock

        mock_rpc_manager = MagicMock()
        scorer = RPCHealthScorer(mock_rpc_manager)

        # Add an endpoint
        scorer._endpoint_health["test_endpoint"] = EndpointHealth(
            label="test_endpoint",
            consecutive_errors=0
        )

        # Record failed request
        scorer.record_request(
            endpoint_label="test_endpoint",
            success=False,
            latency_ms=1000.0
        )

        assert scorer._endpoint_health["test_endpoint"].consecutive_errors == 1


class TestRaceFailureDetector:
    """Tests for Race Failure Detector"""

    def test_detector_initialization(self):
        """Test detector initializes with correct config"""
        config = RaceConfig(
            price_deviation_threshold_pct=5.0,
            max_acceptable_slot_delay=2
        )
        detector = RaceFailureDetector(config)

        assert detector.config.price_deviation_threshold_pct == 5.0
        assert detector.config.max_acceptable_slot_delay == 2

    def test_detect_race_won_by_price(self):
        """Test race detection when price is acceptable"""
        detector = RaceFailureDetector()

        result = detector.detect_race_loss_by_price(
            expected_price=0.000030,
            actual_fill_price=0.000031  # Only 3.3% higher
        )

        assert result.won_race is True
        assert result.price_deviation_pct < 5.0

    def test_detect_race_lost_by_price(self):
        """Test race detection when price too high"""
        detector = RaceFailureDetector()

        result = detector.detect_race_loss_by_price(
            expected_price=0.000030,
            actual_fill_price=0.000040  # 33% higher
        )

        assert result.won_race is False
        assert result.price_deviation_pct > 5.0

    def test_detect_race_won_by_timing_same_slot(self):
        """Test race won when filled in same slot as dev"""
        detector = RaceFailureDetector()

        result = detector.analyze_fill_timing(
            dev_buy_slot=12345,
            our_fill_slot=12345  # Same slot
        )

        assert result.won_race is True
        assert result.slot_delay == 0

    def test_detect_race_won_by_timing_next_slot(self):
        """Test race won when filled 1 slot after dev"""
        detector = RaceFailureDetector()

        result = detector.analyze_fill_timing(
            dev_buy_slot=12345,
            our_fill_slot=12346  # 1 slot later
        )

        assert result.won_race is True
        assert result.slot_delay == 1

    def test_detect_race_lost_by_timing_too_late(self):
        """Test race lost when filled too late"""
        detector = RaceFailureDetector()

        result = detector.analyze_fill_timing(
            dev_buy_slot=12345,
            our_fill_slot=12350  # 5 slots later
        )

        assert result.won_race is False
        assert result.slot_delay == 5

    def test_analyze_combined_both_win(self):
        """Test combined analysis when both methods show win"""
        detector = RaceFailureDetector()

        result = detector.analyze_combined(
            expected_price=0.000030,
            actual_fill_price=0.000031,  # Good price
            dev_buy_slot=12345,
            our_fill_slot=12346  # Good timing
        )

        assert result.won_race is True

    def test_analyze_combined_price_loss(self):
        """Test combined analysis when price shows loss"""
        detector = RaceFailureDetector()

        result = detector.analyze_combined(
            expected_price=0.000030,
            actual_fill_price=0.000040,  # Bad price
            dev_buy_slot=12345,
            our_fill_slot=12346  # Good timing
        )

        assert result.won_race is False  # Overall loss due to price

    def test_analyze_combined_timing_loss(self):
        """Test combined analysis when timing shows loss"""
        detector = RaceFailureDetector()

        result = detector.analyze_combined(
            expected_price=0.000030,
            actual_fill_price=0.000031,  # Good price
            dev_buy_slot=12345,
            our_fill_slot=12350  # Bad timing
        )

        assert result.won_race is False  # Overall loss due to timing


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
