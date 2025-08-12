"""
Test Wallet Attribution Fix

Run: python -m pytest tests/unit/test_attribution.py -v

Expected output:
test_full_mint_storage ✓ (Full mint addresses are stored correctly)
test_identical_prefix_attribution ✓ (Two mints with same 16-char prefix attribute correctly)
test_multiple_buys_same_mint ✓ (Multiple buys by same wallet handled correctly)
test_outcome_matching_with_lookback ✓ (Outcomes match within time window)
test_no_attribution_outside_window ✓ (No attribution beyond lookback window)
"""

import pytest
import time
import tempfile
import os
from src.utils.wallet_performance import WalletPerformanceTracker


class TestWalletAttribution:
    """Test suite for wallet attribution fixes"""
    
    @pytest.fixture
    def tracker(self):
        """Create temporary tracker for testing"""
        # Use temporary file for testing
        fd, temp_path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        
        tracker = WalletPerformanceTracker(path=temp_path)
        yield tracker
        
        # Cleanup
        try:
            os.remove(temp_path)
        except:
            pass
    
    def test_full_mint_storage(self, tracker):
        """Test that full mint addresses are stored correctly"""
        wallet = "8xY3Qz4nV2pL5mK9jF6wT1hR7dC3bA2gE4iU6oP9sN1x"
        mint = "So11111111111111111111111111111111111111112"  # Full SOL mint
        
        tracker.record_trade_follow(wallet, mint, 75.0, tx_signature="test_tx_1")
        
        # Check that full mint is stored
        wallet_data = tracker.data["wallets"][wallet]
        trade = wallet_data["trade_history"][0]
        
        assert trade["mint"] == mint, f"Expected full mint {mint}, got {trade['mint']}"
        assert trade["tx_signature"] == "test_tx_1"
        assert len(trade["mint"]) == 44  # Full Solana address length
    
    def test_identical_prefix_attribution(self, tracker):
        """Test that two mints with identical first 16 chars attribute correctly"""
        wallet = "8xY3Qz4nV2pL5mK9jF6wT1hR7dC3bA2gE4iU6oP9sN1x"
        
        # Two different mints with same first 16 characters
        mint1 = "So11111111111111111111111111111111111111112"  # Real SOL
        mint2 = "So11111111111111DIFFERENT_ENDING_PART_HERE"   # Fake but same prefix
        
        # Record trades for both mints
        tracker.record_trade_follow(wallet, mint1, 75.0, tx_signature="tx_mint1")
        tracker.record_trade_follow(wallet, mint2, 80.0, tx_signature="tx_mint2")
        
        # Record outcomes
        tracker.record_trade_outcome(mint1, 25.0)  # +25% on mint1
        tracker.record_trade_outcome(mint2, -10.0)  # -10% on mint2
        
        # Check that each trade got the correct outcome
        wallet_data = tracker.data["wallets"][wallet]
        trades = wallet_data["trade_history"]
        
        mint1_trade = next(t for t in trades if t["mint"] == mint1)
        mint2_trade = next(t for t in trades if t["mint"] == mint2)
        
        assert mint1_trade["pnl_pct"] == 25.0, "Mint1 should have +25% outcome"
        assert mint2_trade["pnl_pct"] == -10.0, "Mint2 should have -10% outcome"
        assert mint1_trade["outcome"] == "win"
        assert mint2_trade["outcome"] == "loss"
    
    def test_multiple_buys_same_mint(self, tracker):
        """Test that multiple buys by same wallet on same mint are handled"""
        wallet = "8xY3Qz4nV2pL5mK9jF6wT1hR7dC3bA2gE4iU6oP9sN1x"
        mint = "So11111111111111111111111111111111111111112"
        
        # Record multiple trades on same mint
        tracker.record_trade_follow(wallet, mint, 75.0, tx_signature="tx1")
        time.sleep(0.1)  # Small delay
        tracker.record_trade_follow(wallet, mint, 80.0, tx_signature="tx2")
        
        # Record outcome - should match to first unmatched trade
        tracker.record_trade_outcome(mint, 15.0)
        
        wallet_data = tracker.data["wallets"][wallet]
        trades = [t for t in wallet_data["trade_history"] if t["mint"] == mint]
        
        # First trade should have outcome, second should be None
        assert trades[0]["outcome"] == "win"
        assert trades[0]["pnl_pct"] == 15.0
        assert trades[1]["outcome"] is None
        assert trades[1]["pnl_pct"] is None
    
    def test_outcome_matching_with_lookback(self, tracker):
        """Test that outcomes match only within lookback window"""
        wallet = "8xY3Qz4nV2pL5mK9jF6wT1hR7dC3bA2gE4iU6oP9sN1x"
        mint = "So11111111111111111111111111111111111111112"
        
        # Record trade with custom lookback
        tracker.record_trade_follow(wallet, mint, 75.0, tx_signature="tx1")
        trade_time = time.time()
        
        # Test outcome within window (should match)
        exit_time = trade_time + 300  # 5 minutes later
        tracker.record_trade_outcome(mint, 20.0, exit_timestamp=exit_time)
        
        wallet_data = tracker.data["wallets"][wallet]
        trade = wallet_data["trade_history"][0]
        
        assert trade["outcome"] == "win"
        assert trade["pnl_pct"] == 20.0
    
    def test_no_attribution_outside_window(self, tracker):
        """Test that no attribution happens beyond lookback window"""
        wallet = "8xY3Qz4nV2pL5mK9jF6wT1hR7dC3bA2gE4iU6oP9sN1x"
        mint = "So11111111111111111111111111111111111111112"
        
        # Record trade
        tracker.record_trade_follow(wallet, mint, 75.0, tx_signature="tx1")
        trade_time = time.time()
        
        # Test outcome outside window (should NOT match)
        exit_time = trade_time + 1000  # 16+ minutes later (beyond 15min window)
        tracker.record_trade_outcome(mint, 20.0, exit_timestamp=exit_time)
        
        wallet_data = tracker.data["wallets"][wallet]
        trade = wallet_data["trade_history"][0]
        
        # Should not have been attributed
        assert trade["outcome"] is None
        assert trade["pnl_pct"] is None
    
    def test_reorg_safety(self, tracker):
        """Test that outcomes before trades don't get attributed"""
        wallet = "8xY3Qz4nV2pL5mK9jF6wT1hR7dC3bA2gE4iU6oP9sN1x"
        mint = "So11111111111111111111111111111111111111112"
        
        trade_time = time.time()
        
        # Record outcome BEFORE trade (blockchain reorg scenario)
        exit_time = trade_time - 100  # 100 seconds BEFORE trade
        tracker.record_trade_outcome(mint, 20.0, exit_timestamp=exit_time)
        
        # Now record the trade
        tracker.record_trade_follow(wallet, mint, 75.0, tx_signature="tx1")
        
        wallet_data = tracker.data["wallets"][wallet]
        trade = wallet_data["trade_history"][0]
        
        # Should not have been attributed (exit before trade)
        assert trade["outcome"] is None
        assert trade["pnl_pct"] is None
    
    def test_attribution_logging(self, tracker, caplog):
        """Test that attribution results are logged"""
        wallet = "8xY3Qz4nV2pL5mK9jF6wT1hR7dC3bA2gE4iU6oP9sN1x"
        mint = "So11111111111111111111111111111111111111112"
        
        # Record trade and outcome
        tracker.record_trade_follow(wallet, mint, 75.0)
        tracker.record_trade_outcome(mint, 25.0)
        
        # Check that success was logged
        assert "Trade outcome recorded" in caplog.text
        assert "matched to 1 wallets" in caplog.text
        
        # Test no attribution scenario
        different_mint = "DifferentMintAddress1111111111111111111111"
        tracker.record_trade_outcome(different_mint, 10.0)
        
        # Check that failure was logged
        assert "No attribution found" in caplog.text


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])