"""
Unit tests for sniper bot core functionality
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sniper.storage import Store
from sniper.filters import SafetyFilters
from sniper.risk import RiskManager
from sniper.metrics import Metrics


class TestStore:
    """Test storage functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'path': self.temp_dir,
            'rotate_jsonl_mb': 1  # Small size for testing
        }
        self.store = Store(self.config)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_position_management(self):
        """Test position creation and management"""
        # Create position
        position_data = {
            'token': 'test_token_mint',
            'ticker': 'TEST',
            'dev_wallet': 'test_dev_wallet',
            'entry_sol': 0.2,
            'entry_price': 0.000001
        }
        
        position_id = self.store.add_position(position_data)
        assert position_id is not None
        assert position_id.startswith('test_token_mint_')
        
        # Retrieve position
        retrieved = self.store.get_position(position_id)
        assert retrieved is not None
        assert retrieved['ticker'] == 'TEST'
        assert retrieved['status'] == 'open'
        
        # Update position
        self.store.update_position(position_id, {'current_price': 0.000002})
        updated = self.store.get_position(position_id)
        assert updated['current_price'] == 0.000002
        
        # Close position
        self.store.close_position(position_id, 'test_exit', 0.05)
        closed = self.store.get_position(position_id)
        assert closed['status'] == 'closed'
        assert closed['realized_pnl'] == 0.05
    
    def test_whitelist_blacklist(self):
        """Test whitelist/blacklist functionality"""
        dev_wallet = 'test_dev_wallet'
        
        # Initially neither
        assert not self.store.is_whitelisted(dev_wallet)
        assert not self.store.is_blacklisted(dev_wallet)
        
        # Add to whitelist
        self.store.add_to_whitelist(dev_wallet, 75.0)
        assert self.store.is_whitelisted(dev_wallet)
        
        # Add to blacklist
        self.store.add_to_blacklist(dev_wallet, 'Test reason')
        assert self.store.is_blacklisted(dev_wallet)
        
        # Remove from blacklist
        self.store.remove_from_blacklist(dev_wallet)
        assert not self.store.is_blacklisted(dev_wallet)
    
    def test_dev_profile_caching(self):
        """Test dev profile caching with TTL"""
        dev_wallet = 'test_dev_wallet'
        
        # No profile initially
        assert self.store.get_dev_profile(dev_wallet) is None
        
        # Add profile
        profile = {
            'best_peak_mc_usd': 5000000,
            'num_rugs_90d': 0,
            'tokens_launched_7d': 1
        }
        self.store.update_dev_profile(dev_wallet, profile)
        
        # Should retrieve fresh profile
        retrieved = self.store.get_dev_profile(dev_wallet)
        assert retrieved is not None
        assert retrieved['best_peak_mc_usd'] == 5000000
    
    def test_event_logging(self):
        """Test event logging"""
        event_data = {
            'ticker': 'TEST',
            'token_mint': 'test_mint'
        }
        
        # Log event
        self.store.log_event('ENTRY_OK', event_data, 'test_reason')
        
        # Check file exists
        assert self.store.events_file.exists()
        
        # Read and verify content
        with open(self.store.events_file, 'r') as f:
            content = f.read()
            assert 'ENTRY_OK' in content
            assert 'TEST' in content


class TestSafetyFilters:
    """Test safety filter functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'safety': {
                'dev_min_hold_pct': 1.0,
                'dev_max_hold_pct': 20.0,
                'dev_min_liq_sol': 5.0,
                'dev_max_initial_buy_sol': 2.0,
                'dev_no_rug_history_days': 90,
                'max_tax_bps': 200,
                'max_tokens_launched_7d': 5,
                'min_unique_buyers_first_5min': 12,
                'blocklist': {'wallets': [], 'tickers': []},
                'allowlist': {'wallets': []}
            },
            'storage': {'path': self.temp_dir}
        }
        self.store = Store(self.config['storage'])
        self.metrics = Metrics({'metrics': {'enabled': False}})
        self.filters = SafetyFilters(self.config, self.store, self.metrics)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_good_event_passes(self):
        """Test that a good event passes all filters"""
        event = {
            'dev_wallet': 'good_dev_wallet',
            'ticker': 'GOOD',
            'lp_init': 10.0,
            'dev_hold_pct': 5.0,
            'taxes': {'buy_tax_bps': 50, 'sell_tax_bps': 50},
            'dev_initial_buy_sol': 1.0,
            'profile': {
                'num_rugs_90d': 0,
                'tokens_launched_7d': 2
            }
        }
        
        assert self.filters.passes_all(event)
    
    def test_blacklisted_dev_fails(self):
        """Test that blacklisted dev fails"""
        dev_wallet = 'bad_dev_wallet'
        self.store.add_to_blacklist(dev_wallet, 'Test reason')
        
        event = {
            'dev_wallet': dev_wallet,
            'ticker': 'BAD',
            'lp_init': 10.0,
            'dev_hold_pct': 5.0,
            'taxes': {'buy_tax_bps': 50, 'sell_tax_bps': 50}
        }
        
        assert not self.filters.passes_all(event)
        assert self.filters.last_reason == "DEV_BLACKLISTED"
    
    def test_high_tax_fails(self):
        """Test that high tax tokens fail"""
        event = {
            'dev_wallet': 'test_dev_wallet',
            'ticker': 'HIGHTAX',
            'lp_init': 10.0,
            'dev_hold_pct': 5.0,
            'taxes': {'buy_tax_bps': 300, 'sell_tax_bps': 300},  # 6% total
            'profile': {'num_rugs_90d': 0}
        }
        
        assert not self.filters.passes_all(event)
        assert self.filters.last_reason == "TAX_TOO_HIGH"
    
    def test_low_liquidity_fails(self):
        """Test that low liquidity tokens fail"""
        event = {
            'dev_wallet': 'test_dev_wallet',
            'ticker': 'LOWLIQ',
            'lp_init': 2.0,  # Below 5 SOL minimum
            'dev_hold_pct': 5.0,
            'taxes': {'buy_tax_bps': 50, 'sell_tax_bps': 50},
            'profile': {'num_rugs_90d': 0}
        }
        
        assert not self.filters.passes_all(event)
        assert self.filters.last_reason == "INSUFFICIENT_LIQUIDITY"


class TestRiskManager:
    """Test risk management functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'risk_scoring': {
                'weights': {
                    'peak_mc': 30,
                    'rug_count': 20,
                    'holder_count': 15,
                    'launch_freq': 15,
                    'lp_quality': 10,
                    'social': 10
                },
                'min_score_to_trade': 50,
                'dev_cooldown_hours': 24
            },
            'trade': {
                'max_open_positions': 1,
                'max_trades_per_hour': 6,
                'max_risk_per_hour_sol': 1.0,
                'base_position_size_sol': 0.2,
                'size_by_score': [
                    {'min_score': 80, 'size_sol': 0.4},
                    {'min_score': 65, 'size_sol': 0.3},
                    {'min_score': 50, 'size_sol': 0.2},
                    {'min_score': 0, 'size_sol': 0.15}
                ],
                'priority_fee_sol': {
                    'base': 0.002,
                    'max': 0.01,
                    'by_score': []
                },
                'congestion': {
                    'multipliers': [1.0, 1.5, 2.0, 3.0]
                }
            },
            'storage': {'path': self.temp_dir}
        }
        self.store = Store(self.config['storage'])
        self.risk = RiskManager(self.config, self.store)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_dev_scoring(self):
        """Test dev scoring logic"""
        # High quality dev
        good_event = {
            'profile': {
                'best_peak_mc_usd': 5000000,  # $5M peak
                'num_rugs_90d': 0,  # No rugs
                'avg_holder_count_30m': 500,  # Good holders
                'tokens_launched_7d': 1,  # Low frequency
            },
            'lp_init': 15.0  # Good LP
        }
        
        good_score = self.risk.score_dev('good_dev', good_event)
        assert good_score >= 70  # Should score well
        
        # Poor quality dev
        bad_event = {
            'profile': {
                'best_peak_mc_usd': 100000,  # Low peak
                'num_rugs_90d': 1,  # Has rugs
                'avg_holder_count_30m': 50,  # Few holders
                'tokens_launched_7d': 5,  # High frequency
            },
            'lp_init': 3.0  # Low LP
        }
        
        bad_score = self.risk.score_dev('bad_dev', bad_event)
        assert bad_score < 40  # Should score poorly
    
    def test_risk_gates(self):
        """Test risk gate functionality"""
        # Good score should pass
        assert self.risk.allowed(75.0, 'test_dev', 0.2)
        
        # Low score should fail
        assert not self.risk.allowed(30.0, 'test_dev', 0.2)
    
    def test_position_sizing(self):
        """Test dynamic position sizing"""
        # High score gets larger position
        size_high = self.risk.get_position_size(85.0, self.config)
        assert size_high == 0.4
        
        # Medium score gets medium position
        size_med = self.risk.get_position_size(67.0, self.config)
        assert size_med == 0.3
        
        # Low score gets small position
        size_low = self.risk.get_position_size(45.0, self.config)
        assert size_low == 0.15


class TestMetrics:
    """Test metrics functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.config = {
            'metrics': {'enabled': False},  # Disable HTTP server for tests
            'alerts': {}
        }
        self.metrics = Metrics(self.config)
    
    def test_counter_metrics(self):
        """Test counter functionality"""
        self.metrics.inc('test_counter')
        assert self.metrics.counters['test_counter'] == 1
        
        self.metrics.inc('test_counter', 5)
        assert self.metrics.counters['test_counter'] == 6
    
    def test_gauge_metrics(self):
        """Test gauge functionality"""
        self.metrics.set('test_gauge', 42.5)
        assert self.metrics.gauges['test_gauge'] == 42.5
    
    def test_histogram_metrics(self):
        """Test histogram functionality"""
        self.metrics.observe('test_histogram', 10.0)
        self.metrics.observe('test_histogram', 20.0)
        
        assert len(self.metrics.histograms['test_histogram']) == 2
        assert 10.0 in self.metrics.histograms['test_histogram']
        assert 20.0 in self.metrics.histograms['test_histogram']
    
    def test_pnl_tracking(self):
        """Test PnL tracking"""
        # Simulate winning trade
        self.metrics.total_pnl_sol = 0.05
        self.metrics.winning_trades = 1
        
        summary = self.metrics.get_summary()
        assert summary['total_pnl_sol'] == 0.05
        assert summary['winning_trades'] == 1
        assert summary['win_rate'] == 100.0


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])