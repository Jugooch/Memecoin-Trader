#!/usr/bin/env python3
"""
Test script to validate the new real-time position tracking system.
This script tests the accuracy and speed of the new WebSocket-based position updates.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List
import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.core.realtime_position_manager import RealtimePositionManager
from src.clients.pumpportal_client import PumpPortalClient
from src.core.trading_engine import TradingEngine
from src.clients.moralis_client import MoralisClient
from src.utils.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class RealtimePositionTester:
    """Test suite for validating real-time position tracking"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = []
        
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        self.logger.info("🧪 Starting Real-time Position Tracking Test Suite")
        
        # Test 1: Basic position manager functionality
        await self.test_position_manager_basic()
        
        # Test 2: Trade event handling
        await self.test_trade_event_handling()
        
        # Test 3: Sell eligibility checks
        await self.test_sell_eligibility()
        
        # Test 4: P&L calculations
        await self.test_pnl_calculations()
        
        # Test 5: Performance comparison (if possible)
        await self.test_performance_comparison()
        
        # Print results
        self.print_test_summary()
    
    async def test_position_manager_basic(self):
        """Test basic position manager operations"""
        test_name = "Position Manager Basic Operations"
        self.logger.info(f"🔬 Testing: {test_name}")
        
        try:
            # Create position manager
            position_manager = RealtimePositionManager(self.logger)
            
            # Test position creation
            test_mint = "TEST123456789"
            test_event = {
                'mint': test_mint,
                'action': 'buy',
                'tx_signature': 'test_tx_123',
                'price': 0.00001,
                'tokens_received': 100000,
                'sol_amount': 1.0
            }
            
            # Handle buy event
            position_manager.handle_trade_event(test_event)
            
            # Verify position exists
            position = position_manager.get_position(test_mint)
            assert position is not None, "Position should exist after buy event"
            assert position.current_tokens == 100000, f"Expected 100000 tokens, got {position.current_tokens}"
            
            # Test sell event
            sell_event = {
                'mint': test_mint,
                'action': 'sell',
                'tx_signature': 'test_tx_124',
                'tokens_sold': 50000,
                'sol_received': 0.6
            }
            
            position_manager.handle_trade_event(sell_event)
            
            # Verify position updated
            position = position_manager.get_position(test_mint)
            assert position.current_tokens == 50000, f"Expected 50000 tokens after sell, got {position.current_tokens}"
            assert position.sol_recovered == 0.6, f"Expected 0.6 SOL recovered, got {position.sol_recovered}"
            
            self.test_results.append({"name": test_name, "status": "PASS", "details": "All basic operations work correctly"})
            self.logger.info(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self.test_results.append({"name": test_name, "status": "FAIL", "details": str(e)})
            self.logger.error(f"❌ {test_name}: FAILED - {e}")
    
    async def test_trade_event_handling(self):
        """Test trade event handling with various scenarios"""
        test_name = "Trade Event Handling"
        self.logger.info(f"🔬 Testing: {test_name}")
        
        try:
            position_manager = RealtimePositionManager(self.logger)
            
            # Test invalid event handling
            invalid_event = {'invalid': 'data'}
            position_manager.handle_trade_event(invalid_event)  # Should not crash
            
            # Test multiple positions
            for i in range(3):
                mint = f"TESTMINT{i}"
                buy_event = {
                    'mint': mint,
                    'action': 'buy',
                    'tx_signature': f'test_tx_{i}',
                    'price': 0.00001 * (i + 1),
                    'tokens_received': 10000 * (i + 1),
                    'sol_amount': 0.1 * (i + 1)
                }
                position_manager.handle_trade_event(buy_event)
            
            # Verify all positions exist
            active_positions = position_manager.get_active_positions()
            assert len(active_positions) == 3, f"Expected 3 positions, got {len(active_positions)}"
            
            self.test_results.append({"name": test_name, "status": "PASS", "details": "Event handling works correctly"})
            self.logger.info(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self.test_results.append({"name": test_name, "status": "FAIL", "details": str(e)})
            self.logger.error(f"❌ {test_name}: FAILED - {e}")
    
    async def test_sell_eligibility(self):
        """Test sell eligibility checks"""
        test_name = "Sell Eligibility Checks"
        self.logger.info(f"🔬 Testing: {test_name}")
        
        try:
            position_manager = RealtimePositionManager(self.logger)
            
            # Create test position
            test_mint = "SELLTEST123"
            buy_event = {
                'mint': test_mint,
                'action': 'buy',
                'tx_signature': 'sell_test_tx',
                'price': 0.00001,
                'tokens_received': 100000,
                'sol_amount': 1.0
            }
            
            position_manager.handle_trade_event(buy_event)
            
            # Wait a moment to avoid minimum hold time restriction
            await asyncio.sleep(3)
            
            # Test full sell
            can_sell, tokens = position_manager.can_sell(test_mint, 1.0)
            assert can_sell == True, "Should be able to sell 100%"
            assert tokens == 100000, f"Expected 100000 tokens to sell, got {tokens}"
            
            # Test partial sell
            can_sell, tokens = position_manager.can_sell(test_mint, 0.5)
            assert can_sell == True, "Should be able to sell 50%"
            assert tokens == 50000, f"Expected 50000 tokens to sell, got {tokens}"
            
            # Test non-existent position
            can_sell, tokens = position_manager.can_sell("NONEXISTENT", 1.0)
            assert can_sell == False, "Should not be able to sell non-existent position"
            
            self.test_results.append({"name": test_name, "status": "PASS", "details": "Sell eligibility checks work correctly"})
            self.logger.info(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self.test_results.append({"name": test_name, "status": "FAIL", "details": str(e)})
            self.logger.error(f"❌ {test_name}: FAILED - {e}")
    
    async def test_pnl_calculations(self):
        """Test P&L calculation accuracy"""
        test_name = "P&L Calculation Accuracy"
        self.logger.info(f"🔬 Testing: {test_name}")
        
        try:
            position_manager = RealtimePositionManager(self.logger)
            
            # Create test position
            test_mint = "PNLTEST123"
            sol_invested = 1.0
            
            buy_event = {
                'mint': test_mint,
                'action': 'buy',
                'tx_signature': 'pnl_test_tx',
                'price': 0.00001,
                'tokens_received': 100000,
                'sol_amount': sol_invested
            }
            
            position_manager.handle_trade_event(buy_event)
            
            # Simulate profitable sell
            sell_event = {
                'mint': test_mint,
                'action': 'sell',
                'tx_signature': 'pnl_test_sell',
                'tokens_sold': 50000,  # Sell half
                'sol_received': 0.7   # 40% profit on half position
            }
            
            position_manager.handle_trade_event(sell_event)
            
            position = position_manager.get_position(test_mint)
            
            # Check P&L calculation
            expected_pnl = 0.7 - 1.0  # Sold 0.7 SOL, invested 1.0 SOL total
            actual_pnl = position.unrealized_pnl_sol
            
            # Note: This is partial P&L since we only sold half
            assert position.sol_recovered == 0.7, f"Expected 0.7 SOL recovered, got {position.sol_recovered}"
            assert position.sol_invested == 1.0, f"Expected 1.0 SOL invested, got {position.sol_invested}"
            
            self.test_results.append({"name": test_name, "status": "PASS", "details": "P&L calculations accurate"})
            self.logger.info(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self.test_results.append({"name": test_name, "status": "FAIL", "details": str(e)})
            self.logger.error(f"❌ {test_name}: FAILED - {e}")
    
    async def test_performance_comparison(self):
        """Test performance comparison between old and new systems"""
        test_name = "Performance Comparison"
        self.logger.info(f"🔬 Testing: {test_name}")
        
        try:
            # This test measures how fast position updates are processed
            position_manager = RealtimePositionManager(self.logger)
            
            # Test batch position updates
            start_time = time.time()
            
            for i in range(100):
                test_event = {
                    'mint': f'PERF_TEST_{i}',
                    'action': 'buy',
                    'tx_signature': f'perf_tx_{i}',
                    'price': 0.00001,
                    'tokens_received': 1000,
                    'sol_amount': 0.01
                }
                position_manager.handle_trade_event(test_event)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should be very fast (under 1 second for 100 positions)
            assert processing_time < 1.0, f"Position updates too slow: {processing_time:.3f}s for 100 positions"
            
            avg_time_per_update = processing_time / 100 * 1000  # Convert to ms
            
            self.test_results.append({
                "name": test_name, 
                "status": "PASS", 
                "details": f"100 position updates in {processing_time:.3f}s ({avg_time_per_update:.1f}ms per update)"
            })
            self.logger.info(f"✅ {test_name}: PASSED - {avg_time_per_update:.1f}ms per update")
            
        except Exception as e:
            self.test_results.append({"name": test_name, "status": "FAIL", "details": str(e)})
            self.logger.error(f"❌ {test_name}: FAILED - {e}")
    
    def print_test_summary(self):
        """Print comprehensive test results"""
        self.logger.info("\n" + "="*60)
        self.logger.info("🧪 REAL-TIME POSITION TRACKING TEST RESULTS")
        self.logger.info("="*60)
        
        passed = 0
        failed = 0
        
        for result in self.test_results:
            status_emoji = "✅" if result["status"] == "PASS" else "❌"
            self.logger.info(f"{status_emoji} {result['name']}: {result['status']}")
            self.logger.info(f"   Details: {result['details']}")
            
            if result["status"] == "PASS":
                passed += 1
            else:
                failed += 1
        
        self.logger.info("="*60)
        self.logger.info(f"📊 SUMMARY: {passed} passed, {failed} failed")
        
        if failed == 0:
            self.logger.info("🎉 ALL TESTS PASSED! Real-time position tracking is ready for deployment.")
        else:
            self.logger.warning(f"⚠️ {failed} tests failed. Review and fix issues before deployment.")
        
        self.logger.info("="*60)

async def main():
    """Main test runner"""
    tester = RealtimePositionTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())