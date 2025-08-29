#!/usr/bin/env python3
"""
Strategy Simulator - Phase 1: Basic Structure
Minimal working version with logging and config setup
This is the foundation that future phases will build upon
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sims.sim_config import load_simulation_config, save_example_config

@dataclass
class SimulatedPosition:
    """Represents a simulated trading position - Phase 1: Basic structure"""
    mint_address: str
    symbol: str
    entry_time: datetime
    entry_price: float
    tokens: float
    usd_invested: float
    strategy: str
    alpha_count: int
    alpha_delay_seconds: float
    
    # Status tracking
    is_active: bool = True
    current_price: float = 0.0
    peak_price: float = 0.0
    
    # Exit data (will be populated in later phases)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    profit_usd: Optional[float] = None
    profit_pct: Optional[float] = None

@dataclass 
class StrategyResults:
    """Results summary for a strategy - Phase 1: Basic metrics"""
    strategy_name: str
    total_trades: int = 0
    active_trades: int = 0
    completed_trades: int = 0
    total_profit_usd: float = 0.0
    total_invested_usd: float = 0.0
    
    def calculate_metrics(self):
        """Calculate derived metrics"""
        self.win_rate = 0.0  # Will implement in Phase 3
        self.avg_profit = 0.0  # Will implement in Phase 3

class BasicSimulator:
    """Phase 1: Basic simulator with logging and storage foundation"""
    
    def __init__(self, config_path: str):
        """Initialize basic simulator"""
        print("🚀 Initializing Strategy Simulator (Phase 1)")
        
        # Load simulation configuration
        self.config = load_simulation_config(config_path)
        
        # Create example config for reference
        save_example_config()
        
        # Setup directories
        self.sims_dir = Path("sims")
        self.data_dir = self.sims_dir / "data"
        self.logs_dir = self.sims_dir / "logs"
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize data structures
        self.positions = {
            "current": {},      # Active positions by mint
            "buffer": {},
            "late_alpha": {}
        }
        
        self.completed_positions = {
            "current": [],      # List of completed positions
            "buffer": [],
            "late_alpha": []
        }
        
        # Results files
        self.results_files = {
            "current": self.data_dir / "strategy_sim_current.json",
            "buffer": self.data_dir / "strategy_sim_buffer.json", 
            "late_alpha": self.data_dir / "strategy_sim_alphas.json"
        }
        
        # Phase 1: Basic validation
        self.validate_setup()
        
    def setup_logging(self):
        """Setup logging system"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.logs_dir / f"simulation_{timestamp}.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger("StrategySimulator")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("📝 Logging system initialized")
        self.logger.info(f"📁 Log file: {log_file}")
        
    def validate_setup(self):
        """Validate Phase 1 setup requirements"""
        self.logger.info("🔍 Validating Phase 1 setup...")
        
        # Check config requirements (graceful handling)
        has_pumpportal = bool(self.config.pumpportal.get('api_key', '').replace('your_api_key_here', ''))
        has_moralis = bool(self.config.moralis_keys and 
                          self.config.moralis_keys[0].replace('your_moralis_key_here', ''))
        
        if not has_pumpportal:
            self.logger.warning("⚠️ PumpPortal API key not configured - will use demo data")
        if not has_moralis:
            self.logger.warning("⚠️ Moralis API key not configured - will use demo data")
        
        # Check alpha wallets
        wallet_count = len(self.config.watched_wallets)
        self.logger.info(f"📊 Alpha wallets loaded: {wallet_count}")
        
        if wallet_count == 0:
            self.logger.warning("⚠️ No alpha wallets configured - using demo mode")
        
        # Test JSON file creation
        self.test_json_storage()
        
        self.logger.info("✅ Phase 1 validation complete")
        
    def test_json_storage(self):
        """Test JSON file writing capabilities"""
        try:
            test_data = {
                "phase": 1,
                "test_time": datetime.now().isoformat(),
                "status": "testing"
            }
            
            for strategy, file_path in self.results_files.items():
                with open(file_path, 'w') as f:
                    json.dump(test_data, f, indent=2)
                
                self.logger.debug(f"✅ JSON test successful: {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ JSON storage test failed: {e}")
            raise
    
    def create_test_position(self, strategy: str) -> SimulatedPosition:
        """Create a test position for Phase 1 validation"""
        test_mint = f"test_mint_{strategy}_{int(datetime.now().timestamp())}"
        
        position = SimulatedPosition(
            mint_address=test_mint,
            symbol=f"TEST_{strategy.upper()}",
            entry_time=datetime.now(),
            entry_price=0.00001234,
            tokens=1000000.0,
            usd_invested=30.0,
            strategy=strategy,
            alpha_count=1,
            alpha_delay_seconds=10.0,
            current_price=0.00001234,
            peak_price=0.00001234
        )
        
        return position
    
    def calculate_strategy_results(self, strategy: str) -> StrategyResults:
        """Calculate basic results for a strategy"""
        active_count = len(self.positions[strategy])
        completed_count = len(self.completed_positions[strategy])
        
        results = StrategyResults(
            strategy_name=strategy,
            total_trades=active_count + completed_count,
            active_trades=active_count,
            completed_trades=completed_count
        )
        
        # Calculate totals from completed positions
        for position in self.completed_positions[strategy]:
            results.total_invested_usd += position.usd_invested
            if position.profit_usd:
                results.total_profit_usd += position.profit_usd
        
        results.calculate_metrics()
        return results
    
    def save_results(self):
        """Save current results to JSON files"""
        try:
            for strategy in ["current", "buffer", "late_alpha"]:
                results = self.calculate_strategy_results(strategy)
                
                # Convert positions to dicts for JSON serialization
                active_positions_data = []
                for mint, position in self.positions[strategy].items():
                    pos_dict = asdict(position)
                    pos_dict['entry_time'] = position.entry_time.isoformat()
                    if position.exit_time:
                        pos_dict['exit_time'] = position.exit_time.isoformat()
                    active_positions_data.append(pos_dict)
                
                completed_positions_data = []
                for position in self.completed_positions[strategy]:
                    pos_dict = asdict(position)
                    pos_dict['entry_time'] = position.entry_time.isoformat()
                    if position.exit_time:
                        pos_dict['exit_time'] = position.exit_time.isoformat()
                    completed_positions_data.append(pos_dict)
                
                # Prepare final data
                data = {
                    "last_updated": datetime.now().isoformat(),
                    "phase": 1,
                    "results": asdict(results),
                    "active_positions": active_positions_data,
                    "completed_positions": completed_positions_data,
                    "summary": {
                        "total_positions": len(active_positions_data) + len(completed_positions_data),
                        "implementation_status": "Phase 1 - Basic Structure"
                    }
                }
                
                # Save to file
                with open(self.results_files[strategy], 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    
                self.logger.debug(f"💾 Saved {strategy} results: {results.total_trades} total trades")
            
            self.logger.info("💾 All strategy results saved successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving results: {e}")
    
    async def run_phase1_demo(self, duration_seconds: int = 60):
        """Run a Phase 1 demonstration with test data"""
        self.logger.info(f"🎯 Starting Phase 1 demo for {duration_seconds} seconds...")
        
        # Create test positions for each strategy
        strategies = ["current", "buffer", "late_alpha"]
        
        for i, strategy in enumerate(strategies):
            # Simulate discovering a position
            position = self.create_test_position(strategy)
            self.positions[strategy][position.mint_address] = position
            
            self.logger.info(f"📈 {strategy.upper()}: Created test position {position.symbol}")
            
            # Simulate some price movement (Phase 1 - just basic updates)
            await asyncio.sleep(2)
            position.current_price = position.entry_price * (1.0 + (i * 0.05))  # 0%, 5%, 10%
            position.peak_price = max(position.peak_price, position.current_price)
        
        # Run for specified duration with periodic saves
        start_time = datetime.now()
        save_interval = 10  # Save every 10 seconds
        last_save = start_time
        
        while (datetime.now() - start_time).total_seconds() < duration_seconds:
            current_time = datetime.now()
            
            # Periodic saves
            if (current_time - last_save).total_seconds() >= save_interval:
                self.save_results()
                last_save = current_time
                
                # Show status
                total_positions = sum(len(self.positions[s]) for s in strategies)
                self.logger.info(f"📊 Status: {total_positions} active positions across all strategies")
            
            await asyncio.sleep(1)
        
        # Final save
        self.save_results()
        
        self.logger.info("✅ Phase 1 demo completed!")
        self.logger.info("📋 Summary:")
        for strategy in strategies:
            results = self.calculate_strategy_results(strategy)
            self.logger.info(f"  {strategy}: {results.total_trades} positions")
        
        return True
    
    async def cleanup(self):
        """Cleanup Phase 1 resources"""
        self.logger.info("🧹 Phase 1 cleanup...")
        self.save_results()
        self.logger.info("✅ Phase 1 cleanup complete")

async def main():
    """Phase 1 main entry point"""
    print("🚀 Strategy Simulator - Phase 1: Basic Structure")
    print("=" * 60)
    
    config_path = "config/config.yml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("💡 Run test_setup.py first to validate your setup")
        return False
    
    try:
        simulator = BasicSimulator(config_path)
        
        print("\nPhase 1 capabilities:")
        print("✅ Config loading and validation")
        print("✅ Logging system setup")
        print("✅ Directory structure creation")
        print("✅ JSON storage system")
        print("✅ Basic position tracking")
        print("⏳ Real-time monitoring (Phase 2)")
        print("⏳ Strategy logic (Phase 3)")
        print("⏳ Advanced analytics (Phase 4)")
        
        # Run demo
        duration = int(input("\nEnter demo duration in seconds (default 30): ") or "30")
        await simulator.run_phase1_demo(duration)
        
        print(f"\n📁 Results saved in: sims/data/")
        print(f"📝 Logs saved in: sims/logs/")
        print("\n🎯 Next Phase: Real-time monitoring implementation")
        
        await simulator.cleanup()
        return True
        
    except KeyboardInterrupt:
        print("\n⛔ Phase 1 stopped by user")
        return False
    except Exception as e:
        print(f"❌ Phase 1 error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)