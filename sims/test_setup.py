#!/usr/bin/env python3
"""
Phase 1 Test: Basic Setup and Import Validation
Tests that all required dependencies and imports work correctly
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required imports work"""
    print("🧪 Testing imports...")
    
    try:
        # Core Python imports
        import asyncio
        import json
        import logging
        from datetime import datetime, timedelta
        from typing import Dict, List, Optional, Any
        from dataclasses import dataclass, asdict
        print("  ✅ Core Python imports: OK")
        
        # Bot imports
        from src.clients.pumpportal_client import PumpPortalClient
        print("  ✅ PumpPortal client: OK")
        
        from src.clients.moralis_client import MoralisClient
        print("  ✅ Moralis client: OK")
        
        from sims.sim_config import load_simulation_config, SimConfig
        print("  ✅ Simulation config loader: OK")
        
        from src.utils.discord_notifier import DiscordNotifier
        print("  ✅ Discord notifier: OK")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_config_loading():
    """Test config file loading with graceful fallbacks"""
    print("\n📝 Testing simulation config loading...")
    
    config_path = "config/config.yml"
    
    try:
        from sims.sim_config import load_simulation_config
        config = load_simulation_config(config_path)
        
        # Check key fields (with graceful handling)
        wallet_count = len(config.watched_wallets)
        print(f"  📊 Alpha wallets loaded: {wallet_count}")
        
        if wallet_count == 0:
            print("  ⚠️ No alpha wallets - using test wallets for simulation")
        
        has_pumpportal = bool(config.pumpportal.get('api_key', '').replace('your_api_key_here', ''))
        print(f"  🔑 PumpPortal API key configured: {has_pumpportal}")
        
        has_moralis = bool(config.moralis_keys and 
                          config.moralis_keys[0].replace('your_moralis_key_here', ''))
        print(f"  🔑 Moralis API key configured: {has_moralis}")
        
        if not has_pumpportal or not has_moralis:
            print("  💡 Missing API keys - simulation will run in demo mode")
        
        print("  ✅ Config loading successful (with fallbacks)")
        return True
        
    except Exception as e:
        print(f"  ❌ Config loading error: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist or can be created"""
    print("\n📁 Testing directory structure...")
    
    base_dir = Path("sims")
    required_dirs = [
        base_dir / "data",
        base_dir / "logs", 
        base_dir / "configs"
    ]
    
    try:
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Directory ready: {directory}")
        
        # Test write permissions
        test_file = base_dir / "data" / "test_write.json"
        with open(test_file, 'w') as f:
            json.dump({"test": True}, f)
        test_file.unlink()  # Delete test file
        
        print("  ✅ Write permissions: OK")
        return True
        
    except Exception as e:
        print(f"  ❌ Directory setup error: {e}")
        return False

def test_dataclass_structure():
    """Test that dataclass structures work correctly"""
    print("\n🏗️ Testing dataclass structures...")
    
    try:
        from datetime import datetime
        from dataclasses import dataclass, asdict
        
        @dataclass
        class TestPosition:
            mint_address: str
            entry_time: datetime
            entry_price: float
            
        # Test creation
        test_pos = TestPosition(
            mint_address="test123",
            entry_time=datetime.now(),
            entry_price=0.00001234
        )
        
        # Test serialization
        pos_dict = asdict(test_pos)
        
        print("  ✅ Dataclass creation: OK")
        print("  ✅ Dataclass serialization: OK")
        return True
        
    except Exception as e:
        print(f"  ❌ Dataclass test error: {e}")
        return False

def main():
    """Run all Phase 1 tests"""
    print("🚀 Phase 1: Setup & Structure Tests")
    print("=" * 50)
    
    tests = [
        ("Import validation", test_imports),
        ("Config loading", test_config_loading), 
        ("Directory structure", test_directory_structure),
        ("Dataclass structure", test_dataclass_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Phase 1 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ Phase 1 COMPLETE - Ready for Phase 2!")
        print("\nNext steps:")
        print("1. Verify PumpPortal API connectivity")
        print("2. Test alpha wallet monitoring") 
        print("3. Implement real-time data collection")
        return True
    else:
        print("❌ Phase 1 INCOMPLETE - Fix issues above before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)