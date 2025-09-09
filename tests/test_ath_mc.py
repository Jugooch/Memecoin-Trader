#!/usr/bin/env python3
"""
Test script to verify ATH market cap fetching from Bitquery
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clients.bitquery_client import BitqueryClient
from src.utils.config_loader import load_config
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_ath_mc():
    """Test getting ATH market cap for known tokens"""
    
    # Load config
    config = load_config('config/config_sniper.yml')
    
    # Initialize Bitquery client
    bitquery = BitqueryClient(
        api_key=config['bitquery']['api_key'],
        api_url=config['bitquery']['api_url']
    )
    
    # Test with some known pump.fun tokens that had significant runs
    test_tokens = [
        # Add some known pump.fun token addresses here
        # These are examples - replace with actual tokens you want to test
        "7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr",  # POPCAT
        "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",  # WIF
    ]
    
    print("\n=== Testing ATH Market Cap Fetching ===\n")
    
    for token in test_tokens:
        print(f"Testing token: {token[:8]}...")
        try:
            ath_mc = await bitquery.get_token_ath_market_cap(token)
            if ath_mc > 0:
                print(f"✅ ATH Market Cap: ${ath_mc:,.0f}")
            else:
                print(f"❌ No ATH data found")
        except Exception as e:
            print(f"❌ Error: {e}")
        print()
    
    # Test with a known developer wallet
    test_dev = "HHjTVadwjMLkN7U9PR3gYUuqeuFTUx8zHucVBhCYPKJu"  # From your logs
    
    print(f"\n=== Testing Developer History with ATH MC ===\n")
    print(f"Developer: {test_dev[:8]}...\n")
    
    try:
        history = await bitquery.get_dev_tokens_with_ath_mc(test_dev, lookback_days=60)
        
        print(f"Total tokens launched: {history.get('total_tokens', 0)}")
        print(f"Tokens in last 7 days: {history.get('recent_launches', 0)}")
        print(f"Best ATH Market Cap: ${history.get('best_peak_mc', 0):,.0f}")
        print(f"Median ATH Market Cap: ${history.get('median_peak_mc', 0):,.0f}")
        print(f"Tokens with MC data: {history.get('tokens_with_mc_data', 0)}")
        
        if history.get('token_addresses'):
            print(f"\nRecent tokens:")
            for addr in history['token_addresses'][:5]:
                print(f"  - {addr}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ath_mc())