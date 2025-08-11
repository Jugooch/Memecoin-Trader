#!/usr/bin/env python3
"""
Test script to verify Bitquery token rotation functionality
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.clients.bitquery_client import BitqueryClient
from src.utils.config_loader import load_config


async def test_rotation():
    """Test Bitquery token rotation"""
    print("Loading configuration...")
    config = load_config('config/config.yml')
    
    # Get bitquery tokens
    bitquery_tokens = config.get('bitquery_tokens', config.get('bitquery_token'))
    
    if isinstance(bitquery_tokens, str):
        bitquery_tokens = [bitquery_tokens]
    
    print(f"Found {len(bitquery_tokens)} Bitquery API token(s)")
    
    # Initialize client
    print("\nInitializing Bitquery client...")
    client = BitqueryClient(bitquery_tokens)
    
    try:
        await client.initialize()
        print("Client initialized successfully!")
        
        # Show initial status
        print("\nInitial token status:")
        status = client.get_token_status()
        for token in status['tokens']:
            print(f"  Token #{token['index']}: {token['status']}")
        
        # Try to fetch some data
        print("\nFetching recent token launches...")
        trades = await client.get_recent_token_launches(limit=5)
        
        if trades:
            print(f"Successfully fetched {len(trades)} trades")
        else:
            print("No trades returned (might be due to rate limits)")
        
        # Show status after query
        print("\nToken status after query:")
        status = client.get_token_status()
        for token in status['tokens']:
            status_str = f"  Token #{token['index']}: {token['status']}"
            if token['calls_today'] > 0:
                status_str += f" (calls: {token['calls_today']})"
            if token.get('reset_in'):
                status_str += f" (resets in {token['reset_in']}s)"
            print(status_str)
        
        print(f"\nAvailable tokens: {status['available_tokens']}/{status['total_tokens']}")
        
        if status['all_exhausted']:
            print("WARNING: All tokens are exhausted!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*60)
    print("Bitquery Token Rotation Test")
    print("="*60)
    asyncio.run(test_rotation())