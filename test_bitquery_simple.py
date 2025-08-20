#!/usr/bin/env python3
"""Simple test to check if Bitquery returns prices for pump.fun tokens"""

import asyncio
import aiohttp
import yaml
import json
from datetime import datetime, timedelta

async def test_simple():
    # Load config
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    tokens = config.get('bitquery_tokens', [])
    if not tokens:
        token = config.get('bitquery_token')
        if token:
            tokens = [token]
    
    if not tokens:
        print("No Bitquery tokens found!")
        return
    
    access_token = tokens[5]
    
    # Get trades from last 10 minutes
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=10)
    
    query = f"""
    query {{
      Solana {{
        DEXTrades(
          where: {{
            Trade: {{ Dex: {{ ProtocolName: {{ is: "pump" }} }} }}
            Transaction: {{ Result: {{ Success: true }} }}
            Block: {{ Time: {{ since: "{start_time.isoformat()}Z", till: "{end_time.isoformat()}Z" }} }}
          }}
          limit: {{ count: 10 }}
          orderBy: {{ descending: Block_Time }}
        ) {{
          Block {{ Time }}
          Trade {{
            Buy {{
              AmountInUSD
              PriceInUSD
              Currency {{
                Symbol
                MintAddress
              }}
            }}
            Sell {{
              AmountInUSD
              PriceInUSD
              Currency {{
                Symbol
                MintAddress
              }}
            }}
          }}
        }}
      }}
    }}
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    url = "https://streaming.bitquery.io/eap"
    
    print("Checking last 10 pump.fun trades for price data...")
    print("-" * 50)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"query": query}, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                
                if 'data' in data and data['data']:
                    trades = data['data'].get('Solana', {}).get('DEXTrades', [])
                    
                    print(f"Got {len(trades)} trades\n")
                    
                    for i, trade in enumerate(trades[:5]):  # Show first 5
                        print(f"Trade {i+1}:")
                        print(f"  Time: {trade['Block']['Time']}")
                        
                        buy = trade['Trade']['Buy']
                        sell = trade['Trade']['Sell']
                        
                        print(f"  Buy side:")
                        print(f"    Token: {buy['Currency']['Symbol']} ({buy['Currency']['MintAddress'][:8]}...)")
                        print(f"    PriceInUSD: {buy.get('PriceInUSD', 'MISSING')}")
                        print(f"    AmountInUSD: {buy.get('AmountInUSD', 'MISSING')}")
                        
                        print(f"  Sell side:")
                        print(f"    Token: {sell['Currency']['Symbol']} ({sell['Currency']['MintAddress'][:8]}...)")
                        print(f"    PriceInUSD: {sell.get('PriceInUSD', 'MISSING')}")
                        print(f"    AmountInUSD: {sell.get('AmountInUSD', 'MISSING')}")
                        print()
                        
                    # Count how many have prices
                    trades_with_buy_price = sum(1 for t in trades if t['Trade']['Buy'].get('PriceInUSD') is not None)
                    trades_with_sell_price = sum(1 for t in trades if t['Trade']['Sell'].get('PriceInUSD') is not None)
                    
                    print(f"Summary:")
                    print(f"  Trades with Buy PriceInUSD: {trades_with_buy_price}/{len(trades)}")
                    print(f"  Trades with Sell PriceInUSD: {trades_with_sell_price}/{len(trades)}")
                    
                else:
                    print("No data returned")
            else:
                print(f"Error: {response.status}")

if __name__ == "__main__":
    asyncio.run(test_simple())