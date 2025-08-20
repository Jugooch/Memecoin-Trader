#!/usr/bin/env python3
"""Test Bitquery API directly to see what's being returned"""

import asyncio
import aiohttp
import yaml
import json
from datetime import datetime, timedelta

async def test_bitquery_api():
    # Load config to get access tokens
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get first access token from the list
    tokens = config.get('bitquery_tokens', [])
    if not tokens:
        # Fallback to single token
        token = config.get('bitquery_token')
        if token:
            tokens = [token]
    
    if not tokens:
        print("No Bitquery tokens found in config!")
        return
    
    access_token = tokens[5]  # Use first token
    print(f"Using access token: {access_token[:20]}...")
    
    # Test with a token that's actually trading right now (from the output above)
    # You can change this to any token you see in the output
    test_token = "48AmAerMuxU3t3ngSrFJDxtQXpKBjWmUuavtpump"  # Ank token from the trades
    
    # Set time window similar to your discovery
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=30)  # 30 minute window
    
    # Use the same query format as your actual code
    query = f"""
    query {{
      Solana {{
        DEXTrades(
          where: {{
            Trade: {{ Dex: {{ ProtocolName: {{ is: "pump" }} }} }}
            Transaction: {{ Result: {{ Success: true }} }}
            Block: {{ Time: {{ since: "{start_time.isoformat()}Z", till: "{end_time.isoformat()}Z" }} }}
          }}
          limit: {{ count: 100 }}
          orderBy: {{ descending: Block_Time }}
        ) {{
          Block {{ Time }}
          Trade {{
            Dex {{
              ProtocolFamily
              ProtocolName
            }}
            Buy {{
              Amount
              AmountInUSD
              Price
              PriceInUSD
              Account {{ Address }}
              Currency {{
                Symbol
                MintAddress
              }}
            }}
            Sell {{
              Amount
              AmountInUSD
              Price
              PriceInUSD
              Account {{ Address }}
              Currency {{
                Symbol
                MintAddress
              }}
            }}
          }}
          Transaction {{
            Signature
            Signer
          }}
        }}
      }}
    }}
    """
    
    # No variables needed for this query format
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    # Use the EAP endpoint like your actual code does
    url = "https://streaming.bitquery.io/eap"
    
    print(f"Testing Bitquery API for token: {test_token}")
    print(f"Time window: {start_time} to {end_time}")
    print("-" * 50)
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                url,
                json={"query": query},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                print(f"Status: {response.status}")
                
                text = await response.text()
                
                if response.status == 200:
                    try:
                        data = json.loads(text)
                        
                        if 'errors' in data:
                            print(f"GraphQL Errors: {data['errors']}")
                        
                        if 'data' in data and data['data']:
                            trades = data['data'].get('Solana', {}).get('DEXTrades', [])
                            print(f"Number of trades returned: {len(trades)}")
                            
                            # Filter for the specific token we're interested in
                            token_trades = []
                            for trade in trades:
                                buy_mint = trade.get('Trade', {}).get('Buy', {}).get('Currency', {}).get('MintAddress')
                                sell_mint = trade.get('Trade', {}).get('Sell', {}).get('Currency', {}).get('MintAddress')
                                
                                # Debug: show what we're comparing
                                if buy_mint and test_token[:8] in buy_mint[:8]:
                                    print(f"DEBUG: Comparing buy_mint '{buy_mint}' with test_token '{test_token}' - Match: {buy_mint == test_token}")
                                if sell_mint and test_token[:8] in sell_mint[:8]:
                                    print(f"DEBUG: Comparing sell_mint '{sell_mint}' with test_token '{test_token}' - Match: {sell_mint == test_token}")
                                
                                if buy_mint == test_token or sell_mint == test_token:
                                    token_trades.append(trade)
                            
                            print(f"Trades for token {test_token[:8]}...: {len(token_trades)}")
                            
                            # Show what tokens ARE being returned
                            if len(token_trades) == 0 and len(trades) > 0:
                                print("\nTokens actually returned in the trades:")
                                seen_tokens = set()
                                trades_with_price = 0
                                for trade in trades[:20]:  # Check first 20
                                    buy_mint = trade.get('Trade', {}).get('Buy', {}).get('Currency', {}).get('MintAddress')
                                    sell_mint = trade.get('Trade', {}).get('Sell', {}).get('Currency', {}).get('MintAddress')
                                    if buy_mint and buy_mint not in seen_tokens:
                                        buy_symbol = trade.get('Trade', {}).get('Buy', {}).get('Currency', {}).get('Symbol', 'Unknown')
                                        # Show full address for Ank token
                                        if buy_symbol == 'Ank':
                                            print(f"  Buy: {buy_mint} ({buy_symbol}) <-- FULL ADDRESS")
                                        else:
                                            print(f"  Buy: {buy_mint[:8]}... ({buy_symbol})")
                                        seen_tokens.add(buy_mint)
                                    if sell_mint and sell_mint not in seen_tokens:
                                        sell_symbol = trade.get('Trade', {}).get('Sell', {}).get('Currency', {}).get('Symbol', 'Unknown')
                                        print(f"  Sell: {sell_mint[:8]}... ({sell_symbol})")
                                        seen_tokens.add(sell_mint)
                                    
                                    # Check if this trade has price data
                                    buy_price = trade.get('Trade', {}).get('Buy', {}).get('PriceInUSD')
                                    sell_price = trade.get('Trade', {}).get('Sell', {}).get('PriceInUSD')
                                    if buy_price or sell_price:
                                        trades_with_price += 1
                                
                                print(f"\nTrades with PriceInUSD data: {trades_with_price}/{len(trades[:20])}")
                            
                            # Extract price progression
                            price_progression = []
                            for trade in token_trades[::-1]:  # Reverse to get chronological order
                                # Get price from Buy or Sell side depending on which has our token
                                buy_mint = trade.get('Trade', {}).get('Buy', {}).get('Currency', {}).get('MintAddress')
                                if buy_mint == test_token:
                                    price = trade.get('Trade', {}).get('Buy', {}).get('PriceInUSD')
                                else:
                                    price = trade.get('Trade', {}).get('Sell', {}).get('PriceInUSD')
                                
                                if price:
                                    price_progression.append(float(price))
                            
                            print(f"Price progression (first 10): {price_progression[:10]}")
                            
                            # Calculate performance like the code does
                            if len(price_progression) >= 2:
                                early_prices = [p for p in price_progression[:2] if p > 0]
                                later_prices = [p for p in price_progression[2:] if p > 0]
                                
                                if early_prices and later_prices:
                                    early_price = early_prices[0]
                                    peak_price = max(later_prices)
                                    performance_multiplier = peak_price / early_price
                                    print(f"Early price: ${early_price:.8f}")
                                    print(f"Peak price: ${peak_price:.8f}")
                                    print(f"Performance multiplier: {performance_multiplier:.2f}x")
                                else:
                                    print("Not enough valid prices for performance calculation")
                            else:
                                print(f"Not enough trades for performance calculation (only {len(price_progression)} prices)")
                            
                            # Show first few trades for our token
                            print("\nFirst 3 trades for our token:")
                            for i, trade in enumerate(token_trades[:3]):
                                print(f"  Trade {i+1}:")
                                print(f"    Time: {trade['Block']['Time']}")
                                buy_mint = trade.get('Trade', {}).get('Buy', {}).get('Currency', {}).get('MintAddress')
                                if buy_mint == test_token:
                                    print(f"    Buy Price USD: {trade['Trade']['Buy'].get('PriceInUSD')}")
                                    print(f"    Buy Amount USD: {trade['Trade']['Buy'].get('AmountInUSD')}")
                                else:
                                    print(f"    Sell Price USD: {trade['Trade']['Sell'].get('PriceInUSD')}")
                                    print(f"    Sell Amount USD: {trade['Trade']['Sell'].get('AmountInUSD')}")
                        else:
                            print("No data returned")
                            print(f"Full response: {json.dumps(data, indent=2)[:1000]}")
                            
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON: {e}")
                        print(f"Raw response: {text[:500]}")
                else:
                    print(f"Error response: {text[:500]}")
                    
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_bitquery_api())