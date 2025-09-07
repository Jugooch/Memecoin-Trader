"""
Bitquery GraphQL client for monitoring Pump.fun token launches
"""

import asyncio
import aiohttp
import json
import logging
import time
from typing import Dict, List, AsyncGenerator, Union
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.websockets import WebsocketsTransport


class BitqueryClient:
    def __init__(self, api_tokens: Union[str, List[str]]):
        # Support both single token (string) and multiple tokens (list)
        if isinstance(api_tokens, str):
            self.api_tokens = [api_tokens]
        else:
            self.api_tokens = api_tokens
            
        self.logger = logging.getLogger(__name__)
        self._init_lock = asyncio.Lock()
        self._concurrent_semaphore = asyncio.Semaphore(8)  # Allow up to 8 concurrent queries
        
        # GraphQL endpoint (using EAP for Solana data)
        self.endpoint = "https://streaming.bitquery.io/eap"
        self.ws_endpoint = "wss://streaming.bitquery.io/eap"
        
        # Pump.fun program address
        self.pumpfun_program = "6EF8rrecthHAuSStzpf6aXr9HWs8jgPVrj5S6fqF6P"
        
        # Token rotation management
        self.current_token_index = 0
        self.token_stats = {}
        for i, token in enumerate(self.api_tokens):
            self.token_stats[i] = {
                'calls_today': 0,
                'rate_limited': False,
                'reset_time': 0,
                'token': token,
                'payment_required': False  # Track 402 errors
            }
        
        # Global rate limiting
        self.rate_limited = False
        self.rate_limit_reset_time = 0
        
        self.client = None
        self.ws_client = None
        self.current_client_token_index = None

    def _get_next_available_token(self) -> tuple:
        """Get the next available API token that's not rate limited or payment required"""
        for _ in range(len(self.api_tokens)):
            token_info = self.token_stats[self.current_token_index]
            
            # Check if this token has payment issues (402)
            if token_info['payment_required']:
                self.logger.debug(f"Token {self.current_token_index} has payment issues, trying next...")
                self.current_token_index = (self.current_token_index + 1) % len(self.api_tokens)
                continue
            
            # Check if this token is rate limited
            if token_info['rate_limited'] and time.time() < token_info['reset_time']:
                self.logger.debug(f"Token {self.current_token_index} still rate limited, trying next...")
                self.current_token_index = (self.current_token_index + 1) % len(self.api_tokens)
                continue
            
            # This token is available
            return self.current_token_index, token_info['token']
        
        # All tokens are either rate limited or have payment issues
        self.rate_limited = True
        self.rate_limit_reset_time = min(
            token['reset_time'] for token in self.token_stats.values() 
            if not token['payment_required']
        ) if any(not token['payment_required'] for token in self.token_stats.values()) else 0
        return None, None

    async def initialize(self):
        """Initialize GraphQL clients with current available token"""
        token_index, api_token = self._get_next_available_token()
        
        if not api_token:
            self.logger.error("All Bitquery API tokens exhausted or have payment issues!")
            raise Exception("No available Bitquery API tokens")
        
        # Only reinitialize if token changed
        if self.current_client_token_index == token_index and self.client:
            return
            
        self.logger.info(f"Initializing Bitquery client with token #{token_index}")
        
        # HTTP transport for queries (using Bearer token as per Bitquery docs)
        http_transport = AIOHTTPTransport(
            url=self.endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_token}"
            }
        )
        self.client = Client(transport=http_transport)
        
        # Skip WebSocket client initialization - it's not needed for historical queries
        # and it causes hangs during startup. We'll initialize it on-demand if needed.
        self.ws_client = None
        self.current_client_token_index = token_index
        
        self.logger.info(f"Bitquery HTTP client initialized successfully (WebSocket skipped)")

    async def subscribe_token_launches(self) -> AsyncGenerator[Dict, None]:
        """Subscribe to new Pump.fun token creation events"""
        # Try WebSocket first with proper Bitquery protocol
        try:
            async for token_data in self._websocket_subscription():
                yield token_data
        except Exception as e:
            self.logger.error(f"WebSocket subscription failed: {e}")
            self.logger.info("Falling back to polling mode...")
            
            # Fallback to polling recent launches
            while True:
                try:
                    recent_trades = await self.get_recent_token_launches(limit=10)
                    for trade in recent_trades:
                        # Parse the raw trade data to extract token info
                        token_data = self._parse_dex_trade(trade)
                        if token_data:
                            yield token_data
                    await asyncio.sleep(30)  # Poll every 30 seconds
                except Exception as poll_error:
                    self.logger.error(f"Polling error: {poll_error}")
                    await asyncio.sleep(60)

    async def _websocket_subscription(self) -> AsyncGenerator[Dict, None]:
        """Direct WebSocket implementation following Bitquery docs exactly"""
        import websockets
        import json
        
        # Get current token
        token_index, api_token = self._get_next_available_token()
        if not api_token:
            raise Exception("No available Bitquery API tokens")
        
        # EAP endpoint with token in URL (as per Bitquery docs)
        ws_url = f"wss://streaming.bitquery.io/eap?token={api_token}"
        
        # Required headers (only these two)
        headers = {
            "Content-Type": "application/json"
        }
        
        self.logger.info(f"Connecting to Bitquery WebSocket: {ws_url[:50]}...")
        
        async with websockets.connect(ws_url, additional_headers=headers) as websocket:
            # Step 1: Initialize connection
            await websocket.send(json.dumps({"type": "connection_init"}))
            self.logger.debug("Sent connection_init")
            
            # Step 2: Wait for acknowledgment
            while True:
                response = await websocket.recv()
                response_data = json.loads(response)
                if response_data.get("type") == "connection_ack":
                    self.logger.info("WebSocket connection acknowledged")
                    break
                elif response_data.get("type") == "error":
                    raise Exception(f"Connection error: {response_data}")
            
            # Step 3: Send subscription
            subscription_query = """
            subscription {
              Solana {
                DEXTrades(
                  where: {
                    Trade: { Dex: { ProtocolName: { is: "pump" } } }
                    Transaction: { Result: { Success: true } }
                  }
                ) {
                  Block { Time }
                  Trade {
                    Dex {
                      ProtocolFamily
                      ProtocolName
                    }
                    Buy {
                      Amount
                      Account { Address }
                      Currency {
                        Symbol
                        MintAddress
                      }
                    }
                    Sell {
                      Amount
                      Account { Address }
                      Currency {
                        Symbol
                        MintAddress
                      }
                    }
                  }
                  Transaction { Signature }
                }
              }
            }
            """
            
            await websocket.send(json.dumps({
                "type": "start",
                "id": "1", 
                "payload": {"query": subscription_query}
            }))
            
            self.logger.info("Bitquery WebSocket subscription started - listening for pump.fun trades...")
            
            # Step 4: Listen for messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    if message_type == "data":
                        # Extract trades from the payload
                        trades = data.get("payload", {}).get("data", {}).get("Solana", {}).get("DEXTrades", [])
                        for trade in trades:
                            token_data = self._parse_dex_trade(trade)
                            if token_data:
                                yield token_data
                                
                    elif message_type == "ka":
                        # Keep-alive message - just log occasionally
                        pass
                        
                    elif message_type == "error":
                        self.logger.error(f"WebSocket error: {data}")
                        raise Exception(f"Subscription error: {data}")
                        
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing WebSocket message: {e}")
                    raise

    def _parse_dex_trade(self, trade: Dict) -> Dict:
        """Parse DEX trade data to extract new token info"""
        try:
            transaction = trade.get('Transaction', {})
            block = trade.get('Block', {})
            trade_data = trade.get('Trade', {})
            
            # Extract buy currency (new token) - prefer buy side for new tokens
            buy_data = trade_data.get('Buy', {})
            sell_data = trade_data.get('Sell', {})
            
            # For pump.fun, the "buy" is usually SOL->Token, so token is buy currency
            buy_currency = buy_data.get('Currency', {})
            mint_address = buy_currency.get('MintAddress')
            
            # If buy currency is SOL, check sell currency for the token
            if not mint_address or mint_address in ['So11111111111111111111111111111112', '11111111111111111111111111111111']:
                sell_currency = sell_data.get('Currency', {})
                mint_address = sell_currency.get('MintAddress')
                if mint_address:
                    buy_currency = sell_currency
            
            # Filter out SOL and other system tokens
            if not mint_address or mint_address in ['So11111111111111111111111111111112', '11111111111111111111111111111111']:
                return None
                
            # Filter out very short addresses (invalid)
            if len(mint_address) < 40:
                return None
                
            return {
                'mint': mint_address,
                'name': buy_currency.get('Name', ''),
                'symbol': buy_currency.get('Symbol', ''),
                'deployer': transaction.get('Signer'),
                'timestamp': block.get('Time'),
                'signature': transaction.get('Signature'),
                'buyer': buy_data.get('Account', {}).get('Address'),
                'buy_amount': buy_data.get('Amount', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing DEX trade: {e}")
            return None

    def _parse_token_creation(self, instruction: Dict) -> Dict:
        """Parse token creation instruction data"""
        try:
            accounts = instruction.get('Instruction', {}).get('Accounts', [])
            transaction = instruction.get('Transaction', {})
            block = instruction.get('Block', {})
            
            # Extract mint address and deployer
            mint_address = None
            deployer = transaction.get('Signer')
            
            for account in accounts:
                if account.get('Token', {}).get('Mint'):
                    mint_address = account['Token']['Mint']
                    break
            
            if not mint_address:
                return None
                
            return {
                'mint': mint_address,
                'deployer': deployer,
                'timestamp': block.get('Time'),
                'block_number': block.get('Number'),
                'signature': transaction.get('Signature')
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing token creation: {e}")
            return None

    async def get_recent_token_launches(self, limit: int = 50, start_time: str = None, end_time: str = None) -> List[Dict]:
        """Get recent token launches via query with optional time filtering"""
        if not self.client:
            await self.initialize()
        
        # Build time filter if provided (BitQuery uses 'since' and 'till', not 'between')
        time_filter = ""
        if start_time and end_time:
            time_filter = f'Block: {{ Time: {{ since: "{start_time}", till: "{end_time}" }} }}'
        elif start_time:
            time_filter = f'Block: {{ Time: {{ since: "{start_time}" }} }}'
        elif end_time:
            time_filter = f'Block: {{ Time: {{ till: "{end_time}" }} }}'
            
        query = gql(f"""
            query {{
              Solana {{
                DEXTrades(
                  where: {{
                    Trade: {{ Dex: {{ ProtocolName: {{ is: "pump" }} }} }}
                    Transaction: {{ Result: {{ Success: true }} }}
                    {time_filter}
                  }}
                  limit: {{ count: {limit} }}
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
                        MintAddress
                        Symbol
                        Name
                      }}
                    }}
                    Sell {{
                      Amount
                      AmountInUSD
                      Price
                      PriceInUSD
                      Account {{ Address }}
                      Currency {{
                        MintAddress
                        Symbol
                        Name
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
        """)
        
        try:
            # Debug: Log the GraphQL time filter being used
            self.logger.info(f"Executing BitQuery with filter: '{time_filter if time_filter else 'NO FILTER'}'")
            result = await self.client.execute_async(query)
            
            # Return raw trades for alpha discovery - let caller handle parsing
            trades = result['Solana']['DEXTrades']
            
            # Log summary instead of full data to avoid flooding console
            self.logger.info(f"BitQuery client returned {len(trades)} trades")
            
            # Update stats for successful call
            if self.current_client_token_index is not None:
                self.token_stats[self.current_client_token_index]['calls_today'] += 1
            
            return trades
            
        except Exception as e:
            error_str = str(e)
            self.logger.error(f"Error fetching recent launches: {error_str}")
            
            # Check for specific error types
            if '402' in error_str or 'Payment Required' in error_str:
                # Mark current token as payment required
                if self.current_client_token_index is not None:
                    self.token_stats[self.current_client_token_index]['payment_required'] = True
                    self.logger.warning(f"Token #{self.current_client_token_index} marked as payment required")
                
                # Try to rotate to next token
                self.current_token_index = (self.current_token_index + 1) % len(self.api_tokens)
                
                # Reinitialize with new token if available
                try:
                    await self.initialize()
                    # Retry the query with new token
                    self.logger.info("Retrying query with new token...")
                    return await self.get_recent_token_launches(limit, start_time, end_time)
                except Exception as reinit_error:
                    self.logger.error(f"Failed to reinitialize with new token: {reinit_error}")
                    return []
            
            return []
    
    async def get_token_price_history(self, mint_address: str, start_iso: str, end_iso: str, limit: int = 500) -> List[Dict]:
        """Get historical price data for a specific token using DEX trades - EFFICIENT for API credits
        
        This method is designed to be credit-efficient for Bitquery usage:
        - Limited to 500 trades by default (adjustable)
        - Focuses on successful pump.fun trades only
        - Gets actual transaction prices from DEX data
        
        Args:
            mint_address: Token mint address
            start_iso: Start time in ISO format (e.g., "2024-01-01T00:00:00Z")
            end_iso: End time in ISO format  
            limit: Maximum number of trades (default 500 to conserve credits)
            
        Returns:
            List of price points: [{'timestamp': str, 'price': float, 'volume_usd': float}, ...]
        """
        if not self.client:
            await self.initialize()
        
        try:
            query = gql(f"""
                query {{
                  Solana {{
                    DEXTrades(
                      where: {{
                        Trade: {{ 
                          Dex: {{ ProtocolName: {{ is: "pump" }} }}
                          Buy: {{ Currency: {{ MintAddress: {{ is: "{mint_address}" }} }} }}
                        }}
                        Transaction: {{ Result: {{ Success: true }} }}
                        Block: {{ Time: {{ since: "{start_iso}", till: "{end_iso}" }} }}
                      }}
                      limit: {{ count: {limit} }}
                      orderBy: {{ ascending: Block_Time }}
                    ) {{
                      Block {{ Time }}
                      Trade {{
                        Buy {{
                          Amount
                          AmountInUSD
                          PriceInUSD
                        }}
                      }}
                    }}
                  }}
                }}
            """)
            
            # Use current token for the API call
            current_token_info = self.token_stats[self.current_token_index]
            
            if current_token_info.get('payment_required', False):
                self.logger.warning(f"Skipping payment required token #{self.current_token_index}")
                return []
                
            if current_token_info.get('rate_limited', False):
                if time.time() < current_token_info.get('reset_time', 0):
                    self.logger.warning(f"Token {self.current_token_index} is rate limited")
                    return []
                else:
                    current_token_info['rate_limited'] = False
            
            result = await self.client.execute_async(query)
            
            # Update stats
            current_token_info['calls_today'] += 1
            self.logger.debug(f"Bitquery API call #{current_token_info['calls_today']} for token #{self.current_token_index}")
            
            trades = result.get('Solana', {}).get('DEXTrades', [])
            
            # Convert to standardized price history format
            price_history = []
            for trade in trades:
                buy_data = trade.get('Trade', {}).get('Buy', {})
                price_usd = buy_data.get('PriceInUSD')
                
                if price_usd and float(price_usd) > 0:
                    price_history.append({
                        'timestamp': trade['Block']['Time'],
                        'price': float(price_usd),
                        'volume_usd': float(buy_data.get('AmountInUSD', 0))
                    })
            
            self.logger.info(f"Bitquery: Retrieved {len(price_history)} price points for {mint_address[:8]}...")
            return price_history
            
        except Exception as e:
            error_str = str(e)
            self.logger.error(f"Bitquery price history error: {error_str}")
            
            # Handle API limits and errors
            if '402' in error_str or 'Payment Required' in error_str:
                current_token_info['payment_required'] = True
                self.logger.warning(f"[PAYMENT] Bitquery payment required for token #{self.current_token_index}")
            elif '429' in error_str or 'rate limit' in error_str.lower():
                current_token_info['rate_limited'] = True
                current_token_info['reset_time'] = time.time() + 3600  # 1 hour
                self.logger.warning(f"[RATE_LIMIT] Bitquery rate limited for token #{self.current_token_index}")
            
            return []
    
    async def get_trades_windowed_paginated(self, start_iso: str, end_iso: str, page_limit: int = 3000, max_pages: int = 20) -> List[Dict]:
        """Paginate Bitquery to actually cover the full time window
        
        Args:
            start_iso: Start time in ISO format (e.g., "2024-01-01T00:00:00Z")
            end_iso: End time in ISO format
            page_limit: Number of trades per page
            max_pages: Maximum number of pages to fetch
            
        Returns:
            List of all trades in the window
        """
        if not self.client:
            await self.initialize()
            
        all_trades = []
        till = end_iso
        
        self.logger.info(f"Starting paginated fetch from {start_iso} to {end_iso} (max {max_pages} pages)")
        
        page = 0
        while page < max_pages:
            try:
                # Build time filter for this page
                time_filter = f'Block: {{ Time: {{ since: "{start_iso}", till: "{till}" }} }}'
                
                query = gql(f"""
                    query {{
                      Solana {{
                        DEXTrades(
                          where: {{
                            Trade: {{ Dex: {{ ProtocolName: {{ is: "pump" }} }} }}
                            Transaction: {{ Result: {{ Success: true }} }}
                            {time_filter}
                          }}
                          limit: {{ count: {page_limit} }}
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
                                MintAddress
                                Symbol
                                Name
                              }}
                            }}
                            Sell {{
                              Amount
                              AmountInUSD
                              Price
                              PriceInUSD
                              Account {{ Address }}
                              Currency {{
                                MintAddress
                                Symbol
                                Name
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
                """)
                
                self.logger.debug(f"Fetching page {page + 1}/{max_pages} with till={till}")
                result = await self.client.execute_async(query)
                batch = result['Solana']['DEXTrades']
                
                if not batch:
                    self.logger.info(f"No more trades found at page {page + 1}, stopping pagination")
                    break
                    
                all_trades.extend(batch)
                
                # Find minimum Block.Time in this batch to step the window back
                min_ts = None
                for trade in batch:
                    time_str = trade.get('Block', {}).get('Time')
                    if time_str:
                        ts = self._parse_iso_timestamp(time_str)
                        if min_ts is None or ts < min_ts:
                            min_ts = ts
                
                if min_ts is None:
                    self.logger.warning(f"No valid timestamps in batch {page + 1}, stopping pagination")
                    break
                
                # Move 'till' to just before min_ts to avoid overlap
                from datetime import datetime, timedelta
                new_till_dt = datetime.utcfromtimestamp(min_ts) - timedelta(seconds=1)
                till = new_till_dt.isoformat() + "Z"
                
                # Stop if we've gone past the start time
                if self._parse_iso_timestamp(till) <= self._parse_iso_timestamp(start_iso):
                    self.logger.info(f"Reached start time at page {page + 1}, stopping pagination")
                    break
                    
                self.logger.info(f"Page {page + 1}: Got {len(batch)} trades, total={len(all_trades)}, next_till={till}")
                
                # Update stats for successful call
                if self.current_client_token_index is not None:
                    self.token_stats[self.current_client_token_index]['calls_today'] += 1
                
                # Successfully processed this page, move to next
                page += 1
                    
            except Exception as e:
                error_str = str(e)
                self.logger.error(f"Error on page {page + 1}: {error_str}")
                
                # Handle token rotation for 402 and 403 errors
                if '402' in error_str or 'Payment Required' in error_str or '403' in error_str or 'Forbidden' in error_str:
                    if self.current_client_token_index is not None:
                        if '402' in error_str or 'Payment Required' in error_str:
                            self.token_stats[self.current_client_token_index]['payment_required'] = True
                            self.logger.warning(f"Token #{self.current_client_token_index} marked as payment required")
                        elif '403' in error_str or 'Forbidden' in error_str:
                            self.token_stats[self.current_client_token_index]['forbidden'] = True
                            self.logger.warning(f"Token #{self.current_client_token_index} marked as forbidden/invalid")
                    
                    self.current_token_index = (self.current_token_index + 1) % len(self.api_tokens)
                    
                    try:
                        await self.initialize()
                        self.logger.info("Retrying same page with new token...")
                        # Don't increment page, will retry same page
                    except Exception as reinit_error:
                        self.logger.error(f"Failed to reinitialize: {reinit_error}")
                        break
                else:
                    # Non-recoverable error, stop pagination
                    break
        
        # Calculate actual coverage
        if all_trades:
            times = [t.get("Block", {}).get("Time") for t in all_trades if t.get("Block", {}).get("Time")]
            if times:
                tnums = [self._parse_iso_timestamp(x) for x in times]
                min_time = datetime.utcfromtimestamp(min(tnums))
                max_time = datetime.utcfromtimestamp(max(tnums))
                coverage_minutes = (max(tnums) - min(tnums)) / 60
                self.logger.info(f"Paginated fetch complete: {len(all_trades)} trades covering {coverage_minutes:.1f} minutes ({min_time}Z to {max_time}Z)")
            else:
                self.logger.warning("No valid timestamps in paginated results")
        else:
            self.logger.warning("Paginated fetch returned no trades")
        
        return all_trades
    
    def _parse_iso_timestamp(self, timestamp_str) -> float:
        """Parse ISO timestamp to unix timestamp"""
        try:
            from datetime import datetime
            if isinstance(timestamp_str, (int, float)):
                return float(timestamp_str)
            
            if isinstance(timestamp_str, str):
                if 'T' in timestamp_str:
                    # ISO format
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    return dt.timestamp()
                else:
                    # Try to parse as number string
                    return float(timestamp_str)
        except:
            pass
        import time
        return time.time()
    
    def get_token_status(self) -> Dict:
        """Get status of all API tokens for debugging"""
        status = {
            'total_tokens': len(self.api_tokens),
            'current_index': self.current_token_index,
            'tokens': []
        }
        
        for i, token_info in self.token_stats.items():
            token_status = {
                'index': i,
                'calls_today': token_info['calls_today'],
                'rate_limited': token_info['rate_limited'],
                'payment_required': token_info['payment_required'],
                'status': 'OK'
            }
            
            if token_info['payment_required']:
                token_status['status'] = 'PAYMENT_REQUIRED'
            elif token_info['rate_limited']:
                if time.time() < token_info['reset_time']:
                    token_status['status'] = 'RATE_LIMITED'
                    token_status['reset_in'] = int(token_info['reset_time'] - time.time())
                else:
                    token_status['status'] = 'RECOVERED'
            
            status['tokens'].append(token_status)
        
        # Overall status
        available_tokens = sum(1 for t in status['tokens'] if t['status'] in ['OK', 'RECOVERED'])
        status['available_tokens'] = available_tokens
        status['all_exhausted'] = available_tokens == 0
        
        return status

    async def get_token_transactions(self, mint_address: str, limit: int = 100) -> List[Dict]:
        """Get transactions for a specific token"""
        if not self.client:
            await self.initialize()
            
        query = gql("""
            query($mint: String!, $limit: Int!) {
              Solana {
                Transfers(
                  limit: $limit
                  orderBy: {descending: Block_Time}
                  where: {
                    Transfer: {
                      Currency: {
                        MintAddress: {is: $mint}
                      }
                    }
                  }
                ) {
                  Block {
                    Time
                  }
                  Transaction {
                    Signer
                    Signature
                  }
                  Transfer {
                    Amount
                    Currency {
                      MintAddress
                      Symbol
                      Name
                    }
                    Receiver {
                      Address
                    }
                    Sender {
                      Address
                    }
                  }
                }
              }
            }
        """)
        
        try:
            result = await self.client.execute_async(
                query, 
                variable_values={"mint": mint_address, "limit": limit}
            )
            
            # Update stats for successful call
            if self.current_client_token_index is not None:
                self.token_stats[self.current_client_token_index]['calls_today'] += 1
            
            return result['Solana']['Transfers']
            
        except Exception as e:
            error_str = str(e)
            self.logger.error(f"Error fetching token transactions: {error_str}")
            
            # Handle 402 and rate limiting same as above
            if '402' in error_str or 'Payment Required' in error_str:
                if self.current_client_token_index is not None:
                    self.token_stats[self.current_client_token_index]['payment_required'] = True
                    self.logger.warning(f"Token #{self.current_client_token_index} marked as payment required")
                
                self.current_token_index = (self.current_token_index + 1) % len(self.api_tokens)
                
                try:
                    await self.initialize()
                    return await self.get_token_transactions(mint_address, limit)
                except Exception as reinit_error:
                    self.logger.error(f"Failed to reinitialize with new token: {reinit_error}")
                    return []
            
            return []
    
    async def get_wallet_token_trades(self, wallet_address: str, token_address: str, limit: int = 10) -> List[Dict]:
        """Get trades for a specific wallet on a specific token"""
        async with self._concurrent_semaphore:
            return await self._get_wallet_token_trades_impl(wallet_address, token_address, limit)
    
    async def _get_wallet_token_trades_impl(self, wallet_address: str, token_address: str, limit: int = 10) -> List[Dict]:
        """Implementation of get_wallet_token_trades with corrected query structure"""
        query = gql("""
            query($wallet: String!, $mint: String!, $limit: Int!) {
              Solana {
                DEXTrades(
                  limit: { count: $limit }
                  orderBy: {descending: Block_Time}
                  where: {
                    Trade: {
                      Dex: { ProtocolName: { is: "pump" } }
                      Currency: { MintAddress: { is: $mint } }
                      Or: [
                        {
                          Buy: {
                            Account: { Address: { is: $wallet } }
                          }
                        }
                        {
                          Sell: {
                            Account: { Address: { is: $wallet } }
                          }
                        }
                      ]
                    }
                    Transaction: { Result: { Success: true } }
                  }
                ) {
                  Block {
                    Time
                  }
                  Transaction {
                    Signature
                    Signer
                  }
                  Trade {
                    Buy {
                      Amount
                      AmountInUSD
                      Price
                      PriceInUSD
                      Currency {
                        MintAddress
                        Symbol
                        Name
                      }
                      Account {
                        Address
                      }
                    }
                    Sell {
                      Amount
                      AmountInUSD
                      Price
                      PriceInUSD
                      Currency {
                        MintAddress
                        Symbol
                        Name
                      }
                      Account {
                        Address
                      }
                    }
                    Dex {
                      ProtocolFamily
                      ProtocolName
                    }
                  }
                }
              }
            }
        """)
        
        try:
            result = await self.client.execute_async(
                query, 
                variable_values={
                    "wallet": wallet_address, 
                    "mint": token_address, 
                    "limit": limit
                }
            )
            
            if self.current_client_token_index is not None:
                self.token_stats[self.current_client_token_index]['calls_today'] += 1
            
            trades = result['Solana']['DEXTrades']
            parsed_trades = []
            
            for trade in trades:
                wallet_bought = trade['Trade']['Buy']['Account']['Address'] == wallet_address
                
                parsed_trade = {
                    'timestamp': self._parse_iso_timestamp(trade['Block']['Time']),
                    'side': 'buy' if wallet_bought else 'sell',
                    'price': float(trade['Trade']['Buy']['Price']) if wallet_bought else float(trade['Trade']['Sell']['Price']),
                    'amount': float(trade['Trade']['Buy']['Amount']) if wallet_bought else float(trade['Trade']['Sell']['Amount']),
                    'token_address': token_address,
                    'wallet': wallet_address,
                    'tx_hash': trade['Transaction']['Signature']
                }
                parsed_trades.append(parsed_trade)
            
            return parsed_trades
            
        except Exception as e:
            self.logger.error(f"Error fetching wallet token trades: {e}")
            return []
    
    async def get_token_trades_in_window(self, token_address: str, start_time: float, end_time: float) -> List[Dict]:
        """Get all trades for a token within a time window"""
        # Convert timestamps to ISO format (following existing pattern)
        from datetime import datetime, timezone
        start_dt = datetime.fromtimestamp(start_time, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_time, tz=timezone.utc)
        start_iso = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_iso = end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        query = gql(f"""
            query {{
              Solana {{
                DEXTrades(
                  limit: {{ count: 100 }}
                  orderBy: {{ ascending: Block_Time }}
                  where: {{
                    Trade: {{
                      Dex: {{ ProtocolName: {{ is: "pump" }} }}
                      Buy: {{
                        Currency: {{
                          MintAddress: {{is: "{token_address}"}}
                        }}
                      }}
                    }}
                    Transaction: {{ Result: {{ Success: true }} }}
                    Block: {{
                      Time: {{ since: "{start_iso}", till: "{end_iso}" }}
                    }}
                  }}
                ) {{
                  Block {{ Time }}
                  Transaction {{
                    Signature
                    Signer
                  }}
                  Trade {{
                    Buy {{
                      Amount
                      Currency {{
                        MintAddress
                        Symbol
                        Name
                      }}
                      Price
                      Account {{
                        Address
                      }}
                    }}
                    Sell {{
                      Amount
                      Currency {{
                        MintAddress
                        Symbol
                        Name
                      }}
                      Price
                      Account {{
                        Address
                      }}
                    }}
                    Dex {{
                      ProtocolFamily
                      ProtocolName
                    }}
                  }}
                }}
              }}
            }}
        """)
        
        try:
            result = await self.client.execute_async(query)
            
            if self.current_client_token_index is not None:
                self.token_stats[self.current_client_token_index]['calls_today'] += 1
            
            trades = result['Solana']['DEXTrades']
            parsed_trades = []
            
            for trade in trades:
                parsed_trade = {
                    'timestamp': self._parse_iso_timestamp(trade['Block']['Time']),
                    'side': 'buy',  # This query filters for buys
                    'price': float(trade['Trade']['Buy']['Price']),
                    'amount': float(trade['Trade']['Buy']['Amount']),
                    'token_address': token_address,
                    'wallet': trade['Trade']['Buy']['Account']['Address'],
                    'tx_hash': trade['Transaction']['Signature']
                }
                parsed_trades.append(parsed_trade)
            
            return parsed_trades
            
        except Exception as e:
            self.logger.error(f"Error fetching token trades in window: {e}")
            return []
    
    async def get_wallet_trades(self, wallet_address: str, limit: int = 50) -> List[Dict]:
        """Get all recent trades for a specific wallet across all tokens"""
        query = gql("""
            query($wallet: String!, $limit: Int!) {
              Solana {
                DEXTrades(
                  limit: { count: $limit }
                  orderBy: {descending: Block_Time}
                  where: {
                    Trade: {
                      Dex: { ProtocolName: { is: "pump" } }
                      Or: [
                        {
                          Buy: {
                            Account: {
                              Address: {is: $wallet}
                            }
                          }
                        }
                        {
                          Sell: {
                            Account: {
                              Address: {is: $wallet}
                            }
                          }
                        }
                      ]
                    }
                    Transaction: { Result: { Success: true } }
                  }
                ) {
                  Block {
                    Time
                  }
                  Transaction {
                    Signature
                    Signer
                  }
                  Trade {
                    Buy {
                      Amount
                      AmountInUSD
                      Price
                      PriceInUSD
                      Currency {
                        MintAddress
                        Symbol
                        Name
                      }
                      Account {
                        Address
                      }
                    }
                    Sell {
                      Amount
                      AmountInUSD
                      Price
                      PriceInUSD
                      Currency {
                        MintAddress
                        Symbol
                        Name
                      }
                      Account {
                        Address
                      }
                    }
                    Dex {
                      ProtocolFamily
                      ProtocolName
                    }
                  }
                }
              }
            }
        """)
        
        try:
            result = await self.client.execute_async(
                query, 
                variable_values={
                    "wallet": wallet_address, 
                    "limit": limit
                }
            )
            
            if self.current_client_token_index is not None:
                self.token_stats[self.current_client_token_index]['calls_today'] += 1
            
            trades = result['Solana']['DEXTrades']
            parsed_trades = []
            
            for trade in trades:
                wallet_bought = trade['Trade']['Buy']['Account']['Address'] == wallet_address
                
                parsed_trade = {
                    'timestamp': self._parse_iso_timestamp(trade['Block']['Time']),
                    'side': 'buy' if wallet_bought else 'sell',
                    'token_address': trade['Trade']['Buy']['Currency']['MintAddress'] if wallet_bought else trade['Trade']['Sell']['Currency']['MintAddress'],
                    'price': float(trade['Trade']['Buy']['Price']) if wallet_bought else float(trade['Trade']['Sell']['Price']),
                    'amount': float(trade['Trade']['Buy']['Amount']) if wallet_bought else float(trade['Trade']['Sell']['Amount']),
                    'wallet': wallet_address,
                    'tx_hash': trade['Transaction']['Signature']
                }
                parsed_trades.append(parsed_trade)
            
            return parsed_trades
            
        except Exception as e:
            self.logger.error(f"Error fetching wallet trades: {e}")
            return []

    def _timestamp_to_iso(self, timestamp: float) -> str:
        """Convert Unix timestamp to ISO format for BitQuery"""
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
