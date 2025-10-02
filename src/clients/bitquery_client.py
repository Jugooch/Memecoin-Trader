"""
Bitquery GraphQL client for monitoring Pump.fun token launches
"""

import asyncio
import aiohttp
import json
import logging
import time
from typing import Dict, List, AsyncGenerator, Union, Optional, Tuple
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
        
        # Reduce GQL library verbosity to avoid log spam
        logging.getLogger('gql').setLevel(logging.WARNING)
        logging.getLogger('gql.transport').setLevel(logging.WARNING)
        logging.getLogger('gql.client').setLevel(logging.WARNING)
        
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
                'payment_required': False,  # Track 402 errors
                'forbidden': False  # Track 403 errors
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
            
            # Check if this token is forbidden (403)
            if token_info.get('forbidden', False):
                self.logger.debug(f"Token {self.current_token_index} is forbidden, trying next...")
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
        """Implementation of get_wallet_token_trades with fresh transport per request"""
        # Create a fresh transport for this specific request to avoid "Transport is already connected"
        token_index, api_token = self._get_next_available_token()
        
        if not api_token:
            self.logger.error("No available Bitquery API tokens for wallet trades")
            return []
        
        from gql.transport.aiohttp import AIOHTTPTransport
        fresh_transport = AIOHTTPTransport(
            url=self.endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_token}"
            }
        )
        
        fresh_client = Client(transport=fresh_transport)
        
        # We need two separate queries since GraphQL doesn't support OR conditions at this level
        buy_query = gql("""
            query($wallet: String!, $mint: String!, $limit: Int!) {
              Solana {
                DEXTrades(
                  limit: { count: $limit }
                  orderBy: {descending: Block_Time}
                  where: {
                    Trade: {
                      Dex: { ProtocolName: { is: "pump" } }
                      Buy: { 
                        Currency: { MintAddress: { is: $mint } }
                        Account: { Address: { is: $wallet } }
                      }
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
        
        sell_query = gql("""
            query($wallet: String!, $mint: String!, $limit: Int!) {
              Solana {
                DEXTrades(
                  limit: { count: $limit }
                  orderBy: {descending: Block_Time}
                  where: {
                    Trade: {
                      Dex: { ProtocolName: { is: "pump" } }
                      Sell: { 
                        Currency: { MintAddress: { is: $mint } }
                        Account: { Address: { is: $wallet } }
                      }
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
            # Execute both queries
            buy_result = await fresh_client.execute_async(
                buy_query, 
                variable_values={
                    "wallet": wallet_address, 
                    "mint": token_address, 
                    "limit": limit // 2  # Split limit between buy and sell queries
                }
            )
            
            sell_result = await fresh_client.execute_async(
                sell_query, 
                variable_values={
                    "wallet": wallet_address, 
                    "mint": token_address, 
                    "limit": limit // 2
                }
            )
            
            # Update stats for the token we used (count as 2 calls since we made 2 requests)
            if token_index is not None:
                self.token_stats[token_index]['calls_today'] += 2
            
            # Combine trades from both queries
            all_trades = []
            
            # Process buy trades
            buy_trades = buy_result['Solana']['DEXTrades']
            for trade in buy_trades:
                parsed_trade = {
                    'timestamp': self._parse_iso_timestamp(trade['Block']['Time']),
                    'side': 'buy',
                    'price': float(trade['Trade']['Buy']['Price']) if trade['Trade']['Buy']['Price'] else 0,
                    'amount': float(trade['Trade']['Buy']['Amount']) if trade['Trade']['Buy']['Amount'] else 0,
                    'token_address': token_address,
                    'wallet': wallet_address,
                    'tx_hash': trade['Transaction']['Signature']
                }
                all_trades.append(parsed_trade)
            
            # Process sell trades
            sell_trades = sell_result['Solana']['DEXTrades']
            for trade in sell_trades:
                parsed_trade = {
                    'timestamp': self._parse_iso_timestamp(trade['Block']['Time']),
                    'side': 'sell',
                    'price': float(trade['Trade']['Sell']['Price']) if trade['Trade']['Sell']['Price'] else 0,
                    'amount': float(trade['Trade']['Sell']['Amount']) if trade['Trade']['Sell']['Amount'] else 0,
                    'token_address': token_address,
                    'wallet': wallet_address,
                    'tx_hash': trade['Transaction']['Signature']
                }
                all_trades.append(parsed_trade)
            
            # Sort by timestamp (newest first)
            all_trades.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Limit to requested number of trades
            return all_trades[:limit]
            
        except Exception as e:
            error_str = str(e)
            self.logger.error(f"Error fetching wallet token trades: {error_str}")
            
            # Handle 402 and 403 errors and token rotation same as other methods
            if '402' in error_str or 'Payment Required' in error_str or '403' in error_str or 'Forbidden' in error_str:
                if token_index is not None:
                    if '402' in error_str or 'Payment Required' in error_str:
                        self.token_stats[token_index]['payment_required'] = True
                        self.logger.warning(f"Token #{token_index} marked as payment required")
                    elif '403' in error_str or 'Forbidden' in error_str:
                        self.token_stats[token_index]['forbidden'] = True
                        self.logger.warning(f"Token #{token_index} marked as forbidden/invalid")
                
                # Try to rotate to next token
                self.current_token_index = (self.current_token_index + 1) % len(self.api_tokens)
                
                # Close current transport before retrying
                try:
                    await fresh_transport.close()
                except:
                    pass
                
                # Reinitialize with new token if available
                try:
                    await self.initialize()
                    self.logger.info("Retrying wallet token trades query with new token...")
                    # Retry the query with new token
                    return await self.get_wallet_token_trades(wallet_address, token_address, limit)
                except Exception as reinit_error:
                    self.logger.error(f"Failed to reinitialize with new token: {reinit_error}")
                    return []
            
            return []
        finally:
            # Always close the transport to prevent connection leaks
            try:
                await fresh_transport.close()
            except Exception as cleanup_error:
                self.logger.debug(f"Error closing transport: {cleanup_error}")
    
    async def get_token_trades_in_window(self, token_address: str, start_time: float, end_time: float) -> List[Dict]:
        """Get all trades for a token within a time window"""
        # Convert timestamps to ISO format (following existing pattern)
        from datetime import datetime, timezone
        start_dt = datetime.fromtimestamp(start_time, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_time, tz=timezone.utc)
        start_iso = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_iso = end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Use the working pattern from existing methods - combine buy and sell in one query
        # Following the exact structure from get_recent_trades_paginated()
        query = gql(f"""
            query {{
              Solana {{
                DEXTrades(
                  limit: {{ count: 100 }}
                  orderBy: {{ ascending: Block_Time }}
                  where: {{
                    Trade: {{
                      Dex: {{ ProtocolName: {{ is: "pump" }} }}
                      Buy: {{ Currency: {{ MintAddress: {{ is: "{token_address}" }} }} }}
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
            
            # Parse trades using the same logic as existing methods
            for trade in trades:
                # Determine if this is a buy or sell based on which side has data
                buy_data = trade['Trade']['Buy']
                sell_data = trade['Trade']['Sell']
                
                # Check which side has the token we're looking for
                is_buy = (buy_data.get('Currency', {}).get('MintAddress') == token_address and 
                         buy_data.get('Amount') and buy_data.get('Price'))
                
                if is_buy:
                    parsed_trade = {
                        'timestamp': self._parse_iso_timestamp(trade['Block']['Time']),
                        'side': 'buy',
                        'price': float(buy_data['Price']) if buy_data.get('Price') else 0,
                        'amount': float(buy_data['Amount']) if buy_data.get('Amount') else 0,
                        'token_address': token_address,
                        'wallet': buy_data['Account']['Address'],
                        'tx_hash': trade['Transaction']['Signature']
                    }
                    parsed_trades.append(parsed_trade)
                elif sell_data.get('Currency', {}).get('MintAddress') == token_address:
                    parsed_trade = {
                        'timestamp': self._parse_iso_timestamp(trade['Block']['Time']),
                        'side': 'sell',
                        'price': float(sell_data['Price']) if sell_data.get('Price') else 0,
                        'amount': float(sell_data['Amount']) if sell_data.get('Amount') else 0,
                        'token_address': token_address,
                        'wallet': sell_data['Account']['Address'],
                        'tx_hash': trade['Transaction']['Signature']
                    }
                    parsed_trades.append(parsed_trade)
            
            return parsed_trades
            
        except Exception as e:
            self.logger.error(f"Error fetching token trades in window: {e}")
            return []
    
    async def get_wallet_trades(self, wallet_address: str, limit: int = 50) -> List[Dict]:
        """Get all recent trades for a specific wallet across all tokens"""
        async with self._concurrent_semaphore:
            return await self._get_wallet_trades_impl(wallet_address, limit)
    
    async def _get_wallet_trades_impl(self, wallet_address: str, limit: int = 50) -> List[Dict]:
        """Implementation of get_wallet_trades with fresh transport per request"""
        # Create fresh transport to avoid connection issues
        token_index, api_token = self._get_next_available_token()
        
        if not api_token:
            self.logger.error("No available Bitquery API tokens for wallet trades")
            return []
        
        from gql.transport.aiohttp import AIOHTTPTransport
        fresh_transport = AIOHTTPTransport(
            url=self.endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_token}"
            }
        )
        
        fresh_client = Client(transport=fresh_transport)
        
        # Split into two separate queries for buy and sell trades 
        buy_query = gql("""
            query($wallet: String!, $limit: Int!) {
              Solana {
                DEXTrades(
                  limit: { count: $limit }
                  orderBy: {descending: Block_Time}
                  where: {
                    Trade: {
                      Dex: { ProtocolName: { is: "pump" } }
                      Buy: { Account: { Address: { is: $wallet } } }
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
        
        sell_query = gql("""
            query($wallet: String!, $limit: Int!) {
              Solana {
                DEXTrades(
                  limit: { count: $limit }
                  orderBy: {descending: Block_Time}
                  where: {
                    Trade: {
                      Dex: { ProtocolName: { is: "pump" } }
                      Sell: { Account: { Address: { is: $wallet } } }
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
            # Execute both queries
            buy_result = await fresh_client.execute_async(
                buy_query, 
                variable_values={
                    "wallet": wallet_address, 
                    "limit": limit // 2
                }
            )
            
            sell_result = await fresh_client.execute_async(
                sell_query, 
                variable_values={
                    "wallet": wallet_address, 
                    "limit": limit // 2
                }
            )
            
            # Update stats (count as 2 calls)
            if token_index is not None:
                self.token_stats[token_index]['calls_today'] += 2
            
            # Combine and parse trades
            all_trades = []
            
            # Process buy trades
            buy_trades = buy_result['Solana']['DEXTrades']
            for trade in buy_trades:
                parsed_trade = {
                    'timestamp': self._parse_iso_timestamp(trade['Block']['Time']),
                    'side': 'buy',
                    'token_address': trade['Trade']['Buy']['Currency']['MintAddress'],
                    'price': float(trade['Trade']['Buy']['Price']) if trade['Trade']['Buy']['Price'] else 0,
                    'amount': float(trade['Trade']['Buy']['Amount']) if trade['Trade']['Buy']['Amount'] else 0,
                    'wallet': wallet_address,
                    'tx_hash': trade['Transaction']['Signature']
                }
                all_trades.append(parsed_trade)
            
            # Process sell trades
            sell_trades = sell_result['Solana']['DEXTrades']
            for trade in sell_trades:
                parsed_trade = {
                    'timestamp': self._parse_iso_timestamp(trade['Block']['Time']),
                    'side': 'sell',
                    'token_address': trade['Trade']['Sell']['Currency']['MintAddress'],
                    'price': float(trade['Trade']['Sell']['Price']) if trade['Trade']['Sell']['Price'] else 0,
                    'amount': float(trade['Trade']['Sell']['Amount']) if trade['Trade']['Sell']['Amount'] else 0,
                    'wallet': wallet_address,
                    'tx_hash': trade['Transaction']['Signature']
                }
                all_trades.append(parsed_trade)
            
            # Sort by timestamp (newest first) and limit
            all_trades.sort(key=lambda x: x['timestamp'], reverse=True)
            return all_trades[:limit]
            
        except Exception as e:
            error_str = str(e)
            self.logger.error(f"Error fetching wallet trades: {error_str}")
            
            # Handle 402 and 403 errors and token rotation same as other methods
            if '402' in error_str or 'Payment Required' in error_str or '403' in error_str or 'Forbidden' in error_str:
                if token_index is not None:
                    if '402' in error_str or 'Payment Required' in error_str:
                        self.token_stats[token_index]['payment_required'] = True
                        self.logger.warning(f"Token #{token_index} marked as payment required")
                    elif '403' in error_str or 'Forbidden' in error_str:
                        self.token_stats[token_index]['forbidden'] = True
                        self.logger.warning(f"Token #{token_index} marked as forbidden/invalid")
                
                # Try to rotate to next token
                self.current_token_index = (self.current_token_index + 1) % len(self.api_tokens)
                
                # Close current transport before retrying
                try:
                    await fresh_transport.close()
                except:
                    pass
                
                # Reinitialize with new token if available
                try:
                    await self.initialize()
                    self.logger.info("Retrying wallet trades query with new token...")
                    # Retry the query with new token
                    return await self.get_wallet_trades(wallet_address, limit)
                except Exception as reinit_error:
                    self.logger.error(f"Failed to reinitialize with new token: {reinit_error}")
                    return []
            
            return []
        finally:
            # Always close the transport
            try:
                await fresh_transport.close()
            except Exception as cleanup_error:
                self.logger.debug(f"Error closing transport: {cleanup_error}")

    def _timestamp_to_iso(self, timestamp: float) -> str:
        """Convert Unix timestamp to ISO format for BitQuery"""
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    async def get_tokens_created_by_dev(self, dev_wallet: str, lookback_days: int = 30) -> List[Dict]:
        """Find tokens created/deployed by this dev wallet"""
        # Calculate time range
        from datetime import datetime, timedelta, timezone
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - timedelta(days=lookback_days)
        start_iso = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_iso = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Get token with concurrent semaphore
        async with self._concurrent_semaphore:
            return await self._get_tokens_created_by_dev_impl(dev_wallet, start_iso, end_iso)
    
    async def _get_tokens_created_by_dev_impl(self, dev_wallet: str, start_iso: str, end_iso: str) -> List[Dict]:
        """Implementation of get_tokens_created_by_dev with fresh transport"""
        token_index, api_token = self._get_next_available_token()
        
        if not api_token:
            self.logger.error("No available Bitquery API tokens for dev token creation")
            return []
        
        from gql.transport.aiohttp import AIOHTTPTransport
        fresh_transport = AIOHTTPTransport(
            url=self.endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_token}"
            }
        )
        
        fresh_client = Client(transport=fresh_transport)
        
        # Query for DEX trades where this wallet was the transaction signer (token creator)
        # We look at the first buy trade for each token, which happens when token is created
        query = gql("""
            query($dev_wallet: String!, $start: DateTime!, $end: DateTime!, $limit: Int!) {
              Solana {
                DEXTrades(
                  limit: { count: $limit }
                  orderBy: { descending: Block_Time }
                  where: {
                    Trade: { Dex: { ProtocolName: { is: "pump" } } }
                    Transaction: { 
                      Result: { Success: true }
                      Signer: { is: $dev_wallet }
                    }
                    Block: { 
                      Time: { since: $start, till: $end }
                    }
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
                      Currency {
                        MintAddress
                        Symbol
                        Name
                      }
                      Amount
                      AmountInUSD
                    }
                    Sell {
                      Currency {
                        MintAddress
                        Symbol
                        Name
                      }
                      Amount
                      AmountInUSD
                    }
                  }
                }
              }
            }
        """)
        
        try:
            result = await fresh_client.execute_async(
                query, 
                variable_values={
                    "dev_wallet": dev_wallet,
                    "start": start_iso,
                    "end": end_iso,
                    "limit": 100
                }
            )
            
            trades = result.get("Solana", {}).get("DEXTrades", [])
            
            # Group by token to get unique tokens created by this dev
            tokens_created = {}
            
            for trade in trades:
                trade_data = trade.get("Trade", {})
                buy_currency = trade_data.get("Buy", {}).get("Currency", {})
                sell_currency = trade_data.get("Sell", {}).get("Currency", {})
                
                # Find the token mint (not SOL)
                mint_address = None
                token_symbol = None
                token_name = None
                
                # SOL addresses to exclude
                sol_addresses = [None, 'So11111111111111111111111111111112', '11111111111111111111111111111111']
                
                if buy_currency.get('MintAddress') not in sol_addresses:
                    mint_address = buy_currency.get('MintAddress')
                    token_symbol = buy_currency.get('Symbol', '')
                    token_name = buy_currency.get('Name', '')
                elif sell_currency.get('MintAddress') not in sol_addresses:
                    mint_address = sell_currency.get('MintAddress')
                    token_symbol = sell_currency.get('Symbol', '')
                    token_name = sell_currency.get('Name', '')
                
                if mint_address:
                    # Get USD amounts from both sides of the trade
                    buy_usd = float(trade_data.get("Buy", {}).get("AmountInUSD", 0) or 0)
                    sell_usd = float(trade_data.get("Sell", {}).get("AmountInUSD", 0) or 0)
                    max_usd_this_trade = max(buy_usd, sell_usd)
                    
                    if mint_address not in tokens_created:
                        tokens_created[mint_address] = {
                            'mint_address': mint_address,
                            'symbol': token_symbol,
                            'name': token_name,
                            'created_at': trade.get("Block", {}).get("Time"),
                            'signature': trade.get("Transaction", {}).get("Signature"),
                            'dev_wallet': dev_wallet,
                            'max_usd_amount': max_usd_this_trade
                        }
                    else:
                        # Update max USD if this trade is larger
                        tokens_created[mint_address]['max_usd_amount'] = max(
                            tokens_created[mint_address].get('max_usd_amount', 0),
                            max_usd_this_trade
                        )
            
            tokens_list = list(tokens_created.values())
            self.logger.debug(f"Found {len(tokens_list)} tokens created by {dev_wallet[:8]}...")
            
            return tokens_list
            
        except Exception as e:
            error_str = str(e)
            self.logger.error(f"Error fetching tokens created by dev {dev_wallet[:8]}...: {e}")
            
            # Handle 402 and 403 errors and token rotation same as other methods
            if '402' in error_str or 'Payment Required' in error_str or '403' in error_str or 'Forbidden' in error_str:
                if token_index is not None:
                    if '402' in error_str or 'Payment Required' in error_str:
                        self.token_stats[token_index]['payment_required'] = True
                    elif '403' in error_str or 'Forbidden' in error_str:
                        self.token_stats[token_index]['forbidden'] = True
                    
                    self.logger.warning(f"Token #{token_index} marked as {'payment required' if '402' in error_str else 'forbidden/invalid'}")
                    
                    # Try next token
                    next_token_index, next_api_token = self._get_next_available_token()
                    if next_api_token:
                        self.logger.info(f"Retrying dev token creation query with token #{next_token_index}...")
                        try:
                            return await self._get_tokens_created_by_dev_impl(dev_wallet, start_iso, end_iso)
                        except Exception as retry_error:
                            self.logger.error(f"Retry also failed: {retry_error}")
            
            return []
        
        finally:
            await fresh_transport.close()
    
    async def get_dev_token_history(self, dev_wallet: str, lookback_days: int = 30) -> Dict:
        """Get tokens created by this developer wallet"""
        try:
            # Get tokens actually created by this dev wallet
            tokens_created = await self.get_tokens_created_by_dev(dev_wallet, lookback_days)
            
            if not tokens_created:
                return {
                    'total_tokens': 0,
                    'best_peak_mc': 0,
                    'recent_launches': 0,
                    'token_addresses': []
                }
            
            # Count recent launches (last 7 days)
            from datetime import datetime, timedelta, timezone
            seven_days_ago = datetime.now(tz=timezone.utc) - timedelta(days=7)
            recent_launches = 0
            
            for token_data in tokens_created:
                created_at = token_data.get('created_at')
                if isinstance(created_at, str):
                    from dateutil import parser
                    creation_time = parser.parse(created_at)
                    if creation_time >= seven_days_ago:
                        recent_launches += 1
            
            token_addresses = [t.get('mint_address') for t in tokens_created if t.get('mint_address')]
            
            # Calculate max transaction USD from the tokens data
            max_transaction_usd = 0
            for token_data in tokens_created:
                if 'max_usd_amount' in token_data:
                    max_transaction_usd = max(max_transaction_usd, token_data['max_usd_amount'])
            
            return {
                'total_tokens': len(tokens_created),
                'best_peak_mc': 0,  # Will be updated by Moralis with real MC data
                'recent_launches': recent_launches,
                'token_addresses': token_addresses,
                'max_transaction_usd': max_transaction_usd
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dev token history for {dev_wallet[:8]}...: {e}")
            return {
                'total_tokens': 0,
                'best_peak_mc': 0,
                'recent_launches': 0,
                'token_addresses': []
            }
    
    async def get_batch_dev_profiles(self, dev_wallets: List[str], lookback_days: int = 60) -> Dict[str, Dict]:
        """
        Get multiple dev profiles in a single efficient query.
        This dramatically reduces API usage compared to individual calls.
        
        Args:
            dev_wallets: List of developer wallet addresses
            lookback_days: How many days to look back
        
        Returns:
            Dict mapping dev_wallet -> profile data
        """
        if not dev_wallets:
            return {}
        
        # Get fresh token
        token_index, api_token = self._get_next_available_token()
        if not api_token:
            self.logger.error("No available Bitquery API tokens for batch dev profiles")
            return {}
        
        async with self._concurrent_semaphore:
            # Build WHERE clause for multiple wallets - correct GraphQL syntax
            wallet_list = ', '.join([f'"{wallet}"' for wallet in dev_wallets])
            
            from datetime import datetime, timedelta, timezone
            end_time = datetime.now(tz=timezone.utc)
            start_time = end_time - timedelta(days=lookback_days)
            start_iso = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_iso = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Single query to get all devs' token launches with prices
            query = gql(f"""
                query {{
                  Solana {{
                    DEXTrades(
                      limit: {{ count: 2000 }}
                      orderBy: {{ descending: Block_Time }}
                      where: {{
                        Trade: {{ 
                          Dex: {{ ProtocolName: {{ is: "pump" }} }}
                        }}
                        Transaction: {{ 
                          Result: {{ Success: true }}
                          Signer: {{ in: [{wallet_list}] }}
                        }}
                        Block: {{ Time: {{ since: "{start_iso}", till: "{end_iso}" }} }}
                      }}
                    ) {{
                      Block {{ Time }}
                      Transaction {{ Signer }}
                      Trade {{
                        Buy {{
                          Currency {{ MintAddress Symbol }}
                          PriceInUSD
                        }}
                        Sell {{
                          Currency {{ MintAddress Symbol }}
                          PriceInUSD
                        }}
                      }}
                    }}
                  }}
                }}
            """)
            
            from gql.transport.aiohttp import AIOHTTPTransport
            transport = AIOHTTPTransport(
                url=self.endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_token}"
                }
            )
            
            # Retry logic for 402/403 errors  
            max_retries = len(self.api_tokens)  # Try all available tokens
            retry_count = 0
            
            try:
                while retry_count < max_retries:
                    try:
                        from gql import Client
                        client = Client(transport=transport)
                        result = await client.execute_async(query)
                        
                        # Process results into dev profiles
                        dev_profiles = {wallet: {
                            'dev_wallet': wallet,
                            'num_tokens_launched': 0,
                            'best_peak_mc_usd': 0,
                            'tokens_launched_7d': 0,
                            'token_prices': {},  # token_address -> max_price
                            'updated_at': datetime.now().isoformat()
                        } for wallet in dev_wallets}
                        
                        if result and 'Solana' in result and 'DEXTrades' in result['Solana']:
                            trades = result['Solana']['DEXTrades']
                            
                            # Group by dev and token
                            seven_days_ago = datetime.now(tz=timezone.utc) - timedelta(days=7)
                            seven_days_ago_timestamp = seven_days_ago.timestamp()
                            
                            for trade in trades:
                                dev_wallet = trade['Transaction']['Signer']
                                if dev_wallet not in dev_profiles:
                                    continue
                                    
                                # Get token address and price
                                token_address = None
                                price_usd = 0
                                
                                # Check buy side (token being bought)
                                if trade['Trade']['Buy']['Currency']['MintAddress'] != "So11111111111111111111111111111111111111112":
                                    token_address = trade['Trade']['Buy']['Currency']['MintAddress']
                                    price_usd = float(trade['Trade']['Buy']['PriceInUSD'] or 0)
                                # Check sell side (token being sold)
                                elif trade['Trade']['Sell']['Currency']['MintAddress'] != "So11111111111111111111111111111111111111112":
                                    token_address = trade['Trade']['Sell']['Currency']['MintAddress']
                                    price_usd = float(trade['Trade']['Sell']['PriceInUSD'] or 0)
                                
                                if not token_address or price_usd <= 0:
                                    continue
                                
                                profile = dev_profiles[dev_wallet]
                                
                                # Track max price per token
                                if token_address not in profile['token_prices']:
                                    profile['token_prices'][token_address] = price_usd
                                    profile['num_tokens_launched'] += 1
                                    
                                    # Check if recent launch
                                    trade_time = self._parse_iso_timestamp(trade['Block']['Time'])
                                    if trade_time >= seven_days_ago_timestamp:
                                        profile['tokens_launched_7d'] += 1
                                else:
                                    # Update max price for this token
                                    profile['token_prices'][token_address] = max(
                                        profile['token_prices'][token_address], 
                                        price_usd
                                    )
                        
                        # Calculate best peak MC for each dev
                        for profile in dev_profiles.values():
                            if profile['token_prices']:
                                max_price = max(profile['token_prices'].values())
                                # Pump.fun standard: 1B token supply
                                profile['best_peak_mc_usd'] = max_price * 1_000_000_000
                            
                            # Clean up intermediate data
                            del profile['token_prices']
                        
                        # Success! Track API usage and return
                        if token_index is not None:
                            self.token_stats[token_index]['calls_today'] += 1
                        
                        self.logger.info(f"Batch enriched {len(dev_wallets)} devs with 1 API call (vs {len(dev_wallets) * 11} individual calls)")
                        return dev_profiles
                        
                    except Exception as e:
                        error_str = str(e)
                        
                        # Check if this is a 402/403 error that we should retry
                        if '402' in error_str or 'Payment Required' in error_str or '403' in error_str or 'Forbidden' in error_str:
                            if token_index is not None:
                                if '402' in error_str or 'Payment Required' in error_str:
                                    self.token_stats[token_index]['payment_required'] = True
                                    self.logger.warning(f"Token #{token_index} marked as payment required")
                                elif '403' in error_str or 'Forbidden' in error_str:
                                    self.token_stats[token_index]['forbidden'] = True
                                    self.logger.warning(f"Token #{token_index} marked as forbidden/invalid")
                            
                            # Rotate to next token and retry
                            self.current_token_index = (self.current_token_index + 1) % len(self.api_tokens)
                            retry_count += 1
                            
                            if retry_count < max_retries:
                                # Get next token and create fresh transport for retry
                                token_index, api_token = self._get_next_available_token()
                                if api_token:
                                    self.logger.info(f"Retrying batch query with token #{token_index} (attempt {retry_count + 1}/{max_retries})")
                                    
                                    # Close old transport
                                    try:
                                        await transport.close()
                                    except:
                                        pass
                                    
                                    # Create new transport with fresh token
                                    from gql.transport.aiohttp import AIOHTTPTransport
                                    transport = AIOHTTPTransport(
                                        url=self.endpoint,
                                        headers={
                                            "Content-Type": "application/json",
                                            "Authorization": f"Bearer {api_token}"
                                        }
                                    )
                                    continue
                                else:
                                    self.logger.error("No more available tokens for retry")
                                    break
                        else:
                            # Non-auth error, don't retry
                            self.logger.error(f"Error in batch dev profile query (non-auth): {error_str}")
                            break
                
                # All retries failed
                self.logger.error(f"Batch dev profile query failed after {retry_count} retries")
                return {}
                
            except Exception as e:
                # Catch any unexpected errors at the top level
                self.logger.error(f"Unexpected error in batch dev profile method: {e}")
                return {}
            
            finally:
                # Clean up transport
                if 'transport' in locals():
                    try:
                        await transport.close()
                    except:
                        pass
    
    async def get_token_ath_market_cap(self, mint_address: str, end_date: str = None) -> float:
        """
        Get the all-time high market cap for a token.
        For pump.fun tokens: MC = Price × 1B tokens
        
        Args:
            mint_address: Token mint address
            end_date: Optional end date to get ATH up to that point (ISO format)
            
        Returns:
            ATH market cap in USD
        """
        # Get fresh token to avoid connection issues
        token_index, api_token = self._get_next_available_token()
        
        if not api_token:
            self.logger.error("No available Bitquery API tokens for ATH market cap")
            return 0
        
        # Use concurrency control
        async with self._concurrent_semaphore:
            # Build the where clause
            where_clause = f'Trade: {{ Currency: {{ MintAddress: {{ is: "{mint_address}" }} }} }}'
            if end_date:
                where_clause += f', Block: {{ Time: {{ till: "{end_date}" }} }}'
            
            query = gql(f"""
                query {{
                  Solana {{
                    DEXTradeByTokens(
                      where: {{ {where_clause} }}
                      limit: {{ count: 1 }}
                      orderBy: {{ descending: Trade_Price }}
                    ) {{
                      Trade {{
                        Price
                        PriceInUSD
                      }}
                      Block {{
                        Time
                      }}
                    }}
                  }}
                }}
            """)
            
            # Create fresh transport to avoid connection issues
            from gql.transport.aiohttp import AIOHTTPTransport
            transport = AIOHTTPTransport(
                url=self.endpoint,
                headers={
                    "Content-Type": "application/json", 
                    "Authorization": f"Bearer {api_token}"
                }
            )
            
            try:
                from gql import Client
                client = Client(transport=transport)
                result = await client.execute_async(query)
                
                if result and 'Solana' in result and 'DEXTradeByTokens' in result['Solana']:
                    trades = result['Solana']['DEXTradeByTokens']
                    if trades and len(trades) > 0:
                        ath_price_usd = float(trades[0]['Trade'].get('PriceInUSD', 0))
                        
                        # For pump.fun tokens, MC = price × 1B
                        # This is standard for pump.fun as they all have 1B supply
                        market_cap = ath_price_usd * 1_000_000_000
                        
                        self.logger.debug(f"Token {mint_address[:8]}... ATH price: ${ath_price_usd:.8f}, MC: ${market_cap:,.0f}")
                        return market_cap
                
                return 0
                
            except Exception as e:
                error_str = str(e)
                self.logger.error(f"Error getting ATH market cap for {mint_address}: {error_str}")
                
                # Handle 402 and 403 errors and token rotation same as other methods
                if '402' in error_str or 'Payment Required' in error_str or '403' in error_str or 'Forbidden' in error_str:
                    if token_index is not None:
                        if '402' in error_str or 'Payment Required' in error_str:
                            self.token_stats[token_index]['payment_required'] = True
                            self.logger.warning(f"Token #{token_index} marked as payment required")
                        elif '403' in error_str or 'Forbidden' in error_str:
                            self.token_stats[token_index]['forbidden'] = True
                            self.logger.warning(f"Token #{token_index} marked as forbidden/invalid")
                    
                    # Try to rotate to next token
                    self.current_token_index = (self.current_token_index + 1) % len(self.api_tokens)
                
                return 0
            finally:
                # Clean up transport
                if 'transport' in locals():
                    try:
                        await transport.close()
                    except:
                        pass
    
    async def get_dev_tokens_with_ath_mc(self, dev_wallet: str, lookback_days: int = 60) -> Dict:
        """
        Get developer's token history with accurate ATH market caps.

        Args:
            dev_wallet: Developer wallet address
            lookback_days: How many days to look back

        Returns:
            Dict with token history including best peak MC
        """
        # First get the basic history
        history = await self.get_dev_token_history(dev_wallet, lookback_days)

        if not history or not history.get('token_addresses'):
            return history

        # Get ATH MC for tokens in parallel (respects semaphore limit)
        token_addresses = history['token_addresses'][:10]  # Limit to 10 most recent

        async def get_token_ath_safe(token_address):
            try:
                ath_mc = await self.get_token_ath_market_cap(token_address)
                if ath_mc > 0:
                    self.logger.debug(f"Token {token_address[:8]}... ATH MC: ${ath_mc:,.0f}")
                    return ath_mc
            except Exception as e:
                self.logger.error(f"Error getting ATH MC for {token_address}: {e}")
            return 0

        # Execute all ATH fetches concurrently (semaphore will limit concurrency)
        ath_results = await asyncio.gather(*[get_token_ath_safe(addr) for addr in token_addresses])
        token_ath_mcs = [mc for mc in ath_results if mc > 0]

        # Update history with real peak MC data
        if token_ath_mcs:
            history['best_peak_mc'] = max(token_ath_mcs)
            history['median_peak_mc'] = sorted(token_ath_mcs)[len(token_ath_mcs)//2]
            history['tokens_with_mc_data'] = len(token_ath_mcs)
            self.logger.debug(f"Dev {dev_wallet[:8]}... Best ATH MC: ${history['best_peak_mc']:,.0f} from {len(token_ath_mcs)} tokens")

        return history

    async def get_all_tokens_created_by_wallet(self, creator_wallet: str, limit: int = 100, _retry_count: int = 0) -> List[str]:
        """
        Get ALL tokens created by a wallet using the Instructions API.
        This finds tokens the wallet created, not just tokens they hold/held.

        Args:
            creator_wallet: The wallet address that created tokens
            limit: Maximum number of tokens to return (default 100)
            _retry_count: Internal retry counter (do not use)

        Returns:
            List of token mint addresses created by this wallet
        """
        # Prevent infinite retries
        if _retry_count >= len(self.api_tokens):
            self.logger.error(f"All {len(self.api_tokens)} tokens exhausted for wallet {creator_wallet[:8]}...")
            return []

        # Get fresh token
        token_index, api_token = self._get_next_available_token()

        if not api_token:
            self.logger.error("No available Bitquery API tokens for token creation query")
            return []

        self.logger.info(f"About to acquire semaphore for {creator_wallet[:8]}... (token #{token_index})")

        # Track if we should retry
        should_retry = False

        async with self._concurrent_semaphore:
            self.logger.info(f"Semaphore acquired for {creator_wallet[:8]}...")
            # Query pump.fun program instructions signed by this wallet
            # Program address: 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P
            query = gql("""
                query ($creator: String!, $limit: Int!) {
                  Solana {
                    Instructions(
                      where: {
                        Instruction: {
                          Program: {
                            Address: {is: "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"}
                          }
                        }
                        Transaction: {
                          Signer: {is: $creator}
                          Result: {Success: true}
                        }
                      }
                      orderBy: {descending: Block_Time}
                      limit: {count: $limit}
                    ) {
                      Block {
                        Time
                      }
                      Transaction {
                        Signature
                        Signer
                      }
                      Instruction {
                        Accounts {
                          Address
                          IsWritable
                        }
                        Data
                      }
                    }
                  }
                }
            """)

            # Try EAP endpoint - the Instructions API might be there
            # BitQuery has multiple endpoints with different schemas
            instructions_endpoint = "https://streaming.bitquery.io/eap"

            # Create fresh transport with correct endpoint
            from gql.transport.aiohttp import AIOHTTPTransport
            transport = AIOHTTPTransport(
                url=instructions_endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_token}"
                }
            )

            try:
                from gql import Client
                client = Client(transport=transport)

                # Add timeout to prevent hanging
                self.logger.info(f"Executing Instructions query for {creator_wallet[:8]}... (15s timeout)")
                try:
                    result = await asyncio.wait_for(
                        client.execute_async(
                            query,
                            variable_values={
                                "creator": creator_wallet,
                                "limit": limit
                            }
                        ),
                        timeout=15.0  # 15 second timeout
                    )
                    self.logger.info(f"Query completed successfully for {creator_wallet[:8]}...")
                except asyncio.TimeoutError:
                    self.logger.error(f"BitQuery Instructions query timed out after 15s for {creator_wallet[:8]}...")
                    raise  # Re-raise to be caught by outer except
                except Exception as query_error:
                    self.logger.error(f"Query execution error for {creator_wallet[:8]}...: {query_error}")
                    raise  # Re-raise to be caught by outer except

                # Update stats
                if token_index is not None:
                    self.token_stats[token_index]['calls_today'] += 1

                # Debug: Log what we got back
                instructions = result.get('Solana', {}).get('Instructions', [])
                self.logger.info(f"BitQuery returned {len(instructions)} instructions for {creator_wallet[:8]}...")

                if len(instructions) == 0:
                    self.logger.warning(f"No instructions found for creator wallet {creator_wallet[:8]}... - wallet may not have created any pump.fun tokens")
                    return []

                # Extract token mint addresses from instructions
                token_mints = set()  # Use set to avoid duplicates

                for idx, instruction in enumerate(instructions):
                    accounts = instruction.get('Instruction', {}).get('Accounts', [])

                    if idx < 3:  # Debug first 3 instructions
                        self.logger.debug(f"Instruction {idx}: {len(accounts)} accounts")

                    # The token mint is typically the first writable account
                    # that's not the program or system account
                    for account in accounts:
                        address = account.get('Address')
                        is_writable = account.get('IsWritable', False)

                        # Filter out known system addresses and only include writable accounts
                        if (address and is_writable and
                            address not in ['11111111111111111111111111111111',
                                          'So11111111111111111111111111111112',
                                          '6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P'] and
                            len(address) >= 32):  # Valid Solana address length
                            token_mints.add(address)
                            self.logger.debug(f"Found token mint: {address[:8]}...")
                            break  # Take the first valid writable account per instruction

                token_list = list(token_mints)
                self.logger.info(f"Found {len(token_list)} tokens created by wallet {creator_wallet[:8]}...")

                return token_list

            except Exception as e:
                error_str = str(e)
                self.logger.error(f"Error fetching tokens created by {creator_wallet[:8]}...: {error_str}")

                # Handle 402 and 403 errors and token rotation
                if '402' in error_str or 'Payment Required' in error_str or '403' in error_str or 'Forbidden' in error_str:
                    if token_index is not None:
                        if '402' in error_str or 'Payment Required' in error_str:
                            self.token_stats[token_index]['payment_required'] = True
                            self.logger.warning(f"Token #{token_index} marked as payment required")
                        elif '403' in error_str or 'Forbidden' in error_str:
                            self.token_stats[token_index]['forbidden'] = True
                            self.logger.warning(f"Token #{token_index} marked as forbidden/invalid")

                    # Try to rotate to next token
                    self.current_token_index = (self.current_token_index + 1) % len(self.api_tokens)

                    # Set flag to retry OUTSIDE semaphore context
                    should_retry = True
                # Don't return here - let the code flow to the retry section below

            finally:
                # Clean up transport
                try:
                    await transport.close()
                except:
                    pass

        # Retry OUTSIDE the semaphore context to avoid deadlock
        if should_retry:
            try:
                await self.initialize()
                self.logger.info(f"Retrying token creation query with new token (attempt {_retry_count + 2}/{len(self.api_tokens)})...")
                return await self.get_all_tokens_created_by_wallet(creator_wallet, limit, _retry_count=_retry_count + 1)
            except Exception as retry_error:
                self.logger.error(f"Retry failed: {retry_error}")
                return []

        return []

