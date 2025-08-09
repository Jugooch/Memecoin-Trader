"""
Bitquery GraphQL client for monitoring Pump.fun token launches
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, AsyncGenerator
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.websockets import WebsocketsTransport


class BitqueryClient:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.logger = logging.getLogger(__name__)
        
        # GraphQL endpoint (using EAP for Solana data)
        self.endpoint = "https://streaming.bitquery.io/eap"
        self.ws_endpoint = "wss://streaming.bitquery.io/eap"
        
        # Pump.fun program address
        self.pumpfun_program = "6EF8rrecthHAuSStzpf6aXr9HWs8jgPVrj5S6fqF6P"
        
        self.client = None
        self.ws_client = None

    async def initialize(self):
        """Initialize GraphQL clients"""
        # HTTP transport for queries (using Bearer token as per Bitquery docs)
        http_transport = AIOHTTPTransport(
            url=self.endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}"
            }
        )
        self.client = Client(transport=http_transport)
        
        # WebSocket transport for subscriptions (using EAP endpoint for Solana)
        ws_url = f"wss://streaming.bitquery.io/eap?token={self.api_token}"
        ws_transport = WebsocketsTransport(
            url=ws_url,
            headers={
                "Sec-WebSocket-Protocol": "graphql-ws",
                "Content-Type": "application/json"
            }
        )
        self.ws_client = Client(transport=ws_transport)

    async def subscribe_token_launches(self) -> AsyncGenerator[Dict, None]:
        """Subscribe to new Pump.fun token creation events"""
        if not self.ws_client:
            await self.initialize()
        
        # GraphQL subscription using correct V2 schema for pump.fun
        subscription = gql("""
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
                    }
                    Sell {
                      Amount
                      Account { Address }
                    }
                  }
                  Transaction { Signature }
                }
              }
            }
        """)
        
        self.logger.info("Starting Bitquery subscription for token launches")
        
        try:
            # Connect to WebSocket first
            await self.ws_client.transport.connect()
            
            # Now subscribe and iterate
            subscription_iterator = self.ws_client.subscribe(subscription)
            async for result in subscription_iterator:
                if 'Solana' in result and 'DEXTrades' in result['Solana']:
                    for trade in result['Solana']['DEXTrades']:
                        token_data = self._parse_dex_trade(trade)
                        if token_data:
                            yield token_data
                            
        except Exception as e:
            self.logger.error(f"Bitquery subscription error: {e}")
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
                      Account {{ Address }}
                      Currency {{
                        MintAddress
                        Symbol
                        Name
                      }}
                    }}
                    Sell {{
                      Amount
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
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error fetching recent launches: {e}")
            return []

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
            
            return result['Solana']['Transfers']
            
        except Exception as e:
            self.logger.error(f"Error fetching token transactions: {e}")
            return []