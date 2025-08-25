"""
Pump.fun API client for trade execution via QuickNode
"""

import aiohttp
import asyncio
import logging
import json
import base64
from typing import Dict, Optional
import time


class PumpFunClient:
    def __init__(self, quicknode_endpoint: str, api_key: str):
        self.quicknode_endpoint = quicknode_endpoint
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 2 requests per second
        
        self.session = None

    async def _get_session(self):
        """Get or create aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={
                    "Content-Type": "application/json"
                    # Note: QuickNode uses API key in URL, not Authorization header
                }
            )
        return self.session

    async def _rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    async def _make_request(self, url: str, data: Dict) -> Dict:
        """Make HTTP request with error handling and retries"""
        session = await self._get_session()
        await self._rate_limit()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.post(url, json=data) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        return json.loads(response_text)
                    elif response.status == 530:
                        self.logger.warning(f"Pump.fun API 530 error (attempt {attempt + 1})")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    elif response.status == 429:
                        self.logger.warning(f"Rate limited (attempt {attempt + 1})")
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    else:
                        self.logger.error(f"API error {response.status}: {response_text}")
                        return {"error": f"HTTP {response.status}", "message": response_text}
                        
            except Exception as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return {"error": "request_failed", "message": str(e)}
        
        return {"error": "max_retries_exceeded"}

    async def create_buy_transaction(self, wallet_pubkey: str, mint_address: str, sol_amount: float, slippage_bps: int = 100) -> Dict:
        """Create a buy transaction via QuickNode Pump.fun API"""
        # Use public Jupiter API endpoint (from QuickNode free tier)
        url = "https://public.jupiterapi.com/pump-fun/swap"
        
        # Convert SOL to lamports (1 SOL = 1_000_000_000 lamports)
        lamports = int(sol_amount * 1_000_000_000)
        
        payload = {
            "wallet": wallet_pubkey,
            "type": "BUY",
            "mint": mint_address,
            "inAmount": str(lamports),
            "priorityFeeLevel": "high",
            "slippageBps": str(slippage_bps)
        }
        
        self.logger.info(f"Creating buy transaction: {sol_amount} SOL for {mint_address}")
        self.logger.info(f"DEBUG: Request URL: {url}")
        self.logger.info(f"DEBUG: Request payload: {payload}")
        
        try:
            response = await self._make_request(url, payload)
            
            if "error" in response:
                self.logger.error(f"Buy transaction failed: {response}")
                return {
                    "success": False,
                    "error": response["error"],
                    "message": response.get("message", "Unknown error")
                }
            
            # Extract transaction data
            transaction_b64 = response.get("transaction")
            if not transaction_b64:
                return {
                    "success": False,
                    "error": "no_transaction",
                    "message": "No transaction returned from API"
                }
            
            return {
                "success": True,
                "transaction": transaction_b64,
                "mint": mint_address,
                "sol_amount": sol_amount,
                "slippage_bps": slippage_bps,
                "estimated_tokens": response.get("estimatedTokens", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating buy transaction: {e}")
            return {
                "success": False,
                "error": "exception",
                "message": str(e)
            }

    async def create_sell_transaction(self, wallet_pubkey: str, mint_address: str, token_amount: float, slippage_bps: int = 100) -> Dict:
        """Create a sell transaction via QuickNode Pump.fun API"""
        # Use public Jupiter API endpoint (from QuickNode free tier)
        url = "https://public.jupiterapi.com/pump-fun/swap"
        
        # Convert token amount to smallest unit (usually with 6 or 9 decimals)
        # This would need to be adjusted based on token decimals
        token_units = int(token_amount * 1_000_000)  # Assuming 6 decimals
        
        payload = {
            "wallet": wallet_pubkey,
            "type": "SELL",
            "mint": mint_address,
            "inAmount": str(token_units),
            "priorityFeeLevel": "high",
            "slippageBps": str(slippage_bps)
        }
        
        self.logger.info(f"Creating sell transaction: {token_amount} tokens for {mint_address}")
        
        try:
            response = await self._make_request(url, payload)
            
            if "error" in response:
                self.logger.error(f"Sell transaction failed: {response}")
                return {
                    "success": False,
                    "error": response["error"],
                    "message": response.get("message", "Unknown error")
                }
            
            transaction_b64 = response.get("transaction")
            if not transaction_b64:
                return {
                    "success": False,
                    "error": "no_transaction",
                    "message": "No transaction returned from API"
                }
            
            return {
                "success": True,
                "transaction": transaction_b64,
                "mint": mint_address,
                "token_amount": token_amount,
                "slippage_bps": slippage_bps,
                "estimated_sol": response.get("estimatedSol", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating sell transaction: {e}")
            return {
                "success": False,
                "error": "exception",
                "message": str(e)
            }

    async def simulate_transaction(self, transaction_b64: str) -> Dict:
        """Simulate a transaction before sending"""
        try:
            # This would use Solana RPC to simulate the transaction
            # For now, return a mock successful simulation
            return {
                "success": True,
                "gas_used": 5000,
                "accounts_accessed": []
            }
        except Exception as e:
            self.logger.error(f"Error simulating transaction: {e}")
            return {"success": False, "error": str(e)}

    async def get_token_info(self, mint_address: str) -> Dict:
        """Get token information from Pump.fun"""
        # This would be implemented if Pump.fun provides token info endpoints
        # For now, return empty dict and rely on Moralis
        return {}

    async def get_price_quote(self, mint_address: str, sol_amount: float, action: str = "BUY") -> Dict:
        """Get price quote for a trade"""
        try:
            # This could use the same swap endpoint with a dry-run flag
            # Or a separate quote endpoint if available
            # For now, return a mock quote
            estimated_tokens = sol_amount * 1000000  # Mock calculation
            
            return {
                "success": True,
                "mint": mint_address,
                "sol_amount": sol_amount,
                "estimated_tokens": estimated_tokens,
                "price_per_token": sol_amount / estimated_tokens if estimated_tokens > 0 else 0,
                "slippage_estimate": 0.02
            }
            
        except Exception as e:
            self.logger.error(f"Error getting price quote: {e}")
            return {"success": False, "error": str(e)}

    async def check_api_status(self) -> bool:
        """Check if Pump.fun API is available"""
        try:
            # Make a simple request to check API status
            # This would be a health check endpoint if available
            session = await self._get_session()
            
            async with session.get(f"{self.quicknode_endpoint}/health") as response:
                return response.status == 200
                
        except Exception as e:
            self.logger.error(f"API status check failed: {e}")
            return False

    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None