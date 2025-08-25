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
        """Create a buy transaction via Jupiter API"""
        try:
            # Step 1: Get quote from Jupiter
            quote_response = await self._get_jupiter_quote(
                input_mint="So11111111111111111111111111111111111111112",  # SOL
                output_mint=mint_address,
                amount=int(sol_amount * 1_000_000_000),  # Convert to lamports
                slippage_bps=slippage_bps
            )
            
            if "error" in quote_response:
                return quote_response
            
            # Step 2: Create swap transaction with quote
            swap_response = await self._create_jupiter_swap(
                user_public_key=wallet_pubkey,
                quote_response=quote_response
            )
            
            if "error" in swap_response:
                return swap_response
            
            return {
                "success": True,
                "transaction": swap_response["swapTransaction"],
                "mint": mint_address,
                "sol_amount": sol_amount,
                "slippage_bps": slippage_bps,
                "estimated_tokens": quote_response.get("outAmount", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating buy transaction: {e}")
            return {
                "success": False,
                "error": "exception",
                "message": str(e)
            }

    async def create_sell_transaction(self, wallet_pubkey: str, mint_address: str, token_amount: float, slippage_bps: int = 100) -> Dict:
        """Create a sell transaction via Jupiter API"""
        try:
            # Convert token amount to smallest unit (assuming 6 decimals for most tokens)
            token_units = int(token_amount * 1_000_000)
            
            # Step 1: Get quote from Jupiter
            quote_response = await self._get_jupiter_quote(
                input_mint=mint_address,  # Token to sell
                output_mint="So11111111111111111111111111111111111111112",  # SOL
                amount=token_units,
                slippage_bps=slippage_bps
            )
            
            if "error" in quote_response:
                return quote_response
            
            # Step 2: Create swap transaction with quote
            swap_response = await self._create_jupiter_swap(
                user_public_key=wallet_pubkey,
                quote_response=quote_response
            )
            
            if "error" in swap_response:
                return swap_response
            
            return {
                "success": True,
                "transaction": swap_response["swapTransaction"],
                "mint": mint_address,
                "token_amount": token_amount,
                "slippage_bps": slippage_bps,
                "estimated_sol": float(quote_response.get("outAmount", 0)) / 1_000_000_000
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

    async def _get_jupiter_quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int) -> Dict:
        """Get a quote from Jupiter API"""
        url = "https://public.jupiterapi.com/quote"
        
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": str(slippage_bps)
        }
        
        session = await self._get_session()
        await self._rate_limit()
        
        try:
            async with session.get(url, params=params) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    return json.loads(response_text)
                else:
                    self.logger.error(f"Quote API error {response.status}: {response_text}")
                    return {"error": f"HTTP {response.status}", "message": response_text}
                    
        except Exception as e:
            self.logger.error(f"Quote request failed: {e}")
            return {"error": "request_failed", "message": str(e)}
    
    async def _create_jupiter_swap(self, user_public_key: str, quote_response: Dict) -> Dict:
        """Create a swap transaction using Jupiter API"""
        url = "https://public.jupiterapi.com/swap"
        
        payload = {
            "userPublicKey": user_public_key,
            "wrapAndUnwrapSol": True,
            "prioritizationFeeLamports": {
                "priorityLevel": "high"
            },
            "quoteResponse": quote_response
        }
        
        self.logger.info(f"Creating Jupiter swap transaction")
        self.logger.info(f"DEBUG: Request URL: {url}")
        self.logger.info(f"DEBUG: Quote response keys: {list(quote_response.keys())}")
        
        try:
            response = await self._make_request(url, payload)
            self.logger.info(f"DEBUG: Swap API Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
            
            if "error" in response:
                self.logger.error(f"Jupiter swap failed: {response}")
                return {
                    "success": False,
                    "error": response["error"],
                    "message": response.get("message", "Unknown error")
                }
            
            if "swapTransaction" not in response:
                self.logger.error(f"No swapTransaction found in response. Available keys: {list(response.keys())}")
                return {
                    "success": False,
                    "error": "no_transaction",
                    "message": f"No swapTransaction returned from API. Response keys: {list(response.keys())}"
                }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error creating Jupiter swap: {e}")
            return {"error": "exception", "message": str(e)}

    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None