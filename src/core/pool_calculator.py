"""
Pool-based token calculation for immediate position creation
Calculates expected tokens using pump.fun's bonding curve (x*y=k)
"""

import logging
from typing import Dict, Optional
import aiohttp

class PoolCalculator:
    """Calculate expected tokens using AMM bonding curve math"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
    
    async def _get_session(self):
        """Get or create HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_expected_tokens(self, mint_address: str, sol_amount: float) -> Dict:
        """
        Calculate expected tokens using pump.fun bonding curve
        Returns both expected and min tokens (with safety margin)
        """
        try:
            # Get pool reserves (with fallback to standard curve)
            reserves = await self._get_pool_reserves(mint_address)
            
            # Calculate using AMM formula
            result = self._calculate_amm_output(reserves, sol_amount)
            
            return {
                "success": True,
                "expected_tokens": result["expected_tokens"],
                "min_tokens": result["min_tokens"],
                "price_impact": result["price_impact"],
                "reserves_used": reserves
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating expected tokens: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_pool_reserves(self, mint_address: str) -> Dict:
        """Get pool reserves with fallback to standard pump.fun curve"""
        try:
            session = await self._get_session()
            
            # Try to get real reserves from Pump Portal
            url = f"https://pumpportal.fun/api/data/coins/{mint_address}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    sol_reserves = data.get('virtual_sol_reserves', 0)
                    token_reserves = data.get('virtual_token_reserves', 0)
                    
                    if sol_reserves > 0 and token_reserves > 0:
                        return {
                            "sol_reserves": sol_reserves / 1e9,  # Convert to SOL
                            "token_reserves": token_reserves / 1e6,  # Convert to tokens
                            "source": "api"
                        }
            
            # Fallback to standard pump.fun bonding curve
            return {
                "sol_reserves": 30.0,  # Standard virtual SOL
                "token_reserves": 1073000000.0,  # Standard virtual tokens  
                "source": "default"
            }
            
        except Exception as e:
            self.logger.warning(f"Could not get pool reserves, using default: {e}")
            # Always return standard curve as fallback
            return {
                "sol_reserves": 30.0,
                "token_reserves": 1073000000.0,
                "source": "default"
            }
    
    def _calculate_amm_output(self, reserves: Dict, sol_input: float) -> Dict:
        """Calculate AMM output using constant product formula"""
        
        sol_reserves = reserves["sol_reserves"]
        token_reserves = reserves["token_reserves"]
        
        # Constant product: k = x * y
        k = sol_reserves * token_reserves
        
        # New reserves after adding SOL
        new_sol_reserves = sol_reserves + sol_input
        new_token_reserves = k / new_sol_reserves
        
        # Tokens out (before fees)
        tokens_out = token_reserves - new_token_reserves
        
        # Apply 1% pump.fun fee
        tokens_after_fee = tokens_out * 0.99
        
        # Apply 5% safety margin for position creation
        min_tokens = tokens_after_fee * 0.95
        
        # Calculate price impact
        old_price = sol_reserves / token_reserves
        new_price = new_sol_reserves / new_token_reserves
        price_impact = ((new_price - old_price) / old_price) * 100
        
        return {
            "expected_tokens": tokens_after_fee,
            "min_tokens": min_tokens,
            "price_impact": price_impact
        }
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            self.session = None