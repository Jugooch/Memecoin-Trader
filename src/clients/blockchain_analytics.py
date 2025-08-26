"""
Blockchain Analytics Client for accurate P&L and wallet balance tracking
Uses QuickNode RPC to fetch real-time blockchain data
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import httpx
import json
import base64


class BlockchainAnalytics:
    """Direct blockchain analytics for accurate P&L tracking"""
    
    def __init__(self, rpc_endpoint: str, api_key: Optional[str] = None):
        self.rpc_endpoint = rpc_endpoint
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Set up headers with API key if provided
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
            
        self._client = httpx.AsyncClient(timeout=30)
        
        # SOL decimals and conversion
        self.SOL_DECIMALS = 9
        self.LAMPORTS_PER_SOL = 10 ** self.SOL_DECIMALS
        
        # Cache for SOL price
        self._sol_price_cache = {"price": 0.0, "timestamp": 0}
        self._sol_price_cache_duration = 60  # 60 seconds cache
        
        # Store daily starting balance
        self._daily_starting_balance = None
        self._daily_reset_time = None
        
        # Transaction cache to avoid re-fetching
        self._tx_cache = {}  # signature -> transaction details
        self._cached_stats = {"data": None, "timestamp": 0}
        self._stats_cache_duration = 120  # Cache stats for 2 minutes
        
    async def get_wallet_sol_balance(self, wallet_address: str) -> float:
        """
        Get current SOL balance for a wallet
        
        Args:
            wallet_address: The wallet public key
            
        Returns:
            SOL balance as float
        """
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBalance",
                "params": [wallet_address]
            }
            
            response = await self._client.post(
                self.rpc_endpoint,
                json=payload,
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data and "value" in data["result"]:
                    lamports = data["result"]["value"]
                    sol_balance = lamports / self.LAMPORTS_PER_SOL
                    return sol_balance
            
            self.logger.error(f"Failed to get wallet balance: {response.text}")
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error fetching wallet balance: {e}")
            return 0.0
    
    async def _get_recent_signatures(self, wallet_address: str) -> List[Dict]:
        """
        Get just transaction signatures without full details (1 RPC call)
        """
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [
                    wallet_address,
                    {"limit": 100}  # Get last 100 transactions
                ]
            }
            
            response = await self._client.post(
                self.rpc_endpoint,
                json=payload,
                headers=self.headers
            )
            
            if response.status_code != 200:
                return []
                
            data = response.json()
            if "result" not in data:
                return []
                
            # Filter to last 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            cutoff_timestamp = int(cutoff_time.timestamp())
            
            today_sigs = []
            for sig_info in data["result"]:
                if sig_info.get("blockTime", 0) >= cutoff_timestamp:
                    today_sigs.append(sig_info)
                else:
                    break  # Ordered by time
            
            return today_sigs
            
        except Exception as e:
            self.logger.error(f"Error fetching signatures: {e}")
            return []
    
    async def get_wallet_transactions_today(self, wallet_address: str) -> List[Dict]:
        """
        Get all transactions for a wallet from today (last 24 hours)
        
        Args:
            wallet_address: The wallet public key
            
        Returns:
            List of transaction details
        """
        try:
            # Get transactions signatures for the last 24 hours
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [
                    wallet_address,
                    {
                        "limit": 1000  # Max transactions to fetch
                    }
                ]
            }
            
            response = await self._client.post(
                self.rpc_endpoint,
                json=payload,
                headers=self.headers
            )
            
            if response.status_code != 200:
                self.logger.error(f"Failed to get signatures: {response.text}")
                return []
                
            data = response.json()
            if "result" not in data or not data["result"]:
                self.logger.info("No transaction signatures found")
                return []
                
            signatures = data["result"]
            if not isinstance(signatures, list):
                self.logger.warning(f"Unexpected signatures format: {type(signatures)}")
                return []
            
            # Filter to last 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            cutoff_timestamp = int(cutoff_time.timestamp())
            
            today_sigs = []
            for sig_info in signatures:
                if not isinstance(sig_info, dict):
                    continue
                if sig_info.get("blockTime", 0) >= cutoff_timestamp:
                    today_sigs.append(sig_info)  # Keep the full sig_info object
                else:
                    break  # Signatures are ordered by time, so we can break early
            
            if not today_sigs:
                self.logger.info("No transactions found in the last 24 hours")
                return []
            
            # Get transaction details, using cache for already-fetched transactions
            transactions = []
            new_signatures = []
            
            # Check which signatures we need to fetch
            for sig_info in today_sigs[:50]:  # Look at up to 50 most recent
                if not isinstance(sig_info, dict) or "signature" not in sig_info:
                    continue
                    
                sig = sig_info["signature"]
                if sig in self._tx_cache:
                    # Use cached transaction
                    transactions.append(self._tx_cache[sig])
                else:
                    new_signatures.append(sig)
            
            # Fetch only new transactions (limit to 10 new ones per call)
            self.logger.info(f"Found {len(transactions)} cached txs, fetching {min(len(new_signatures), 10)} new txs")
            
            for i, sig in enumerate(new_signatures[:10]):
                tx_details = await self._get_transaction_details(sig)
                if tx_details:
                    transactions.append(tx_details)
                    self._tx_cache[sig] = tx_details  # Cache it
                
                # Small delay every 3 transactions
                if (i + 1) % 3 == 0 and i < len(new_signatures[:10]) - 1:
                    await asyncio.sleep(0.2)
            
            return transactions
            
        except Exception as e:
            self.logger.error(f"Error fetching wallet transactions: {e}")
            return []
    
    async def _get_transaction_details(self, signature: str) -> Optional[Dict]:
        """Get detailed transaction information"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [
                    signature,
                    {
                        "encoding": "jsonParsed",
                        "maxSupportedTransactionVersion": 0
                    }
                ]
            }
            
            response = await self._client.post(
                self.rpc_endpoint,
                json=payload,
                headers=self.headers
            )
            
            if response.status_code != 200:
                return None
                
            data = response.json()
            if "result" not in data or not data["result"]:
                return None
                
            return data["result"]
            
        except Exception as e:
            self.logger.error(f"Error getting transaction details: {e}")
            return None
    
    async def get_accurate_portfolio_stats(self, wallet_address: str) -> Dict:
        """
        Get accurate portfolio stats from blockchain for Discord summaries
        Uses caching to minimize RPC calls
        
        Args:
            wallet_address: The wallet public key
            
        Returns:
            Dict with accurate P&L and equity metrics
        """
        try:
            # Check if we have cached stats that are still fresh
            current_time = datetime.utcnow().timestamp()
            if (self._cached_stats["data"] and 
                current_time - self._cached_stats["timestamp"] < self._stats_cache_duration):
                self.logger.info("Using cached portfolio stats (< 2 minutes old)")
                # Just update the current balance and equity
                cached = self._cached_stats["data"].copy()
                current_balance = await self.get_wallet_sol_balance(wallet_address)
                sol_price = await self.get_sol_price()
                cached["current_balance_sol"] = round(current_balance, 4)
                cached["current_equity_usd"] = round(current_balance * sol_price, 2)
                cached["sol_price"] = round(sol_price, 2)
                return cached
            # Get current balance and SOL price (2 calls, price is cached)
            current_balance = await self.get_wallet_sol_balance(wallet_address)
            sol_price = await self.get_sol_price()
            
            # Get today's transactions (optimized)
            transactions = await self.get_wallet_transactions_today(wallet_address)
            
            # Handle case where there are no transactions
            if not transactions:
                self.logger.info("No transactions found today - wallet balance unchanged")
                # Return current balance as both starting and ending (no P&L)
                return {
                    "current_balance_sol": round(current_balance, 4),
                    "starting_balance_sol": round(current_balance, 4),
                    "net_sol_change": 0.0,
                    "net_usd_change": 0.0,
                    "total_sol_in": 0.0,
                    "total_sol_out": 0.0,
                    "fees_paid_sol": 0.0,
                    "fees_paid_usd": 0.0,
                    "transaction_count": 0,
                    "swap_count": 0,
                    "buy_trades": 0,
                    "sell_trades": 0,
                    "current_equity_usd": round(current_balance * sol_price, 2),
                    "sol_price": round(sol_price, 2)
                }
            
            # Track SOL movements for accurate P&L
            total_sol_in = 0.0
            total_sol_out = 0.0
            swap_count = 0
            buy_count = 0
            sell_count = 0
            fees_paid = 0.0
            
            for tx in transactions:
                if not tx or "meta" not in tx:
                    continue
                    
                meta = tx["meta"]
                if meta.get("err"):
                    continue  # Skip failed transactions
                    
                # Calculate SOL changes from pre/post balances
                pre_balances = meta.get("preBalances", [])
                post_balances = meta.get("postBalances", [])
                
                # Get wallet's account index
                account_keys = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])
                wallet_index = None
                
                for i, key in enumerate(account_keys):
                    key_str = key if isinstance(key, str) else key.get("pubkey", "")
                    if key_str == wallet_address:
                        wallet_index = i
                        break
                
                if wallet_index is not None and wallet_index < len(pre_balances):
                    pre_sol = pre_balances[wallet_index] / self.LAMPORTS_PER_SOL
                    post_sol = post_balances[wallet_index] / self.LAMPORTS_PER_SOL
                    sol_change = post_sol - pre_sol
                    
                    # Account for fees
                    fee = meta.get("fee", 0) / self.LAMPORTS_PER_SOL
                    fees_paid += fee
                    
                    # Track gross movements (before fees)
                    gross_change = sol_change + fee
                    
                    if gross_change > 0.001:  # Received SOL (likely a sell)
                        total_sol_in += gross_change
                        sell_count += 1
                    elif gross_change < -0.001:  # Sent SOL (likely a buy)
                        total_sol_out += abs(gross_change)
                        buy_count += 1
                
                # Count swaps by checking for DEX programs
                instructions = tx.get("transaction", {}).get("message", {}).get("instructions", [])
                for inst in instructions:
                    program = inst.get("programId", "")
                    # Common DEX program IDs
                    dex_programs = [
                        "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB",  # Jupiter v4
                        "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",  # Jupiter v6
                        "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",  # Orca Whirlpool
                        "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Raydium AMM
                        "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBymtb1U"  # Pump.fun
                    ]
                    if any(dex in str(program) for dex in dex_programs):
                        swap_count += 1
                        break
            
            # Calculate net P&L
            net_sol_change = total_sol_in - total_sol_out - fees_paid
            net_usd_change = net_sol_change * sol_price
            
            # Calculate starting balance
            implied_starting_balance = current_balance - net_sol_change
            
            result = {
                "current_balance_sol": round(current_balance, 4),
                "starting_balance_sol": round(implied_starting_balance, 4),
                "net_sol_change": round(net_sol_change, 4),
                "net_usd_change": round(net_usd_change, 2),
                "total_sol_in": round(total_sol_in, 4),
                "total_sol_out": round(total_sol_out, 4),
                "fees_paid_sol": round(fees_paid, 4),
                "fees_paid_usd": round(fees_paid * sol_price, 2),
                "transaction_count": len(transactions),
                "swap_count": swap_count,
                "buy_trades": buy_count,
                "sell_trades": sell_count,
                "current_equity_usd": round(current_balance * sol_price, 2),
                "sol_price": round(sol_price, 2)
            }
            
            # Cache the result
            self._cached_stats = {"data": result, "timestamp": current_time}
            
            # Clean up old cached transactions (keep only last 100)
            if len(self._tx_cache) > 100:
                # Keep only the 100 most recent
                recent_sigs = [sig["signature"] if isinstance(sig, dict) else sig for sig in today_sigs[:100]]
                self._tx_cache = {k: v for k, v in self._tx_cache.items() if k in recent_sigs}
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating daily P&L: {e}")
            return {
                "current_balance_sol": 0.0,
                "starting_balance_sol": 0.0,
                "net_sol_change": 0.0,
                "net_usd_change": 0.0,
                "total_sol_in": 0.0,
                "total_sol_out": 0.0,
                "fees_paid_sol": 0.0,
                "fees_paid_usd": 0.0,
                "transaction_count": 0,
                "swap_count": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "current_equity_usd": 0.0,
                "sol_price": 0.0
            }
    
    async def get_sol_price(self) -> float:
        """Get current SOL price in USD with caching"""
        try:
            # Check cache
            current_time = datetime.utcnow().timestamp()
            if (self._sol_price_cache["price"] > 0 and 
                current_time - self._sol_price_cache["timestamp"] < self._sol_price_cache_duration):
                return self._sol_price_cache["price"]
            
            # Try multiple price sources for reliability
            price = await self._fetch_sol_price_coingecko()
            
            if price <= 0:
                # Fallback to Coinbase
                price = await self._fetch_sol_price_coinbase()
            
            if price > 0:
                self._sol_price_cache = {"price": price, "timestamp": current_time}
                return price
            
            # Fallback to last known price or default
            if self._sol_price_cache["price"] > 0:
                return self._sol_price_cache["price"]
            
            # Default fallback (approximate SOL price)
            return 200.0
            
        except Exception as e:
            self.logger.error(f"Error fetching SOL price: {e}")
            return self._sol_price_cache.get("price", 200.0)
    
    async def _fetch_sol_price_coingecko(self) -> float:
        """Fetch SOL price from CoinGecko"""
        try:
            response = await self._client.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("solana", {}).get("usd", 0.0)
            return 0.0
        except:
            return 0.0
    
    async def _fetch_sol_price_coinbase(self) -> float:
        """Fetch SOL price from Coinbase as backup"""
        try:
            response = await self._client.get(
                "https://api.coinbase.com/v2/exchange-rates?currency=SOL",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                return float(data.get("data", {}).get("rates", {}).get("USD", 0))
            return 0.0
        except:
            return 0.0
    
    async def get_token_holdings(self, wallet_address: str) -> List[Dict]:
        """
        Get all SPL token holdings for a wallet
        
        Args:
            wallet_address: The wallet public key
            
        Returns:
            List of token holdings with balances
        """
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    wallet_address,
                    {
                        "programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
                    },
                    {
                        "encoding": "jsonParsed"
                    }
                ]
            }
            
            response = await self._client.post(
                self.rpc_endpoint,
                json=payload,
                headers=self.headers
            )
            
            if response.status_code != 200:
                self.logger.error(f"Failed to get token accounts: {response.text}")
                return []
            
            data = response.json()
            if "result" not in data or "value" not in data["result"]:
                return []
            
            token_accounts = []
            for account in data["result"]["value"]:
                parsed = account.get("account", {}).get("data", {}).get("parsed", {})
                info = parsed.get("info", {})
                
                if info:
                    token_amount = info.get("tokenAmount", {})
                    ui_amount = token_amount.get("uiAmount", 0)
                    
                    # Only include tokens with non-zero balance
                    if ui_amount > 0:
                        token_accounts.append({
                            "mint": info.get("mint", ""),
                            "owner": info.get("owner", ""),
                            "balance": ui_amount,
                            "decimals": token_amount.get("decimals", 0),
                            "amount": token_amount.get("amount", "0")
                        })
            
            return token_accounts
            
        except Exception as e:
            self.logger.error(f"Error fetching token holdings: {e}")
            return []
    
    async def reset_daily_tracking(self, wallet_address: str):
        """Reset daily tracking at start of new trading day"""
        current_balance = await self.get_wallet_sol_balance(wallet_address)
        self._daily_starting_balance = current_balance
        self._daily_reset_time = datetime.utcnow()
        self.logger.info(f"Daily tracking reset. Starting balance: {current_balance:.4f} SOL")
    
    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()