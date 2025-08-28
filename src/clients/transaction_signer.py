"""
Transaction signing and sending for live trading using QuickNode
"""

import base64
import logging
import json
import aiohttp
from typing import Dict, Optional
from solders.keypair import Keypair
from solders.transaction import Transaction
import base58


class TransactionSigner:
    """Handles signing and sending transactions via QuickNode"""
    
    def __init__(self, quicknode_endpoint: str, quicknode_api_key: str, private_key_base58: str = None):
        """
        Initialize transaction signer with QuickNode
        
        Args:
            quicknode_endpoint: QuickNode RPC endpoint URL
            quicknode_api_key: QuickNode API key
            private_key_base58: Base58 encoded private key from pumpportal config
        """
        self.logger = logging.getLogger(__name__)
        self.quicknode_endpoint = quicknode_endpoint
        self.quicknode_api_key = quicknode_api_key
        
        # Initialize keypair if private key provided
        self.keypair = None
        if private_key_base58:
            try:
                # Decode base58 private key to bytes
                private_key_bytes = base58.b58decode(private_key_base58)
                # Create keypair from private key (solders uses different method)
                self.keypair = Keypair.from_bytes(private_key_bytes)
                self.logger.info(f"Wallet loaded: {str(self.keypair.pubkey())[:8]}...")
            except Exception as e:
                self.logger.error(f"Failed to load wallet from private key: {e}")
                raise
        
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session with QuickNode headers"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={
                    "Content-Type": "application/json"
                    # Note: QuickNode uses API key in URL, not headers
                }
            )
        return self.session
    
    async def _make_rpc_request(self, method: str, params: list) -> Dict:
        """Make RPC request to QuickNode"""
        session = await self._get_session()
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        try:
            async with session.post(self.quicknode_endpoint, json=payload) as response:
                result = await response.json()
                if "error" in result:
                    self.logger.error(f"RPC error: {result['error']}")
                    return {"error": result["error"]}
                return result.get("result", {})
        except Exception as e:
            self.logger.error(f"RPC request failed: {e}")
            return {"error": str(e)}
    
    async def simulate_transaction(self, transaction_b64: str) -> Dict:
        """Simulate a transaction to get expected results without sending it"""
        try:
            if not self.keypair:
                return {"error": "No keypair available"}
            
            # Decode and sign the transaction for simulation
            transaction_bytes = base64.b64decode(transaction_b64)
            
            # Sign it (simulation requires signatures)
            from solders.transaction import VersionedTransaction
            
            tx = VersionedTransaction.from_bytes(transaction_bytes)
            # For VersionedTransaction, create a new signed version
            signed_tx = VersionedTransaction(tx.message, [self.keypair])
            
            # Simulate the signed transaction with token balance tracking
            result = await self._make_rpc_request(
                "simulateTransaction",
                [
                    base64.b64encode(bytes(signed_tx)).decode('utf-8'),
                    {
                        "encoding": "base64",
                        "commitment": "processed",
                        "replaceRecentBlockhash": True,  # Use current blockhash for simulation
                        "innerInstructions": True  # Get detailed instruction data
                        # Note: Removed accounts parameter to avoid "missing addresses" error
                        # We'll rely on preTokenBalances/postTokenBalances + innerInstructions
                    }
                ]
            )
            
            if "error" in result:
                return {"success": False, "error": result.get("error")}
                
            # Parse simulation results
            if result and "value" in result:
                sim_result = result["value"]
                
                # Check if simulation succeeded
                if sim_result.get("err"):
                    return {"success": False, "error": f"Simulation failed: {sim_result.get('err')}"}
                
                # Extract useful data from logs and token balances
                logs = sim_result.get("logs", [])
                accounts = sim_result.get("accounts", [])
                pre_token_balances = sim_result.get("preTokenBalances", [])
                post_token_balances = sim_result.get("postTokenBalances", [])
                inner_instructions = sim_result.get("innerInstructions", [])
                
                # Parse innerInstructions for token transfer data (primary method)
                estimated_tokens = 0
                token_mint = None
                
                if inner_instructions:
                    self.logger.debug("📋 Parsing innerInstructions for token transfers...")
                    for instruction_group in inner_instructions:
                        for instruction in instruction_group.get('instructions', []):
                            # Look for SPL token transfer instructions
                            if (instruction.get('program') == 'spl-token' and 
                                instruction.get('parsed', {}).get('type') == 'transfer'):
                                
                                parsed_info = instruction.get('parsed', {}).get('info', {})
                                amount_str = parsed_info.get('amount', '0')
                                
                                try:
                                    amount = int(amount_str)
                                    if amount > 0:
                                        # Convert from raw token units to actual tokens
                                        # Pump.fun tokens typically use 6 decimals
                                        estimated_tokens = amount / (10 ** 6)
                                        
                                        self.logger.info(f"📊 Found token transfer: {amount} raw units = {estimated_tokens:,.0f} tokens")
                                        
                                        # Take the first significant token transfer we find
                                        # (This is our buy since we're simulating our own transaction)
                                        if estimated_tokens > 10:  # Any meaningful amount
                                            break
                                except (ValueError, TypeError):
                                    continue
                        
                        if estimated_tokens > 0:
                            break
                
                if estimated_tokens > 0:
                    self.logger.info(f"✅ Simulation extracted {estimated_tokens:,.0f} tokens")
                else:
                    self.logger.warning("⚠️ Simulation successful but no token amounts found")
                    self.logger.debug(f"Pre-balances: {len(pre_token_balances)}, Post-balances: {len(post_token_balances)}, InnerInstructions: {len(inner_instructions)}")
                
                return {
                    "success": True,
                    "logs": logs[-10:],  # Last 10 logs
                    "accounts": accounts,
                    "units_consumed": sim_result.get("unitsConsumed", 0),
                    "estimated_tokens": estimated_tokens,
                    "token_mint": token_mint,
                    "pre_token_balances": pre_token_balances,
                    "post_token_balances": post_token_balances,
                    "inner_instructions": inner_instructions
                }
            
            return {"success": False, "error": "Invalid simulation response"}
            
        except Exception as e:
            self.logger.error(f"Error simulating transaction: {e}")
            return {"success": False, "error": str(e)}
    
    async def sign_and_send_transaction(self, transaction_b64: str) -> Dict:
        """
        Sign and send a base64 encoded transaction via QuickNode
        
        Args:
            transaction_b64: Base64 encoded transaction from QuickNode pump-fun API
            
        Returns:
            Dictionary with transaction result
        """
        try:
            if not self.keypair:
                return {
                    "success": False,
                    "error": "No wallet configured for signing"
                }
            
            # Decode base64 transaction
            transaction_bytes = base64.b64decode(transaction_b64)
            
            # Try to deserialize as VersionedTransaction first (Pump Portal format)
            try:
                from solders.transaction import VersionedTransaction
                transaction = VersionedTransaction.from_bytes(transaction_bytes)
                # For VersionedTransaction, we need to sign differently
                signed_tx = VersionedTransaction(transaction.message, [self.keypair])
                signed_tx_bytes = bytes(signed_tx)
                self.logger.info("Signed as VersionedTransaction")
            except Exception as versioned_error:
                # Fallback to regular Transaction
                try:
                    transaction = Transaction.from_bytes(transaction_bytes)
                    transaction.sign(self.keypair)
                    signed_tx_bytes = bytes(transaction)
                    self.logger.info("Signed as regular Transaction")
                except Exception as regular_error:
                    self.logger.error(f"Failed to parse transaction as VersionedTransaction: {versioned_error}")
                    self.logger.error(f"Failed to parse transaction as regular Transaction: {regular_error}")
                    return {
                        "success": False,
                        "error": f"Transaction parsing failed: {regular_error}"
                    }
            
            # Serialize signed transaction back to base64 (signed_tx_bytes already set above)
            signed_tx_b64 = base64.b64encode(signed_tx_bytes).decode('utf-8')
            
            # Send transaction via QuickNode RPC
            self.logger.info("Sending signed transaction via QuickNode...")
            
            # Use sendTransaction RPC method
            result = await self._make_rpc_request(
                "sendTransaction",
                [
                    signed_tx_b64,
                    {
                        "encoding": "base64",
                        "skipPreflight": False,
                        "preflightCommitment": "confirmed",
                        "maxRetries": 3
                    }
                ]
            )
            
            if "error" in result:
                self.logger.error(f"Transaction failed: {result['error']}")
                return {
                    "success": False,
                    "error": result.get("error", {}).get("message", "Unknown error")
                }
            
            # Get transaction signature - if we get this back, the transaction succeeded
            tx_signature = result
            self.logger.info(f"✅ Transaction sent successfully: {tx_signature}")
            
            return {
                "success": True,
                "signature": tx_signature
            }
                
        except Exception as e:
            self.logger.error(f"Error signing/sending transaction: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_wallet_balance(self) -> Optional[float]:
        """Get SOL balance of the wallet via QuickNode"""
        try:
            if not self.keypair:
                return None
            
            result = await self._make_rpc_request(
                "getBalance",
                [str(self.keypair.pubkey())]
            )
            
            if "error" not in result and "value" in result:
                lamports = result["value"]
                sol_balance = lamports / 1_000_000_000  # Convert lamports to SOL
                return sol_balance
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting wallet balance: {e}")
            return None
    
    async def get_token_balance(self, mint_address: str) -> Optional[float]:
        """Get token balance for a specific mint"""
        try:
            if not self.keypair:
                return None
            
            # Get token accounts for wallet
            result = await self._make_rpc_request(
                "getTokenAccountsByOwner",
                [
                    str(self.keypair.pubkey()),
                    {"mint": mint_address},
                    {"encoding": "jsonParsed"}
                ]
            )
            
            if "error" in result:
                self.logger.warning(f"RPC error getting token balance: {result.get('error')}")
                return None
                
            if "value" in result:
                accounts = result.get("value", [])
                if accounts:
                    # Get the first token account (should only be one per mint)
                    account_info = accounts[0].get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
                    token_amount = account_info.get("tokenAmount", {})
                    ui_amount = token_amount.get("uiAmount", 0)
                    self.logger.debug(f"Found token balance: {ui_amount} for mint {mint_address[:8]}...")
                    return float(ui_amount)
                else:
                    self.logger.debug(f"No token accounts found for mint {mint_address[:8]}... (wallet: {str(self.keypair.pubkey())[:8]}...)")
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting token balance: {e}")
            return None
    
    def get_wallet_address(self) -> Optional[str]:
        """Get the wallet's public address"""
        if self.keypair:
            return str(self.keypair.pubkey())
        return None
    
    async def get_transaction_details(self, tx_signature: str) -> Dict:
        """Get transaction details including logs to parse actual token amounts"""
        try:
            result = await self._make_rpc_request(
                "getTransaction",
                [
                    tx_signature,
                    {
                        "encoding": "jsonParsed",
                        "maxSupportedTransactionVersion": 0
                    }
                ]
            )
            
            if result and "error" not in result:
                # Check if transaction failed (has error in meta)
                meta = result.get("meta", {})
                if meta and meta.get("err") is not None:
                    self.logger.warning(f"❌ Transaction {tx_signature} failed with error: {meta.get('err')}")
                    result["_transaction_failed"] = True
                    result["_error_detail"] = meta.get("err")
                return result
            else:
                self.logger.error(f"Failed to get transaction details: {result.get('error', 'Unknown error')}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting transaction details: {e}")
            return {}
    
    def parse_token_transfer_from_logs(self, transaction_data: Dict, mint_address: str, wallet_address: str) -> float:
        """Parse transaction logs to extract exact token transfer amount"""
        try:
            # Check if we got valid transaction data
            if not transaction_data:
                self.logger.warning("No transaction data provided to parse")
                return 0.0
                
            # Get the transaction meta and logs
            meta = transaction_data.get("meta", {})
            
            # Parse postTokenBalances to see the balance change
            post_balances = meta.get("postTokenBalances", [])
            pre_balances = meta.get("preTokenBalances", [])
            
            # Find our wallet's token account for this mint
            post_balance = 0
            pre_balance = 0
            
            for balance in post_balances:
                if (balance.get("mint") == mint_address and 
                    balance.get("owner") == wallet_address):
                    post_balance = float(balance.get("uiTokenAmount", {}).get("uiAmount", 0))
                    break
            
            for balance in pre_balances:
                if (balance.get("mint") == mint_address and 
                    balance.get("owner") == wallet_address):
                    pre_balance = float(balance.get("uiTokenAmount", {}).get("uiAmount", 0))
                    break
            
            # Calculate the actual tokens received
            tokens_received = post_balance - pre_balance
            
            if tokens_received > 0:
                self.logger.info(f"Parsed token transfer: {tokens_received} tokens received")
                return tokens_received
            else:
                self.logger.warning(f"No positive token transfer found in transaction logs")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error parsing token transfer from logs: {e}")
            return 0.0

    def parse_sol_change_from_logs(self, transaction_data: Dict, wallet_address: str) -> float:
        """Parse transaction logs to extract exact SOL balance change"""
        try:
            # Get the transaction meta
            meta = transaction_data.get("meta", {})
            
            # Parse pre and post balances for SOL (account balances, not token balances)
            post_balances = meta.get("postBalances", [])
            pre_balances = meta.get("preBalances", [])
            
            # Get account keys to match wallet address
            account_keys = transaction_data.get("transaction", {}).get("message", {}).get("accountKeys", [])
            if not account_keys:
                # Try versioned transaction format
                account_keys = transaction_data.get("transaction", {}).get("message", {}).get("staticAccountKeys", [])
            
            # Find our wallet's account index - handle both string and dict formats
            wallet_index = -1
            for i, account_key in enumerate(account_keys):
                # Handle jsonParsed format (dict) vs json format (string)
                key_to_compare = account_key
                if isinstance(account_key, dict):
                    key_to_compare = account_key.get('pubkey', '')
                    
                if key_to_compare == wallet_address:
                    wallet_index = i
                    break
            
            if wallet_index == -1:
                self.logger.warning(f"Wallet address {wallet_address} not found in transaction accounts")
                return 0.0
            
            # Get pre and post SOL balances for our wallet
            if wallet_index < len(pre_balances) and wallet_index < len(post_balances):
                pre_balance_lamports = pre_balances[wallet_index]
                post_balance_lamports = post_balances[wallet_index]
                
                # Convert lamports to SOL
                pre_balance_sol = pre_balance_lamports / 1_000_000_000
                post_balance_sol = post_balance_lamports / 1_000_000_000
                
                sol_change = post_balance_sol - pre_balance_sol
                
                self.logger.info(f"Parsed SOL change: {sol_change:+.6f} SOL ({pre_balance_sol:.6f} → {post_balance_sol:.6f})")
                return sol_change
            else:
                self.logger.warning(f"Balance indices out of range: wallet_index={wallet_index}, pre_len={len(pre_balances)}, post_len={len(post_balances)}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error parsing SOL change from logs: {e}")
            return 0.0

    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None