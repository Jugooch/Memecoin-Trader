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
                # Create keypair from private key
                self.keypair = Keypair.from_secret_key(private_key_bytes)
                self.logger.info(f"Wallet loaded: {str(self.keypair.public_key)[:8]}...")
            except Exception as e:
                self.logger.error(f"Failed to load wallet from private key: {e}")
                raise
        
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session with QuickNode headers"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.quicknode_api_key  # QuickNode auth
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
            
            # Deserialize transaction  
            transaction = Transaction.from_bytes(transaction_bytes)
            
            # Sign the transaction with our keypair
            transaction.sign(self.keypair)
            
            # Serialize signed transaction back to base64
            signed_tx_bytes = bytes(transaction)
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
            
            # Get transaction signature
            tx_signature = result
            self.logger.info(f"Transaction sent: {tx_signature}")
            
            # Confirm transaction
            self.logger.info("Waiting for confirmation...")
            confirmation = await self._make_rpc_request(
                "confirmTransaction",
                [tx_signature, "confirmed"]
            )
            
            if confirmation and not confirmation.get("error"):
                self.logger.info(f"✅ Transaction confirmed: {tx_signature}")
                return {
                    "success": True,
                    "signature": tx_signature,
                    "confirmed": True
                }
            else:
                self.logger.warning(f"Transaction sent but not confirmed: {tx_signature}")
                return {
                    "success": True,  # Transaction was sent
                    "signature": tx_signature,
                    "confirmed": False,
                    "warning": "Transaction sent but confirmation pending"
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
                [str(self.keypair.public_key)]
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
                    str(self.keypair.public_key),
                    {"mint": mint_address},
                    {"encoding": "jsonParsed"}
                ]
            )
            
            if "error" not in result and "value" in result:
                accounts = result.get("value", [])
                if accounts:
                    # Get the first token account (should only be one per mint)
                    account_info = accounts[0].get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
                    token_amount = account_info.get("tokenAmount", {})
                    ui_amount = token_amount.get("uiAmount", 0)
                    return float(ui_amount)
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting token balance: {e}")
            return None
    
    def get_wallet_address(self) -> Optional[str]:
        """Get the wallet's public address"""
        if self.keypair:
            return str(self.keypair.public_key)
        return None
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None