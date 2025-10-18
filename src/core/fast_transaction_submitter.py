"""
Fast Transaction Submitter - Helius + Jito transaction execution

This module ONLY handles transaction building and submission.
All position tracking, P&L, Discord, etc. is handled by TradingEngine.

Based on working frontrun implementation:
- frontrun/core/tx_submitter.py (submission pattern)
- frontrun/clients/pumpfun_client.py (instruction building)
"""

import asyncio
import logging
import time
import aiohttp
import base64
import json
import random
import struct
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import base58
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction
from solders.hash import Hash
from solders.system_program import transfer, TransferParams

# Import frontrun components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "frontrun"))

try:
    from clients.pump_fun_direct import PumpFunDirectClient
    from core.tx_builder import TransactionBuilder, TransactionBuildConfig
    from core.bonding_curve import BondingCurveCalculator
    from core.rpc_manager import RPCManager
    from core.config import RPCConfig, RPCEndpoint
    FRONTRUN_AVAILABLE = True
except ImportError:
    FRONTRUN_AVAILABLE = False
    logging.warning("Frontrun components not available")


# Helius Sender Jito tip accounts (required for Helius Sender endpoint)
JITO_TIP_ACCOUNTS = [
    "4ACfpUFoaSD9bfPdeu6DBt89gB6ENTeHBXCAi87NhDEE",
    "D2L6yPZ2FmmmTKPgzaMKdhu6EWZcTpLy1Vhx8uvZe7NZ",
    "9bnz4RShgq1hAnLnZbP8kbgBg1kEmcJBYQq3gQbmnSta",
    "5VY91ws6B2hMmBFRsXkoAAdsPHBJwRfBht4DXox3xkwn",
    "2nyhqdwKcJZR2vcqCyrYsaPVdAnFoJjiksCXJ7hfEYgD",
    "2q5pghRs6arqVjRvT5gfgWfWcHWmw1ZuCzphgd5KfWGJ",
    "wyvPkWjVZz1M8fHQnMMCDTQDbkManefNNhweYk5WkcF",
    "4TQLFNWK8AovT1gFvda5jfw2oJeRMKEmw7aH6MGBJ3or",
    "3KCKozbAaF75qEU33jtzozcJ29yJuaLJTy2jFdzUY8bT",
]


@dataclass
class TransactionResult:
    """Result of transaction submission"""
    success: bool
    signature: Optional[str] = None
    tokens_received: Optional[float] = None  # For buy transactions
    sol_received: Optional[float] = None  # For sell transactions
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


@dataclass
class TransactionVerification:
    """Result of transaction verification"""
    status: str  # "success", "failed_on_chain", "not_found"
    sol_amount: Optional[float] = None
    error_detail: Optional[str] = None  # Error code/message if failed


class FastTransactionSubmitter:
    """
    Fast transaction builder and submitter using Helius + Jito

    This class ONLY handles:
    - Building Pump.fun buy/sell instructions
    - Building transactions with compute budget
    - Signing transactions
    - Submitting via Helius Sender endpoint

    It does NOT handle:
    - Position tracking
    - P&L calculations
    - Discord notifications
    - TP/SL monitoring
    - Multi-tier exits
    - Dump protection
    - Volatility buffers
    - etc.

    All that logic stays in TradingEngine.
    """

    def __init__(self, config: Dict):
        """
        Initialize fast transaction submitter

        Args:
            config: Configuration dict with fast_execution section
        """
        self.logger = logging.getLogger(__name__)

        if not FRONTRUN_AVAILABLE:
            raise RuntimeError("Frontrun components not available - cannot use fast execution")

        # Extract fast execution config
        fast_config = config.get('fast_execution', {})
        self.helius_rpc = fast_config.get('helius_rpc_endpoint')
        self.helius_sender = fast_config.get('helius_sender_endpoint', self.helius_rpc)
        self.slippage_bps = fast_config.get('slippage_bps', 2000)  # 20% default (simulation validates before submission)
        self.priority_fee_lamports = fast_config.get('priority_fee_lamports', 10_000_000)  # 0.01 SOL
        self.jito_tip_lamports = fast_config.get('jito_tip_lamports', 1_000_000)  # 0.001 SOL default

        # Wallet configuration
        wallet_private_key = fast_config.get('wallet_private_key')
        if not wallet_private_key:
            raise ValueError("wallet_private_key required for fast execution")

        # Initialize wallet keypair
        try:
            private_key_bytes = base58.b58decode(wallet_private_key)

            # Handle both formats:
            # - 32 bytes: secret key only -> use from_seed()
            # - 64 bytes: full keypair (secret + public) -> use from_bytes()
            if len(private_key_bytes) == 32:
                # Secret key only - create keypair from seed
                self.wallet = Keypair.from_seed(private_key_bytes)
            elif len(private_key_bytes) == 64:
                # Full keypair - use directly
                self.wallet = Keypair.from_bytes(private_key_bytes)
            else:
                raise ValueError(f"Invalid key length: {len(private_key_bytes)} bytes (expected 32 or 64)")

            self.wallet_pubkey = str(self.wallet.pubkey())
        except Exception as e:
            raise ValueError(f"Invalid wallet private key: {e}")

        # Initialize frontrun components
        self.calculator = BondingCurveCalculator()

        # Initialize RPC manager for PumpFunDirectClient
        helius_endpoint = RPCEndpoint(
            url=self.helius_rpc,
            websocket_url=self.helius_rpc.replace('https://', 'wss://').replace('http://', 'ws://'),
            priority=0,
            label="helius"
        )
        rpc_config = RPCConfig(endpoints=[helius_endpoint])
        self.rpc_manager = RPCManager(rpc_config)

        # Initialize PumpFunDirectClient (builds complete transactions with correct format)
        # High-priority client: Fast execution with Jito tips
        self.pump_client = PumpFunDirectClient(
            self.rpc_manager,
            enable_jito_tips=True,
            jito_tip_lamports=self.jito_tip_lamports
        )

        # Low-priority client: Minimal fees, no Jito tips (for stale positions)
        self.pump_client_low_priority = PumpFunDirectClient(
            self.rpc_manager,
            enable_jito_tips=False,  # Skip Jito tips to save $0.20-1.00
            jito_tip_lamports=0
        )

        self.logger.info("FastTransactionSubmitter initialized")
        self.logger.info(f"  Wallet: {self.wallet_pubkey[:8]}...")
        self.logger.info(f"  Helius RPC: {self.helius_rpc}")
        self.logger.info(f"  Helius Sender: {self.helius_sender}")
        self.logger.info(f"  Slippage: {self.slippage_bps / 100:.1f}%")
        self.logger.info(f"  Jito Tip: {self.jito_tip_lamports / 1e9:.4f} SOL ({self.jito_tip_lamports:,} lamports)")
        self.logger.info(f"  Low-priority mode: 1,000 lamports priority fee, no Jito tips")

    async def ensure_rpc_started(self):
        """Ensure RPC manager is started (lazy initialization)"""
        if not self.rpc_manager._running:
            await self.rpc_manager.start()

    async def _get_recent_blockhash(self) -> Hash:
        """Get recent blockhash from Helius RPC (using working frontrun pattern)"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getLatestBlockhash",
                "params": [{"commitment": "finalized"}]
            }

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.helius_rpc, json=payload) as response:
                    result = await response.json()

                    if "error" in result:
                        raise Exception(f"RPC error: {result['error']}")

                    blockhash_str = result["result"]["value"]["blockhash"]
                    return Hash.from_string(blockhash_str)
        except Exception as e:
            self.logger.error(f"Failed to get blockhash: {e}")
            raise

    async def _submit_transaction(self, signed_tx: Transaction, use_jito: bool = True) -> Optional[str]:
        """
        Submit transaction via Helius Sender (Jito) or regular RPC

        Args:
            signed_tx: Signed transaction to submit
            use_jito: If True, use Helius Sender (requires Jito tip). If False, use regular RPC.

        (Using working frontrun pattern from tx_submitter.py)
        """
        try:
            # Serialize transaction
            tx_bytes = bytes(signed_tx)
            tx_base64 = base64.b64encode(tx_bytes).decode('utf-8')

            # Choose endpoint based on whether we're using Jito
            endpoint = self.helius_sender if use_jito else self.helius_rpc
            endpoint_type = "Helius Sender (Jito)" if use_jito else "Regular RPC"

            self.logger.debug(f"Submitting transaction ({len(tx_bytes)} bytes) to {endpoint_type}")

            # Build RPC request payload
            # Note: Helius Sender doesn't support preflight simulation (speed optimization)
            # High slippage tolerance (80% buys, 50% sells) prevents most failures
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendTransaction",
                "params": [
                    tx_base64,
                    {
                        "skipPreflight": True,
                        "encoding": "base64",
                        "maxRetries": 0
                    }
                ]
            }

            # Submit via selected endpoint
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, json=payload) as response:
                    response_text = await response.text()
                    self.logger.debug(f"RPC response status: {response.status}, body: {response_text[:500]}")

                    # Parse response
                    try:
                        result = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse RPC response: {e}, response: {response_text[:200]}")
                        return None

                    if "error" in result:
                        error_data = result["error"]
                        error_msg = error_data.get("message", str(error_data))
                        error_code = error_data.get("code", "unknown")
                        self.logger.error(f"Transaction submission RPC error (code {error_code}): {error_msg}")
                        return None

                    signature = result.get("result")
                    if signature:
                        self.logger.info(f"Transaction submitted successfully: {signature}")
                    else:
                        self.logger.warning(f"No signature in result: {result}")

                    return signature

        except Exception as e:
            self.logger.error(f"Failed to submit transaction: {e}", exc_info=True)
            return None

    async def _simulate_transaction(self, signed_tx: Transaction) -> Dict:
        """
        Simulate a transaction to check if it would succeed and extract token amounts

        Args:
            signed_tx: Signed transaction to simulate

        Returns:
            Dict with:
                - success: bool
                - error: Optional[str]
                - compute_units: Optional[int]
                - tokens_received: Optional[float] (for buy transactions)
        """
        try:
            # Serialize transaction to base64
            tx_bytes = bytes(signed_tx)
            tx_base64 = base64.b64encode(tx_bytes).decode('utf-8')

            # Build simulateTransaction RPC request
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "simulateTransaction",
                "params": [
                    tx_base64,
                    {
                        "encoding": "base64",
                        "replaceRecentBlockhash": True,  # Use latest blockhash
                        "sigVerify": False,  # Skip sig verification for speed
                        "accounts": {
                            "encoding": "jsonParsed",
                            "addresses": []  # Return all account states
                        }
                    }
                ]
            }

            # Send simulation request with short timeout
            timeout = aiohttp.ClientTimeout(total=3)  # 3 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.helius_rpc, json=payload) as response:
                    result = await response.json()

                    # Check for RPC errors
                    if "error" in result:
                        error_msg = result["error"].get("message", str(result["error"]))
                        return {
                            "success": False,
                            "error": error_msg
                        }

                    # Parse simulation result
                    sim_result = result.get("result", {}).get("value", {})

                    # Check if simulation failed
                    if sim_result.get("err"):
                        error_detail = str(sim_result["err"])
                        return {
                            "success": False,
                            "error": error_detail
                        }

                    # Simulation succeeded
                    compute_units = sim_result.get("unitsConsumed", 0)
                    accounts = sim_result.get("accounts", [])
                    logs = sim_result.get("logs", [])

                    # Extract token amount from Pump.fun TradeEvent
                    # Pump.fun emits "Program data:" with base64-encoded TradeEvent struct
                    # Token amount is at offset 48 as a little-endian u64
                    tokens_received = None
                    for log in logs:
                        if log.startswith("Program data: "):
                            try:
                                # Extract base64 data after "Program data: "
                                data_b64 = log.split("Program data: ", 1)[1].strip()
                                data = base64.b64decode(data_b64)

                                # Token amount is at offset 48 as little-endian u64
                                if len(data) >= 56:  # Need at least 48 + 8 bytes
                                    token_amount = struct.unpack('<Q', data[48:56])[0]
                                    tokens_received = token_amount  # Raw tokens (with 6 decimals)
                                    self.logger.debug(f"Extracted token amount from TradeEvent: {token_amount:,}")
                                    break
                            except Exception as e:
                                self.logger.debug(f"Failed to parse Program data: {e}")
                                pass

                    return {
                        "success": True,
                        "compute_units": compute_units,
                        "tokens_received": tokens_received
                    }

        except Exception as e:
            self.logger.error(f"Simulation exception: {e}")
            return {
                "success": False,
                "error": f"Simulation exception: {str(e)}"
            }

    async def buy(
        self,
        mint: str,
        sol_amount: float,
        slippage_bps: Optional[int] = None
    ) -> TransactionResult:
        """
        Build and submit a buy transaction using PumpFunDirectClient

        Args:
            mint: Token mint address
            sol_amount: Amount of SOL to spend
            slippage_bps: Slippage in basis points (optional, uses config default)

        Returns:
            TransactionResult with success status and EXPECTED token estimate
        """
        start_time = time.time()
        slippage = slippage_bps if slippage_bps is not None else self.slippage_bps

        try:
            # Ensure RPC manager is started
            await self.ensure_rpc_started()

            # CRITICAL: Calculate expected tokens BEFORE building transaction
            # PumpFunDirectClient calculates this internally but doesn't return it
            # We need this value for position tracking
            from core.bonding_curve import fetch_bonding_curve_state

            # Retry bonding curve fetch for brand new tokens (Helius indexing delay)
            # If not indexed in ~1 second, opportunity is GONE - move on!
            curve_data = None
            max_retries = 3  # Fast retries only
            retry_delays = [0.2, 0.4, 0.6]  # Total: ~1.2 seconds max

            for attempt in range(max_retries):
                curve_data = await fetch_bonding_curve_state(self.rpc_manager, mint)
                if curve_data:
                    if attempt > 0:
                        self.logger.info(f"✅ Bonding curve found after {attempt} retries ({sum(retry_delays[:attempt]):.1f}s total)")
                    break

                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    self.logger.debug(f"Bonding curve not found for {mint[:8]}..., retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)

            if not curve_data:
                return TransactionResult(
                    success=False,
                    error=f"Token not indexed after {sum(retry_delays):.1f}s - opportunity missed or not on Pump.fun"
                )

            curve_state, _, _, _ = curve_data

            # Calculate tokens out (same logic as PumpFunDirectClient line 159-163)
            sol_lamports = int(sol_amount * 1e9)
            quote = self.calculator.calculate_buy_price(curve_state, sol_lamports)
            expected_tokens = int(quote.tokens_out)

            self.logger.info(f"💰 Expected tokens from bonding curve: {expected_tokens:,.0f} ({expected_tokens / 1e6:.2f}M)")

            # Build complete transaction using PumpFunDirectClient (correct format with 16 accounts + 2 params)
            signed_tx = await self.pump_client.create_buy_transaction(
                keypair=self.wallet,
                mint_str=mint,
                sol_amount=sol_amount,
                slippage_bps=slippage,
                priority_fee_lamports=self.priority_fee_lamports,
                compute_units=200_000,
                # Pass curve state to avoid re-fetching (saves ~50ms)
                curve_state=curve_state,
                bonding_curve_address=str(curve_data[1]),
                associated_bonding_curve_address=str(curve_data[2]),
                creator_address=str(curve_data[3])
            )

            if not signed_tx:
                return TransactionResult(
                    success=False,
                    error="Failed to build transaction"
                )

            # SIMULATE TRANSACTION BEFORE SUBMITTING (prevents ghost positions from slippage failures)
            # This adds ~50-100ms but saves us from creating positions that fail
            self.logger.debug(f"Simulating buy transaction before submission...")
            sim_result = await self._simulate_transaction(signed_tx)

            if not sim_result["success"]:
                # Simulation failed - transaction would fail on-chain
                error_msg = sim_result.get("error", "Unknown simulation error")
                self.logger.warning(f"❌ Buy simulation FAILED: {error_msg}")

                # Check if it's a slippage error
                if "slippage" in error_msg.lower() or "0x1771" in error_msg:
                    self.logger.warning(f"⚠️ Slippage exceeded - price moved too much, skipping buy")

                return TransactionResult(
                    success=False,
                    error=f"Simulation failed: {error_msg}"
                )

            # Simulation succeeded - extract actual token amounts
            simulated_tokens = sim_result.get("tokens_received")
            cu_used = sim_result.get("compute_units", 0)

            if simulated_tokens:
                self.logger.info(f"✅ Simulation OK - Actual tokens: {simulated_tokens:,.0f}, CU: {cu_used:,}")
                # Use simulated token amount instead of bonding curve estimate (more accurate!)
                expected_tokens = simulated_tokens
            else:
                # Couldn't extract tokens from simulation logs - use bonding curve estimate
                self.logger.warning(f"⚠️ Simulation OK but couldn't extract token amount - using bonding curve estimate: {expected_tokens:,.0f}")
                # Keep expected_tokens as is (bonding curve estimate)

            # Submit via Helius Sender (always use Jito for buys - need speed!)
            signature = await self._submit_transaction(signed_tx, use_jito=True)

            if not signature:
                return TransactionResult(
                    success=False,
                    error="Transaction submission failed"
                )

            execution_time = (time.time() - start_time) * 1000

            # Return success with expected tokens from bonding curve calculation
            # NOTE: Actual tokens may differ if other txs land first
            return TransactionResult(
                success=True,
                signature=signature,
                tokens_received=expected_tokens,  # Expected tokens from curve at submission time
                execution_time_ms=execution_time
            )

        except Exception as e:
            self.logger.error(f"Fast buy error: {e}")
            return TransactionResult(
                success=False,
                error=str(e)
            )

    async def sell(
        self,
        mint: str,
        token_amount: float,
        slippage_bps: Optional[int] = None,
        low_priority: bool = False
    ) -> TransactionResult:
        """
        Build and submit a sell transaction

        Args:
            mint: Token mint address
            token_amount: Amount of tokens to sell (UI amount)
            slippage_bps: Slippage in basis points (optional, uses config default)
            low_priority: If True, use minimal fees and skip Jito (for stale positions)

        Returns:
            TransactionResult with success status and actual SOL expected from bonding curve
        """
        start_time = time.time()
        slippage = slippage_bps if slippage_bps is not None else self.slippage_bps

        # Use minimal fees for low-priority sells (dead tokens with no activity)
        if low_priority:
            priority_fee = 1_000  # Minimal: 1,000 lamports (~$0.0002)
            use_jito = False
            self.logger.info(f"🐌 Low-priority sell: Using minimal fee (1,000 lamports) and skipping Jito")
        else:
            priority_fee = self.priority_fee_lamports
            use_jito = True

        try:
            # Ensure RPC manager is started
            await self.ensure_rpc_started()

            # CRITICAL: Calculate expected SOL BEFORE building transaction
            # We need this for accurate P&L calculations
            from core.bonding_curve import fetch_bonding_curve_state

            # Retry bonding curve fetch for brand new tokens (Helius indexing delay)
            # If not indexed in ~1 second, opportunity is GONE - move on!
            curve_data = None
            max_retries = 3  # Fast retries only
            retry_delays = [0.2, 0.4, 0.6]  # Total: ~1.2 seconds max

            for attempt in range(max_retries):
                curve_data = await fetch_bonding_curve_state(self.rpc_manager, mint)
                if curve_data:
                    if attempt > 0:
                        self.logger.info(f"✅ Bonding curve found after {attempt} retries ({sum(retry_delays[:attempt]):.1f}s total)")
                    break

                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    self.logger.debug(f"Bonding curve not found for {mint[:8]}..., retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)

            if not curve_data:
                return TransactionResult(
                    success=False,
                    error=f"Token not indexed after {sum(retry_delays):.1f}s - opportunity missed or not on Pump.fun"
                )

            curve_state, _, _, _ = curve_data

            # Calculate SOL out (same logic as PumpFunDirectClient)
            token_amount_raw = int(token_amount * 1e6)  # Convert UI to raw (6 decimals)
            quote = self.calculator.calculate_sell_price(curve_state, token_amount_raw)
            expected_sol_lamports = quote.sol_out
            expected_sol = expected_sol_lamports / 1e9  # Convert lamports to SOL

            self.logger.info(f"💵 Expected SOL from bonding curve: {expected_sol:.6f} SOL ({expected_sol_lamports:,} lamports)")

            # Choose client based on priority level
            # Low-priority: Minimal fees, no Jito (for dead tokens)
            # High-priority: Standard fees with Jito (for active trading)
            client = self.pump_client_low_priority if low_priority else self.pump_client

            # Build complete transaction using PumpFunDirectClient (correct format with 14 accounts + 2 params)
            signed_tx = await client.create_sell_transaction(
                keypair=self.wallet,
                mint_str=mint,
                token_amount=token_amount,
                slippage_bps=slippage,
                priority_fee_lamports=priority_fee,  # Use low fee for stale positions
                compute_units=200_000,
                # Pass curve state to avoid re-fetching (saves ~50ms)
                curve_state=curve_state,
                bonding_curve_address=str(curve_data[1]),
                associated_bonding_curve_address=str(curve_data[2]),
                creator_address=str(curve_data[3])
            )

            if not signed_tx:
                return TransactionResult(
                    success=False,
                    error="Failed to build transaction"
                )

            # Submit via appropriate endpoint
            # Low-priority: Regular RPC (no Jito requirement)
            # High-priority: Helius Sender (Jito bundle for speed)
            signature = await self._submit_transaction(signed_tx, use_jito=use_jito)

            if not signature:
                return TransactionResult(
                    success=False,
                    error="Transaction submission failed"
                )

            execution_time = (time.time() - start_time) * 1000

            # Return success with expected SOL from bonding curve calculation
            return TransactionResult(
                success=True,
                signature=signature,
                sol_received=expected_sol,  # Actual expected SOL from bonding curve
                execution_time_ms=execution_time
            )

        except Exception as e:
            self.logger.error(f"Fast sell error: {e}")
            return TransactionResult(
                success=False,
                error=str(e)
            )

    async def get_wallet_balance(self) -> Optional[float]:
        """Get wallet SOL balance"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBalance",
                "params": [self.wallet_pubkey]
            }

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.helius_rpc, json=payload) as response:
                    result = await response.json()

                    if "error" in result:
                        return None

                    lamports = result.get("result", {}).get("value", 0)
                    return lamports / 1e9  # Convert to SOL

        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            return None

    async def get_token_balance(self, mint_address: str) -> Optional[float]:
        """
        Get actual token balance from wallet's token account
        Used to verify tokens received after buy transaction
        """
        try:
            # Get associated token account address
            from spl.token.instructions import get_associated_token_address

            user_pubkey = self.wallet.pubkey()
            mint_pubkey = Pubkey.from_string(mint_address)
            token_account = get_associated_token_address(user_pubkey, mint_pubkey)

            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountBalance",
                "params": [str(token_account)]
            }

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.helius_rpc, json=payload) as response:
                    result = await response.json()

                    if "error" in result:
                        # Token account doesn't exist or other error
                        return None

                    token_data = result.get("result", {}).get("value")
                    if not token_data:
                        return None

                    # Get UI amount (already divided by decimals)
                    ui_amount = token_data.get("uiAmount", 0)
                    return float(ui_amount) if ui_amount else None

        except Exception as e:
            self.logger.error(f"Failed to get token balance for {mint_address[:8]}...: {e}")
            return None

    async def get_transaction_sol_delta(self, signature: str, is_buy: bool = True) -> TransactionVerification:
        """
        Get ACTUAL SOL spent/received from transaction receipt.
        This is 100% accurate and includes ALL fees (Jito, priority, network, Pump.fun).

        Args:
            signature: Transaction signature
            is_buy: True for buy (returns positive SOL spent), False for sell (returns positive SOL received)

        Returns:
            TransactionVerification with:
                - status: "success", "failed_on_chain", or "not_found"
                - sol_amount: Actual SOL delta (if successful)
                - error_detail: Error message (if failed)
        """
        # Retry up to 10 times with 400ms delays (total 4 seconds max - gives more time for indexing)
        for attempt in range(10):
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

                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(self.helius_rpc, json=payload) as response:
                        result = await response.json()

                        if "error" in result:
                            # Transaction not found yet - may still be indexing
                            error_msg = result.get("error", {})
                            if attempt < 9:
                                if attempt % 3 == 0:  # Log every 3rd attempt
                                    self.logger.info(f"Transaction {signature[:8]}... not indexed yet (attempt {attempt + 1}/10, error: {error_msg})")
                                await asyncio.sleep(0.4)  # Wait 400ms for indexing
                                continue
                            self.logger.warning(f"❌ Transaction {signature[:8]}... not found after {attempt + 1} attempts (4s timeout), error: {error_msg}")
                            return TransactionVerification(status="not_found", error_detail=str(error_msg))

                        tx_data = result.get("result")
                        if not tx_data or tx_data is None:
                            # Transaction not indexed yet
                            if attempt < 9:
                                if attempt % 3 == 0:  # Log every 3rd attempt
                                    self.logger.info(f"Transaction {signature[:8]}... result is null (attempt {attempt + 1}/10)")
                                await asyncio.sleep(0.4)
                                continue
                            self.logger.warning(f"❌ Transaction {signature[:8]}... returned null after {attempt + 1} attempts (4s timeout)")
                            return TransactionVerification(status="not_found", error_detail="Transaction result is null")

                        # Get balance changes from transaction metadata
                        meta = tx_data.get("meta")
                        if not meta:
                            self.logger.error(f"❌ No metadata in transaction {signature[:8]}... (transaction malformed?)")
                            return TransactionVerification(status="not_found", error_detail="No transaction metadata")

                        # Check if transaction succeeded
                        err = meta.get("err")
                        if err:
                            # Transaction definitively failed - this is a ghost position!
                            self.logger.error(f"❌ Transaction {signature[:8]}... FAILED ON-CHAIN: {err}")
                            return TransactionVerification(status="failed_on_chain", error_detail=str(err))

                        # preBalances[0] and postBalances[0] are YOUR wallet (signer)
                        pre_balances = meta.get("preBalances", [])
                        post_balances = meta.get("postBalances", [])

                        if not pre_balances or not post_balances:
                            self.logger.error(f"❌ Missing balance data in transaction {signature[:8]}...")
                            return TransactionVerification(status="not_found", error_detail="Missing balance data")

                        # Calculate SOL delta
                        pre_balance_lamports = pre_balances[0]
                        post_balance_lamports = post_balances[0]

                        if is_buy:
                            # Buy: pre > post (spent SOL)
                            sol_spent_lamports = pre_balance_lamports - post_balance_lamports
                            sol_spent = sol_spent_lamports / 1e9

                            if attempt > 0:
                                self.logger.info(f"✅ Transaction indexed after {attempt} retries ({attempt * 0.4:.1f}s)")

                            self.logger.info(f"✅ Buy receipt: {sol_spent:.6f} SOL spent (pre: {pre_balance_lamports/1e9:.6f}, post: {post_balance_lamports/1e9:.6f})")
                            return TransactionVerification(status="success", sol_amount=sol_spent)
                        else:
                            # Sell: post > pre (received SOL)
                            sol_received_lamports = post_balance_lamports - pre_balance_lamports
                            sol_received = sol_received_lamports / 1e9

                            if attempt > 0:
                                self.logger.info(f"✅ Transaction indexed after {attempt} retries ({attempt * 0.4:.1f}s)")

                            self.logger.info(f"✅ Sell receipt: {sol_received:.6f} SOL received (pre: {pre_balance_lamports/1e9:.6f}, post: {post_balance_lamports/1e9:.6f})")
                            return TransactionVerification(status="success", sol_amount=sol_received)

            except Exception as e:
                if attempt < 9:
                    if attempt % 3 == 0:  # Log every 3rd attempt
                        self.logger.warning(f"Transaction query attempt {attempt + 1}/10 failed: {type(e).__name__}: {e}")
                    await asyncio.sleep(0.4)
                    continue
                self.logger.error(f"❌ Failed to get transaction after {attempt + 1} attempts (4s): {type(e).__name__}: {e}")
                return TransactionVerification(status="not_found", error_detail=f"Exception: {type(e).__name__}: {e}")

        self.logger.warning(f"❌ Transaction {signature[:8]}... query exhausted all retries")
        return TransactionVerification(status="not_found", error_detail="Query exhausted all retries")
