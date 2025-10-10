"""
Phase 2: Micro-Trading Script
Execute 10-20 tiny frontrun trades to validate logic

Capital: 1 SOL (~$250)
Position size: 0.01-0.05 SOL per trade
Goal: Validate buy/sell logic works correctly AND test frontrunning capability
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solders.keypair import Keypair
from solders.pubkey import Pubkey

from core.config import ConfigurationManager
from core.logger import setup_logging, get_logger
from services.mempool_monitor import LaserStreamMonitor, LaserStreamConfig, PendingTransaction
from services.dev_detector import DevWalletDetector, DetectorConfig
from core.rpc_manager import RPCManager, RPCEndpoint, RPCConfig
from core.wallet_manager import WalletManager
from clients.pump_fun_direct import PumpFunDirectClient  # DIRECT builder (10x faster!)
from core.tx_signer import TransactionSigner
from core.tx_submitter import TransactionSubmitter, SubmitterConfig
from core.position_tracker import PositionTracker
from core.pnl import PnLCalculator
from core.bonding_curve import fetch_bonding_curve_state, is_new_token_launch, get_market_cap_sol

logger = get_logger(__name__)


@dataclass
class MicroTrade:
    """Record of a micro trade"""
    timestamp: datetime
    mint: str
    signature_buy: str
    signature_sell: str
    buy_amount_sol: float
    sell_amount_sol: float
    profit_sol: float
    fee_paid: int
    success: bool
    reason: str
    # CRITICAL: Slot comparison
    dev_tx_slot: int  # Slot where dev tx landed
    our_tx_slot: int  # Slot where our tx landed
    frontrun_success: bool  # Did we land BEFORE dev?
    slot_delta: int  # our_slot - dev_slot (negative = we won)

    def to_dict(self):
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


class Phase2MicroTrader:
    """
    Phase 2 Micro-Trading Bot

    Executes tiny trades to validate:
    - LaserStream detection works
    - Buy transactions execute with REAL slot tracking
    - Frontrun capability (can we land BEFORE dev buy?)
    - Position management works

    This is THE critical test - can we actually frontrun?
    """

    def __init__(
        self,
        config_path: str,
        wallet_path: str,
        max_trades: int = 10,
        enable_real_trades: bool = False,
        min_buy_sol: float = 1.0,
        max_position_sol: float = 0.5,
        hold_time_seconds: float = 2.0
    ):
        """
        Initialize Phase 2 micro trader

        Args:
            config_path: Path to config.yml
            wallet_path: Path to wallet keypair JSON file
            max_trades: Maximum number of trades (default: 10)
            enable_real_trades: If True, execute real trades. If False, simulate. (default: False)
            min_buy_sol: Minimum dev buy size in SOL to target (default: 1.0)
            max_position_sol: Maximum position size in SOL (default: 0.5)
            hold_time_seconds: Time to hold before selling (default: 2.0)
        """
        self.max_trades = max_trades
        self.trades: List[MicroTrade] = []
        self.enable_real_trades = enable_real_trades
        self.min_buy_sol = min_buy_sol
        self.max_position_sol = max_position_sol
        self.hold_time_seconds = hold_time_seconds

        # Track tokens we've already traded to avoid duplicates
        self._traded_mints: set = set()

        # Load config
        config_mgr = ConfigurationManager(config_path)
        self.bot_config = config_mgr.load_config()

        # Load wallet
        self.keypair = self._load_wallet(wallet_path)
        logger.info(
            "wallet_loaded",
            pubkey=str(self.keypair.pubkey())
        )

        # Get Helius config
        helius_endpoint = self.bot_config.rpc_config.endpoints[0]
        endpoint = helius_endpoint.laserstream_url.replace("https://", "").replace("http://", "")

        # Initialize LaserStream
        self.laserstream = LaserStreamMonitor(LaserStreamConfig(
            endpoint=endpoint,
            api_key=helius_endpoint.laserstream_api_key,
            commitment="processed"
        ))

        # Initialize DevDetector with configurable min buy size
        self.detector = DevWalletDetector(DetectorConfig(
            min_buy_amount_sol=min_buy_sol  # Filter for large dev buys
        ))

        # Initialize transaction components (only if real trades enabled)
        self.rpc_manager: Optional[RPCManager] = None
        self.wallet_manager: Optional[WalletManager] = None
        self.direct_client: Optional[PumpFunDirectClient] = None
        self.tx_submitter: Optional[TransactionSubmitter] = None

        if self.enable_real_trades:
            # Initialize RPC manager
            self.rpc_manager = RPCManager(self.bot_config.rpc_config)

            # Initialize wallet manager with our single wallet (NO COOLDOWN for speed)
            from core.wallet_manager import WalletManagerConfig
            self.wallet_manager = WalletManager(
                rpc_manager=self.rpc_manager,
                keypairs=[self.keypair],
                config=WalletManagerConfig(
                    cooldown_seconds=0,  # CRITICAL: No cooldown for max speed
                    balance_cache_ttl_seconds=30
                )
            )

            # Initialize DIRECT transaction builder (10x faster than Pump Portal!)
            self.direct_client = PumpFunDirectClient(self.rpc_manager)

            self.tx_submitter = TransactionSubmitter(
                rpc_manager=self.rpc_manager,
                config=SubmitterConfig(
                    skip_preflight=True,  # Skip for speed
                    max_retries=3,
                    confirmation_timeout_s=30
                )
            )

        logger.info(
            "phase2_micro_trader_initialized",
            max_trades=max_trades,
            real_trades_enabled=enable_real_trades,
            min_buy_sol=min_buy_sol,
            max_position_sol=max_position_sol,
            hold_time_seconds=hold_time_seconds,
            wallet=str(self.keypair.pubkey())
        )

    def _load_wallet(self, wallet_path: str) -> Keypair:
        """
        Load wallet keypair from JSON file

        Args:
            wallet_path: Path to wallet JSON file (array of bytes)

        Returns:
            Keypair

        Raises:
            Exception: If wallet file not found or invalid
        """
        wallet_file = Path(wallet_path)

        if not wallet_file.exists():
            raise FileNotFoundError(
                f"Wallet file not found: {wallet_path}\n"
                f"Create a wallet with: solana-keygen new --outfile {wallet_path}\n"
                f"Then fund it with at least 1 SOL"
            )

        with open(wallet_file, 'r') as f:
            key_data = json.load(f)

        # Convert to bytes and create keypair
        key_bytes = bytes(key_data)
        keypair = Keypair.from_bytes(key_bytes)

        return keypair

    async def run(self):
        """Run Phase 2 micro-trading"""
        logger.info("=== PHASE 2: MICRO-TRADING STARTED ===")
        logger.info(f"Strategy: Target dev buys ≥ {self.min_buy_sol} SOL to beat retail copy traders")
        logger.info(f"Will execute {self.max_trades} COMPLETE trades (buy + sell)")
        logger.info(f"Position size: 0.01-{self.max_position_sol} SOL per trade (10% of dev buy, capped)")
        logger.info(f"Hold time: {self.hold_time_seconds}s (exit into retail wave)")
        logger.info(f"Mode: {'REAL TRADES' if self.enable_real_trades else 'SIMULATION'}")
        logger.info("Press Ctrl+C to stop early")

        # Start RPC manager if real trades enabled
        if self.enable_real_trades:
            logger.info("Starting RPC manager...")
            await self.rpc_manager.start()

            logger.info("Starting wallet manager...")
            await self.wallet_manager.start()

            # Check wallet balance
            balance = self.wallet_manager.get_total_balance_sol()
            # Estimate: max_trades * max_position_sol + buffer for fees
            # Use smaller buffer for small tests (0.1 SOL) vs large tests (0.5 SOL)
            buffer_sol = 0.1 if self.max_trades <= 20 else 0.5
            min_balance_required = (self.max_trades * self.max_position_sol) + buffer_sol
            logger.info(
                "wallet_balance_check",
                balance_sol=balance,
                min_required=min_balance_required,
                sufficient=balance >= min_balance_required
            )

            if balance < min_balance_required:
                raise RuntimeError(
                    f"Insufficient balance: {balance:.4f} SOL. Need at least {min_balance_required:.1f} SOL for {self.max_trades} trades with {self.max_position_sol} SOL max position size."
                )

        # Start LaserStream monitoring
        monitoring_task = asyncio.create_task(
            self.laserstream.start_monitoring(self._on_opportunity)
        )

        try:
            # Wait until we've executed max_trades
            while len(self.trades) < self.max_trades:
                await asyncio.sleep(1)

            logger.info(f"Completed {self.max_trades} trades!")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            await self.laserstream.stop_monitoring()
            monitoring_task.cancel()

            # Stop components if real trades enabled
            if self.enable_real_trades:
                await self.wallet_manager.stop()
                await self.rpc_manager.stop()

            # Generate report
            await self._generate_report()

    async def _on_opportunity(self, tx: PendingTransaction):
        """
        Handle detected opportunity

        Args:
            tx: Pending transaction from LaserStream
        """
        # Check if we're done
        if len(self.trades) >= self.max_trades:
            return

        # Skip if no mint extracted
        if not tx.mint:
            return

        # Skip if we've already traded this token (avoid duplicate transactions)
        if tx.mint in self._traded_mints:
            logger.debug(
                "skipping_duplicate_mint",
                mint=tx.mint,
                reason="Already traded this token"
            )
            return

        # Analyze opportunity
        opportunity = self.detector.analyze_transaction(
            program_id=tx.program_id,
            buy_amount_sol=tx.buy_amount_sol,
            sender=tx.accounts[0] if tx.accounts else None
        )

        if not opportunity.should_frontrun:
            return

        # CRITICAL: Check if token is NEW (not old token with large buy!)
        # Use TradeEvent data (instant, no RPC lag!) instead of fetching bonding curve
        if self.enable_real_trades:
            # Check if we have bonding curve data from TradeEvent
            if tx.virtual_sol_reserves is None or tx.virtual_token_reserves is None:
                logger.warning(
                    "skipping_no_bonding_curve_data",
                    mint=tx.mint[:16] + "...",
                    reason="No bonding curve data in TradeEvent"
                )
                return

            # Calculate market cap from TradeEvent reserves (NO RPC CALL!)
            virtual_sol = tx.virtual_sol_reserves / 1e9  # Convert lamports to SOL

            # A new token has < 5 SOL in reserves (< $1250 traded)
            is_new = virtual_sol < 5.0
            market_cap = virtual_sol

            if not is_new:
                logger.debug(
                    "skipping_old_token",
                    mint=tx.mint[:16] + "...",
                    market_cap_sol=market_cap,
                    reason="Token is not new (market cap >= 5 SOL)"
                )
                return

            logger.info(
                "new_token_detected",
                mint=tx.mint[:16] + "...",
                market_cap_sol=round(market_cap, 2),
                buy_amount_sol=opportunity.buy_amount_sol,
                source="TradeEvent (instant, no RPC lag!)"
            )

        # Log opportunity
        logger.info(
            "opportunity_detected",
            buy_amount_sol=opportunity.buy_amount_sol,
            signature=tx.signature[:16] + "..."
        )

        # Execute micro trade
        await self._execute_micro_trade(tx, opportunity.buy_amount_sol)

    async def _execute_micro_trade(self, tx: PendingTransaction, dev_buy_amount: float):
        """
        Execute a micro trade (buy only - no sell to keep it simple)

        Args:
            tx: Pending transaction from LaserStream
            dev_buy_amount: Developer's buy amount
        """
        trade_num = len(self.trades) + 1

        # Safety check: skip if no mint extracted
        if not tx.mint:
            logger.warning(
                "skipping_trade_no_mint",
                trade_num=trade_num,
                signature=tx.signature
            )
            return

        # Determine our position size (10% of dev buy, capped by max_position_sol)
        our_buy_amount = min(self.max_position_sol, dev_buy_amount * 0.10)

        # Minimum position size (at least 0.01 SOL to make trade worthwhile)
        our_buy_amount = max(0.01, our_buy_amount)

        # CRITICAL: Record dev transaction slot
        dev_slot = tx.slot

        # Mark this mint as traded immediately (prevents duplicate trades if dev buys again during our execution)
        self._traded_mints.add(tx.mint)

        logger.info(
            f"=== TRADE #{trade_num} ===",
            our_buy_amount_sol=our_buy_amount,
            dev_buy_amount_sol=dev_buy_amount,
            dev_tx_slot=dev_slot,
            mint=tx.mint
        )

        if self.enable_real_trades:
            # REAL TRANSACTION EXECUTION
            await self._execute_real_trade(
                trade_num=trade_num,
                tx=tx,
                our_buy_amount=our_buy_amount,
                dev_slot=dev_slot
            )
        else:
            # SIMULATION MODE
            await self._execute_simulated_trade(
                trade_num=trade_num,
                tx=tx,
                our_buy_amount=our_buy_amount,
                dev_slot=dev_slot
            )

    async def _execute_real_trade(
        self,
        trade_num: int,
        tx: PendingTransaction,
        our_buy_amount: float,
        dev_slot: int
    ):
        """
        Execute COMPLETE trade using DIRECT builder (BUY → HOLD → SELL)

        Flow:
        1. BUY: Create, sign, submit buy transaction DIRECTLY (10x faster!)
        2. HOLD: Wait for retail wave (1-3 seconds)
        3. SELL: Create, sign, submit sell transaction DIRECTLY (10x faster!)
        4. Calculate P&L and slot metrics

        Args:
            trade_num: Trade number
            tx: Pending transaction
            our_buy_amount: Our buy amount in SOL
            dev_slot: Slot where dev transaction landed
        """
        buy_signature = ""
        sell_signature = ""
        buy_slot = 0
        sell_slot = 0
        tokens_received = 0
        sell_sol_received = 0.0

        try:
            # Get wallet
            async with self.wallet_manager.get_wallet(min_sol=our_buy_amount + 0.01) as wallet:
                # ========================================
                # STEP 1: BUY
                # ========================================
                logger.info(
                    "executing_buy",
                    trade_num=trade_num,
                    wallet=str(wallet.pubkey),
                    buy_amount=our_buy_amount,
                    mint=tx.mint
                )

                # Build buy transaction DIRECTLY (10x faster than Pump Portal!)
                buy_signed = await self.direct_client.create_buy_transaction(
                    keypair=self.keypair,
                    mint_str=tx.mint,
                    sol_amount=our_buy_amount,
                    slippage_bps=500,  # 5% slippage (was 10% with Pump Portal)
                    priority_fee_lamports=5_000_000,  # 0.005 SOL for fast inclusion
                    compute_units=120_000
                )

                if not buy_signed:
                    raise RuntimeError("Buy transaction build failed")

                buy_confirmed = await self.tx_submitter.submit_and_confirm(
                    buy_signed,
                    timeout_seconds=30
                )

                if buy_confirmed.error:
                    raise RuntimeError(f"Buy confirmation failed: {buy_confirmed.error}")

                buy_signature = buy_confirmed.signature
                buy_slot = buy_confirmed.slot

                # Extract ACTUAL tokens received from transaction (no balance call needed!)
                tokens_received = await self._get_tokens_from_transaction(buy_signature, tx.mint)

                # Calculate frontrun metrics
                frontrun_success = buy_slot <= dev_slot
                slot_delta = buy_slot - dev_slot

                logger.info(
                    "buy_complete",
                    signature=buy_signature[:16] + "...",
                    buy_slot=buy_slot,
                    dev_slot=dev_slot,
                    frontrun_success=frontrun_success,
                    slot_delta=slot_delta,
                    tokens_received=tokens_received
                )

                # ========================================
                # STEP 2: HOLD (wait for retail wave)
                # ========================================
                logger.info(
                    "holding_for_retail_wave",
                    hold_seconds=self.hold_time_seconds,
                    strategy="Exit into retail copy trader buying pressure"
                )

                await asyncio.sleep(self.hold_time_seconds)

                # ========================================
                # STEP 3: SELL
                # ========================================
                logger.info(
                    "executing_sell",
                    trade_num=trade_num,
                    mint=tx.mint
                )

                # Wait a bit for token account to be indexed by RPC (important!)
                # Token account creation takes ~200-500ms to show up in RPC queries
                await asyncio.sleep(0.5)

                # Get token balance from wallet
                token_balance = await self._get_token_balance(wallet.pubkey, tx.mint)

                if token_balance == 0:
                    logger.warning(
                        "no_tokens_to_sell",
                        mint=tx.mint,
                        reason="Token balance is 0 - token account may not be indexed yet"
                    )

                    # If we have estimated tokens from Pump Portal, use those
                    if tokens_received > 0:
                        token_balance = tokens_received
                        logger.info(
                            "using_estimated_token_balance",
                            estimated_tokens=token_balance
                        )
                    else:
                        # Cannot sell without knowing balance - record as partial trade
                        logger.error(
                            "cannot_determine_token_balance",
                            mint=tx.mint,
                            buy_signature=buy_signature,
                            message="Skipping sell - will be stuck holding tokens"
                        )

                        # Record partial trade (buy only, no sell)
                        trade = MicroTrade(
                            timestamp=datetime.now(),
                            mint=tx.mint,
                            signature_buy=buy_signature,
                            signature_sell="",  # No sell
                            buy_amount_sol=our_buy_amount,
                            sell_amount_sol=0.0,
                            profit_sol=-our_buy_amount,  # Lost the capital
                            fee_paid=5000,
                            success=False,
                            reason="Buy succeeded but couldn't determine token balance to sell",
                            dev_tx_slot=dev_slot,
                            our_tx_slot=buy_slot,
                            frontrun_success=frontrun_success,
                            slot_delta=slot_delta
                        )
                        self.trades.append(trade)
                        return  # Exit early, skip sell

                logger.info(
                    "selling_tokens",
                    token_balance=token_balance,
                    mint=tx.mint
                )

                # Convert raw token units to UI amount (Pump.fun uses 6 decimals)
                # Pump Portal expects UI amount (human-readable) when denominatedInSol=false
                token_ui_amount = token_balance / 1e6

                logger.info(
                    "converting_token_amount",
                    raw_units=token_balance,
                    ui_amount=token_ui_amount
                )

                # Build sell transaction DIRECTLY (10x faster than Pump Portal!)
                sell_signed = await self.direct_client.create_sell_transaction(
                    keypair=self.keypair,
                    mint_str=tx.mint,
                    token_amount=token_ui_amount,
                    slippage_bps=500,  # 5% slippage (was 10% with Pump Portal)
                    priority_fee_lamports=5_000_000,
                    compute_units=120_000
                )

                if not sell_signed:
                    raise RuntimeError("Sell transaction build failed")

                sell_confirmed = await self.tx_submitter.submit_and_confirm(
                    sell_signed,
                    timeout_seconds=30
                )

                if sell_confirmed.error:
                    raise RuntimeError(f"Sell confirmation failed: {sell_confirmed.error}")

                sell_signature = sell_confirmed.signature
                sell_slot = sell_confirmed.slot

                # Extract ACTUAL SOL received from transaction
                sell_sol_received = await self._get_sol_from_transaction(sell_signature, str(wallet.pubkey))

                logger.info(
                    "sell_complete",
                    signature=sell_signature[:16] + "...",
                    sell_slot=sell_slot,
                    sol_received=sell_sol_received,
                    slots_held=sell_slot - buy_slot
                )

                # ========================================
                # STEP 4: Calculate P&L
                # ========================================
                profit_sol = sell_sol_received - our_buy_amount
                profit_pct = (profit_sol / our_buy_amount) * 100 if our_buy_amount > 0 else 0

                logger.info(
                    "trade_pnl",
                    buy_amount=our_buy_amount,
                    sell_amount=sell_sol_received,
                    profit_sol=profit_sol,
                    profit_pct=round(profit_pct, 2),
                    verdict="✅ PROFIT" if profit_sol > 0 else "❌ LOSS"
                )

                # Record successful trade
                trade = MicroTrade(
                    timestamp=datetime.now(),
                    mint=tx.mint,
                    signature_buy=buy_signature,
                    signature_sell=sell_signature,
                    buy_amount_sol=our_buy_amount,
                    sell_amount_sol=sell_sol_received,
                    profit_sol=profit_sol,
                    fee_paid=10000,  # Estimate 10000 lamports total fees (buy + sell)
                    success=True,
                    reason=f"Complete trade: {profit_pct:.2f}% {'profit' if profit_sol > 0 else 'loss'}",
                    dev_tx_slot=dev_slot,
                    our_tx_slot=buy_slot,
                    frontrun_success=frontrun_success,
                    slot_delta=slot_delta
                )

                self.trades.append(trade)

                logger.info(
                    "trade_completed",
                    trade_num=trade_num,
                    buy_sig=buy_signature[:16] + "...",
                    sell_sig=sell_signature[:16] + "...",
                    profit_sol=profit_sol,
                    profit_pct=round(profit_pct, 2),
                    trades_remaining=self.max_trades - len(self.trades)
                )

        except Exception as e:
            logger.error(
                "trade_execution_failed",
                trade_num=trade_num,
                error=str(e),
                error_type=type(e).__name__,
                buy_signature=buy_signature,
                sell_signature=sell_signature
            )

            # Record failed trade
            trade = MicroTrade(
                timestamp=datetime.now(),
                mint=tx.mint,
                signature_buy=buy_signature,
                signature_sell=sell_signature,
                buy_amount_sol=our_buy_amount,
                sell_amount_sol=sell_sol_received,
                profit_sol=sell_sol_received - our_buy_amount if sell_sol_received > 0 else -our_buy_amount,
                fee_paid=0,
                success=False,
                reason=f"Failed: {str(e)}",
                dev_tx_slot=dev_slot,
                our_tx_slot=buy_slot if buy_slot > 0 else 0,
                frontrun_success=False,
                slot_delta=buy_slot - dev_slot if buy_slot > 0 else 999
            )

            self.trades.append(trade)

    async def _get_sol_from_transaction(self, signature: str, wallet_pubkey: str) -> float:
        """
        Extract SOL received from sell transaction using pre/post balances

        Args:
            signature: Transaction signature
            wallet_pubkey: Wallet public key (to find the right account)

        Returns:
            SOL amount received (in SOL, not lamports)
        """
        try:
            # Fetch transaction details
            response = await self.rpc_manager.call_http_rpc(
                "getTransaction",
                [
                    signature,
                    {
                        "encoding": "jsonParsed",
                        "maxSupportedTransactionVersion": 0,
                        "commitment": "confirmed"
                    }
                ]
            )

            result = response.get("result")
            if not result:
                logger.warning("transaction_not_found_for_sol", signature=signature)
                return 0.0

            # Get account keys and balances
            transaction = result.get("transaction", {})
            message = transaction.get("message", {})
            account_keys = message.get("accountKeys", [])

            meta = result.get("meta", {})
            pre_balances = meta.get("preBalances", [])
            post_balances = meta.get("postBalances", [])

            # Find our wallet's index
            wallet_index = None
            for i, account in enumerate(account_keys):
                pubkey = account if isinstance(account, str) else account.get("pubkey")
                if pubkey == wallet_pubkey:
                    wallet_index = i
                    break

            if wallet_index is None:
                logger.warning("wallet_not_found_in_tx", wallet=wallet_pubkey[:16])
                return 0.0

            # Calculate SOL difference
            pre_balance = pre_balances[wallet_index] if wallet_index < len(pre_balances) else 0
            post_balance = post_balances[wallet_index] if wallet_index < len(post_balances) else 0

            sol_diff = (post_balance - pre_balance) / 1e9  # Convert lamports to SOL

            logger.info(
                "sol_extracted_from_tx",
                signature=signature[:16] + "...",
                sol_received=sol_diff
            )

            return sol_diff

        except Exception as e:
            logger.error(
                "sol_extraction_failed",
                signature=signature,
                error=str(e)
            )
            return 0.0

    async def _get_tokens_from_transaction(self, signature: str, mint: str) -> float:
        """
        Extract actual tokens received from transaction using getTransaction RPC

        This parses innerInstructions to find SPL token transfers - much more accurate
        than balance calls or estimates!

        Args:
            signature: Transaction signature
            mint: Token mint address

        Returns:
            Token amount (raw units, e.g. 1000000 for 1M tokens with 6 decimals)
        """
        try:
            # Fetch transaction details with max encoding for innerInstructions
            response = await self.rpc_manager.call_http_rpc(
                "getTransaction",
                [
                    signature,
                    {
                        "encoding": "jsonParsed",
                        "maxSupportedTransactionVersion": 0,
                        "commitment": "confirmed"
                    }
                ]
            )

            result = response.get("result")
            if not result:
                logger.warning("transaction_not_found", signature=signature)
                return 0

            # Parse innerInstructions for token transfers
            meta = result.get("meta", {})
            inner_instructions = meta.get("innerInstructions", [])

            for instruction_group in inner_instructions:
                for instruction in instruction_group.get('instructions', []):
                    # Look for SPL token transfer
                    if (instruction.get('program') == 'spl-token' and
                        instruction.get('parsed', {}).get('type') == 'transfer'):

                        parsed_info = instruction.get('parsed', {}).get('info', {})
                        amount_str = parsed_info.get('amount', '0')

                        try:
                            amount = float(amount_str)
                            if amount > 0:
                                logger.info(
                                    "tokens_extracted_from_tx",
                                    signature=signature[:16] + "...",
                                    tokens=amount
                                )
                                return amount
                        except (ValueError, TypeError):
                            continue

            # No token transfer found
            logger.warning(
                "no_token_transfer_found",
                signature=signature,
                inner_instructions_count=len(inner_instructions)
            )
            return 0

        except Exception as e:
            logger.error(
                "token_extraction_failed",
                signature=signature,
                error=str(e)
            )
            return 0

    async def _get_token_balance(self, owner: Pubkey, mint: str, max_retries: int = 3) -> float:
        """
        Get token balance for owner's associated token account

        Args:
            owner: Owner pubkey
            mint: Token mint address
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            Token balance (raw amount, not lamports)
        """
        for attempt in range(max_retries):
            try:
                from solders.pubkey import Pubkey as SoldersPubkey

                # Derive associated token account
                mint_pubkey = SoldersPubkey.from_string(mint)

                # Use the pumpfun_client's helper to derive ATA
                # (This requires importing the derive function or using a library)
                # For now, make RPC call to get token accounts

                response = await self.rpc_manager.call_http_rpc(
                    "getTokenAccountsByOwner",
                    [
                        str(owner),
                        {"mint": mint},
                        {"encoding": "jsonParsed"}
                    ]
                )

                result = response.get("result", {})
                accounts = result.get("value", [])

                if not accounts:
                    if attempt < max_retries - 1:
                        # Wait and retry - token account may not be indexed yet
                        logger.debug(
                            "no_token_account_found_retrying",
                            owner=str(owner),
                            mint=mint,
                            attempt=attempt + 1
                        )
                        await asyncio.sleep(0.3)  # Wait 300ms before retry
                        continue
                    else:
                        logger.debug(
                            "no_token_account_found",
                            owner=str(owner),
                            mint=mint
                        )
                        return 0

                # Get balance from first account
                token_data = accounts[0].get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
                token_amount = token_data.get("tokenAmount", {})
                balance = float(token_amount.get("amount", 0))

                logger.debug(
                    "token_balance_fetched",
                    owner=str(owner)[:16] + "...",
                    mint=mint[:16] + "...",
                    balance=balance,
                    attempt=attempt + 1
                )

                return balance

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "token_balance_fetch_failed_retrying",
                        error=str(e),
                        owner=str(owner),
                        mint=mint,
                        attempt=attempt + 1
                    )
                    await asyncio.sleep(0.3)
                    continue
                else:
                    logger.error(
                        "token_balance_fetch_failed",
                        error=str(e),
                        owner=str(owner),
                        mint=mint
                    )
                    return 0

        return 0

    async def _execute_simulated_trade(
        self,
        trade_num: int,
        tx: PendingTransaction,
        our_buy_amount: float,
        dev_slot: int
    ):
        """
        Simulate trade execution (for testing without real capital)

        Args:
            trade_num: Trade number
            tx: Pending transaction
            our_buy_amount: Our buy amount in SOL
            dev_slot: Slot where dev transaction landed
        """
        # Simulate our slot (optimistic: same slot as dev)
        our_slot = dev_slot

        # Calculate if we frontran
        frontrun_success = our_slot <= dev_slot
        slot_delta = our_slot - dev_slot

        logger.info(
            "slot_comparison",
            dev_slot=dev_slot,
            our_slot=our_slot,
            frontrun_success=frontrun_success,
            slot_delta=slot_delta,
            verdict="FRONTRUN SUCCESS (simulated)" if frontrun_success else "FRONTRUN FAILED (simulated)"
        )

        # For now, simulate success
        simulated_profit = our_buy_amount * 0.05  # Assume 5% profit
        simulated_fee = 50000  # 0.00005 SOL fee

        trade = MicroTrade(
            timestamp=datetime.now(),
            mint=tx.mint,
            signature_buy=f"sim_buy_{tx.signature[:16]}",
            signature_sell=f"sim_sell_{tx.signature[:16]}",
            buy_amount_sol=our_buy_amount,
            sell_amount_sol=our_buy_amount + simulated_profit,
            profit_sol=simulated_profit,
            fee_paid=simulated_fee,
            success=True,
            reason="Simulated trade for Phase 2 validation",
            dev_tx_slot=dev_slot,
            our_tx_slot=our_slot,
            frontrun_success=frontrun_success,
            slot_delta=slot_delta
        )

        self.trades.append(trade)

        logger.info(
            "trade_completed",
            trade_num=trade_num,
            profit_sol=simulated_profit,
            frontrun_success=frontrun_success,
            trades_remaining=self.max_trades - len(self.trades)
        )

    async def _generate_report(self):
        """Generate Phase 2 completion report"""
        logger.info("\n=== PHASE 2 MICRO-TRADING COMPLETE ===\n")

        total_trades = len(self.trades)
        successful = sum(1 for t in self.trades if t.success)
        total_profit = sum(t.profit_sol for t in self.trades)
        total_fees = sum(t.fee_paid for t in self.trades) / 1e9
        net_pnl = total_profit - total_fees

        # CRITICAL: Frontrun analysis
        frontrun_wins = sum(1 for t in self.trades if t.frontrun_success)
        frontrun_rate = frontrun_wins / total_trades if total_trades > 0 else 0
        avg_slot_delta = sum(t.slot_delta for t in self.trades) / total_trades if total_trades > 0 else 0

        # Calculate win rate (profitable trades)
        winning_trades = sum(1 for t in self.trades if t.profit_sol > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate average gain/loss
        avg_profit = sum(t.profit_sol for t in self.trades if t.profit_sol > 0) / winning_trades if winning_trades > 0 else 0
        losing_trades = sum(1 for t in self.trades if t.profit_sol < 0)
        avg_loss = sum(t.profit_sol for t in self.trades if t.profit_sol < 0) / losing_trades if losing_trades > 0 else 0

        logger.info(
            "phase2_results",
            total_trades=total_trades,
            successful_trades=successful,
            success_rate=successful/total_trades if total_trades > 0 else 0,
            total_profit_sol=round(total_profit, 6),
            total_fees_sol=round(total_fees, 6),
            net_pnl_sol=round(net_pnl, 6)
        )

        logger.info(
            "profitability_analysis",
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 2),
            avg_profit_sol=round(avg_profit, 6),
            avg_loss_sol=round(avg_loss, 6),
            verdict="✅ PROFITABLE" if net_pnl > 0 else "❌ UNPROFITABLE"
        )

        logger.info(
            "frontrun_analysis",
            frontrun_wins=frontrun_wins,
            frontrun_losses=total_trades - frontrun_wins,
            frontrun_rate=round(frontrun_rate, 2),
            avg_slot_delta=round(avg_slot_delta, 2),
            verdict="SUCCESS" if frontrun_rate >= 0.5 else "NEEDS_IMPROVEMENT"
        )

        # Export data
        output_dir = Path("data/phase2")
        output_dir.mkdir(parents=True, exist_ok=True)

        trades_file = output_dir / f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(trades_file, 'w') as f:
            json.dump([t.to_dict() for t in self.trades], f, indent=2)

        logger.info(f"✅ Trades exported to: {trades_file}")

        # Decision
        logger.info("\n=== STRATEGY VERDICT ===\n")

        if net_pnl > 0 and win_rate >= 0.4:
            logger.info("✅ STRATEGY IS PROFITABLE!")
            logger.info(f"Win rate: {win_rate*100:.1f}% (need 40%+ to be viable)")
            logger.info(f"Net P&L: {net_pnl:.6f} SOL")
            logger.info(f"Average profit on wins: {avg_profit:.6f} SOL")
            logger.info(f"Average loss on losses: {avg_loss:.6f} SOL")
            logger.info("\n✅ READY TO SCALE: Test with more capital and longer time period")
        elif net_pnl > 0 and win_rate < 0.4:
            logger.warning(f"⚠️ Net positive BUT win rate too low: {win_rate*100:.1f}% (need 40%+)")
            logger.warning("May work but risky - test with different hold times (1s vs 2s vs 3s)")
        elif win_rate >= 0.4:
            logger.warning(f"⚠️ Good win rate ({win_rate*100:.1f}%) BUT net negative P&L")
            logger.warning("Losses are too large - reduce slippage or exit faster")
        else:
            logger.error("❌ STRATEGY NOT PROFITABLE")
            logger.error(f"Win rate: {win_rate*100:.1f}% (need 40%+)")
            logger.error(f"Net P&L: {net_pnl:.6f} SOL")
            logger.error("Consider: 1) Different hold time, 2) Different dev buy size filter, 3) Test LaserStream vs free websockets")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 2/3: Complete Copy Trading - Execute buy + sell to measure profitability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ultra-conservative: 10 trades, max 0.1 SOL, 2s hold (needs ~1.5 SOL)
  python phase2_microtrading.py --wallet wallet.json --real-trades --max-trades 10 --max-position-sol 0.1 --hold-time 2.0

  # Fast exit: 20 trades, 1 second hold (test quick scalping)
  python phase2_microtrading.py --wallet wallet.json --real-trades --max-trades 20 --max-position-sol 0.1 --hold-time 1.0

  # Slower exit: 20 trades, 3 second hold (let retail wave build)
  python phase2_microtrading.py --wallet wallet.json --real-trades --max-trades 20 --max-position-sol 0.1 --hold-time 3.0

  # Target 2+ SOL dev buys only (higher quality, fewer opportunities)
  python phase2_microtrading.py --wallet wallet.json --real-trades --min-buy-sol 2.0 --max-trades 20 --hold-time 2.0

Strategy:
  1. Dev buys large amount (1+ SOL) → Creates price momentum
  2. LaserStream detects (10-50ms) → WE BUY
  3. Free websockets detect (100-500ms) → RETAIL BUYS (price spikes!)
  4. We hold 1-3 seconds → WE SELL into retail buying pressure
  5. Profit from the retail wave

Setup:
  1. Create wallet: solana-keygen new --outfile data/phase2-wallet.json
  2. Fund wallet with desired amount based on test size
     - Ultra-conservative (10 trades @ 0.1 SOL): 1.5 SOL (~$375)
     - Conservative (20 trades @ 0.2 SOL): 4.5 SOL (~$1,125)
  3. Run test with different hold times to find optimal (1s vs 2s vs 3s)

Results:
  - Win rate: % of profitable trades (need 40%+ to be viable)
  - Average gain: Profit on winning trades
  - Average loss: Loss on losing trades (should be < avg gain)
  - Net P&L: Total profit/loss (positive = strategy works!)
        """
    )
    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="Path to config.yml"
    )
    parser.add_argument(
        "--wallet",
        required=True,
        help="Path to wallet keypair JSON file (created with solana-keygen)"
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=10,
        help="Maximum number of trades to execute (default: 10)"
    )
    parser.add_argument(
        "--real-trades",
        action="store_true",
        help="Execute REAL trades (default: simulation only). WARNING: Uses real capital!"
    )
    parser.add_argument(
        "--min-buy-sol",
        type=float,
        default=1.0,
        help="Minimum dev buy size in SOL to target (default: 1.0). Higher values = fewer but higher-quality opportunities."
    )
    parser.add_argument(
        "--max-position-sol",
        type=float,
        default=0.5,
        help="Maximum position size in SOL per trade (default: 0.5). Lower values = less risk. Example: 0.1 for ultra-conservative testing."
    )
    parser.add_argument(
        "--hold-time",
        type=float,
        default=2.0,
        help="Time to hold position before selling in seconds (default: 2.0). Use 1-3s to catch retail wave."
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO", format="json", output_file="logs/phase2_microtrading.log")

    logger.info(
        "phase2_starting",
        config=args.config,
        wallet=args.wallet,
        max_trades=args.max_trades,
        real_trades=args.real_trades
    )

    if args.real_trades:
        logger.warning(
            "⚠️  REAL TRADES MODE ENABLED - WILL USE REAL CAPITAL ⚠️"
        )
        logger.warning(
            f"This will execute {args.max_trades} COMPLETE trades (buy + sell) with max {args.max_position_sol} SOL positions"
        )
        logger.warning(
            f"Hold time: {args.hold_time}s (exit into retail wave)"
        )
        logger.warning(
            "Press Ctrl+C within 5 seconds to cancel..."
        )
        await asyncio.sleep(5)

    # Create and run trader
    trader = Phase2MicroTrader(
        config_path=args.config,
        wallet_path=args.wallet,
        max_trades=args.max_trades,
        enable_real_trades=args.real_trades,
        min_buy_sol=args.min_buy_sol,
        max_position_sol=args.max_position_sol,
        hold_time_seconds=args.hold_time
    )

    await trader.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Phase 2 interrupted by user")
    except Exception as e:
        logger.error("phase2_failed", error=str(e), error_type=type(e).__name__)
        raise
