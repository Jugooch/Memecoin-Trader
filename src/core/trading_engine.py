"""
Trading engine for executing buy/sell orders and managing positions
"""

import asyncio
import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.clients.pumpfun_client import PumpFunClient
from src.clients.moralis_client import MoralisClient
from src.clients.transaction_signer import TransactionSigner
from src.clients.blockchain_analytics import BlockchainAnalytics
from src.utils.discord_notifier import DiscordNotifier
from src.utils.pnl_store import PnLStore
from .realtime_position_manager import RealtimePositionManager
from .pool_calculator import PoolCalculator


@dataclass
class Position:
    mint: str
    symbol: str  # Token name/symbol for display
    entry_price: float
    amount: float
    sol_invested: float
    entry_time: datetime
    tp_price: float
    sl_price: float
    peak_price: float
    paper_mode: bool
    tokens_initial: float
    cost_usd_remaining: float
    avg_cost_per_token: float
    tp1_hit_time: Optional[datetime] = None  # Track when TP1 was hit
    tp1_percentage_sold: float = 0  # Track how much was sold at TP1
    tp2_hit_time: Optional[datetime] = None  # Track when TP2 was hit (50%)
    tp2_percentage_sold: float = 0  # Track how much was sold at TP2
    tp3_hit_time: Optional[datetime] = None  # Track when TP3 was hit (100%)
    tp3_percentage_sold: float = 0  # Track how much was sold at TP3
    break_even_armed: bool = False  # Track if break-even stop is armed
    break_even_armed_time: Optional[datetime] = None  # When break-even was armed
    trailing_stop_active: bool = False  # Track if trailing stop is active
    high_gain_peak: float = 0  # Track highest gain percentage achieved
    is_selling: bool = False  # Prevent race conditions in sell operations
    
    # Blockchain verification data (for live trading accuracy)
    buy_tx_signature: Optional[str] = None  # Source of truth for cost basis
    verified_from_blockchain: bool = False   # True if data comes from tx logs
    
    # Smart reconciliation system
    needs_reconciliation: bool = False  # Flag for position balance reconciliation
    pre_reconcile_state: Optional[dict] = None  # Store TP state before reconciliation
    
    # Buffer strategy fields
    buffer_period_seconds: int = 0  # Volatility buffer period (no stop losses)
    buffer_end_time: Optional[datetime] = None  # When buffer period ends


class TradingEngine:
    def __init__(self, config, moralis_client: MoralisClient):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients - Pump Portal Local API doesn't need API key
        self.pumpfun = PumpFunClient(config.quicknode_endpoint, config.quicknode_api_key)
        
        self.logger.info("✅ Pump Portal Local Trading API client initialized (no API key needed)")
        self.moralis = moralis_client  # Use shared client instead of creating new one
        
        # Initialize transaction signer for live trading (using wallet keys from pumpportal)
        self.transaction_signer = None

        # Initialize fast transaction submitter if fast execution is enabled
        self.fast_transaction_submitter = None
        self.fast_execution_enabled = False

        if hasattr(config, 'fast_execution') and config.fast_execution:
            if config.fast_execution.get('enabled', False):
                try:
                    from src.core.fast_transaction_submitter import FastTransactionSubmitter
                    self.fast_transaction_submitter = FastTransactionSubmitter(config.__dict__)
                    self.fast_execution_enabled = True
                    self.logger.info("✅ Fast execution enabled (Helius + Jito)")
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize fast execution: {e}")
                    self.fast_execution_enabled = False

        # Debug logging
        self.logger.info(f"Checking transaction signer initialization...")
        self.logger.info(f"Has pumpportal config: {hasattr(config, 'pumpportal') and config.pumpportal is not None}")
        self.logger.info(f"QuickNode endpoint configured: {bool(getattr(config, 'quicknode_endpoint', None))}")
        
        if hasattr(config, 'pumpportal') and config.pumpportal:
            wallet_private_key = config.pumpportal.get('wallet_private_key')
            wallet_public_key = config.pumpportal.get('wallet_public_key')
            
            self.logger.info(f"Wallet private key found: {bool(wallet_private_key)}")
            self.logger.info(f"Wallet public key found: {bool(wallet_public_key)}")
            
            if wallet_private_key and config.quicknode_endpoint:
                try:
                    self.transaction_signer = TransactionSigner(
                        quicknode_endpoint=config.quicknode_endpoint,
                        quicknode_api_key=config.quicknode_api_key,
                        private_key_base58=wallet_private_key
                    )
                    self.logger.info(f"✅ Live trading wallet initialized: {self.transaction_signer.get_wallet_address()[:8]}...")
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize transaction signer: {e}")
                    self.transaction_signer = None
            else:
                missing = []
                if not wallet_private_key:
                    missing.append("wallet_private_key")
                if not config.quicknode_endpoint:
                    missing.append("quicknode_endpoint")
                self.logger.error(f"❌ Transaction signer not initialized - missing: {', '.join(missing)}")
        else:
            self.logger.error(f"❌ No pumpportal configuration found for live trading")
        
        # Initialize Discord notifier
        webhook_url = None

        # Try multiple config locations for webhook URL
        if hasattr(config, 'discord') and config.discord and isinstance(config.discord, dict):
            # Standard location: discord.webhook_url
            webhook_url = config.discord.get('webhook_url', None)
            if webhook_url:
                self.logger.info(f"DISCORD_DEBUG: Found discord.webhook_url: {webhook_url[:50]}...")
        elif hasattr(config, 'notifications') and config.notifications:
            # Legacy location: notifications.discord_webhook_url
            webhook_url = config.notifications.get('discord_webhook_url', None)
            if webhook_url:
                self.logger.info(f"DISCORD_DEBUG: Found notifications.discord_webhook_url: {webhook_url[:50]}...")
        elif hasattr(config, 'discord_webhook_url'):
            # Root level location
            webhook_url = config.discord_webhook_url
            if webhook_url:
                self.logger.info(f"DISCORD_DEBUG: Found root discord_webhook_url: {webhook_url[:50]}...")

        if not webhook_url:
            self.logger.warning("DISCORD_DEBUG: No Discord webhook URL found in config")
            self.logger.info(f"DISCORD_DEBUG: Checked discord, notifications, and root level")

        # Check if webhook URL is actually set and not empty
        if webhook_url and webhook_url.strip():
            self.notifier = DiscordNotifier(webhook_url.strip())
            self.logger.info(f"✅ Discord notifier initialized")
        else:
            self.notifier = None
            self.logger.warning(f"⚠️ Discord notifier not initialized - no webhook URL configured")
        
        # Initialize P&L store
        self.pnl_store = PnLStore(
            path="data/pnl_state.json",
            initial_capital=config.initial_capital
        )
        
        # Initialize blockchain analytics for SOL price and P&L tracking
        # Uses Helius RPC (if fast execution enabled) or QuickNode RPC
        # Falls back to SOL price only if no RPC configured
        self.blockchain_analytics = None
        try:
            rpc_endpoint = None
            api_key = None
            rpc_source = None

            # Prefer Helius RPC if fast execution is enabled
            if self.fast_execution_enabled and hasattr(config, 'fast_execution'):
                helius_rpc = config.fast_execution.get('helius_rpc_endpoint')
                if helius_rpc:
                    rpc_endpoint = helius_rpc
                    rpc_source = "Helius"
                    # Helius includes API key in URL, no separate key needed
                    api_key = None

            # Fallback to QuickNode if no Helius or fast execution disabled
            if not rpc_endpoint and config.quicknode_endpoint:
                rpc_endpoint = config.quicknode_endpoint
                api_key = config.quicknode_api_key if hasattr(config, 'quicknode_api_key') else None
                rpc_source = "QuickNode"

            # Always initialize (for SOL price at minimum)
            if not rpc_endpoint:
                rpc_endpoint = "https://placeholder"  # Placeholder - won't be used for RPC calls
                rpc_source = None

            self.blockchain_analytics = BlockchainAnalytics(
                rpc_endpoint=rpc_endpoint,
                api_key=api_key
            )

            if rpc_source:
                self.logger.info(f"✅ Blockchain analytics initialized with full RPC support ({rpc_source})")
            else:
                self.logger.info("✅ Blockchain analytics initialized (SOL price only, no RPC)")
        except Exception as e:
            self.logger.warning(f"⚠️ Blockchain analytics not available: {e}")
        
        # Position tracking - OLD SYSTEM (will be phased out)
        self.active_positions = {}
        self.paper_capital = self.pnl_store.current_equity  # Use P&L store's equity
        self.total_trades = 0
        self.winning_trades = 0
        
        # NEW: Real-time position tracking system
        self.realtime_positions = RealtimePositionManager(self.logger)
        
        # Set up callbacks for position events
        self.realtime_positions.on_position_opened = self._on_realtime_position_opened
        self.realtime_positions.on_position_updated = self._on_realtime_position_updated  
        self.realtime_positions.on_position_closed = self._on_realtime_position_closed
        
        # Feature flag for gradual transition
        self.use_realtime_positions = getattr(config, 'use_realtime_positions', False)
        
        # Pool calculator for immediate position creation
        self.pool_calculator = PoolCalculator()
        
        # Hybrid safety configuration
        self.safety_hybrid = getattr(config, 'safety_hybrid', {})
        self.use_hybrid_safety = self.safety_hybrid.get('enabled', False)
    
    def _on_realtime_position_opened(self, position):
        """Callback when a new position is opened via realtime tracking"""
        self.logger.info(f"📈 Realtime position opened: {position.mint[:8]}... {position.current_tokens:,.0f} tokens")
        
        # Send buy notification to Discord after WSS confirmation
        if hasattr(self, 'notifier') and self.notifier:
            # Use current price for USD value estimate  
            usd_value = position.current_tokens * position.entry_price if position.entry_price else 0
            
            # If symbol is dummy from ultra fast execution, fetch real metadata now
            if position.symbol == 'UF':
                asyncio.create_task(self._send_notification_with_real_metadata(
                    position, usd_value
                ))
            else:
                # Use existing symbol for immediate notification
                asyncio.create_task(self.notifier.send_trade_notification(
                    side="BUY",
                    symbol=position.symbol,  # Use proper token symbol
                    mint_address=position.mint,
                    quantity=position.current_tokens,
                    price=position.entry_price,
                    usd_amount=usd_value,
                    equity=self.pnl_store.current_equity,
                    paper_mode=False
                ))
    
    async def _send_notification_with_real_metadata(self, position, usd_value):
        """Fetch real metadata and send Discord notification with proper symbol"""
        try:
            # Fetch real metadata from Moralis
            self.logger.info(f"🔍 Fetching real metadata for {position.mint[:8]}...")
            metadata = await self.moralis.get_token_metadata(position.mint)
            
            if metadata and metadata.get('symbol'):
                real_symbol = metadata['symbol']
                self.logger.info(f"✅ Updated symbol: {position.mint[:8]}... UF → {real_symbol}")
                
                # Update the position's symbol for future use
                position.symbol = real_symbol
                
                # Also update main position if it exists
                if position.mint in self.active_positions:
                    self.active_positions[position.mint].symbol = real_symbol
                    self.logger.debug(f"📋 Updated active position symbol: {position.mint[:8]}... → {real_symbol}")
                
            else:
                real_symbol = f"{position.mint[:8]}..."  # Fallback to mint address
                self.logger.warning(f"⚠️ Could not fetch metadata for {position.mint[:8]}..., using mint address")
            
            # Send Discord notification with real symbol
            await self.notifier.send_trade_notification(
                side="BUY",
                symbol=real_symbol,
                mint_address=position.mint,
                quantity=position.current_tokens,
                price=position.entry_price,
                usd_amount=usd_value,
                equity=self.pnl_store.current_equity,
                paper_mode=False
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching metadata for {position.mint[:8]}...: {e}")
            # Send notification with fallback symbol
            await self.notifier.send_trade_notification(
                side="BUY",
                symbol=f"{position.mint[:8]}...",
                mint_address=position.mint,
                quantity=position.current_tokens,
                price=position.entry_price,
                usd_amount=usd_value,
                equity=self.pnl_store.current_equity,
                paper_mode=False
            )
        
    def _on_realtime_position_updated(self, position):
        """Callback when a position is updated via realtime tracking"""
        pnl_sol = position.unrealized_pnl_sol
        self.logger.info(f"📊 Realtime position updated: {position.mint[:8]}... P&L: {pnl_sol:+.4f} SOL")
        
    def _on_realtime_position_closed(self, position):
        """Callback when a position is closed via realtime tracking"""
        pnl_sol = position.unrealized_pnl_sol
        self.logger.info(f"🏁 Realtime position closed: {position.mint[:8]}... Final P&L: {pnl_sol:+.4f} SOL")
        
        # Update trading statistics
        self.total_trades += 1
        if pnl_sol > 0:
            self.winning_trades += 1
    
    async def handle_self_trade_event(self, trade_event: Dict) -> None:
        """
        🚨 CRITICAL: WSS event triggers QuickNode verification and position creation.
        This is the ONLY way we know when our transaction went through!
        """
        try:
            mint = trade_event.get('mint')
            action = trade_event.get('action')
            tx_signature = trade_event.get('tx_signature')
            
            self.logger.info(f"⚡ WSS detected our {action}: {mint[:8]}... TX: {tx_signature[:16]}...")
            
            if action == 'buy':
                # This is the trigger! WSS detected our buy went through
                self.logger.info(f"🚀 WSS buy trigger received for {mint[:8]}...")
                
                # Get stored transaction data (prevents duplicates and ensures accurate pricing)
                if not hasattr(self, '_pending_transactions'):
                    self._pending_transactions = {}
                
                tx_data = self._pending_transactions.get(tx_signature)
                if not tx_data:
                    self.logger.warning(f"⚠️ No pending transaction data for {tx_signature[:16]}... - ignoring WSS event")
                    return
                
                # Remove from pending (prevents duplicate processing)
                del self._pending_transactions[tx_signature]
                
                self.logger.info(f"🚀 WSS buy trigger - starting verification for {tx_data['symbol']}")
                
                # Start reconciliation (verify and adjust if needed)
                if tx_data.get('has_immediate_position'):
                    asyncio.create_task(self._reconcile_position(
                        mint_address=tx_data['mint_address'],
                        tx_signature=tx_signature,
                        symbol=tx_data['symbol']
                    ))
                else:
                    # Fallback to old verification flow if no immediate position
                    asyncio.create_task(self._create_verified_position(
                        mint_address=tx_data['mint_address'],
                        tx_signature=tx_signature,
                        sol_amount=tx_data['sol_amount'],
                        usd_amount=tx_data['usd_amount'],
                        symbol=tx_data['symbol']
                    ))
                
            elif action == 'sell':
                # Handle sell confirmations if needed
                self.logger.info(f"✅ WSS confirmed sell: {mint[:8]}...")
                
        except Exception as e:
            self.logger.error(f"Error handling WSS trade event: {e}")
            self.logger.error(f"Event data: {trade_event}")
    
    async def _verify_transaction_amounts(self, tx_signature: str, mint: str, action: str) -> Dict:
        """Verify transaction amounts using live account balance - fast and accurate"""
        try:
            if not self.transaction_signer and not (self.fast_execution_enabled and self.fast_transaction_submitter):
                return {}

            verified_amounts = {}

            if action == 'buy':
                # Get current token balance from live account data
                # Support both fast execution and legacy paths
                if self.fast_execution_enabled and self.fast_transaction_submitter:
                    current_balance = await self.fast_transaction_submitter.get_token_balance(mint)
                elif self.transaction_signer:
                    current_balance = await self.transaction_signer.get_token_balance(mint)
                else:
                    current_balance = None
                if current_balance and current_balance > 0:
                    verified_amounts['tokens_received'] = current_balance
                    self.logger.info(f"✅ Verified buy: {current_balance:,.0f} tokens from live balance")
                else:
                    self.logger.warning(f"⚠️ Token balance query returned {current_balance} for {mint[:8]}...")
                    # Log more details about the failure
                    self.logger.warning(f"  Mint: {mint}")
                    self.logger.warning(f"  TX: {tx_signature[:16]}...")
                    self.logger.warning(f"  This likely means QuickNode hasn't indexed the ATA yet")
                    
            elif action == 'sell':
                # For sells, we'll track the token balance change
                # For now, just log that we detected a sell
                self.logger.info(f"✅ Verified sell: {mint[:8]}... (balance tracking)")
                # Note: Sell verification can be enhanced later with before/after balance comparison
                
            return verified_amounts
            
        except Exception as e:
            self.logger.warning(f"Balance verification failed for {mint[:8]}...: {e}")
            return {}
    
    async def _get_verified_token_balance(self, mint_address: str, position: Position) -> float:
        """
        DEPRECATED: Get token balance with smart ATA indexing awareness.

        This function is being phased out in favor of verification-first position creation.

        Handles the delay between token purchase and ATA indexing.
        """
        # Try blockchain query first
        # Support both fast execution and legacy paths
        if self.fast_execution_enabled and self.fast_transaction_submitter:
            blockchain_balance = await self.fast_transaction_submitter.get_token_balance(mint_address)
        elif self.transaction_signer:
            blockchain_balance = await self.transaction_signer.get_token_balance(mint_address)
        else:
            blockchain_balance = None
        
        # If blockchain query succeeds and returns tokens, use it
        if blockchain_balance is not None and blockchain_balance > 0:
            return blockchain_balance
        
        # If blockchain returns 0, but position tracking shows tokens, check timing
        position_balance = position.amount
        if position_balance > 0:
            # Check if this is a recent buy (ATA might not be indexed yet)
            seconds_since_buy = (datetime.now() - position.entry_time).total_seconds()
            
            if seconds_since_buy < 60:  # Within 1 minute of buy
                self.logger.warning(f"🔄 ATA Indexing Issue: blockchain={blockchain_balance}, position={position_balance}")
                self.logger.warning(f"🕒 Recent buy ({seconds_since_buy:.0f}s ago) - retrying with delays...")
                
                # Retry with exponential backoff and detailed diagnostics
                for attempt in range(4):  # More attempts for critical operation
                    delay = min(2 ** attempt, 8)  # Cap at 8 second delay
                    await asyncio.sleep(delay)

                    self.logger.info(f"🔍 Balance query attempt {attempt + 2} (after {delay}s delay)...")
                    # Support both fast execution and legacy paths
                    if self.fast_execution_enabled and self.fast_transaction_submitter:
                        retry_balance = await self.fast_transaction_submitter.get_token_balance(mint_address)
                    elif self.transaction_signer:
                        retry_balance = await self.transaction_signer.get_token_balance(mint_address)
                    else:
                        retry_balance = None
                    
                    if retry_balance is not None and retry_balance > 0:
                        self.logger.info(f"✅ Success! Found {retry_balance} tokens on attempt {attempt + 2}")
                        return retry_balance
                    
                    self.logger.warning(f"⏳ Attempt {attempt + 2}: still {retry_balance} tokens")
                
                # After all retries failed - this is a critical system issue
                self.logger.error(f"🚨 SYSTEM FAILURE: All blockchain queries failed for {mint_address[:8]}...")
                self.logger.error(f"📊 Position data: {position_balance} tokens, buy_tx: {position.buy_tx_signature}")
                self.logger.error(f"⛔ SELL BLOCKED: Cannot verify token balance after {seconds_since_buy:.0f}s")
                
                # Send critical error notification
                if hasattr(self, 'notifier') and self.notifier:
                    await self.notifier.send_error_notification(
                        "🚨 CRITICAL: Blockchain verification failed",
                        {"mint": mint_address[:8], "seconds_since_buy": seconds_since_buy}
                    )
                
                return 0  # Force safe failure
            
            else:
                # Older position, blockchain should be accurate
                self.logger.error(f"Position-blockchain mismatch: position {position_balance}, blockchain {blockchain_balance}")
                return blockchain_balance if blockchain_balance is not None else 0
        
        # Both blockchain and position show 0 tokens
        return 0.0
    
    async def _get_post_transaction_balance(self, mint_address: str, pre_sell_balance: float = None, tokens_sold: float = None) -> float:
        """
        DEPRECATED: Get token balance after a transaction with retry logic.

        This function is being phased out in favor of verification-first position creation.

        Used after sells to verify remaining balance.

        Args:
            mint_address: Token mint address
            pre_sell_balance: Balance before sell (for stale data detection)
            tokens_sold: Amount of tokens sold (for stale data detection)
        """
        # Try multiple times with delays (transaction might need time to settle)
        for attempt in range(4):
            # Support both fast execution and legacy paths
            if self.fast_execution_enabled and self.fast_transaction_submitter:
                balance = await self.fast_transaction_submitter.get_token_balance(mint_address)
            elif self.transaction_signer:
                balance = await self.transaction_signer.get_token_balance(mint_address)
            else:
                balance = None

            if balance is not None:
                # CRITICAL: Detect stale data - if balance hasn't decreased after a sell, RPC is stale!
                if pre_sell_balance is not None and tokens_sold is not None and tokens_sold > 0:
                    # Balance should have decreased by approximately tokens_sold
                    expected_remaining = pre_sell_balance - tokens_sold
                    # Allow 1% tolerance for rounding
                    if balance > pre_sell_balance * 0.99:
                        self.logger.warning(f"⚠️ STALE BALANCE DETECTED: Query returned {balance:.0f} but pre-sell was {pre_sell_balance:.0f} (sold {tokens_sold:.0f})")
                        self.logger.warning(f"   RPC hasn't indexed the sell yet, retrying...")
                        balance = None  # Treat as failed query

                if balance is not None:
                    return balance

            if attempt < 3:  # Don't sleep on last attempt
                await asyncio.sleep(1 + attempt)  # 1s, 2s, 3s delays
                self.logger.warning(f"Post-transaction balance query attempt {attempt + 1} failed, retrying...")

        # If all attempts failed, return None (cannot verify balance)
        self.logger.error("❌ All post-transaction balance queries failed - cannot verify balance!")
        return None

    async def buy_token(self, mint_address: str, usd_amount: float, paper_mode: bool = True, symbol: str = "UNKNOWN", confidence_score: float = None) -> Dict:
        """Execute a buy order for a token"""
        try:
            self.logger.info(f"Executing BUY for {mint_address}, amount: ${usd_amount}")

            # CRITICAL PATH OPTIMIZATION: Skip dump protection in fast execution mode
            # Dump protection adds 700-1200ms (2 Moralis API calls) which causes us to arrive 7 slots late
            # Fast execution relies on SPEED to get good prices, not pre-trade analysis
            skip_dump_protection = self.fast_execution_enabled

            if not skip_dump_protection:
                # DUMP PROTECTION - Check before any other logic (legacy mode only)
                dump_check = await self._check_dump_protection(mint_address)
                if not dump_check.get('safe_to_enter', True):
                    reason = dump_check.get('reason', 'Unknown dump protection trigger')
                    self.logger.warning(f"🚨 DUMP PROTECTION: Skipping {mint_address[:8]}... - {reason}")
                    return {
                        "success": False,
                        "error": f"Dump protection: {reason}",
                        "dump_protection": dump_check
                    }
            else:
                self.logger.debug("⚡ Dump protection skipped (fast execution mode)")

            if paper_mode:
                return await self._execute_paper_buy(mint_address, usd_amount, symbol, confidence_score)
            else:
                return await self._execute_real_buy(mint_address, usd_amount, symbol)
                
        except Exception as e:
            self.logger.error(f"Error executing buy: {e}")
            
            # Send error notification
            if self.notifier:
                await self.notifier.send_error_notification(
                    f"Failed to execute buy order: {str(e)}",
                    {"token": mint_address[:8], "amount": f"${usd_amount}"}
                )
            
            return {"success": False, "error": str(e)}
    
    async def _check_dump_protection(self, mint_address: str) -> Dict:
        """
        Check if token is dumping and should be avoided
        Runs BEFORE safety checks in aggressive mode
        """
        try:
            # Get recent trades for analysis
            recent_trades = await self.moralis.get_token_swaps(mint_address, limit=50)
            if not recent_trades:
                return {'safe_to_enter': True, 'reason': 'No trade data available'}
            
            # Get current price
            current_price = await self.moralis.get_current_price(mint_address)
            if not current_price or current_price <= 0:
                return {'safe_to_enter': True, 'reason': 'No current price available'}
            
            import time
            current_time = time.time()
            
            # Get dump protection config
            dump_config = getattr(self.config, 'dump_protection', {})
            momentum_window = dump_config.get('momentum_check_seconds', 30)
            cutoff_time = current_time - momentum_window
            
            prices_in_window = []
            buy_volume = 0
            sell_volume = 0
            
            for trade in recent_trades:
                trade_time = self._parse_timestamp_to_unix(trade.get('timestamp', ''))
                if trade_time < cutoff_time:
                    continue
                    
                price = trade.get('price', 0)
                if price > 0:
                    prices_in_window.append(price)
                    
                # Track buy/sell volume
                if trade.get('side') == 'buy':
                    buy_volume += trade.get('amount_usd', 0)
                else:
                    sell_volume += trade.get('amount_usd', 0)
            
            # Calculate momentum
            momentum = 0
            if len(prices_in_window) >= 2:
                price_30s_ago = prices_in_window[0]
                momentum = (current_price - price_30s_ago) / price_30s_ago if price_30s_ago > 0 else 0
            
            # Calculate buy/sell ratio
            total_volume = buy_volume + sell_volume
            buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.5
            
            # Find recent peak (5 min window)
            peak_window = 300
            peak_cutoff = current_time - peak_window
            recent_peak = 0
            peak_time = 0
            
            for trade in recent_trades:
                trade_time = self._parse_timestamp_to_unix(trade.get('timestamp', ''))
                if trade_time < peak_cutoff:
                    continue
                price = trade.get('price', 0)
                if price > recent_peak:
                    recent_peak = price
                    peak_time = trade_time
            
            # Get configuration values first
            min_momentum = dump_config.get('min_momentum_threshold', -0.15)
            max_sell_ratio = dump_config.get('max_sell_ratio', 0.7)
            avoid_peaks_seconds = dump_config.get('avoid_peaks_seconds', 30)
            
            # Check if we're too close to a recent peak
            time_since_peak = current_time - peak_time if peak_time > 0 else 999
            near_peak = time_since_peak < avoid_peaks_seconds and current_price > recent_peak * 0.95
            
            reasons = []
            if momentum < min_momentum:
                reasons.append(f"Price momentum: {momentum:.1%} (dropped >{abs(min_momentum):.0%} in {momentum_window}s)")
            if buy_ratio < (1 - max_sell_ratio):
                reasons.append(f"Sell pressure: {buy_ratio:.1%} buy ratio (>{max_sell_ratio:.0%} sells)")
            if near_peak:
                reasons.append(f"Near peak: {time_since_peak:.0f}s since peak at ${recent_peak:.8f}")
            
            is_dumping = len(reasons) > 0
            
            result = {
                'safe_to_enter': not is_dumping,
                'momentum_30s': momentum,
                'buy_ratio': buy_ratio,
                'near_peak': near_peak,
                'time_since_peak': time_since_peak,
                'recent_peak': recent_peak,
                'current_price': current_price,
                'reason': '; '.join(reasons) if reasons else 'Safe to enter'
            }
            
            if is_dumping:
                self.logger.warning(f"🚨 Dump detected for {mint_address[:8]}...: {'; '.join(reasons)}")
            else:
                self.logger.debug(f"✅ Dump protection passed for {mint_address[:8]}...")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error in dump protection check: {e}")
            return {'safe_to_enter': True, 'reason': f'Error in check: {e}'}
    
    def _parse_timestamp_to_unix(self, timestamp) -> float:
        """
        Parse timestamp to unix timestamp
        """
        try:
            if isinstance(timestamp, (int, float)):
                return float(timestamp)
            if isinstance(timestamp, str):
                if 'T' in timestamp:
                    # ISO format
                    from datetime import datetime
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    return dt.timestamp()
                else:
                    return float(timestamp)
        except:
            pass
        import time
        return time.time()

    async def sell_token(self, mint_address: str, percentage: float, paper_mode: bool = True, symbol: str = "UNKNOWN", exit_reason: str = "unknown", current_price: float = None, low_priority: bool = False) -> Dict:
        """Execute a sell order for a token

        Args:
            current_price: Optional current price (e.g., from LaserStream bonding curve).
                          If provided, uses this price for P&L calculation instead of fetching from Moralis.
            low_priority: If True, uses minimum fees and skips Jito (for dead tokens with no activity)
                          This ensures we use the exact price that triggered the exit decision.
        """
        try:
            # ENHANCED: Comprehensive exit logging with multi-tier details
            position_info = ""
            if mint_address in self.active_positions:
                pos = self.active_positions[mint_address]
                hold_time = (datetime.now() - pos.entry_time).total_seconds()

                # Use provided price (from LaserStream) if available, otherwise fetch from Moralis
                # This ensures we use the same price that triggered the exit decision
                if current_price is not None and current_price > 0:
                    # LaserStream provided real-time bonding curve price
                    pass  # Use the provided price
                else:
                    # Fallback to Moralis for backward compatibility
                    current_price = await self.moralis.get_current_price(mint_address, fresh=True) or pos.entry_price

                pnl_pct = ((current_price / pos.entry_price) - 1) * 100 if pos.entry_price > 0 else 0
                
                # Add multi-tier status to logging
                multi_tier_config = getattr(self.config, 'multi_tier_exits', {})
                if multi_tier_config.get('enabled', False):
                    tier_status = self.get_position_tier_status(pos)
                    tier_info = f"tier={tier_status['current_tier']} sold={tier_status['total_sold_pct']:.0%} remaining={tier_status['remaining_pct']:.0%}"
                    position_info = f"hold_sec={hold_time:.0f} pnl={pnl_pct:+.1f}% {tier_info} entry=${pos.entry_price:.8f} current=${current_price:.8f}"
                else:
                    position_info = f"hold_sec={hold_time:.0f} pnl={pnl_pct:+.1f}% entry=${pos.entry_price:.8f} current=${current_price:.8f}"
            
            self.logger.info(f"Exit: mint={mint_address[:8]}... reason={exit_reason} percentage={percentage*100:.0f}% {position_info}")
            
            if paper_mode:
                return await self._execute_paper_sell(mint_address, percentage, symbol, exit_reason)
            else:
                return await self._execute_real_sell(mint_address, percentage, symbol, exit_reason, low_priority)
                
        except Exception as e:
            self.logger.error(f"Error executing sell: {e}")
            
            # Send error notification
            if self.notifier:
                await self.notifier.send_error_notification(
                    f"Failed to execute sell order: {str(e)}",
                    {"token": mint_address[:8], "percentage": f"{percentage*100}%"}
                )
            
            return {"success": False, "error": str(e)}

    async def _execute_paper_buy(self, mint_address: str, usd_amount: float, symbol: str = "UNKNOWN", confidence_score: float = None) -> Dict:
        """Execute a paper trading buy"""
        # Get current price
        price = await self.moralis.get_current_price(mint_address)
        if price <= 0:
            return {"success": False, "error": "Could not get price"}
        
        # Calculate SOL amount using current estimate
        sol_price = getattr(self.config, "paper_trading", {}).get("sol_price_estimate", 140)
        sol_amount = usd_amount / sol_price
        
        # Apply realistic fees for alpha-following strategy (not sniping)
        paper_config = getattr(self.config, "paper_trading", {})
        
        # DEX trading fee (pump.fun ~0.30%)
        dex_fee_bps = paper_config.get("dex_fee_bps", 30)
        
        # Network fees (less aggressive since we're following, not sniping)
        sol_price_usd = paper_config.get("sol_price_estimate", 140)  # Current SOL price estimate
        base_tx_fee_sol = paper_config.get("base_tx_fee_sol", 0.000005)  # Base Solana tx fee
        priority_fee_sol = paper_config.get("priority_fee_sol", 0.002)    # Medium priority (not high)
        network_fee_usd = (base_tx_fee_sol + priority_fee_sol) * sol_price_usd
        
        # Slippage (lower since we're not competing with MEV bots)
        buy_slip_bps = paper_config.get("buy_slippage_bps", 300)  # 3% buy slippage
        max_slippage_bps = paper_config.get("max_slippage_bps", 150)
        
        # Slippage rejection gate
        if buy_slip_bps > max_slippage_bps:
            self.logger.warning(f"Trade rejected: estimated slippage {buy_slip_bps}bps > max {max_slippage_bps}bps")
            return {"success": False, "error": f"Slippage too high: {buy_slip_bps/100:.1f}%"}
        
        # Calculate fill price with slippage
        fill_price = price * (1 + buy_slip_bps/10000)
        
        # Calculate total fees: DEX fee + network fee
        dex_fee_usd = usd_amount * (dex_fee_bps/10000)
        total_fee_usd = dex_fee_usd + network_fee_usd
        
        # Calculate tokens received after all fees
        net_usd_amount = usd_amount - total_fee_usd
        if net_usd_amount <= 0:
            return {"success": False, "error": f"Trade too small: fees (${total_fee_usd:.2f}) exceed trade size"}
        
        tokens_received = net_usd_amount / fill_price
        
        # Log fee breakdown for transparency
        self.logger.debug(f"Buy fees: DEX ${dex_fee_usd:.3f} + Network ${network_fee_usd:.3f} = ${total_fee_usd:.3f} (on ${usd_amount})")
        
        # Update paper capital
        if self.paper_capital < usd_amount:
            return {"success": False, "error": "Insufficient paper capital"}
        
        self.paper_capital -= usd_amount
        
        # Create position with proper cost tracking (use fill_price as entry)
        # Get buffer configuration
        buffer_config = getattr(self.config, 'volatility_buffer', {})
        buffer_seconds = buffer_config.get('buffer_seconds', 0) if buffer_config.get('enabled', False) else 0
        entry_time = datetime.now()
        
        position = Position(
            mint=mint_address,
            symbol=symbol,
            entry_price=fill_price,
            amount=tokens_received,
            sol_invested=sol_amount,
            entry_time=entry_time,
            tp_price=fill_price * self.config.tp_multiplier,
            sl_price=fill_price * self.config.stop_loss_pct,
            peak_price=fill_price,
            paper_mode=True,
            tokens_initial=tokens_received,
            cost_usd_remaining=usd_amount,
            avg_cost_per_token=usd_amount / tokens_received,
            buffer_period_seconds=buffer_seconds,
            buffer_end_time=datetime.fromtimestamp(entry_time.timestamp() + buffer_seconds) if buffer_seconds > 0 else None
        )
        
        self.active_positions[mint_address] = position
        
        # Record in P&L store
        self.pnl_store.add_trade(
            action="BUY",
            symbol=symbol,
            mint_address=mint_address,
            amount=tokens_received,
            price=fill_price,  # Use actual fill price for consistency
            usd_value=usd_amount,
            paper_mode=True
        )
        
        # Send Discord notification
        if self.notifier:
            await self.notifier.send_trade_notification(
                side="BUY",
                symbol=symbol,
                mint_address=mint_address,
                quantity=tokens_received,
                price=fill_price,  # Use actual fill price for consistency
                usd_amount=usd_amount,
                equity=self.pnl_store.current_equity,
                confidence_score=confidence_score,
                paper_mode=True
            )
        
        self.logger.info(f"Paper buy executed: {tokens_received} tokens at ${fill_price:.8f} (market: ${price:.8f})")
        
        return {
            "success": True,
            "price": fill_price,  # Return actual fill price for consistency
            "tokens_received": tokens_received,
            "sol_amount": sol_amount,
            "usd_amount": usd_amount,
            "symbol": symbol,
            "paper_mode": True
        }

    async def _execute_real_buy(self, mint_address: str, usd_amount: float, symbol: str = "UNKNOWN") -> Dict:
        """Execute a real buy transaction via QuickNode/PumpFun API"""
        try:
            # Check trading mode
            trading_mode = getattr(self.config, 'trading_mode', 'simulation')

            if trading_mode in ['simulation', 'paper']:
                self.logger.warning(f"Trading mode is '{trading_mode}', not executing real trade")
                return await self._execute_paper_buy(mint_address, usd_amount, symbol)

            # Check execution method
            if self.fast_execution_enabled:
                # Fast execution mode - uses Helius + Jito
                if not self.fast_transaction_submitter:
                    self.logger.error("Fast transaction submitter not initialized")
                    return {"success": False, "error": "Fast execution not configured"}

                # Fast execution uses wallet from fast_transaction_submitter (no pumpportal config needed)
                wallet_pubkey = None  # Not needed for fast execution
            else:
                # Legacy execution mode - uses QuickNode + PumpPortal
                if not self.transaction_signer:
                    self.logger.error("Transaction signer not initialized for live trading")
                    return {"success": False, "error": "Transaction signer not configured"}

                if not self.pumpfun or not self.config.quicknode_endpoint:
                    self.logger.error("QuickNode not configured for live trading")
                    return {"success": False, "error": "QuickNode not configured"}

                # Get wallet public key from pumpportal config (legacy mode only)
                wallet_pubkey = self.config.pumpportal.get('wallet_public_key') if hasattr(self.config, 'pumpportal') else None

                if not wallet_pubkey:
                    self.logger.error("Wallet public key not configured in pumpportal section")
                    return {"success": False, "error": "Wallet public key not configured"}
            
            # Convert USD to SOL using real-time price
            sol_price = await self._get_sol_price()
            if sol_price is None:
                self.logger.error("Cannot get SOL price for buy calculation")
                return {"success": False, "error": "Cannot get SOL price"}
            
            sol_amount = usd_amount / sol_price
            
            # We'll get the actual price from transaction simulation instead of Moralis
            # This removes latency and uses real-time pump.fun pricing
            self.logger.info(f"Creating transaction for ${usd_amount} ({sol_amount:.4f} SOL)")

            # Check wallet balance (legacy mode only - fast execution handles this internally)
            if not self.fast_execution_enabled:
                wallet_balance = await self.transaction_signer.get_wallet_balance()
                if wallet_balance is None or wallet_balance < sol_amount:
                    self.logger.error(f"Insufficient balance: {wallet_balance:.4f} SOL < {sol_amount:.4f} SOL needed")
                    return {"success": False, "error": f"Insufficient balance: {wallet_balance:.4f} SOL"}
            
            # Execute transaction based on execution mode
            if self.fast_execution_enabled:
                # FAST EXECUTION: Use Helius + Jito
                self.logger.info(f"⚡ Fast execution: ${usd_amount} ({sol_amount:.4f} SOL) for {symbol}")

                tx_result = await self.fast_transaction_submitter.buy(
                    mint=mint_address,
                    sol_amount=sol_amount,
                    slippage_bps=2000  # 20% slippage - simulation pre-validates, preventing ghost positions
                )

                if not tx_result.success:
                    self.logger.error(f"Fast buy failed: {tx_result.error}")
                    return {"success": False, "error": tx_result.error}

                tx_signature = tx_result.signature
                estimated_tokens = tx_result.tokens_received or 0
                sim_result = {"success": True, "estimated_tokens": estimated_tokens}  # Mimic simulation result

                self.logger.info(f"✅ Fast buy executed: {estimated_tokens:,.0f} tokens - TX: {tx_signature}")

            else:
                # LEGACY EXECUTION: Use QuickNode + PumpPortal
                self.logger.info(f"Creating live buy transaction: ${usd_amount} ({sol_amount:.4f} SOL) for {symbol}")

                # Higher slippage for aggressive entry fills
                slippage_bps = 5000  # 50% slippage for ultra-volatile pump.fun tokens

                tx_result = await self.pumpfun.create_buy_transaction(
                    wallet_pubkey=wallet_pubkey,
                    mint_address=mint_address,
                    sol_amount=sol_amount,
                    slippage_bps=slippage_bps
                )

                if not tx_result.get("success"):
                    self.logger.error(f"Failed to create buy transaction: {tx_result.get('error')}")
                    return tx_result

                # Get the base64 transaction
                transaction_b64 = tx_result.get("transaction")
                if not transaction_b64:
                    return {"success": False, "error": "No transaction returned"}

                # Simulate transaction to get accurate token estimate BEFORE signing
                self.logger.info("Simulating transaction to get accurate token estimate...")
                sim_result = await self.transaction_signer.simulate_transaction(transaction_b64)

                estimated_tokens = 0
                if sim_result.get("success"):
                    estimated_tokens = sim_result.get("estimated_tokens", 0)
                    if estimated_tokens > 0:
                        self.logger.info(f"✅ Simulation: Will receive {estimated_tokens:,.0f} tokens")
                    else:
                        self.logger.warning("⚠️ Simulation successful but no tokens found - proceeding anyway")
                else:
                    self.logger.warning(f"⚠️ Simulation failed: {sim_result.get('error')} - proceeding with send")

                # Sign and send transaction
                self.logger.info("Signing and sending transaction...")
                send_result = await self.transaction_signer.sign_and_send_transaction(transaction_b64)

                if not send_result.get("success"):
                    self.logger.error(f"Transaction failed: {send_result.get('error')}")
                    return send_result

                tx_signature = send_result.get("signature")

            # Common path continues here (position creation, P&L, Discord, etc.)
            if tx_signature:
                self.logger.info(f"✅ Live buy executed: {symbol} for ${usd_amount} - TX: {tx_signature}")

                # FAST PATH: Use bonding curve quotes for immediate position creation
                # BACKGROUND: Launch async task to get actual costs from transaction receipt
                # This ensures we don't slow down the bot while still getting accurate P&L eventually

                # NEW: Create position with simulation-based token estimate OR price fallback
                position_created = False
                current_price = None

                if estimated_tokens > 0:
                    # Use simulation results for accurate position creation
                    # Note: estimated_tokens from bonding curve is in raw units (with 6 decimals)
                    # Convert to UI units for position tracking (sell functions expect UI units)
                    estimated_tokens_ui = estimated_tokens / 1e6  # Pump.fun uses 6 decimals
                    current_price = usd_amount / estimated_tokens_ui  # Derive actual fill price per UI token
                    self.logger.info(f"🎯 Using bonding curve quote: {estimated_tokens:,.0f} raw tokens ({estimated_tokens_ui:,.2f} UI) at derived price ${current_price:.8f}")

                    # Store UI tokens in position (sell functions expect UI amounts)
                    estimated_tokens = estimated_tokens_ui
                else:
                    # CRITICAL PATH OPTIMIZATION: ONE quick Moralis call without retries
                    # This path should rarely execute (only if bonding curve quote fails)
                    self.logger.warning("⚠️ Bonding curve quote unavailable - attempting quick Moralis fallback")
                    try:
                        current_price = await asyncio.wait_for(
                            self.moralis.get_current_price(mint_address, fresh=True),
                            timeout=1.0  # 1 second timeout instead of 6+ seconds with retries
                        )
                    except asyncio.TimeoutError:
                        self.logger.error("Moralis price fetch timed out after 1s")
                        current_price = 0

                    if current_price > 0:
                        # Calculate estimated tokens based on current price with fee/slippage buffer
                        # Account for: 0.95% protocol + 0.30% dev fee + slippage = ~5% total buffer
                        effective_usd = usd_amount * 0.95  # Conservative after fees
                        estimated_tokens = effective_usd / current_price
                        self.logger.info(f"📊 Using Moralis fallback: {estimated_tokens:,.0f} tokens at ${current_price:.8f}")
                
                if estimated_tokens > 0 and current_price > 0:
                    
                    # Get buffer configuration
                    buffer_config = getattr(self.config, 'volatility_buffer', {})
                    buffer_seconds = buffer_config.get('buffer_seconds', 0) if buffer_config.get('enabled', False) else 0
                    entry_time = datetime.now()
                    
                    position = Position(
                        mint=mint_address,
                        symbol=symbol,
                        entry_price=current_price,
                        amount=estimated_tokens,
                        sol_invested=sol_amount,
                        entry_time=entry_time,
                        tp_price=current_price * self.config.tp_multiplier,
                        sl_price=current_price * self.config.stop_loss_pct,
                        peak_price=current_price,
                        paper_mode=False,
                        tokens_initial=estimated_tokens,
                        cost_usd_remaining=usd_amount,
                        avg_cost_per_token=current_price,
                        buy_tx_signature=tx_signature,
                        verified_from_blockchain=False,
                        buffer_period_seconds=buffer_seconds,
                        buffer_end_time=datetime.fromtimestamp(entry_time.timestamp() + buffer_seconds) if buffer_seconds > 0 else None
                    )
                    
                    # Mark if this position was created with simulation data for better tracking
                    position.simulation_verified = sim_result.get("success") and sim_result.get("estimated_tokens", 0) > 0

                    # CRITICAL: Mark position as UNVERIFIED (using bonding curve quotes)
                    # Background task will update with actual costs from transaction receipt
                    position.actual_costs_verified = False
                    position.unverified_sol_invested = sol_amount  # Store quote-based estimate

                    self.active_positions[mint_address] = position
                    estimation_method = "simulation-based" if sim_result.get("success") and sim_result.get("estimated_tokens", 0) > 0 else "price-based estimate"
                    self.logger.info(f"🚀 Position created: {estimated_tokens:,.0f} tokens at ${current_price:.8f} ({estimation_method})")
                    self.logger.warning(f"⚠️ Position costs UNVERIFIED - using bonding curve quote ({sol_amount:.6f} SOL)")
                    position_created = True

                    # CRITICAL: Record BUY trade in P&L store immediately after position creation
                    # This ensures the trade is tracked even if the position is created with simulation data
                    self.pnl_store.add_trade(
                        action="BUY",
                        symbol=symbol,
                        mint_address=mint_address,
                        amount=estimated_tokens,
                        price=current_price,
                        usd_value=usd_amount,
                        paper_mode=False
                    )
                    self.logger.info(f"✅ BUY trade recorded in P&L store: {estimated_tokens:,.0f} tokens at ${current_price:.8f}")

                    # BACKGROUND TASK: Verify actual costs from transaction receipt
                    # This keeps retrying until it succeeds, ensuring accurate P&L eventually
                    if self.fast_execution_enabled and self.fast_transaction_submitter:
                        asyncio.create_task(self._verify_buy_costs_async(mint_address, tx_signature, symbol, sol_price))

                else:
                    # No position created - wait for WSS verification
                    self.logger.warning(f"❌ Could not get price for {symbol} after {max_price_retries} attempts - will wait for WSS verification")
                    self.logger.info(f"📝 Transaction successful but position creation deferred until WSS confirmation")
                
                # Create realtime position if we have a position
                if position_created and self.use_realtime_positions:
                    trade_event = {
                        'mint': mint_address,
                        'symbol': symbol,  # Add symbol for RealtimePosition
                        'action': 'buy',
                        'tokens_received': position.amount,
                        'sol_amount': sol_amount,
                        'price': position.entry_price,
                        'tx_signature': tx_signature
                    }
                    self.realtime_positions.handle_trade_event(trade_event)
                
                # Store pending transaction data for WSS reconciliation
                if not hasattr(self, '_pending_transactions'):
                    self._pending_transactions = {}
                
                self._pending_transactions[tx_signature] = {
                    'mint_address': mint_address,
                    'sol_amount': sol_amount,
                    'usd_amount': usd_amount,
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'has_immediate_position': position_created
                }
                
                # P&L and notifications will be handled after verification in _create_verified_position
                
                return {
                    "success": True,
                    "price": current_price if position_created else 0.0,
                    "tx_signature": tx_signature,
                    "sol_amount": sol_amount,
                    "usd_amount": usd_amount,
                    "tokens_received": position.amount if position_created else 0.0,
                    "symbol": symbol,
                    "paper_mode": False,
                    "status": "immediate_position" if position_created else "wss_verification_pending",
                    "message": f"Position created with price data" if position_created else f"Transaction sent, position creation deferred until WSS confirmation"
                }
            else:
                error_msg = send_result.get('error', '')
                self.logger.error(f"Failed to send buy transaction: {error_msg}")
                
                # Check if this is a slippage error - log but don't retry (we're too late)
                if self._is_slippage_error(error_msg):
                    self.logger.warning(f"🚫 SLIPPAGE ERROR - Skipping {symbol}: Too late to follow alpha wallet")
                    return {"success": False, "error": "slippage_skip", "message": f"Slippage error on {symbol} - alpha signal expired"}
                
                # CRITICAL: Do not create position if buy failed
                return {"success": False, "error": error_msg}
            
        except Exception as e:
            self.logger.error(f"Error in real buy execution: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_paper_sell(self, mint_address: str, percentage: float, symbol: str = "UNKNOWN", exit_reason: str = "unknown") -> Dict:
        """Execute a paper trading sell"""
        if mint_address not in self.active_positions:
            return {"success": False, "error": "No position found"}
        
        position = self.active_positions[mint_address]
        
        # Always use the position's symbol if it's been updated from "UF"
        actual_symbol = position.symbol if hasattr(position, 'symbol') and position.symbol != "UF" else symbol
        
        # QOL: Show "Unknown" instead of "UF" for better user experience
        if actual_symbol == "UF":
            actual_symbol = "Unknown"
            
        self.logger.debug(f"🏷️ Using symbol: {actual_symbol} (position.symbol={getattr(position, 'symbol', 'missing')}, passed={symbol})")
        
        # Get current price
        current_price = await self.moralis.get_current_price(mint_address, fresh=True)
        if current_price <= 0:
            return {"success": False, "error": "Could not get current price"}
        
        # Calculate sell amount (percentage of current amount)
        tokens_to_sell = position.amount * percentage
        if tokens_to_sell <= 0:
            return {"success": False, "error": "Nothing to sell"}
        
        # Apply realistic fees for alpha-following strategy
        paper_config = getattr(self.config, "paper_trading", {})
        
        # DEX trading fee
        dex_fee_bps = paper_config.get("dex_fee_bps", 30)
        
        # Network fees (same as buy)
        sol_price_usd = paper_config.get("sol_price_estimate", 140)
        base_tx_fee_sol = paper_config.get("base_tx_fee_sol", 0.000005)
        priority_fee_sol = paper_config.get("priority_fee_sol", 0.002)
        network_fee_usd = (base_tx_fee_sol + priority_fee_sol) * sol_price_usd
        
        # Slippage (slightly higher on sells due to market impact)
        sell_slip_bps = paper_config.get("sell_slippage_bps", 300)  # 3% sell slippage
        
        # Calculate fill price with slippage
        fill_price = current_price * (1 - sell_slip_bps/10000)
        
        # Calculate USD received after all fees
        cost_basis_usd = tokens_to_sell * position.avg_cost_per_token
        gross_usd = tokens_to_sell * fill_price
        dex_fee_usd = gross_usd * (dex_fee_bps/10000)
        total_fee_usd = dex_fee_usd + network_fee_usd
        usd_received = gross_usd - total_fee_usd
        
        # Log fee breakdown
        self.logger.debug(f"Sell fees: DEX ${dex_fee_usd:.3f} + Network ${network_fee_usd:.3f} = ${total_fee_usd:.3f} (on ${gross_usd:.2f})")
        
        # Update position with proper accounting
        position.amount -= tokens_to_sell
        position.cost_usd_remaining -= cost_basis_usd
        if position.amount > 0:
            position.avg_cost_per_token = position.cost_usd_remaining / position.amount
        else:
            position.avg_cost_per_token = 0.0
        
        self.paper_capital += usd_received
        
        # Calculate profit/loss with correct cost basis
        profit = usd_received - cost_basis_usd
        profit_pct = (profit / cost_basis_usd) * 100 if cost_basis_usd > 0 else 0
        
        # Record in P&L store
        self.pnl_store.add_trade(
            action="SELL",
            symbol=actual_symbol,
            mint_address=mint_address,
            amount=tokens_to_sell,
            price=current_price,
            usd_value=usd_received,
            realized_pnl=profit,
            paper_mode=True
        )
        
        # Send Discord notification
        if self.notifier:
            await self.notifier.send_trade_notification(
                side="SELL",
                symbol=actual_symbol,
                mint_address=mint_address,
                quantity=tokens_to_sell,
                price=current_price,
                usd_amount=usd_received,
                equity=self.pnl_store.current_equity,
                realized_pnl=profit,
                paper_mode=True
            )
        
        self.logger.info(f"Paper sell executed: {tokens_to_sell} tokens at ${current_price}, profit: ${profit:.2f} ({profit_pct:.2f}%)")
        
        # Remove position if fully sold
        if position.amount <= 1e-9:  # Small threshold for floating point precision
            del self.active_positions[mint_address]
            
            # Update stats
            self.total_trades += 1
            if profit > 0:
                self.winning_trades += 1
        
        return {
            "success": True,
            "tokens_sold": tokens_to_sell,
            "price": current_price,
            "usd_amount": usd_received,
            "usd_received": usd_received,
            "sol_amount": usd_received / 20,  # Mock SOL price
            "profit": profit,
            "profit_pct": profit_pct,
            "symbol": symbol,
            "paper_mode": True
        }

    async def _execute_real_sell(self, mint_address: str, percentage: float, symbol: str = "UNKNOWN", exit_reason: str = "unknown", low_priority: bool = False) -> Dict:
        """Execute a real sell transaction via QuickNode/PumpFun API"""
        try:
            # Check trading mode
            trading_mode = getattr(self.config, 'trading_mode', 'simulation')

            if trading_mode in ['simulation', 'paper']:
                self.logger.warning(f"Trading mode is '{trading_mode}', not executing real trade")
                return await self._execute_paper_sell(mint_address, percentage, symbol, exit_reason)
            
            # Check if we have the position
            if mint_address not in self.active_positions:
                self.logger.error(f"No active position for {mint_address}")
                return {"success": False, "error": "No active position"}
            
            position = self.active_positions[mint_address]

            # CRITICAL: Skip 0% sells (TP3 moonshot mode)
            if percentage <= 0.0001:  # Effectively 0%
                self.logger.info(f"⏭️ Skipping 0% sell for {mint_address[:8]}... (moonshot mode - holding position)")
                position.is_selling = False  # Reset flag if it was set
                return {"success": False, "error": "zero_percent_sell", "message": "Moonshot mode - holding position"}

            # Always use the position's symbol if it's been updated from "UF"
            actual_symbol = position.symbol if hasattr(position, 'symbol') and position.symbol != "UF" else symbol

            # QOL: Show "Unknown" instead of "UF" for better user experience
            if actual_symbol == "UF":
                actual_symbol = "Unknown"

            self.logger.debug(f"🏷️ Using symbol: {actual_symbol} (position.symbol={getattr(position, 'symbol', 'missing')}, passed={symbol})")

            # Check if reconciliation is needed - pause all sells until resolved
            if hasattr(position, 'needs_reconciliation') and position.needs_reconciliation:
                self.logger.info(f"⏸️ Skipping sell for {actual_symbol} - reconciliation in progress")
                return {"success": False, "error": "awaiting_reconciliation"}
            
            # Prevent race conditions - check if position is already being sold
            if hasattr(position, 'is_selling') and position.is_selling:
                # Check if selling flag is stuck (over 60 seconds)
                if hasattr(position, 'selling_started_time'):
                    time_since_sell_start = (datetime.now() - position.selling_started_time).total_seconds()
                    if time_since_sell_start > 60:  # 60 second timeout
                        self.logger.warning(f"⚠️ STUCK SELL FLAG: Resetting {actual_symbol} after {time_since_sell_start:.0f}s")
                        position.is_selling = False
                        delattr(position, 'selling_started_time')
                    else:
                        self.logger.warning(f"🔒 Sell blocked - {actual_symbol} position already being sold ({time_since_sell_start:.0f}s ago)")
                        return {"success": False, "error": "Position already being sold"}
                else:
                    # No timestamp - assume it's stuck, reset it
                    self.logger.warning(f"⚠️ STUCK SELL FLAG: Resetting {actual_symbol} (no timestamp)")
                    position.is_selling = False
            
            # Mark as selling to prevent concurrent sells
            position.is_selling = True
            position.selling_started_time = datetime.now()  # Track when selling started
            
            # Check execution method
            if self.fast_execution_enabled:
                # Fast execution mode - uses Helius + Jito
                if not self.fast_transaction_submitter:
                    self.logger.error("Fast transaction submitter not initialized")
                    position.is_selling = False
                    return {"success": False, "error": "Fast execution not configured"}
            else:
                # Legacy execution mode - uses QuickNode + PumpPortal
                if not self.transaction_signer:
                    self.logger.error("Transaction signer not initialized for live trading")
                    position.is_selling = False
                    return {"success": False, "error": "Transaction signer not configured"}

                if not self.pumpfun or not self.config.quicknode_endpoint:
                    self.logger.error("QuickNode not configured for live trading")
                    position.is_selling = False
                    return {"success": False, "error": "QuickNode not configured"}

                # Get wallet public key from pumpportal config
                wallet_pubkey = self.config.pumpportal.get('wallet_public_key') if hasattr(self.config, 'pumpportal') else None

                if not wallet_pubkey:
                    self.logger.error("Wallet public key not configured")
                    position.is_selling = False
                    return {"success": False, "error": "Wallet public key not configured"}
            
            # CRITICAL: Never attempt to sell positions with 0 tokens (ghost positions)
            if position.amount <= 0:
                buy_age = (datetime.now() - position.entry_time).total_seconds()
                self.logger.error(f"👻 GHOST POSITION DETECTED - {symbol}: {position.amount} tokens, buy age {buy_age:.0f}s")
                
                # If this is a very recent buy (< 2 minutes) that might still be indexing, keep position
                if buy_age < 120:
                    self.logger.warning(f"⚠️ {symbol}: Recent buy with 0 tokens - likely indexing delay, keeping position")
                    position.is_selling = False
                    return {"success": False, "error": "indexing_delay", "message": "Recent buy still indexing"}
                else:
                    # Old position with 0 tokens - remove it entirely
                    self.logger.error(f"🧙 {symbol}: Removing old ghost position (age: {buy_age:.0f}s)")
                    if mint_address in self.active_positions:
                        del self.active_positions[mint_address]
                    return {"success": False, "error": "ghost_position_removed"}
            
            # Use internal tracking for sell calculations (fast, no blockchain calls)
            tokens_to_sell = position.amount * percentage
            
            if tokens_to_sell <= 0:
                self.logger.error(f"No tokens to sell: internal balance {position.amount}, percentage {percentage}")
                return {"success": False, "error": "No tokens to sell - position empty"}
            
            # Conservative sell amount - adjust based on position verification status
            # If position is verified from blockchain, we can be more aggressive
            # For 100% sells, use actual amount to ensure position closes
            if percentage >= 0.999:  # Selling 100%
                conservative_sell_amount = tokens_to_sell  # Use full amount for complete exit
                self.logger.info(f"📤 Full exit: selling all {tokens_to_sell:.0f} tokens")
            elif hasattr(position, 'verified_from_blockchain') and position.verified_from_blockchain:
                conservative_sell_amount = tokens_to_sell * 0.995  # Only 0.5% buffer for verified positions
                self.logger.info(f"✅ Verified position: using 99.5% conservative amount")
            else:
                conservative_sell_amount = tokens_to_sell * 0.985  # 1.5% buffer for unverified positions
                self.logger.info(f"⚠️ Unverified position: using 98.5% conservative amount")
            
            self.logger.info(f"Selling {percentage*100:.1f}% of position: {tokens_to_sell:.0f} tokens (internal tracking), conservative amount: {conservative_sell_amount:.0f}")
            
            # Get current price for logging
            current_price = await self.moralis.get_current_price(mint_address, fresh=True)
            if current_price <= 0:
                current_price = position.entry_price
            
            # Smart token availability check for high-risk sells
            if await self._should_check_token_availability(mint_address, position, exit_reason):
                availability_result = await self._smart_availability_check(mint_address, int(conservative_sell_amount))
                
                if not availability_result['available'] and availability_result['confidence'] in ['high', 'medium']:
                    # High confidence tokens aren't available - trigger smart retry
                    self.logger.warning(f"🔍 Tokens not available ({availability_result['balance']} < {conservative_sell_amount:.0f}) - triggering smart retry")
                    retry_result = await self._smart_sell_retry(mint_address, symbol, exit_reason)
                    return retry_result
                elif not availability_result['available']:
                    # Low confidence - proceed but expect potential failure
                    self.logger.warning(f"❓ Uncertain token availability - proceeding with sell attempt")
            
            # Execute transaction based on execution mode
            if self.fast_execution_enabled:
                # FAST EXECUTION: Use Helius + Jito (or low-priority for stale positions)
                if low_priority:
                    self.logger.info(f"🐌 Low-priority execution: Dumping stale {symbol} ({tokens_to_sell:.2f} tokens) with minimal fees")
                else:
                    self.logger.info(f"⚡ Fast execution: Selling {percentage*100:.0f}% of {symbol} ({tokens_to_sell:.2f} tokens)")

                tx_result = await self.fast_transaction_submitter.sell(
                    mint=mint_address,
                    token_amount=conservative_sell_amount,
                    slippage_bps=5000,  # 50% slippage - ensures sells execute (prevents stuck positions)
                    low_priority=low_priority  # Use minimal fees for stale positions
                )

                if not tx_result.success:
                    error_msg = tx_result.error or ""
                    self.logger.error(f"Fast sell failed: {error_msg}")

                    # Check for error 3012 (account not initialized) - RPC hasn't indexed token account yet
                    if "3012" in error_msg or "not initialized" in error_msg.lower() or "expected this account to be already initialized" in error_msg.lower():
                        self.logger.warning(f"🔄 ERROR 3012 DETECTED - {symbol}: Token account not indexed yet, triggering smart retry")
                        retry_result = await self._smart_sell_retry(
                            mint_address, symbol, exit_reason
                        )
                        if retry_result.get('success'):
                            return retry_result
                        elif retry_result.get('error') == 'position_recovered':
                            # Position recovered during retry - this is SUCCESS, not failure!
                            self.logger.info(f"🎆 {symbol}: Position recovered during smart retry - sell canceled successfully")
                            position.is_selling = False
                            return retry_result
                        # If still failing after smart retries AND not recovered, fall through to other error handling

                    # Use same error handling as legacy path
                    # Check for balance/indexing issues - trigger smart retry
                    if 'NotEnoughTokensToSell' in error_msg or '0x1787' in error_msg or "not enough tokens" in error_msg.lower() or "insufficient" in error_msg.lower():
                        self.logger.warning(f"🔄 BALANCE/INDEXING ERROR - {symbol}: Triggering smart retry")
                        retry_result = await self._smart_sell_retry(
                            mint_address, symbol, exit_reason
                        )
                        if retry_result.get('success'):
                            return retry_result
                        elif retry_result.get('error') == 'position_recovered':
                            # Position recovered during retry - this is SUCCESS, not failure!
                            self.logger.info(f"🎆 {symbol}: Position recovered during smart retry - sell canceled successfully")
                            position.is_selling = False
                            return retry_result  # Don't do reconciliation for recovered positions
                        # If still failing after smart retries AND not recovered, fall through to reconciliation

                    # Check if this is a traditional slippage error - retry with higher slippage
                    elif self._is_slippage_error(error_msg) and not ('NotEnoughTokensToSell' in error_msg or '0x1787' in error_msg):
                        self.logger.warning(f"🔄 SELL SLIPPAGE ERROR - {symbol}: Retrying with higher slippage (must exit position)")
                        retry_result = await self._retry_sell_with_higher_slippage(
                            mint_address, percentage, symbol, exit_reason
                        )
                        if retry_result.get('success'):
                            return retry_result
                        else:
                            # All retries failed - position is stuck, but keep trying later
                            self.logger.error(f"❌ SELL RETRIES FAILED - {symbol}: Position may be stuck")
                            position.is_selling = False  # Reset flag so we can try again later
                            return retry_result

                    # Final fallback: trigger reconciliation for persistent balance issues
                    if 'NotEnoughTokensToSell' in error_msg or '0x1787' in error_msg or "not enough tokens" in error_msg.lower():
                        self.logger.warning(f"🚫 Balance mismatch detected - entering reconciliation mode for {symbol}")

                        # Mark position as needing reconciliation and preserve current state
                        position.needs_reconciliation = True
                        position.is_selling = False
                        position.pre_reconcile_state = {
                            'tp1_hit': position.tp1_hit_time is not None,
                            'tp2_hit': position.tp2_hit_time is not None,
                            'tp3_hit': position.tp3_hit_time is not None,
                            'last_exit_reason': exit_reason,
                            'attempted_percentage': percentage
                        }

                        self.logger.info(f"⏸️ PAUSING {symbol}: Will reconcile balance and resume monitoring")
                        self.logger.info(f"   Current TP state: TP1={position.tp1_hit_time is not None}, TP2={position.tp2_hit_time is not None}, TP3={position.tp3_hit_time is not None}")

                        # Start reconciliation in background (don't wait)
                        asyncio.create_task(self._reconcile_and_resume(mint_address, symbol))

                        return {"success": False, "error": "reconciliation_triggered", "message": f"Position reconciliation started for {symbol}"}

                    position.is_selling = False  # Reset selling flag on failure
                    return {"success": False, "error": error_msg}
                else:
                    # Sell succeeded on first try
                    tx_signature = tx_result.signature
                    self.logger.info(f"✅ Fast sell executed: {percentage*100:.0f}% of {symbol} - TX: {tx_signature}")

            else:
                # LEGACY EXECUTION: Use QuickNode + PumpPortal
                self.logger.info(f"Creating live sell transaction: {percentage*100:.0f}% of {symbol} position ({tokens_to_sell:.2f} tokens)")

                # Use 20% slippage for live sells (was 3%, too tight for volatile tokens)
                slippage_bps = 2000  # 20% slippage

                tx_result = await self.pumpfun.create_sell_transaction(
                    wallet_pubkey=wallet_pubkey,
                    mint_address=mint_address,
                    token_amount=conservative_sell_amount,
                    slippage_bps=slippage_bps
                )

                if not tx_result.get("success"):
                    self.logger.error(f"Failed to create sell transaction: {tx_result.get('error')}")
                    position.is_selling = False
                    return tx_result

                # Get the base64 transaction
                transaction_b64 = tx_result.get("transaction")
                if not transaction_b64:
                    position.is_selling = False
                    return {"success": False, "error": "No transaction returned"}

                # Sign and send transaction
                self.logger.info("Signing and sending sell transaction...")
                send_result = await self.transaction_signer.sign_and_send_transaction(transaction_b64)

                if not send_result.get("success"):
                    error_msg = send_result.get('error', '')
                    self.logger.error(f"Sell transaction failed: {error_msg}")

                    # Check for error 3012 (account not initialized) - RPC hasn't indexed token account yet
                    if "3012" in str(error_msg) or "not initialized" in str(error_msg).lower() or "expected this account to be already initialized" in str(error_msg).lower():
                        self.logger.warning(f"🔄 ERROR 3012 DETECTED - {symbol}: Token account not indexed yet, triggering smart retry")
                        retry_result = await self._smart_sell_retry(
                            mint_address, symbol, exit_reason
                        )
                        if retry_result.get('success'):
                            return retry_result
                        elif retry_result.get('error') == 'position_recovered':
                            # Position recovered during retry - this is SUCCESS, not failure!
                            self.logger.info(f"🎆 {symbol}: Position recovered during smart retry - sell canceled successfully")
                            position.is_selling = False
                            return retry_result
                        # If still failing after smart retries, fall through to return error

                    position.is_selling = False
                    return send_result

                tx_signature = send_result.get("signature")

            # Common path continues here (P&L calculation, verification, Discord, etc.)
            if tx_signature:
                pnl_pct = ((current_price / position.entry_price) - 1) * 100

                self.logger.info(f"✅ Live sell executed: {percentage*100:.0f}% of {symbol} at {pnl_pct:+.1f}% P&L - TX: {tx_signature}")

                # FAST PATH: Use bonding curve quotes for immediate P&L calculation
                # WAIT FOR VERIFICATION: Before Discord notification, we'll wait for actual proceeds
                # This ensures Discord always gets accurate P&L without slowing down the bot

                # Get SOL price for verification
                sol_price_for_verification = await self._get_sol_price()
                verification_task = None
                if sol_price_for_verification and self.fast_execution_enabled and self.fast_transaction_submitter:
                    # Store verification task so we can await it before Discord notification
                    verification_task = asyncio.create_task(
                        self._verify_sell_proceeds_async(mint_address, tx_signature, symbol, sol_price_for_verification)
                    )

                # Get actual SOL received from transaction logs (most accurate)
                # Note: For fast execution, we'll use bonding curve quote for initial P&L
                actual_sol_received = 0.0

                if self.fast_execution_enabled:
                    # Use bonding curve quote from TransactionResult
                    actual_sol_received = tx_result.sol_received if tx_result.sol_received else 0
                    if actual_sol_received > 0:
                        self.logger.info(f"💵 Bonding curve SOL: {actual_sol_received:.6f} SOL (will verify before Discord)")
                    else:
                        self.logger.warning(f"⚠️ No bonding curve quote available - P&L will be unverified")

                elif self.transaction_signer:
                    # Legacy execution: Parse transaction logs for exact SOL received
                    self.logger.info("Getting exact SOL received from transaction logs...")

                    max_retries = 5
                    retry_delays = [1, 2, 2, 3, 5]  # Progressive delays

                    wallet_address = self.transaction_signer.get_wallet_address()
                    for attempt in range(max_retries):
                        try:
                            # Wait before checking (progressive delays)
                            await asyncio.sleep(retry_delays[attempt])

                            tx_details = await self.transaction_signer.get_transaction_details(tx_signature)

                            if tx_details and wallet_address:
                                actual_sol_received = self.transaction_signer.parse_sol_change_from_logs(
                                    tx_details, wallet_address
                                )
                                if actual_sol_received > 0:
                                    self.logger.info(f"✅ Parsed actual SOL received: {actual_sol_received:.6f} SOL (attempt {attempt+1})")
                                    break
                                else:
                                    self.logger.warning(f"Attempt {attempt+1}: SOL parsing returned 0")
                            else:
                                self.logger.warning(f"Attempt {attempt+1}: Transaction not fully indexed yet")
                        except Exception as e:
                            self.logger.warning(f"Attempt {attempt+1} failed: {e}")

                    if actual_sol_received <= 0:
                        self.logger.error("❌ CRITICAL: Failed to parse actual SOL after all retries - DO NOT TRUST P&L")
                        # NEVER use market estimate for sells - better to report unknown P&L
                        actual_sol_received = 0.0
                else:
                    self.logger.warning("⚠️ No transaction signer available - cannot parse SOL received")
                
                # Calculate cost basis using actual SOL invested (more accurate)
                if hasattr(position, 'sol_invested') and position.sol_invested > 0:
                    # Calculate cost basis from actual SOL invested
                    percentage_sold = conservative_sell_amount / position.tokens_initial if position.tokens_initial > 0 else 1.0
                    sol_cost_basis = position.sol_invested * percentage_sold
                    
                    # Convert to USD using current SOL price
                    sol_price_usd = await self._get_sol_price()
                    
                    if sol_price_usd is None:
                        self.logger.error("Cannot get SOL price - using position tracking for cost basis")
                        cost_basis_usd = conservative_sell_amount * position.avg_cost_per_token
                    else:
                        cost_basis_usd = sol_cost_basis * sol_price_usd
                        self.logger.info(f"Cost basis from SOL: {sol_cost_basis:.6f} SOL (${cost_basis_usd:.2f} at ${sol_price_usd}/SOL)")
                else:
                    # Fallback to position tracking
                    cost_basis_usd = conservative_sell_amount * position.avg_cost_per_token
                    self.logger.warning(f"Using position tracking for cost basis: ${cost_basis_usd:.2f}")
                
                if actual_sol_received > 0:
                    # Get real-time SOL price from blockchain analytics
                    sol_price_usd = await self._get_sol_price()

                    if sol_price_usd is None:
                        self.logger.error("Cannot get SOL price for P&L calculation - marking as unverified")
                        usd_value = 0
                        profit_usd = None  # Mark as unknown
                        profit_pct = 0
                    else:
                        # CRITICAL: When using transaction receipts, actual_sol_spent and actual_sol_received
                        # already include ALL fees (they are wallet deltas). Don't subtract fees again!

                        # Get actual cost basis from position (which now uses actual_sol_spent from receipt)
                        percentage_sold_calc = conservative_sell_amount / position.tokens_initial if position.tokens_initial > 0 else 1.0
                        actual_sol_cost = position.sol_invested * percentage_sold_calc  # This is actual SOL from buy receipt

                        # Simple and 100% accurate P&L calculation using wallet deltas
                        total_cost = actual_sol_cost * sol_price_usd  # What we spent (including all fees)
                        net_proceeds = actual_sol_received * sol_price_usd  # What we received (net of all fees)

                        profit_usd = net_proceeds - total_cost
                        profit_pct = (profit_usd / total_cost) * 100 if total_cost > 0 else 0

                        self.logger.info(f"📊 WALLET DELTA P&L (100% accurate from receipts):")
                        self.logger.info(f"   Total cost: {actual_sol_cost:.6f} SOL = ${total_cost:.2f}")
                        self.logger.info(f"   Total received: {actual_sol_received:.6f} SOL = ${net_proceeds:.2f}")
                        self.logger.info(f"   Profit: {actual_sol_received - actual_sol_cost:+.6f} SOL = ${profit_usd:+.2f} ({profit_pct:+.1f}%)")

                        # Update usd_value for downstream usage
                        usd_value = net_proceeds  # Use net proceeds for consistency

                    if profit_usd is not None:
                        self.logger.info(f"✅ TRUE P&L (with fees): ${total_cost:.2f} total cost → ${net_proceeds:.2f} net received = ${profit_usd:+.2f} ({profit_pct:+.1f}%)")
                    else:
                        self.logger.warning(f"UNVERIFIED Sell P&L: Cannot calculate without SOL price")
                else:
                    # CRITICAL: Cannot determine actual P&L
                    self.logger.error("⚠️ CANNOT VERIFY ACTUAL P&L - Transaction parsing failed")
                    self.logger.error("⚠️ Discord P&L will show UNKNOWN - check wallet manually")
                    usd_value = 0  # Don't report fake values
                    profit_usd = None  # Mark as unknown
                    profit_pct = 0
                
                # Update position tracking
                if self.use_realtime_positions:
                    # NEW: Realtime positions are updated automatically via WebSocket
                    # Just update the old position system for compatibility
                    # For 100% sells, ensure position is fully closed
                    if percentage >= 0.999:
                        position.amount = 0  # Force to 0 for complete exits
                        position.cost_usd_remaining = 0
                        self.logger.info(f"⚡ Position fully closed via 100% sell")
                    else:
                        position.amount = max(0, position.amount - conservative_sell_amount)
                        position.cost_usd_remaining -= cost_basis_usd
                        self.logger.info(f"⚡ Position updated via realtime tracking")
                else:
                    # OLD: Update position using post-transaction balance verification (SLOW)
                    # OPTIMIZATION: For 100% sells, skip verification - we know balance = 0
                    if percentage >= 0.999:
                        self.logger.info(f"⚡ 100% sell - skipping balance verification (position will close)")
                        remaining_balance = 0  # We sold everything
                    else:
                        # For partial sells, verify remaining balance with stale data detection
                        remaining_balance = await self._get_post_transaction_balance(
                            mint_address,
                            pre_sell_balance=position.amount,
                            tokens_sold=conservative_sell_amount
                        )

                    # CRITICAL: If balance query failed, trigger reconciliation to recover!
                    if remaining_balance is None:
                        self.logger.error(f"❌ CRITICAL: Cannot verify balance for {mint_address[:8]}... - triggering reconciliation!")

                        # Calculate estimated remaining balance to continue tracking
                        estimated_remaining = max(0, position.amount - conservative_sell_amount)
                        position.amount = estimated_remaining
                        position.cost_usd_remaining -= cost_basis_usd

                        # Mark for reconciliation to verify actual balance
                        position.needs_reconciliation = True
                        position.is_selling = False
                        position.pre_reconcile_state = {
                            'tp1_hit': position.tp1_hit_time is not None,
                            'tp2_hit': position.tp2_hit_time is not None,
                            'tp3_hit': position.tp3_hit_time is not None,
                            'last_exit_reason': exit_reason,
                            'attempted_percentage': percentage
                        }

                        self.logger.warning(f"⚠️ Using estimated balance: {estimated_remaining:.0f} tokens (sold {conservative_sell_amount:.0f})")
                        self.logger.info(f"⏸️ PAUSING {symbol}: Will reconcile actual balance and resume monitoring")

                        # Start reconciliation in background (don't wait)
                        asyncio.create_task(self._reconcile_and_resume(mint_address, symbol))

                        # Continue execution to log trade and update P&L with estimated values
                        # Reconciliation will correct the balance once it completes
                    else:
                        # Update position to match verified reality
                        position.amount = remaining_balance
                        position.cost_usd_remaining -= cost_basis_usd
                
                # Also update SOL invested amount if we have it
                if hasattr(position, 'sol_invested') and position.sol_invested > 0:
                    percentage_sold = conservative_sell_amount / position.tokens_initial if position.tokens_initial > 0 else 1.0
                    sol_sold = position.sol_invested * percentage_sold
                    position.sol_invested -= sol_sold
                    self.logger.info(f"Updated SOL invested: -{sol_sold:.6f} SOL (remaining: {position.sol_invested:.6f} SOL)")
                
                if position.amount > 0:
                    position.avg_cost_per_token = position.cost_usd_remaining / position.amount
                else:
                    position.avg_cost_per_token = 0.0
                
                # Remove position if no tokens remain OR if we sold 100%
                if position.amount <= 0.0001 or percentage >= 0.999:  # Account for dust or full exit
                    # CRITICAL: Save cost basis for verification task before deleting position
                    # Verification might still be running in background and needs this data
                    if not hasattr(self, '_recent_closed_positions'):
                        self._recent_closed_positions = {}

                    self._recent_closed_positions[mint_address] = {
                        'sol_invested': getattr(position, 'sol_invested', 0),
                        'cost_usd': getattr(position, 'cost_usd_remaining', 0),
                        'closed_at': datetime.now(),
                        'symbol': symbol
                    }

                    # Cleanup old closed positions (keep only last 60 seconds for verification)
                    cutoff_time = datetime.now() - timedelta(seconds=60)
                    self._recent_closed_positions = {
                        k: v for k, v in self._recent_closed_positions.items()
                        if v['closed_at'] > cutoff_time
                    }

                    del self.active_positions[mint_address]
                    tracking_method = "realtime" if self.use_realtime_positions else "blockchain verified"
                    self.logger.info(f"✅ Position fully closed for {mint_address[:8]}... ({tracking_method})")

                    # Also clean up from realtime positions if used
                    if self.use_realtime_positions and hasattr(self, 'realtime_positions') and mint_address in self.realtime_positions.positions:
                        del self.realtime_positions.positions[mint_address]
                        self.logger.info(f"🧹 Cleaned up realtime position for {mint_address[:8]}...")
                else:
                    # Reset selling flag for partial sells
                    position.is_selling = False
                    self.logger.info(f"📊 Partial sell: {position.amount:.0f} tokens remaining")
                
                # Update P&L
                self.pnl_store.add_trade(
                    action="SELL",
                    symbol=actual_symbol,
                    mint_address=mint_address,
                    amount=tokens_to_sell,
                    price=current_price,
                    usd_value=usd_value,
                    paper_mode=False
                )
                
                # Update win rate
                self.total_trades += 1
                if profit_usd is not None and profit_usd > 0:
                    self.winning_trades += 1

                # CRITICAL: Wait for sell verification before sending Discord notification
                # This ensures Discord always gets 100% accurate P&L
                verified_sol_received = None
                verified_profit_usd = profit_usd  # Default to bonding curve estimate
                verified_profit_pct = profit_pct

                if verification_task:
                    try:
                        self.logger.info(f"⏳ Waiting for sell verification before Discord notification...")
                        # Wait up to 15 seconds for verification
                        verified_data = await asyncio.wait_for(verification_task, timeout=15.0)

                        if verified_data:
                            verified_sol_received = verified_data.get('sol_received')
                            verified_cost_sol = verified_data.get('cost_sol')

                            if verified_sol_received and verified_cost_sol and sol_price_for_verification:
                                # Recalculate P&L with verified amounts
                                verified_proceeds = verified_sol_received * sol_price_for_verification
                                verified_cost = verified_cost_sol * sol_price_for_verification
                                verified_profit_usd = verified_proceeds - verified_cost
                                verified_profit_pct = (verified_profit_usd / verified_cost * 100) if verified_cost > 0 else 0

                                self.logger.info(f"✅ Using VERIFIED P&L for Discord: ${verified_profit_usd:+.2f} ({verified_profit_pct:+.1f}%)")

                                # Update variables for Discord notification
                                usd_value = verified_proceeds
                                profit_usd = verified_profit_usd
                                profit_pct = verified_profit_pct

                                # CRITICAL: Update P&L store with verified amounts
                                # This replaces the bonding curve quote-based SELL entry with accurate data
                                self.pnl_store.add_trade(
                                    action="SELL_VERIFIED",
                                    symbol=actual_symbol,
                                    mint_address=mint_address,
                                    amount=tokens_to_sell,
                                    price=current_price,
                                    usd_value=verified_proceeds,  # Use verified proceeds, not quote!
                                    realized_pnl=verified_profit_usd,  # Use verified P&L!
                                    paper_mode=False
                                )
                                self.logger.info(f"💾 P&L store updated with VERIFIED amounts: ${verified_proceeds:.2f} proceeds, ${verified_profit_usd:+.2f} P&L")
                    except asyncio.TimeoutError:
                        self.logger.warning(f"⚠️ Verification timeout (15s) - using bonding curve estimate for Discord")
                    except Exception as e:
                        self.logger.error(f"Error waiting for verification: {e}")

                # Send Discord notification (now with verified P&L if available)
                if self.notifier:
                    await self.notifier.send_trade_notification(
                        side="SELL",
                        symbol=actual_symbol,
                        mint_address=mint_address,
                        quantity=tokens_to_sell,
                        price=current_price,
                        usd_amount=usd_value,
                        equity=self.pnl_store.current_equity,
                        realized_pnl=profit_usd,  # Now uses verified P&L if available
                        paper_mode=False
                    )
                
                return {
                    "success": True,
                    "price": current_price,  # Add missing price key for consistency
                    "tx_signature": tx_signature,
                    "tokens_sold": tokens_to_sell,
                    "usd_received": usd_value,  # Renamed for clarity
                    "usd_value": usd_value,     # Keep both for compatibility
                    "profit": profit_usd,       # Match paper trading format
                    "profit_pct": profit_pct,
                    "symbol": symbol,
                    "paper_mode": False
                }
            else:
                error_msg = send_result.get('error', '')
                self.logger.error(f"Failed to send sell transaction: {error_msg}")
                
                # Handle sell failures - distinguish between slippage and indexing delays
                buy_age = (datetime.now() - position.entry_time).total_seconds()
                
                # For NotEnoughTokensToSell on recent buys, it's likely an indexing delay
                if ('NotEnoughTokensToSell' in error_msg or '0x1787' in error_msg) and buy_age < 120:
                    # Record this failure for future availability checks
                    self._record_sell_failure(mint_address, 'NotEnoughTokensToSell')
                    self.logger.warning(f"🕐 INDEXING DELAY - {symbol}: Buy is {buy_age:.0f}s old, retrying with smart evaluation")
                    retry_result = await self._smart_sell_retry(
                        mint_address, symbol, exit_reason
                    )
                    if retry_result.get('success'):
                        return retry_result
                    elif retry_result.get('error') == 'position_recovered':
                        # Position recovered during retry - this is SUCCESS, not failure!
                        self.logger.info(f"🎆 {symbol}: Position recovered during smart retry - sell canceled successfully")
                        position.is_selling = False
                        return retry_result  # Don't do reconciliation for recovered positions
                    # If still failing after smart retries AND not recovered, fall through to reconciliation
                    
                # Check if this is a traditional slippage error - retry with higher slippage
                elif self._is_slippage_error(error_msg) and not ('NotEnoughTokensToSell' in error_msg or '0x1787' in error_msg):
                    self.logger.warning(f"🔄 SELL SLIPPAGE ERROR - {symbol}: Retrying with higher slippage (must exit position)")
                    retry_result = await self._retry_sell_with_higher_slippage(
                        mint_address, percentage, symbol, exit_reason
                    )
                    if retry_result.get('success'):
                        return retry_result
                    else:
                        # All retries failed - position is stuck, but keep trying later
                        self.logger.error(f"❌ SELL RETRIES FAILED - {symbol}: Position may be stuck")
                        position.is_selling = False  # Reset flag so we can try again later
                        return retry_result
                
                # Final fallback: trigger reconciliation for persistent balance issues
                if 'NotEnoughTokensToSell' in error_msg or '0x1787' in error_msg:
                    self.logger.warning(f"🚫 Balance mismatch detected - entering reconciliation mode for {symbol}")
                    
                    # Mark position as needing reconciliation and preserve current state
                    position.needs_reconciliation = True
                    position.is_selling = False
                    position.pre_reconcile_state = {
                        'tp1_hit': position.tp1_hit_time is not None,
                        'tp2_hit': position.tp2_hit_time is not None,
                        'tp3_hit': position.tp3_hit_time is not None,
                        'last_exit_reason': exit_reason,
                        'attempted_percentage': percentage
                    }
                    
                    self.logger.info(f"⏸️ PAUSING {symbol}: Will reconcile balance and resume monitoring")
                    self.logger.info(f"   Current TP state: TP1={position.tp1_hit_time is not None}, TP2={position.tp2_hit_time is not None}, TP3={position.tp3_hit_time is not None}")
                    
                    # Start reconciliation in background (don't wait)
                    asyncio.create_task(self._reconcile_and_resume(mint_address, symbol))
                    
                    return {"success": False, "error": "reconciliation_triggered", "message": f"Position reconciliation started for {symbol}"}
                
                position.is_selling = False  # Reset selling flag on failure
                return send_result
            
        except Exception as e:
            self.logger.error(f"Error in real sell execution: {e}")
            # Reset selling flag on exception
            if mint_address in self.active_positions:
                self.active_positions[mint_address].is_selling = False
            return {"success": False, "error": str(e)}

    async def check_exit_conditions(self, mint_address: str, current_price: float = 0) -> Optional[tuple]:
        """
        Enhanced exit strategy with dynamic TP1 sizing and intelligent trailing stops
        Returns: (exit_reason, sell_percentage) or None
        """
        if mint_address not in self.active_positions:
            return None
        
        position = self.active_positions[mint_address]
        
        # Get current price
        if current_price <= 0:
            return None
        
        # Calculate current gain/loss percentage
        current_gain_pct = ((current_price / position.entry_price) - 1) * 100
        hold_time_seconds = (datetime.now() - position.entry_time).total_seconds()
        
        # Enhanced logging for exit decision monitoring
        self.logger.debug(f"EXIT CHECK {mint_address[:8]}... "
                         f"Price: ${current_price:.8f}, "
                         f"Gain: {current_gain_pct:+.1f}%, "
                         f"Hold: {hold_time_seconds:.0f}s, "
                         f"Peak: {position.high_gain_peak:.1f}%, "
                         f"TP1Hit: {position.tp1_hit_time is not None}")
        
        # Update peak price and high gain
        if current_price > position.peak_price:
            position.peak_price = current_price
            position.high_gain_peak = max(position.high_gain_peak, current_gain_pct)
        
        # NEW: Check for scratch exit conditions (early weakness detection)
        scratch_enabled = getattr(self.config, 'scratch_rule', {}).get('enabled', True)
        if scratch_enabled and hold_time_seconds <= 45:
            scratch_result = self.should_scratch_exit(position, current_price)
            
            if scratch_result['should_scratch']:
                self.logger.info(f"🔸 SCRATCH EXIT triggered for {mint_address[:8]}...: {scratch_result['reason']}")
                self.logger.info(f"   Exiting at {current_gain_pct:.1f}% loss to prevent further drawdown")
                return ("scratch", 1.0)
        
        # PHASE 3.1: Multi-Tier Exit Strategy
        multi_tier_config = getattr(self.config, 'multi_tier_exits', {})
        use_multi_tier = multi_tier_config.get('enabled', False)
        
        if use_multi_tier:
            # Multi-tier exit strategy for aggressive mode with buffer strategy values
            # Use buffer strategy TP values if enabled, otherwise use original aggressive values
            buffer_config = getattr(self.config, 'volatility_buffer', {})
            if buffer_config.get('enabled', False):
                # Buffer strategy TPs: 20%, 45%, 90%
                tp1_multiplier = getattr(self.config, 'tp_multiplier', 1.20)  # Default to 20% gain
                tp2_price = position.entry_price * getattr(self.config, 'tp2_multiplier', 1.45)  # 45% gain
                tp3_price = position.entry_price * getattr(self.config, 'tp3_multiplier', 1.90)  # 90% gain
                
                # Buffer strategy sell percentages: 40%, 35%, 20%
                tp1_sell_pct = multi_tier_config.get('tp1_sell_pct', 0.40)
                tp2_sell_pct = multi_tier_config.get('tp2_sell_pct', 0.35)
                tp3_sell_pct = multi_tier_config.get('tp3_sell_pct', 0.20)
            else:
                # Original aggressive mode values
                tp2_price = position.entry_price * getattr(self.config, 'tp2_multiplier', 1.50)
                tp3_price = position.entry_price * getattr(self.config, 'tp3_multiplier', 2.00)
                tp1_sell_pct = multi_tier_config.get('tp1_sell_pct', 0.45)
                tp2_sell_pct = multi_tier_config.get('tp2_sell_pct', 0.35)
                tp3_sell_pct = multi_tier_config.get('tp3_sell_pct', 0.15)
            
            # TP1 - First take profit
            # SAFETY MARGIN: Increase TP threshold by 5% for unverified positions (compensates for quote inaccuracy)
            is_verified = getattr(position, 'actual_costs_verified', False)
            safety_margin = 1.0 if is_verified else 1.05  # 5% extra buffer for unverified
            tp1_price_adjusted = position.tp_price * safety_margin

            if current_price >= tp1_price_adjusted and position.tp1_hit_time is None:
                position.tp1_hit_time = datetime.now()
                position.tp1_percentage_sold = tp1_sell_pct

                if not is_verified:
                    self.logger.warning(f"⚠️ TP1 triggered with UNVERIFIED costs (+5% safety margin applied)")
                    self.logger.info(f"   Target: ${position.tp_price:.8f}, Adjusted: ${tp1_price_adjusted:.8f}")

                self.logger.info(f"🎯 TP1 HIT: Selling {tp1_sell_pct*100:.0f}% at +{current_gain_pct:.1f}% gain")
                self.logger.info(f"   Current price: ${current_price:.8f}, TP1 target: ${position.tp_price:.8f}")
                return ("take_profit_partial", tp1_sell_pct)

            # TP2 - Second take profit
            if current_price >= tp2_price and position.tp2_hit_time is None and position.tp1_hit_time is not None:
                position.tp2_hit_time = datetime.now()
                position.tp2_percentage_sold = tp2_sell_pct
                self.logger.info(f"🚀 TP2 HIT: Selling {tp2_sell_pct*100:.0f}% at +{current_gain_pct:.1f}% gain")
                self.logger.info(f"   Current price: ${current_price:.8f}, TP2 target: ${tp2_price:.8f}")
                return ("take_profit_partial", tp2_sell_pct)

            # TP3 - Third take profit
            if current_price >= tp3_price and position.tp3_hit_time is None and position.tp2_hit_time is not None:
                position.tp3_hit_time = datetime.now()
                position.tp3_percentage_sold = tp3_sell_pct
                self.logger.info(f"🌙 TP3 HIT: Selling {tp3_sell_pct*100:.0f}% at +{current_gain_pct:.1f}% gain (Moonshot!)")
                self.logger.info(f"   Current price: ${current_price:.8f}, TP3 target: ${tp3_price:.8f}")
                return ("take_profit_partial", tp3_sell_pct)
        else:
            # Original single TP1 logic for conservative mode
            # SAFETY MARGIN: Apply same 5% buffer for unverified positions
            is_verified = getattr(position, 'actual_costs_verified', False)
            safety_margin = 1.0 if is_verified else 1.05
            tp_price_adjusted = position.tp_price * safety_margin

            if current_price >= tp_price_adjusted and position.tp1_hit_time is None:
                if not is_verified:
                    self.logger.warning(f"⚠️ TP triggered with UNVERIFIED costs (+5% safety margin)")

                position.tp1_hit_time = datetime.now()
                time_to_tp1 = (position.tp1_hit_time - position.entry_time).total_seconds()
                
                if time_to_tp1 < 30:  # Ultra-fast spike
                    tp1_percentage = 0.08
                elif time_to_tp1 < 60:  # Fast move
                    tp1_percentage = 0.12
                elif time_to_tp1 < 180:  # Normal move
                    tp1_percentage = 0.20
                else:  # Slow move
                    tp1_percentage = 0.25
                
                position.tp1_percentage_sold = tp1_percentage
                self.logger.info(f"TP1: Selling {tp1_percentage*100:.0f}% after {time_to_tp1:.0f}s")
                return ("take_profit_partial", tp1_percentage)
        
        # PHASE 3.2: Optimized Intelligent Trailing Stops
        
        # Check velocity stop FIRST (rapid crash protection)
        if await self._check_velocity_stop(position, current_price):
            return ("velocity_stop", 1.0)
        
        # Check profit floor (ratchet system)
        profit_floor = self._get_profit_floor(current_gain_pct)
        if current_gain_pct < profit_floor:
            self.logger.info(f"💎 PROFIT FLOOR: Protecting gains at {profit_floor:.0f}% minimum")
            return ("profit_floor", 1.0)
        
        # MOONSHOT trailing with wider stops as requested
        if position.high_gain_peak >= 400:  # Massive moonshot
            trailing_pct = 0.60  # Trail at 60% of peak - 40% drawdown allowed
            trailing_stop = position.peak_price * trailing_pct
            if current_price <= trailing_stop:
                self.logger.info(f"🚀 MOONSHOT EXIT (400%+): {position.high_gain_peak:.0f}% peak, exiting at {current_gain_pct:.0f}%")
                return ("trailing_stop_moonshot_massive", 1.0)
        elif position.high_gain_peak >= 200:  # Strong moonshot
            trailing_pct = 0.65  # Trail at 65% of peak - 35% drawdown allowed
            trailing_stop = position.peak_price * trailing_pct
            if current_price <= trailing_stop:
                self.logger.info(f"🌙 MOONSHOT EXIT (200%+): {position.high_gain_peak:.0f}% peak, exiting at {current_gain_pct:.0f}%")
                return ("trailing_stop_moonshot_strong", 1.0)
        elif position.high_gain_peak >= 100:  # Regular moonshot
            trailing_pct = 0.70  # Trail at 70% of peak - 30% drawdown allowed
            trailing_stop = position.peak_price * trailing_pct
            if current_price <= trailing_stop:
                self.logger.info(f"✨ MOONSHOT EXIT (100%+): {position.high_gain_peak:.0f}% peak, exiting at {current_gain_pct:.0f}%")
                return ("trailing_stop_moonshot", 1.0)
        elif position.high_gain_peak >= 60:  # High gains
            trailing_pct = 0.80  # Trail at 80% of peak
            trailing_stop = position.peak_price * trailing_pct
            if current_price <= trailing_stop:
                return ("trailing_stop_high_gain", 1.0)
        elif position.high_gain_peak >= 30 and hold_time_seconds < 120:  # +30% in first 2 min
            trailing_pct = 0.85  # Trail at 85% of peak
            trailing_stop = position.peak_price * trailing_pct
            if current_price <= trailing_stop:
                return ("trailing_stop_fast_gain", 1.0)
        
        # Multi-tier aware break-even and trailing logic
        total_sold_pct = position.tp1_percentage_sold + position.tp2_percentage_sold + position.tp3_percentage_sold
        has_banked_profits = total_sold_pct > 0
        
        # Check if we're in buffer period first - no break-even or stop losses during buffer
        in_buffer_period = False
        if position.buffer_end_time:
            in_buffer_period = datetime.now() < position.buffer_end_time
            if in_buffer_period:
                time_remaining = (position.buffer_end_time - datetime.now()).total_seconds()
                self.logger.debug(f"🛡️ Buffer period active: {time_remaining:.0f}s remaining (no stops/break-even)")
        
        # Skip ALL stop/break-even logic during buffer period (but allow TPs)
        if in_buffer_period:
            # During buffer, we only check for take profits (TPs are handled earlier in the function)
            # No break-even stops, no trailing stops, no stop losses
            pass
        # Check for buffer strategy's remaining position management (AFTER buffer period)
        elif getattr(self.config, 'volatility_buffer', {}).get('enabled', False) and \
             getattr(self.config, 'remaining_position', {}).get('enabled', False) and has_banked_profits:
            remaining_config = getattr(self.config, 'remaining_position', {})
            # Buffer strategy: After TP1, protect gains but let winners run
            profit_protection_level = remaining_config.get('profit_protection_level', 15.0)  # Default 15%
            trailing_stop_pct = remaining_config.get('trailing_stop_percentage', 20.0)  # Default 20% trailing
            max_remaining_hold = remaining_config.get('max_remaining_hold_seconds', 10800)  # 3 hours default
            
            # Check if we should exit based on profit protection level
            if position.tp1_hit_time and current_gain_pct < profit_protection_level:
                self.logger.info(f"📉 PROFIT PROTECTION: Price dropped below +{profit_protection_level:.0f}% after TP1")
                return ("profit_protection", 1.0)
            
            # Apply wider trailing stop for remaining position
            if position.tp1_hit_time:
                trailing_threshold = position.peak_price * (1 - trailing_stop_pct / 100)
                if current_price <= trailing_threshold:
                    self.logger.info(f"📊 REMAINING POSITION TRAILING: {trailing_stop_pct:.0f}% drawdown from peak")
                    return ("remaining_trailing_stop", 1.0)
            
            # Check max hold time for remaining position
            if position.tp1_hit_time:
                time_since_tp1 = (datetime.now() - position.tp1_hit_time).total_seconds()
                if time_since_tp1 >= max_remaining_hold:
                    self.logger.info(f"⏰ MAX REMAINING HOLD: {max_remaining_hold/3600:.1f} hours reached for remaining position")
                    return ("max_remaining_hold", 1.0)
        else:
            # Original break-even and trailing logic for non-buffer strategies
            # Break-even stop logic - more aggressive after taking profits
            break_even_threshold = 5 if has_banked_profits else 8
            if current_gain_pct >= break_even_threshold and not position.break_even_armed:
                position.break_even_armed = True
                position.break_even_armed_time = datetime.now()
                self.logger.info(f"Break-even stop armed for {mint_address[:8]}... at +{current_gain_pct:.1f}% (banked: {has_banked_profits})")
            
            if position.break_even_armed:
                time_since_armed = (datetime.now() - position.break_even_armed_time).total_seconds()
                if time_since_armed <= 60:  # Break-even protection for 60 seconds
                    buffer = 1.005 if has_banked_profits else 1.01  # Tighter buffer after profits
                    if current_price <= position.entry_price * buffer:
                        return ("break_even_stop", 1.0)
                else:
                    # After 60 seconds, switch to normal trailing
                    position.trailing_stop_active = True
                    position.break_even_armed = False
            
            # Adaptive trailing stop based on profit-taking progress
            if position.trailing_stop_active or current_gain_pct >= 15:
                # Tighter trailing after taking multi-tier profits
                if has_banked_profits:
                    if total_sold_pct >= 0.80:  # After TP1+TP2 (80% sold)
                        trailing_pct = 0.95  # Very tight 5% trail on final 20%
                    elif total_sold_pct >= 0.45:  # After TP1 (45% sold)
                        trailing_pct = 0.90  # Moderate 10% trail on remaining 55%
                    else:
                        trailing_pct = 0.85  # Standard trail
                else:
                    trailing_pct = 0.85  # Standard trail for conservative mode
                    
                trailing_stop = position.peak_price * trailing_pct
                if current_price <= trailing_stop:
                    trail_pct = int((1 - trailing_pct) * 100)
                    self.logger.info(f"Trailing stop hit: {trail_pct}% drawdown from peak (banked {total_sold_pct:.0%})")
                    return ("trailing_stop", 1.0)
        
        # STOP LOSS SYSTEM - No buffer period, straight stops
        # Hard 35% catastrophe stop
        catastrophe_stop = position.entry_price * 0.65
        if current_price <= catastrophe_stop:
            self.logger.warning(f"🛑 CATASTROPHE STOP: -35% hard stop hit")
            return ("catastrophe_stop", 1.0)
        
        # ATR-based dynamic stop (if we have enough data)
        atr_stop = await self._calculate_atr_stop(position, hold_time_seconds)
        if atr_stop > 0 and current_price <= atr_stop:
            stop_pct = ((atr_stop / position.entry_price) - 1) * 100
            self.logger.info(f"📊 ATR STOP: Volatility-adjusted stop at {stop_pct:.1f}%")
            return ("atr_stop", 1.0)
        
        # AGGRESSIVE TIME-BASED EXITS (Friend's Strategy Refined)
        max_hold = getattr(self.config, 'max_hold_seconds', 1800)  # 30 minutes for aggressive
        hold_time_minutes = hold_time_seconds / 60
        
        # Aggressive time-based take profits
        if hold_time_minutes >= 20:
            # After 20 minutes: Take profit at 10% (refined from 5%)
            if current_gain_pct >= 10:
                self.logger.info(f"⏰ TIME-BASED TP: 20min+ hold, taking 10% profit")
                return ("time_based_tp_20min", 1.0)
        elif hold_time_minutes >= 15:
            # After 15 minutes: Take profit at 15% (refined from 5%)
            if current_gain_pct >= 15:
                self.logger.info(f"⏰ TIME-BASED TP: 15min+ hold, taking 15% profit")
                return ("time_based_tp_15min", 1.0)
        
        # Final safety exit after 30 minutes
        if hold_time_seconds > max_hold:
            # After 30 minutes: Take profit at 5% (refined from -5% loss)
            if current_gain_pct >= 5:
                self.logger.info(f"⏰ MAX HOLD REACHED: Taking 5% profit after {hold_time_minutes:.0f} minutes")
                return ("max_hold_tp", 1.0)
            # If not profitable, let stop loss handle it
        
        return None

    def should_scratch_exit(self, position: Position, current_price: float, 
                           recent_trades: list = None) -> Dict:
        """
        Determine if position should be scratched due to early weakness
        
        Args:
            position: Current position data
            current_price: Current token price
            recent_trades: Recent trade data for buyer acceleration
            
        Returns:
            Dictionary with scratch decision
        """
        current_time = datetime.now()
        entry_time = position.entry_time
        
        # Only consider scratch in first 45 seconds
        hold_time_seconds = (current_time - entry_time).total_seconds()
        if hold_time_seconds > 45:
            return {
                'should_scratch': False,
                'reason': 'Outside scratch window (>45s)'
            }
        
        if position.entry_price <= 0 or current_price <= 0:
            return {
                'should_scratch': False,
                'reason': 'Invalid price data'
            }
        
        # Calculate current P&L
        current_pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        # Calculate peak-to-current drawdown
        peak_drawdown = (position.peak_price - current_price) / position.peak_price if position.peak_price > 0 else 0
        
        # Check if drawdown exceeds threshold
        drawdown_threshold = 0.04  # 4%
        if peak_drawdown < drawdown_threshold:
            return {
                'should_scratch': False,
                'reason': f'Drawdown {peak_drawdown:.1%} < {drawdown_threshold:.1%}'
            }
        
        # Check buyer acceleration (simplified - we'll estimate without recent_trades for now)
        buyer_accel = self._estimate_buyer_acceleration(position.mint)
        
        # Scratch conditions
        is_negative_accel = buyer_accel < 0
        scratch_threshold_low = -0.02   # -2%
        scratch_threshold_high = -0.03  # -3%
        
        should_scratch = (
            peak_drawdown >= drawdown_threshold and
            is_negative_accel and
            scratch_threshold_high <= current_pnl_pct <= scratch_threshold_low
        )
        
        return {
            'should_scratch': should_scratch,
            'reason': f'Drawdown: {peak_drawdown:.1%}, Accel: {buyer_accel:.1f}, P&L: {current_pnl_pct:.1%}',
            'peak_drawdown': peak_drawdown,
            'buyer_acceleration': buyer_accel,
            'current_pnl_pct': current_pnl_pct
        }

    def _estimate_buyer_acceleration(self, mint_address: str) -> float:
        """
        Estimate buyer acceleration without external data
        For now, return 0 (neutral) - can be enhanced later with real-time data
        
        Returns:
            Buyer acceleration (positive = accelerating, negative = decelerating)
        """
        # TODO: Implement real buyer acceleration calculation when real-time data is available
        # For now, return slightly negative to be conservative
        return -0.5
    
    async def _calculate_atr_stop(self, position: Position, hold_time_seconds: float) -> float:
        """
        Calculate ATR-based trailing stop that adapts to volatility
        
        Args:
            position: Current position
            hold_time_seconds: How long we've held the position
            
        Returns:
            ATR-based stop price
        """
        try:
            # Get recent price history if available
            if not hasattr(position, 'price_history'):
                # Initialize price history tracking if not present
                position.price_history = []
                return 0  # No ATR calculation possible yet
            
            # Need at least 20 price points for ATR (40 seconds at 2s intervals)
            # This prevents premature ATR calculations on brand new positions
            if len(position.price_history) < 20:
                return 0
            
            # Calculate simple ATR (average true range)
            price_changes = []
            for i in range(1, len(position.price_history[-45:])):
                change = abs(position.price_history[i] - position.price_history[i-1])
                price_changes.append(change)
            
            if not price_changes:
                return 0
                
            atr = sum(price_changes) / len(price_changes)
            
            # Determine multiplier based on unrealized gain
            current_price = position.price_history[-1] if position.price_history else position.entry_price
            unrealized_gain = (current_price / position.entry_price - 1) * 100
            
            # AGGRESSIVE TRADING: Much wider stops to handle volatility
            # Only protect against real dumps, not normal price action
            if unrealized_gain >= 300:
                k = 6.0  # Ultra wide for moonshots
            elif unrealized_gain >= 100:
                k = 5.0  # Very wide for strong gains
            elif unrealized_gain >= 50:
                k = 4.5  # Wide for good gains
            elif hold_time_seconds < 60:
                # First minute: ULTRA WIDE tolerance - Pump.fun tokens are extremely volatile
                # Need to overcome ~$3-5 in fees per round trip
                k = 12.0  # Ultra ultra wide - only catch catastrophic -15%+ dumps
            elif hold_time_seconds < 180:
                # First 3 minutes: still very tolerant
                k = 8.0  # Ultra wide for new positions establishing direction
            else:
                # After 3 minutes: still more tolerant than before
                k = 6.0  # Very wide stop for aggressive trading
            
            # Calculate ATR-based stop
            atr_stop = position.peak_price - (k * atr)
            
            return atr_stop
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR stop: {e}")
            return 0
    
    async def _check_velocity_stop(self, position: Position, current_price: float) -> bool:
        """
        Check for rapid price drops (velocity stop)
        
        Args:
            position: Current position
            current_price: Current token price
            
        Returns:
            True if velocity stop triggered
        """
        try:
            # Store price history for velocity calculation
            if not hasattr(position, 'price_history'):
                position.price_history = []
            
            # Add current price to history
            position.price_history.append(current_price)
            
            # Keep only last 60 seconds of prices (assuming 2s polling)
            max_history = 30  # 30 prices = ~60 seconds at 2s intervals
            if len(position.price_history) > max_history:
                position.price_history = position.price_history[-max_history:]
            
            # Need at least 5 price points (10 seconds) for velocity
            if len(position.price_history) < 5:
                return False
            
            # Check 10-second velocity (5 price points at 2s intervals)
            price_10s_ago = position.price_history[-5]
            velocity = (current_price - price_10s_ago) / price_10s_ago
            
            # Trigger on rapid drops
            velocity_threshold = -0.10  # -10% in 10 seconds
            
            if velocity < velocity_threshold:
                self.logger.warning(f"🚨 VELOCITY STOP: {velocity*100:.1f}% drop in 10s")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking velocity stop: {e}")
            return False
    
    def _get_profit_floor(self, unrealized_gain_pct: float) -> float:
        """
        Get the profit floor (ratchet) based on unrealized gains
        
        Args:
            unrealized_gain_pct: Current unrealized gain percentage
            
        Returns:
            Minimum acceptable gain percentage (profit floor)
        """
        # Profit ratchet system - lock in gains at milestones
        if unrealized_gain_pct >= 400:
            return 250  # Lock in 250% minimum
        elif unrealized_gain_pct >= 200:
            return 120  # Lock in 120% minimum
        elif unrealized_gain_pct >= 100:
            return 40   # Lock in 40% minimum
        else:
            return -35  # Default to -35% stop loss
    
    async def _calculate_dynamic_stop_loss(self, mint_address: str, position: Position, hold_time_seconds: float) -> float:
        """
        Calculate stop loss based on buffer strategy configuration
        
        Args:
            mint_address: Token mint address
            position: Current position
            hold_time_seconds: How long we've held the position
            
        Returns:
            Stop loss price level
        """
        try:
            # Always use hard 35% stop as catastrophe protection
            catastrophe_stop = position.entry_price * 0.65  # 35% stop loss
            
            # Calculate ATR-based stop if we have price history
            atr_stop = await self._calculate_atr_stop(position, hold_time_seconds)
            
            # Use the tighter of the two stops
            stop_price = max(catastrophe_stop, atr_stop) if atr_stop > 0 else catastrophe_stop
            
            # Never let stop price go above entry (no positive stops)
            stop_price = min(stop_price, position.entry_price * 0.99)
            
            return stop_price
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            # Fallback to 15% stop
            return position.entry_price * 0.85

    # Note: update_position_prices removed - position monitoring is now handled by main.py:monitor_position
    # which calls check_exit_conditions directly for better integration

    def get_portfolio_summary(self) -> Dict:
        """Get summary of current portfolio"""
        total_invested = 0
        total_current_value = 0
        position_count = len(self.active_positions)
        
        for position in self.active_positions.values():
            invested = position.sol_invested * 20  # Mock SOL price
            total_invested += invested
            
            # Would need current price to calculate current value
            # For now, use entry price
            current_value = position.amount * position.entry_price
            total_current_value += current_value
        
        unrealized_pnl = total_current_value - total_invested
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            "paper_capital": self.paper_capital,
            "total_invested": total_invested,
            "current_value": total_current_value,
            "unrealized_pnl": unrealized_pnl,
            "position_count": position_count,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate
        }

    async def _attempt_reconciliation(self, mint_address: str) -> Optional[float]:
        """Try to get actual balance with exponential backoff until successful"""
        position = self.active_positions.get(mint_address)
        if not position:
            return None

        max_attempts = 10
        base_delay = 1  # Start with 1 second
        buy_age = (datetime.now() - position.entry_time).total_seconds()

        for attempt in range(max_attempts):
            try:
                # Support both fast execution and legacy paths
                actual_balance = None
                if self.fast_execution_enabled and self.fast_transaction_submitter:
                    actual_balance = await self.fast_transaction_submitter.get_token_balance(mint_address)
                elif self.transaction_signer:
                    actual_balance = await self.transaction_signer.get_token_balance(mint_address)

                if actual_balance is not None:
                    self.logger.info(f"📊 Reconciliation attempt {attempt + 1}: {actual_balance:,.0f} tokens (buy age: {buy_age:.0f}s)")
                    
                    # CONSERVATIVE: Don't accept 0 balance for recent successful buys
                    if actual_balance == 0 and buy_age < 180:  # 3 minutes
                        self.logger.warning(f"⚠️ Got 0 balance for recent buy ({buy_age:.0f}s old) - likely indexing delay, continuing...")
                        # Don't return 0, continue retrying
                    else:
                        # Accept any valid balance > 0, or 0 for old positions
                        return actual_balance
            except Exception as e:
                self.logger.warning(f"Reconciliation attempt {attempt + 1} failed: {e}")
            
            # Exponential backoff: 1s, 2s, 4s, 8s, 16s, etc.
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)
                self.logger.info(f"⏳ Retrying reconciliation in {delay}s...")
                await asyncio.sleep(delay)
        
        return None  # Failed to reconcile
    
    async def _resume_after_reconciliation(self, mint_address: str, actual_balance: float) -> None:
        """Resume monitoring from current state, preserving TP progress"""
        position = self.active_positions.get(mint_address)
        if not position:
            return
            
        old_balance = position.amount
        # Position doesn't have symbol field, use mint address
        symbol = mint_address[:8] + "..."
        buy_age = (datetime.now() - position.entry_time).total_seconds()
        
        # CRITICAL: Never create ghost positions with 0 tokens
        if actual_balance == 0:
            self.logger.error(f"🚫 RECONCILIATION FAILED - {symbol}: Got 0 tokens for {buy_age:.0f}s old position")
            self.logger.error(f"   Original estimate: {old_balance:,.0f} tokens")
            self.logger.error(f"   Removing position to prevent ghost trading")
            
            # Remove the position entirely rather than creating a ghost position
            if mint_address in self.active_positions:
                del self.active_positions[mint_address]
            return
        
        # Update position with reality
        position.amount = actual_balance
        position.needs_reconciliation = False
        
        # Recalculate cost basis proportionally
        if position.tokens_initial > 0:
            balance_ratio = actual_balance / position.tokens_initial
            position.cost_usd_remaining = position.cost_usd_remaining * balance_ratio
            position.avg_cost_per_token = position.cost_usd_remaining / actual_balance if actual_balance > 0 else 0
        
        # CRITICAL: Don't reset TP progress - keep existing TP states intact
        self.logger.info(f"✅ RECONCILED {symbol}: {old_balance:,.0f} → {actual_balance:,.0f} tokens")
        self.logger.info(f"   TP Status: TP1={position.tp1_hit_time is not None}, "
                        f"TP2={position.tp2_hit_time is not None}, TP3={position.tp3_hit_time is not None}")
        self.logger.info(f"   Cost basis: ${position.cost_usd_remaining:.2f} (${position.avg_cost_per_token:.6f}/token)")
        
        # Resume normal monitoring - it will continue from current TP level
        self.logger.info(f"🎯 {symbol} monitoring resumed - ready for next exit signal")
    
    async def _reconcile_and_resume(self, mint_address: str, symbol: str) -> None:
        """Background task to reconcile and resume trading"""
        self.logger.info(f"🔄 Starting reconciliation for {symbol}...")

        # Check if position still exists before attempting reconciliation
        position = self.active_positions.get(mint_address)
        if not position:
            self.logger.warning(f"⚠️ Reconciliation skipped for {symbol} - position already closed/removed")
            return

        actual_balance = await self._attempt_reconciliation(mint_address)

        if actual_balance is not None:
            await self._resume_after_reconciliation(mint_address, actual_balance)
        else:
            self.logger.error(f"❌ Failed to reconcile {symbol} - all balance queries failed after retries")
            # Keep position paused but don't break it completely
            position.needs_reconciliation = True  # Keep it paused

    async def _get_transaction_details_with_retry(self, tx_signature: str, mint_address: str) -> Optional[Dict]:
        """Get transaction details with exponential backoff retry"""
        max_attempts = 5
        base_delay = 1  # Start with 1 second
        
        for attempt in range(max_attempts):
            try:
                # Try to get transaction amounts first (most accurate)
                verified_data = await self._get_transaction_token_amounts(tx_signature)
                
                # Check if transaction failed
                if verified_data and verified_data.get('transaction_failed'):
                    self.logger.error(f"❌ Buy transaction failed: {verified_data.get('error', 'Unknown error')}")
                    return verified_data  # Return immediately with failure flag
                
                if verified_data and verified_data.get('tokens_received', 0) > 0:
                    self.logger.info(f"✅ Got transaction details on attempt {attempt + 1}: {verified_data['tokens_received']:,.0f} tokens")
                    return verified_data
                    
                # Fallback to balance query if transaction details not ready
                if self.transaction_signer:
                    actual_balance = await self.transaction_signer.get_token_balance(mint_address)
                    if actual_balance and actual_balance > 0:
                        verified_data = {'tokens_received': actual_balance}
                        self.logger.info(f"✅ Got balance on attempt {attempt + 1}: {actual_balance:,.0f} tokens")
                        return verified_data
                    elif actual_balance == 0:
                        # Balance is 0 - check if transaction actually failed with exponential backoff
                        self.logger.warning(f"⚠️ Balance is 0 for {mint_address[:8]}... - checking transaction status")
                        
                        # Try to get transaction details with retries
                        tx_check_attempts = 4
                        tx_check_delay = 1.0  # Start with 1 second
                        
                        for tx_attempt in range(tx_check_attempts):
                            tx_details = await self.transaction_signer.get_transaction_details(tx_signature)
                            
                            # If transaction found and we can check status
                            if tx_details and not tx_details.get("_not_found"):
                                if tx_details.get("_transaction_failed"):
                                    self.logger.error(f"❌ Transaction FAILED: {tx_details.get('_error_detail', 'Unknown error')}")
                                    return {"transaction_failed": True, "error": tx_details.get("_error_detail")}
                                else:
                                    # Transaction succeeded - trust it even if balance shows 0
                                    self.logger.warning(f"⚠️ Transaction SUCCEEDED but balance shows 0 - likely indexing delay")
                                    # Return None to continue main retry loop which will keep checking balance
                                    return None
                            
                            # If not the last attempt, wait before retrying
                            if tx_attempt < tx_check_attempts - 1:
                                self.logger.debug(f"Transaction not indexed yet, retrying in {tx_check_delay}s...")
                                await asyncio.sleep(tx_check_delay)
                                # Progressive backoff: 1s -> 3s -> 5s -> 8s
                                if tx_check_delay < 5:
                                    tx_check_delay = min(tx_check_delay + 2, 5)
                                else:
                                    tx_check_delay = 8  # Final longer wait
                        
                        # If we still can't confirm status after all attempts, continue retrying balance
                        # Don't assume failure just because transaction isn't indexed yet
                        self.logger.warning(f"⚠️ Transaction not indexed after {tx_check_attempts} checks, continuing balance retry")
                        # Return None to continue main retry loop
                        
            except Exception as e:
                self.logger.warning(f"Transaction details attempt {attempt + 1} failed: {e}")
            
            # Exponential backoff: 1s, 2s, 4s, 8s, 16s
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)
                self.logger.info(f"⏳ Retrying transaction details in {delay}s...")
                await asyncio.sleep(delay)
        
        self.logger.error(f"❌ Failed to get transaction details after {max_attempts} attempts")
        return None

    def get_position_details(self, mint_address: str) -> Optional[Dict]:
        """Get details for a specific position"""
        if mint_address not in self.active_positions:
            return None
        
        position = self.active_positions[mint_address]
        
        return {
            "mint": position.mint,
            "entry_price": position.entry_price,
            "amount": position.amount,
            "sol_invested": position.sol_invested,
            "entry_time": position.entry_time.isoformat(),
            "tp_price": position.tp_price,
            "sl_price": position.sl_price,
            "peak_price": position.peak_price,
            "paper_mode": position.paper_mode
        }

    async def close_all_positions(self):
        """Close all active positions"""
        for mint_address in list(self.active_positions.keys()):
            try:
                position = self.active_positions[mint_address]
                await self.sell_token(mint_address, 1.0, position.paper_mode)
                self.logger.info(f"Closed position {mint_address}")
            except Exception as e:
                self.logger.error(f"Error closing position {mint_address}: {e}")

    async def send_error_notification(self, message: str, context: Dict = None):
        """Send error notification to Discord"""
        if self.notifier:
            await self.notifier.send_error_notification(message, context)

    async def _create_verified_position(self, mint_address: str, tx_signature: str, 
                                      sol_amount: float, usd_amount: float, symbol: str):
        """
        🚀 NEW CLEAN FLOW: Create position only after QuickNode verification
        
        This method:
        1. Waits for QuickNode to index the transaction 
        2. Gets verified token amounts from blockchain
        3. Creates position with accurate data
        4. Catches up position state (TP/SL checks)
        5. Sends notifications
        """
        try:
            self.logger.info(f"🔍 Starting verification for {symbol} TX: {tx_signature[:16]}...")
            
            # Wait for QuickNode verification with exponential backoff
            verified_data = await self._wait_for_verification(mint_address, tx_signature)
            
            # Check if transaction failed
            if verified_data and verified_data.get('transaction_failed'):
                self.logger.error(f"🚫 Buy transaction FAILED for {symbol} - not creating position")
                await self.send_error_notification(f"Buy transaction failed: {symbol}", {
                    'mint': mint_address,
                    'tx_signature': tx_signature,
                    'error': verified_data.get('error', 'Unknown error')
                })
                return
            
            if not verified_data:
                self.logger.error(f"❌ Verification failed for {symbol} - no position created")
                await self.send_error_notification(f"Position verification failed: {symbol}", {
                    'mint': mint_address,
                    'tx_signature': tx_signature
                })
                return
            
            # Extract verified data
            actual_tokens = verified_data['tokens_received']
            
            # Safety check: Prevent division by zero
            if actual_tokens <= 0:
                self.logger.error(f"❌ Invalid token amount: {actual_tokens} - cannot create position")
                await self.send_error_notification(f"Invalid token verification: {symbol}", {
                    'mint': mint_address,
                    'tx_signature': tx_signature,
                    'tokens_received': actual_tokens
                })
                return
                
            actual_fill_price = usd_amount / actual_tokens
            
            # Safety check: Prevent zero entry price
            if actual_fill_price <= 0:
                self.logger.error(f"❌ Invalid fill price: ${actual_fill_price} - cannot create position")
                await self.send_error_notification(f"Invalid fill price: {symbol}", {
                    'mint': mint_address,
                    'tx_signature': tx_signature,
                    'fill_price': actual_fill_price
                })
                return
            
            self.logger.info(f"✅ Verification complete: {actual_tokens:,.0f} tokens at ${actual_fill_price:.8f}")
            
            # Create verified position
            # Get buffer configuration
            buffer_config = getattr(self.config, 'volatility_buffer', {})
            buffer_seconds = buffer_config.get('buffer_seconds', 0) if buffer_config.get('enabled', False) else 0
            entry_time = datetime.now()
            
            position = Position(
                mint=mint_address,
                symbol=symbol,
                entry_price=actual_fill_price,
                amount=actual_tokens,
                sol_invested=sol_amount,
                entry_time=entry_time,
                tp_price=actual_fill_price * self.config.tp_multiplier,
                sl_price=actual_fill_price * self.config.stop_loss_pct,
                peak_price=actual_fill_price,
                paper_mode=False,
                tokens_initial=actual_tokens,
                cost_usd_remaining=usd_amount,
                avg_cost_per_token=actual_fill_price,
                buy_tx_signature=tx_signature,
                verified_from_blockchain=True,
                buffer_period_seconds=buffer_seconds,
                buffer_end_time=datetime.fromtimestamp(entry_time.timestamp() + buffer_seconds) if buffer_seconds > 0 else None
            )
            
            self.active_positions[mint_address] = position
            
            # Record in P&L store
            self.pnl_store.add_trade(
                action="BUY",
                symbol=symbol,
                mint_address=mint_address,
                amount=actual_tokens,
                price=actual_fill_price,
                usd_value=usd_amount,
                paper_mode=False
            )
            
            self.logger.info(f"📊 Position created: {symbol} - {actual_tokens:,.0f} tokens")
            
            # Register position with realtime system if enabled
            if self.use_realtime_positions:
                trade_event = {
                    'mint': mint_address,
                    'action': 'buy',
                    'tx_signature': tx_signature,
                    'price': actual_fill_price,
                    'tokens_received': actual_tokens,
                    'sol_amount': sol_amount
                }
                self.realtime_positions.handle_trade_event(trade_event)
                self.logger.info(f"📊 Position registered with realtime system: {symbol}")
            
            # 🎯 CATCH-UP LOGIC: Check if we should immediately TP/SL based on current price
            await self._catchup_position_state(mint_address, symbol)
            
            # Send success notification
            if self.notifier:
                await self.notifier.send_trade_notification(
                    side="BUY",
                    symbol=symbol,
                    mint_address=mint_address,
                    quantity=actual_tokens,
                    price=actual_fill_price,
                    usd_amount=usd_amount,
                    equity=self.pnl_store.current_equity,
                    paper_mode=False
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error in verified position creation: {e}")
            await self.send_error_notification(f"Position creation error: {symbol}", {
                'error': str(e),
                'mint': mint_address,
                'tx_signature': tx_signature
            })

    async def _wait_for_verification(self, mint_address: str, tx_signature: str) -> Optional[Dict]:
        """
        Wait for QuickNode to index the buy transaction with robust retry logic
        Returns verified data or None if verification fails
        """
        max_attempts = 12  # Up to ~2 minutes total
        base_delay = 2
        
        for attempt in range(max_attempts):
            try:
                # Get current token balance from blockchain
                # Support both fast execution and legacy paths
                if self.fast_execution_enabled and self.fast_transaction_submitter:
                    current_balance = await self.fast_transaction_submitter.get_token_balance(mint_address)
                elif self.transaction_signer:
                    current_balance = await self.transaction_signer.get_token_balance(mint_address)
                else:
                    current_balance = None
                
                if current_balance and current_balance > 0:
                    self.logger.info(f"✅ Verification successful on attempt {attempt + 1}: {current_balance:,.0f} tokens")
                    return {
                        'tokens_received': current_balance,
                        'verified_at': datetime.now(),
                        'verification_attempts': attempt + 1
                    }
                
                # After a few attempts, check if the transaction actually failed
                if attempt >= 3:  # Check after 3 attempts (~15-20 seconds)
                    tx_details = await self.transaction_signer.get_transaction_details(tx_signature)
                    if tx_details and tx_details.get("_transaction_failed"):
                        self.logger.error(f"🚫 Buy transaction FAILED: {tx_details.get('_error_detail', 'Unknown error')}")
                        return {"transaction_failed": True, "error": tx_details.get("_error_detail")}
                
                if attempt < max_attempts - 1:
                    # Exponential backoff with jitter
                    delay = min(base_delay * (1.5 ** attempt), 15)  # Cap at 15s
                    self.logger.info(f"⏳ Attempt {attempt + 1}/{max_attempts}: No tokens yet, retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Verification attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(base_delay * (1.2 ** attempt))
        
        self.logger.error(f"❌ Verification failed after {max_attempts} attempts")
        return None
    
    async def _reconcile_position(self, mint_address: str, tx_signature: str, symbol: str):
        """
        Reconcile immediate position with actual transaction data
        Only adjusts if there's a significant difference
        """
        try:
            self.logger.info(f"🔄 Reconciling position: {symbol}")
            
            if mint_address not in self.active_positions:
                self.logger.warning(f"No position to reconcile for {mint_address}")
                return
            
            position = self.active_positions[mint_address]
            
            # Get actual tokens with retry logic (includes transaction details + balance fallback)
            verified_data = await self._get_transaction_details_with_retry(tx_signature, mint_address)
            
            # Check if transaction failed
            if verified_data and verified_data.get('transaction_failed'):
                self.logger.error(f"🚫 Buy transaction FAILED for {symbol} - canceling position")
                # Remove the failed position
                if mint_address in self.active_positions:
                    # Log any erroneous TP hits before removal
                    if position.tp1_hit_time:
                        self.logger.warning(f"⚠️ Removing false TP1 hit for failed transaction")
                    del self.active_positions[mint_address]
                    self.logger.info(f"✅ Removed failed position for {symbol}")
                    
                    # Also remove from any sell locks
                    if hasattr(self, '_active_sells') and mint_address in self._active_sells:
                        del self._active_sells[mint_address]
                return
            
            if verified_data and verified_data.get('tokens_received', 0) > 0:
                actual_tokens = verified_data['tokens_received']
                estimated_tokens = position.amount
                
                # Check if adjustment needed (>2% difference)
                diff_pct = abs(actual_tokens - estimated_tokens) / estimated_tokens * 100
                
                if diff_pct > 2:
                    self.logger.info(f"📊 Adjusting position: {estimated_tokens:,.0f} → {actual_tokens:,.0f} tokens ({diff_pct:.1f}% difference)")

                    # Adjust position amounts
                    old_entry_price = position.entry_price
                    new_entry_price = position.cost_usd_remaining / actual_tokens

                    position.amount = actual_tokens
                    position.tokens_initial = actual_tokens
                    position.entry_price = new_entry_price
                    position.avg_cost_per_token = new_entry_price
                    position.tp_price = new_entry_price * self.config.tp_multiplier
                    position.sl_price = new_entry_price * self.config.stop_loss_pct
                    position.verified_from_blockchain = True

                    self.logger.info(f"✅ Position reconciled: entry price ${old_entry_price:.8f} → ${new_entry_price:.8f}")
                else:
                    self.logger.info(f"✅ Position accurate: {diff_pct:.1f}% difference (no adjustment needed)")
                    position.verified_from_blockchain = True

                # CRITICAL FIX: Record BUY trade in P&L store (was missing!)
                # This position was created immediately, so we need to record it now after verification
                self.pnl_store.add_trade(
                    action="BUY",
                    symbol=symbol,
                    mint_address=mint_address,
                    amount=actual_tokens,
                    price=position.entry_price,
                    usd_value=position.cost_usd_remaining,
                    paper_mode=False
                )
                self.logger.info(f"✅ BUY trade recorded in P&L store: {actual_tokens:,.0f} tokens at ${position.entry_price:.8f}")
            else:
                # Could not verify - keep position but mark as unverified
                self.logger.warning(f"⚠️ Could not verify transaction for {symbol} - keeping position with estimated amounts")
                position.verified_from_blockchain = False
                # Don't remove needs_reconciliation flag - keep it paused for safety
                
        except Exception as e:
            self.logger.error(f"Error reconciling position: {e}")
    
    async def _get_transaction_token_amounts(self, tx_signature: str) -> Optional[Dict]:
        """Get token amounts from transaction logs (faster than ATA query)"""
        try:
            if not self.transaction_signer:
                return None
            
            tx_details = await self.transaction_signer.get_transaction_details(tx_signature)
            if not tx_details:
                return None
            
            # Check if transaction failed
            if tx_details.get("_transaction_failed"):
                self.logger.warning(f"🚫 Transaction failed: {tx_details.get('_error_detail', 'Unknown error')}")
                return {"transaction_failed": True, "error": tx_details.get("_error_detail")}
            
            # Get wallet address
            wallet_address = self.transaction_signer.get_wallet_address()
            
            # Parse transaction logs for token amounts
            meta = tx_details.get("meta", {})
            post_balances = meta.get("postTokenBalances", [])
            pre_balances = meta.get("preTokenBalances", [])
            
            # Calculate token balance change
            post_amount = 0
            pre_amount = 0
            
            for balance in post_balances:
                if balance.get("owner") == wallet_address:
                    post_amount = float(balance.get("uiTokenAmount", {}).get("uiAmount", 0))
                    break
            
            for balance in pre_balances:
                if balance.get("owner") == wallet_address:
                    pre_amount = float(balance.get("uiTokenAmount", {}).get("uiAmount", 0))
                    break
            
            tokens_received = post_amount - pre_amount
            
            if tokens_received > 0:
                return {"tokens_received": tokens_received}
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting transaction amounts: {e}")
            return None

    async def _catchup_position_state(self, mint_address: str, symbol: str):
        """
        🎯 Catch-up logic: Check if position should immediately TP/SL based on current market price
        This handles the case where price moved significantly during the verification delay
        """
        try:
            if mint_address not in self.active_positions:
                return
                
            position = self.active_positions[mint_address]
            current_price = await self.moralis.get_current_price(mint_address, fresh=True)
            
            if current_price <= 0:
                self.logger.warning(f"⚠️ Cannot get current price for catchup: {symbol}")
                return
            
            # Safety check: Prevent division by zero
            if position.entry_price <= 0:
                self.logger.error(f"❌ Invalid entry price {position.entry_price} - skipping catchup for {symbol}")
                return
            
            # Update peak price if needed
            if current_price > position.peak_price:
                position.peak_price = current_price
                self.logger.info(f"📈 New peak: {symbol} ${current_price:.8f}")
            
            # Calculate current P&L (now safe from division by zero)
            current_pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            
            self.logger.info(f"🎯 Catchup check: {symbol} entry=${position.entry_price:.8f}, "
                           f"current=${current_price:.8f}, P&L={current_pnl_pct:+.1f}%")
            
            # Check TP conditions
            tp_threshold_pct = ((position.tp_price - position.entry_price) / position.entry_price) * 100
            
            if current_pnl_pct >= tp_threshold_pct:
                self.logger.info(f"🎯 CATCHUP TP TRIGGER: {symbol} hit {current_pnl_pct:+.1f}% (TP at {tp_threshold_pct:+.1f}%)")
                
                # Execute TP1 (use your existing TP percentages)
                tp1_percentage = 0.45  # 45% as mentioned in your strategy
                
                sell_result = await self.sell_token(
                    mint_address=mint_address,
                    percentage=tp1_percentage,
                    paper_mode=False,
                    symbol=symbol,
                    exit_reason="catchup_tp1"
                )
                
                if sell_result.get("success"):
                    position.tp1_hit_time = datetime.now()
                    position.tp1_percentage_sold = tp1_percentage
                    self.logger.info(f"✅ Catchup TP1 executed: {symbol} sold {tp1_percentage*100:.0f}%")
                else:
                    self.logger.error(f"❌ Catchup TP1 failed: {sell_result.get('error')}")
            
            # Check SL conditions
            sl_threshold_pct = ((position.sl_price - position.entry_price) / position.entry_price) * 100
            
            if current_pnl_pct <= sl_threshold_pct:
                self.logger.warning(f"🎯 CATCHUP SL TRIGGER: {symbol} hit {current_pnl_pct:+.1f}% (SL at {sl_threshold_pct:+.1f}%)")
                
                # Mark position as selling to prevent race conditions
                if hasattr(position, 'is_selling'):
                    if position.is_selling:
                        self.logger.info(f"🔒 Catchup SL skipped - {symbol} already being sold")
                        return
                    position.is_selling = True
                else:
                    position.is_selling = True
                
                sell_result = await self.sell_token(
                    mint_address=mint_address,
                    percentage=1.0,  # Full position
                    paper_mode=False,
                    symbol=symbol,
                    exit_reason="catchup_stop_loss"
                )
                
                if sell_result.get("success"):
                    self.logger.info(f"✅ Catchup SL executed: {symbol} position closed")
                else:
                    self.logger.error(f"❌ Catchup SL failed: {sell_result.get('error')}")
                    # Reset selling flag on failure
                    position.is_selling = False
            
        except Exception as e:
            self.logger.error(f"❌ Error in catchup logic for {symbol}: {e}")
    
    async def send_summary(self):
        """Send portfolio summary to Discord with blockchain-verified data"""
        if not self.notifier:
            return
            
        # Try to get blockchain-verified data if available (non-blocking)
        if self.blockchain_analytics and self.transaction_signer:
            try:
                # Get wallet address
                wallet_address = self.transaction_signer.get_wallet_address()
                
                # Fetch blockchain data asynchronously (with timeout to prevent blocking)
                blockchain_task = asyncio.create_task(
                    self.blockchain_analytics.get_accurate_portfolio_stats(wallet_address)
                )
                
                # Wait max 5 seconds for blockchain data
                try:
                    blockchain_data = await asyncio.wait_for(blockchain_task, timeout=5.0)
                    
                    # Calculate win rate from today's trades
                    wins = self.pnl_store.data.get("winning_trades", 0)
                    losses = self.pnl_store.data.get("losing_trades", 0) 
                    total_trades = wins + losses
                    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
                    
                    # Use blockchain-verified data for Discord
                    await self.notifier.send_summary(
                        equity=blockchain_data["current_equity_usd"],
                        daily_pnl=blockchain_data["net_usd_change"],
                        total_trades=total_trades,
                        win_rate=win_rate,
                        active_positions=len(self.active_positions)
                    )
                    
                    self.logger.info(f"📊 Sent blockchain-verified summary: ${blockchain_data['net_usd_change']:+.2f} P&L")
                    return
                    
                except asyncio.TimeoutError:
                    self.logger.warning("⚠️ Blockchain data fetch timed out, using local data")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Could not fetch blockchain data: {e}")
        
        # Fallback to local P&L store data
        summary = self.pnl_store.get_summary()
        await self.notifier.send_summary(
            equity=summary["equity"],
            daily_pnl=summary["daily_pnl"],
            total_trades=summary["total_trades"],
            win_rate=summary["win_rate"],
            active_positions=len(self.active_positions)
        )
    
    def get_position_tier_status(self, position: Position) -> Dict:
        """
        Get current tier status for multi-tier exit strategy
        
        Returns:
            Dictionary with tier information and remaining position percentage
        """
        total_sold = position.tp1_percentage_sold + position.tp2_percentage_sold + position.tp3_percentage_sold
        remaining_pct = 1.0 - total_sold
        
        tier_status = {
            'tp1_hit': position.tp1_hit_time is not None,
            'tp2_hit': position.tp2_hit_time is not None, 
            'tp3_hit': position.tp3_hit_time is not None,
            'total_sold_pct': total_sold,
            'remaining_pct': remaining_pct,
            'current_tier': 'runner' if position.tp3_hit_time else 'tp3_zone' if position.tp2_hit_time else 'tp2_zone' if position.tp1_hit_time else 'accumulation'
        }
        
        return tier_status
    
    def should_skip_safety_checks(self, wallet_confidence: float, wallet_tier: str, signal_strength: float) -> str:
        """
        Determine what safety checks to skip based on wallet intelligence
        
        Args:
            wallet_confidence: Confidence score of the triggering wallet (0-100)
            wallet_tier: Tier of the wallet (S, A, B, C, Unknown)
            signal_strength: Combined signal strength
            
        Returns:
            "all" - Skip all safety checks
            "partial" - Skip sellability only (keep price impact)
            "none" - Full safety checks
        """
        if not self.use_hybrid_safety:
            return "none"  # Conservative mode - full safety
            
        # Ultra-high confidence: Skip all safety
        ultra_threshold = self.safety_hybrid.get('ultra_confidence_threshold', 80)
        signal_bypass = self.safety_hybrid.get('signal_strength_bypass', 4.0)
        s_tier_bypass = self.safety_hybrid.get('s_tier_bypass', True)
        
        if (wallet_confidence >= ultra_threshold and 
            wallet_tier == "S" and 
            signal_strength >= signal_bypass and
            s_tier_bypass):
            self.logger.info(f"🚀 ULTRA-FAST EXECUTION: S-tier wallet ({wallet_confidence:.0f}% confidence, {signal_strength:.1f} signal)")
            return "all"
        
        # High confidence: Skip sellability only
        high_threshold = self.safety_hybrid.get('high_confidence_threshold', 70)
        if wallet_confidence >= high_threshold and signal_strength >= 3.5:
            self.logger.info(f"⚡ FAST EXECUTION: High confidence ({wallet_confidence:.0f}%, {signal_strength:.1f} signal)")
            return "partial"
        
        # Everything else: Full safety checks
        self.logger.info(f"🛡️  SAFE EXECUTION: Full safety checks ({wallet_confidence:.0f}%, {signal_strength:.1f} signal)")
        return "none"

    async def _verify_buy_costs_async(self, mint_address: str, tx_signature: str, symbol: str, sol_price: float):
        """
        Background task to verify actual SOL spent from transaction receipt.
        Keeps retrying until success to ensure accurate P&L tracking.
        """
        try:
            self.logger.info(f"🔄 Background verification started for {symbol} buy (tx: {tx_signature[:8]}...)")

            # Retry for up to 15 seconds (more generous for indexing delay)
            max_attempts = 30
            delay = 0.5  # 500ms between attempts

            for attempt in range(max_attempts):
                # Check if position still exists (might have been sold already)
                if mint_address not in self.active_positions:
                    self.logger.info(f"⏭️ Position {symbol} already closed - skipping verification")
                    return

                position = self.active_positions[mint_address]

                # Check if already verified (shouldn't happen, but safety check)
                if getattr(position, 'actual_costs_verified', False):
                    self.logger.debug(f"✅ Position {symbol} already verified - skipping")
                    return

                # Try to get actual SOL spent from transaction receipt
                verification = await self.fast_transaction_submitter.get_transaction_sol_delta(tx_signature, is_buy=True)

                # Check if transaction definitively FAILED on-chain (ghost position!)
                if verification.status == "failed_on_chain":
                    self.logger.error(f"👻 GHOST POSITION DETECTED: Buy transaction {tx_signature[:8]}... FAILED ON-CHAIN!")
                    self.logger.error(f"   Token: {symbol} ({mint_address[:8]}...)")
                    self.logger.error(f"   Error: {verification.error_detail}")
                    self.logger.error(f"   Removing ghost position from tracking...")

                    # Delete the ghost position
                    if mint_address in self.active_positions:
                        del self.active_positions[mint_address]
                        self.logger.info(f"✅ Ghost position removed for {symbol}")

                        # Clean up related tracking structures
                        if hasattr(self, '_active_sells') and mint_address in self._active_sells:
                            del self._active_sells[mint_address]

                        # Remove from realtime positions if using that system
                        if self.use_realtime_positions and hasattr(self, 'realtime_positions') and mint_address in self.realtime_positions.positions:
                            del self.realtime_positions.positions[mint_address]

                    # Log a failed trade in P&L (no cost since transaction failed)
                    self.pnl_store.add_trade(
                        action="BUY_FAILED",
                        symbol=symbol,
                        mint_address=mint_address,
                        amount=0,
                        price=0,
                        usd_value=0,
                        paper_mode=False
                    )

                    self.logger.info(f"✅ Ghost position cleanup complete for {symbol}")
                    return  # Position deleted, exit verification

                # Check if transaction succeeded
                if verification.status == "success" and verification.sol_amount:
                    # SUCCESS! Update position with actual costs
                    actual_sol_spent = verification.sol_amount
                    quote_estimate = getattr(position, 'unverified_sol_invested', position.sol_invested)
                    discrepancy = actual_sol_spent - quote_estimate
                    discrepancy_pct = (discrepancy / quote_estimate * 100) if quote_estimate > 0 else 0

                    self.logger.info(f"✅ VERIFIED {symbol} buy costs after {attempt + 1} attempts ({(attempt + 1) * delay:.1f}s)")
                    self.logger.info(f"   Quote estimate: {quote_estimate:.6f} SOL")
                    self.logger.info(f"   Actual spent: {actual_sol_spent:.6f} SOL")
                    self.logger.info(f"   Discrepancy: {discrepancy:+.6f} SOL ({discrepancy_pct:+.1f}%)")

                    # Update position with actual costs
                    position.sol_invested = actual_sol_spent
                    position.actual_costs_verified = True

                    # Update cost basis in USD
                    actual_usd_cost = actual_sol_spent * sol_price
                    position.cost_usd_remaining = actual_usd_cost
                    position.avg_cost_per_token = actual_usd_cost / position.amount if position.amount > 0 else 0

                    # Update P&L store with corrected values
                    self.pnl_store.add_trade(
                        action="BUY_VERIFIED",
                        symbol=symbol,
                        mint_address=mint_address,
                        amount=position.amount,
                        price=position.avg_cost_per_token,
                        usd_value=actual_usd_cost,
                        paper_mode=False
                    )

                    return  # SUCCESS!

                # Status is "not_found" - transaction not indexed yet, wait and retry
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay)

            # Failed to verify after all attempts
            self.logger.warning(f"⚠️ Failed to verify {symbol} buy costs after {max_attempts} attempts ({max_attempts * delay:.0f}s)")
            self.logger.warning(f"   Position will use unverified bonding curve quote for P&L calculations")

        except Exception as e:
            self.logger.error(f"Error in buy cost verification for {symbol}: {e}", exc_info=True)

    async def _verify_sell_proceeds_async(self, mint_address: str, tx_signature: str, symbol: str, sol_price: float):
        """
        Background task to verify actual SOL received from transaction receipt.
        Keeps retrying until success to ensure accurate P&L tracking.

        Returns:
            dict with 'sol_received' and 'cost_sol' if successful, None otherwise
        """
        try:
            self.logger.info(f"🔄 Background verification started for {symbol} sell (tx: {tx_signature[:8]}...)")

            # Retry for up to 15 seconds
            max_attempts = 30
            delay = 0.5

            for attempt in range(max_attempts):
                # Try to get actual SOL received from transaction receipt
                verification = await self.fast_transaction_submitter.get_transaction_sol_delta(tx_signature, is_buy=False)

                # Check if sell transaction failed on-chain
                if verification.status == "failed_on_chain":
                    self.logger.error(f"❌ Sell transaction {tx_signature[:8]}... FAILED ON-CHAIN!")
                    self.logger.error(f"   Token: {symbol} ({mint_address[:8]}...)")
                    self.logger.error(f"   Error: {verification.error_detail}")
                    self.logger.error(f"   Sell failed - position still exists")
                    # Note: Position still exists, sell just failed
                    return None

                # Check if sell succeeded
                if verification.status == "success" and verification.sol_amount and verification.sol_amount > 0:
                    # SUCCESS! Get cost basis from position (if still exists) or from buy verification
                    actual_sol_received = verification.sol_amount
                    actual_usd_received = actual_sol_received * sol_price

                    self.logger.info(f"✅ VERIFIED {symbol} sell proceeds after {attempt + 1} attempts ({(attempt + 1) * delay:.1f}s)")
                    self.logger.info(f"   Actual received: {actual_sol_received:.6f} SOL (${actual_usd_received:.2f})")

                    # Try to get cost basis from active position (might still exist during partial sells)
                    # or from recently closed position data
                    cost_sol = None
                    if mint_address in self.active_positions:
                        position = self.active_positions[mint_address]
                        if hasattr(position, 'sol_invested'):
                            cost_sol = position.sol_invested
                    elif hasattr(self, '_recent_closed_positions') and mint_address in self._recent_closed_positions:
                        # Position was closed - get cost from cache
                        cost_sol = self._recent_closed_positions[mint_address].get('sol_invested')
                        self.logger.debug(f"Retrieved cost basis from closed position cache: {cost_sol:.6f} SOL")

                    return {
                        'sol_received': actual_sol_received,
                        'cost_sol': cost_sol,
                        'usd_received': actual_usd_received
                    }

                # Status is "not_found" - transaction not indexed yet, wait and retry
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay)

            # Failed to verify after all attempts
            self.logger.warning(f"⚠️ Failed to verify {symbol} sell proceeds after {max_attempts} attempts ({max_attempts * delay:.0f}s)")
            return None

        except Exception as e:
            self.logger.error(f"Error in sell proceeds verification for {symbol}: {e}", exc_info=True)
            return None

    async def _get_sol_price(self) -> Optional[float]:
        """Get current SOL price in USD"""
        if self.blockchain_analytics:
            price = await self.blockchain_analytics.get_sol_price()
            return price if price > 0 else None
        return None
    
    def _is_slippage_error(self, error_msg: str) -> bool:
        """
        Check if an error is due to slippage/price movement
        
        Args:
            error_msg: Error message from transaction failure
            
        Returns:
            True if error is likely due to slippage/price movement
        """
        if not error_msg:
            return False
            
        error_lower = error_msg.lower()
        
        # Slippage-related errors (don't retry these - we're too late)
        slippage_indicators = [
            'slippage',
            'toomuchsolrequired',
            '0x1772',  # TooMuchSolRequired error code
            '0x1787',  # NotEnoughTokensToSell - can be slippage-related
            'notenoughtokenstosell',
            'price impact',
            'price moved',
            'insufficient output',
            'minimum received',
            'swap failed'
        ]
        
        for indicator in slippage_indicators:
            if indicator in error_lower:
                self.logger.info(f"🚫 Detected slippage error: '{indicator}' in '{error_msg}'")
                return True
                
        return False
    
    async def _retry_sell_with_higher_slippage(self, mint_address: str, percentage: float, 
                                              symbol: str, exit_reason: str, max_retries: int = 2) -> Dict:
        """
        Retry sell transaction with progressively higher slippage tolerance
        We MUST exit positions somehow, so we retry sells even if we skip buy retries
        """
        if mint_address not in self.active_positions:
            return {"success": False, "error": "No position found for sell retry"}
            
        position = self.active_positions[mint_address]
        base_slippage = 150  # Start with 1.5% for sells (higher than buys)
        
        for retry_attempt in range(max_retries):
            # Increase slippage: 1.5% -> 2.5% -> 3.5%
            retry_slippage = base_slippage + ((retry_attempt + 1) * 100)
            
            self.logger.warning(f"🔄 SELL Retry {retry_attempt + 1}/{max_retries}: Increasing slippage to {retry_slippage/100:.1f}%")
            
            try:
                # Get wallet info
                wallet_pubkey = await self.transaction_signer.get_wallet_pubkey()
                
                # Calculate tokens to sell
                tokens_to_sell = position.amount * percentage
                
                # Create sell transaction with higher slippage
                tx_result = await self.pumpfun.create_sell_transaction(
                    wallet_pubkey=wallet_pubkey,
                    mint_address=mint_address,
                    token_amount=tokens_to_sell,
                    slippage_bps=retry_slippage
                )
                
                if not tx_result.get("success"):
                    error_msg = tx_result.get('error', '')
                    self.logger.warning(f"🔄 SELL Retry {retry_attempt + 1} creation failed: {error_msg}")
                    
                    if self._is_slippage_error(error_msg):
                        continue  # Try with even higher slippage
                    else:
                        return tx_result  # Non-slippage error, give up
                
                # Try to send the transaction
                transaction_b64 = tx_result.get("transaction")
                if not transaction_b64:
                    self.logger.error(f"🔄 SELL Retry {retry_attempt + 1}: No transaction returned")
                    continue
                
                send_result = await self.transaction_signer.sign_and_send_transaction(transaction_b64)
                
                if send_result.get("success"):
                    tx_signature = send_result.get("signature")
                    self.logger.info(f"✅ SELL Retry {retry_attempt + 1} SUCCESS: {percentage*100:.0f}% sold with {retry_slippage/100:.1f}% slippage - TX: {tx_signature}")
                    
                    # Get current price for P&L calculation
                    current_price = await self.moralis.get_current_price(mint_address, fresh=True)
                    
                    # Update position and handle P&L (simplified but functional)
                    old_amount = position.amount
                    tokens_to_sell = old_amount * percentage
                    position.amount = old_amount * (1 - percentage)
                    position.is_selling = False
                    
                    # Calculate P&L
                    usd_value = tokens_to_sell * current_price if current_price > 0 else 0
                    cost_basis = tokens_to_sell * position.avg_cost_per_token
                    profit_usd = usd_value - cost_basis
                    profit_pct = (profit_usd / cost_basis * 100) if cost_basis > 0 else 0
                    
                    # Update P&L store
                    if hasattr(self, 'pnl_store'):
                        self.pnl_store.add_trade(
                            action="SELL",
                            symbol=symbol,
                            mint_address=mint_address,
                            amount=tokens_to_sell,
                            price=current_price,
                            usd_value=usd_value,
                            paper_mode=False
                        )
                    
                    # If position fully sold, remove it
                    if position.amount <= 0.01:  # Small threshold for rounding
                        del self.active_positions[mint_address]
                        self.logger.info(f"🎯 Position {symbol} fully closed via slippage retry")
                    
                    return {
                        "success": True,
                        "tx_signature": tx_signature,
                        "tokens_sold": tokens_to_sell,
                        "usd_received": usd_value,
                        "profit": profit_usd,
                        "profit_pct": profit_pct,
                        "symbol": symbol,
                        "slippage_used": retry_slippage,
                        "retry_attempt": retry_attempt + 1,
                        "paper_mode": False
                    }
                else:
                    retry_error = send_result.get('error', '')
                    self.logger.warning(f"🔄 SELL Retry {retry_attempt + 1} send failed: {retry_error}")
                    
                    if not self._is_slippage_error(retry_error):
                        return send_result  # Non-slippage error, give up
                        
            except Exception as e:
                self.logger.error(f"🔄 SELL Retry {retry_attempt + 1} exception: {e}")
                continue
        
        self.logger.error(f"❌ All {max_retries} SELL retries exhausted for {symbol}")
        position.is_selling = False  # Reset flag so monitoring will try again later
        return {"success": False, "error": f"All {max_retries} sell retries failed"}
    
    async def _smart_sell_retry(self, mint_address: str, symbol: str, original_exit_reason: str, max_retries: int = 4) -> Dict:
        """
        Smart sell retry that re-evaluates position conditions on each attempt
        Handles indexing delays while ensuring we don't sell at the wrong price/conditions
        """
        if mint_address not in self.active_positions:
            return {"success": False, "error": "No position found for smart retry"}
            
        position = self.active_positions[mint_address]
        base_delay = 2  # Start with 2 second delays
        
        self.logger.info(f"🧠 SMART RETRY starting for {symbol}: will re-evaluate conditions on each attempt")
        
        for retry_attempt in range(max_retries):
            # Wait for indexing to catch up
            delay = base_delay * (1.5 ** retry_attempt)  # 2s, 3s, 4.5s, 6.8s
            if retry_attempt > 0:
                self.logger.info(f"⏳ Smart retry {retry_attempt + 1}/{max_retries}: waiting {delay:.1f}s for indexing...")
                await asyncio.sleep(delay)
            
            try:
                # Re-evaluate current price and position conditions
                current_price = await self.moralis.get_current_price(mint_address, fresh=True)
                if current_price <= 0:
                    self.logger.warning(f"⚠️ Smart retry {retry_attempt + 1}: Could not get current price, continuing with retry")
                    continue
                
                # Check if position conditions have changed
                exit_conditions = await self.check_exit_conditions(mint_address, current_price)
                
                if exit_conditions:
                    new_exit_reason, new_percentage = exit_conditions
                    current_gain_pct = ((current_price / position.entry_price) - 1) * 100
                    
                    self.logger.info(f"🔄 Smart retry {retry_attempt + 1}: Conditions changed!")
                    self.logger.info(f"   Original: {original_exit_reason} (unknown%)") 
                    self.logger.info(f"   Current:  {new_exit_reason} at {current_gain_pct:+.1f}% (sell {new_percentage*100:.0f}%)")
                    
                    # Use the NEW conditions, not the original ones
                    percentage = new_percentage
                    exit_reason = new_exit_reason
                else:
                    # No exit conditions met - position recovered!
                    current_gain_pct = ((current_price / position.entry_price) - 1) * 100
                    self.logger.info(f"🚀 Smart retry {retry_attempt + 1}: Position recovered to {current_gain_pct:+.1f}% - canceling sell!")
                    position.is_selling = False
                    return {"success": False, "error": "position_recovered", "message": f"Position recovered to {current_gain_pct:+.1f}%, sell canceled"}
                
                # Try to execute the sell with current conditions
                tokens_to_sell = position.amount * percentage

                # Check if we have tokens available now (simple balance check)
                # Support both fast execution and legacy paths
                actual_balance = None
                if self.fast_execution_enabled and self.fast_transaction_submitter:
                    actual_balance = await self.fast_transaction_submitter.get_token_balance(mint_address)
                elif self.transaction_signer:
                    actual_balance = await self.transaction_signer.get_token_balance(mint_address)

                if actual_balance is not None and actual_balance > 0:
                    self.logger.info(f"✅ Smart retry {retry_attempt + 1}: Balance available! {actual_balance:,.0f} tokens")

                    # Update position with actual balance if significantly different
                    if abs(actual_balance - position.amount) / position.amount > 0.05:  # 5% difference
                        self.logger.info(f"📊 Adjusting position: {position.amount:,.0f} → {actual_balance:,.0f} tokens")
                        position.amount = actual_balance
                        tokens_to_sell = actual_balance * percentage
                        
                        # Try the actual sell - use simplified transaction execution
                        tx_result = await self._execute_sell_transaction(
                            mint_address, tokens_to_sell, percentage, symbol, exit_reason, current_price
                        )
                        
                        if tx_result.get('success'):
                            self.logger.info(f"✅ Smart retry {retry_attempt + 1} SUCCESS: {exit_reason} executed!")
                            
                            # PROCESS THE SELL LIKE NORMAL SELL FLOW (lines 1377-1404 in normal sell)
                            tx_signature = tx_result.get('tx_signature')
                            
                            # Calculate P&L (simplified version of normal sell logic)
                            usd_value = tokens_to_sell * current_price
                            cost_basis = (position.cost_usd_remaining or position.sol_invested) * percentage if percentage < 1 else (position.cost_usd_remaining or position.sol_invested)
                            profit_usd = usd_value - cost_basis
                            profit_pct = (profit_usd / cost_basis) * 100 if cost_basis > 0 else 0
                            
                            self.logger.info(f"📊 Smart retry P&L: ${cost_basis:.2f} cost → ${usd_value:.2f} received = ${profit_usd:+.2f} ({profit_pct:+.1f}%)")
                            
                            # Update position (like normal sell flow)
                            if percentage >= 0.99:  # Full sell
                                if mint_address in self.active_positions:
                                    del self.active_positions[mint_address]
                                    self.logger.info(f"✅ Position fully closed: {symbol}")
                            else:  # Partial sell
                                position.amount -= tokens_to_sell
                                position.cost_usd_remaining -= cost_basis
                                position.avg_cost_per_token = position.cost_usd_remaining / position.amount if position.amount > 0 else 0
                                self.logger.info(f"📊 Partial sell: {position.amount:.0f} tokens remaining")
                            
                            # Record in P&L store (like normal sell flow)
                            self.pnl_store.add_trade(
                                action="SELL",
                                symbol=symbol,
                                mint_address=mint_address,
                                amount=tokens_to_sell,
                                price=current_price,
                                usd_value=usd_value,
                                realized_pnl=profit_usd,
                                paper_mode=False
                            )
                            
                            # Send Discord notification (like normal sell flow)
                            if self.notifier:
                                await self.notifier.send_trade_notification(
                                    side="SELL",
                                    symbol=symbol,
                                    mint_address=mint_address,
                                    quantity=tokens_to_sell,
                                    price=current_price,
                                    usd_amount=usd_value,
                                    equity=self.pnl_store.current_equity,
                                    realized_pnl=profit_usd,
                                    paper_mode=False
                                )
                            
                            # Reset selling flag and timestamp
                            position.is_selling = False
                            if hasattr(position, 'selling_started_time'):
                                delattr(position, 'selling_started_time')
                            
                            return {
                                "success": True,
                                "tx_signature": tx_signature,
                                "tokens_sold": tokens_to_sell,
                                "exit_reason": exit_reason,
                                "price": current_price,
                                "usd_value": usd_value,
                                "profit": profit_usd,
                                "profit_pct": profit_pct,
                                "realized_pnl": profit_usd,
                                "paper_mode": False
                            }
                        else:
                            retry_error = tx_result.get('error', '')
                            self.logger.warning(f"🔄 Smart retry {retry_attempt + 1} failed: {retry_error}")

                            # If it's still a balance error, continue retrying
                            if 'NotEnoughTokensToSell' not in retry_error and '0x1787' not in retry_error:
                                # Different error type, don't keep retrying
                                break
                    else:
                        self.logger.warning(f"⏳ Smart retry {retry_attempt + 1}: Still 0 tokens available, continuing...")
                        
            except Exception as e:
                self.logger.error(f"🔄 Smart retry {retry_attempt + 1} exception: {e}")
                continue
        
        # ENHANCED: After all retries failed, check for ghost position
        self.logger.error(f"❌ All {max_retries} smart retries exhausted for {symbol}")
        
        # Final ghost position detection
        ghost_detected = await self._detect_ghost_position(mint_address, symbol, position)
        if ghost_detected:
            self.logger.error(f"👻 GHOST POSITION CONFIRMED: Removing {symbol} from tracking")
            if mint_address in self.active_positions:
                del self.active_positions[mint_address]
            return {"success": False, "error": "ghost_position_removed", "ghost_position": True}
        
        # CRITICAL: Always reset selling flag and clear timestamp
        position.is_selling = False
        if hasattr(position, 'selling_started_time'):
            delattr(position, 'selling_started_time')
        return {"success": False, "error": f"All {max_retries} smart retries failed"}
    
    async def _execute_sell_transaction(self, mint_address: str, tokens_to_sell: float, 
                                      percentage: float, symbol: str, exit_reason: str, 
                                      current_price: float) -> Dict:
        """
        Execute the actual sell transaction (extracted from main sell logic)
        """
        try:
            wallet_pubkey = self.config.pumpportal.get('wallet_public_key') if hasattr(self.config, 'pumpportal') else None
            if not wallet_pubkey:
                return {"success": False, "error": "Wallet public key not configured"}
            
            # Create sell transaction with standard slippage
            slippage_bps = 150  # 1.5% slippage
            
            tx_result = await self.pumpfun.create_sell_transaction(
                wallet_pubkey=wallet_pubkey,
                mint_address=mint_address,
                token_amount=tokens_to_sell,
                slippage_bps=slippage_bps
            )
            
            if not tx_result.get("success"):
                return tx_result
            
            transaction_b64 = tx_result.get("transaction")
            if not transaction_b64:
                return {"success": False, "error": "No transaction returned"}
            
            # Sign and send transaction
            send_result = await self.transaction_signer.sign_and_send_transaction(transaction_b64)
            
            if send_result.get("success"):
                tx_signature = send_result.get("signature")
                pnl_pct = ((current_price / self.active_positions[mint_address].entry_price) - 1) * 100
                
                self.logger.info(f"✅ Smart sell executed: {percentage*100:.0f}% of {symbol} at {pnl_pct:+.1f}% P&L - TX: {tx_signature}")
                
                return {
                    "success": True,
                    "tx_signature": tx_signature,
                    "tokens_sold": tokens_to_sell,
                    "exit_reason": exit_reason,
                    "price": current_price,
                    "profit_pct": pnl_pct,
                    "symbol": symbol,
                    "paper_mode": False
                }
            else:
                return send_result
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.notifier:
            await self.notifier.close()
        if self.blockchain_analytics:
            await self.blockchain_analytics.close()
        await self.pumpfun.close()
        await self.moralis.close()
        await self.pool_calculator.close()
    
    async def _detect_ghost_position(self, mint_address: str, symbol: str, position: Position) -> bool:
        """
        Comprehensive ghost position detection after multiple sell failures
        Returns True if position is confirmed to be a ghost (we don't actually own the tokens)
        """
        try:
            self.logger.info(f"🔍 GHOST DETECTION: Analyzing {symbol} position...")
            
            # Check 1: Multiple balance checks
            balance_checks = []
            for i in range(3):
                try:
                    # Support both fast execution and legacy paths
                    if self.fast_execution_enabled and self.fast_transaction_submitter:
                        balance = await self.fast_transaction_submitter.get_token_balance(mint_address)
                    elif self.transaction_signer:
                        balance = await self.transaction_signer.get_token_balance(mint_address)
                    else:
                        balance = None
                    balance_checks.append(balance if balance is not None else 0)
                    if i < 2:  # Don't wait after last check
                        await asyncio.sleep(2)  # Wait 2s between checks
                except Exception as e:
                    self.logger.warning(f"Ghost detection balance check {i+1} failed: {e}")
                    balance_checks.append(0)
            
            avg_balance = sum(balance_checks) / len(balance_checks)
            max_balance = max(balance_checks)
            
            self.logger.info(f"🔍 Balance checks: {balance_checks}, avg={avg_balance:.0f}, max={max_balance:.0f}")
            
            # Check 2: Position age analysis
            position_age = (datetime.now() - position.entry_time).total_seconds()
            
            # Check 3: Compare expected vs actual balance
            expected_balance = position.amount
            balance_discrepancy = abs(max_balance - expected_balance) / expected_balance if expected_balance > 0 else 1.0
            
            # Ghost position criteria:
            ghost_indicators = 0
            reasons = []
            
            # Indicator 1: Consistently zero or very low balance
            if max_balance < expected_balance * 0.01:  # Less than 1% of expected
                ghost_indicators += 2
                reasons.append(f"Very low balance: {max_balance:.0f} vs expected {expected_balance:.0f}")
            elif max_balance < expected_balance * 0.1:  # Less than 10% of expected
                ghost_indicators += 1
                reasons.append(f"Low balance: {max_balance:.0f} vs expected {expected_balance:.0f}")
            
            # Indicator 2: Old position that should be indexed by now
            if position_age > 300:  # 5+ minutes old
                ghost_indicators += 1
                reasons.append(f"Old position ({position_age/60:.1f}min) still failing")
            
            # Indicator 3: High balance discrepancy
            if balance_discrepancy > 0.9:  # 90%+ difference
                ghost_indicators += 1
                reasons.append(f"High discrepancy: {balance_discrepancy:.1%}")
            
            # Ghost detection threshold
            is_ghost = ghost_indicators >= 2
            
            if is_ghost:
                self.logger.warning(f"👻 GHOST POSITION DETECTED for {symbol}: {'; '.join(reasons)}")
                
                # Send ghost position alert
                if self.notifier:
                    await self.notifier.send_text(
                        f"👻 **Ghost Position Removed**\n"
                        f"Token: {symbol}\n"
                        f"Mint: {mint_address[:8]}...\n"
                        f"Expected: {expected_balance:,.0f} tokens\n"
                        f"Actual: {max_balance:,.0f} tokens\n"
                        f"Reasons: {'; '.join(reasons)}"
                    )
            else:
                self.logger.info(f"🔍 Not a ghost position: {ghost_indicators}/2 indicators ({'; '.join(reasons)})")
            
            return is_ghost
            
        except Exception as e:
            self.logger.error(f"Error in ghost position detection: {e}")
            return False  # Conservative: don't remove position on detection errors

    async def _should_check_token_availability(self, mint_address: str, position: Position, exit_reason: str) -> bool:
        """
        Determine if we need to check token availability based on risk factors.
        Only check when there's high risk of indexing delays.
        """
        hold_time_seconds = (datetime.now() - position.entry_time).total_seconds()
        
        # 1. Very fresh position (high indexing delay risk)
        if hold_time_seconds < 30:
            self.logger.debug(f"🔍 Checking availability: Fresh position ({hold_time_seconds:.0f}s old)")
            return True
            
        # 2. Take profit on fast-moving tokens (hit TP very quickly)
        if exit_reason.startswith("take_profit") and hold_time_seconds < 60:
            self.logger.debug(f"🔍 Checking availability: Fast TP trigger ({hold_time_seconds:.0f}s)")
            return True
            
        # 3. Position was created from simulation data (not verified from blockchain)
        if hasattr(position, 'verified_from_blockchain') and not position.verified_from_blockchain:
            self.logger.debug(f"🔍 Checking availability: Unverified position")
            return True
            
        # 4. Previous sell failures on this token
        if self._has_recent_sell_failures(mint_address):
            self.logger.debug(f"🔍 Checking availability: Recent sell failures")
            return True
            
        # Low risk - skip availability check for speed
        return False

    async def _smart_availability_check(self, mint_address: str, expected_amount: int) -> Dict:
        """
        Multi-tier availability check with intelligent fallbacks.
        Returns: {'available': bool, 'confidence': str, 'balance': int}
        """
        # Tier 1: Quick primary check (500ms timeout)
        try:
            # Support both fast execution and legacy paths
            if self.fast_execution_enabled and self.fast_transaction_submitter:
                balance_coro = self.fast_transaction_submitter.get_token_balance(mint_address)
            elif self.transaction_signer:
                balance_coro = self.transaction_signer.get_token_balance(mint_address)
            else:
                # No balance checker available - skip check
                return {'available': False, 'confidence': 'none', 'balance': 0}

            balance = await asyncio.wait_for(balance_coro, timeout=0.5)
            if balance >= expected_amount:
                return {'available': True, 'confidence': 'high', 'balance': balance}
            elif balance is not None:
                return {'available': False, 'confidence': 'high', 'balance': balance}
        except asyncio.TimeoutError:
            self.logger.warning("⏱️ Primary balance check timed out")
        except Exception as e:
            self.logger.debug(f"⚠️ Primary balance check failed: {e}")
        
        # Tier 2: Fallback to a simple existence check
        try:
            # Support both fast execution and legacy paths
            if self.fast_execution_enabled and self.fast_transaction_submitter:
                balance_coro = self.fast_transaction_submitter.get_token_balance(mint_address)
            elif self.transaction_signer:
                balance_coro = self.transaction_signer.get_token_balance(mint_address)
            else:
                # No balance checker available
                return {'available': False, 'confidence': 'none', 'balance': 0}

            balance = await asyncio.wait_for(balance_coro, timeout=1.0)
            if balance is not None and balance >= expected_amount:
                return {'available': True, 'confidence': 'medium', 'balance': balance}
            elif balance is not None:
                return {'available': False, 'confidence': 'medium', 'balance': balance}
        except Exception as e:
            self.logger.warning(f"⚠️ All balance checks failed: {e}")
        
        # If all checks fail, assume NOT available (safer than assuming available)
        return {'available': False, 'confidence': 'none', 'balance': 0}

    def _has_recent_sell_failures(self, mint_address: str) -> bool:
        """Track tokens that have had recent sell failures"""
        if not hasattr(self, '_sell_failure_cache'):
            self._sell_failure_cache = {}
            
        failure_record = self._sell_failure_cache.get(mint_address)
        if not failure_record:
            return False
            
        # Consider failures in last 5 minutes as "recent"
        recent_failures = [f for f in failure_record if (datetime.now() - f).total_seconds() < 300]
        return len(recent_failures) > 0

    def _record_sell_failure(self, mint_address: str, error_type: str):
        """Record sell failure for this token"""
        if not hasattr(self, '_sell_failure_cache'):
            self._sell_failure_cache = {}
            
        if mint_address not in self._sell_failure_cache:
            self._sell_failure_cache[mint_address] = []
            
        self._sell_failure_cache[mint_address].append(datetime.now())
        
        # Only keep last 10 failures to prevent memory bloat
        if len(self._sell_failure_cache[mint_address]) > 10:
            self._sell_failure_cache[mint_address] = self._sell_failure_cache[mint_address][-10:]