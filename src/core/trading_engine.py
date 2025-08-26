"""
Trading engine for executing buy/sell orders and managing positions
"""

import asyncio
import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from src.clients.pumpfun_client import PumpFunClient
from src.clients.moralis_client import MoralisClient
from src.clients.transaction_signer import TransactionSigner
from src.utils.discord_notifier import DiscordNotifier
from src.utils.pnl_store import PnLStore


@dataclass
class Position:
    mint: str
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
    
    # Blockchain verification data (for live trading accuracy)
    buy_tx_signature: Optional[str] = None  # Source of truth for cost basis
    verified_from_blockchain: bool = False   # True if data comes from tx logs


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
        if hasattr(config, 'notifications') and config.notifications:
            webhook_url = config.notifications.get('discord_webhook_url', None)
            self.logger.info(f"DISCORD_DEBUG: Found notifications section, webhook_url: {webhook_url[:50] if webhook_url else 'None'}")
        elif hasattr(config, 'discord_webhook_url'):
            webhook_url = config.discord_webhook_url
            self.logger.info(f"DISCORD_DEBUG: Found root discord_webhook_url: {webhook_url[:50] if webhook_url else 'None'}")
        else:
            self.logger.warning("DISCORD_DEBUG: No Discord webhook URL found in config")
            self.logger.info(f"DISCORD_DEBUG: Config notifications attr exists: {hasattr(config, 'notifications')}, value: {getattr(config, 'notifications', 'NONE')}")
        
        # Check if webhook URL is actually set and not empty
        if webhook_url and webhook_url.strip():
            self.notifier = DiscordNotifier(webhook_url.strip())
            self.logger.info(f"DISCORD_DEBUG: Notifier initialized successfully, enabled: {self.notifier.enabled}")
        else:
            self.notifier = None
            self.logger.warning(f"DISCORD_DEBUG: Notifier not initialized - webhook_url is empty or None: '{webhook_url}'")
        
        # Initialize P&L store
        self.pnl_store = PnLStore(
            path="data/pnl_state.json",
            initial_capital=config.initial_capital
        )
        
        # Position tracking
        self.active_positions = {}
        self.paper_capital = self.pnl_store.current_equity  # Use P&L store's equity
        self.total_trades = 0
        self.winning_trades = 0
        
        # Hybrid safety configuration
        self.safety_hybrid = getattr(config, 'safety_hybrid', {})
        self.use_hybrid_safety = self.safety_hybrid.get('enabled', False)
    
    async def _get_verified_token_balance(self, mint_address: str, position: Position) -> float:
        """
        Get token balance with smart ATA indexing awareness.
        Handles the delay between token purchase and ATA indexing.
        """
        # Try blockchain query first
        blockchain_balance = await self.transaction_signer.get_token_balance(mint_address)
        
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
                    retry_balance = await self.transaction_signer.get_token_balance(mint_address)
                    
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
    
    async def _get_post_transaction_balance(self, mint_address: str) -> float:
        """
        Get token balance after a transaction with retry logic.
        Used after sells to verify remaining balance.
        """
        # Try multiple times with delays (transaction might need time to settle)
        for attempt in range(4):
            balance = await self.transaction_signer.get_token_balance(mint_address)
            
            if balance is not None:
                return balance
            
            if attempt < 3:  # Don't sleep on last attempt
                await asyncio.sleep(1 + attempt)  # 1s, 2s, 3s delays
                self.logger.warning(f"Post-transaction balance query attempt {attempt + 1} failed, retrying...")
        
        # If all attempts failed, return 0 (assume fully sold)
        self.logger.warning("All post-transaction balance queries failed, assuming 0")
        return 0.0

    async def buy_token(self, mint_address: str, usd_amount: float, paper_mode: bool = True, symbol: str = "UNKNOWN", confidence_score: float = None) -> Dict:
        """Execute a buy order for a token"""
        try:
            self.logger.info(f"Executing BUY for {mint_address}, amount: ${usd_amount}")
            
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

    async def sell_token(self, mint_address: str, percentage: float, paper_mode: bool = True, symbol: str = "UNKNOWN", exit_reason: str = "unknown") -> Dict:
        """Execute a sell order for a token"""
        try:
            # ENHANCED: Comprehensive exit logging with multi-tier details
            position_info = ""
            if mint_address in self.active_positions:
                pos = self.active_positions[mint_address]
                hold_time = (datetime.now() - pos.entry_time).total_seconds()
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
                return await self._execute_real_sell(mint_address, percentage, symbol, exit_reason)
                
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
        buy_slip_bps = paper_config.get("buy_slippage_bps", 50)  # Reduced from 75
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
        position = Position(
            mint=mint_address,
            entry_price=fill_price,
            amount=tokens_received,
            sol_invested=sol_amount,
            entry_time=datetime.now(),
            tp_price=fill_price * self.config.tp_multiplier,
            sl_price=fill_price * self.config.stop_loss_pct,
            peak_price=fill_price,
            paper_mode=True,
            tokens_initial=tokens_received,
            cost_usd_remaining=usd_amount,
            avg_cost_per_token=usd_amount / tokens_received
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
            
            if trading_mode != 'auto':
                self.logger.warning(f"Trading mode is '{trading_mode}', not executing real trade")
                return await self._execute_paper_buy(mint_address, usd_amount, symbol)
            
            # Check if we have transaction signer
            if not self.transaction_signer:
                self.logger.error("Transaction signer not initialized for live trading")
                return {"success": False, "error": "Transaction signer not configured"}
            
            # Check if we have QuickNode configuration
            if not self.pumpfun or not self.config.quicknode_endpoint:
                self.logger.error("QuickNode not configured for live trading")
                return {"success": False, "error": "QuickNode not configured"}
            
            # Get wallet public key from pumpportal config
            wallet_pubkey = self.config.pumpportal.get('wallet_public_key') if hasattr(self.config, 'pumpportal') else None
            
            if not wallet_pubkey:
                self.logger.error("Wallet public key not configured in pumpportal section")
                return {"success": False, "error": "Wallet public key not configured"}
            
            # Convert USD to SOL with dynamic sizing for liquidity constraints
            sol_price = getattr(self.config, "paper_trading", {}).get("sol_price_estimate", 140)
            sol_amount = usd_amount / sol_price
            
            # Get current price for reference (restored from working implementation)
            current_price = await self.moralis.get_current_price(mint_address)
            if current_price <= 0:
                return {"success": False, "error": "Could not get current price"}
            
            self.logger.info(f"Got current price ${current_price:.8f} for live trade calculations")
            
            # Check wallet balance
            wallet_balance = await self.transaction_signer.get_wallet_balance()
            if wallet_balance is None or wallet_balance < sol_amount:
                self.logger.error(f"Insufficient balance: {wallet_balance:.4f} SOL < {sol_amount:.4f} SOL needed")
                return {"success": False, "error": f"Insufficient balance: {wallet_balance:.4f} SOL"}
            
            # Create buy transaction via QuickNode pump-fun API
            self.logger.info(f"Creating live buy transaction: ${usd_amount} ({sol_amount:.4f} SOL) for {symbol}")
            
            # Tightened slippage for better fills and more accurate P&L
            slippage_bps = 100  # 1% slippage - tightened from 2%
            
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
            
            # Sign and send transaction
            self.logger.info("Signing and sending transaction...")
            send_result = await self.transaction_signer.sign_and_send_transaction(transaction_b64)
            
            if send_result.get("success"):
                tx_signature = send_result.get("signature")
                self.logger.info(f"✅ Live buy executed: {symbol} for ${usd_amount} - TX: {tx_signature}")
                
                # Get exact tokens received by parsing transaction logs (most accurate)
                self.logger.info("Getting exact token amount from transaction logs...")
                
                # Wait a moment for transaction to be indexed
                await asyncio.sleep(2)
                
                # Get transaction details and parse the actual token transfer
                tx_details = await self.transaction_signer.get_transaction_details(tx_signature)
                wallet_address = self.transaction_signer.get_wallet_address()
                
                actual_tokens = 0.0
                if tx_details and wallet_address:
                    actual_tokens = self.transaction_signer.parse_token_transfer_from_logs(
                        tx_details, mint_address, wallet_address
                    )
                
                # Fallback to estimated if parsing fails
                if actual_tokens <= 0:
                    self.logger.warning("Could not parse exact tokens from logs, using price-based estimate")
                    actual_tokens = usd_amount / current_price
                
                self.logger.info(f"Exact tokens received: {actual_tokens}")
                
                # Calculate actual fill price from transaction (USD per token)
                fill_price = usd_amount / actual_tokens if actual_tokens > 0 else current_price
                self.logger.info(f"Actual fill price: ${fill_price:.8f} (vs market ${current_price:.8f})")
                
                position = Position(
                    mint=mint_address,
                    entry_price=fill_price,  # Use actual fill price, not market price
                    amount=actual_tokens,
                    sol_invested=sol_amount,
                    entry_time=datetime.now(),
                    tp_price=fill_price * self.config.tp_multiplier,
                    sl_price=fill_price * self.config.stop_loss_pct,
                    peak_price=fill_price,
                    paper_mode=False,  # This is live trading
                    tokens_initial=actual_tokens,
                    cost_usd_remaining=usd_amount,
                    avg_cost_per_token=fill_price,
                    buy_tx_signature=tx_signature,  # Store for verification
                    verified_from_blockchain=True   # Mark as blockchain-verified
                )
                
                self.active_positions[mint_address] = position
                
                # Record in P&L store
                self.pnl_store.add_trade(
                    action="BUY",
                    symbol=symbol,
                    mint_address=mint_address,
                    amount=actual_tokens,
                    price=fill_price,  # Use actual fill price
                    usd_value=usd_amount,
                    paper_mode=False
                )
                
                # Send Discord notification
                if self.notifier:
                    await self.notifier.send_trade_notification(
                        side="BUY",
                        symbol=symbol,
                        mint_address=mint_address,
                        quantity=actual_tokens,
                        price=fill_price,  # Use actual fill price
                        usd_amount=usd_amount,
                        equity=self.pnl_store.current_equity,
                        paper_mode=False
                    )
                
                return {
                    "success": True,
                    "price": fill_price,  # Use actual execution price for accurate records
                    "tx_signature": tx_signature,
                    "sol_amount": sol_amount,
                    "usd_amount": usd_amount,
                    "tokens_received": actual_tokens,
                    "symbol": symbol,
                    "paper_mode": False
                }
            else:
                self.logger.error(f"Failed to send transaction: {send_result.get('error')}")
                return send_result
            
        except Exception as e:
            self.logger.error(f"Error in real buy execution: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_paper_sell(self, mint_address: str, percentage: float, symbol: str = "UNKNOWN", exit_reason: str = "unknown") -> Dict:
        """Execute a paper trading sell"""
        if mint_address not in self.active_positions:
            return {"success": False, "error": "No position found"}
        
        position = self.active_positions[mint_address]
        
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
        sell_slip_bps = paper_config.get("sell_slippage_bps", 75)  # Reduced from 100
        
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
            symbol=symbol,
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
                symbol=symbol,
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

    async def _execute_real_sell(self, mint_address: str, percentage: float, symbol: str = "UNKNOWN", exit_reason: str = "unknown") -> Dict:
        """Execute a real sell transaction via QuickNode/PumpFun API"""
        try:
            # Check trading mode
            trading_mode = getattr(self.config, 'trading_mode', 'simulation')
            
            if trading_mode != 'auto':
                self.logger.warning(f"Trading mode is '{trading_mode}', not executing real trade")
                return await self._execute_paper_sell(mint_address, percentage, symbol, exit_reason)
            
            # Check if we have the position
            if mint_address not in self.active_positions:
                self.logger.error(f"No active position for {mint_address}")
                return {"success": False, "error": "No active position"}
            
            position = self.active_positions[mint_address]
            
            # Check if we have transaction signer
            if not self.transaction_signer:
                self.logger.error("Transaction signer not initialized for live trading")
                return {"success": False, "error": "Transaction signer not configured"}
            
            # Check if we have QuickNode configuration
            if not self.pumpfun or not self.config.quicknode_endpoint:
                self.logger.error("QuickNode not configured for live trading")
                return {"success": False, "error": "QuickNode not configured"}
            
            # Get wallet public key from pumpportal config
            wallet_pubkey = self.config.pumpportal.get('wallet_public_key') if hasattr(self.config, 'pumpportal') else None
            
            if not wallet_pubkey:
                self.logger.error("Wallet public key not configured")
                return {"success": False, "error": "Wallet public key not configured"}
            
            # Smart blockchain verification with ATA indexing awareness
            actual_balance = await self._get_verified_token_balance(mint_address, position)
            
            if actual_balance <= 0:
                self.logger.error(f"❌ SELL FAILED: Verified balance is {actual_balance}")
                
                # Provide actionable error message
                if position.amount > 0:
                    self.logger.error(f"💡 ISSUE: Position shows {position.amount} tokens but blockchain verification failed")
                    self.logger.error(f"🔧 ACTION: Check QuickNode connectivity and ATA indexing status")
                    error_msg = f"Blockchain verification failed (position: {position.amount}, verified: {actual_balance})"
                else:
                    error_msg = f"No tokens to sell: {actual_balance}"
                
                return {"success": False, "error": error_msg}
            
            # Use verified balance (blockchain truth with fallbacks)
            tokens_to_sell = actual_balance * percentage
            
            # Update position amount to match verified reality
            if abs(position.amount - actual_balance) > 0.01:
                self.logger.warning(f"Position sync: tracking {position.amount}, verified {actual_balance}")
                position.amount = actual_balance
            
            if tokens_to_sell <= 0:
                self.logger.error(f"No tokens to sell: actual balance {actual_balance}, percentage {percentage}")
                return {"success": False, "error": "No tokens to sell"}
            
            self.logger.info(f"Selling {percentage*100:.1f}% of position: {tokens_to_sell} tokens (blockchain verified: {actual_balance})")
            
            # Get current price for logging
            current_price = await self.moralis.get_current_price(mint_address, fresh=True)
            if current_price <= 0:
                current_price = position.entry_price
            
            # Create sell transaction
            self.logger.info(f"Creating live sell transaction: {percentage*100:.0f}% of {symbol} position ({tokens_to_sell:.2f} tokens)")
            
            # Tightened sell slippage for better fills
            slippage_bps = 150  # 1.5% slippage - tightened from 3%
            
            tx_result = await self.pumpfun.create_sell_transaction(
                wallet_pubkey=wallet_pubkey,
                mint_address=mint_address,
                token_amount=tokens_to_sell,
                slippage_bps=slippage_bps
            )
            
            if not tx_result.get("success"):
                self.logger.error(f"Failed to create sell transaction: {tx_result.get('error')}")
                return tx_result
            
            # Get the base64 transaction
            transaction_b64 = tx_result.get("transaction")
            if not transaction_b64:
                return {"success": False, "error": "No transaction returned"}
            
            # Sign and send transaction
            self.logger.info("Signing and sending sell transaction...")
            send_result = await self.transaction_signer.sign_and_send_transaction(transaction_b64)
            
            if send_result.get("success"):
                tx_signature = send_result.get("signature")
                pnl_pct = ((current_price / position.entry_price) - 1) * 100
                
                self.logger.info(f"✅ Live sell executed: {percentage*100:.0f}% of {symbol} at {pnl_pct:+.1f}% P&L - TX: {tx_signature}")
                
                # Get actual SOL received from transaction logs (most accurate)
                self.logger.info("Getting exact SOL received from transaction logs...")
                
                # Wait a moment for transaction to be indexed
                await asyncio.sleep(2)
                
                # Get transaction details and parse actual SOL received
                tx_details = await self.transaction_signer.get_transaction_details(tx_signature)
                wallet_address = self.transaction_signer.get_wallet_address()
                
                actual_sol_received = 0.0
                if tx_details and wallet_address:
                    actual_sol_received = self.transaction_signer.parse_sol_change_from_logs(
                        tx_details, wallet_address
                    )
                
                # Calculate actual USD value using real-time SOL price
                cost_basis_usd = tokens_to_sell * position.avg_cost_per_token
                
                if actual_sol_received > 0:
                    # Get real-time SOL price from Moralis or use estimate
                    sol_price_usd = getattr(self.config, "paper_trading", {}).get("sol_price_estimate", 250)
                    usd_value = actual_sol_received * sol_price_usd
                    self.logger.info(f"Actual SOL received: {actual_sol_received:.6f} SOL (${usd_value:.2f} at ${sol_price_usd}/SOL)")
                else:
                    # Fallback to market price estimate if parsing fails
                    self.logger.warning("Could not parse actual SOL received, using market price estimate")
                    usd_value = tokens_to_sell * current_price
                
                profit_usd = usd_value - cost_basis_usd
                profit_pct = (profit_usd / cost_basis_usd) * 100 if cost_basis_usd > 0 else 0
                
                self.logger.info(f"Sell P&L: ${cost_basis_usd:.2f} cost → ${usd_value:.2f} received = ${profit_usd:+.2f} ({profit_pct:+.1f}%)")
                
                # Update position using post-transaction balance verification
                remaining_balance = await self._get_post_transaction_balance(mint_address)
                
                # Update position to match verified reality
                position.amount = remaining_balance
                position.cost_usd_remaining -= cost_basis_usd
                
                if position.amount > 0:
                    position.avg_cost_per_token = position.cost_usd_remaining / position.amount
                else:
                    position.avg_cost_per_token = 0.0
                
                # Remove position if no tokens remain
                if remaining_balance <= 0.0001:  # Account for dust
                    del self.active_positions[mint_address]
                    self.logger.info(f"Position fully closed for {mint_address[:8]}... (blockchain verified)")
                
                # Update P&L
                self.pnl_store.add_trade(
                    action="SELL",
                    symbol=symbol,
                    mint_address=mint_address,
                    amount=tokens_to_sell,
                    price=current_price,
                    usd_value=usd_value,
                    paper_mode=False
                )
                
                # Update win rate
                self.total_trades += 1
                if profit_usd > 0:
                    self.winning_trades += 1
                
                # Send Discord notification
                if self.notifier:
                    await self.notifier.send_trade_notification(
                        side="SELL",
                        symbol=symbol,
                        mint_address=mint_address,
                        quantity=tokens_to_sell,
                        price=current_price,
                        usd_amount=usd_value,
                        equity=self.pnl_store.current_equity,
                        realized_pnl=profit_usd,  # Use realized_pnl parameter for profit
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
                self.logger.error(f"Failed to send sell transaction: {send_result.get('error')}")
                return send_result
            
        except Exception as e:
            self.logger.error(f"Error in real sell execution: {e}")
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
            # Multi-tier exit strategy for aggressive mode
            tp2_price = position.entry_price * getattr(self.config, 'tp2_multiplier', 1.50)
            tp3_price = position.entry_price * getattr(self.config, 'tp3_multiplier', 2.00)
            
            # TP1 at 25% - Sell 45%
            if current_price >= position.tp_price and position.tp1_hit_time is None:
                position.tp1_hit_time = datetime.now()
                tp1_percentage = multi_tier_config.get('tp1_sell_pct', 0.45)
                position.tp1_percentage_sold = tp1_percentage
                self.logger.info(f"🎯 TP1 HIT: Selling {tp1_percentage*100:.0f}% at +{current_gain_pct:.1f}% gain")
                return ("take_profit_partial", tp1_percentage)
            
            # TP2 at 50% - Sell 35%  
            if current_price >= tp2_price and position.tp2_hit_time is None and position.tp1_hit_time is not None:
                position.tp2_hit_time = datetime.now()
                tp2_percentage = multi_tier_config.get('tp2_sell_pct', 0.35)
                position.tp2_percentage_sold = tp2_percentage
                self.logger.info(f"🚀 TP2 HIT: Selling {tp2_percentage*100:.0f}% at +{current_gain_pct:.1f}% gain")
                return ("take_profit_partial", tp2_percentage)
            
            # TP3 at 100% - Sell 15%
            if current_price >= tp3_price and position.tp3_hit_time is None and position.tp2_hit_time is not None:
                position.tp3_hit_time = datetime.now()
                tp3_percentage = multi_tier_config.get('tp3_sell_pct', 0.15)
                position.tp3_percentage_sold = tp3_percentage
                self.logger.info(f"🌙 TP3 HIT: Selling {tp3_percentage*100:.0f}% at +{current_gain_pct:.1f}% gain (Moonshot!)")
                return ("take_profit_partial", tp3_percentage)
        else:
            # Original single TP1 logic for conservative mode
            if current_price >= position.tp_price and position.tp1_hit_time is None:
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
        
        # MOONSHOT trailing for exceptional gains (looser to capture full runs)
        if position.high_gain_peak >= 100:  # Triple digit gains
            trailing_pct = 0.75  # Trail at 75% of peak - very loose for moonshots
            trailing_stop = position.peak_price * trailing_pct
            if current_price <= trailing_stop:
                self.logger.info(f"MOONSHOT EXIT: {position.high_gain_peak:.0f}% peak, exiting at {current_gain_pct:.0f}%")
                return ("trailing_stop_moonshot", 1.0)
        elif position.high_gain_peak >= 60:  # High gains
            trailing_pct = 0.80  # Trail at 80% of peak (loosened from 82%)
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
        
        # VOLATILITY-BASED STOP LOSS
        # Instead of fixed 8%, adjust based on token volatility and time
        volatility_stop = await self._calculate_dynamic_stop_loss(mint_address, position, hold_time_seconds)
        
        if current_price <= volatility_stop:
            stop_pct = ((volatility_stop / position.entry_price) - 1) * 100
            self.logger.info(f"DYNAMIC STOP HIT: Volatility-adjusted stop at {stop_pct:.1f}%")
            return ("stop_loss", 1.0)
        
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
    
    async def _calculate_dynamic_stop_loss(self, mint_address: str, position: Position, hold_time_seconds: float) -> float:
        """
        Calculate aggressive stop loss based on friend's strategy
        
        Args:
            mint_address: Token mint address
            position: Current position
            hold_time_seconds: How long we've held the position
            
        Returns:
            Stop loss price level
        """
        try:
            # AGGRESSIVE STOP LOSS STRATEGY (Friend's Plan Refined)
            hold_time_minutes = hold_time_seconds / 60
            
            if hold_time_minutes < 5:
                # 0-5 minutes: 15% stop loss (refined from 25%)
                base_stop_pct = 0.85
                self.logger.debug(f"Aggressive early stop: 15% (hold time: {hold_time_minutes:.1f}m)")
            elif hold_time_minutes < 10:
                # 5-10 minutes: 20% stop loss (refined from 40%)  
                base_stop_pct = 0.80
                self.logger.debug(f"Aggressive mid stop: 20% (hold time: {hold_time_minutes:.1f}m)")
            else:
                # 10+ minutes: 25% stop loss
                base_stop_pct = 0.75
                self.logger.debug(f"Aggressive late stop: 25% (hold time: {hold_time_minutes:.1f}m)")
            
            # Calculate stop price
            stop_price = position.entry_price * base_stop_pct
            
            # Never let stop price go above entry (no positive stops)
            stop_price = min(stop_price, position.entry_price * 0.99)
            
            return stop_price
            
        except Exception as e:
            self.logger.error(f"Error calculating aggressive stop loss: {e}")
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
    
    async def send_summary(self):
        """Send portfolio summary to Discord"""
        if self.notifier:
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
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.notifier:
            await self.notifier.close()
        await self.pumpfun.close()
        await self.moralis.close()