"""
Fast execution module for entry and exit trades
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clients.pumpfun_client import PumpFunClient
from src.clients.transaction_signer import TransactionSigner
from src.core.pool_calculator import PoolCalculator


class Executor:
    """High-speed trade execution optimized for sniping"""
    
    def __init__(self, config: Dict, store, exits, metrics):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.store = store
        self.exits = exits
        self.metrics = metrics
        
        # Execution settings
        self.enable_live = config['trade']['enable_live']
        self.slippage_bps = config['trade']['slippage_bps']
        self.max_impact_bps = config['trade']['max_buy_impact_bps']
        
        # Clients (initialized in setup)
        self.pumpfun = None
        self.signer = None
        self.pool_calc = PoolCalculator()
        
        # Track last error for reporting
        self.last_reason = None
        
        # RPC health tracking (using QuickNode, no rotation needed)
        self.quicknode_endpoint = None
        
        # Performance metrics
        self.latency_tracker = []
    
    async def initialize(self, pumpfun_config: Dict, wallet_private_key: str):
        """Initialize execution clients"""
        try:
            # Initialize PumpFun client
            self.pumpfun = PumpFunClient(
                quicknode_endpoint=pumpfun_config.get('quicknode_endpoint', ''),
                api_key=pumpfun_config.get('api_key', '')
            )
            
            # Initialize transaction signer with QuickNode endpoint (not free RPC)
            self.quicknode_endpoint = pumpfun_config.get('quicknode_endpoint', '')
            quicknode_api_key = pumpfun_config.get('api_key', '')
            
            if not self.quicknode_endpoint:
                raise ValueError("QuickNode endpoint is required for transaction signing")
            
            self.signer = TransactionSigner(
                quicknode_endpoint=self.quicknode_endpoint,
                quicknode_api_key=quicknode_api_key,
                private_key_base58=wallet_private_key
            )
            
            self.wallet_pubkey = str(self.signer.keypair.pubkey())
            self.logger.info(f"Executor initialized with wallet: {self.wallet_pubkey[:8]}...")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize executor: {e}")
            raise
    
    
    async def try_enter(self, evt: Dict, position_size_sol: float, priority_fee: float) -> bool:
        """
        Try to enter position with optimized speed
        Returns True if successful
        """
        start_time = time.time()
        
        try:
            # Extract data
            token_mint = evt['token_mint']
            ticker = evt.get('ticker', 'UNKNOWN')
            
            self.logger.info(f"Attempting entry: {ticker} ({token_mint[:8]}...) "
                           f"Size: {position_size_sol} SOL, Priority: {priority_fee} SOL")
            
            # Estimate impact (quick check)
            if not await self._check_impact(evt, position_size_sol):
                return False
            
            # Paper trading mode
            if not self.enable_live:
                self.logger.info(f"PAPER TRADE: Would buy {ticker} for {position_size_sol} SOL")
                
                # Create paper position
                position_id = self.store.add_position({
                    'token': token_mint,
                    'ticker': ticker,
                    'dev_wallet': evt['dev_wallet'],
                    'entry_sol': position_size_sol,
                    'entry_price': 0.000001,  # Placeholder
                    'score': evt.get('total_score', 0),
                    'paper_mode': True
                })
                
                # Register with exit manager
                self.exits.register_position(evt, position_id, position_size_sol)
                
                self.metrics.inc("execution.paper_entry")
                return True
            
            # LIVE TRADING
            # Create buy transaction
            tx_result = await self.pumpfun.create_buy_transaction(
                wallet_pubkey=self.wallet_pubkey,
                mint_address=token_mint,
                sol_amount=position_size_sol,
                slippage_bps=self.slippage_bps
            )
            
            if not tx_result.get('success'):
                self.last_reason = f"TX_CREATE_FAILED: {tx_result.get('error')}"
                self.logger.error(self.last_reason)
                self.metrics.inc("execution.tx_create_failed")
                return False
            
            transaction_b64 = tx_result.get('transaction')
            if not transaction_b64:
                self.last_reason = "NO_TRANSACTION"
                self.metrics.inc("execution.no_transaction")
                return False
            
            # Quick simulation for token estimate
            sim_result = await self.signer.simulate_transaction(transaction_b64)
            estimated_tokens = sim_result.get('estimated_tokens', 0)
            
            if estimated_tokens > 0:
                self.logger.info(f"Simulation: Will receive {estimated_tokens:,.0f} tokens")
            
            # Sign and send with priority fee
            send_result = await self.signer.sign_and_send_transaction(
                transaction_b64,
                priority_fee_lamports=int(priority_fee * 1e9)  # Convert SOL to lamports
            )
            
            if not send_result.get('success'):
                self.last_reason = f"SEND_FAILED: {send_result.get('error')}"
                self.logger.error(self.last_reason)
                
                # Log RPC error (using QuickNode, no rotation needed)
                
                self.metrics.inc("execution.send_failed")
                return False
            
            # SUCCESS!
            tx_signature = send_result.get('signature')
            entry_price = position_size_sol / estimated_tokens if estimated_tokens > 0 else 0.000001
            
            self.logger.info(f"✅ LIVE BUY EXECUTED: {ticker} TX: {tx_signature[:16]}...")
            
            # Create position record
            position_id = self.store.add_position({
                'token': token_mint,
                'ticker': ticker,
                'dev_wallet': evt['dev_wallet'],
                'entry_sol': position_size_sol,
                'entry_price': entry_price,
                'entry_tokens': estimated_tokens,
                'tx_signature': tx_signature,
                'score': evt.get('total_score', 0),
                'paper_mode': False
            })
            
            # Register with exit manager
            self.exits.register_position(evt, position_id, position_size_sol)
            
            # Track performance
            latency_ms = (time.time() - start_time) * 1000
            self.latency_tracker.append(latency_ms)
            self.metrics.observe("execution.entry_latency_ms", latency_ms)
            self.metrics.inc("execution.live_entry")
            
            # Log fill
            self.store.add_fill(position_id, 'buy', estimated_tokens, entry_price, priority_fee)
            
            return True
            
        except Exception as e:
            self.last_reason = f"EXCEPTION: {str(e)}"
            self.logger.error(f"Entry execution error: {e}")
            self.metrics.inc("execution.exception")
            return False
    
    async def _check_impact(self, evt: Dict, sol_amount: float) -> bool:
        """Quick impact check before entry"""
        try:
            # Get pool data if available
            lp_sol = evt.get('lp_init', 10)  # Default 10 SOL if unknown
            
            # Rough impact estimate
            estimated_impact_bps = (sol_amount / (lp_sol * 2)) * 10000
            
            if estimated_impact_bps > self.max_impact_bps:
                self.last_reason = f"IMPACT_TOO_HIGH: {estimated_impact_bps:.0f} bps"
                self.logger.warning(self.last_reason)
                self.metrics.inc("execution.impact_rejected")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Impact check failed: {e}, proceeding anyway")
            return True
    
    async def execute_exit(self, position_id: str, sell_pct: float, reason: str) -> bool:
        """
        Execute exit trade
        Returns True if successful
        """
        try:
            position = self.store.get_position(position_id)
            if not position:
                self.logger.error(f"Position {position_id} not found")
                return False
            
            # Paper trading
            if position.get('paper_mode'):
                self.logger.info(f"PAPER EXIT: {position['ticker']} {sell_pct}% - Reason: {reason}")
                
                # Update position
                self.store.update_position(position_id, {
                    'exit_reason': reason,
                    'exit_pct': sell_pct,
                    'exit_time': datetime.now().isoformat()
                })
                
                if sell_pct >= 100:
                    self.store.close_position(position_id, reason, 0)
                
                return True
            
            # LIVE TRADING
            if not self.enable_live:
                return False
            
            tokens_to_sell = position.get('entry_tokens', 0) * (sell_pct / 100)
            
            # Create sell transaction
            tx_result = await self.pumpfun.create_sell_transaction(
                wallet_pubkey=self.wallet_pubkey,
                mint_address=position['token'],
                token_amount=tokens_to_sell,
                slippage_bps=self.slippage_bps * 2  # Higher slippage for exits
            )
            
            if not tx_result.get('success'):
                self.logger.error(f"Sell tx creation failed: {tx_result.get('error')}")
                self.metrics.inc("execution.sell_failed")
                return False
            
            # Sign and send
            transaction_b64 = tx_result.get('transaction')
            send_result = await self.signer.sign_and_send_transaction(
                transaction_b64,
                priority_fee_lamports=int(0.002 * 1e9)  # Fixed 0.002 SOL for exits
            )
            
            if not send_result.get('success'):
                self.logger.error(f"Sell send failed: {send_result.get('error')}")
                self.metrics.inc("execution.sell_send_failed")
                return False
            
            tx_signature = send_result.get('signature')
            self.logger.info(f"✅ SELL EXECUTED: {position['ticker']} {sell_pct}% TX: {tx_signature[:16]}...")
            
            # Update position
            self.store.update_position(position_id, {
                'exit_reason': reason,
                'exit_pct': sell_pct,
                'exit_tx': tx_signature,
                'exit_time': datetime.now().isoformat()
            })
            
            # Close if full exit
            if sell_pct >= 100:
                # Calculate PnL (simplified)
                exit_sol = send_result.get('sol_received', 0)
                entry_sol = position.get('entry_sol', 0)
                pnl = exit_sol - entry_sol
                
                self.store.close_position(position_id, reason, pnl)
                self.metrics.inc("execution.position_closed")
            
            # Log fill
            self.store.add_fill(position_id, 'sell', tokens_to_sell, 0, 0.002)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Exit execution error: {e}")
            self.metrics.inc("execution.exit_exception")
            return False
    
    
    async def healthcheck_loop(self):
        """Background health monitoring"""
        while True:
            try:
                await asyncio.sleep(30)
                
                # Check RPC health
                if self.signer:
                    try:
                        # Simple health check - get recent blockhash
                        result = await self.signer._make_rpc_request("getRecentBlockhash", [])
                        if 'error' in result:
                            self.logger.debug(f"QuickNode RPC health check failed: {result.get('error')}")
                    except Exception as e:
                        self.logger.debug(f"QuickNode RPC health check exception: {e}")
                
                # Calculate average latency
                if self.latency_tracker:
                    avg_latency = sum(self.latency_tracker[-10:]) / min(10, len(self.latency_tracker))
                    self.metrics.set("execution.avg_latency_ms", avg_latency)
                
            except Exception as e:
                self.logger.error(f"Healthcheck error: {e}")
                await asyncio.sleep(5)
    
    def get_stats(self) -> Dict:
        """Get execution statistics"""
        avg_latency = 0
        if self.latency_tracker:
            avg_latency = sum(self.latency_tracker[-10:]) / min(10, len(self.latency_tracker))
        
        return {
            'live_mode': self.enable_live,
            'current_rpc': self.quicknode_endpoint or 'none',
            'avg_latency_ms': avg_latency,
            'last_error': self.last_reason
        }