#!/usr/bin/env python3
"""
Sniper Bot - Main orchestrator for early token sniping
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from sniper.storage import Store
from sniper.discovery import DevDiscovery
from sniper.filters import SafetyFilters
from sniper.execution import Executor
from sniper.exits import ExitManager
from sniper.risk import RiskManager
from sniper.metrics import Metrics
from src.utils.config_loader import load_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sniper.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class SniperBot:
    """Main orchestrator for sniper bot"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.tasks = []
        
        # Load configuration
        self.logger.info("Loading configuration...")
        self.config = self._load_sniper_config()
        
        # Initialize components
        self.logger.info("Initializing components...")
        self.store = Store(self.config['storage'])
        self.metrics = Metrics(self.config)
        self.risk = RiskManager(self.config, self.store)
        self.exits = ExitManager(self.config, self.store, self.metrics)
        self.executor = Executor(self.config, self.store, self.exits, self.metrics)
        self.discovery = DevDiscovery(self.config, self.store, self.metrics)
        self.safety = SafetyFilters(self.config, self.store, self.metrics)
        
        # Link executor to exit manager
        self.exits.set_executor(self.executor)
        
        # Stats tracking
        self.start_time = datetime.now()
        self.processed_events = 0
        self.successful_entries = 0
    
    def _load_sniper_config(self) -> dict:
        """Load and merge configurations exactly like main.py"""
        try:
            # Load sniper config
            sniper_config = load_config('config_sniper.yml')
            
            # Load main config EXACTLY like main.py does
            main_config = load_config('config.yml')
            
            # Store main config for client initialization (like main.py)
            self.main_config = main_config
            
            # Just merge discord webhook from sniper config
            return sniper_config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    async def initialize(self):
        """Initialize all async components exactly like main.py"""
        self.logger.info("Initializing async components...")
        
        try:
            # Initialize like main.py - create RealtimeClient with realtime config
            realtime_config = {
                'bitquery_tokens': self.main_config.get('bitquery_tokens'),
                'pumpportal': self.main_config.get('pumpportal', {}),
                'realtime_source': self.main_config.get('realtime_source', 'pumpportal')
            }
            
            # Import RealtimeClient like main.py
            from src.clients.realtime_client import RealtimeClient
            self.realtime_client = RealtimeClient(realtime_config)
            
            # Initialize realtime client like main.py
            await self.realtime_client.initialize()
            self.logger.info(f"Realtime client initialized with source: {self.realtime_client.get_source()}")
            
            # Initialize moralis client like main.py  
            from src.clients.moralis_client import MoralisClient
            self.moralis_client = MoralisClient(
                self.main_config.get('moralis_keys', []), 
                self.main_config.get('api_optimization', {})
            )
            
            # Use the realtime client's bitquery client (already initialized with token rotation)
            # This ensures consistent token rotation with the main bot
            self.bitquery_client = self.realtime_client.bitquery_client
            
            # Set the clients in discovery module
            self.discovery.set_clients(
                self.realtime_client,
                self.bitquery_client,
                self.moralis_client
            )
            
            # Initialize executor with real config like main.py
            await self.executor.initialize(
                {
                    'quicknode_endpoint': self.main_config.get('quicknode_endpoint', ''),
                    'api_key': self.main_config.get('quicknode_api_key', '')
                },
                self.main_config.get('pumpportal', {}).get('wallet_private_key', '')
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            raise
    
    async def run(self):
        """Main event loop"""
        self.running = True
        self.logger.info("=" * 50)
        self.logger.info("SNIPER BOT STARTED")
        self.logger.info(f"Mode: {'LIVE' if self.config['trade']['enable_live'] else 'PAPER'}")
        self.logger.info(f"Position size: {self.config['trade']['base_position_size_sol']} SOL")
        self.logger.info(f"Min score: {self.config['risk_scoring']['min_score_to_trade']}")
        self.logger.info("=" * 50)
        
        # Start background tasks
        self.tasks = [
            asyncio.create_task(self.discovery.loop()),
            asyncio.create_task(self.executor.healthcheck_loop()),
            asyncio.create_task(self.metrics.serve_http()),
            asyncio.create_task(self._status_loop())
        ]
        
        # Main event processing loop
        try:
            async for event in self.discovery.stream_candidates():
                if not self.running:
                    break
                
                self.processed_events += 1
                
                # Process candidate
                await self._process_candidate(event)
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
        finally:
            await self.shutdown()
    
    async def _process_candidate(self, event: dict):
        """Process a candidate token launch"""
        try:
            ticker = event.get('ticker', 'UNKNOWN')
            dev_wallet = event.get('dev_wallet')
            
            self.logger.info(f"\n{'='*40}")
            self.logger.info(f"Processing candidate: {ticker}")
            self.logger.info(f"Dev: {dev_wallet[:8]}...")
            
            # 1. Safety checks
            if not self.safety.passes_all(event):
                reason = self.safety.last_reason
                self.logger.warning(f"Safety check failed: {reason}")
                
                await self.metrics.send_alert("ENTRY_ABORTED", {
                    'ticker': ticker,
                    'name': event.get('name', ''),
                    'reason': reason,
                    'dev_wallet': dev_wallet,
                    'token_mint': event.get('token_mint', ''),
                    'market_cap_usd': event.get('market_cap_usd', 0),
                    'lp_sol': event.get('lp_init', 0)
                })
                
                self.store.log_event("ENTRY_ABORTED", event, reason=reason)
                return
            
            # 2. Risk scoring
            score = self.risk.score_dev(dev_wallet, event)
            
            # 3. Risk gates
            position_size = self.risk.get_position_size(score, self.config)
            
            if not self.risk.allowed(score, dev_wallet, position_size):
                reason = f"Risk gate failed (score: {score:.1f})"
                self.logger.warning(f"Risk gate failed: {reason}")
                
                await self.metrics.send_alert("ENTRY_ABORTED", {
                    'ticker': ticker,
                    'name': event.get('name', ''),
                    'reason': reason,
                    'score': score,
                    'dev_wallet': dev_wallet,
                    'token_mint': event.get('token_mint', ''),
                    'lp_sol': event.get('lp_init', 0),
                    'dev_hold_pct': event.get('dev_hold_pct', 0)
                })
                
                self.store.log_event("ENTRY_ABORTED", event, reason=reason)
                return
            
            # 4. Calculate priority fee
            congestion_level = 0  # TODO: Implement congestion detection
            priority_fee = self.risk.get_priority_fee(score, congestion_level, self.config)
            
            # 5. Execute entry
            self.logger.info(f"Attempting entry with score {score:.1f}")
            
            success = await self.executor.try_enter(event, position_size, priority_fee)
            
            if success:
                self.successful_entries += 1
                self.risk.record_attempt(dev_wallet, position_size)
                
                await self.metrics.send_alert("ENTRY_OK", {
                    'ticker': ticker,
                    'name': event.get('name', ''),
                    'token_mint': event.get('token_mint'),
                    'score': score,
                    'position_sol': position_size,
                    'entry_price': 0.000001,  # Will be updated
                    'tx_signature': 'pending',
                    'dev_wallet': dev_wallet,
                    'lp_sol': event.get('lp_init', 0),
                    'dev_hold_pct': event.get('dev_hold_pct', 0)
                })
                
                self.store.log_event("ENTRY_OK", event)
                self.logger.info(f"Entry successful for {ticker}")
            else:
                reason = self.executor.last_reason
                self.logger.warning(f"Entry failed: {reason}")
                
                await self.metrics.send_alert("ENTRY_ABORTED", {
                    'ticker': ticker,
                    'name': event.get('name', ''),
                    'reason': reason,
                    'score': score,
                    'dev_wallet': dev_wallet,
                    'token_mint': event.get('token_mint', ''),
                    'lp_sol': event.get('lp_init', 0),
                    'dev_hold_pct': event.get('dev_hold_pct', 0)
                })
                
                self.store.log_event("ENTRY_ABORTED", event, reason=reason)
            
        except Exception as e:
            self.logger.error(f"Error processing candidate: {e}")
            self.metrics.inc("processing.errors")
    
    async def _status_loop(self):
        """Periodic status reporting"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                # Gather stats
                uptime = (datetime.now() - self.start_time).seconds / 60
                metrics_summary = self.metrics.get_summary()
                risk_stats = self.risk.get_stats()
                executor_stats = self.executor.get_stats()
                exit_stats = self.exits.get_stats()
                storage_stats = self.store.get_stats()
                
                self.logger.info("\n" + "="*50)
                self.logger.info("STATUS REPORT")
                self.logger.info(f"Uptime: {uptime:.1f} minutes")
                self.logger.info(f"Events processed: {self.processed_events}")
                self.logger.info(f"Successful entries: {self.successful_entries}")
                self.logger.info(f"Active positions: {exit_stats['active_positions']}")
                self.logger.info(f"PnL: {metrics_summary['total_pnl_sol']:+.4f} SOL")
                self.logger.info(f"Win rate: {metrics_summary['win_rate']:.1f}%")
                self.logger.info(f"Hourly risk: {risk_stats['hourly_risk_sol']:.2f}/{risk_stats['max_hourly_risk_sol']} SOL")
                self.logger.info("="*50 + "\n")
                
            except Exception as e:
                self.logger.error(f"Status loop error: {e}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down sniper bot...")
        self.running = False
        
        # Cancel background tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Final stats
        metrics_summary = self.metrics.get_summary()
        self.logger.info("\n" + "="*50)
        self.logger.info("FINAL STATISTICS")
        self.logger.info(f"Total PnL: {metrics_summary['total_pnl_sol']:+.4f} SOL")
        self.logger.info(f"Win rate: {metrics_summary['win_rate']:.1f}%")
        self.logger.info(f"Total trades: {metrics_summary['winning_trades'] + metrics_summary['losing_trades']}")
        self.logger.info("="*50)
        
        self.logger.info("Sniper bot stopped")


async def main():
    """Main entry point"""
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    bot = SniperBot()
    
    try:
        await bot.initialize()
        await bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run bot
    asyncio.run(main())