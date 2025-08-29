#!/usr/bin/env python3
"""
Strategy Simulation Engine - Phase 2: Real-time Monitoring
Monitors real-time alpha buys and simulates 3 different trading strategies with actual price data
"""

import asyncio
import json
import logging
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clients.pumpportal_client import PumpPortalClient
from src.clients.moralis_client import MoralisClient
from src.clients.realtime_client import RealtimeClient
from src.utils.config_loader import load_config
from src.utils.discord_notifier import DiscordNotifier

@dataclass
class SimulatedPosition:
    """Represents a simulated trading position"""
    mint_address: str
    symbol: str
    entry_time: datetime
    entry_price: float
    tokens: float
    usd_invested: float
    strategy: str
    alpha_count: int
    alpha_delay_seconds: float
    
    # Exit tracking
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    profit_usd: Optional[float] = None
    profit_pct: Optional[float] = None
    
    # Multi-tier exit tracking
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    remaining_percentage: float = 1.0
    total_sold_pct: float = 0.0
    
    # Cumulative P&L tracking for partial exits
    cumulative_profit_usd: float = 0.0
    cumulative_profit_pct: float = 0.0
    
    # Price tracking
    peak_price: float = 0.0
    peak_gain_pct: float = 0.0
    current_price: float = 0.0
    
    # Volatility buffer tracking
    buffer_end_time: Optional[datetime] = None
    is_in_buffer: bool = False

@dataclass
class StrategyConfig:
    """Configuration for a strategy loaded from YAML"""
    strategy_name: str
    entry_config: Dict
    exit_config: Dict
    simulation_config: Dict

@dataclass
class AlphaSignal:
    """Track alpha wallet signals for each token"""
    mint_address: str
    first_alpha_time: datetime
    alpha_wallets: List[str]
    total_buy_amount: float = 0.0
    
    @property
    def alpha_count(self) -> int:
        return len(self.alpha_wallets)

class StrategySimulator:
    """Phase 2: Real-time monitoring with actual market data"""
    
    def __init__(self, config_path: str = "config.yml"):
        """Initialize simulator with real-time monitoring"""
        print("Initializing Strategy Simulator (Phase 2 - Real-time)")
        
        # Load main config
        config_data = load_config(config_path)
        
        # Create a simple config object for compatibility
        self.config = type('Config', (), config_data)()
        self.logger = logging.getLogger(__name__)
        
        # Setup directories
        self.sims_dir = Path("sims")
        self.data_dir = self.sims_dir / "data"
        self.logs_dir = self.sims_dir / "logs"
        self.configs_dir = self.sims_dir / "configs"
        
        for directory in [self.data_dir, self.logs_dir, self.configs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load strategy configurations
        self.strategies = self.load_strategy_configs()
        self.logger.info(f"Loaded {len(self.strategies)} strategy configurations")
        
        # Initialize clients using same pattern as main bot
        self.realtime_client = None
        self.moralis = None
        self.discord = None
        
        # Data structures for delayed analysis approach
        self.alpha_signals = {}  # mint_address -> AlphaSignal
        self.pending_analysis = []  # List of tokens waiting for price data
        self.active_positions = {name: {} for name in self.strategies.keys()}
        self.completed_positions = {name: [] for name in self.strategies.keys()}
        
        # Batch processing settings
        self.last_batch_process = datetime.utcnow()
        self.batch_interval_seconds = 120  # Process every 2 minutes
        self.analysis_delay_minutes = 5  # Wait 5 minutes before analyzing (for price data to be available)
        
        # Simulation state
        self.running = False
        self.start_time = None
        self.stats = {
            'total_alpha_signals': 0,
            'positions_created': 0,
            'positions_closed': 0,
            'tokens_analyzed': 0
        }
    
    def setup_logging(self):
        """Setup logging exactly like main bot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"strategy_simulation_{timestamp}.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler  
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Suppress debug logging from websockets library to avoid binary data encoding issues
        logging.getLogger('websockets').setLevel(logging.WARNING)
        logging.getLogger('websockets.protocol').setLevel(logging.WARNING)
        logging.getLogger('websockets.client').setLevel(logging.WARNING)
        logging.getLogger('websockets.server').setLevel(logging.WARNING)
        
        self.logger.info(f"Logging initialized - file: {log_file}")
    
    def load_strategy_configs(self) -> Dict[str, StrategyConfig]:
        """Load strategy configurations from YAML files"""
        strategies = {}
        
        for config_file in self.configs_dir.glob("*.yml"):
            try:
                with open(config_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                strategy_name = data.get('strategy_name')
                if strategy_name:
                    strategies[strategy_name] = StrategyConfig(
                        strategy_name=strategy_name,
                        entry_config=data.get('entry', {}),
                        exit_config=data.get('exits', {}),
                        simulation_config=data.get('simulation', {})
                    )
                    self.logger.info(f"Loaded strategy config: {strategy_name}")
                    
            except Exception as e:
                self.logger.error(f"ERROR: Failed to load config {config_file}: {e}")
        
        if not strategies:
            raise ValueError("No strategy configurations found in sims/configs/")
            
        return strategies
    
    async def initialize_clients(self):
        """Initialize real-time clients using exact patterns from main bot"""
        try:
            # Initialize Moralis client (same pattern as main.py)
            moralis_config = {
                'api_key': self.config.moralis_api_key,
                'base_url': 'https://deep-index.moralis.io/api/v2.2'
            }
            self.moralis = MoralisClient(moralis_config)
            self.logger.info("Moralis client initialized")
            
            # Initialize realtime client (same pattern as main.py)
            realtime_config = {
                'pumpportal': self.config.pumpportal,
                'realtime_source': 'pumpportal',
                'bitquery_tokens': getattr(self.config, 'bitquery_tokens', [])
            }
            self.realtime_client = RealtimeClient(realtime_config)
            await self.realtime_client.initialize()
            self.logger.info("Realtime client initialized")
            
            # Initialize Discord notifier if configured
            if hasattr(self.config, 'discord') and self.config.discord:
                webhook_url = self.config.discord.get('webhook_url', '') if isinstance(self.config.discord, dict) else ''
                if webhook_url:
                    self.discord = DiscordNotifier(webhook_url)
                    self.logger.info("Discord notifier initialized")
                else:
                    self.logger.info("Discord webhook URL not configured")
            
        except Exception as e:
            self.logger.error(f"ERROR: Failed to initialize clients: {e}")
            raise
    
    async def initialize_bitquery_client(self):
        """Initialize Bitquery client early to test token availability"""
        try:
            from src.clients.bitquery_client import BitqueryClient
            
            bitquery_tokens = getattr(self.config, 'bitquery_tokens', [])
            if not bitquery_tokens:
                self.logger.warning("No Bitquery tokens configured")
                return
                
            self.bitquery_client = BitqueryClient(bitquery_tokens)
            # Start at last token index instead of 0 to use fresh tokens first
            if hasattr(self.bitquery_client, 'api_tokens') and self.bitquery_client.api_tokens:
                self.bitquery_client.current_token_index = len(self.bitquery_client.api_tokens) - 1
                self.logger.info(f"Bitquery client initialized starting with token #{self.bitquery_client.current_token_index} (last token)")
            else:
                self.logger.info("Bitquery client initialized")
            
        except Exception as e:
            self.logger.error(f"ERROR: Bitquery client initialization failed: {e}")
            raise
    
    async def monitor_alpha_trades(self, duration_hours: float = 24.0):
        """Monitor alpha wallet trades using same pattern as main bot"""
        self.logger.info(f"Starting alpha trade monitoring for {duration_hours} hours")
        
        watched_wallets = getattr(self.config, 'watched_wallets', [])
        if not watched_wallets:
            raise ValueError("No watched_wallets configured in config.yml")
            
        self.logger.info(f"Monitoring {len(watched_wallets)} alpha wallets")
        
        # Set monitoring duration
        end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        event_count = 0
        
        try:
            # Use exact same pattern as main.py for PumpPortal subscription
            async for event in self.realtime_client.pumpportal_client.subscribe_all_events(watched_wallets):
                event_count += 1
                
                if event_count <= 5:  # Log first few events
                    self.logger.info(f"Received event #{event_count}: {event.get('event_type', 'unknown')}")
                
                if not self.running or datetime.utcnow() > end_time:
                    break
                
                # Process events exactly like main bot
                try:
                    if event.get('event_type') == 'trade':
                        await self.process_alpha_trade_event(event)
                    elif event.get('event_type') == 'token_launch':
                        # We can ignore token launches for now, focus on trades
                        pass
                        
                except Exception as e:
                    self.logger.error(f"Error processing event: {e}")
                    
        except Exception as e:
            self.logger.error(f"ERROR: Alpha monitoring error: {e}")
            raise
    
    async def process_alpha_trade_event(self, event: Dict):
        """Process alpha trade event and update signals"""
        try:
            mint = event.get('mint')
            trader = event.get('buyer') or event.get('seller')
            timestamp_str = event.get('timestamp', datetime.utcnow().isoformat())
            
            if not mint or not trader:
                return
                
            # Parse timestamp (same as main bot) - ensure timezone-naive UTC for consistency
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).replace(tzinfo=None)
            else:
                timestamp = datetime.utcnow()
            
            # Only track buy events for alpha signals
            if event.get('buyer'):  # This is a buy event
                buy_amount = float(event.get('buy_amount', 0))
                
                # Update alpha signals
                if mint not in self.alpha_signals:
                    self.alpha_signals[mint] = AlphaSignal(
                        mint_address=mint,
                        first_alpha_time=timestamp,
                        alpha_wallets=[trader],
                        total_buy_amount=buy_amount
                    )
                    self.stats['total_alpha_signals'] += 1
                    self.logger.info(f"New alpha signal: {mint[:8]}... from {trader[:8]}...")
                else:
                    signal = self.alpha_signals[mint]
                    if trader not in signal.alpha_wallets:
                        signal.alpha_wallets.append(trader)
                        signal.total_buy_amount += buy_amount
                        self.logger.debug(f"Alpha count increased: {mint[:8]}... now has {signal.alpha_count} alphas")
                
                # Add to pending analysis queue instead of immediate position creation
                await self.queue_for_analysis(mint)
                
        except Exception as e:
            self.logger.error(f"ERROR: Error processing alpha trade: {e}")
    
    async def queue_for_analysis(self, mint: str):
        """Queue a token for delayed analysis instead of immediate position creation"""
        if mint not in self.alpha_signals:
            return
            
        signal = self.alpha_signals[mint]
        
        # Check if already queued
        if mint not in [item['mint'] for item in self.pending_analysis]:
            analysis_item = {
                'mint': mint,
                'signal_time': signal.first_alpha_time,
                'alpha_count': signal.alpha_count,
                'alpha_wallets': signal.alpha_wallets.copy(),
                'total_buy_amount': signal.total_buy_amount,
                'queued_at': datetime.utcnow()
            }
            self.pending_analysis.append(analysis_item)
            self.logger.debug(f"Queued {mint[:8]}... for delayed analysis (alpha count: {signal.alpha_count})")

    async def process_batch_analysis(self):
        """Process pending analysis items in batches"""
        if not self.pending_analysis:
            return
            
        current_time = datetime.utcnow()
        
        # Find items ready for analysis (older than analysis_delay_minutes)
        ready_for_analysis = [
            item for item in self.pending_analysis 
            if (current_time - item['queued_at']).total_seconds() > (self.analysis_delay_minutes * 60)
        ]
        
        if not ready_for_analysis:
            self.logger.info(f"No tokens ready for analysis yet (need 5+ minute delay)")
            return
            
        self.logger.info(f"Processing batch analysis for {len(ready_for_analysis)} tokens")
        
        # Batch fetch historical price data for all ready tokens
        self.logger.info(f"DEBUG: About to fetch historical data for {len(ready_for_analysis)} tokens")
        historical_data = await self.fetch_historical_price_data(ready_for_analysis)
        self.logger.info(f"DEBUG: Received historical data for {len(historical_data)} tokens")
        
        # Analyze each token with its complete price history
        for item in ready_for_analysis:
            mint = item['mint']
            token_data = historical_data.get(mint)
            
            self.logger.info(f"DEBUG: Token {mint[:8]}... - has data: {bool(token_data)}")
            if token_data and token_data.get('price_history'):
                price_count = len(token_data['price_history'])
                self.logger.info(f"DEBUG: Analyzing {mint[:8]}... with {price_count} price points")
                await self.analyze_token_with_history(item, token_data)
            else:
                self.logger.warning(f"No historical price data available for {mint[:8]}... - skipping analysis")
            
            # Remove from pending queue
            self.pending_analysis.remove(item)
    
    async def fetch_historical_price_data(self, analysis_items: list) -> dict:
        """Fetch historical price data for tokens from their signal time to now"""
        historical_data = {}
        
        for item in analysis_items:
            mint = item['mint']
            start_time = item['signal_time']  # When alpha wallet bought (in UTC)
            end_time = datetime.utcnow()  # Use UTC to match signal_time
            
            try:
                # Try to get historical price data
                price_history = await self.get_token_price_history(mint, start_time, end_time)
                if price_history:
                    historical_data[mint] = {
                        'price_history': price_history,
                        'entry_price': price_history[0]['price'] if price_history else None,
                        'current_price': price_history[-1]['price'] if price_history else None
                    }
                    
                # Rate limiting
                await asyncio.sleep(0.2)  # Slower for historical data calls
                
            except Exception as e:
                self.logger.debug(f"Error fetching historical data for {mint[:8]}...: {e}")
                # Fallback: try to get just current price
                try:
                    current_price = await self.moralis.get_current_price(mint, fresh=True)
                    if current_price:
                        # Create minimal historical data with just current price
                        historical_data[mint] = {
                            'price_history': [{'timestamp': end_time, 'price': current_price}],
                            'entry_price': current_price,  # Not ideal but better than nothing
                            'current_price': current_price
                        }
                except Exception as fallback_error:
                    self.logger.debug(f"Fallback price fetch failed for {mint[:8]}...: {fallback_error}")
                    
        return historical_data
    
    async def get_token_price_history(self, mint: str, start_time: datetime, end_time: datetime) -> list:
        """Get historical price data for a token between two timestamps"""
        try:
            # Try Bitquery first (most likely to have pump.fun data)
            if hasattr(self, 'bitquery_client') or hasattr(self.config, 'bitquery_tokens'):
                return await self.get_bitquery_price_history(mint, start_time, end_time)
                
            # Fallback to QuickNode (if available)
            elif hasattr(self.config, 'quicknode_endpoint') and self.config.quicknode_endpoint:
                return await self.get_quicknode_price_history(mint, start_time, end_time)
            
            else:
                self.logger.debug(f"No historical price API configured for {mint[:8]}...")
                return []
            
        except Exception as e:
            self.logger.debug(f"Error getting price history for {mint[:8]}...: {e}")
            return []
    
    async def get_bitquery_price_history(self, mint: str, start_time: datetime, end_time: datetime) -> list:
        """Get price history from Bitquery - optimized for API credit usage"""
        try:
            # Use the bitquery client initialized at startup (like alpha discovery does)
            if not hasattr(self, 'bitquery_client'):
                self.logger.error("Bitquery client not initialized - should have been done at startup")
                return []
            
            # Convert datetime to ISO format for Bitquery (match alpha discovery format)
            # Ensure we're working with UTC times
            start_iso = start_time.isoformat(timespec="seconds") + "Z"
            end_iso = end_time.isoformat(timespec="seconds") + "Z"
            
            # Debug log to verify timestamp order
            self.logger.debug(f"Fetching price history for {mint[:8]}... from {start_iso} to {end_iso}")
            
            # Optimize for API efficiency - most memecoin action happens in first 5 minutes
            # Limit time window to 5 minutes max to save API credits
            max_analysis_duration = 5 * 60  # 5 minutes in seconds
            actual_duration = min((end_time - start_time).total_seconds(), max_analysis_duration)
            
            # Adjust end time if needed
            if actual_duration < (end_time - start_time).total_seconds():
                end_time = start_time + timedelta(seconds=actual_duration)
                end_iso = end_time.isoformat() + "Z"
                self.logger.debug(f"Limited analysis window to {actual_duration/60:.1f} minutes for API efficiency")
            
            # Conservative but sufficient limits - most tokens don't need 2000 trades
            # Even very active tokens rarely exceed 500 trades in 5 minutes
            limit = 800  # Sufficient for granular data without excessive API usage
            
            price_history = await self.bitquery_client.get_token_price_history(
                mint_address=mint,
                start_iso=start_iso,
                end_iso=end_iso,
                limit=limit
            )
            
            # If no data, retry multiple times to force token rotation
            retry_count = 0
            max_retries = 5  # Try 5 more tokens
            while not price_history and retry_count < max_retries and hasattr(self.bitquery_client, 'token_stats'):
                retry_count += 1
                self.logger.debug(f"Retry #{retry_count} for {mint[:8]}... - forcing token rotation")
                price_history = await self.bitquery_client.get_token_price_history(
                    mint_address=mint,
                    start_iso=start_iso,
                    end_iso=end_iso,
                    limit=limit
                )
            
            if price_history:
                # Convert Bitquery format to our standard format
                converted_history = []
                for point in price_history:
                    try:
                        # Parse Bitquery timestamp format
                        from datetime import datetime
                        if isinstance(point['timestamp'], str):
                            timestamp = datetime.fromisoformat(point['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None)
                        else:
                            timestamp = point['timestamp']
                            
                        converted_history.append({
                            'timestamp': timestamp,
                            'price': point['price'],
                            'volume': point.get('volume_usd', 0)
                        })
                    except Exception as parse_error:
                        self.logger.debug(f"Error parsing price point: {parse_error}")
                        continue
                
                # Log the actual granularity we achieved
                if len(converted_history) >= 2:
                    time_span = (converted_history[-1]['timestamp'] - converted_history[0]['timestamp']).total_seconds()
                    avg_interval = time_span / len(converted_history) if len(converted_history) > 1 else 0
                    self.logger.info(f"Bitquery: {len(converted_history)} trades over {time_span:.0f}s (avg {avg_interval:.1f}s/trade) for {mint[:8]}...")
                else:
                    self.logger.info(f"Bitquery returned {len(converted_history)} price points for {mint[:8]}...")
                
                return converted_history
            
            return []
            
        except Exception as e:
            self.logger.debug(f"Bitquery price history failed for {mint[:8]}...: {e}")
            return []
    
    async def get_quicknode_price_history(self, mint: str, start_time: datetime, end_time: datetime) -> list:
        """Get price history from QuickNode"""
        try:
            import aiohttp
            
            payload = {
                "method": "qn_getTokenPriceHistory",
                "params": {
                    "token": mint,
                    "startTime": int(start_time.timestamp()),
                    "endTime": int(end_time.timestamp()),
                    "interval": "1m"
                },
                "id": 1,
                "jsonrpc": "2.0"
            }
            
            headers = {"Content-Type": "application/json"}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.quicknode_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'result' in data and data['result']:
                            # Convert to standard format
                            return [
                                {
                                    'timestamp': datetime.fromtimestamp(item['timestamp']),
                                    'price': float(item['price']),
                                    'volume': float(item.get('volume', 0))
                                }
                                for item in data['result']
                            ]
            
            return []
            
        except Exception as e:
            self.logger.debug(f"QuickNode price history failed: {e}")
            return []
    
    
    async def analyze_token_with_history(self, analysis_item: dict, token_data: dict):
        """Analyze token using complete historical price data"""
        mint = analysis_item['mint']
        signal_time = analysis_item['signal_time']
        alpha_count = analysis_item['alpha_count']
        price_history = token_data['price_history']
        
        # Track token analysis
        self.stats['tokens_analyzed'] += 1
        entry_price = token_data['entry_price']
        
        if not entry_price or entry_price <= 0:
            self.logger.debug(f"Invalid entry price for {mint[:8]}...")
            return
            
        self.logger.info(f"Analyzing {mint[:8]}... with {len(price_history)} price points")
        self.logger.info(f"    Entry would have been at: ${entry_price:.8f} ({signal_time.strftime('%H:%M:%S')})")
        
        # Check which strategies would have entered and simulate their complete journey
        for strategy_name, strategy_config in self.strategies.items():
            entry_config = strategy_config.entry_config
            min_alphas = entry_config.get('min_alpha_wallets', 1)
            max_delay = entry_config.get('max_entry_delay_seconds', 15)
            
            # Calculate delay from first alpha (this should always be ≤ max_delay since we're processing retroactively)
            # But we'll check anyway to be consistent with strategy rules
            entry_delay = 0  # We're assuming we would have entered immediately after alpha signal
            
            # Check if this strategy would have entered
            if alpha_count >= min_alphas and entry_delay <= max_delay:
                await self.simulate_complete_strategy_journey(
                    mint=mint,
                    strategy_name=strategy_name, 
                    analysis_item=analysis_item,
                    entry_price=entry_price,
                    price_history=price_history
                )

    async def simulate_complete_strategy_journey(self, mint: str, strategy_name: str, analysis_item: dict, entry_price: float, price_history: list):
        """Simulate the complete trading journey from entry to exit using historical prices"""
        try:
            strategy_config = self.strategies[strategy_name]
            usd_amount = strategy_config.entry_config.get('position_size_usd', 30.0)
            
            # Get token metadata
            token_metadata = await self.moralis.get_token_metadata(mint)
            symbol = token_metadata.get('symbol', 'UNKNOWN') if token_metadata else 'UNKNOWN'
            
            # Calculate position size
            estimated_tokens = usd_amount / entry_price
            
            # Create initial position
            position = SimulatedPosition(
                mint_address=mint,
                symbol=symbol,
                entry_time=analysis_item['signal_time'],
                entry_price=entry_price,
                tokens=estimated_tokens,
                usd_invested=usd_amount,
                strategy=strategy_name,
                alpha_count=analysis_item['alpha_count'],
                alpha_delay_seconds=0,  # Assuming immediate entry after alpha
                current_price=entry_price,
                peak_price=entry_price
            )
            
            # Send Discord notification for position entry
            if self.discord:
                await self.send_position_entry_notification(position, strategy_name)
            
            # Set up volatility buffer if strategy uses it
            exit_config = strategy_config.exit_config
            buffer_config = exit_config.get('volatility_buffer', {})
            if buffer_config.get('enabled', False):
                buffer_seconds = buffer_config.get('buffer_seconds', 60)
                position.buffer_end_time = position.entry_time + timedelta(seconds=buffer_seconds)
                position.is_in_buffer = True
            
            # Walk through price history and simulate strategy execution with GRANULAR precision
            exit_result = None
            trade_count = 0
            significant_moves = 0
            
            for i, price_point in enumerate(price_history):
                timestamp = price_point['timestamp']
                price = price_point['price']
                trade_count += 1
                
                # Update position tracking
                previous_price = position.current_price
                position.current_price = price
                position.peak_price = max(position.peak_price, price)
                position.peak_gain_pct = ((position.peak_price / position.entry_price) - 1) * 100
                
                # Track significant price movements (>5% in single trade)
                if previous_price > 0:
                    price_change_pct = abs((price - previous_price) / previous_price * 100)
                    if price_change_pct > 5:
                        significant_moves += 1
                
                # Update buffer status
                if position.is_in_buffer and position.buffer_end_time:
                    if timestamp > position.buffer_end_time:
                        position.is_in_buffer = False
                        self.logger.debug(f"Buffer ended at trade #{trade_count} for {position.symbol}")
                
                # Check exit conditions at this EXACT price point
                exit_result = await self.check_exit_conditions_historical(position, strategy_name, timestamp)
                if exit_result:
                    # Exit triggered - finalize position
                    seconds_held = (timestamp - position.entry_time).total_seconds()
                    current_gain = ((price / position.entry_price) - 1) * 100
                    self.logger.info(f"[EXIT] {strategy_name.upper()}: Exit triggered after {trade_count} trades ({seconds_held:.0f}s)")
                    self.logger.info(f"    Reason: {exit_result['reason']} | Gain: {current_gain:+.1f}% | Significant moves: {significant_moves}")
                    
                    await self.finalize_historical_position(position, exit_result, timestamp, price)
                    self.active_positions[strategy_name][mint] = position
                    break
            
            # If no exit was triggered, position is still active (shouldn't happen in backtesting)
            if not exit_result:
                # Position would still be active - use final price
                final_price = price_history[-1]['price']
                final_timestamp = price_history[-1]['timestamp']
                position.current_price = final_price
                
                # Mark as active position (will be tracked going forward)
                self.active_positions[strategy_name][mint] = position
                self.logger.info(f"[ACTIVE] {strategy_name.upper()}: POSITION {symbol} | Current: {((final_price/entry_price-1)*100):+.1f}%")
            
            self.stats['positions_created'] += 1
            
            # Save results after creating/updating positions
            await self.save_results()
            
        except Exception as e:
            self.logger.error(f"ERROR: Error simulating strategy journey: {e}")

    async def check_exit_conditions_historical(self, position: SimulatedPosition, strategy: str, timestamp: datetime) -> dict:
        """Check exit conditions for a specific point in time during historical analysis"""
        # This uses the same logic as regular check_exit_conditions but for historical simulation
        strategy_config = self.strategies[strategy]
        exit_config = strategy_config.exit_config
        current_price = position.current_price
        entry_price = position.entry_price
        
        if current_price <= 0 or entry_price <= 0:
            return None
        
        current_gain_pct = ((current_price / entry_price) - 1) * 100
        hold_time = (timestamp - position.entry_time).total_seconds()
        
        # Check stop loss (skip if in buffer period)
        stop_loss_config = exit_config.get('stop_loss', {})
        if (stop_loss_config.get('enabled', False) and 
            not position.is_in_buffer and 
            current_gain_pct <= stop_loss_config.get('percentage', -15.0)):
            return {
                'reason': 'stop_loss',
                'exit_price': current_price,
                'percentage': position.remaining_percentage if position.total_sold_pct > 0 else 1.0
            }
        
        # Check take profit levels 
        tp_config = exit_config.get('take_profit', {})
        if tp_config.get('enabled', False) and tp_config.get('multi_tier_enabled', False):
            
            # TP1
            tp1_pct = tp_config.get('tp1_percentage', 25.0)
            if current_gain_pct >= tp1_pct and not position.tp1_hit:
                tp1_sell_amount = tp_config.get('tp1_sell_amount', 0.45)
                return {
                    'reason': 'take_profit_1',
                    'exit_price': current_price,
                    'percentage': tp1_sell_amount
                }
            
            # TP2 (only if TP1 hit)
            tp2_pct = tp_config.get('tp2_percentage', 50.0)
            if current_gain_pct >= tp2_pct and position.tp1_hit and not position.tp2_hit:
                tp2_sell_amount = tp_config.get('tp2_sell_amount', 0.35)
                return {
                    'reason': 'take_profit_2',
                    'exit_price': current_price,
                    'percentage': tp2_sell_amount
                }
            
            # TP3 (only if TP2 hit)
            tp3_pct = tp_config.get('tp3_percentage', 100.0)
            if current_gain_pct >= tp3_pct and position.tp2_hit and not position.tp3_hit:
                tp3_sell_amount = tp_config.get('tp3_sell_amount', 0.15)
                return {
                    'reason': 'take_profit_3',
                    'exit_price': current_price,
                    'percentage': tp3_sell_amount
                }
            
            # Trailing stop (if enabled and after TP hits)
            trailing_config = tp_config.get('trailing_stop_enabled', False)
            if trailing_config and position.tp1_hit:
                trailing_pct = tp_config.get('trailing_stop_percentage', 15.0)
                drawdown_from_peak = ((position.peak_price - current_price) / position.peak_price) * 100
                if drawdown_from_peak >= trailing_pct:
                    return {
                        'reason': 'trailing_stop',
                        'exit_price': current_price,
                        'percentage': position.remaining_percentage
                    }
        
        # REMAINING POSITION MANAGEMENT - Same as live trading
        remaining_config = exit_config.get('remaining_position', {})
        if remaining_config.get('enabled', False) and position.total_sold_pct > 0:
            
            # Profit protection - exit if price drops below protection level
            protection_level = remaining_config.get('profit_protection_level', 10.0)
            if current_gain_pct < protection_level:
                return {
                    'reason': 'profit_protection',
                    'exit_price': current_price,
                    'percentage': position.remaining_percentage,
                    'details': f'Price dropped below {protection_level}% protection level'
                }
            
            # Remaining position trailing stop
            remaining_trailing_pct = remaining_config.get('trailing_stop_percentage', 20.0)
            if position.peak_price > 0:
                remaining_drawdown = ((position.peak_price - current_price) / position.peak_price) * 100
                if remaining_drawdown >= remaining_trailing_pct:
                    return {
                        'reason': 'remaining_trailing_stop',
                        'exit_price': current_price,
                        'percentage': position.remaining_percentage,
                        'details': f'{remaining_drawdown:.1f}% drawdown from peak'
                    }
            
            # Remaining position time-based exit
            max_remaining_hold = remaining_config.get('max_remaining_hold_seconds', 3600)
            if hold_time >= max_remaining_hold:
                return {
                    'reason': 'remaining_max_hold_time',
                    'exit_price': current_price,
                    'percentage': position.remaining_percentage,
                    'details': f'Held remaining position for {hold_time/60:.1f} minutes'
                }
        
        # Check time-based exit
        time_config = exit_config.get('time_based', {})
        max_hold_seconds = time_config.get('max_hold_seconds', 1800)
        
        if hold_time >= max_hold_seconds and time_config.get('force_exit_at_max', False):
            exit_percentage = position.remaining_percentage if position.total_sold_pct > 0 else 1.0
            return {
                'reason': 'max_hold_time',
                'exit_price': current_price,
                'percentage': exit_percentage
            }
        
        return None

    async def finalize_historical_position(self, position: SimulatedPosition, exit_result: dict, exit_time: datetime, exit_price: float):
        """Finalize a position that exited during historical simulation"""
        exit_reason = exit_result['reason']
        percentage = exit_result['percentage']
        
        # Calculate profit/loss
        tokens_sold = position.tokens * percentage
        usd_received = tokens_sold * exit_price
        cost_basis = position.usd_invested * percentage
        profit_usd = usd_received - cost_basis
        profit_pct = (profit_usd / cost_basis) * 100 if cost_basis > 0 else 0
        
        # Update position
        position.exit_time = exit_time
        position.exit_price = exit_price
        position.exit_reason = exit_reason
        position.profit_usd = profit_usd
        position.profit_pct = profit_pct
        
        # Update tracking for partial exits
        if exit_reason == 'take_profit_1':
            position.tp1_hit = True
        elif exit_reason == 'take_profit_2':
            position.tp2_hit = True
        elif exit_reason == 'take_profit_3':
            position.tp3_hit = True
            
        position.total_sold_pct += percentage
        position.remaining_percentage -= percentage
        
        # If fully closed, move to completed positions
        if percentage >= 1.0 or position.remaining_percentage <= 0.01:
            strategy = position.strategy
            self.completed_positions[strategy].append(position)
            self.stats['positions_closed'] += 1
            
            hold_time_minutes = (exit_time - position.entry_time).total_seconds() / 60
            self.logger.info(f"[CLOSED] {strategy.upper()}: HISTORICAL EXIT {position.symbol}")
            self.logger.info(f"    {exit_reason.upper()} at ${exit_price:.8f} | P&L: {profit_pct:+.1f}% (${profit_usd:+.2f}) | Hold: {hold_time_minutes:.1f}m")

    async def analyze_token_retroactively(self, analysis_item: dict, current_price: float):
        """Analyze what would have happened if we bought this token"""
        mint = analysis_item['mint']
        signal_time = analysis_item['signal_time']
        alpha_count = analysis_item['alpha_count']
        
        current_time = datetime.utcnow()
        
        # Check which strategies would have entered
        for strategy_name, strategy_config in self.strategies.items():
            entry_config = strategy_config.entry_config
            min_alphas = entry_config.get('min_alpha_wallets', 1)
            max_delay = entry_config.get('max_entry_delay_seconds', 15)
            
            # Calculate delay from first alpha to now (representing when we would have bought)
            entry_delay = (current_time - signal_time).total_seconds()
            
            # Check if this strategy would have entered
            if alpha_count >= min_alphas and entry_delay <= max_delay:
                await self.create_retroactive_position(mint, strategy_name, analysis_item, current_price)

    async def create_retroactive_position(self, mint: str, strategy: str, analysis_item: dict, entry_price: float):
        """Create a position based on what would have happened"""
        try:
            strategy_config = self.strategies[strategy]
            usd_amount = strategy_config.entry_config.get('position_size_usd', 30.0)
            
            # Calculate tokens we would have received
            estimated_tokens = usd_amount / entry_price
            
            # Get token metadata
            token_metadata = await self.moralis.get_token_metadata(mint)
            symbol = token_metadata.get('symbol', 'UNKNOWN') if token_metadata else 'UNKNOWN'
            
            # Create position with retroactive entry time
            position = SimulatedPosition(
                mint_address=mint,
                symbol=symbol,
                entry_time=analysis_item['signal_time'],  # Use original signal time as entry
                entry_price=entry_price,
                tokens=estimated_tokens,
                usd_invested=usd_amount,
                strategy=strategy,
                alpha_count=analysis_item['alpha_count'],
                alpha_delay_seconds=analysis_item.get('delay_seconds', 0),
                current_price=entry_price,
                peak_price=entry_price
            )
            
            # Set up volatility buffer if strategy uses it
            exit_config = strategy_config.exit_config
            buffer_config = exit_config.get('volatility_buffer', {})
            if buffer_config.get('enabled', False):
                buffer_seconds = buffer_config.get('buffer_seconds', 60)
                position.buffer_end_time = position.entry_time + timedelta(seconds=buffer_seconds)
                position.is_in_buffer = True
            
            self.active_positions[strategy][mint] = position
            self.stats['positions_created'] += 1
            
            self.logger.info(f"🔄 {strategy.upper()}: RETROACTIVE POSITION {symbol} ({mint[:8]}...)")
            self.logger.info(f"    Entry: ${entry_price:.8f} | Size: {estimated_tokens:.0f} tokens | Alphas: {analysis_item['alpha_count']}")
            
            # Save updated results after creating position
            await self.save_results()
            
        except Exception as e:
            self.logger.error(f"ERROR: Error creating retroactive position: {e}")
    
    
    async def update_positions(self):
        """Update all active positions with current prices and check exit conditions"""
        for strategy_name in self.strategies.keys():
            positions_to_remove = []
            
            for mint, position in self.active_positions[strategy_name].items():
                try:
                    # Get current price
                    current_price = await self.moralis.get_current_price(mint, fresh=True)
                    if current_price and current_price > 0:
                        position.current_price = current_price
                        position.peak_price = max(position.peak_price, current_price)
                        position.peak_gain_pct = ((position.peak_price / position.entry_price) - 1) * 100
                        
                        # Update buffer status
                        if position.is_in_buffer and position.buffer_end_time:
                            if datetime.utcnow() > position.buffer_end_time:
                                position.is_in_buffer = False
                                self.logger.debug(f"Buffer expired for {position.symbol} ({mint[:8]}...)")
                        
                        # Check exit conditions
                        exit_result = await self.check_exit_conditions(position, strategy_name)
                        if exit_result:
                            await self.close_position(mint, strategy_name, exit_result)
                            positions_to_remove.append(mint)
                    
                except Exception as e:
                    self.logger.error(f"ERROR: Error updating position {mint[:8]}...: {e}")
            
            # Remove closed positions
            for mint in positions_to_remove:
                del self.active_positions[strategy_name][mint]
    
    async def check_exit_conditions(self, position: SimulatedPosition, strategy: str) -> Optional[Dict]:
        """Check if position should exit based on strategy configuration"""
        strategy_config = self.strategies[strategy]
        exit_config = strategy_config.exit_config
        current_price = position.current_price
        entry_price = position.entry_price
        
        if current_price <= 0 or entry_price <= 0:
            return None
        
        current_gain_pct = ((current_price / entry_price) - 1) * 100
        current_time = datetime.utcnow()
        
        # Check stop loss (skip if in buffer period)
        stop_loss_config = exit_config.get('stop_loss', {})
        if (stop_loss_config.get('enabled', False) and 
            not position.is_in_buffer and 
            current_gain_pct <= stop_loss_config.get('percentage', -15.0)):
            return {
                'reason': 'stop_loss',
                'exit_price': current_price,
                'percentage': 1.0  # Close entire position
            }
        
        # Check take profit levels (multi-tier system)
        tp_config = exit_config.get('take_profit', {})
        if tp_config.get('enabled', False) and tp_config.get('multi_tier_enabled', False):
            
            # TP1
            tp1_pct = tp_config.get('tp1_percentage', 25.0)
            if current_gain_pct >= tp1_pct and not position.tp1_hit:
                tp1_sell_amount = tp_config.get('tp1_sell_amount', 0.45)
                return {
                    'reason': 'take_profit_1',
                    'exit_price': current_price,
                    'percentage': tp1_sell_amount
                }
            
            # TP2 (only if TP1 hit)
            tp2_pct = tp_config.get('tp2_percentage', 50.0)
            if current_gain_pct >= tp2_pct and position.tp1_hit and not position.tp2_hit:
                tp2_sell_amount = tp_config.get('tp2_sell_amount', 0.35)
                return {
                    'reason': 'take_profit_2',
                    'exit_price': current_price,
                    'percentage': tp2_sell_amount
                }
            
            # TP3 (only if TP2 hit)
            tp3_pct = tp_config.get('tp3_percentage', 100.0)
            if current_gain_pct >= tp3_pct and position.tp2_hit and not position.tp3_hit:
                tp3_sell_amount = tp_config.get('tp3_sell_amount', 0.15)
                return {
                    'reason': 'take_profit_3',
                    'exit_price': current_price,
                    'percentage': tp3_sell_amount
                }
            
            # Trailing stop (if enabled and after TP hits)
            trailing_config = tp_config.get('trailing_stop_enabled', False)
            if trailing_config and position.tp1_hit:
                trailing_pct = tp_config.get('trailing_stop_percentage', 15.0)
                drawdown_from_peak = ((position.peak_price - current_price) / position.peak_price) * 100
                if drawdown_from_peak >= trailing_pct:
                    return {
                        'reason': 'trailing_stop',
                        'exit_price': current_price,
                        'percentage': position.remaining_percentage  # Sell remaining position
                    }
        
        # REMAINING POSITION MANAGEMENT - Strategy-specific logic for partial exits
        remaining_config = exit_config.get('remaining_position', {})
        if remaining_config.get('enabled', False) and position.total_sold_pct > 0:
            
            # Profit protection - exit if price drops below protection level
            protection_level = remaining_config.get('profit_protection_level', 10.0)
            if current_gain_pct < protection_level:
                return {
                    'reason': 'profit_protection',
                    'exit_price': current_price,
                    'percentage': position.remaining_percentage,
                    'details': f'Price dropped below {protection_level}% protection level'
                }
            
            # Remaining position trailing stop (different from main trailing stop)
            remaining_trailing_pct = remaining_config.get('trailing_stop_percentage', 20.0)
            if position.peak_price > 0:
                remaining_drawdown = ((position.peak_price - current_price) / position.peak_price) * 100
                if remaining_drawdown >= remaining_trailing_pct:
                    return {
                        'reason': 'remaining_trailing_stop',
                        'exit_price': current_price,
                        'percentage': position.remaining_percentage,
                        'details': f'{remaining_drawdown:.1f}% drawdown from peak'
                    }
            
            # Remaining position time-based exit
            max_remaining_hold = remaining_config.get('max_remaining_hold_seconds', 3600)
            if hold_time >= max_remaining_hold:
                return {
                    'reason': 'remaining_max_hold_time',
                    'exit_price': current_price,
                    'percentage': position.remaining_percentage,
                    'details': f'Held remaining position for {hold_time/60:.1f} minutes'
                }
        
        # Check time-based exit (only for full positions or if forced)
        time_config = exit_config.get('time_based', {})
        max_hold_seconds = time_config.get('max_hold_seconds', 1800)
        hold_time = (current_time - position.entry_time).total_seconds()
        
        if hold_time >= max_hold_seconds and time_config.get('force_exit_at_max', False):
            # Exit remaining percentage only (may be full position if no partial exits yet)
            exit_percentage = position.remaining_percentage if position.total_sold_pct > 0 else 1.0
            return {
                'reason': 'max_hold_time',
                'exit_price': current_price,
                'percentage': exit_percentage
            }
        
        return None
    
    async def close_position(self, mint: str, strategy: str, exit_result: Dict):
        """Close or partially close a position"""
        position = self.active_positions[strategy][mint]
        exit_reason = exit_result['reason']
        exit_price = exit_result['exit_price']
        percentage = exit_result['percentage']
        
        # Calculate profit/loss for THIS exit
        tokens_sold = position.tokens * percentage
        usd_received = tokens_sold * exit_price
        cost_basis = position.usd_invested * percentage
        profit_usd = usd_received - cost_basis
        profit_pct = (profit_usd / cost_basis) * 100 if cost_basis > 0 else 0
        
        # Send Discord notification for this exit event
        if self.discord:
            await self.send_exit_notification(position, strategy, exit_reason, exit_price, percentage, profit_pct, profit_usd)
        
        # ACCUMULATE profits from all partial exits
        position.cumulative_profit_usd += profit_usd
        # Weight the percentage by the portion sold for accurate cumulative percentage
        position.cumulative_profit_pct += profit_pct * percentage
        
        # Update position tracking for partial exits
        if exit_reason == 'take_profit_1':
            position.tp1_hit = True
        elif exit_reason == 'take_profit_2':
            position.tp2_hit = True
        elif exit_reason == 'take_profit_3':
            position.tp3_hit = True
        
        position.total_sold_pct += percentage
        position.remaining_percentage -= percentage
        
        details = exit_result.get('details', '')
        details_str = f" | {details}" if details else ""
        
        self.logger.info(f"🔄 {strategy.upper()}: {exit_reason.upper()} {position.symbol} ({mint[:8]}...)")
        self.logger.info(f"    Exit: ${exit_price:.8f} | Sold: {percentage:.1%} | This Exit P&L: {profit_pct:+.1f}% (${profit_usd:+.2f}){details_str}")
        self.logger.info(f"    Cumulative P&L: {position.cumulative_profit_pct:+.1f}% (${position.cumulative_profit_usd:+.2f}) | Remaining: {position.remaining_percentage:.1%}")
        
        # If fully closed, move to completed positions
        if percentage >= 1.0 or position.remaining_percentage <= 0.01:
            position.exit_time = datetime.utcnow()
            position.exit_price = exit_price
            position.exit_reason = exit_reason
            # Use CUMULATIVE profits, not just the final exit
            position.profit_usd = position.cumulative_profit_usd
            position.profit_pct = position.cumulative_profit_pct
            
            self.completed_positions[strategy].append(position)
            self.stats['positions_closed'] += 1
            
            self.logger.info(f"[CLOSED] {strategy.upper()}: POSITION FULLY CLOSED {position.symbol} | Final P&L: {position.cumulative_profit_pct:+.1f}%")
            
            # Send Discord summary for fully closed position
            if self.discord:
                await self.send_position_closed_summary(position, strategy)
        
        # Save updated results
        await self.save_results()
    
    async def save_results(self):
        """Save current results to JSON files"""
        try:
            for strategy_name in self.strategies.keys():
                # Calculate current strategy results
                active_positions = list(self.active_positions[strategy_name].values())
                completed_positions = self.completed_positions[strategy_name]
                
                # Calculate metrics
                total_trades = len(completed_positions)
                winning_trades = len([p for p in completed_positions if p.profit_pct and p.profit_pct > 0])
                total_profit = sum([p.profit_usd for p in completed_positions if p.profit_usd])
                
                results = {
                    'last_updated': datetime.utcnow().isoformat(),
                    'strategy_name': strategy_name,
                    'results': {
                        'total_trades': total_trades,
                        'active_trades': len(active_positions),
                        'winning_trades': winning_trades,
                        'losing_trades': total_trades - winning_trades,
                        'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                        'total_profit_usd': total_profit,
                        'avg_profit_per_trade': total_profit / total_trades if total_trades > 0 else 0,
                    },
                    'active_positions': [asdict(p) for p in active_positions],
                    'completed_positions': [asdict(p) for p in completed_positions],
                    'stats': self.stats.copy()
                }
                
                # Save to file
                results_file = self.data_dir / f"strategy_sim_{strategy_name}.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"ERROR: Failed to save results: {e}")
    
    async def send_discord_summary(self):
        """Send strategy performance summary to Discord"""
        if not self.discord:
            return
        
        try:
            # Calculate overall metrics
            total_positions = 0
            total_wins = 0
            total_losses = 0
            total_pnl = 0.0
            total_active = 0
            
            strategy_details = []
            
            for strategy_name in self.strategies.keys():
                active = list(self.active_positions[strategy_name].values())
                completed = self.completed_positions[strategy_name]
                
                strategy_wins = 0
                strategy_losses = 0
                strategy_pnl = 0.0
                
                # Count wins/losses from BOTH completed positions AND partial profits from active positions
                for pos in completed:
                    # Use cumulative P&L which properly accounts for all partial exits
                    profit_pct = pos.cumulative_profit_pct if pos.cumulative_profit_pct != 0 else (
                        ((pos.exit_price / pos.entry_price) - 1) * 100 if pos.exit_price and pos.entry_price else 0
                    )
                    strategy_pnl += profit_pct
                    if profit_pct > 0:
                        strategy_wins += 1
                    else:
                        strategy_losses += 1
                
                # ALSO count partial profits from active positions as wins
                for pos in active:
                    if pos.cumulative_profit_pct > 0:  # Position has taken some profit (TP1/TP2/TP3 hits)
                        strategy_wins += 1  # Count as a win
                        strategy_pnl += pos.cumulative_profit_pct  # Add realized profit
                    # Note: We don't count active positions as losses since they might recover
                
                # Calculate FIXED unrealized P&L for active positions (remaining position only)
                unrealized_pnl = 0.0
                for pos in active:
                    if pos.current_price and pos.entry_price and pos.remaining_percentage > 0:
                        # Only the REMAINING position's unrealized P&L (don't double-count realized profits)
                        remaining_pnl = ((pos.current_price / pos.entry_price) - 1) * 100 * pos.remaining_percentage
                        unrealized_pnl += remaining_pnl  # Don't add cumulative_profit_pct here - that's realized!
                
                win_rate = (strategy_wins / (strategy_wins + strategy_losses) * 100) if (strategy_wins + strategy_losses) > 0 else 0
                
                strategy_details.append({
                    'name': strategy_name.upper(),
                    'active': len(active),
                    'completed': len(completed),
                    'wins': strategy_wins,
                    'losses': strategy_losses,
                    'win_rate': win_rate,
                    'realized_pnl': strategy_pnl,
                    'unrealized_pnl': unrealized_pnl,
                    'total_pnl': strategy_pnl + unrealized_pnl
                })
                
                total_positions += len(active) + len(completed)
                total_wins += strategy_wins
                total_losses += strategy_losses
                total_pnl += strategy_pnl + unrealized_pnl
                total_active += len(active)
            
            # Create Discord embed
            embed_color = 0x00ff00 if total_pnl >= 0 else 0xff0000  # Green if profit, red if loss
            
            description = f"**Simulation Runtime:** {((datetime.utcnow() - self.start_time).total_seconds() / 3600):.1f} hours\n"
            description += f"**Alpha Signals Tracked:** {len(self.alpha_signals)}\n"
            description += f"**Tokens Analyzed:** {self.stats['tokens_analyzed']}\n\n"
            
            fields = {}
            
            # Add overall summary
            overall_win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
            fields["Overall Performance"] = (
                f"Total Positions: {total_positions}\n"
                f"Active: {total_active} | Closed: {total_wins + total_losses}\n"
                f"Win Rate: {overall_win_rate:.1f}%\n"
                f"**Total P&L: {total_pnl:+.2f}%**"
            )
            
            # Add each strategy's performance
            for strategy in strategy_details:
                status_indicator = "[+]" if strategy['total_pnl'] >= 0 else "[-]"
                fields[f"{status_indicator} {strategy['name']}"] = (
                    f"Active: {strategy['active']} | Closed: {strategy['completed']}\n"
                    f"W/L: {strategy['wins']}/{strategy['losses']} ({strategy['win_rate']:.1f}%)\n"
                    f"Realized: {strategy['realized_pnl']:+.2f}%\n"
                    f"Unrealized: {strategy['unrealized_pnl']:+.2f}%\n"
                    f"**Total: {strategy['total_pnl']:+.2f}%**"
                )
            
            # Send the embed
            await self.discord.send_embed(
                title="[SIM] Strategy Simulation Update",
                description=description,
                fields=fields,
                color=embed_color,
                footer=f"Next update in 10 minutes | {self.stats['total_alpha_signals']} alpha signals"
            )
            
            self.logger.info("Discord summary sent successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to send Discord summary: {e}")

    async def send_final_discord_summary(self):
        """Send final summary when simulation ends"""
        if not self.discord:
            return
        
        try:
            # Similar to periodic summary but with more detail
            total_positions = 0
            total_wins = 0
            total_losses = 0
            total_pnl = 0.0
            best_trade = {'pnl': -999, 'strategy': '', 'token': ''}
            worst_trade = {'pnl': 999, 'strategy': '', 'token': ''}
            
            strategy_details = []
            
            for strategy_name in self.strategies.keys():
                completed = self.completed_positions[strategy_name]
                
                strategy_wins = 0
                strategy_losses = 0
                strategy_pnl = 0.0
                
                for pos in completed:
                    if pos.exit_price and pos.entry_price:
                        profit_pct = ((pos.exit_price / pos.entry_price) - 1) * 100
                        strategy_pnl += profit_pct
                        if profit_pct > 0:
                            strategy_wins += 1
                        else:
                            strategy_losses += 1
                        
                        # Track best and worst trades
                        if profit_pct > best_trade['pnl']:
                            best_trade = {'pnl': profit_pct, 'strategy': strategy_name, 'token': pos.mint_address[:8]}
                        if profit_pct < worst_trade['pnl']:
                            worst_trade = {'pnl': profit_pct, 'strategy': strategy_name, 'token': pos.mint_address[:8]}
                
                win_rate = (strategy_wins / (strategy_wins + strategy_losses) * 100) if (strategy_wins + strategy_losses) > 0 else 0
                avg_profit = strategy_pnl / len(completed) if completed else 0
                
                strategy_details.append({
                    'name': strategy_name.upper(),
                    'completed': len(completed),
                    'wins': strategy_wins,
                    'losses': strategy_losses,
                    'win_rate': win_rate,
                    'total_pnl': strategy_pnl,
                    'avg_profit': avg_profit
                })
                
                total_positions += len(completed)
                total_wins += strategy_wins
                total_losses += strategy_losses
                total_pnl += strategy_pnl
            
            # Create final summary embed
            embed_color = 0x00ff00 if total_pnl >= 0 else 0xff0000
            runtime_hours = (datetime.utcnow() - self.start_time).total_seconds() / 3600
            
            description = f"**Final Results after {runtime_hours:.1f} hours**\n\n"
            description += f"Total Alpha Signals: {len(self.alpha_signals)}\n"
            description += f"Tokens Analyzed: {self.stats['tokens_analyzed']}\n"
            description += f"Total Positions: {total_positions}\n"
            description += f"Overall Win Rate: {(total_wins/(total_wins+total_losses)*100) if (total_wins+total_losses) > 0 else 0:.1f}%\n"
            description += f"**Total P&L: {total_pnl:+.2f}%**\n\n"
            
            if best_trade['pnl'] > -999:
                description += f"Best Trade: {best_trade['strategy'].upper()} on {best_trade['token']}... ({best_trade['pnl']:+.1f}%)\n"
            if worst_trade['pnl'] < 999:
                description += f"Worst Trade: {worst_trade['strategy'].upper()} on {worst_trade['token']}... ({worst_trade['pnl']:+.1f}%)\n"
            
            fields = {}
            
            # Rank strategies by performance
            strategy_details.sort(key=lambda x: x['total_pnl'], reverse=True)
            
            for i, strategy in enumerate(strategy_details, 1):
                medal = "[1st]" if i == 1 else "[2nd]" if i == 2 else "[3rd]" if i == 3 else f"[{i}th]"
                fields[f"{medal} {strategy['name']}"] = (
                    f"Positions: {strategy['completed']}\n"
                    f"Win Rate: {strategy['win_rate']:.1f}%\n"
                    f"Avg P&L per Trade: {strategy['avg_profit']:+.2f}%\n"
                    f"**Total P&L: {strategy['total_pnl']:+.2f}%**"
                )
            
            # Send final summary
            await self.discord.send_embed(
                title="[COMPLETE] Simulation Final Results",
                description=description,
                fields=fields,
                color=embed_color,
                footer=f"Simulation completed | Data saved to sims/data/"
            )
            
            self.logger.info("Final Discord summary sent")
            
        except Exception as e:
            self.logger.error(f"Failed to send final Discord summary: {e}")

    async def run_simulation(self, duration_hours: float = 1.0):
        """Main simulation loop"""
        self.logger.info(f"Starting {duration_hours}-hour strategy simulation")
        self.start_time = datetime.utcnow()
        self.running = True
        
        try:
            # Initialize clients
            await self.initialize_clients()
            
            # Initialize bitquery client early to test token rotation
            await self.initialize_bitquery_client()
            
            # Start monitoring and updating tasks
            tasks = [
                self.monitor_alpha_trades(duration_hours),
                self.periodic_position_updates()
            ]
            
            # Run all tasks concurrently
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"ERROR: Simulation error: {e}")
            raise
        finally:
            self.running = False
            await self.save_results()
            await self.send_final_discord_summary()
            self.logger.info("Simulation completed")
    
    async def periodic_position_updates(self):
        """Periodically update positions and process batch analysis"""
        last_discord_update = datetime.utcnow()
        discord_update_interval = 600  # 10 minutes in seconds
        
        while self.running:
            try:
                # Update existing positions
                await self.update_positions()
                
                # Check if it's time for batch processing
                current_time = datetime.utcnow()
                if (current_time - self.last_batch_process).total_seconds() >= self.batch_interval_seconds:
                    await self.process_batch_analysis()
                    self.last_batch_process = current_time
                    
                    # Log queue status
                    if self.pending_analysis:
                        self.logger.info(f"Batch processing status: {len(self.pending_analysis)} tokens queued for analysis")
                
                # Check if it's time for Discord update
                if (current_time - last_discord_update).total_seconds() >= discord_update_interval:
                    await self.send_discord_summary()
                    last_discord_update = current_time
                
                await asyncio.sleep(5)  # Update every 5 seconds
            except Exception as e:
                self.logger.error(f"ERROR: Error in position updates: {e}")
                await asyncio.sleep(5)
    
    async def send_position_entry_notification(self, position: SimulatedPosition, strategy: str):
        """Send Discord notification when entering a position"""
        if not self.discord:
            return
        
        try:
            description = (
                f"**Token:** {position.symbol} ({position.mint_address[:8]}...)\n"
                f"**Entry Price:** ${position.entry_price:.8f}\n"
                f"**Position Size:** ${position.usd_invested:.2f}\n"
                f"**Alpha Count:** {position.alpha_count}\n"
                f"**Time:** {position.entry_time.strftime('%H:%M:%S UTC')}"
            )
            
            await self.discord.send_embed(
                title=f"[{strategy.upper()}] 🎯 POSITION OPENED",
                description=description,
                color=0x3498db,  # Blue for entry
                fields={},
                footer=f"Simulation | Entry Signal"
            )
        except Exception as e:
            self.logger.debug(f"Failed to send entry notification: {e}")
    
    async def send_exit_notification(self, position: SimulatedPosition, strategy: str, exit_reason: str, 
                                    exit_price: float, percentage: float, profit_pct: float, profit_usd: float):
        """Send Discord notification for any exit event (partial or full)"""
        if not self.discord:
            return
        
        try:
            # Determine exit type and emoji
            if 'take_profit' in exit_reason:
                emoji = "💰"
                tp_level = exit_reason.replace('take_profit_', 'TP')
                title = f"[{strategy.upper()}] {emoji} {tp_level} HIT"
                color = 0x00ff00  # Green
            elif 'stop_loss' in exit_reason:
                emoji = "🛑"
                title = f"[{strategy.upper()}] {emoji} STOP LOSS"
                color = 0xff0000  # Red
            elif 'trailing_stop' in exit_reason:
                emoji = "📉"
                title = f"[{strategy.upper()}] {emoji} TRAILING STOP"
                color = 0xffa500  # Orange
            elif 'profit_protection' in exit_reason:
                emoji = "🛡️"
                title = f"[{strategy.upper()}] {emoji} PROFIT PROTECTION"
                color = 0xffff00  # Yellow
            elif 'max_hold_time' in exit_reason:
                emoji = "⏰"
                title = f"[{strategy.upper()}] {emoji} TIME EXIT"
                color = 0x9932cc  # Purple
            else:
                emoji = "🔄"
                title = f"[{strategy.upper()}] {emoji} {exit_reason.upper()}"
                color = 0x808080  # Gray
            
            # Calculate hold time
            hold_time = (datetime.utcnow() - position.entry_time).total_seconds()
            hold_time_str = f"{hold_time/60:.1f}m" if hold_time >= 60 else f"{hold_time:.0f}s"
            
            description = (
                f"**Token:** {position.symbol} ({position.mint_address[:8]}...)\n"
                f"**Exit Price:** ${exit_price:.8f}\n"
                f"**Sold:** {percentage:.1%} of position\n"
                f"**P&L This Exit:** {profit_pct:+.1f}% (${profit_usd:+.2f})\n"
                f"**Entry → Exit:** ${position.entry_price:.8f} → ${exit_price:.8f}\n"
                f"**Hold Time:** {hold_time_str}\n"
                f"**Remaining Position:** {position.remaining_percentage:.1%}"
            )
            
            # Add cumulative stats if there were previous exits
            if position.total_sold_pct > percentage:
                description += f"\n**Total P&L So Far:** {position.cumulative_profit_pct:+.1f}% (${position.cumulative_profit_usd:+.2f})"
            
            await self.discord.send_embed(
                title=title,
                description=description,
                color=color,
                fields={},
                footer=f"Peak: {position.peak_gain_pct:+.1f}% | Alpha: {position.alpha_count}"
            )
        except Exception as e:
            self.logger.debug(f"Failed to send exit notification: {e}")
    
    async def send_position_closed_summary(self, position: SimulatedPosition, strategy: str):
        """Send comprehensive summary when position is fully closed"""
        if not self.discord:
            return
        
        try:
            # Determine overall result
            if position.cumulative_profit_pct > 0:
                emoji = "✅"
                color = 0x00ff00  # Green
                result = "PROFIT"
            else:
                emoji = "❌"
                color = 0xff0000  # Red
                result = "LOSS"
            
            # Calculate total hold time
            total_hold_time = (position.exit_time - position.entry_time).total_seconds()
            hold_str = f"{total_hold_time/60:.1f} minutes" if total_hold_time >= 60 else f"{total_hold_time:.0f} seconds"
            
            # Build exit history
            exit_history = ""
            if position.tp1_hit:
                exit_history += "✓ TP1 Hit\n"
            if position.tp2_hit:
                exit_history += "✓ TP2 Hit\n"
            if position.tp3_hit:
                exit_history += "✓ TP3 Hit\n"
            if not exit_history:
                exit_history = "No take profits hit\n"
            
            description = (
                f"**Token:** {position.symbol} ({position.mint_address[:8]}...)\n\n"
                f"**Entry:** ${position.entry_price:.8f} @ {position.entry_time.strftime('%H:%M:%S')}\n"
                f"**Final Exit:** ${position.exit_price:.8f} @ {position.exit_time.strftime('%H:%M:%S')}\n"
                f"**Total Hold Time:** {hold_str}\n\n"
                f"**Exit History:**\n{exit_history}\n"
                f"**Final Exit Reason:** {position.exit_reason.replace('_', ' ').title()}\n\n"
                f"**Peak Gain:** {position.peak_gain_pct:+.1f}%\n"
                f"**Final P&L:** {position.cumulative_profit_pct:+.1f}% (${position.cumulative_profit_usd:+.2f})\n"
                f"**Alpha Count:** {position.alpha_count}"
            )
            
            await self.discord.send_embed(
                title=f"[{strategy.upper()}] {emoji} POSITION CLOSED - {result}",
                description=description,
                color=color,
                fields={},
                footer=f"Simulation Complete for {position.symbol}"
            )
        except Exception as e:
            self.logger.debug(f"Failed to send closed position summary: {e}")

async def main():
    """Main entry point"""
    print("Strategy Simulation Engine - Phase 2")
    print("=" * 50)
    
    try:
        # Get simulation duration
        duration_input = input("Enter simulation duration in hours (default: 1): ").strip()
        duration_hours = float(duration_input) if duration_input else 1.0
        
        # Initialize and run simulator
        simulator = StrategySimulator()
        await simulator.run_simulation(duration_hours)
        
        print(f"\n[SUCCESS] Simulation completed! Check sims/data/ for results.")
        
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Simulation failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())