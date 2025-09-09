"""
Dev discovery module with WebSocket integration and smart caching
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, AsyncGenerator, Optional
from collections import deque
import heapq
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clients.pumpportal_client import PumpPortalClient
from src.clients.bitquery_client import BitqueryClient
from src.clients.moralis_client import MoralisClient


class DevDiscovery:
    """Discovers and tracks quality devs via WebSocket events and API enrichment"""
    
    def __init__(self, config: Dict, store, metrics):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.store = store
        self.metrics = metrics
        
        # API clients (will be initialized in setup)
        self.pumpportal = None
        self.bitquery = None
        self.moralis = None
        
        # Priority queues for batch processing
        self.urgent_queue = []  # Priority queue for new devs (heapq)
        self.stale_queue = deque()  # Regular queue for stale refreshes
        self.processing_set = set()  # Track devs currently being processed
        self.enrichment_lock = asyncio.Lock()
        
        # Performance tracking
        self.processed_count = 0
        self.last_throughput_check = datetime.now()
        
        # Bootstrap wallets
        self.bootstrap_wallets = set(config['discovery'].get('watchlist_bootstrap', []))
        self.bootstrap_only = config['discovery'].get('bootstrap_only', False)
        
        # Quality thresholds
        self.min_peak_mc = config['discovery']['min_prev_peak_mc_usd']
        self.max_launches_24h = config['discovery']['max_dev_launches_24h']
        self.lookback_days = config['discovery']['lookback_days']
        
        # Cache settings
        self.cache_ttl_hours = config['discovery']['enrichment']['cache_ttl_hours']
        self.batch_interval = config['discovery']['enrichment']['batch_interval_sec']
        self.max_batch_size = config['discovery']['enrichment']['max_batch_size']
        
        # Track recent launches to prevent spam
        self.recent_launches = {}  # dev_wallet -> list of timestamps
        
    def set_clients(self, realtime_client, bitquery_client, moralis_client):
        """Set the already initialized clients from main orchestrator"""
        self.realtime_client = realtime_client
        self.bitquery = bitquery_client  
        self.moralis = moralis_client
        self.logger.info("Discovery module initialized with shared clients")
    
    async def loop(self):
        """Optimized background loop for high-throughput batch enrichment"""
        while True:
            try:
                # Process urgent devs more frequently (every 10s)
                if self.urgent_queue or self.stale_queue:
                    await self._process_enrichment_queues()
                
                # Add stale profiles every batch interval
                if (datetime.now() - self.last_throughput_check).total_seconds() >= self.batch_interval:
                    await self._queue_stale_profiles()
                    self._log_throughput()
                
                # Short sleep for responsive processing
                await asyncio.sleep(10)  # Check every 10s instead of 60s
                
            except Exception as e:
                self.logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(5)
    
    async def stream_candidates(self) -> AsyncGenerator[Dict, None]:
        """Stream token creation events and filter for quality devs"""
        if not self.realtime_client:
            self.logger.error("RealtimeClient not initialized")
            return
        
        self.logger.info("Starting candidate stream from RealtimeClient")
        
        # Use the realtime client's pumpportal stream like main.py does
        event_count = 0
        async for event in self.realtime_client.pumpportal_client.subscribe_all_events([]):
            try:
                event_count += 1
                event_type = event.get('event_type')
                
                # Debug: Log first few events to see what we're getting
                if event_count <= 10:
                    self.logger.info(f"Event #{event_count}: type={event_type}, keys={list(event.keys())[:5]}")
                    self.logger.info(f"Full event data: {event}")
                
                # Parse token launch event like main.py does
                if event_type != 'token_launch':
                    self.logger.debug(f"Skipping non-launch event: {event_type}")
                    continue
                
                # PumpPortal uses 'traderPublicKey' for token creator, not 'deployer'
                dev_wallet = event.get('deployer') or event.get('traderPublicKey')
                token_mint = event.get('mint')
                
                if not dev_wallet or not token_mint:
                    self.logger.debug(f"Skipping - missing data: dev={bool(dev_wallet)}, mint={bool(token_mint)}")
                    continue
                
                self.logger.info(f"Processing launch: {event.get('symbol', 'UNK')} by {dev_wallet[:8]}...")
                
                # Check if dev is blacklisted
                if self.store.is_blacklisted(dev_wallet):
                    self.logger.info(f"Skipping blacklisted dev: {dev_wallet[:8]}...")
                    self.metrics.inc("discovery.blacklisted_skip")
                    continue
                
                # Check recent launches (anti-spam)
                if not self._check_launch_frequency(dev_wallet):
                    self.logger.info(f"Dev {dev_wallet[:8]}... launching too frequently")
                    self.metrics.inc("discovery.spam_filtered")
                    continue
                
                # Get cached profile or queue for enrichment
                profile = self.store.get_dev_profile(dev_wallet)
                
                # If bootstrap_only mode, only allow bootstrap wallets
                if self.bootstrap_only:
                    if dev_wallet not in self.bootstrap_wallets:
                        self.logger.info(f"Bootstrap mode: skipping non-bootstrap dev {dev_wallet[:8]}...")
                        continue
                    # Bootstrap wallets always pass
                    self.logger.info(f"Bootstrap dev approved: {dev_wallet[:8]}...")
                    candidate = await self._build_candidate_event(event, dev_wallet, profile)
                    self.metrics.inc("discovery.candidate_emitted")
                    yield candidate
                    continue
                
                # Check if dev is explicitly whitelisted or in bootstrap
                if self.store.is_whitelisted(dev_wallet) or dev_wallet in self.bootstrap_wallets:
                    self.logger.info(f"Whitelisted/bootstrap dev approved: {dev_wallet[:8]}...")
                    candidate = await self._build_candidate_event(event, dev_wallet, profile)
                    self.metrics.inc("discovery.candidate_emitted")
                    yield candidate
                    continue
                
                # If no profile, queue for urgent enrichment and skip for now
                if not profile:
                    self.logger.info(f"Unknown dev {dev_wallet[:8]}... - queuing for urgent enrichment")
                    await self._queue_urgent_dev(dev_wallet)
                    self.metrics.inc("discovery.unknown_dev")
                    continue
                
                # Check if profile meets quality thresholds
                if self._is_quality_dev(profile):
                    self.logger.info(f"Quality dev approved: {dev_wallet[:8]}... (score meets thresholds)")
                    candidate = await self._build_candidate_event(event, dev_wallet, profile)
                    self.metrics.inc("discovery.candidate_emitted")
                    yield candidate
                else:
                    self.logger.info(f"Dev {dev_wallet[:8]}... doesn't meet quality thresholds")
                    self.metrics.inc("discovery.quality_filtered")
                
            except Exception as e:
                self.logger.error(f"Error processing token creation event: {e}")
                self.metrics.inc("discovery.event_error")
    
    def _check_launch_frequency(self, dev_wallet: str) -> bool:
        """Check if dev is launching too frequently"""
        now = datetime.now()
        
        # Clean old entries
        if dev_wallet in self.recent_launches:
            self.recent_launches[dev_wallet] = [
                ts for ts in self.recent_launches[dev_wallet]
                if now - ts < timedelta(hours=24)
            ]
        else:
            self.recent_launches[dev_wallet] = []
        
        # Check frequency
        launches_24h = len(self.recent_launches[dev_wallet])
        if launches_24h >= self.max_launches_24h:
            # Calculate time since last launch for better logging
            if self.recent_launches[dev_wallet]:
                last_launch = max(self.recent_launches[dev_wallet])
                minutes_ago = int((now - last_launch).total_seconds() / 60)
                self.logger.info(f"Dev {dev_wallet[:8]}... SPAM: {launches_24h+1} launches in 24h (max: {self.max_launches_24h}), last launch {minutes_ago}min ago")
            else:
                self.logger.info(f"Dev {dev_wallet[:8]}... SPAM: {launches_24h+1} launches in 24h (max: {self.max_launches_24h})")
            return False
        
        # Record this launch
        self.recent_launches[dev_wallet].append(now)
        return True
    
    def _is_quality_dev(self, profile: Dict) -> bool:
        """Check if dev profile meets quality thresholds"""
        dev_wallet = profile.get('dev_wallet', 'unknown')
        
        # Check peak market cap
        best_peak_mc = profile.get('best_peak_mc_usd', 0)
        num_tokens = profile.get('num_tokens_launched', 0)
        
        # Special case: new dev with no history
        if best_peak_mc == 0 and num_tokens == 0:
            self.logger.info(f"Dev {dev_wallet[:8]}... NEW DEV: No token history found (first launch?)")
            return False  # Could change this to True if you want to gamble on new devs
        elif best_peak_mc < self.min_peak_mc:
            self.logger.info(f"Dev {dev_wallet[:8]}... REJECTED: peak MC ${best_peak_mc:,.0f} < ${self.min_peak_mc:,.0f} required (from {num_tokens} tokens)")
            return False
        
        # For now, skip rug detection - would need more sophisticated analysis
        # TODO: Add proper rug detection based on token trading patterns
        
        # Check launch frequency
        tokens_7d = profile.get('tokens_launched_7d', 0)
        if tokens_7d > 5:  # Hardcoded max from config
            self.logger.info(f"Dev {dev_wallet[:8]}... REJECTED: {tokens_7d} tokens in 7 days (max: 5)")
            return False
        
        # Passed all checks
        self.logger.info(f"Dev {dev_wallet[:8]}... APPROVED: peak MC ${best_peak_mc:,.0f}, {tokens_7d} tokens/7d")
        return True
    
    async def _build_candidate_event(self, event: Dict, dev_wallet: str, profile: Optional[Dict]) -> Dict:
        """Build candidate event with all necessary data"""
        # Get token metadata if available
        token_metadata = {}
        if self.moralis and event.get('mint'):
            try:
                metadata = await self.moralis.get_token_metadata(event['mint'])
                if metadata:
                    token_metadata = {
                        'symbol': metadata.get('symbol', 'UNKNOWN'),
                        'name': metadata.get('name', ''),
                        'decimals': metadata.get('decimals', 9)
                    }
            except Exception as e:
                self.logger.debug(f"Could not fetch token metadata: {e}")
        
        return {
            'dev_wallet': dev_wallet,
            'token_mint': event.get('mint'),
            'ticker': token_metadata.get('symbol', event.get('symbol', 'UNKNOWN')),
            'name': token_metadata.get('name', ''),
            'lp_init': event.get('initial_liquidity_sol', 0),
            'taxes': event.get('tax_info', {}),
            'dev_hold_pct': event.get('creator_holding_pct', 0),
            'timestamp': datetime.now(),
            'profile': profile or {},
            'source': 'pumpfun',
            'raw_event': event
        }
    
    async def _queue_urgent_dev(self, dev_wallet: str):
        """Add dev to urgent processing queue (for new token launches)"""
        async with self.enrichment_lock:
            if dev_wallet not in self.processing_set:
                # Use timestamp as priority (newer = higher priority)
                priority = -datetime.now().timestamp()  # Negative for min-heap
                heapq.heappush(self.urgent_queue, (priority, dev_wallet))
                self.processing_set.add(dev_wallet)
    
    async def _queue_stale_profiles(self):
        """Add stale profiles to refresh queue"""
        stale_devs = self.store.get_stale_dev_profiles(self.cache_ttl_hours)
        async with self.enrichment_lock:
            for dev in stale_devs:
                if dev not in self.processing_set:
                    self.stale_queue.append(dev)
                    self.processing_set.add(dev)
    
    async def _process_enrichment_queues(self):
        """Process both urgent and stale queues with optimal batching"""
        batch = []
        
        async with self.enrichment_lock:
            # Prioritize urgent devs (new launches)
            while self.urgent_queue and len(batch) < self.max_batch_size * 2:  # Allow larger batches
                _, dev_wallet = heapq.heappop(self.urgent_queue)
                batch.append(('urgent', dev_wallet))
            
            # Fill remaining batch with stale refreshes
            while self.stale_queue and len(batch) < self.max_batch_size * 2:
                dev_wallet = self.stale_queue.popleft()
                batch.append(('stale', dev_wallet))
        
        if not batch:
            return
        
        self.logger.info(f"Processing enrichment batch of {len(batch)} devs ({sum(1 for t, _ in batch if t == 'urgent')} urgent, {sum(1 for t, _ in batch if t == 'stale')} stale)")
        
        # Process devs in parallel using asyncio.gather
        tasks = []
        for batch_type, dev_wallet in batch:
            task = self._enrich_dev_profile_safe(dev_wallet, batch_type)
            tasks.append(task)
        
        # Use efficient batch API call instead of individual calls
        dev_wallets = [dev_wallet for _, dev_wallet in batch]
        
        try:
            # Single API call for all devs in batch
            batch_results = await self.bitquery.get_batch_dev_profiles(
                dev_wallets, 
                lookback_days=self.lookback_days
            )
            
            success_count = 0
            for batch_type, dev_wallet in batch:
                try:
                    profile = batch_results.get(dev_wallet)
                    if profile:
                        success_count += 1
                        self.store.update_dev_profile(dev_wallet, profile)
                        
                        # Auto-whitelist if meets criteria
                        if self._is_quality_dev(profile):
                            self.store.add_to_whitelist(dev_wallet, profile.get('score', 0))
                            self.logger.info(f"Auto-whitelisted quality dev: {dev_wallet[:8]}...")
                            self.metrics.inc("discovery.auto_whitelisted")
                    else:
                        self.logger.debug(f"No profile data returned for {batch_type} dev {dev_wallet[:8]}...")
                    
                    # Remove from processing set
                    async with self.enrichment_lock:
                        self.processing_set.discard(dev_wallet)
                        
                except Exception as e:
                    self.logger.error(f"Error processing batch result for {dev_wallet[:8]}...: {e}")
            
            self.processed_count += len(batch)
            self.logger.info(f"Batch completed: {success_count}/{len(batch)} successful enrichments (1 API call vs {len(batch) * 11} individual calls)")
            
        except Exception as e:
            self.logger.error(f"Batch enrichment failed: {e}")
            self.metrics.inc("discovery.enrichment_error")
            
            # Fallback to individual processing if batch fails
            self.logger.info("Falling back to individual enrichment due to batch failure")
            tasks = []
            for batch_type, dev_wallet in batch:
                task = self._enrich_dev_profile_safe(dev_wallet, batch_type)
                tasks.append(task)
            
            # Execute fallback enrichments
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = 0
            for (batch_type, dev_wallet), result in zip(batch, results):
                try:
                    if isinstance(result, Exception):
                        self.logger.error(f"Failed to enrich {batch_type} dev {dev_wallet[:8]}...: {result}")
                        self.metrics.inc("discovery.enrichment_error")
                    elif result:
                        success_count += 1
                        self.store.update_dev_profile(dev_wallet, result)
                        
                        # Auto-whitelist if meets criteria
                        if self._is_quality_dev(result):
                            self.store.add_to_whitelist(dev_wallet, result.get('score', 0))
                            self.logger.info(f"Auto-whitelisted quality dev: {dev_wallet[:8]}...")
                            self.metrics.inc("discovery.auto_whitelisted")
                    
                    # Remove from processing set
                    async with self.enrichment_lock:
                        self.processing_set.discard(dev_wallet)
                        
                except Exception as e:
                    self.logger.error(f"Error processing fallback result for {dev_wallet[:8]}...: {e}")
            
            self.processed_count += len(batch)
            self.logger.info(f"Fallback batch completed: {success_count}/{len(batch)} successful enrichments")
    
    async def _enrich_dev_profile_safe(self, dev_wallet: str, batch_type: str) -> Optional[Dict]:
        """Safe wrapper for dev enrichment with error handling"""
        try:
            return await self._enrich_dev_profile(dev_wallet)
        except Exception as e:
            self.logger.error(f"Failed to enrich {batch_type} dev {dev_wallet[:8]}...: {e}")
            self.metrics.inc("discovery.enrichment_error")
            return None
    
    def _log_throughput(self):
        """Log processing throughput metrics"""
        now = datetime.now()
        elapsed = (now - self.last_throughput_check).total_seconds()
        if elapsed > 0 and self.processed_count > 0:
            throughput = self.processed_count / elapsed * 3600  # devs per hour
            self.logger.info(f"Enrichment throughput: {throughput:.0f} devs/hour ({self.processed_count} devs in {elapsed:.0f}s)")
            self.processed_count = 0
        self.last_throughput_check = now
    
    async def _enrich_dev_profile(self, dev_wallet: str) -> Optional[Dict]:
        """Fetch and build comprehensive dev profile"""
        profile = {
            'dev_wallet': dev_wallet,
            'updated_at': datetime.now().isoformat()
        }
        
        try:
            # Fetch data from Bitquery
            if self.bitquery:
                # Use the new method that gets ATH market caps directly from Bitquery
                history = await self.bitquery.get_dev_tokens_with_ath_mc(
                    dev_wallet,
                    lookback_days=self.lookback_days
                )
                
                if history:
                    profile.update({
                        'num_tokens_launched': history.get('total_tokens', 0),
                        'best_peak_mc_usd': history.get('best_peak_mc', 0),
                        'tokens_launched_7d': history.get('recent_launches', 0),
                        'median_peak_mc_usd': history.get('median_peak_mc', 0),
                        'tokens_with_mc_data': history.get('tokens_with_mc_data', 0)
                    })
                    
                    # Log enrichment details for debugging
                    self.logger.debug(f"Dev {dev_wallet[:8]}... enriched: {history.get('total_tokens', 0)} tokens, "
                                     f"best ATH MC ${profile.get('best_peak_mc_usd', 0):,.0f}, "
                                     f"median ATH MC ${profile.get('median_peak_mc_usd', 0):,.0f}, "
                                     f"{history.get('recent_launches', 0)} recent launches, "
                                     f"{history.get('tokens_with_mc_data', 0)} tokens with MC data")
                else:
                    self.logger.debug(f"Dev {dev_wallet[:8]}... no history returned from BitQuery")
            
            # Additional Moralis data if needed
            if self.moralis:
                # Can fetch additional on-chain data here
                pass
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error enriching profile for {dev_wallet[:8]}...: {e}")
            return None
    
    async def refresh_dev_scores(self):
        """Periodic refresh of dev scores and whitelist"""
        self.logger.info("Refreshing dev scores and whitelist")
        
        # Get all cached profiles
        for dev_wallet in list(self.store.dev_profiles_cache.keys()):
            profile = self.store.get_dev_profile(dev_wallet)
            
            if not profile:
                continue
            
            # Check if still meets quality criteria
            if self._is_quality_dev(profile):
                # Update whitelist if not already there
                if not self.store.is_whitelisted(dev_wallet):
                    self.store.add_to_whitelist(dev_wallet, profile.get('score', 0))
            else:
                # Remove from whitelist if quality degraded
                if self.store.is_whitelisted(dev_wallet):
                    self.store.remove_from_whitelist(dev_wallet)
                    self.logger.info(f"Removed {dev_wallet[:8]}... from whitelist (quality degraded)")
        
        self.metrics.set("discovery.whitelisted_devs", len(self.store.whitelist_cache))
        self.metrics.set("discovery.cached_profiles", len(self.store.dev_profiles_cache))