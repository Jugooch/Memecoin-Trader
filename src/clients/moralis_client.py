"""
Moralis API client for token metadata, liquidity, and price data
"""

import aiohttp
import asyncio
import logging
from typing import Dict, List, Optional
import time
import hashlib
import json


class MoralisClient:
    def __init__(self, api_keys):
        # Support both single key (string) and multiple keys (list)
        if isinstance(api_keys, str):
            self.api_keys = [api_keys]
        else:
            self.api_keys = api_keys
            
        self.base_url = "https://solana-gateway.moralis.io"
        self.logger = logging.getLogger(__name__)

        # Key rotation management
        self.current_key_index = 0
        self.key_stats = {}
        for i, key in enumerate(self.api_keys):
            self.key_stats[i] = {
                'calls_today': 0,
                'rate_limited': False,
                'reset_time': 0,
                'key': key
            }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 5 requests per second (conservative)     

        # In-flight request coalescing and concurrency control
        self._inflight = {}  # key -> asyncio.Task
        self._sem = asyncio.Semaphore(3)  # cap concurrent requests
        
        # Smart caching to reduce API calls - mode-aware TTLs
        self.cache = {}
        self.cache_ttl = {
            'metadata': 7200,     # Cache metadata for 2 hours (very stable data)
            'liquidity': 600,     # Cache liquidity for 10 minutes (reduce from 15)
            'price': 8,           # Cache price for 8 seconds (use fresh=True for monitoring)
            'swaps': 30,          # Cache swaps for 30 seconds (reduce calls during discovery)
            'swaps_alpha': 10,    # Short TTL for alpha checks (more frequent updates)
            'holders': 1800       # Cache holders for 30 minutes (changes slowly)
        }
        
        # Cache cleanup tracking
        self.last_cache_cleanup = time.time()
        self.cache_cleanup_interval = 3600  # Clean cache every hour
        
        # Global rate limiting (legacy compatibility)
        self.rate_limited = False
        self.rate_limit_reset_time = 0
        
        self.session = None

    def _get_next_available_key(self) -> tuple:
        """Get the next available API key that's not rate limited"""
        for _ in range(len(self.api_keys)):
            key_info = self.key_stats[self.current_key_index]
            
            # Check if this key is rate limited
            if key_info['rate_limited'] and time.time() < key_info['reset_time']:
                self.logger.debug(f"Key {self.current_key_index} still rate limited, trying next...")
                self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                continue
            
            # This key is available
            return self.current_key_index, key_info['key']
        
        # All keys are rate limited
        self.rate_limited = True
        self.rate_limit_reset_time = min(key['reset_time'] for key in self.key_stats.values())
        return None, None

    async def _get_session(self):
        """Get or create aiohttp session with current API key"""
        key_index, api_key = self._get_next_available_key()
        
        if not api_key:
            self.logger.warning("All API keys rate limited!")
            return None
            
        # Create new session if key changed or session doesn't exist
        if not self.session or getattr(self.session, '_current_key_index', None) != key_index:
            if self.session:
                await self.session.close()
                
            self.session = aiohttp.ClientSession(
                headers={
                    "Accept": "application/json",
                    "X-API-Key": api_key
                }
            )
            self.session._current_key_index = key_index
            self.logger.debug(f"Switched to API key #{key_index}")
        
        return self.session

    async def _rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def _get_cache_key(self, endpoint: str, params: Dict = None) -> str:
        """Generate cache key for request"""
        key_data = f"{endpoint}_{json.dumps(params, sort_keys=True) if params else ''}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict, cache_type: str) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False
        cache_time = cache_entry.get('timestamp', 0)
        ttl = self.cache_ttl.get(cache_type, 300)
        return time.time() - cache_time < ttl
    
    def _cleanup_expired_cache(self):
        """Remove expired cache entries to prevent memory bloat"""
        if time.time() - self.last_cache_cleanup < self.cache_cleanup_interval:
            return
            
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry.get('timestamp', 0) > 7200:  # Remove entries older than 2 hours
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
        self.last_cache_cleanup = current_time
    
    async def _make_request(self, url: str, params: Dict = None, cache_type: str = None) -> Dict:
        """Make HTTP request with intelligent caching, coalescing, and error handling"""

        self.logger.debug(f"Starting _make_request: url={url}, params={params}, cache_type={cache_type}")

        # Periodic cache cleanup
        self._cleanup_expired_cache()

        # Create request key for caching and coalescing
        cache_key = self._get_cache_key(url, params)
        self.logger.debug(f"Computed cache key: {cache_key}")

        # Check cache first
        if cache_type:
            cache_entry = self.cache.get(cache_key)
            if cache_entry:
                self.logger.debug(f"Found cache entry for {cache_type}")
            else:
                self.logger.debug(f"No cache entry for {cache_type}")

            if self._is_cache_valid(cache_entry, cache_type):
                self.logger.debug(f"Cache hit for {cache_type}, returning cached data")
                return cache_entry['data']

        # In-flight coalescing - check if same request is already running
        if cache_key in self._inflight:
            self.logger.debug(f"Coalescing request: {url}")
            data = await self._inflight[cache_key]
            self.logger.debug(f"Coalesced request completed: data={data}")
            return data

        # Create and track the actual HTTP request task
        async def _fetch():
            async with self._sem:  # Global concurrency cap
                self.logger.debug(f"Acquired semaphore for request: {url}")

                session = await self._get_session()
                if not session:
                    self.logger.warning("No available API keys — returning empty data")
                    return {}

                self.logger.debug(f"Using session {session} for request: {url}")
                await self._rate_limit()

                self.logger.debug(f"Making HTTP request to {url} with params={params}")
                data = await self._execute_request(session, url, params, cache_type, cache_key)
                self.logger.debug(f"HTTP request complete: {url}, received data={data}")
                return data

        # Store the task for coalescing
        task = asyncio.create_task(_fetch())
        self._inflight[cache_key] = task

        try:
            data = await task
            self.logger.debug(f"_make_request completed for {url} — returning data: {data}")
            return data
        finally:
            # Clean up after request completes
            self.logger.debug(f"Cleaning up inflight entry for {cache_key}")
            self._inflight.pop(cache_key, None)

    
    async def _execute_request(self, session, url, params, cache_type, cache_key):
       """Execute the actual HTTP request with retry logic"""
       current_key_index = getattr(session, '_current_key_index', 0)
       self.logger.debug(f"Starting _execute_request: url={url}, params={params}, key_index={current_key_index}")
    
       max_retries = len(self.api_keys)  # Try all keys before giving up
       for attempt in range(max_retries):
           self.logger.debug(f"Attempt {attempt + 1}/{max_retries} with key #{current_key_index}")
    
           try:
               async with session.get(url, params=params) as response:
                   self.logger.debug(f"HTTP status {response.status} for {url} (key #{current_key_index})")
    
                   # Track usage for current key
                   self.key_stats[current_key_index]['calls_today'] += 1
    
                   if response.status == 200:
                       data = await response.json()
                       self.logger.debug(f"Received data from {url}: {data}")
    
                       # Cache successful response
                       if cache_type:
                           self.cache[cache_key] = {
                               'data': data,
                               'timestamp': time.time()
                           }
                           self.logger.debug(f"Cached response for cache_type={cache_type}, key={cache_key}")
    
                       return data
    
                   elif response.status == 429:  # Rate limited
                       self.logger.warning(f"Key #{current_key_index} rate limited, rotating...")
                       self.key_stats[current_key_index]['rate_limited'] = True
                       self.key_stats[current_key_index]['reset_time'] = time.time() + 60
    
                       # Try next key
                       self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                       session = await self._get_session()
                       current_key_index = getattr(session, '_current_key_index', 0)
                       if not session:
                           self.logger.error("No session available after rate limit rotation — returning empty")
                           return {}
                       continue
                    
                   elif response.status == 401:  # API limit hit
                       error_text = await response.text()
                       self.logger.warning(f"Key #{current_key_index} got 401: {error_text}")
    
                       if "consumed" in error_text or "upgrade" in error_text:
                           self.logger.warning(f"Key #{current_key_index} daily limit reached, rotating...")
                           self.key_stats[current_key_index]['rate_limited'] = True
                           self.key_stats[current_key_index]['reset_time'] = time.time() + 86400  # 24 hours
    
                           # Try next key
                           self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                           session = await self._get_session()
                           current_key_index = getattr(session, '_current_key_index', 0)
                           if not session:
                               self.logger.error("All API keys exhausted — returning empty")
                               return {}
                           continue
                       else:
                           self.logger.error("401 unauthorized but not daily limit — returning empty")
                           return {}
    
                   else:
                       # Log 404s at info for now (debug before)
                       if response.status == 404:
                           self.logger.info(f"API 404 for {url} — token metadata not found")
                       else:
                           error_body = await response.text()
                           self.logger.error(f"API error {response.status}: {error_body}")
                       return {}
    
           except Exception as e:
               self.logger.error(f"Request failed (attempt {attempt + 1}): {e}")
    
               if attempt < max_retries - 1:
                   self.logger.debug("Rotating to next key due to request failure")
                   self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                   session = await self._get_session()
                   current_key_index = getattr(session, '_current_key_index', 0)
                   if not session:
                       self.logger.error("No session available after exception rotation — returning empty")
                       return {}
                   await asyncio.sleep(1)  # Small delay before retry
               else:
                   self.logger.error("Max retries reached — returning empty")
                   return {}
    
       self.logger.error("All retries exhausted — returning empty")
       return {}
    

    async def get_token_metadata(self, mint_address: str) -> Dict:
        """Get token metadata including name, symbol, supply"""
        url = f"{self.base_url}/token/mainnet/{mint_address}/metadata"
        
        try:
            data = await self._make_request(url, cache_type='metadata')
            
            return {
                'mint': mint_address,
                'name': data.get('name', ''),
                'symbol': data.get('symbol', ''),
                'decimals': data.get('decimals', 9),
                'supply': data.get('supply', 0),
                'logo': data.get('logoURI', ''),
                'description': data.get('description', ''),
                'tags': data.get('tags', [])
            }
            
        except Exception as e:
            self.logger.error(f"Error getting token metadata for {mint_address}: {e}")
            return {}

    async def get_token_liquidity(self, mint_address: str) -> Dict:
        """Get token liquidity information"""
        # Get token pairs
        pairs_url = f"{self.base_url}/token/mainnet/{mint_address}/pairs"
        
        try:
            pairs_data = await self._make_request(pairs_url, cache_type='liquidity')
            
            total_liquidity_usd = 0
            pools = []
            
            for pair in pairs_data.get('pairs', []):
                usd_value = pair.get('liquidityUsd', 0)
                total_liquidity_usd += usd_value
                
                pools.append({
                    'exchange': pair.get('exchangeName', ''),
                    'pair_address': pair.get('pairAddress', ''),
                    'liquidity_usd': usd_value,
                    'pair_label': pair.get('pairLabel', ''),
                    'usd_price': pair.get('usdPrice', 0),
                    'price_change_24h': pair.get('usdPrice24hrPercentChange', 0)
                })
            
            return {
                'mint': mint_address,
                'total_liquidity_usd': total_liquidity_usd,
                'pools': pools,
                'pool_count': len(pools)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting token liquidity for {mint_address}: {e}")
            return {'total_liquidity_usd': 0, 'pools': [], 'pool_count': 0}

    async def get_current_price(self, mint_address: str, fresh: bool = False) -> float:
        """Get current token price in USD"""
        url = f"{self.base_url}/token/mainnet/{mint_address}/price"
        
        try:
            # Skip cache if fresh=True (for position monitoring)
            cache_type = None if fresh else 'price'
            data = await self._make_request(url, cache_type=cache_type)
            self.logger.debug(f"Current data for {mint_address}: {data}")
            return float(data.get('usdPrice', 0))
            
        except Exception as e:
            self.logger.error(f"Error getting price for {mint_address}: {e}")
            return 0.0

    async def get_token_ohlcv(self, mint_address: str, timeframe: str = "1m", limit: int = 100) -> List[Dict]:
        """Get OHLCV data for token"""
        url = f"{self.base_url}/token/{mint_address}/ohlcv"
        params = {
            'timeframe': timeframe,
            'limit': limit
        }
        
        try:
            data = await self._make_request(url, params)
            return data.get('result', [])
            
        except Exception as e:
            self.logger.error(f"Error getting OHLCV for {mint_address}: {e}")
            return []

    async def get_token_swaps(self, mint_address: str, limit: int = 100, ttl_override: str = None) -> List[Dict]:
        """Get recent swap transactions for token with optional TTL override"""
        url = f"{self.base_url}/token/mainnet/{mint_address}/swaps"
        params = {'limit': limit}
        
        # Use override TTL for alpha checks (10s) vs discovery (30s)
        cache_type = ttl_override if ttl_override else 'swaps'
        
        try:
            data = await self._make_request(url, params, cache_type=cache_type)
            
            swaps = []
            for swap in data.get('result', []):
                # Extract token info from bought/sold objects
                bought = swap.get('bought', {})
                sold = swap.get('sold', {})
                
                # Determine which token is our target token and which is the other
                if bought.get('address') == mint_address:
                    # Trader bought our token with something else → BUY
                    side = 'buy'
                    from_token = sold.get('address', '')
                    to_token = mint_address
                    from_amount = float(sold.get('amount', 0))
                    to_amount = float(bought.get('amount', 0))
                elif sold.get('address') == mint_address:
                    # Trader sold our token for something else → SELL
                    side = 'sell'
                    from_token = mint_address
                    to_token = bought.get('address', '')
                    from_amount = float(sold.get('amount', 0))
                    to_amount = float(bought.get('amount', 0))
                else:
                    # Not directly our pair; skip
                    continue
                
                swaps.append({
                    'signature': swap.get('transactionHash', ''),
                    'timestamp': swap.get('blockTimestamp', ''),
                    'wallet': swap.get('walletAddress', ''),
                    'from_token': from_token,
                    'to_token': to_token,
                    'from_amount': from_amount,
                    'to_amount': to_amount,
                    'usd_value': float(swap.get('totalValueUsd', 0)),
                    'side': side
                })
            
            return swaps
            
        except Exception as e:
            self.logger.error(f"Error getting swaps for {mint_address}: {e}")
            return []


    async def get_wallet_portfolio(self, wallet_address: str) -> Dict:
        """Get wallet's token portfolio"""
        url = f"{self.base_url}/account/mainnet/{wallet_address}/portfolio"
        
        try:
            data = await self._make_request(url)
            
            tokens = []
            total_value = 0
            
            # Get native SOL balance
            native_balance = data.get('nativeBalance', {})
            sol_balance = float(native_balance.get('solana', 0))
            
            # Process tokens
            for token in data.get('tokens', []):
                # Note: New API doesn't have usdValue directly, would need to calculate or get from price API
                # For now, we'll use amount and mark USD value as 0 since it's not in the response
                amount = float(token.get('amount', 0))
                
                tokens.append({
                    'mint': token.get('mint', ''),
                    'symbol': token.get('symbol', ''),
                    'name': token.get('name', ''),
                    'amount': amount,
                    'amount_raw': token.get('amountRaw', ''),
                    'decimals': token.get('decimals', 9),
                    'logo': token.get('logo', ''),
                    'usd_value': 0,  # Would need separate price lookup
                    'possible_spam': token.get('possibleSpam', False)
                })
            
            return {
                'wallet': wallet_address,
                'sol_balance': sol_balance,
                'total_value_usd': total_value,  # Would need price calculations
                'tokens': tokens,
                'token_count': len(tokens),
                'native_balance': native_balance
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio for {wallet_address}: {e}")
            return {}

    async def get_token_holders(self, mint_address: str) -> Dict:
        """Get token holder statistics"""
        url = f"{self.base_url}/token/mainnet/holders/{mint_address}"
        
        try:
            data = await self._make_request(url, cache_type='holders')
            
            if not data:
                return {'holder_count': 0, 'holder_stats': {}}
            
            # Extract holder count directly from response
            holder_count = data.get('totalHolders', 0)
            
            # Get additional holder statistics
            holder_stats = {
                'total_holders': holder_count,
                'acquisition_breakdown': data.get('holdersByAcquisition', {}),
                'holder_changes': data.get('holderChange', {}),
                'holder_distribution': data.get('holderDistribution', {})
            }
            
            return {
                'holder_count': holder_count,
                'holder_stats': holder_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting holders for {mint_address}: {e}")
            return {'holder_count': 0, 'holder_stats': {}}


    async def close(self):
        """Close the HTTP session"""
        if self.session:
            try:
                await self.session.close()
                # Wait a bit for the underlying connection to close
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.debug(f"Error closing session: {e}")
            finally:
                self.session = None