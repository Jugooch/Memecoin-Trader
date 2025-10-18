"""
Funding graph builder for sybil detection

Tracks wallet funding relationships using RPC lookups with caching
"""

import asyncio
from typing import Dict, Set, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass
import base58


@dataclass
class FundingRelationship:
    """A funding relationship between two wallets"""
    parent: str  # Who funded
    child: str   # Who was funded
    amount_sol: float
    timestamp: datetime
    signature: str


class FundingGraph:
    """
    Maintains wallet funding relationships for sybil detection

    Uses LRU cache + SMART trigger-based RPC lookups
    Only queries RPC for wallets showing suspicious coordinated patterns
    """

    def __init__(self, rpc_manager=None, cache_ttl_hours: float = 24.0,
                 max_cache_size: int = 10000, trigger_mode: bool = True):
        self.rpc_manager = rpc_manager

        # Cache: child -> FundingRelationship
        self.funding_cache: Dict[str, FundingRelationship] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.max_cache_size = max_cache_size

        # Graph structure: parent -> {children}
        self.parent_to_children: Dict[str, Set[str]] = defaultdict(set)

        # Pending lookups (rate limiting)
        self.pending_lookups: Set[str] = set()
        self.lookup_queue: asyncio.Queue = asyncio.Queue()
        self.lookup_worker_running = False

        # Trigger-based lookup mode (default: True)
        self.trigger_mode = trigger_mode
        self.suspicious_wallets: Set[str] = set()  # Wallets flagged for lookup

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.rpc_calls = 0
        self.rpc_calls_avoided = 0  # How many we saved with smart triggers

    async def start_lookup_worker(self):
        """Start background worker for RPC lookups"""
        if self.lookup_worker_running:
            return

        self.lookup_worker_running = True
        asyncio.create_task(self._lookup_worker())

    async def _lookup_worker(self):
        """Background worker that processes RPC lookups with rate limiting"""
        while self.lookup_worker_running:
            try:
                wallet = await asyncio.wait_for(
                    self.lookup_queue.get(),
                    timeout=1.0
                )

                await self._fetch_funding_source(wallet)

                # Rate limit: 10 RPS
                await asyncio.sleep(0.1)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                # Log but continue
                continue

    def flag_suspicious_wallet(self, wallet: str, reason: str = ""):
        """
        Flag a wallet as suspicious and queue for RPC lookup

        Args:
            wallet: Wallet address
            reason: Why it's suspicious (for debugging)
        """
        if wallet not in self.suspicious_wallets:
            self.suspicious_wallets.add(wallet)
            # Queue for lookup if not already cached or pending
            if wallet not in self.funding_cache and wallet not in self.pending_lookups:
                # Add to queue (will be processed by worker)
                try:
                    self.lookup_queue.put_nowait(wallet)
                    self.pending_lookups.add(wallet)
                except asyncio.QueueFull:
                    pass  # Skip if queue full

    async def get_funding_parent(self, wallet: str, max_wait_seconds: float = 0.5) -> Optional[str]:
        """
        Get funding parent for a wallet (async, cached)

        In trigger_mode: Only queries RPC for wallets flagged as suspicious
        Otherwise: Queries all wallets (legacy behavior)

        Args:
            wallet: Wallet address to lookup
            max_wait_seconds: Max time to wait for RPC (0 = don't wait)

        Returns:
            Parent wallet address, or None if unknown
        """
        # Check cache
        if wallet in self.funding_cache:
            cache_age = datetime.now() - self.cache_timestamps[wallet]
            if cache_age < self.cache_ttl:
                self.cache_hits += 1
                return self.funding_cache[wallet].parent

        self.cache_misses += 1

        # OPTIMIZATION: In trigger mode, only lookup if flagged as suspicious
        if self.trigger_mode and wallet not in self.suspicious_wallets:
            self.rpc_calls_avoided += 1
            return None

        # Check if already pending
        if wallet in self.pending_lookups:
            if max_wait_seconds > 0:
                # Wait briefly
                await asyncio.sleep(max_wait_seconds)
                if wallet in self.funding_cache:
                    return self.funding_cache[wallet].parent
            return None

        # Queue for lookup
        self.pending_lookups.add(wallet)
        await self.lookup_queue.put(wallet)

        # Optionally wait
        if max_wait_seconds > 0:
            await asyncio.sleep(max_wait_seconds)
            if wallet in self.funding_cache:
                return self.funding_cache[wallet].parent

        return None

    def get_funding_parent_sync(self, wallet: str) -> Optional[str]:
        """
        Get funding parent synchronously (cache-only, no RPC)

        Args:
            wallet: Wallet address

        Returns:
            Parent wallet address if in cache, else None
        """
        if wallet in self.funding_cache:
            cache_age = datetime.now() - self.cache_timestamps[wallet]
            if cache_age < self.cache_ttl:
                return self.funding_cache[wallet].parent

        return None

    async def _fetch_funding_source(self, wallet: str):
        """Fetch funding source via RPC"""
        if not self.rpc_manager:
            # No RPC manager, can't fetch
            self.pending_lookups.discard(wallet)
            return

        try:
            self.rpc_calls += 1

            # Get last 5 signatures
            sigs_response = await self.rpc_manager.call_http_rpc(
                method="getSignaturesForAddress",
                params=[wallet, {"limit": 5}],
                timeout=2.0
            )

            if not sigs_response or 'result' not in sigs_response:
                self.pending_lookups.discard(wallet)
                return

            signatures = sigs_response['result']
            if not signatures:
                self.pending_lookups.discard(wallet)
                return

            # Find first inbound SOL transfer
            for sig_info in reversed(signatures):  # Oldest first
                sig = sig_info['signature']

                tx_response = await self.rpc_manager.call_http_rpc(
                    method="getTransaction",
                    params=[sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
                    timeout=2.0
                )

                if not tx_response or 'result' not in tx_response:
                    continue

                result = tx_response['result']
                if not result or result.get('meta', {}).get('err'):
                    continue

                # Parse for SOL transfer TO this wallet
                funding = self._parse_funding_from_tx(result, wallet)
                if funding:
                    # Cache it
                    self._add_to_cache(funding)
                    break

        except Exception as e:
            # Silently fail
            pass
        finally:
            self.pending_lookups.discard(wallet)

    def _parse_funding_from_tx(self, tx_result: Dict, target_wallet: str) -> Optional[FundingRelationship]:
        """Parse funding relationship from transaction"""
        try:
            meta = tx_result.get('meta', {})
            transaction = tx_result.get('transaction', {})
            message = transaction.get('message', {})
            instructions = message.get('instructions', [])

            # Look for System Program transfer
            for ix in instructions:
                if ix.get('program') == 'system' and ix.get('parsed', {}).get('type') == 'transfer':
                    info = ix['parsed']['info']
                    destination = info.get('destination')
                    source = info.get('source')
                    lamports = info.get('lamports', 0)

                    if destination == target_wallet and lamports > 0:
                        # Found funding transaction
                        return FundingRelationship(
                            parent=source,
                            child=target_wallet,
                            amount_sol=lamports / 1e9,
                            timestamp=datetime.now(),  # Approximate
                            signature=tx_result.get('transaction', {}).get('signatures', [''])[0]
                        )

        except Exception:
            pass

        return None

    def _add_to_cache(self, funding: FundingRelationship):
        """Add funding relationship to cache"""
        # Evict if cache full
        if len(self.funding_cache) >= self.max_cache_size:
            # Evict oldest
            oldest_wallet = min(self.cache_timestamps.keys(),
                              key=lambda w: self.cache_timestamps[w])
            del self.funding_cache[oldest_wallet]
            del self.cache_timestamps[oldest_wallet]

            # Remove from graph
            parent = self.funding_cache.get(oldest_wallet)
            if parent:
                self.parent_to_children[parent.parent].discard(oldest_wallet)

        # Add to cache
        self.funding_cache[funding.child] = funding
        self.cache_timestamps[funding.child] = datetime.now()

        # Add to graph
        self.parent_to_children[funding.parent].add(funding.child)

    def get_siblings(self, wallet: str, max_age_seconds: float = 90.0) -> Set[str]:
        """
        Get sibling wallets (funded by same parent within time window)

        Args:
            wallet: Wallet to check
            max_age_seconds: Max age difference for siblings

        Returns:
            Set of sibling wallet addresses
        """
        parent = self.get_funding_parent_sync(wallet)
        if not parent:
            return set()

        siblings = set()
        wallet_funding = self.funding_cache.get(wallet)
        if not wallet_funding:
            return siblings

        # Find all children of this parent
        for child in self.parent_to_children[parent]:
            if child == wallet:
                continue

            child_funding = self.funding_cache.get(child)
            if not child_funding:
                continue

            # Check time proximity
            time_diff = abs((child_funding.timestamp - wallet_funding.timestamp).total_seconds())
            if time_diff <= max_age_seconds:
                siblings.add(child)

        return siblings

    def get_cluster_size(self, wallet: str, max_age_seconds: float = 90.0) -> int:
        """Get size of funding cluster (1 + siblings)"""
        siblings = self.get_siblings(wallet, max_age_seconds)
        return 1 + len(siblings)

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        return {
            'cached_wallets': len(self.funding_cache),
            'parent_nodes': len(self.parent_to_children),
            'suspicious_flagged': len(self.suspicious_wallets),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / total_requests if total_requests > 0 else 0.0,
            'rpc_calls': self.rpc_calls,
            'rpc_calls_avoided': self.rpc_calls_avoided,
            'efficiency': f"{self.rpc_calls_avoided / (self.rpc_calls + self.rpc_calls_avoided) * 100:.1f}%"
                         if (self.rpc_calls + self.rpc_calls_avoided) > 0 else "0.0%",
        }


def detect_coordinated_cluster(wallets: Set[str], funding_graph: FundingGraph,
                               min_cluster_size: int = 6) -> Tuple[bool, Set[str]]:
    """
    Detect if a group of wallets are part of a coordinated sybil cluster

    Args:
        wallets: Set of wallet addresses to check
        funding_graph: FundingGraph instance
        min_cluster_size: Minimum cluster size to flag

    Returns:
        (is_cluster, cluster_wallets)
    """
    # Group by funding parent
    parent_groups: Dict[str, Set[str]] = defaultdict(set)

    for wallet in wallets:
        parent = funding_graph.get_funding_parent_sync(wallet)
        if parent:
            parent_groups[parent].add(wallet)

    # Find largest cluster
    if not parent_groups:
        return (False, set())

    largest_cluster = max(parent_groups.values(), key=len)

    is_cluster = len(largest_cluster) >= min_cluster_size

    return (is_cluster, largest_cluster if is_cluster else set())
