"""
Simple RPC Manager for funding graph lookups

Minimal HTTP RPC client for getSignaturesForAddress and getTransaction
"""

import aiohttp
import asyncio
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SimpleRPCManager:
    """
    Lightweight RPC manager for funding graph queries

    Uses HTTP RPC endpoint (not gRPC) for getSignaturesForAddress and getTransaction
    """

    def __init__(self, rpc_endpoint: str, max_concurrent: int = 10):
        """
        Args:
            rpc_endpoint: HTTP RPC endpoint (e.g., https://api.mainnet-beta.solana.com)
            max_concurrent: Max concurrent RPC calls
        """
        self.rpc_endpoint = rpc_endpoint
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_count = 0

    async def start(self):
        """Initialize aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def stop(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def call_http_rpc(self, method: str, params: list, timeout: float = 2.0) -> Optional[Dict]:
        """
        Call Solana HTTP RPC

        Args:
            method: RPC method (e.g., "getSignaturesForAddress")
            params: RPC params list
            timeout: Request timeout in seconds

        Returns:
            RPC response dict or None on error
        """
        if not self.session:
            await self.start()

        async with self.semaphore:
            try:
                self.request_count += 1

                payload = {
                    "jsonrpc": "2.0",
                    "id": self.request_count,
                    "method": method,
                    "params": params
                }

                async with self.session.post(
                    self.rpc_endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"RPC call failed: {response.status}")
                        return None

            except asyncio.TimeoutError:
                logger.warning(f"RPC timeout: {method}")
                return None
            except Exception as e:
                logger.warning(f"RPC error: {e}")
                return None
