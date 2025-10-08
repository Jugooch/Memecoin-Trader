"""
Multi-Wallet Manager for Frontrun Bot
Manages pool of trading wallets with rotation and locking to prevent concurrent usage
"""

import asyncio
import time
from typing import Dict, List, Optional, AsyncIterator
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from datetime import datetime

from solders.keypair import Keypair
from solders.pubkey import Pubkey

from core.rpc_manager import RPCManager
from core.logger import get_logger
from core.metrics import get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


LAMPORTS_PER_SOL = 1_000_000_000


@dataclass
class WalletInfo:
    """Information about a managed wallet"""
    pubkey: Pubkey
    keypair: Keypair
    balance_lamports: int = 0
    balance_cached_at: float = 0
    last_used_at: float = 0
    in_cooldown_until: float = 0
    total_uses: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def balance_sol(self) -> float:
        """Balance in SOL"""
        return self.balance_lamports / LAMPORTS_PER_SOL

    @property
    def is_locked(self) -> bool:
        """Whether wallet is currently locked"""
        return self.lock.locked()

    @property
    def is_in_cooldown(self) -> bool:
        """Whether wallet is in cooldown period"""
        return time.time() < self.in_cooldown_until

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "pubkey": str(self.pubkey),
            "balance_sol": self.balance_sol,
            "last_used_at": self.last_used_at,
            "in_cooldown": self.is_in_cooldown,
            "is_locked": self.is_locked,
            "total_uses": self.total_uses
        }


@dataclass
class WalletManagerConfig:
    """Configuration for wallet manager"""
    balance_cache_ttl_seconds: int = 30
    cooldown_seconds: int = 5
    min_balance_threshold_lamports: int = 10_000_000  # 0.01 SOL


class WalletManager:
    """
    Manages multiple trading wallets with rotation and locking

    Features:
    - asyncio.Lock per wallet (prevents concurrent usage)
    - Balance tracking with caching
    - Cooldown periods between uses
    - Round-robin rotation
    - Minimum balance enforcement

    CRITICAL: Every wallet operation acquires a lock. NO wallet should
    ever be used concurrently. This is enforced by asyncio.Lock.

    Usage:
        manager = WalletManager(rpc_manager, keypairs)
        await manager.start()

        # Get available wallet (with lock acquired)
        async with manager.get_wallet(min_sol=0.1) as wallet:
            # Use wallet for trade
            # Lock is automatically released on exit
            pass
    """

    def __init__(
        self,
        rpc_manager: RPCManager,
        keypairs: List[Keypair],
        config: Optional[WalletManagerConfig] = None
    ):
        """
        Initialize wallet manager

        Args:
            rpc_manager: RPC manager for balance queries
            keypairs: List of keypairs to manage
            config: Wallet manager configuration (optional)
        """
        self.rpc_manager = rpc_manager
        self.config = config or WalletManagerConfig()

        # Initialize wallet info for each keypair
        self.wallets: Dict[Pubkey, WalletInfo] = {}
        for keypair in keypairs:
            pubkey = keypair.pubkey()
            self.wallets[pubkey] = WalletInfo(
                pubkey=pubkey,
                keypair=keypair
            )

        self._running = False
        self._last_rotation_index = 0

        logger.info(
            "wallet_manager_initialized",
            wallet_count=len(self.wallets),
            cooldown_seconds=self.config.cooldown_seconds,
            balance_cache_ttl=self.config.balance_cache_ttl_seconds
        )

    async def start(self) -> None:
        """Start wallet manager (fetch initial balances)"""
        if self._running:
            logger.warning("wallet_manager_already_running")
            return

        self._running = True
        logger.info("wallet_manager_starting")

        # Fetch initial balances
        await self.refresh_balances()

        logger.info("wallet_manager_started", total_balance_sol=self.get_total_balance_sol())

    async def stop(self) -> None:
        """Stop wallet manager"""
        if not self._running:
            return

        logger.info("wallet_manager_stopping")
        self._running = False

        logger.info("wallet_manager_stopped")

    @asynccontextmanager
    async def get_wallet(
        self,
        min_sol: float = 0.0,
        timeout_seconds: float = 10.0
    ) -> AsyncIterator[WalletInfo]:
        """
        Get an available wallet with lock acquired (context manager)

        Args:
            min_sol: Minimum SOL balance required
            timeout_seconds: Timeout for acquiring wallet lock

        Yields:
            WalletInfo with lock acquired

        Raises:
            TimeoutError: If no wallet available within timeout
            RuntimeError: If no wallets meet requirements

        Example:
            async with manager.get_wallet(min_sol=0.1) as wallet:
                # Use wallet.keypair for signing
                # Lock is automatically released on exit
                pass
        """
        min_balance_lamports = int(min_sol * LAMPORTS_PER_SOL)

        wallet = await self._get_available_wallet(
            min_balance_lamports=min_balance_lamports,
            timeout_seconds=timeout_seconds
        )

        if wallet is None:
            raise RuntimeError(
                f"No wallet available with min balance {min_sol} SOL within {timeout_seconds}s"
            )

        # Lock is already acquired by _get_available_wallet
        try:
            # Update usage stats
            wallet.last_used_at = time.time()
            wallet.total_uses += 1

            logger.debug(
                "wallet_acquired",
                pubkey=str(wallet.pubkey),
                balance_sol=wallet.balance_sol,
                total_uses=wallet.total_uses
            )

            metrics.increment_counter(
                "wallet_acquisitions",
                labels={"pubkey": str(wallet.pubkey)[:8]}
            )

            yield wallet

        finally:
            # Set cooldown
            wallet.in_cooldown_until = time.time() + self.config.cooldown_seconds

            # Release lock
            wallet.lock.release()

            logger.debug(
                "wallet_released",
                pubkey=str(wallet.pubkey),
                cooldown_seconds=self.config.cooldown_seconds
            )

    async def _get_available_wallet(
        self,
        min_balance_lamports: int,
        timeout_seconds: float
    ) -> Optional[WalletInfo]:
        """
        Get an available wallet (internal method)

        Acquires lock before returning. Caller MUST release lock.

        Args:
            min_balance_lamports: Minimum balance required
            timeout_seconds: Timeout for acquiring lock

        Returns:
            WalletInfo with lock acquired, or None if timeout
        """
        start_time = time.time()

        # Try to find available wallet (round-robin starting from last rotation index)
        wallet_list = list(self.wallets.values())
        num_wallets = len(wallet_list)

        # Check for empty wallet list
        if num_wallets == 0:
            raise RuntimeError("No wallets available")

        attempt = 0
        while time.time() - start_time < timeout_seconds:
            # Get next wallet in rotation
            wallet = wallet_list[self._last_rotation_index % num_wallets]
            self._last_rotation_index += 1

            # Check if wallet meets requirements
            if wallet.is_in_cooldown:
                await asyncio.sleep(0.1)
                attempt += 1
                continue

            # Refresh balance if cache expired
            if await self._should_refresh_balance(wallet):
                await self._refresh_wallet_balance(wallet)

            # Check balance
            if wallet.balance_lamports < min_balance_lamports:
                logger.debug(
                    "wallet_insufficient_balance",
                    pubkey=str(wallet.pubkey),
                    balance_sol=wallet.balance_sol,
                    required_sol=min_balance_lamports / LAMPORTS_PER_SOL
                )
                await asyncio.sleep(0.1)
                attempt += 1
                continue

            # Try to acquire lock
            try:
                await asyncio.wait_for(
                    wallet.lock.acquire(),
                    timeout=min(0.5, timeout_seconds - (time.time() - start_time))
                )

                # Success! Return wallet with lock acquired
                return wallet

            except asyncio.TimeoutError:
                # Wallet is locked by another task
                logger.debug(
                    "wallet_lock_timeout",
                    pubkey=str(wallet.pubkey)
                )
                await asyncio.sleep(0.1)
                attempt += 1
                continue

        # Timeout - no wallet available
        logger.warning(
            "no_wallet_available_timeout",
            timeout_seconds=timeout_seconds,
            min_balance_sol=min_balance_lamports / LAMPORTS_PER_SOL,
            attempts=attempt
        )
        metrics.increment_counter("wallet_acquisition_timeouts")

        return None

    async def refresh_balances(self) -> Dict[Pubkey, float]:
        """
        Refresh balances for all wallets

        Returns:
            Dictionary of pubkey -> balance in SOL
        """
        logger.debug("refreshing_wallet_balances", wallet_count=len(self.wallets))

        balances = {}

        # Refresh each wallet balance
        for pubkey, wallet in self.wallets.items():
            balance_sol = await self._refresh_wallet_balance(wallet)
            balances[pubkey] = balance_sol

        logger.info(
            "wallet_balances_refreshed",
            total_balance_sol=sum(balances.values()),
            wallet_count=len(balances)
        )

        metrics.set_gauge("total_wallet_balance_sol", sum(balances.values()))

        return balances

    async def _refresh_wallet_balance(self, wallet: WalletInfo) -> float:
        """
        Refresh balance for a single wallet

        Args:
            wallet: Wallet to refresh

        Returns:
            Balance in SOL
        """
        try:
            # Get balance via RPC
            response = await self.rpc_manager.call_http_rpc(
                "getBalance",
                [str(wallet.pubkey)],
                timeout=5.0
            )

            balance_lamports = response.get("result", {}).get("value", 0)

            # Update wallet info
            wallet.balance_lamports = balance_lamports
            wallet.balance_cached_at = time.time()

            logger.debug(
                "wallet_balance_updated",
                pubkey=str(wallet.pubkey),
                balance_sol=wallet.balance_sol
            )

            return wallet.balance_sol

        except Exception as e:
            logger.error(
                "wallet_balance_refresh_failed",
                pubkey=str(wallet.pubkey),
                error=str(e)
            )
            return wallet.balance_sol  # Return cached balance

    async def _should_refresh_balance(self, wallet: WalletInfo) -> bool:
        """
        Check if wallet balance cache has expired

        Args:
            wallet: Wallet to check

        Returns:
            True if balance should be refreshed
        """
        if wallet.balance_cached_at == 0:
            return True  # Never cached

        age = time.time() - wallet.balance_cached_at
        return age > self.config.balance_cache_ttl_seconds

    def get_wallet_info(self, pubkey: Pubkey) -> Optional[WalletInfo]:
        """
        Get wallet info by public key

        Args:
            pubkey: Public key to lookup

        Returns:
            WalletInfo if found, None otherwise
        """
        return self.wallets.get(pubkey)

    def get_all_wallets(self) -> List[WalletInfo]:
        """
        Get all managed wallets

        Returns:
            List of all WalletInfo
        """
        return list(self.wallets.values())

    def get_total_balance_sol(self) -> float:
        """
        Get total balance across all wallets (cached)

        Returns:
            Total balance in SOL
        """
        return sum(wallet.balance_sol for wallet in self.wallets.values())

    def get_available_wallet_count(self, min_balance_lamports: int = 0) -> int:
        """
        Get count of available wallets (not locked, not in cooldown, sufficient balance)

        Args:
            min_balance_lamports: Minimum balance required

        Returns:
            Count of available wallets
        """
        available = 0
        for wallet in self.wallets.values():
            if (
                not wallet.is_locked and
                not wallet.is_in_cooldown and
                wallet.balance_lamports >= min_balance_lamports
            ):
                available += 1

        return available

    def get_stats(self) -> dict:
        """
        Get wallet manager statistics

        Returns:
            Dictionary with stats
        """
        locked_count = sum(1 for w in self.wallets.values() if w.is_locked)
        cooldown_count = sum(1 for w in self.wallets.values() if w.is_in_cooldown)
        total_uses = sum(w.total_uses for w in self.wallets.values())

        return {
            "total_wallets": len(self.wallets),
            "locked_wallets": locked_count,
            "cooldown_wallets": cooldown_count,
            "available_wallets": len(self.wallets) - locked_count - cooldown_count,
            "total_balance_sol": self.get_total_balance_sol(),
            "total_uses": total_uses
        }


# Example usage
if __name__ == "__main__":
    from core.config import ConfigurationManager
    from core.logger import setup_logging

    async def main():
        # Setup logging
        setup_logging(level="INFO", format="console")

        # Load config
        config_manager = ConfigurationManager("config/config.yml")
        bot_config = config_manager.load_config()

        # Initialize RPC manager
        rpc_manager = RPCManager(bot_config.rpc_config)
        await rpc_manager.start()

        # Create some test keypairs
        keypairs = [Keypair() for _ in range(3)]

        # Initialize wallet manager
        manager = WalletManager(rpc_manager, keypairs)
        await manager.start()

        try:
            logger.info("wallet_manager_test_started")

            # Test wallet acquisition
            async with manager.get_wallet(min_sol=0.0) as wallet:
                logger.info(
                    "wallet_acquired_for_use",
                    pubkey=str(wallet.pubkey),
                    balance_sol=wallet.balance_sol
                )

                # Simulate some work
                await asyncio.sleep(1)

            # Get stats
            stats = manager.get_stats()
            logger.info("wallet_manager_stats", **stats)

            logger.info("wallet_manager_test_completed")

        except Exception as e:
            logger.error("test_error", error=str(e), exc_info=True)
        finally:
            await manager.stop()
            await rpc_manager.stop()

    asyncio.run(main())
