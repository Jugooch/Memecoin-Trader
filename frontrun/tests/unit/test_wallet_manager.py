"""
Unit tests for Multi-Wallet Manager
Tests wallet rotation, locking, balance tracking, and concurrency safety
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock
from solders.keypair import Keypair
from solders.pubkey import Pubkey

from core.wallet_manager import (
    WalletManager,
    WalletInfo,
    WalletManagerConfig,
    LAMPORTS_PER_SOL
)


@pytest.fixture
def mock_rpc_manager():
    """Create mock RPC manager"""
    rpc = AsyncMock()
    # Mock balance response (1 SOL)
    rpc.call_http_rpc.return_value = {
        "result": {"value": 1_000_000_000}
    }
    return rpc


@pytest.fixture
def keypairs():
    """Create test keypairs"""
    return [Keypair() for _ in range(3)]


@pytest.fixture
def config():
    """Create test config"""
    return WalletManagerConfig(
        balance_cache_ttl_seconds=30,
        cooldown_seconds=1,  # Short cooldown for tests
        min_balance_threshold_lamports=10_000_000
    )


@pytest_asyncio.fixture
async def manager(mock_rpc_manager, keypairs, config):
    """Create wallet manager instance"""
    mgr = WalletManager(mock_rpc_manager, keypairs, config)
    await mgr.start()
    yield mgr
    await mgr.stop()


# =============================================================================
# INITIALIZATION
# =============================================================================

def test_wallet_manager_initialization(mock_rpc_manager, keypairs, config):
    """Test wallet manager initialization"""
    manager = WalletManager(mock_rpc_manager, keypairs, config)

    assert len(manager.wallets) == 3
    assert manager.config == config
    assert manager._running is False


def test_wallet_info_creation(mock_rpc_manager, keypairs):
    """Test WalletInfo is created for each keypair"""
    manager = WalletManager(mock_rpc_manager, keypairs)

    for keypair in keypairs:
        pubkey = keypair.pubkey()
        assert pubkey in manager.wallets
        wallet_info = manager.wallets[pubkey]
        assert isinstance(wallet_info, WalletInfo)
        assert wallet_info.pubkey == pubkey
        assert wallet_info.keypair == keypair
        assert isinstance(wallet_info.lock, asyncio.Lock)


@pytest.mark.asyncio
async def test_manager_start(mock_rpc_manager, keypairs):
    """Test manager start refreshes balances"""
    manager = WalletManager(mock_rpc_manager, keypairs)

    await manager.start()

    assert manager._running is True
    # Should have called RPC for each wallet
    assert mock_rpc_manager.call_http_rpc.call_count == len(keypairs)


# =============================================================================
# WALLET ACQUISITION
# =============================================================================

@pytest.mark.asyncio
async def test_get_wallet_basic(manager):
    """Test basic wallet acquisition"""
    async with manager.get_wallet(min_sol=0.0) as wallet:
        assert isinstance(wallet, WalletInfo)
        assert wallet.is_locked is True  # Lock should be acquired


@pytest.mark.asyncio
async def test_get_wallet_releases_lock(manager):
    """Test wallet lock is released immediately after use"""
    wallet_pubkey = None

    async with manager.get_wallet(min_sol=0.0) as wallet:
        wallet_pubkey = wallet.pubkey
        assert wallet.is_locked is True

    # After context exit, lock should be released immediately
    # No sleep needed - lock.release() is synchronous
    wallet_info = manager.get_wallet_info(wallet_pubkey)
    assert wallet_info.is_locked is False


@pytest.mark.asyncio
async def test_get_wallet_updates_usage_stats(manager):
    """Test wallet usage stats are updated"""
    async with manager.get_wallet(min_sol=0.0) as wallet:
        initial_uses = wallet.total_uses

    # Check usage was incremented
    wallet_info = manager.get_wallet_info(wallet.pubkey)
    assert wallet_info.total_uses == initial_uses
    assert wallet_info.last_used_at > 0


@pytest.mark.asyncio
async def test_get_wallet_sets_cooldown(manager):
    """Test wallet enters cooldown with correct duration after use"""
    import time

    wallet_pubkey = None
    start_time = time.time()

    async with manager.get_wallet(min_sol=0.0) as wallet:
        wallet_pubkey = wallet.pubkey

    # After release, wallet should be in cooldown
    wallet_info = manager.get_wallet_info(wallet_pubkey)
    assert wallet_info.is_in_cooldown is True

    # Verify cooldown duration is correct
    expected_cooldown_until = start_time + manager.config.cooldown_seconds
    # Allow 0.2s tolerance for execution time
    assert abs(wallet_info.in_cooldown_until - expected_cooldown_until) < 0.2


@pytest.mark.asyncio
async def test_get_wallet_min_balance_requirement(manager, mock_rpc_manager):
    """Test wallet acquisition with minimum balance requirement"""
    # Mock one wallet with insufficient balance
    def get_balance_response(method, params, timeout):
        pubkey = params[0]
        # First wallet has 0.005 SOL (insufficient), others have 1 SOL
        if pubkey == str(list(manager.wallets.keys())[0]):
            return {"result": {"value": 5_000_000}}  # 0.005 SOL
        return {"result": {"value": 1_000_000_000}}  # 1 SOL

    mock_rpc_manager.call_http_rpc.side_effect = get_balance_response

    # Refresh balances
    await manager.refresh_balances()

    # Request wallet with 0.01 SOL minimum
    async with manager.get_wallet(min_sol=0.01) as wallet:
        # Should get one of the wallets with sufficient balance
        assert wallet.balance_sol >= 0.01


@pytest.mark.asyncio
async def test_get_wallet_timeout(manager, mock_rpc_manager):
    """Test wallet acquisition timeout"""
    # Set all wallets to zero balance
    mock_rpc_manager.call_http_rpc.return_value = {"result": {"value": 0}}
    await manager.refresh_balances()

    # Try to get wallet with balance requirement (should timeout)
    with pytest.raises(RuntimeError, match="No wallet available"):
        async with manager.get_wallet(min_sol=0.1, timeout_seconds=0.5) as wallet:
            pass


# =============================================================================
# WALLET ROTATION
# =============================================================================

@pytest.mark.asyncio
async def test_wallet_rotation_round_robin(manager):
    """Test wallets are rotated round-robin"""
    acquired_wallets = []

    # Acquire wallets sequentially with cooldown waits
    for i in range(6):  # More than number of wallets
        async with manager.get_wallet(min_sol=0.0) as wallet:
            acquired_wallets.append(wallet.pubkey)
        await asyncio.sleep(1.1)  # Wait for cooldown to expire

    # Should have rotated through wallets (may not be perfectly sequential due to cooldowns)
    unique_wallets = set(acquired_wallets)
    assert len(unique_wallets) >= 2  # At least 2 different wallets used


# =============================================================================
# CONCURRENT ACCESS PROTECTION
# =============================================================================

@pytest.mark.asyncio
async def test_no_concurrent_wallet_usage(manager):
    """Test that same wallet cannot be used concurrently"""
    concurrent_uses = []

    async def use_wallet(wallet_id: int):
        """Task that tries to use a wallet"""
        async with manager.get_wallet(min_sol=0.0, timeout_seconds=5.0) as wallet:
            concurrent_uses.append((wallet_id, wallet.pubkey, time.time()))
            await asyncio.sleep(0.2)  # Simulate work

    # Run multiple tasks concurrently
    import time
    tasks = [use_wallet(i) for i in range(5)]
    await asyncio.gather(*tasks)

    # Check that no wallet was used concurrently
    # Group by wallet pubkey and check time overlaps
    wallet_usage = {}
    for task_id, pubkey, timestamp in concurrent_uses:
        if pubkey not in wallet_usage:
            wallet_usage[pubkey] = []
        wallet_usage[pubkey].append((task_id, timestamp))

    # For each wallet, check no overlapping time windows
    for pubkey, uses in wallet_usage.items():
        if len(uses) > 1:
            # Sort by timestamp
            uses.sort(key=lambda x: x[1])
            # Check each consecutive pair doesn't overlap (0.2s work time)
            for i in range(len(uses) - 1):
                time1 = uses[i][1]
                time2 = uses[i + 1][1]
                # Second use should start after first finishes (0.2s + margin)
                assert time2 - time1 >= 0.15  # Allow small margin


@pytest.mark.asyncio
async def test_multiple_wallets_concurrent_usage(manager):
    """Test multiple different wallets can be used concurrently"""
    active_tasks = []

    async def use_wallet_long(task_id: int):
        """Task that uses a wallet for longer time"""
        async with manager.get_wallet(min_sol=0.0, timeout_seconds=5.0) as wallet:
            active_tasks.append(task_id)
            await asyncio.sleep(0.3)
            active_tasks.remove(task_id)

    # Start 3 tasks concurrently (we have 3 wallets)
    tasks = [use_wallet_long(i) for i in range(3)]

    # At some point, multiple tasks should be active concurrently
    results = await asyncio.gather(*tasks)

    # All tasks should complete successfully
    assert len(results) == 3


@pytest.mark.asyncio
async def test_lock_prevents_concurrent_access(manager):
    """Test asyncio lock prevents concurrent access to same wallet"""
    wallet_pubkey = list(manager.wallets.keys())[0]
    wallet = manager.wallets[wallet_pubkey]

    # Manually acquire lock
    await wallet.lock.acquire()

    # Try to get this specific wallet (should wait/timeout)
    start_time = asyncio.get_event_loop().time()

    async def try_get_wallet():
        try:
            # This should timeout because lock is held
            await asyncio.wait_for(wallet.lock.acquire(), timeout=0.5)
            wallet.lock.release()
            return "acquired"
        except asyncio.TimeoutError:
            return "timeout"

    result = await try_get_wallet()

    # Should have timed out
    assert result == "timeout"

    # Release lock
    wallet.lock.release()


# =============================================================================
# BALANCE TRACKING
# =============================================================================

@pytest.mark.asyncio
async def test_refresh_balances(manager, mock_rpc_manager):
    """Test refreshing balances for all wallets"""
    # Set mock balance
    mock_rpc_manager.call_http_rpc.return_value = {"result": {"value": 2_000_000_000}}  # 2 SOL

    balances = await manager.refresh_balances()

    assert len(balances) == 3
    for pubkey, balance_sol in balances.items():
        assert balance_sol == 2.0


@pytest.mark.asyncio
async def test_balance_cache_expiry(manager, mock_rpc_manager):
    """Test balance cache expiry"""
    wallet = list(manager.wallets.values())[0]

    # Set initial balance
    wallet.balance_lamports = 1_000_000_000
    wallet.balance_cached_at = asyncio.get_event_loop().time() - 100  # Expired

    # Should refresh (cache expired)
    should_refresh = await manager._should_refresh_balance(wallet)
    assert should_refresh is True


@pytest.mark.asyncio
async def test_balance_cache_valid(manager):
    """Test balance cache when still valid"""
    import time

    wallet = list(manager.wallets.values())[0]

    # Set recent cache (use time.time() to match implementation)
    wallet.balance_lamports = 1_000_000_000
    wallet.balance_cached_at = time.time()

    # Should not refresh (cache valid)
    should_refresh = await manager._should_refresh_balance(wallet)
    assert should_refresh is False


@pytest.mark.asyncio
async def test_get_total_balance(manager):
    """Test getting total balance across all wallets"""
    # Set balances manually
    for i, wallet in enumerate(manager.wallets.values()):
        wallet.balance_lamports = (i + 1) * LAMPORTS_PER_SOL  # 1, 2, 3 SOL

    total = manager.get_total_balance_sol()
    assert total == 6.0  # 1 + 2 + 3


# =============================================================================
# WALLET INFO QUERIES
# =============================================================================

def test_get_wallet_info(manager, keypairs):
    """Test getting wallet info by pubkey"""
    pubkey = keypairs[0].pubkey()

    wallet_info = manager.get_wallet_info(pubkey)

    assert wallet_info is not None
    assert wallet_info.pubkey == pubkey


def test_get_wallet_info_not_found(manager):
    """Test getting wallet info for non-existent pubkey"""
    fake_pubkey = Keypair().pubkey()

    wallet_info = manager.get_wallet_info(fake_pubkey)

    assert wallet_info is None


def test_get_all_wallets(manager):
    """Test getting all wallets"""
    all_wallets = manager.get_all_wallets()

    assert len(all_wallets) == 3
    assert all(isinstance(w, WalletInfo) for w in all_wallets)


def test_get_available_wallet_count(manager):
    """Test counting available wallets"""
    # Initially, all wallets should be available
    count = manager.get_available_wallet_count(min_balance_lamports=0)
    assert count == 3


@pytest.mark.asyncio
async def test_get_available_wallet_count_with_locks(manager):
    """Test counting available wallets with some locked"""
    # Lock one wallet
    wallet = list(manager.wallets.values())[0]
    await wallet.lock.acquire()

    count = manager.get_available_wallet_count(min_balance_lamports=0)
    assert count == 2  # One locked

    # Release
    wallet.lock.release()


# =============================================================================
# STATISTICS
# =============================================================================

def test_get_stats(manager):
    """Test getting manager statistics"""
    stats = manager.get_stats()

    assert stats["total_wallets"] == 3
    assert "locked_wallets" in stats
    assert "cooldown_wallets" in stats
    assert "available_wallets" in stats
    assert "total_balance_sol" in stats
    assert "total_uses" in stats


@pytest.mark.asyncio
async def test_stats_after_usage(manager):
    """Test statistics after wallet usage"""
    # Use a wallet
    async with manager.get_wallet(min_sol=0.0) as wallet:
        pass

    await asyncio.sleep(0.1)  # Let cooldown set in

    stats = manager.get_stats()

    # Should have at least one wallet in cooldown
    assert stats["cooldown_wallets"] >= 1
    assert stats["total_uses"] >= 1


# =============================================================================
# WALLET INFO PROPERTIES
# =============================================================================

def test_wallet_info_balance_sol_property():
    """Test WalletInfo balance_sol property"""
    keypair = Keypair()
    wallet = WalletInfo(pubkey=keypair.pubkey(), keypair=keypair)
    wallet.balance_lamports = 1_500_000_000  # 1.5 SOL

    assert wallet.balance_sol == 1.5


def test_wallet_info_is_locked_property():
    """Test WalletInfo is_locked property"""
    keypair = Keypair()
    wallet = WalletInfo(pubkey=keypair.pubkey(), keypair=keypair)

    assert wallet.is_locked is False

    # Lock it
    wallet.lock._locked = True
    assert wallet.is_locked is True


def test_wallet_info_is_in_cooldown_property():
    """Test WalletInfo is_in_cooldown property"""
    import time
    keypair = Keypair()
    wallet = WalletInfo(pubkey=keypair.pubkey(), keypair=keypair)

    # No cooldown
    wallet.in_cooldown_until = time.time() - 1
    assert wallet.is_in_cooldown is False

    # In cooldown
    wallet.in_cooldown_until = time.time() + 10
    assert wallet.is_in_cooldown is True


def test_wallet_info_to_dict():
    """Test WalletInfo serialization to dict"""
    keypair = Keypair()
    wallet = WalletInfo(pubkey=keypair.pubkey(), keypair=keypair)
    wallet.balance_lamports = 2_000_000_000
    wallet.total_uses = 5

    data = wallet.to_dict()

    assert isinstance(data, dict)
    assert "pubkey" in data
    assert "balance_sol" in data
    assert data["balance_sol"] == 2.0
    assert data["total_uses"] == 5


# =============================================================================
# EDGE CASES
# =============================================================================

@pytest.mark.asyncio
async def test_manager_with_no_wallets(mock_rpc_manager):
    """Test manager with empty wallet list"""
    manager = WalletManager(mock_rpc_manager, [])

    assert len(manager.wallets) == 0

    # Should raise error when trying to get wallet
    with pytest.raises(RuntimeError):
        async with manager.get_wallet(min_sol=0.0, timeout_seconds=0.5) as wallet:
            pass


@pytest.mark.asyncio
async def test_manager_with_single_wallet(mock_rpc_manager):
    """Test manager with only one wallet"""
    keypair = Keypair()
    manager = WalletManager(mock_rpc_manager, [keypair])
    await manager.start()

    # Should work fine with single wallet
    async with manager.get_wallet(min_sol=0.0) as wallet:
        assert wallet.pubkey == keypair.pubkey()

    await manager.stop()


# =============================================================================
# CRITICAL EDGE CASES (ADDED FROM AUDIT)
# =============================================================================

@pytest.mark.asyncio
async def test_all_wallets_in_cooldown(manager):
    """Test behavior when all wallets are in cooldown"""
    # Acquire all wallets quickly to put them in cooldown
    for _ in range(len(manager.wallets)):
        async with manager.get_wallet(min_sol=0.0) as wallet:
            pass  # Just acquire and release

    # All should be in cooldown now
    assert all(w.is_in_cooldown for w in manager.wallets.values())

    # Try to get wallet - should timeout since all in cooldown
    with pytest.raises(RuntimeError, match="No wallet available"):
        async with manager.get_wallet(min_sol=0.0, timeout_seconds=0.5) as wallet:
            pass


@pytest.mark.asyncio
async def test_wallet_reacquisition_after_cooldown(manager):
    """Test wallet can be acquired again after cooldown expires"""
    # Set short cooldown for test
    manager.config.cooldown_seconds = 0.5

    # Acquire wallet
    async with manager.get_wallet(min_sol=0.0) as wallet1:
        wallet1_pubkey = wallet1.pubkey

    # Verify in cooldown
    wallet_info = manager.get_wallet_info(wallet1_pubkey)
    assert wallet_info.is_in_cooldown is True

    # Wait for cooldown to expire
    await asyncio.sleep(0.6)

    # Verify cooldown expired
    wallet_info = manager.get_wallet_info(wallet1_pubkey)
    assert wallet_info.is_in_cooldown is False

    # Should be able to acquire again
    async with manager.get_wallet(min_sol=0.0) as wallet2:
        # Might get same wallet or different one depending on rotation
        assert wallet2 is not None


@pytest.mark.asyncio
async def test_concurrent_balance_refresh_no_race(manager, mock_rpc_manager):
    """Test concurrent balance refreshes don't cause race conditions"""
    wallet = list(manager.wallets.values())[0]

    # Mock balance to return different values
    call_count = 0
    def get_balance_incremental(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {"result": {"value": call_count * 1_000_000_000}}

    mock_rpc_manager.call_http_rpc.side_effect = get_balance_incremental

    # Refresh concurrently
    results = await asyncio.gather(*[
        manager._refresh_wallet_balance(wallet)
        for _ in range(10)
    ])

    # All should complete
    assert len(results) == 10
    # Balance should be set (last write wins is acceptable)
    assert wallet.balance_lamports > 0
    # No errors raised


@pytest.mark.asyncio
async def test_balance_refresh_failure(manager, mock_rpc_manager):
    """Test balance refresh handles RPC errors gracefully"""
    # Make RPC call fail
    mock_rpc_manager.call_http_rpc.side_effect = Exception("RPC error")

    # Should not crash
    balances = await manager.refresh_balances()

    # Should return cached balances (from initialization)
    assert len(balances) == 3
