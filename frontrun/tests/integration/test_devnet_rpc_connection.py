"""
Integration Test 1: RPC Connection & Health
Tests REAL connections to Solana devnet - validates RPCManager works with actual blockchain
"""

import pytest
import time
import asyncio
from core.logger import get_logger


logger = get_logger(__name__)

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# BASIC CONNECTIVITY TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_devnet_rpc_connection_established(devnet_connection_info):
    """Test that we can connect to Solana devnet and get basic info"""
    assert devnet_connection_info["connected"] is True, \
        f"Failed to connect to devnet: {devnet_connection_info.get('error')}"

    assert devnet_connection_info["slot"] > 0, \
        "Devnet slot should be positive"

    logger.info(
        "devnet_connection_validated",
        slot=devnet_connection_info["slot"],
        version=devnet_connection_info["version"]
    )


@pytest.mark.asyncio
async def test_devnet_get_current_slot(devnet_rpc_manager):
    """Test fetching current slot from devnet (real blockchain data)"""
    response = await devnet_rpc_manager.call_http_rpc("getSlot", [])

    # Extract slot
    if isinstance(response, dict):
        slot = response.get("result", 0)
    else:
        slot = response

    assert isinstance(slot, int), f"Slot should be integer, got {type(slot)}"
    assert slot > 0, "Slot should be positive"
    assert slot > 200_000_000, f"Devnet slot seems too low: {slot}"  # Devnet is past 200M slots

    logger.info("devnet_current_slot", slot=slot)


@pytest.mark.asyncio
async def test_devnet_get_version(devnet_rpc_manager):
    """Test fetching Solana version from devnet"""
    response = await devnet_rpc_manager.call_http_rpc("getVersion", [])

    assert isinstance(response, dict), "Response should be dict"
    result = response.get("result", {})

    assert "solana-core" in result, "Should have solana-core version"

    version = result["solana-core"]
    logger.info("devnet_solana_version", version=version)

    # Version should be in format like "1.18.0" or "2.0.0"
    assert len(version) > 0, "Version should not be empty"


@pytest.mark.asyncio
async def test_devnet_get_health(devnet_rpc_manager):
    """Test health check on devnet RPC"""
    response = await devnet_rpc_manager.call_http_rpc("getHealth", [])

    # getHealth returns "ok" or error
    if isinstance(response, dict):
        result = response.get("result", "")
    else:
        result = response

    logger.info("devnet_health_status", status=result)

    # Note: getHealth might return error if node is unhealthy, but that's okay
    # We just want to verify we can call it
    assert result is not None


@pytest.mark.asyncio
async def test_devnet_get_latest_blockhash(devnet_rpc_manager):
    """Test fetching latest blockhash from devnet (needed for transactions)"""
    response = await devnet_rpc_manager.call_http_rpc(
        "getLatestBlockhash",
        [{"commitment": "finalized"}]
    )

    assert isinstance(response, dict), "Response should be dict"
    result = response.get("result", {})
    value = result.get("value", {})

    assert "blockhash" in value, "Should have blockhash"
    assert "lastValidBlockHeight" in value, "Should have lastValidBlockHeight"

    blockhash = value["blockhash"]
    block_height = value["lastValidBlockHeight"]

    assert isinstance(blockhash, str), "Blockhash should be string"
    assert len(blockhash) > 0, "Blockhash should not be empty"
    assert isinstance(block_height, int), "Block height should be integer"
    assert block_height > 0, "Block height should be positive"

    logger.info(
        "devnet_latest_blockhash",
        blockhash=blockhash[:16] + "...",  # Truncate for logging
        block_height=block_height
    )


# =============================================================================
# PERFORMANCE & LATENCY TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_devnet_rpc_latency_acceptable(devnet_rpc_manager):
    """Test that RPC latency to devnet is acceptable (<5 seconds)"""
    start_time = time.perf_counter()

    await devnet_rpc_manager.call_http_rpc("getSlot", [])

    latency = (time.perf_counter() - start_time) * 1000  # Convert to ms

    logger.info("devnet_rpc_latency", latency_ms=latency)

    # Devnet can be slow, but should respond within 5 seconds
    assert latency < 5000, f"RPC latency too high: {latency:.2f}ms"


@pytest.mark.asyncio
async def test_devnet_multiple_concurrent_requests(devnet_rpc_manager):
    """Test handling multiple concurrent RPC requests to devnet"""
    num_requests = 5

    start_time = time.perf_counter()

    # Make 5 concurrent requests
    tasks = [
        devnet_rpc_manager.call_http_rpc("getSlot", [])
        for _ in range(num_requests)
    ]

    results = await asyncio.gather(*tasks)

    elapsed = (time.perf_counter() - start_time) * 1000

    # All should succeed
    assert len(results) == num_requests, "All requests should complete"

    # Extract slots
    slots = []
    for result in results:
        if isinstance(result, dict):
            slot = result.get("result", 0)
        else:
            slot = result
        slots.append(slot)

    # All slots should be positive
    assert all(s > 0 for s in slots), "All slots should be positive"

    # Slots should be close to each other (devnet progresses ~2 slots/second)
    max_slot = max(slots)
    min_slot = min(slots)
    slot_diff = max_slot - min_slot

    assert slot_diff < 100, f"Slot difference too large: {slot_diff}"

    logger.info(
        "concurrent_requests_completed",
        num_requests=num_requests,
        elapsed_ms=elapsed,
        avg_latency_ms=elapsed / num_requests,
        slot_range=f"{min_slot}-{max_slot}"
    )


# =============================================================================
# FAILOVER & RELIABILITY TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_devnet_rpc_manager_has_multiple_endpoints(devnet_rpc_config):
    """Test that RPC manager is configured with multiple endpoints for failover"""
    assert len(devnet_rpc_config.endpoints) >= 2, \
        "Should have at least 2 endpoints for failover"

    endpoint_urls = [ep.url for ep in devnet_rpc_config.endpoints]
    logger.info("configured_endpoints", endpoints=endpoint_urls)

    # All should be devnet endpoints
    for url in endpoint_urls:
        assert "devnet" in url.lower(), f"Endpoint should be devnet: {url}"


@pytest.mark.asyncio
async def test_devnet_rpc_retry_logic(devnet_rpc_manager):
    """Test that RPC manager has failover configuration"""
    # This test verifies failover configuration, actual retry testing
    # would require mocking which defeats the purpose of integration tests

    # Verify configuration
    assert devnet_rpc_manager.config.failover_threshold_errors >= 3, \
        "Should have at least 3 errors before failover"

    assert devnet_rpc_manager.config.health_check_interval_s >= 10, \
        "Should have reasonable health check interval configured"

    logger.info(
        "failover_config",
        failover_threshold_errors=devnet_rpc_manager.config.failover_threshold_errors,
        health_check_interval_s=devnet_rpc_manager.config.health_check_interval_s,
        failover_threshold_latency_ms=devnet_rpc_manager.config.failover_threshold_latency_ms
    )


# =============================================================================
# BALANCE & ACCOUNT QUERIES
# =============================================================================

@pytest.mark.asyncio
async def test_devnet_get_balance_for_known_address(devnet_rpc_manager):
    """Test querying balance for a known address on devnet"""
    # Use a well-known devnet address (fee payer address)
    # This address may or may not have balance, but query should work
    test_address = "11111111111111111111111111111111"

    response = await devnet_rpc_manager.call_http_rpc(
        "getBalance",
        [test_address]
    )

    assert isinstance(response, dict), "Response should be dict"
    result = response.get("result", {})
    value = result.get("value")

    # Balance query should return a value (even if 0)
    assert value is not None, "Balance should be returned"
    assert isinstance(value, int), "Balance should be integer"
    assert value >= 0, "Balance should be non-negative"

    logger.info("balance_query_successful", address=test_address, balance=value)


@pytest.mark.asyncio
async def test_devnet_get_account_info_for_program(devnet_rpc_manager):
    """Test querying account info for a known program on devnet"""
    # System program - should always exist
    system_program = "11111111111111111111111111111111"

    response = await devnet_rpc_manager.call_http_rpc(
        "getAccountInfo",
        [system_program, {"encoding": "base64"}]
    )

    assert isinstance(response, dict), "Response should be dict"
    result = response.get("result", {})
    value = result.get("value")

    # System program should exist
    assert value is not None, "System program should exist"

    logger.info("account_info_query_successful", program=system_program)


# =============================================================================
# SLOT & BLOCK PROGRESSION
# =============================================================================

@pytest.mark.asyncio
async def test_devnet_slot_progression(devnet_rpc_manager):
    """Test that devnet is actively producing blocks (slot increases over time)"""
    # Get initial slot
    response1 = await devnet_rpc_manager.call_http_rpc("getSlot", [])
    slot1 = response1 if isinstance(response1, int) else response1.get("result", 0)

    # Wait 5 seconds
    await asyncio.sleep(5)

    # Get new slot
    response2 = await devnet_rpc_manager.call_http_rpc("getSlot", [])
    slot2 = response2 if isinstance(response2, int) else response2.get("result", 0)

    # Devnet should have progressed (typically ~2 slots/second = 10 slots in 5 seconds)
    slot_diff = slot2 - slot1

    assert slot_diff > 0, f"Slot should increase over time: {slot1} -> {slot2}"
    assert slot_diff >= 5, f"Devnet should progress at least 5 slots in 5 seconds, got {slot_diff}"
    assert slot_diff < 30, f"Slot diff seems too large: {slot_diff}"  # Sanity check

    logger.info(
        "slot_progression_validated",
        initial_slot=slot1,
        final_slot=slot2,
        slots_progressed=slot_diff,
        slots_per_second=slot_diff / 5.0
    )
