"""
Integration tests for Phase 1 - RPC Infrastructure

These tests connect to real Solana RPC endpoints and verify:
- WebSocket connectivity
- Real-time subscriptions
- Failover behavior
- Health monitoring

Requires network access and valid RPC endpoints.

Run with: pytest tests/integration/ -v
Skip with: pytest tests/ --ignore=tests/integration/
"""

import pytest
import asyncio
from core.rpc_manager import RPCManager, ConnectionStatus

# Use fixtures from conftest.py
# NOTE: These tests are replaced by newer test_devnet_*.py tests
# Skipping for now to focus on new test suite
pytestmark = [pytest.mark.integration, pytest.mark.skip(reason="Replaced by test_devnet_*.py tests")]


@pytest.mark.rpc
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_rpc_manager_connects_to_devnet(devnet_rpc_manager):
    """Test that RPC manager can connect to Solana devnet"""
    rpc_manager = devnet_rpc_manager
    # Fixture already starts the manager

    # Wait a moment for connections
    await asyncio.sleep(1)

    # Verify at least one endpoint connected
    connected_count = sum(
        1 for conn in rpc_manager.connections.values()
        if conn.status == ConnectionStatus.CONNECTED
    )

    assert connected_count > 0, "No RPC endpoints connected"

    # Get health stats
    health = rpc_manager.get_health_stats()
    assert len(health) > 0

    # Verify at least one healthy endpoint
    healthy_count = sum(1 for stats in health.values() if stats.is_healthy)
    assert healthy_count > 0, "No healthy endpoints"


@pytest.mark.rpc
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_slot_subscription(devnet_rpc_manager):
    """Test real-time slot subscription"""
    rpc_manager = devnet_rpc_manager

    # Subscribe to slot updates
    slot_count = 0
    slots_seen = []

    async for notification in rpc_manager.subscribe("slotSubscribe", []):
        slot_info = notification.get("result", {})
        slot = slot_info.get("slot")

        if slot:
            slots_seen.append(slot)
            slot_count += 1

        # Exit after 5 slot updates (~2 seconds)
        if slot_count >= 5:
            break

    # Verify we got slot updates
    assert slot_count == 5, f"Expected 5 slots, got {slot_count}"

    # Verify slots are incrementing
    for i in range(1, len(slots_seen)):
        assert slots_seen[i] >= slots_seen[i-1], "Slots should be increasing"


@pytest.mark.rpc
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_healthy_connection_selection(devnet_rpc_manager, devnet_rpc_config):
    """Test that healthy connection is selected correctly"""
    rpc_manager = devnet_rpc_manager

    # Get healthy connection
    conn = await rpc_manager.get_healthy_connection()

    assert conn is not None, "Should get a healthy connection"
    assert conn.status == ConnectionStatus.CONNECTED
    assert conn.consecutive_failures < devnet_rpc_config.failover_threshold_errors


@pytest.mark.rpc
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_health_monitoring(devnet_rpc_manager):
    """Test health monitoring loop"""
    rpc_manager = devnet_rpc_manager

    # Wait for a few health check cycles
    await asyncio.sleep(15)  # ~1.5 health check intervals

    # Verify health stats collected
    health = rpc_manager.get_health_stats()

    for label, stats in health.items():
        # If connected, should have health data
        conn = rpc_manager.connections[label]
        if conn.status == ConnectionStatus.CONNECTED:
            assert stats.is_healthy or stats.consecutive_failures > 0
            # Healthy endpoints should have low error rate
            if stats.is_healthy:
                assert stats.error_rate < 0.5, f"{label} has high error rate: {stats.error_rate}"


@pytest.mark.rpc
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_graceful_shutdown(devnet_rpc_config):
    """Test that manager shuts down cleanly"""
    # Create fresh manager for this test
    rpc_manager = RPCManager(devnet_rpc_config)

    await rpc_manager.start()
    await asyncio.sleep(1)

    # Stop should complete without hanging
    await rpc_manager.stop()

    # Verify all connections closed
    for conn in rpc_manager.connections.values():
        assert conn.websocket is None or conn.websocket.closed
        assert conn.status == ConnectionStatus.DISCONNECTED


@pytest.mark.rpc
@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_multiple_start_stop_cycles(devnet_rpc_config):
    """Test starting and stopping manager multiple times"""
    rpc_manager = RPCManager(devnet_rpc_config)

    # Do 3 start/stop cycles
    for cycle in range(3):
        await rpc_manager.start()
        await asyncio.sleep(2)

        # Verify connected
        connected_count = sum(
            1 for conn in rpc_manager.connections.values()
            if conn.status == ConnectionStatus.CONNECTED
        )
        assert connected_count > 0, f"Cycle {cycle+1}: No connections"

        await rpc_manager.stop()
        await asyncio.sleep(1)


@pytest.mark.rpc
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_all_configured_endpoints(devnet_rpc_manager, devnet_rpc_config):
    """Test that all configured endpoints are attempted"""
    rpc_manager = devnet_rpc_manager

    await asyncio.sleep(2)

    # Verify all endpoints have a connection object
    assert len(rpc_manager.connections) == len(devnet_rpc_config.endpoints)

    # Verify each endpoint was attempted
    for endpoint in devnet_rpc_config.endpoints:
        assert endpoint.label in rpc_manager.connections
        conn = rpc_manager.connections[endpoint.label]

        # Status should not be DISCONNECTED (at least attempted to connect)
        assert conn.status != ConnectionStatus.DISCONNECTED or conn.reconnect_attempt > 0


# Helper function for manual testing
@pytest.mark.rpc
@pytest.mark.slow
@pytest.mark.skip(reason="Manual test only - requires observation")
@pytest.mark.asyncio
async def test_extended_operation(devnet_rpc_manager):
    """
    Extended test for manual observation

    Run this test manually to observe:
    - Sustained connection health
    - Memory usage over time
    - Reconnection behavior

    Run with: pytest tests/integration/test_phase1_integration.py::test_extended_operation -v -s
    """
    rpc_manager = devnet_rpc_manager

    print("\n=== Extended Operation Test (60 seconds) ===")
    print("Monitoring RPC connections...\n")

    for i in range(12):  # 12 x 5s = 60s
        await asyncio.sleep(5)

        health = rpc_manager.get_health_stats()

        print(f"[{i*5}s] Health Status:")
        for label, stats in health.items():
            status = "✓ HEALTHY" if stats.is_healthy else "✗ UNHEALTHY"
            print(f"  {label:30} {status:15} failures={stats.consecutive_failures}")

        print()

    print("=== Test Complete ===")
