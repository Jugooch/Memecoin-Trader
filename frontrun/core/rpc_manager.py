"""
Multi-RPC Connection Manager for Solana
Handles WebSocket connections with automatic failover and health monitoring
"""

import asyncio
import json
import time
import aiohttp
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import websockets

from core.config import RPCEndpoint, RPCConfig
from core.logger import get_logger
from core.metrics import get_metrics, LatencyTimer


logger = get_logger(__name__)
metrics = get_metrics()


class ConnectionStatus(Enum):
    """WebSocket connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class RPCHealthStats:
    """Health statistics for an RPC endpoint"""
    endpoint_url: str
    label: str
    is_healthy: bool
    p50_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    last_successful_call: Optional[float] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    total_errors: int = 0


@dataclass
class RPCConnection:
    """Represents a single RPC connection"""
    endpoint: RPCEndpoint
    websocket: Optional[Any] = None  # websockets.WebSocketClientProtocol
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    last_ping_time: float = 0
    consecutive_failures: int = 0
    reconnect_attempt: int = 0


class RPCManager:
    """
    Manages multiple RPC connections with automatic failover

    Features:
    - Multiple WebSocket connections to different RPCs
    - Automatic reconnection with exponential backoff
    - Health monitoring and failover
    - Connection pooling and load balancing
    """

    def __init__(self, config: RPCConfig):
        """
        Initialize RPC manager

        Args:
            config: RPC configuration
        """
        self.config = config
        self.connections: Dict[str, RPCConnection] = {}
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._subscription_id_counter = 0
        self._active_subscriptions: Dict[int, str] = {}  # sub_id -> endpoint_label

        # HTTP client session for HTTP RPC calls
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Initialize connections
        for endpoint in config.endpoints:
            self.connections[endpoint.label] = RPCConnection(endpoint=endpoint)

        logger.info(
            "rpc_manager_initialized",
            endpoint_count=len(self.connections),
            endpoints=[ep.label for ep in config.endpoints]
        )

    async def start(self) -> None:
        """Start the RPC manager and connect to all endpoints"""
        if self._running:
            logger.warning("rpc_manager_already_running")
            return

        self._running = True
        logger.info("rpc_manager_starting")

        # Create HTTP session for HTTP RPC calls (no default timeout, set per request)
        self._http_session = aiohttp.ClientSession()

        # Connect to all endpoints
        connect_tasks = [
            self._connect_endpoint(label)
            for label in self.connections.keys()
        ]
        await asyncio.gather(*connect_tasks, return_exceptions=True)

        # Start health check loop (DISABLED - only needed for WebSocket subscriptions)
        # self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("rpc_manager_started")

    async def stop(self) -> None:
        """Stop the RPC manager and close all connections"""
        if not self._running:
            return

        logger.info("rpc_manager_stopping")
        self._running = False

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close HTTP session
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        # Close all connections
        close_tasks = [
            self._disconnect_endpoint(label)
            for label in self.connections.keys()
        ]
        await asyncio.gather(*close_tasks, return_exceptions=True)

        logger.info("rpc_manager_stopped")

    async def get_healthy_connection(self) -> Optional[RPCConnection]:
        """
        Get a healthy WebSocket connection with lowest latency

        Returns:
            RPCConnection if available, None otherwise
        """
        # Filter for connected and healthy connections
        healthy_conns = [
            conn for conn in self.connections.values()
            if conn.status == ConnectionStatus.CONNECTED and
            conn.consecutive_failures < self.config.failover_threshold_errors
        ]

        if not healthy_conns:
            logger.warning("no_healthy_connections_available")
            return None

        # Sort by priority (endpoint.priority is already sorted in config)
        # In case of tie, prefer lower consecutive failures
        healthy_conns.sort(key=lambda c: (c.endpoint.priority, c.consecutive_failures))

        return healthy_conns[0]

    async def call_rpc(
        self,
        method: str,
        params: List[Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Make an RPC call with automatic failover

        Args:
            method: JSON-RPC method name
            params: Method parameters
            timeout: Request timeout in seconds

        Returns:
            RPC response dict

        Raises:
            Exception: If all endpoints fail
        """
        last_error = None

        # Try each healthy endpoint
        for conn in sorted(self.connections.values(), key=lambda c: c.endpoint.priority):
            if conn.status != ConnectionStatus.CONNECTED:
                continue

            try:
                with LatencyTimer(metrics, "rpc_call", {"endpoint": conn.endpoint.label, "method": method}):
                    response = await self._send_rpc_request(
                        conn,
                        method,
                        params,
                        timeout
                    )

                # Success - reset failure counter
                conn.consecutive_failures = 0
                metrics.increment_counter("rpc_success", labels={"endpoint": conn.endpoint.label})

                return response

            except Exception as e:
                logger.warning(
                    "rpc_call_failed",
                    endpoint=conn.endpoint.label,
                    method=method,
                    error=str(e)
                )
                conn.consecutive_failures += 1
                metrics.increment_counter("rpc_errors", labels={"endpoint": conn.endpoint.label})

                last_error = e

                # Check if we should failover this endpoint
                if conn.consecutive_failures >= self.config.failover_threshold_errors:
                    logger.error(
                        "rpc_endpoint_failing_over",
                        endpoint=conn.endpoint.label,
                        failures=conn.consecutive_failures
                    )
                    asyncio.create_task(self._reconnect_endpoint(conn.endpoint.label))

        # All endpoints failed
        raise Exception(f"All RPC endpoints failed. Last error: {last_error}")

    async def call_http_rpc(
        self,
        method: str,
        params: List[Any],
        timeout: float = 10.0
    ) -> Dict[str, Any]:
        """
        Make an HTTP RPC call with automatic failover

        Args:
            method: JSON-RPC method name
            params: Method parameters
            timeout: Request timeout in seconds

        Returns:
            RPC response dict

        Raises:
            Exception: If all endpoints fail
        """
        if not self._http_session:
            raise Exception("HTTP session not initialized. Call start() first.")

        last_error = None

        # Try each endpoint in priority order
        for conn in sorted(self.connections.values(), key=lambda c: c.endpoint.priority):
            try:
                with LatencyTimer(metrics, "http_rpc_call", {"endpoint": conn.endpoint.label, "method": method}):
                    # Build JSON-RPC request
                    request_id = int(time.time() * 1000000)
                    payload = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": method,
                        "params": params
                    }

                    # Send HTTP POST request with timeout using asyncio.wait_for
                    async def _make_request():
                        async with self._http_session.post(
                            conn.endpoint.url,
                            json=payload
                        ) as response:
                            return await response.json()

                    result = await asyncio.wait_for(_make_request(), timeout=timeout)

                    # Check for RPC errors
                    if "error" in result:
                        error_msg = result["error"].get("message", str(result["error"]))
                        raise Exception(f"RPC error: {error_msg}")

                    # Success - reset failure counter
                    conn.consecutive_failures = 0
                    metrics.increment_counter("http_rpc_success", labels={"endpoint": conn.endpoint.label})

                    return result

            except Exception as e:
                logger.warning(
                    "http_rpc_call_failed",
                    endpoint=conn.endpoint.label,
                    method=method,
                    error=str(e)
                )
                conn.consecutive_failures += 1
                metrics.increment_counter("http_rpc_errors", labels={"endpoint": conn.endpoint.label})

                last_error = e

                # Check if we should mark endpoint unhealthy
                if conn.consecutive_failures >= self.config.failover_threshold_errors:
                    logger.error(
                        "http_rpc_endpoint_failing_over",
                        endpoint=conn.endpoint.label,
                        failures=conn.consecutive_failures
                    )

        # All endpoints failed
        raise Exception(f"All HTTP RPC endpoints failed. Last error: {last_error}")

    async def subscribe(
        self,
        method: str,
        params: List[Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Subscribe to RPC notifications

        Args:
            method: Subscription method (e.g., "slotSubscribe")
            params: Subscription parameters

        Yields:
            Notification messages
        """
        conn = await self.get_healthy_connection()
        if not conn or not conn.websocket:
            raise Exception("No healthy connection available for subscription")

        # Send subscription request
        sub_id = self._subscription_id_counter
        self._subscription_id_counter += 1

        request = {
            "jsonrpc": "2.0",
            "id": sub_id,
            "method": method,
            "params": params
        }

        await conn.websocket.send(json.dumps(request))

        # Wait for subscription confirmation
        response = await asyncio.wait_for(conn.websocket.recv(), timeout=10.0)
        response_data = json.loads(response)

        if "error" in response_data:
            raise Exception(f"Subscription failed: {response_data['error']}")

        subscription_id = response_data.get("result")
        self._active_subscriptions[subscription_id] = conn.endpoint.label

        logger.info(
            "subscription_created",
            method=method,
            subscription_id=subscription_id,
            endpoint=conn.endpoint.label
        )

        # Yield notifications
        try:
            while self._running:
                message = await conn.websocket.recv()
                data = json.loads(message)

                # Check if this is a notification for our subscription
                if "method" in data and data.get("params", {}).get("subscription") == subscription_id:
                    yield data["params"]

        except websockets.exceptions.ConnectionClosed:
            logger.warning("subscription_connection_closed", subscription_id=subscription_id)
        finally:
            # Unsubscribe
            if subscription_id in self._active_subscriptions:
                del self._active_subscriptions[subscription_id]

    def get_health_stats(self) -> Dict[str, RPCHealthStats]:
        """
        Get health statistics for all endpoints

        Returns:
            Dictionary of endpoint label -> health stats
        """
        stats = {}

        for label, conn in self.connections.items():
            # Get latency stats from metrics
            histogram_stats = metrics.get_histogram_stats(f"rpc_call_{label}")

            stats[label] = RPCHealthStats(
                endpoint_url=conn.endpoint.url,
                label=label,
                is_healthy=(
                    conn.status == ConnectionStatus.CONNECTED and
                    conn.consecutive_failures < self.config.failover_threshold_errors
                ),
                p50_latency_ms=histogram_stats.p50 if histogram_stats else 0,
                p99_latency_ms=histogram_stats.p99 if histogram_stats else 0,
                error_rate=(
                    conn.consecutive_failures / max(histogram_stats.count, 1)
                    if histogram_stats else 0
                ),
                consecutive_failures=conn.consecutive_failures,
                total_requests=histogram_stats.count if histogram_stats else 0,
                total_errors=conn.consecutive_failures
            )

        return stats

    # ============================================================================
    # PRIVATE METHODS
    # ============================================================================

    async def _connect_endpoint(self, label: str) -> None:
        """Connect to a specific RPC endpoint"""
        conn = self.connections[label]
        conn.status = ConnectionStatus.CONNECTING

        try:
            logger.info("connecting_to_endpoint", endpoint=label, url=conn.endpoint.websocket_url)

            conn.websocket = await websockets.connect(
                conn.endpoint.websocket_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )

            conn.status = ConnectionStatus.CONNECTED
            conn.consecutive_failures = 0
            conn.reconnect_attempt = 0

            logger.info("endpoint_connected", endpoint=label)
            metrics.increment_counter("rpc_connections", labels={"endpoint": label})

        except Exception as e:
            conn.status = ConnectionStatus.FAILED
            logger.error("endpoint_connection_failed", endpoint=label, error=str(e))
            metrics.increment_counter("rpc_connection_failures", labels={"endpoint": label})

            # Schedule reconnection
            asyncio.create_task(self._reconnect_endpoint(label))

    async def _disconnect_endpoint(self, label: str) -> None:
        """Disconnect from a specific RPC endpoint"""
        conn = self.connections[label]

        if conn.websocket:
            try:
                await conn.websocket.close()
            except Exception as e:
                logger.warning("endpoint_disconnect_error", endpoint=label, error=str(e))

        conn.websocket = None
        conn.status = ConnectionStatus.DISCONNECTED

        logger.info("endpoint_disconnected", endpoint=label)

    async def _reconnect_endpoint(self, label: str) -> None:
        """Reconnect to an endpoint with exponential backoff"""
        conn = self.connections[label]

        if conn.status == ConnectionStatus.RECONNECTING:
            return  # Already reconnecting

        conn.status = ConnectionStatus.RECONNECTING

        while self._running and conn.status == ConnectionStatus.RECONNECTING:
            # Calculate backoff delay
            backoff_ms = min(
                self.config.reconnect_backoff_base_ms * (2 ** conn.reconnect_attempt),
                self.config.reconnect_backoff_max_ms
            )
            backoff_s = backoff_ms / 1000

            logger.info(
                "reconnecting_to_endpoint",
                endpoint=label,
                attempt=conn.reconnect_attempt + 1,
                backoff_seconds=backoff_s
            )

            await asyncio.sleep(backoff_s)

            # Attempt reconnection
            await self._disconnect_endpoint(label)
            await self._connect_endpoint(label)

            if conn.status == ConnectionStatus.CONNECTED:
                logger.info("endpoint_reconnected", endpoint=label)
                break

            conn.reconnect_attempt += 1

    async def _send_rpc_request(
        self,
        conn: RPCConnection,
        method: str,
        params: List[Any],
        timeout: float
    ) -> Dict[str, Any]:
        """Send RPC request to a specific connection"""
        if not conn.websocket:
            raise Exception(f"No websocket connection for {conn.endpoint.label}")

        request_id = int(time.time() * 1000000)  # Microsecond timestamp as ID
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }

        # Send request
        await conn.websocket.send(json.dumps(request))

        # Wait for response
        response = await asyncio.wait_for(conn.websocket.recv(), timeout=timeout)
        response_data = json.loads(response)

        # Check for errors
        if "error" in response_data:
            raise Exception(f"RPC error: {response_data['error']}")

        return response_data.get("result", {})

    async def _health_check_loop(self) -> None:
        """
        Periodic health check loop

        Performs actual RPC health checks via getHealth or getSlot methods
        to detect RPC issues beyond just WebSocket connectivity.
        """
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval_s)

                # Check each connection status
                for label, conn in self.connections.items():
                    try:
                        # First check: WebSocket connection status
                        websocket_open = True
                        if conn.websocket is not None:
                            # Check if websocket has a 'closed' attribute (not all implementations do)
                            websocket_open = not getattr(conn.websocket, 'closed', False)

                        websocket_connected = (
                            conn.status == ConnectionStatus.CONNECTED and
                            conn.websocket is not None and
                            websocket_open
                        )

                        # Second check: Actual RPC functionality
                        rpc_functional = False
                        if websocket_connected:
                            try:
                                # Try to get slot to verify RPC is functional
                                start_time = time.perf_counter()
                                result = await asyncio.wait_for(
                                    self._send_rpc_request(conn, "getSlot", [], timeout=5.0),
                                    timeout=5.0
                                )
                                latency_ms = (time.perf_counter() - start_time) * 1000

                                # Verify we got a valid slot number
                                slot = result if isinstance(result, int) else result.get("result", 0)
                                rpc_functional = slot > 0

                                # Record latency
                                metrics.record_latency("health_check", latency_ms, labels={"endpoint": label})

                                # Check if latency is too high
                                if latency_ms > self.config.failover_threshold_latency_ms:
                                    logger.warning(
                                        "health_check_high_latency",
                                        endpoint=label,
                                        latency_ms=latency_ms,
                                        threshold_ms=self.config.failover_threshold_latency_ms
                                    )
                                    rpc_functional = False

                            except Exception as e:
                                logger.debug("health_check_rpc_call_failed", endpoint=label, error=str(e))
                                rpc_functional = False

                        # Overall health determination
                        is_healthy = (
                            websocket_connected and
                            rpc_functional and
                            conn.consecutive_failures < self.config.failover_threshold_errors
                        )

                        if is_healthy:
                            metrics.set_gauge("rpc_healthy", 1.0, labels={"endpoint": label})
                            logger.debug("health_check_passed", endpoint=label)
                        else:
                            metrics.set_gauge("rpc_healthy", 0.0, labels={"endpoint": label})
                            logger.warning(
                                "health_check_failed",
                                endpoint=label,
                                status=conn.status.value,
                                websocket_connected=websocket_connected,
                                rpc_functional=rpc_functional
                            )

                            # Trigger reconnect if unhealthy
                            if conn.status == ConnectionStatus.CONNECTED and (
                                (conn.websocket and getattr(conn.websocket, 'closed', False)) or not rpc_functional
                            ):
                                asyncio.create_task(self._reconnect_endpoint(label))

                    except Exception as e:
                        logger.warning("health_check_error", endpoint=label, error=str(e))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_loop_error", error=str(e))


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

        # Start manager
        await rpc_manager.start()

        try:
            logger.info("rpc_manager_test_started")

            # Test WebSocket subscription (proper usage)
            logger.info("testing_slot_subscription")

            slot_count = 0
            async for notification in rpc_manager.subscribe("slotSubscribe", []):
                slot_info = notification.get("result", {})
                slot_count += 1
                logger.info(
                    "slot_update_received",
                    slot=slot_info.get("slot"),
                    parent=slot_info.get("parent"),
                    root=slot_info.get("root"),
                    count=slot_count
                )

                # Get health stats after a few updates
                if slot_count == 5:
                    health = rpc_manager.get_health_stats()
                    for label, stats in health.items():
                        logger.info(
                            "endpoint_health",
                            endpoint=label,
                            healthy=stats.is_healthy,
                            failures=stats.consecutive_failures,
                            p50_latency_ms=stats.p50_latency_ms,
                            p99_latency_ms=stats.p99_latency_ms
                        )

                # Exit after 10 updates (~4-5 seconds)
                if slot_count >= 10:
                    break

            logger.info("rpc_manager_test_completed")

        except KeyboardInterrupt:
            logger.info("test_interrupted_by_user")
        except Exception as e:
            logger.error("test_error", error=str(e), exc_info=True)
        finally:
            await rpc_manager.stop()

    asyncio.run(main())
