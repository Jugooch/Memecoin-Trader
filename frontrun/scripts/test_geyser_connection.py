"""
Quick test script to verify Helius LaserStream connection
Run this BEFORE starting Phase 1 to ensure everything is configured correctly
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import ConfigurationManager
from core.logger import setup_logging, get_logger
from services.mempool_monitor import (
    LaserStreamMonitor,
    LaserStreamConfig,
    PendingTransaction
)


logger = get_logger(__name__)


async def test_connection(config_path: str = "config/config.yml", duration_s: int = 30):
    """
    Test Helius LaserStream connection

    Args:
        config_path: Path to config.yml
        duration_s: How long to test (default: 30 seconds)
    """
    logger.info("=== HELIUS LASERSTREAM CONNECTION TEST ===")
    logger.info(f"Testing connection for {duration_s} seconds...")

    # Load configuration
    config_manager = ConfigurationManager(config_path)
    bot_config = config_manager.load_config()

    # Get Helius endpoint
    helius_endpoint = bot_config.rpc_config.endpoints[0]

    if not helius_endpoint.laserstream_url or not helius_endpoint.laserstream_api_key:
        logger.error(
            "missing_helius_config",
            message="LaserStream URL and API key required in config.yml"
        )
        return False

    # Extract endpoint (remove https://)
    endpoint = helius_endpoint.laserstream_url.replace("https://", "").replace("http://", "")

    logger.info(
        "helius_config_found",
        endpoint=endpoint,
        label=helius_endpoint.label
    )

    # Create LaserStream monitor
    laserstream_config = LaserStreamConfig(
        endpoint=endpoint,
        api_key=helius_endpoint.laserstream_api_key,
        commitment="processed"  # Fastest for testing
    )
    monitor = LaserStreamMonitor(laserstream_config)

    # Track transactions seen
    transactions_seen = []

    async def on_transaction(tx: PendingTransaction):
        """Callback for each transaction"""
        transactions_seen.append(tx)
        logger.info(
            "transaction_received",
            signature=tx.signature[:16] + "...",
            buy_amount_sol=tx.buy_amount_sol,
            fee=tx.fee,
            program_id=str(tx.program_id) if tx.program_id else None
        )

    # Start monitoring
    monitoring_task = asyncio.create_task(
        monitor.start_monitoring(on_transaction)
    )

    # Wait for test duration
    try:
        logger.info("monitoring_started", message="Waiting for transactions...")
        await asyncio.sleep(duration_s)

    except KeyboardInterrupt:
        logger.info("test_interrupted_by_user")

    finally:
        # Stop monitoring
        await monitor.stop_monitoring()
        monitoring_task.cancel()

        # Get stats
        stats = monitor.get_mempool_stats()

        # Print results
        logger.info("=== TEST RESULTS ===")
        logger.info(
            "connection_stats",
            total_transactions=stats.transactions_seen,
            pump_fun_txs=stats.pump_fun_txs_seen,
            reconnections=stats.reconnections,
            errors=stats.errors
        )

        # Evaluate results
        if stats.errors > 0:
            logger.warning(
                "test_completed_with_errors",
                errors=stats.errors,
                recommendation="Check logs and Helius dashboard"
            )
            return False

        if stats.transactions_seen == 0:
            logger.warning(
                "no_transactions_received",
                recommendation="Verify Pump.fun is active or increase test duration"
            )
            return False

        logger.info(
            "test_passed",
            message=f"✓ Successfully received {stats.transactions_seen} transactions",
            pump_fun_txs=stats.pump_fun_txs_seen,
            recommendation="Ready to start Phase 1 monitoring"
        )
        return True


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Helius LaserStream Connection")
    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="Path to config.yml"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Test duration in seconds (default: 30)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO", format="console")

    # Run test
    success = await test_connection(
        config_path=args.config,
        duration_s=args.duration
    )

    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error("test_failed", error=str(e), error_type=type(e).__name__)
        sys.exit(1)
