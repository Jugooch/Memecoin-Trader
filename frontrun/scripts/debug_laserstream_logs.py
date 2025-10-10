"""
Debug script to inspect raw LaserStream logs

This will connect to LaserStream and print ALL logs from Pump.fun transactions
so we can see if "Program data:" logs exist and what format they're in.

Usage:
    python scripts/debug_laserstream_logs.py --duration 60
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import ConfigurationManager
from core.logger import setup_logging, get_logger
from services.mempool_monitor import LaserStreamMonitor, LaserStreamConfig, PendingTransaction

logger = get_logger(__name__)


async def debug_logs(config_path: str = "config/config.yml", duration_s: int = 60):
    """
    Debug LaserStream logs by printing them all

    Args:
        config_path: Path to config.yml
        duration_s: How long to monitor (default: 60 seconds)
    """
    logger.info("=== LASERSTREAM LOG DEBUG ===")
    logger.info(f"Monitoring for {duration_s} seconds...")
    logger.info("This will print ALL logs from Pump.fun transactions")
    logger.info("")

    # Load configuration
    config_manager = ConfigurationManager(config_path)
    bot_config = config_manager.load_config()

    # Get Helius endpoint
    helius_endpoint = bot_config.rpc_config.endpoints[0]

    if not helius_endpoint.laserstream_url or not helius_endpoint.laserstream_api_key:
        logger.error("missing_helius_config")
        return False

    # Extract endpoint (remove https://)
    endpoint = helius_endpoint.laserstream_url.replace("https://", "").replace("http://", "")

    # Create LaserStream monitor
    laserstream_config = LaserStreamConfig(
        endpoint=endpoint,
        api_key=helius_endpoint.laserstream_api_key,
        commitment="processed"
    )

    # IMPORTANT: We'll directly hook into the internal parsing to see raw logs
    monitor = LaserStreamMonitor(laserstream_config)

    # Transaction counter
    tx_count = 0

    async def on_transaction(tx: PendingTransaction):
        """This will be called for successful parses only"""
        nonlocal tx_count
        tx_count += 1
        logger.info(
            "✅ Successfully parsed BUY transaction",
            count=tx_count,
            signature=tx.signature[:16] + "...",
            buy_amount_sol=tx.buy_amount_sol,
            mint=tx.mint[:16] + "..."
        )

    # Start monitoring
    monitoring_task = asyncio.create_task(
        monitor.start_monitoring(on_transaction)
    )

    # Wait for test duration
    try:
        logger.info("monitoring_started")
        logger.info("Waiting for transactions...")
        logger.info("Watch for 'no_trade_event_found' messages - they will show first_5_logs")
        logger.info("")
        await asyncio.sleep(duration_s)

    except KeyboardInterrupt:
        logger.info("interrupted_by_user")

    finally:
        # Stop monitoring
        await monitor.stop_monitoring()
        monitoring_task.cancel()

        # Get stats
        stats = monitor.get_mempool_stats()

        # Print results
        logger.info("")
        logger.info("=== DEBUG RESULTS ===")
        logger.info(
            "stats",
            total_transactions=stats.transactions_seen,
            pump_fun_txs=stats.pump_fun_txs_seen,
            successful_buys=tx_count,
            errors=stats.errors
        )

        logger.info("")
        logger.info("Check the logs above for:")
        logger.info("  1. 'no_trade_event_found' messages showing first_5_logs")
        logger.info("  2. Whether 'has_program_data' is ever True")
        logger.info("  3. The format of log messages (invocation vs data)")

        return True


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Debug LaserStream log format")
    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="Path to config.yml"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds (default: 60)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO", format="console")

    # Run test
    success = await debug_logs(
        config_path=args.config,
        duration_s=args.duration
    )

    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Debug interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error("debug_failed", error=str(e), error_type=type(e).__name__)
        sys.exit(1)
