"""
Test LaserStream with ALL transactions (no filter)
This verifies the connection works before narrowing to Pump.fun
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import ConfigurationManager
from core.logger import setup_logging, get_logger
from services.mempool_monitor import LaserStreamMonitor, LaserStreamConfig, PendingTransaction

logger = get_logger(__name__)


async def test_all_transactions(duration_s: int = 10):
    """Test with ALL transactions (unfiltered)"""

    logger.info("=== TESTING ALL TRANSACTIONS (UNFILTERED) ===")

    # Load config
    config_manager = ConfigurationManager("config/config.yml")
    bot_config = config_manager.load_config()

    helius_endpoint = bot_config.rpc_config.endpoints[0]
    endpoint = helius_endpoint.laserstream_url.replace("https://", "").replace("http://", "")

    # Create monitor with NO program filter (get everything)
    config = LaserStreamConfig(
        endpoint=endpoint,
        api_key=helius_endpoint.laserstream_api_key,
        program_filters=[],  # EMPTY = ALL transactions
        commitment="processed"
    )

    monitor = LaserStreamMonitor(config)

    tx_count = 0
    pump_fun_count = 0

    async def on_tx(tx: PendingTransaction):
        nonlocal tx_count, pump_fun_count
        tx_count += 1

        if tx.program_id and str(tx.program_id) == "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P":
            pump_fun_count += 1
            logger.info(
                "PUMP_FUN_TX_FOUND",
                signature=tx.signature[:16] + "...",
                buy_amount=tx.buy_amount_sol,
                fee=tx.fee
            )
        else:
            logger.debug(f"tx #{tx_count}", signature=tx.signature[:16])

    # Start monitoring
    monitoring_task = asyncio.create_task(monitor.start_monitoring(on_tx))

    try:
        await asyncio.sleep(duration_s)
    finally:
        await monitor.stop_monitoring()
        monitoring_task.cancel()

        logger.info(
            "RESULTS",
            total_txs=tx_count,
            pump_fun_txs=pump_fun_count,
            duration_s=duration_s
        )

        if tx_count == 0:
            logger.error("NO TRANSACTIONS AT ALL - LaserStream issue")
        elif pump_fun_count == 0:
            logger.warning("Got transactions but NO Pump.fun - filter issue")
        else:
            logger.info(f"SUCCESS - Found {pump_fun_count} Pump.fun txs!")


async def main():
    setup_logging(level="INFO", format="console")
    await test_all_transactions(duration_s=10)


if __name__ == "__main__":
    asyncio.run(main())
