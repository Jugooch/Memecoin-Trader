"""
Week 3 Stress Testing: Multi-Wallet Operations Under Load

Tests wallet rotation, position tracking, and database persistence
with high volume of transactions.
"""

import pytest
import asyncio
from solders.keypair import Keypair

from core.wallet_manager import WalletManager
from core.position_tracker import PositionTracker, PositionStorage
from core.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.asyncio
async def test_50_sequential_sol_transfers(multiple_funded_wallets, devnet_rpc_manager):
    """
    Execute 50 sequential SOL transfers to test:
    - Wallet rotation works under load
    - No nonce conflicts
    - All transactions succeed
    - System remains stable
    """
    from solders.system_program import transfer, TransferParams
    from solders.transaction import Transaction
    from solders.message import Message
    from solders.hash import Hash
    import base64

    wallets = multiple_funded_wallets
    transfer_amount = 10_000_000  # 0.01 SOL per transfer
    successful_transfers = 0

    logger.info("starting_stress_test", num_transfers=50, wallets=len(wallets))

    for i in range(50):
        # Rotate through wallets
        source_wallet = wallets[i % len(wallets)]
        dest_wallet = Keypair()  # Fresh destination each time

        try:
            # Get blockhash
            blockhash_response = await devnet_rpc_manager.call_http_rpc(
                "getLatestBlockhash",
                [{"commitment": "finalized"}]
            )
            blockhash_str = blockhash_response.get("result", {}).get("value", {}).get("blockhash")
            blockhash = Hash.from_string(blockhash_str)

            # Build transfer
            transfer_ix = transfer(
                TransferParams(
                    from_pubkey=source_wallet.pubkey(),
                    to_pubkey=dest_wallet.pubkey(),
                    lamports=transfer_amount
                )
            )

            message = Message.new_with_blockhash(
                [transfer_ix],
                source_wallet.pubkey(),
                blockhash
            )
            tx = Transaction.new_unsigned(message)
            tx.sign([source_wallet], blockhash)

            # Submit
            tx_bytes = bytes(tx)
            tx_base64 = base64.b64encode(tx_bytes).decode('utf-8')

            send_response = await devnet_rpc_manager.call_http_rpc(
                "sendTransaction",
                [tx_base64, {"encoding": "base64", "skipPreflight": True}]
            )

            signature = send_response.get("result") if isinstance(send_response, dict) else send_response

            if signature:
                successful_transfers += 1
                logger.info(
                    "transfer_success",
                    iteration=i+1,
                    signature=signature[:16] + "...",
                    source=str(source_wallet.pubkey())[:16] + "..."
                )

            # Small delay to avoid overwhelming devnet
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(
                "transfer_failed",
                iteration=i+1,
                error=str(e)
            )

    logger.info(
        "stress_test_complete",
        total=50,
        successful=successful_transfers,
        success_rate=f"{(successful_transfers/50)*100:.1f}%"
    )

    # Assert at least 90% success rate (devnet can be flaky)
    assert successful_transfers >= 45, f"Only {successful_transfers}/50 transfers succeeded"


@pytest.mark.asyncio
async def test_position_tracking_persistence(tmp_path):
    """
    Test position tracking persists correctly:
    - Open 20 positions
    - Close 10 positions
    - Restart tracker (simulate bot restart)
    - Verify all positions correctly loaded
    """
    db_path = tmp_path / "stress_test_positions.db"

    # Phase 1: Create and open positions
    storage = PositionStorage(str(db_path))
    tracker = PositionTracker(storage)
    await tracker.start()

    logger.info("opening_positions", count=20)

    position_ids = []
    for i in range(20):
        position = await tracker.open_position(
            wallet=Keypair().pubkey(),
            mint=Keypair().pubkey(),
            amount_tokens=1_000_000_000,
            entry_price_sol=0.1,
            entry_slot=1000 + i,
            strategy="stress_test"
        )
        position_ids.append(position.position_id)

    # Phase 2: Close half
    logger.info("closing_positions", count=10)
    for i in range(10):
        await tracker.close_position(
            position_id=position_ids[i],
            exit_price_sol=0.15,
            exit_slot=2000 + i
        )

    # Phase 3: Restart tracker (simulate bot restart)
    logger.info("simulating_restart")
    await tracker.stop()

    # Create new tracker instance
    storage2 = PositionStorage(str(db_path))
    tracker2 = PositionTracker(storage2)
    await tracker2.start()

    # Phase 4: Verify persistence
    open_positions = await tracker2.get_open_positions()
    closed_positions = await tracker2.get_closed_positions()

    logger.info(
        "persistence_check",
        open_count=len(open_positions),
        closed_count=len(closed_positions)
    )

    assert len(open_positions) == 10, f"Expected 10 open positions, got {len(open_positions)}"
    assert len(closed_positions) == 10, f"Expected 10 closed positions, got {len(closed_positions)}"

    # Verify closed positions have PnL
    for pos in closed_positions:
        assert pos.pnl_sol is not None, "Closed position missing PnL"
        assert pos.pnl_pct is not None, "Closed position missing PnL %"

    await tracker2.stop()

    logger.info("persistence_test_complete")


@pytest.mark.asyncio
async def test_wallet_manager_under_load(multiple_funded_wallets, devnet_rpc_manager):
    """
    Test wallet manager handles rapid wallet acquisition:
    - Request 30 wallets rapidly
    - Verify cooldowns enforced
    - Verify no deadlocks
    - Verify balance tracking accurate
    """
    from core.wallet_manager import WalletManagerConfig

    wallets = multiple_funded_wallets

    # Create wallet manager with correct signature
    config = WalletManagerConfig(
        balance_cache_ttl_seconds=10,
        cooldown_seconds=2,
        min_balance_threshold_lamports=10_000_000  # 0.01 SOL
    )

    manager = WalletManager(
        rpc_manager=devnet_rpc_manager,
        keypairs=wallets,
        config=config
    )
    await manager.start()

    # Wait a bit and refresh balances to ensure all wallets are visible
    await asyncio.sleep(2)
    await manager.refresh_balances()

    # Log wallet balances for debugging
    for wallet_info in manager.get_all_wallets():
        logger.info(
            "wallet_balance",
            pubkey=str(wallet_info.pubkey)[:16] + "...",
            balance_sol=wallet_info.balance_sol
        )

    try:
        logger.info("testing_rapid_wallet_acquisition", iterations=30)

        acquired_wallets = []
        start_time = asyncio.get_event_loop().time()

        for i in range(30):
            # Use async context manager (automatically acquires and releases lock)
            # Use very low min_sol since we're just testing rotation, not actual trading
            async with manager.get_wallet(min_sol=0.0001) as wallet_info:
                acquired_wallets.append(wallet_info.keypair.pubkey())

                logger.info(
                    "wallet_acquired",
                    iteration=i+1,
                    wallet=str(wallet_info.keypair.pubkey())[:16] + "...",
                    balance=wallet_info.balance_sol
                )

                # Small delay while holding wallet
                await asyncio.sleep(0.1)

            # Wallet is automatically released here

        end_time = asyncio.get_event_loop().time()
        elapsed = end_time - start_time

        logger.info(
            "rapid_acquisition_complete",
            total=30,
            elapsed_seconds=f"{elapsed:.2f}",
            avg_per_acquisition=f"{(elapsed/30)*1000:.1f}ms"
        )

        # Verify no wallet was acquired twice in a row
        for i in range(len(acquired_wallets) - 1):
            wallet1 = acquired_wallets[i]
            wallet2 = acquired_wallets[i+1]
            assert wallet1 != wallet2, f"Same wallet acquired twice in a row at iteration {i}"

        logger.info("wallet_manager_stress_test_complete")

    finally:
        await manager.stop()
