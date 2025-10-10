"""
Integration Test 2: Wallet Operations
Tests REAL wallet creation, funding (airdrops), transfers, and balance tracking on devnet
"""

import pytest
import asyncio
import time
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import transfer, TransferParams
from solders.transaction import Transaction
from solders.message import Message
from solders.hash import Hash

from core.wallet_manager import WalletManager, WalletManagerConfig
from core.logger import get_logger


logger = get_logger(__name__)

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# WALLET CREATION & KEY GENERATION
# =============================================================================

def test_create_new_keypair():
    """Test creating a new Solana keypair"""
    keypair = Keypair()

    assert keypair is not None, "Keypair should be created"
    assert keypair.pubkey() is not None, "Pubkey should exist"

    pubkey = keypair.pubkey()
    assert isinstance(pubkey, Pubkey), "Pubkey should be Pubkey type"

    pubkey_str = str(pubkey)
    assert len(pubkey_str) > 0, "Pubkey string should not be empty"
    # Base58 encoded Solana pubkeys are typically 43-44 chars (depending on leading zeros)
    assert 32 <= len(pubkey_str) <= 44, f"Pubkey should be 32-44 chars (base58), got {len(pubkey_str)}"

    logger.info("keypair_created", pubkey=pubkey_str)


def test_create_multiple_unique_keypairs():
    """Test that multiple keypairs are unique"""
    keypairs = [Keypair() for _ in range(5)]

    pubkeys = [str(kp.pubkey()) for kp in keypairs]

    # All should be unique
    assert len(set(pubkeys)) == 5, "All keypairs should be unique"

    logger.info("unique_keypairs_created", count=5)


# =============================================================================
# WALLET BALANCE CHECKS
# Tests reading balance from devnet (uses pre-funded wallet)
# =============================================================================

@pytest.mark.asyncio
async def test_check_wallet_balance(funded_wallet, devnet_rpc_manager):
    """Test checking balance of a pre-funded wallet"""
    keypair = funded_wallet
    pubkey = keypair.pubkey()

    # Check balance
    response = await devnet_rpc_manager.call_http_rpc(
        "getBalance",
        [str(pubkey)]
    )

    if isinstance(response, dict):
        balance = response.get("result", {}).get("value", 0)
    else:
        balance = 0

    # Should have some balance (at least 0.1 SOL)
    assert balance >= 100_000_000, \
        f"Balance should be at least 0.1 SOL, got {balance / 1e9:.4f} SOL"

    logger.info(
        "wallet_balance_checked",
        pubkey=str(pubkey),
        balance_sol=balance / 1e9
    )


# =============================================================================
# SOL TRANSFERS
# Tests submitting real transactions to devnet (uses pre-funded wallet)
# =============================================================================

@pytest.mark.asyncio
async def test_transfer_sol_between_wallets(funded_wallet, devnet_rpc_manager):
    """Test transferring SOL from one wallet to another (REAL transaction)"""
    sender = funded_wallet
    receiver = Keypair()

    sender_pubkey = sender.pubkey()
    receiver_pubkey = receiver.pubkey()

    # Get initial balances
    sender_balance_initial_response = await devnet_rpc_manager.call_http_rpc(
        "getBalance",
        [str(sender_pubkey)]
    )
    sender_balance_initial = sender_balance_initial_response.get("result", {}).get("value", 0)

    # Build transfer transaction
    transfer_amount = 500_000_000  # 0.5 SOL

    # Get latest blockhash
    blockhash_response = await devnet_rpc_manager.call_http_rpc(
        "getLatestBlockhash",
        [{"commitment": "finalized"}]
    )
    blockhash_str = blockhash_response.get("result", {}).get("value", {}).get("blockhash")

    assert blockhash_str is not None, "Should have blockhash"
    blockhash = Hash.from_string(blockhash_str)

    # Create transfer instruction
    transfer_ix = transfer(
        TransferParams(
            from_pubkey=sender_pubkey,
            to_pubkey=receiver_pubkey,
            lamports=transfer_amount
        )
    )

    # Build transaction using new API
    message = Message.new_with_blockhash(
        [transfer_ix],
        sender_pubkey,
        blockhash
    )
    tx = Transaction.new_unsigned(message)
    tx.sign([sender], blockhash)

    # Serialize and encode
    tx_bytes = bytes(tx)
    import base64
    tx_base64 = base64.b64encode(tx_bytes).decode('utf-8')

    # Send transaction
    send_response = await devnet_rpc_manager.call_http_rpc(
        "sendTransaction",
        [tx_base64, {"encoding": "base64", "skipPreflight": False}]
    )

    # Check for errors in send response
    if isinstance(send_response, dict):
        if "error" in send_response:
            error_msg = send_response["error"]
            raise AssertionError(f"Transaction send failed: {error_msg}")
        signature = send_response.get("result")
    else:
        signature = send_response

    assert signature is not None, f"Should receive transaction signature, got: {send_response}"

    logger.info("transaction_signature_received", signature=signature)

    logger.info(
        "transfer_submitted",
        from_pubkey=str(sender_pubkey),
        to_pubkey=str(receiver_pubkey),
        amount_sol=transfer_amount / 1e9,
        signature=signature
    )

    # Wait for confirmation (up to 40 seconds)
    confirmed = False
    tx_error = None
    for attempt in range(20):
        await asyncio.sleep(2)

        status_response = await devnet_rpc_manager.call_http_rpc(
            "getSignatureStatuses",
            [[signature], {"searchTransactionHistory": True}]
        )

        if isinstance(status_response, dict):
            result = status_response.get("result", {})
            value = result.get("value", [])

            if value and len(value) > 0 and value[0] is not None:
                status = value[0]
                confirmation_status = status.get("confirmationStatus")
                tx_error = status.get("err")

                logger.info(
                    "transaction_status_check",
                    attempt=attempt + 1,
                    confirmation_status=confirmation_status,
                    error=tx_error
                )

                if confirmation_status in ["confirmed", "finalized"]:
                    if tx_error is None:
                        confirmed = True
                        logger.info("transaction_confirmed", signature=signature)
                        break
                    else:
                        logger.error("transaction_failed", signature=signature, error=tx_error)
                        raise AssertionError(f"Transaction failed with error: {tx_error}")

    if not confirmed:
        raise AssertionError(f"Transaction did not confirm after 40 seconds: {signature}")

    # Get transaction details to verify it executed correctly
    tx_response = await devnet_rpc_manager.call_http_rpc(
        "getTransaction",
        [signature, {"encoding": "json", "maxSupportedTransactionVersion": 0}]
    )

    tx_result = tx_response.get("result") if isinstance(tx_response, dict) else None
    if tx_result:
        tx_meta = tx_result.get("meta", {})
        tx_err = tx_meta.get("err")

        logger.info(
            "transaction_details",
            signature=signature,
            error=tx_err,
            fee=tx_meta.get("fee"),
            pre_balances=tx_meta.get("preBalances"),
            post_balances=tx_meta.get("postBalances")
        )

        if tx_err:
            raise AssertionError(f"Transaction failed on-chain: {tx_err}")

    # Wait for balance to settle on devnet with retry
    receiver_balance = 0
    for retry in range(5):
        await asyncio.sleep(2)

        receiver_balance_response = await devnet_rpc_manager.call_http_rpc(
            "getBalance",
            [str(receiver_pubkey)]
        )
        receiver_balance = receiver_balance_response.get("result", {}).get("value", 0)

        logger.info(
            "balance_check_retry",
            retry=retry + 1,
            receiver_balance_sol=receiver_balance / 1e9
        )

        if receiver_balance >= transfer_amount * 0.99:
            break

    # Account for transaction fees - receiver should get close to transfer_amount
    assert receiver_balance >= transfer_amount * 0.99, \
        f"Receiver should have at least {transfer_amount * 0.99 / 1e9:.4f} SOL, got {receiver_balance / 1e9:.4f} SOL. " \
        f"Transaction: https://explorer.solana.com/tx/{signature}?cluster=devnet"

    logger.info(
        "transfer_completed",
        receiver_balance_sol=receiver_balance / 1e9,
        signature=signature
    )


# =============================================================================
# WALLET MANAGER INTEGRATION
# =============================================================================

@pytest.mark.asyncio
async def test_wallet_manager_with_devnet_wallets(devnet_rpc_manager, multiple_funded_wallets):
    """Test WalletManager with real funded devnet wallets"""
    keypairs = multiple_funded_wallets

    # Create wallet manager
    config = WalletManagerConfig(
        balance_cache_ttl_seconds=10,
        cooldown_seconds=2
    )

    manager = WalletManager(devnet_rpc_manager, keypairs, config)
    await manager.start()

    try:
        # Manager should have 3 wallets
        assert len(manager.wallets) == 3, "Should have 3 wallets"

        # Get total balance (not async)
        total_balance = manager.get_total_balance_sol()
        logger.info("wallet_manager_total_balance", balance_sol=total_balance)

        # Should have ~1 SOL total (0.3 SOL per wallet x 3, plus main wallet remainder)
        # After 0.6 SOL transferred from main wallet, we have ~0.9+ SOL total
        assert total_balance >= 0.8, f"Total balance should be at least 0.8 SOL, got {total_balance}"

        # Test wallet acquisition (lowered requirement to match reduced funding)
        async with manager.get_wallet(min_sol=0.2) as wallet:
            assert wallet is not None, "Should get wallet"
            assert wallet.balance_sol >= 0.2, "Wallet should meet minimum balance"

            logger.info(
                "wallet_acquired",
                pubkey=str(wallet.pubkey),
                balance_sol=wallet.balance_sol
            )

        # Wallet should be in cooldown after use
        wallet_info = manager.get_wallet_info(wallet.pubkey)
        assert wallet_info.is_in_cooldown is True, "Wallet should be in cooldown"

    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_wallet_manager_rotation_with_real_wallets(devnet_rpc_manager, multiple_funded_wallets):
    """Test wallet rotation with real devnet wallets"""
    keypairs = multiple_funded_wallets

    config = WalletManagerConfig(cooldown_seconds=1)
    manager = WalletManager(devnet_rpc_manager, keypairs, config)
    await manager.start()

    try:
        used_wallets = []

        # Acquire wallets 3 times - should rotate through different wallets
        for i in range(3):
            async with manager.get_wallet(min_sol=0.2) as wallet:
                used_wallets.append(str(wallet.pubkey))
                logger.info(f"wallet_rotation_{i+1}", pubkey=str(wallet.pubkey))

        # With 3 funded wallets, should use at least 2 different wallets (rotation)
        assert len(set(used_wallets)) >= 2, \
            f"Should use at least 2 different wallets in rotation, got {len(set(used_wallets))} different wallets"

        # All 3 acquisitions should have succeeded
        assert len(used_wallets) == 3, "Should have acquired wallet 3 times"

    finally:
        await manager.stop()


# =============================================================================
# BALANCE TRACKING & CACHING
# =============================================================================

@pytest.mark.asyncio
async def test_wallet_balance_cache(devnet_rpc_manager, funded_wallet):
    """Test that wallet manager caches balances correctly"""
    keypair = funded_wallet

    config = WalletManagerConfig(balance_cache_ttl_seconds=10)
    manager = WalletManager(devnet_rpc_manager, [keypair], config)
    await manager.start()

    try:
        wallet_info = list(manager.wallets.values())[0]

        # First balance check - should fetch from RPC
        balance1 = wallet_info.balance_sol
        cache_time1 = wallet_info.balance_cached_at

        assert balance1 > 0, "Balance should be positive"
        assert cache_time1 > 0, "Cache time should be set"

        # Immediate second check - should use cache
        await asyncio.sleep(0.1)
        wallet_info2 = list(manager.wallets.values())[0]
        balance2 = wallet_info2.balance_sol
        cache_time2 = wallet_info2.balance_cached_at

        assert balance2 == balance1, "Cached balance should be same"
        assert cache_time2 == cache_time1, "Cache time should not change"

        logger.info(
            "balance_cache_validated",
            balance_sol=balance1,
            cache_age_seconds=time.time() - cache_time1
        )

    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_wallet_balance_refresh_on_cache_expiry(devnet_rpc_manager, funded_wallet):
    """Test that wallet balance refreshes when cache expires"""
    keypair = funded_wallet

    # Short cache TTL for testing
    config = WalletManagerConfig(balance_cache_ttl_seconds=2)
    manager = WalletManager(devnet_rpc_manager, [keypair], config)
    await manager.start()

    try:
        wallet_info = list(manager.wallets.values())[0]

        # Get initial cache time
        initial_cache_time = wallet_info.balance_cached_at

        # Wait for cache to expire
        await asyncio.sleep(3)

        # Force refresh
        await manager.refresh_balances()

        # Cache time should be updated
        new_cache_time = wallet_info.balance_cached_at
        assert new_cache_time > initial_cache_time, "Cache should be refreshed"

        logger.info(
            "balance_cache_refreshed",
            initial_cache_time=initial_cache_time,
            new_cache_time=new_cache_time
        )

    finally:
        await manager.stop()
