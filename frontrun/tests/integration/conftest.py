"""
Shared fixtures for integration tests
Provides real devnet connections, funded wallets, and test infrastructure
"""

import pytest
import pytest_asyncio
import asyncio
import aiohttp
import os
from typing import Optional
from solders.keypair import Keypair
from solders.pubkey import Pubkey

from core.rpc_manager import RPCManager, RPCEndpoint, RPCConfig
from core.bonding_curve import BondingCurveCalculator
from core.slippage import SlippageManager
from core.pnl import PnLCalculator
from core.wallet_manager import WalletManager
from core.logger import setup_logging, get_logger


# Setup logging for integration tests
setup_logging(level="INFO", format="console")
logger = get_logger(__name__)


# =============================================================================
# DEVNET CONFIGURATION
# =============================================================================

# Public devnet RPC endpoints
DEVNET_RPC_ENDPOINTS = [
    "https://api.devnet.solana.com",
    "https://rpc.ankr.com/solana_devnet",
]

# Pump.fun program ID (same on devnet and mainnet)
PUMP_FUN_PROGRAM_ID = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")


# =============================================================================
# SESSION-LEVEL FIXTURES (shared across all tests)
# =============================================================================

@pytest_asyncio.fixture(scope="session")
async def devnet_rpc_config():
    """Create RPC configuration for devnet"""
    # Solana devnet websocket endpoints
    DEVNET_WS_ENDPOINTS = [
        "wss://api.devnet.solana.com",
        "wss://rpc.ankr.com/solana_devnet"
    ]

    endpoints = [
        RPCEndpoint(
            url=DEVNET_RPC_ENDPOINTS[0],
            websocket_url=DEVNET_WS_ENDPOINTS[0],
            label="devnet_primary",
            priority=10
        ),
        RPCEndpoint(
            url=DEVNET_RPC_ENDPOINTS[1],
            websocket_url=DEVNET_WS_ENDPOINTS[1],
            label="devnet_backup",
            priority=5
        )
    ]

    config = RPCConfig(
        endpoints=endpoints,
        failover_threshold_errors=3,
        failover_threshold_latency_ms=500,
        health_check_interval_s=60
    )

    return config


@pytest_asyncio.fixture(scope="function")
async def devnet_rpc_manager(devnet_rpc_config):
    """
    Create RPC manager connected to Solana devnet
    Fresh instance for each test to ensure proper event loop compatibility

    This is a REAL connection to devnet - no mocks!
    """
    manager = RPCManager(devnet_rpc_config)
    await manager.start()

    logger.info(
        "devnet_rpc_manager_started",
        endpoints=[ep.label for ep in devnet_rpc_config.endpoints]
    )

    yield manager

    await manager.stop()
    logger.info("devnet_rpc_manager_stopped")


@pytest_asyncio.fixture(scope="function")
async def devnet_connection_info(devnet_rpc_manager):
    """
    Get devnet connection info (slot, version, etc.)
    Validates we can actually connect to devnet
    """
    try:
        # Get current slot via HTTP RPC (not WebSocket)
        slot_response = await devnet_rpc_manager.call_http_rpc("getSlot", [])
        slot = slot_response if isinstance(slot_response, int) else slot_response.get("result", 0)

        # Get version via HTTP RPC
        version_response = await devnet_rpc_manager.call_http_rpc("getVersion", [])
        version = version_response.get("result", {}) if isinstance(version_response, dict) else {}

        info = {
            "slot": slot,
            "version": version,
            "connected": True
        }

        logger.info(
            "devnet_connected",
            slot=slot,
            version=version.get("solana-core") if isinstance(version, dict) else "unknown"
        )

        return info

    except Exception as e:
        logger.error("devnet_connection_failed", error=str(e))
        return {
            "slot": 0,
            "version": {},
            "connected": False,
            "error": str(e)
        }


# =============================================================================
# TEST-LEVEL FIXTURES (created fresh for each test)
# =============================================================================

@pytest_asyncio.fixture
async def fresh_keypair():
    """Generate a fresh keypair for testing"""
    keypair = Keypair()
    logger.debug("fresh_keypair_generated", pubkey=str(keypair.pubkey()))
    return keypair


@pytest_asyncio.fixture
async def funded_wallet(devnet_rpc_manager):
    """
    Provide a funded wallet for testing.

    Uses pre-funded wallet from environment variable DEVNET_TEST_WALLET_PRIVATE_KEY.
    Format: JSON array like [1,2,3,...] (64 bytes) or base58 string

    If not set, tests requiring funded wallets will skip.

    To set up:
    1. Generate a wallet: solana-keygen new --outfile test-wallet.json
    2. Fund it: Visit https://faucet.solana.com and send SOL to the address
    3. Set env var: export DEVNET_TEST_WALLET_PRIVATE_KEY=$(cat test-wallet.json)
    """
    private_key_str = os.getenv("DEVNET_TEST_WALLET_PRIVATE_KEY")

    if not private_key_str:
        pytest.skip(
            "No funded wallet available. Set DEVNET_TEST_WALLET_PRIVATE_KEY environment variable. "
            "See conftest.py for setup instructions."
        )

    try:
        # Try parsing as JSON array first
        import json
        if private_key_str.strip().startswith('['):
            key_bytes = bytes(json.loads(private_key_str))
            keypair = Keypair.from_bytes(key_bytes)
        else:
            # Try as base58
            import base58
            key_bytes = base58.b58decode(private_key_str)
            keypair = Keypair.from_bytes(key_bytes)
    except Exception as e:
        pytest.skip(f"Failed to load funded wallet from env var: {e}")

    # Verify wallet has balance (need at least 1 SOL for funding 2 additional wallets)
    balance = await _get_balance(devnet_rpc_manager, keypair.pubkey())

    if balance < 1_000_000_000:  # Less than 1 SOL
        pytest.skip(
            f"Funded wallet has insufficient balance: {balance / 1e9:.4f} SOL. "
            f"Need at least 1 SOL to fund additional test wallets. "
            f"Please fund {keypair.pubkey()} at https://faucet.solana.com"
        )

    logger.info(
        "using_prefunded_wallet",
        pubkey=str(keypair.pubkey()),
        balance_sol=balance / 1e9
    )

    return keypair


@pytest_asyncio.fixture(scope="function")
async def multiple_funded_wallets(funded_wallet, devnet_rpc_manager):
    """
    Provide 3 funded wallets for multi-wallet testing (SESSION SCOPED - reused across tests)

    Creates 2 additional wallets ONCE per test session and funds them from the main wallet.
    These wallets are reused across all tests to avoid draining SOL.
    Each wallet will have ~3.3 SOL (10 SOL total / 3 wallets)
    """
    from solders.system_program import transfer, TransferParams
    from solders.transaction import Transaction
    from solders.message import Message
    from solders.hash import Hash
    import base64
    import json

    # Use the pre-funded wallet as the first wallet
    wallet1 = funded_wallet

    # Check if we already have cached wallets from a previous session
    # Store them in a file so they persist across test runs
    cache_file = ".test_wallets_cache.json"

    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                wallet2 = Keypair.from_bytes(bytes(cached['wallet2']))
                wallet3 = Keypair.from_bytes(bytes(cached['wallet3']))

                # Verify wallets still have balance
                balance2 = await _get_balance(devnet_rpc_manager, wallet2.pubkey())
                balance3 = await _get_balance(devnet_rpc_manager, wallet3.pubkey())

                if balance2 >= 100_000_000 and balance3 >= 100_000_000:  # At least 0.1 SOL each
                    logger.info(
                        "reusing_cached_wallets",
                        wallet2=str(wallet2.pubkey()),
                        wallet3=str(wallet3.pubkey()),
                        balance2=balance2 / 1e9,
                        balance3=balance3 / 1e9
                    )
                    return [wallet1, wallet2, wallet3]
    except Exception as e:
        logger.warning("failed_to_load_cached_wallets", error=str(e))

    # Create 2 new keypairs if cache miss or insufficient balance
    wallet2 = Keypair()
    wallet3 = Keypair()

    # Save to cache for next time
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'wallet2': list(bytes(wallet2)),
                'wallet3': list(bytes(wallet3))
            }, f)
    except Exception as e:
        logger.warning("failed_to_cache_wallets", error=str(e))

    # Get initial balance
    balance_response = await devnet_rpc_manager.call_http_rpc(
        "getBalance",
        [str(wallet1.pubkey())]
    )
    initial_balance = balance_response.get("result", {}).get("value", 0)

    # Fund each new wallet with just 0.3 SOL (enough for testing, way cheaper!)
    # Total: 0.3 + 0.3 = 0.6 SOL transferred (saves 6 SOL compared to before!)
    # Each wallet gets 300M lamports = plenty for fees + min balance requirements
    transfer_amount = 300_000_000  # 0.3 SOL in lamports

    for new_wallet in [wallet2, wallet3]:
        # Get latest blockhash
        blockhash_response = await devnet_rpc_manager.call_http_rpc(
            "getLatestBlockhash",
            [{"commitment": "finalized"}]
        )
        blockhash_str = blockhash_response.get("result", {}).get("value", {}).get("blockhash")
        blockhash = Hash.from_string(blockhash_str)

        # Create transfer instruction
        transfer_ix = transfer(
            TransferParams(
                from_pubkey=wallet1.pubkey(),
                to_pubkey=new_wallet.pubkey(),
                lamports=transfer_amount
            )
        )

        # Build transaction
        message = Message.new_with_blockhash(
            [transfer_ix],
            wallet1.pubkey(),
            blockhash
        )
        tx = Transaction.new_unsigned(message)
        tx.sign([wallet1], blockhash)

        # Submit transaction
        tx_bytes = bytes(tx)
        tx_base64 = base64.b64encode(tx_bytes).decode('utf-8')

        send_response = await devnet_rpc_manager.call_http_rpc(
            "sendTransaction",
            [tx_base64, {"encoding": "base64", "skipPreflight": True}]
        )

        signature = send_response.get("result") if isinstance(send_response, dict) else send_response

        # Wait for confirmation
        for _ in range(30):  # Wait up to 30 seconds
            await asyncio.sleep(1)
            status_response = await devnet_rpc_manager.call_http_rpc(
                "getSignatureStatuses",
                [[signature]]
            )

            if isinstance(status_response, dict):
                result = status_response.get("result", {})
                value = result.get("value", [])
                if value and len(value) > 0 and value[0] is not None:
                    status = value[0].get("confirmationStatus")
                    if status in ["confirmed", "finalized"]:
                        break

        # Verify wallet was funded (retry a few times if needed)
        balance = 0
        for retry in range(5):
            await asyncio.sleep(0.5)  # Brief wait before checking
            balance_response = await devnet_rpc_manager.call_http_rpc(
                "getBalance",
                [str(new_wallet.pubkey())]
            )
            balance = balance_response.get("result", {}).get("value", 0)

            if balance >= transfer_amount * 0.9:  # Account for fees
                logger.info(
                    "wallet_funded",
                    pubkey=str(new_wallet.pubkey()),
                    balance_sol=balance / 1e9,
                    signature=signature[:16] + "..." if signature else None
                )
                break

            logger.warning(
                "wallet_funding_retry",
                pubkey=str(new_wallet.pubkey()),
                balance_sol=balance / 1e9,
                retry=retry + 1
            )

        if balance < transfer_amount * 0.9:
            pytest.skip(
                f"Failed to fund wallet {new_wallet.pubkey()}: "
                f"expected ~{transfer_amount / 1e9:.2f} SOL, got {balance / 1e9:.4f} SOL. "
                f"Transaction: {signature}"
            )

    # Final verification - check all balances one more time
    await asyncio.sleep(1)  # Brief pause to ensure RPC state is settled

    final_balances = []
    for i, wallet in enumerate([wallet1, wallet2, wallet3], 1):
        balance_response = await devnet_rpc_manager.call_http_rpc(
            "getBalance",
            [str(wallet.pubkey())]
        )
        balance = balance_response.get("result", {}).get("value", 0)
        final_balances.append(balance / 1e9)

    logger.info(
        "multiple_wallets_created",
        wallet1=str(wallet1.pubkey()),
        wallet2=str(wallet2.pubkey()),
        wallet3=str(wallet3.pubkey()),
        balance1=final_balances[0],
        balance2=final_balances[1],
        balance3=final_balances[2],
        total_balance=sum(final_balances)
    )

    return [wallet1, wallet2, wallet3]


# =============================================================================
# CALCULATOR FIXTURES (stateless, can be session-scoped)
# =============================================================================

@pytest.fixture(scope="session")
def bonding_curve_calculator():
    """Real bonding curve calculator"""
    return BondingCurveCalculator()


@pytest.fixture(scope="session")
def slippage_manager():
    """Real slippage manager"""
    return SlippageManager()


@pytest.fixture(scope="session")
def pnl_calculator():
    """Real PnL calculator"""
    return PnLCalculator()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def _get_balance(rpc_manager: RPCManager, pubkey: Pubkey) -> int:
    """Get SOL balance for pubkey"""
    try:
        response = await rpc_manager.call_http_rpc(
            "getBalance",
            [str(pubkey)]
        )

        if isinstance(response, dict):
            return response.get("result", {}).get("value", 0)
        return 0

    except Exception as e:
        logger.error("get_balance_failed", pubkey=str(pubkey), error=str(e))
        return 0


async def _wait_for_transaction_confirmation(
    rpc_manager: RPCManager,
    signature: str,
    timeout_seconds: int = 60
) -> bool:
    """
    Wait for transaction confirmation on devnet

    Returns True if confirmed, False if timeout
    """
    import time
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        try:
            response = await rpc_manager.call_http_rpc(
                "getSignatureStatuses",
                [[signature], {"searchTransactionHistory": True}]
            )

            if isinstance(response, dict):
                result = response.get("result", {})
                value = result.get("value", [])

                if value and len(value) > 0 and value[0] is not None:
                    status = value[0]
                    confirmation_status = status.get("confirmationStatus")

                    if confirmation_status in ["confirmed", "finalized"]:
                        # Check for errors
                        if status.get("err") is None:
                            return True
                        else:
                            logger.error(
                                "transaction_failed",
                                signature=signature,
                                error=status.get("err")
                            )
                            return False

            await asyncio.sleep(2)

        except Exception as e:
            logger.warning(
                "confirmation_check_error",
                signature=signature,
                error=str(e)
            )
            await asyncio.sleep(2)

    logger.warning("transaction_confirmation_timeout", signature=signature)
    return False
