"""
Buy-Hold-Sell Test

Test the full cycle:
1. Detect token creation
2. Buy immediately
3. Hold for X seconds
4. Sell
5. Calculate P&L

Usage:
    python frontrun/scripts/test_buy_hold_sell.py \
        --wallet frontrun/data/phase2-wallet.json \
        --buy-amount 0.005 \
        --hold-seconds 10 \
        --priority-fee 100000 \
        --real-trades
"""

import asyncio
import sys
import json
import struct
import base58
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solders.keypair import Keypair
from solders.pubkey import Pubkey

from core.config import ConfigurationManager
from core.logger import setup_logging, get_logger
from core.rpc_manager import RPCManager
from core.wallet_manager import WalletManager, WalletManagerConfig
from core.bonding_curve import get_initial_bonding_curve_state
from clients.pump_fun_direct import PumpFunDirectClient
from core.tx_submitter import TransactionSubmitter, SubmitterConfig

# Import Geyser
import grpc
proto_path = Path(__file__).parent.parent / "proto"
sys.path.insert(0, str(proto_path))
import geyser_pb2
import geyser_pb2_grpc

logger = get_logger(__name__)

# Constants
PUMP_CREATE_DISCRIMINATOR = struct.pack("<Q", 8576854823835016728)
PUMP_FUN_PROGRAM = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")


@dataclass
class TokenCreation:
    """Token creation event"""
    mint: str
    bonding_curve: str
    associated_bonding_curve: str
    creator: str
    name: str
    symbol: str
    signature: str
    slot: int
    timestamp: datetime


@dataclass
class TradeResult:
    """Result of buy-hold-sell cycle"""
    mint: str
    symbol: str

    # Buy
    buy_signature: str
    buy_slot: int
    buy_sol: float
    buy_tokens: float

    # Sell
    sell_signature: str
    sell_slot: int
    sell_sol_received: float

    # P&L
    net_sol: float
    pnl_pct: float

    # Timing
    hold_seconds: float

    success: bool
    error: Optional[str] = None


class TokenCreationMonitor:
    """Monitor token creations via Geyser"""

    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self._monitoring = False

    def decode_create_instruction(self, ix_data: bytes, account_keys, accounts):
        """Decode token creation instruction"""
        try:
            offset = 8  # Skip discriminator

            def get_account_key(index: int) -> str:
                if index >= len(accounts):
                    return None
                account_index = accounts[index]
                if account_index >= len(account_keys):
                    return None
                key_bytes = account_keys[account_index]
                return base58.b58encode(bytes(key_bytes)).decode()

            def read_string() -> str:
                nonlocal offset
                length = struct.unpack_from("<I", ix_data, offset)[0]
                offset += 4
                value = ix_data[offset:offset + length].decode('utf-8')
                offset += length
                return value

            def read_pubkey() -> str:
                nonlocal offset
                value = base58.b58encode(ix_data[offset:offset + 32]).decode('utf-8')
                offset += 32
                return value

            name = read_string()
            symbol = read_string()
            uri = read_string()
            creator = read_pubkey()

            return {
                'name': name,
                'symbol': symbol,
                'uri': uri,
                'creator': creator,
                'mint': get_account_key(0),
                'bonding_curve': get_account_key(2),
                'associated_bonding_curve': get_account_key(3),
            }

        except Exception as e:
            logger.debug("create_instruction_parse_failed", error=str(e))
            return None

    async def start_monitoring(self, callback):
        """Start monitoring for token creations"""
        self._monitoring = True

        class HeliusAuthMetadata(grpc.AuthMetadataPlugin):
            def __init__(self, api_key: str):
                self.api_key = api_key

            def __call__(self, context, callback_fn):
                metadata = (('x-token', self.api_key),)
                callback_fn(metadata, None)

        auth = HeliusAuthMetadata(self.api_key)
        call_creds = grpc.metadata_call_credentials(auth)
        ssl_creds = grpc.ssl_channel_credentials()
        combined_creds = grpc.composite_channel_credentials(ssl_creds, call_creds)

        channel = grpc.aio.secure_channel(
            self.endpoint,
            credentials=combined_creds,
            options=[('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        )
        stub = geyser_pb2_grpc.GeyserStub(channel)

        subscription_request = geyser_pb2.SubscribeRequest()
        subscription_request.transactions["pump_creates"].account_include.append(str(PUMP_FUN_PROGRAM))
        subscription_request.transactions["pump_creates"].failed = False
        subscription_request.commitment = geyser_pb2.CommitmentLevel.PROCESSED

        logger.info("subscribing_to_token_creations")

        try:
            async def request_generator():
                yield subscription_request
                while self._monitoring:
                    await asyncio.sleep(30)

            async for message in stub.Subscribe(request_generator()):
                if not message.HasField('transaction'):
                    continue

                try:
                    tx_update = message.transaction
                    transaction = tx_update.transaction

                    signature = base58.b58encode(bytes(transaction.signature)).decode('utf-8')
                    slot = tx_update.slot

                    tx_inner = transaction.transaction
                    msg = getattr(tx_inner, "message", None)

                    if msg is None:
                        continue

                    for ix in msg.instructions:
                        ix_data = bytes(ix.data)

                        if not ix_data.startswith(PUMP_CREATE_DISCRIMINATOR):
                            continue

                        creation_data = self.decode_create_instruction(
                            ix_data,
                            msg.account_keys,
                            list(ix.accounts)
                        )

                        if not creation_data:
                            continue

                        token_creation = TokenCreation(
                            mint=creation_data['mint'],
                            bonding_curve=creation_data['bonding_curve'],
                            associated_bonding_curve=creation_data['associated_bonding_curve'],
                            creator=creation_data['creator'],
                            name=creation_data['name'],
                            symbol=creation_data['symbol'],
                            signature=signature,
                            slot=slot,
                            timestamp=datetime.now()
                        )

                        try:
                            await callback(token_creation)
                        except Exception as e:
                            logger.error("callback_error", error=str(e))

                except Exception as e:
                    logger.error("message_processing_error", error=str(e))

        finally:
            await channel.close()

    async def stop_monitoring(self):
        """Stop monitoring"""
        self._monitoring = False


class BuyHoldSellTester:
    """Test buy-hold-sell cycle"""

    def __init__(
        self,
        config_path: str,
        wallet_path: str,
        buy_amount_sol: float,
        hold_seconds: int,
        priority_fee_lamports: int,
        enable_real_trades: bool
    ):
        self.buy_amount_sol = buy_amount_sol
        self.hold_seconds = hold_seconds
        self.priority_fee_lamports = priority_fee_lamports
        self.enable_real_trades = enable_real_trades
        self.completed = False

        # Load config
        config_mgr = ConfigurationManager(config_path)
        self.bot_config = config_mgr.load_config()

        # Load wallet
        self.keypair = self._load_wallet(wallet_path)
        logger.info("wallet_loaded", pubkey=str(self.keypair.pubkey()))

        # Get Helius config
        helius_endpoint = self.bot_config.rpc_config.endpoints[0]
        self.geyser_endpoint = helius_endpoint.laserstream_url.replace("https://", "").replace("http://", "")
        self.geyser_api_key = helius_endpoint.laserstream_api_key

        # Initialize trading components
        if self.enable_real_trades:
            self.rpc_manager = RPCManager(self.bot_config.rpc_config)
            self.wallet_manager = WalletManager(
                rpc_manager=self.rpc_manager,
                keypairs=[self.keypair],
                config=WalletManagerConfig(cooldown_seconds=0, balance_cache_ttl_seconds=30)
            )
            self.direct_client = PumpFunDirectClient(self.rpc_manager)
            self.tx_submitter = TransactionSubmitter(
                rpc_manager=self.rpc_manager,
                config=SubmitterConfig(skip_preflight=True, max_retries=3, confirmation_timeout_s=30)
            )

    def _load_wallet(self, wallet_path: str) -> Keypair:
        """Load wallet keypair"""
        wallet_file = Path(wallet_path)
        if not wallet_file.exists():
            raise FileNotFoundError(f"Wallet not found: {wallet_path}")

        with open(wallet_file, 'r') as f:
            key_data = json.load(f)

        return Keypair.from_bytes(bytes(key_data))

    async def _get_token_balance(self, mint: str) -> float:
        """Get token balance for our wallet"""
        try:
            response = await self.rpc_manager.call_http_rpc(
                "getTokenAccountsByOwner",
                [
                    str(self.keypair.pubkey()),
                    {"mint": mint},
                    {"encoding": "jsonParsed"}
                ]
            )

            result = response.get("result", {})
            accounts = result.get("value", [])

            if not accounts:
                return 0

            token_data = accounts[0].get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
            token_amount = token_data.get("tokenAmount", {})
            ui_amount = float(token_amount.get("uiAmount", 0))

            return ui_amount

        except Exception as e:
            logger.error("token_balance_fetch_failed", error=str(e))
            return 0

    async def run(self):
        """Run the test"""
        logger.info("=== BUY-HOLD-SELL TEST ===")
        logger.info(f"Buy amount: {self.buy_amount_sol} SOL")
        logger.info(f"Hold time: {self.hold_seconds} seconds")
        logger.info(f"Priority fee: {self.priority_fee_lamports / 1e9:.6f} SOL")
        logger.info(f"Mode: {'REAL TRADES' if self.enable_real_trades else 'SIMULATION'}")
        logger.info("Waiting for next token creation...\n")

        if self.enable_real_trades:
            await self.rpc_manager.start()
            await self.wallet_manager.start()

            balance = self.wallet_manager.get_total_balance_sol()
            logger.info(f"Wallet balance: {balance:.4f} SOL")

        # Start creation monitor
        monitor = TokenCreationMonitor(self.geyser_endpoint, self.geyser_api_key)
        monitoring_task = asyncio.create_task(
            monitor.start_monitoring(self._on_token_creation)
        )

        try:
            while not self.completed:
                await asyncio.sleep(1)

            logger.info(f"\n✅ Test complete!")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            await monitor.stop_monitoring()
            monitoring_task.cancel()

            if self.enable_real_trades:
                await self.wallet_manager.stop()
                await self.rpc_manager.stop()

    async def _on_token_creation(self, creation: TokenCreation):
        """Handle token creation - execute full cycle"""
        if self.completed:
            return

        logger.info(f"\n🎯 TOKEN DETECTED")
        logger.info(f"   Mint: {creation.mint}")
        logger.info(f"   Symbol: {creation.symbol}")
        logger.info(f"   Slot: {creation.slot}")

        if not self.enable_real_trades:
            logger.info("  ⏭️ Simulation mode - skipping")
            self.completed = True
            return

        try:
            start_time = datetime.now()

            # Step 1: BUY
            logger.info(f"\n1️⃣ BUYING {self.buy_amount_sol} SOL...")

            # For NEW tokens, use INITIAL bonding curve (not fetched - won't be indexed yet)
            from core.bonding_curve import BondingCurveCalculator, calculate_curve_state_after_buy

            # Get initial (fresh) bonding curve state
            initial_curve_state = get_initial_bonding_curve_state()

            # Calculate expected tokens from fresh curve
            calculator = BondingCurveCalculator()
            sol_lamports = int(self.buy_amount_sol * 1e9)
            buy_quote = calculator.calculate_buy_price(initial_curve_state, sol_lamports)
            tokens_expected_raw = int(buy_quote.tokens_out)  # Raw tokens (with 6 decimals)
            tokens_expected_ui = tokens_expected_raw / 1e6  # UI amount for logging

            logger.info(f"   Expected tokens: {tokens_expected_ui:,.2f}M ({tokens_expected_raw:,} raw)")

            # Calculate curve state AFTER our buy (for sell transaction later)
            curve_state_after_buy = calculate_curve_state_after_buy(
                initial_curve_state,
                sol_lamports,
                tokens_expected_raw
            )

            async with self.wallet_manager.get_wallet(min_sol=self.buy_amount_sol + 0.01) as wallet:
                buy_signed = await self.direct_client.create_buy_transaction(
                    keypair=self.keypair,
                    mint_str=creation.mint,
                    sol_amount=self.buy_amount_sol,
                    slippage_bps=1000,
                    priority_fee_lamports=self.priority_fee_lamports,
                    compute_units=150_000,
                    curve_state=initial_curve_state,
                    bonding_curve_address=creation.bonding_curve,
                    associated_bonding_curve_address=creation.associated_bonding_curve,
                    creator_address=creation.creator
                )

                if not buy_signed:
                    raise RuntimeError("Buy transaction build failed")

                buy_confirmed = await self.tx_submitter.submit_and_confirm(
                    buy_signed,
                    timeout_seconds=30
                )

                if buy_confirmed.error:
                    raise RuntimeError(f"Buy failed: {buy_confirmed.error}")

                buy_slot = buy_confirmed.slot
                buy_signature = buy_confirmed.signature

                logger.info(f"   ✅ BUY CONFIRMED")
                logger.info(f"      Signature: {buy_signature[:16]}...")
                logger.info(f"      Slot: {buy_slot}")
                logger.info(f"      Tokens: {tokens_expected_ui:,.2f}M ({tokens_expected_raw:,} raw)")

            # Step 2: HOLD
            logger.info(f"\n2️⃣ HOLDING for {self.hold_seconds} seconds...")
            await asyncio.sleep(self.hold_seconds)

            # Step 3: SELL
            logger.info(f"\n3️⃣ SELLING all tokens...")

            async with self.wallet_manager.get_wallet(min_sol=0.01) as wallet:
                sell_signed = await self.direct_client.create_sell_transaction(
                    keypair=self.keypair,
                    mint_str=creation.mint,
                    token_amount=tokens_expected_ui,  # Use calculated amount from buy
                    slippage_bps=1000,
                    priority_fee_lamports=self.priority_fee_lamports,
                    compute_units=120_000,
                    curve_state=curve_state_after_buy,  # Simulated state after our buy
                    bonding_curve_address=creation.bonding_curve,
                    associated_bonding_curve_address=creation.associated_bonding_curve,
                    creator_address=creation.creator
                )

                if not sell_signed:
                    raise RuntimeError("Sell transaction build failed")

                sell_confirmed = await self.tx_submitter.submit_and_confirm(
                    sell_signed,
                    timeout_seconds=30
                )

                if sell_confirmed.error:
                    raise RuntimeError(f"Sell failed: {sell_confirmed.error}")

                sell_slot = sell_confirmed.slot
                sell_signature = sell_confirmed.signature

                logger.info(f"   ✅ SELL CONFIRMED")
                logger.info(f"      Signature: {sell_signature[:16]}...")
                logger.info(f"      Slot: {sell_slot}")

            end_time = datetime.now()
            actual_hold_seconds = (end_time - start_time).total_seconds()

            # Calculate P&L (simplified - check wallet balance change)
            final_balance = self.wallet_manager.get_total_balance_sol()

            # For now, estimate based on transaction fees and price action
            # In reality, we'd parse the transaction logs to get exact SOL received
            logger.info(f"\n📊 RESULTS:")
            logger.info(f"   Buy: {self.buy_amount_sol:.4f} SOL")
            logger.info(f"   Hold time: {actual_hold_seconds:.1f}s")
            logger.info(f"   Final balance: {final_balance:.4f} SOL")
            logger.info(f"\n   Buy sig: {buy_signature}")
            logger.info(f"   Sell sig: {sell_signature}")

            self.completed = True

        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            self.completed = True


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test buy-hold-sell cycle")
    parser.add_argument("--config", default="frontrun/config/config.yml", help="Config path")
    parser.add_argument("--wallet", required=True, help="Wallet keypair JSON")
    parser.add_argument("--buy-amount", type=float, default=0.005, help="SOL to buy")
    parser.add_argument("--hold-seconds", type=int, default=10, help="Seconds to hold")
    parser.add_argument("--priority-fee", type=int, default=100000, help="Priority fee lamports")
    parser.add_argument("--real-trades", action="store_true", help="Execute REAL trades")

    args = parser.parse_args()

    setup_logging(level="INFO", format="json", output_file="logs/buy_hold_sell_test.log")

    if args.real_trades:
        logger.warning("⚠️ REAL TRADES MODE ⚠️")
        logger.warning(f"Will buy {args.buy_amount} SOL, hold {args.hold_seconds}s, then sell")
        logger.warning("Press Ctrl+C within 5 seconds to cancel...")
        await asyncio.sleep(5)

    tester = BuyHoldSellTester(
        config_path=args.config,
        wallet_path=args.wallet,
        buy_amount_sol=args.buy_amount,
        hold_seconds=args.hold_seconds,
        priority_fee_lamports=args.priority_fee,
        enable_real_trades=args.real_trades
    )

    await tester.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted")
    except Exception as e:
        logger.error("test_failed", error=str(e))
        raise
