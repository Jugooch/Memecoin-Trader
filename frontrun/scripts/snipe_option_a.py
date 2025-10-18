"""
Option A: Snipe Everything Strategy

Snipes every new token creation with time-based exits.
- Max 1 concurrent position
- Buy on creation
- Hold for X seconds
- Sell
- Move to next token

Usage:
    python frontrun/scripts/snipe_option_a.py \
        --wallet frontrun/data/phase2-wallet.json \
        --buy-amount 0.5 \
        --hold-seconds 30 \
        --priority-fee 100000 \
        --max-trades 5 \
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
from core.bonding_curve import (
    get_initial_bonding_curve_state,
    BondingCurveCalculator,
    calculate_curve_state_after_buy
)
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
    """Result of a single trade"""
    mint: str
    symbol: str
    buy_signature: str
    buy_slot: int
    buy_sol: float
    sell_signature: str
    sell_slot: int
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


class OptionASniper:
    """Option A: Snipe everything with max 1 concurrent position"""

    def __init__(
        self,
        config_path: str,
        wallet_path: str,
        buy_amount_sol: float,
        hold_seconds: int,
        priority_fee_lamports: int,
        max_trades: int,
        enable_real_trades: bool
    ):
        self.buy_amount_sol = buy_amount_sol
        self.hold_seconds = hold_seconds
        self.priority_fee_lamports = priority_fee_lamports
        self.max_trades = max_trades
        self.enable_real_trades = enable_real_trades

        self.trades_completed = 0
        self.trades_results = []
        self.currently_trading = False  # Max 1 concurrent position
        self.should_stop = False

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
                config=WalletManagerConfig(cooldown_seconds=0, balance_cache_ttl_seconds=0)  # No cache - always fresh
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

    async def run(self):
        """Run the sniper"""
        logger.info("=== OPTION A SNIPER STARTED ===")
        logger.info(f"Buy amount: {self.buy_amount_sol} SOL")
        logger.info(f"Hold time: {self.hold_seconds} seconds")
        logger.info(f"Priority fee: {self.priority_fee_lamports / 1e9:.6f} SOL")
        logger.info(f"Max trades: {self.max_trades}")
        logger.info(f"Mode: {'REAL TRADES' if self.enable_real_trades else 'SIMULATION'}")
        logger.info("Waiting for token creations...\n")

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
            while not self.should_stop:
                await asyncio.sleep(1)

            logger.info(f"\n✅ Sniper complete!")
            logger.info(f"   Trades completed: {self.trades_completed}/{self.max_trades}")

            # Print summary
            successful = sum(1 for r in self.trades_results if r.success)
            logger.info(f"\n📊 SUMMARY:")
            logger.info(f"   Success rate: {successful}/{self.trades_completed} ({successful/self.trades_completed*100:.1f}%)")

            for i, result in enumerate(self.trades_results, 1):
                logger.info(f"\n   Trade {i}: {result.symbol}")
                logger.info(f"      Buy: {result.buy_signature[:16]}...")
                logger.info(f"      Sell: {result.sell_signature[:16]}...")
                logger.info(f"      Hold: {result.hold_seconds:.1f}s")
                logger.info(f"      Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
                if result.error:
                    logger.info(f"      Error: {result.error}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            await monitor.stop_monitoring()
            monitoring_task.cancel()

            if self.enable_real_trades:
                await self.wallet_manager.stop()
                await self.rpc_manager.stop()

    async def _on_token_creation(self, creation: TokenCreation):
        """Handle token creation - execute trade if not currently trading"""
        # Check if we should skip
        if self.currently_trading:
            logger.debug("token_skipped_position_active", mint=creation.mint[:16], symbol=creation.symbol)
            return

        if self.trades_completed >= self.max_trades:
            logger.info("max_trades_reached")
            self.should_stop = True
            return

        # Execute trade
        self.currently_trading = True

        logger.info(f"\n🎯 TOKEN {self.trades_completed + 1}/{self.max_trades}")
        logger.info(f"   Mint: {creation.mint}")
        logger.info(f"   Symbol: {creation.symbol}")
        logger.info(f"   Slot: {creation.slot}")

        if not self.enable_real_trades:
            logger.info("   ⏭️ Simulation mode - skipping")
            self.currently_trading = False
            return

        try:
            result = await self._execute_trade(creation)
            self.trades_results.append(result)
            self.trades_completed += 1

            if result.success:
                logger.info(f"   ✅ Trade {self.trades_completed} complete")
            else:
                logger.error(f"   ❌ Trade {self.trades_completed} failed: {result.error}")

        except Exception as e:
            logger.error(f"   ❌ Trade failed: {e}")
            import traceback
            traceback.print_exc()

            # Record failed trade
            self.trades_results.append(TradeResult(
                mint=creation.mint,
                symbol=creation.symbol,
                buy_signature="",
                buy_slot=0,
                buy_sol=self.buy_amount_sol,
                sell_signature="",
                sell_slot=0,
                hold_seconds=0,
                success=False,
                error=str(e)
            ))
            self.trades_completed += 1

        finally:
            self.currently_trading = False

    async def _execute_trade(self, creation: TokenCreation) -> TradeResult:
        """Execute buy-hold-sell cycle"""
        start_time = datetime.now()

        # Step 1: BUY
        logger.info(f"   1️⃣ BUYING {self.buy_amount_sol} SOL...")

        # Get initial bonding curve state
        initial_curve_state = get_initial_bonding_curve_state()

        # Calculate expected tokens
        calculator = BondingCurveCalculator()
        sol_lamports = int(self.buy_amount_sol * 1e9)
        buy_quote = calculator.calculate_buy_price(initial_curve_state, sol_lamports)
        tokens_expected_raw = int(buy_quote.tokens_out)
        tokens_expected_ui = tokens_expected_raw / 1e6

        # Calculate curve state after buy
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

            logger.info(f"      ✅ Buy confirmed: {buy_signature[:16]}...")

        # Step 2: HOLD
        logger.info(f"   2️⃣ HOLDING for {self.hold_seconds} seconds...")
        await asyncio.sleep(self.hold_seconds)

        # Step 3: SELL
        logger.info(f"   3️⃣ SELLING...")

        async with self.wallet_manager.get_wallet(min_sol=0.01) as wallet:
            sell_signed = await self.direct_client.create_sell_transaction(
                keypair=self.keypair,
                mint_str=creation.mint,
                token_amount=tokens_expected_ui,
                slippage_bps=1000,
                priority_fee_lamports=self.priority_fee_lamports,
                compute_units=120_000,
                curve_state=curve_state_after_buy,
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

            logger.info(f"      ✅ Sell confirmed: {sell_signature[:16]}...")

        end_time = datetime.now()
        hold_seconds = (end_time - start_time).total_seconds()

        return TradeResult(
            mint=creation.mint,
            symbol=creation.symbol,
            buy_signature=buy_signature,
            buy_slot=buy_slot,
            buy_sol=self.buy_amount_sol,
            sell_signature=sell_signature,
            sell_slot=sell_slot,
            hold_seconds=hold_seconds,
            success=True
        )


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Option A: Snipe everything with max 1 concurrent position")
    parser.add_argument("--config", default="frontrun/config/config.yml", help="Config path")
    parser.add_argument("--wallet", required=True, help="Wallet keypair JSON")
    parser.add_argument("--buy-amount", type=float, default=0.5, help="SOL to buy per trade")
    parser.add_argument("--hold-seconds", type=int, default=30, help="Seconds to hold each position")
    parser.add_argument("--priority-fee", type=int, default=100000, help="Priority fee lamports")
    parser.add_argument("--max-trades", type=int, default=5, help="Maximum trades before stopping")
    parser.add_argument("--real-trades", action="store_true", help="Execute REAL trades")

    args = parser.parse_args()

    setup_logging(level="INFO", format="json", output_file="logs/snipe_option_a.log")

    if args.real_trades:
        logger.warning("⚠️ REAL TRADES MODE ⚠️")
        logger.warning(f"Will snipe {args.max_trades} tokens with {args.buy_amount} SOL each")
        logger.warning(f"Hold time: {args.hold_seconds}s")
        logger.warning("Press Ctrl+C within 5 seconds to cancel...")
        await asyncio.sleep(5)

    sniper = OptionASniper(
        config_path=args.config,
        wallet_path=args.wallet,
        buy_amount_sol=args.buy_amount,
        hold_seconds=args.hold_seconds,
        priority_fee_lamports=args.priority_fee,
        max_trades=args.max_trades,
        enable_real_trades=args.real_trades
    )

    await sniper.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Sniper interrupted")
    except Exception as e:
        logger.error("sniper_failed", error=str(e))
        raise
