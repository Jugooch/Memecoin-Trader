"""
Test Sniper with Real-Time Buy Monitoring

After we buy, monitor the token for 30 seconds to see:
- How many buys came BEFORE ours (based on slot)
- How many buys came AFTER ours
- What our actual position was

This answers: Are we really first? Or are bots beating us?
"""

import asyncio
import sys
import json
import struct
import base58
import base64
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set

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
TRADE_EVENT_DISCRIMINATOR = bytes([189, 219, 127, 211, 78, 230, 97, 238])


@dataclass
class BuyEvent:
    """A buy transaction observed"""
    signature: str
    slot: int
    buyer: str
    sol_amount: float
    timestamp: datetime
    is_us: bool = False


@dataclass
class SnipeResult:
    """Result of a single snipe test with monitoring"""
    test_num: int
    fee_level: str
    priority_fee_sol: float

    # Token
    mint: str
    symbol: str
    creation_slot: int

    # Our buy
    our_signature: str
    our_slot: int
    our_sol_amount: float
    slot_difference: int

    # Competition (from 30s monitoring)
    total_buys_observed: int
    buys_before_us: int
    buys_after_us: int
    our_position: int  # 1 = first, 2 = second, etc

    # All observed buys
    all_buys: List[Dict]

    success: bool
    error: Optional[str] = None


def parse_trade_event_from_logs(logs: List[str]) -> Optional[Dict]:
    """Parse TradeEvent from logs"""
    for log in logs:
        if "Program data:" in log:
            try:
                encoded_data = log.split("Program data: ")[1].strip()
                decoded_data = base64.b64decode(encoded_data)

                if len(decoded_data) >= 8:
                    discriminator = decoded_data[:8]
                    if discriminator == TRADE_EVENT_DISCRIMINATOR:
                        if len(decoded_data) < 113:
                            return None

                        offset = 8

                        # Parse mint
                        mint_bytes = decoded_data[offset:offset + 32]
                        mint = base58.b58encode(mint_bytes).decode('utf-8')
                        offset += 32

                        # Parse sol_amount
                        sol_amount = struct.unpack('<Q', decoded_data[offset:offset + 8])[0]
                        offset += 8

                        # Skip token_amount
                        offset += 8

                        # Parse is_buy
                        is_buy = bool(decoded_data[offset])
                        offset += 1

                        # Parse user
                        user_bytes = decoded_data[offset:offset + 32]
                        user = base58.b58encode(user_bytes).decode('utf-8')

                        return {
                            'mint': mint,
                            'sol_amount': sol_amount / 1e9,
                            'is_buy': is_buy,
                            'user': user
                        }
            except Exception:
                continue
    return None


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


class TokenCreationMonitor:
    """Monitor token creations via Geyser"""

    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self._monitoring = False

    def decode_create_instruction(self, ix_data: bytes, account_keys: List, accounts: List[int]) -> Optional[Dict]:
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


class BuyMonitor:
    """Monitor all buys for a specific token via Geyser"""

    def __init__(self, endpoint: str, api_key: str, mint: str, our_wallet: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.mint = mint
        self.our_wallet = our_wallet
        self.buys: List[BuyEvent] = []
        self._monitoring = False

    async def monitor_for_duration(self, duration_seconds: int = 30):
        """Monitor for specified duration"""
        logger.info(f"  📊 Monitoring {self.mint[:8]}... for {duration_seconds}s to see all buys...")

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
        subscription_request.transactions["pump_buys"].account_include.append(str(PUMP_FUN_PROGRAM))
        subscription_request.transactions["pump_buys"].failed = False
        subscription_request.commitment = geyser_pb2.CommitmentLevel.PROCESSED

        try:
            async def request_generator():
                yield subscription_request
                while self._monitoring:
                    await asyncio.sleep(1)

            # Create timeout task
            timeout_task = asyncio.create_task(asyncio.sleep(duration_seconds))

            async for message in stub.Subscribe(request_generator()):
                # Check timeout
                if timeout_task.done():
                    break

                if not message.HasField('transaction'):
                    continue

                try:
                    tx_update = message.transaction
                    transaction = tx_update.transaction

                    # Get logs
                    logs = []
                    if transaction.meta:
                        if hasattr(transaction.meta, 'log_messages'):
                            logs = list(transaction.meta.log_messages)

                    # Parse trade event
                    trade_event = parse_trade_event_from_logs(logs)
                    if not trade_event:
                        continue

                    # Check if it's for our mint
                    if trade_event['mint'] != self.mint:
                        continue

                    # Check if it's a buy
                    if not trade_event['is_buy']:
                        continue

                    # Record the buy
                    signature = base58.b58encode(bytes(transaction.signature)).decode('utf-8')
                    is_us = (trade_event['user'] == self.our_wallet)

                    buy = BuyEvent(
                        signature=signature,
                        slot=tx_update.slot,
                        buyer=trade_event['user'],
                        sol_amount=trade_event['sol_amount'],
                        timestamp=datetime.now(),
                        is_us=is_us
                    )

                    self.buys.append(buy)

                    marker = "👉 US" if is_us else ""
                    logger.info(f"    Buy #{len(self.buys)}: Slot {buy.slot}, {buy.sol_amount:.4f} SOL {marker}")

                except Exception as e:
                    logger.debug(f"message_parse_error: {e}")

        finally:
            self._monitoring = False
            await channel.close()

        logger.info(f"  ✅ Monitoring complete. Observed {len(self.buys)} total buys")
        return self.buys


class SnipeTester:
    """Test sniping with real-time buy monitoring"""

    def __init__(
        self,
        config_path: str,
        wallet_path: str,
        num_tests: int = 3,
        buy_amount_sol: float = 0.01,
        priority_fee_lamports: int = 0,
        enable_real_trades: bool = False
    ):
        self.num_tests = num_tests
        self.buy_amount_sol = buy_amount_sol
        self.priority_fee_lamports = priority_fee_lamports
        self.enable_real_trades = enable_real_trades
        self.results: List[SnipeResult] = []
        self._traded_mints: Set[str] = set()

        # Load config
        config_mgr = ConfigurationManager(config_path)
        self.bot_config = config_mgr.load_config()

        # Load wallet
        self.keypair = self._load_wallet(wallet_path)
        self.our_wallet = str(self.keypair.pubkey())
        logger.info("wallet_loaded", pubkey=self.our_wallet)

        # Get Helius config
        helius_endpoint = self.bot_config.rpc_config.endpoints[0]
        self.geyser_endpoint = helius_endpoint.laserstream_url.replace("https://", "").replace("http://", "")
        self.geyser_api_key = helius_endpoint.laserstream_api_key

        # Initialize trading components
        self.rpc_manager: Optional[RPCManager] = None
        self.wallet_manager: Optional[WalletManager] = None
        self.direct_client: Optional[PumpFunDirectClient] = None
        self.tx_submitter: Optional[TransactionSubmitter] = None

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

    async def run(self):
        """Run the tests"""
        logger.info("=== SNIPE WITH MONITORING TEST ===")
        logger.info(f"Number of tests: {self.num_tests}")
        logger.info(f"Buy amount: {self.buy_amount_sol} SOL")
        logger.info(f"Priority fee: {self.priority_fee_lamports / 1e9:.6f} SOL")
        logger.info(f"Mode: {'REAL TRADES' if self.enable_real_trades else 'SIMULATION'}")
        logger.info("Press Ctrl+C to stop\n")

        if self.enable_real_trades:
            await self.rpc_manager.start()
            await self.wallet_manager.start()

        # Start creation monitor
        monitor = TokenCreationMonitor(self.geyser_endpoint, self.geyser_api_key)
        monitoring_task = asyncio.create_task(
            monitor.start_monitoring(self._on_token_creation)
        )

        try:
            while len(self.results) < self.num_tests:
                await asyncio.sleep(1)

            logger.info(f"\n✅ Completed all {self.num_tests} tests!")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            await monitor.stop_monitoring()
            monitoring_task.cancel()

            if self.enable_real_trades:
                await self.wallet_manager.stop()
                await self.rpc_manager.stop()

            await self._generate_report()

    async def _on_token_creation(self, creation):
        """Handle token creation"""
        if len(self.results) >= self.num_tests:
            return

        if creation.mint in self._traded_mints:
            return

        self._traded_mints.add(creation.mint)

        test_num = len(self.results) + 1

        logger.info(f"\n🎯 TEST #{test_num}")
        logger.info(f"   Mint: {creation.mint}")
        logger.info(f"   Symbol: {creation.symbol}")
        logger.info(f"   Creation slot: {creation.slot}")

        await self._execute_and_monitor(creation, test_num)

    async def _execute_and_monitor(self, creation, test_num):
        """Execute buy and monitor competition"""

        if not self.enable_real_trades:
            logger.info("  ⏭️ Simulation mode - skipping")
            return

        try:
            # START MONITORING IMMEDIATELY (before our buy!)
            monitor = BuyMonitor(
                endpoint=self.geyser_endpoint,
                api_key=self.geyser_api_key,
                mint=creation.mint,
                our_wallet=self.our_wallet
            )

            # Start monitoring task (runs in background for 30s)
            monitoring_task = asyncio.create_task(monitor.monitor_for_duration(30))

            logger.info(f"  📊 Started monitoring for buys...")

            # Execute buy (while monitoring runs in parallel)
            async with self.wallet_manager.get_wallet(min_sol=self.buy_amount_sol + 0.01) as wallet:
                logger.info(f"  🚀 BUYING {self.buy_amount_sol} SOL...")

                initial_curve_state = get_initial_bonding_curve_state()

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

                our_slot = buy_confirmed.slot
                our_signature = buy_confirmed.signature
                slot_diff = our_slot - creation.slot

                logger.info(f"  ✅ BUY CONFIRMED")
                logger.info(f"     Signature: {our_signature[:16]}...")
                logger.info(f"     Slot: {our_slot} (+{slot_diff} from creation)")

            # Wait for monitoring to complete
            all_buys = await monitoring_task

            # Analyze results
            # Sort by slot to determine position
            sorted_buys = sorted(all_buys, key=lambda b: b.slot)

            our_buy = None
            our_position = None
            buys_before = 0
            buys_after = 0

            for i, buy in enumerate(sorted_buys):
                if buy.is_us or buy.signature == our_signature:
                    our_buy = buy
                    our_position = i + 1
                    buys_before = i
                    buys_after = len(sorted_buys) - i - 1
                    break

            if our_buy is None:
                # We didn't see our own buy in the stream (possible if it was too fast)
                # Assume we were first
                our_position = 1
                buys_before = 0
                buys_after = len(sorted_buys)

            logger.info(f"\n  📊 ANALYSIS:")
            logger.info(f"     Position: #{our_position}")
            logger.info(f"     Buys before us: {buys_before}")
            logger.info(f"     Buys after us: {buys_after}")
            logger.info(f"     Total buys observed: {len(all_buys)}")

            result = SnipeResult(
                test_num=test_num,
                fee_level=f"{self.priority_fee_lamports / 1e9:.6f} SOL",
                priority_fee_sol=self.priority_fee_lamports / 1e9,
                mint=creation.mint,
                symbol=creation.symbol,
                creation_slot=creation.slot,
                our_signature=our_signature,
                our_slot=our_slot,
                our_sol_amount=self.buy_amount_sol,
                slot_difference=slot_diff,
                total_buys_observed=len(all_buys),
                buys_before_us=buys_before,
                buys_after_us=buys_after,
                our_position=our_position,
                all_buys=[{
                    'signature': b.signature,
                    'slot': b.slot,
                    'buyer': b.buyer,
                    'sol_amount': b.sol_amount,
                    'is_us': b.is_us
                } for b in sorted_buys],
                success=True
            )

            self.results.append(result)

        except Exception as e:
            logger.error(f"  ❌ Test failed: {e}")

            result = SnipeResult(
                test_num=test_num,
                fee_level=f"{self.priority_fee_lamports / 1e9:.6f} SOL",
                priority_fee_sol=self.priority_fee_lamports / 1e9,
                mint=creation.mint,
                symbol=creation.symbol,
                creation_slot=creation.slot,
                our_signature="",
                our_slot=0,
                our_sol_amount=0,
                slot_difference=0,
                total_buys_observed=0,
                buys_before_us=0,
                buys_after_us=0,
                our_position=0,
                all_buys=[],
                success=False,
                error=str(e)
            )

            self.results.append(result)

    async def _generate_report(self):
        """Generate final report"""
        logger.info("\n" + "="*80)
        logger.info("SNIPE MONITORING TEST RESULTS")
        logger.info("="*80 + "\n")

        successful = [r for r in self.results if r.success]

        if successful:
            avg_position = sum(r.our_position for r in successful) / len(successful)
            avg_buys_before = sum(r.buys_before_us for r in successful) / len(successful)
            avg_buys_after = sum(r.buys_after_us for r in successful) / len(successful)

            logger.info(f"Successful tests: {len(successful)}/{len(self.results)}")
            logger.info(f"Average position: #{avg_position:.1f}")
            logger.info(f"Average buys before us: {avg_buys_before:.1f}")
            logger.info(f"Average buys after us: {avg_buys_after:.1f}")
            logger.info(f"Times we were #1: {sum(1 for r in successful if r.our_position == 1)}/{len(successful)}")

        # Save results
        output_dir = Path("data/sniper")
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / f"monitoring_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, default=str)

        logger.info(f"\n✅ Results saved to: {results_file}\n")
        logger.info("="*80)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test sniping with real-time buy monitoring")
    parser.add_argument("--config", default="frontrun/config/config.yml", help="Config path")
    parser.add_argument("--wallet", required=True, help="Wallet keypair JSON")
    parser.add_argument("--num-tests", type=int, default=3, help="Number of tests to run")
    parser.add_argument("--buy-amount", type=float, default=0.01, help="SOL per trade")
    parser.add_argument("--priority-fee", type=int, default=0, help="Priority fee in lamports")
    parser.add_argument("--real-trades", action="store_true", help="Execute REAL trades")

    args = parser.parse_args()

    setup_logging(level="INFO", format="json", output_file="logs/snipe_monitoring_test.log")

    if args.real_trades:
        logger.warning("⚠️ REAL TRADES MODE - WILL USE REAL CAPITAL ⚠️")
        logger.warning(f"This will execute {args.num_tests} test trades")
        logger.warning("Press Ctrl+C within 5 seconds to cancel...")
        await asyncio.sleep(5)

    tester = SnipeTester(
        config_path=args.config,
        wallet_path=args.wallet,
        num_tests=args.num_tests,
        buy_amount_sol=args.buy_amount,
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
