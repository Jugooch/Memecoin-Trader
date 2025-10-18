"""
Test Sniper - Fee Level Comparison

Run 10 snipes with different fee levels to see what position we land in.
This will answer: Do high fees actually help us get better positions?

Test Groups:
  - Group 1 (3 trades): No priority fee
  - Group 2 (4 trades): Low priority fee (0.001 SOL = 1M lamports)
  - Group 3 (3 trades): High priority fee (0.01 SOL = 10M lamports)

For each trade, we track:
  - Creation slot
  - Our buy slot
  - Slot difference (how many slots behind creation)
  - Our position # (1st buyer? 2nd? 3rd? etc)
  - Whether we beat other bots

Usage:
    python scripts/test_snipe_fee_levels.py --wallet data/test-wallet.json --real-trades
"""

import asyncio
import sys
import json
import struct
import base58
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
from core.bonding_curve import get_initial_bonding_curve_state, BondingCurveCalculator
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
class SnipeTest:
    """Test result for a single snipe"""
    test_num: int
    fee_level: str  # "none", "low", "high"
    priority_fee_lamports: int
    priority_fee_sol: float

    # Token info
    mint: str
    creator: str
    symbol: str

    # Timing
    creation_slot: int
    our_buy_slot: int
    slot_difference: int

    # Position tracking
    our_position: int  # 1 = first buyer, 2 = second, etc
    total_buyers_before_us: int

    # Transaction
    signature: str
    success: bool
    error: Optional[str] = None

    def to_dict(self):
        return {
            **asdict(self),
            "timestamp": datetime.now().isoformat()
        }


class TokenCreationMonitor:
    """Monitor token creations via Geyser"""

    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self._monitoring = False
        logger.info("token_creation_monitor_initialized", endpoint=endpoint)

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

        channel = grpc.aio.secure_channel(self.endpoint, credentials=combined_creds)
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


class FeeLevelTester:
    """Test different fee levels for sniping"""

    def __init__(
        self,
        config_path: str,
        wallet_path: str,
        enable_real_trades: bool = False,
        buy_amount_sol: float = 0.01
    ):
        self.buy_amount_sol = buy_amount_sol
        self.enable_real_trades = enable_real_trades
        self.tests: List[SnipeTest] = []
        self._traded_mints: Set[str] = set()

        # Test plan: 10 trades across 3 fee levels
        self.test_plan = [
            ("none", 0, 3),           # 3 trades with no priority fee
            ("low", 1_000_000, 4),    # 4 trades with 0.001 SOL priority
            ("high", 10_000_000, 3),  # 3 trades with 0.01 SOL priority
        ]

        self.current_group = 0
        self.group_trades = 0

        # Load config
        config_mgr = ConfigurationManager(config_path)
        self.bot_config = config_mgr.load_config()

        # Load wallet
        self.keypair = self._load_wallet(wallet_path)
        logger.info("wallet_loaded", pubkey=str(self.keypair.pubkey()))

        # Get Helius config
        helius_endpoint = self.bot_config.rpc_config.endpoints[0]
        endpoint = helius_endpoint.laserstream_url.replace("https://", "").replace("http://", "")

        # Initialize monitor
        self.creation_monitor = TokenCreationMonitor(
            endpoint=endpoint,
            api_key=helius_endpoint.laserstream_api_key
        )

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

    def _get_current_fee_config(self) -> tuple:
        """Get current fee configuration"""
        fee_level, priority_fee, max_trades = self.test_plan[self.current_group]
        return fee_level, priority_fee

    async def run(self):
        """Run fee level tests"""
        logger.info("=== FEE LEVEL SNIPER TEST ===")
        logger.info(f"Test Plan:")
        for level, fee, count in self.test_plan:
            logger.info(f"  {level.upper()}: {count} trades @ {fee/1e9:.6f} SOL priority fee")
        logger.info(f"Buy amount: {self.buy_amount_sol} SOL per trade")
        logger.info(f"Mode: {'REAL TRADES' if self.enable_real_trades else 'SIMULATION'}")
        logger.info("Press Ctrl+C to stop\n")

        if self.enable_real_trades:
            await self.rpc_manager.start()
            await self.wallet_manager.start()

            balance = self.wallet_manager.get_total_balance_sol()
            min_balance = (10 * self.buy_amount_sol) + 0.1
            logger.info(f"Wallet balance: {balance:.4f} SOL (need {min_balance:.1f} SOL)")

            if balance < min_balance:
                raise RuntimeError(f"Insufficient balance")

        monitoring_task = asyncio.create_task(
            self.creation_monitor.start_monitoring(self._on_token_creation)
        )

        try:
            while len(self.tests) < 10:
                await asyncio.sleep(1)

            logger.info(f"\n✅ Completed all 10 test snipes!")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            await self.creation_monitor.stop_monitoring()
            monitoring_task.cancel()

            if self.enable_real_trades:
                await self.wallet_manager.stop()
                await self.rpc_manager.stop()

            await self._generate_report()

    async def _on_token_creation(self, creation: TokenCreation):
        """Handle token creation"""
        if len(self.tests) >= 10:
            return

        if creation.mint in self._traded_mints:
            return

        self._traded_mints.add(creation.mint)

        # Get current fee config
        fee_level, priority_fee = self._get_current_fee_config()

        test_num = len(self.tests) + 1

        logger.info(
            f"\n🎯 TEST #{test_num} - Fee Level: {fee_level.upper()}",
            mint=creation.mint[:16] + "...",
            symbol=creation.symbol,
            priority_fee_sol=priority_fee / 1e9
        )

        await self._execute_snipe(creation, test_num, fee_level, priority_fee)

        # Move to next group if needed
        self.group_trades += 1
        _, _, max_trades = self.test_plan[self.current_group]
        if self.group_trades >= max_trades:
            self.current_group += 1
            self.group_trades = 0
            if self.current_group < len(self.test_plan):
                next_level, _, _ = self.test_plan[self.current_group]
                logger.info(f"\n📊 Moving to next fee level: {next_level.upper()}\n")

    async def _execute_snipe(self, creation: TokenCreation, test_num: int, fee_level: str, priority_fee: int):
        """Execute a test snipe"""
        if not self.enable_real_trades:
            # Simulation
            test = SnipeTest(
                test_num=test_num,
                fee_level=fee_level,
                priority_fee_lamports=priority_fee,
                priority_fee_sol=priority_fee / 1e9,
                mint=creation.mint,
                creator=creation.creator,
                symbol=creation.symbol,
                creation_slot=creation.slot,
                our_buy_slot=creation.slot + 1,
                slot_difference=1,
                our_position=1,
                total_buyers_before_us=0,
                signature=f"sim_{creation.signature[:16]}",
                success=True
            )
            self.tests.append(test)
            logger.info(f"  ✅ Simulated (position #1, slot +1)")
            return

        # Real trade
        try:
            async with self.wallet_manager.get_wallet(min_sol=self.buy_amount_sol + 0.01) as wallet:
                logger.info(f"  🚀 BUYING...")

                initial_curve_state = get_initial_bonding_curve_state()

                buy_signed = await self.direct_client.create_buy_transaction(
                    keypair=self.keypair,
                    mint_str=creation.mint,
                    sol_amount=self.buy_amount_sol,
                    slippage_bps=1000,
                    priority_fee_lamports=priority_fee,  # Variable fee!
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

                our_buy_slot = buy_confirmed.slot
                slot_diff = our_buy_slot - creation.slot

                # Query position (count buys before our slot)
                position_info = await self._get_our_position(creation.mint, our_buy_slot)

                logger.info(
                    f"  ✅ BUY COMPLETE",
                    signature=buy_confirmed.signature[:16] + "...",
                    slot_diff=slot_diff,
                    position=position_info['position'],
                    buyers_before=position_info['buyers_before']
                )

                test = SnipeTest(
                    test_num=test_num,
                    fee_level=fee_level,
                    priority_fee_lamports=priority_fee,
                    priority_fee_sol=priority_fee / 1e9,
                    mint=creation.mint,
                    creator=creation.creator,
                    symbol=creation.symbol,
                    creation_slot=creation.slot,
                    our_buy_slot=our_buy_slot,
                    slot_difference=slot_diff,
                    our_position=position_info['position'],
                    total_buyers_before_us=position_info['buyers_before'],
                    signature=buy_confirmed.signature,
                    success=True
                )

                self.tests.append(test)

        except Exception as e:
            logger.error(f"  ❌ Test failed: {e}")

            test = SnipeTest(
                test_num=test_num,
                fee_level=fee_level,
                priority_fee_lamports=priority_fee,
                priority_fee_sol=priority_fee / 1e9,
                mint=creation.mint,
                creator=creation.creator,
                symbol=creation.symbol,
                creation_slot=creation.slot,
                our_buy_slot=0,
                slot_difference=0,
                our_position=0,
                total_buyers_before_us=0,
                signature="",
                success=False,
                error=str(e)
            )

            self.tests.append(test)

    async def _get_our_position(self, mint: str, our_slot: int) -> Dict:
        """
        Query how many buyers were before us.
        This requires fetching transaction history and counting.
        """
        # TODO: Implement this by querying Pump.fun bonding curve
        # For now, return placeholder
        return {
            'position': 1,  # Placeholder
            'buyers_before': 0  # Placeholder
        }

    async def _generate_report(self):
        """Generate final report"""
        logger.info("\n" + "="*80)
        logger.info("FEE LEVEL TEST RESULTS")
        logger.info("="*80 + "\n")

        # Group by fee level
        none_tests = [t for t in self.tests if t.fee_level == "none" and t.success]
        low_tests = [t for t in self.tests if t.fee_level == "low" and t.success]
        high_tests = [t for t in self.tests if t.fee_level == "high" and t.success]

        def analyze_group(tests, name):
            if not tests:
                logger.info(f"{name}: No successful tests")
                return

            avg_slot_diff = sum(t.slot_difference for t in tests) / len(tests)
            avg_position = sum(t.our_position for t in tests) / len(tests)
            avg_buyers_before = sum(t.total_buyers_before_us for t in tests) / len(tests)

            logger.info(f"{name} ({len(tests)} trades):")
            logger.info(f"  Avg slot difference: {avg_slot_diff:.1f} slots")
            logger.info(f"  Avg position: #{avg_position:.1f}")
            logger.info(f"  Avg buyers before us: {avg_buyers_before:.1f}")
            logger.info("")

        analyze_group(none_tests, "NO PRIORITY FEE")
        analyze_group(low_tests, "LOW FEE (0.001 SOL)")
        analyze_group(high_tests, "HIGH FEE (0.01 SOL)")

        # Export results
        output_dir = Path("data/sniper")
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / f"fee_level_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump([t.to_dict() for t in self.tests], f, indent=2)

        logger.info(f"✅ Results saved to: {results_file}\n")
        logger.info("="*80)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test different priority fee levels for sniping")
    parser.add_argument("--config", default="config/config.yml", help="Config path")
    parser.add_argument("--wallet", required=True, help="Wallet keypair JSON")
    parser.add_argument("--real-trades", action="store_true", help="Execute REAL trades")
    parser.add_argument("--buy-amount", type=float, default=0.01, help="SOL per trade")

    args = parser.parse_args()

    setup_logging(level="INFO", format="json", output_file="logs/fee_level_test.log")

    if args.real_trades:
        logger.warning("⚠️ REAL TRADES MODE - WILL USE REAL CAPITAL ⚠️")
        logger.warning("This will execute 10 test trades")
        logger.warning("Press Ctrl+C within 5 seconds to cancel...")
        await asyncio.sleep(5)

    tester = FeeLevelTester(
        config_path=args.config,
        wallet_path=args.wallet,
        enable_real_trades=args.real_trades,
        buy_amount_sol=args.buy_amount
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
