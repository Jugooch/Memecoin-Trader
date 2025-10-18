#!/usr/bin/env python3
"""
Anti-Bot Wallet Classifier and Environment Analyzer

Monitors Laserstream in real-time to classify every wallet and detect:
- Sybil/wash trading clusters
- Metric-triggered bots
- Copy trading bots
- Priority fee racers
- Organic traders
- Bait deployer clusters

Usage:
    python anti-bot/analyzer.py --hours 6 --config config.yml
"""

import asyncio
import json
import sys
import argparse
import statistics
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import grpc
import base58

# Import proto files
proto_path = Path(__file__).parent.parent / "frontrun" / "proto"
sys.path.insert(0, str(proto_path))

try:
    import geyser_pb2
    import geyser_pb2_grpc
    GEYSER_AVAILABLE = True
except ImportError:
    GEYSER_AVAILABLE = False
    print("❌ Geyser proto files not available")
    sys.exit(1)

from src.utils.logger_setup import setup_logging
from src.utils.config_loader import load_config

from anti_bot.core.types import WalletLabel
from anti_bot.core.feature_extractor import extract_features_from_geyser_tx
from anti_bot.core.wallet_classifier import WalletClassifier
from anti_bot.core.simple_rpc_manager import SimpleRPCManager
from anti_bot.core.bot_profiler import BotProfiler

import logging
logger = logging.getLogger(__name__)


PUMP_FUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"


class HeliusAuthMetadata(grpc.AuthMetadataPlugin):
    """Authentication plugin for Helius LaserStream"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def __call__(self, context, callback):
        metadata = (('x-token', self.api_key),)
        callback(metadata, None)


class AntiBotAnalyzer:
    """Real-time wallet classifier from Laserstream"""

    def __init__(self, config: dict):
        if not GEYSER_AVAILABLE:
            raise RuntimeError("Geyser proto files not available")

        # Configuration
        self.grpc_endpoint = config.get('helius_grpc_endpoint')
        self.grpc_token = config.get('helius_grpc_token')

        # Use Helius HTTP RPC (not public Solana RPC - will get rate limited!)
        # Convert gRPC endpoint to HTTP RPC endpoint
        if 'helius' in self.grpc_endpoint.lower():
            # Extract base domain and use Helius HTTP RPC
            self.rpc_endpoint = f"https://mainnet.helius-rpc.com?api-key={self.grpc_token}"
        else:
            # Fallback to config or Helius default
            self.rpc_endpoint = config.get('rpc_endpoint', f"https://mainnet.helius-rpc.com?api-key={self.grpc_token}")

        if not self.grpc_endpoint or not self.grpc_token:
            raise ValueError("helius_grpc_endpoint and helius_grpc_token required")

        # Strip https:// prefix if present (for gRPC)
        if self.grpc_endpoint.startswith('https://'):
            self.grpc_endpoint = self.grpc_endpoint.replace('https://', '')
        elif self.grpc_endpoint.startswith('http://'):
            self.grpc_endpoint = self.grpc_endpoint.replace('http://', '')

        # Add :443 port if not specified
        if ':' not in self.grpc_endpoint:
            self.grpc_endpoint = f"{self.grpc_endpoint}:443"

        # Connection state
        self.channel = None
        self.stub = None
        self.running = False

        # RPC Manager for funding graph (smart trigger-based lookups)
        self.rpc_manager = SimpleRPCManager(
            rpc_endpoint=self.rpc_endpoint,
            max_concurrent=10  # Limit concurrent RPC calls
        )

        # Classifier with production-ready components + RPC manager
        self.classifier = WalletClassifier(rpc_manager=self.rpc_manager)

        # Bot profiler for detailed analysis and strategy recommendations
        self.profiler = BotProfiler()

        # Statistics
        self.total_txs_seen = 0
        self.total_trades_seen = 0
        self.start_time = None

        logger.info("AntiBotAnalyzer initialized (Production-Ready v2 + Smart RPC)")
        logger.info(f"  Geyser Endpoint: {self.grpc_endpoint}")
        logger.info(f"  RPC Endpoint: {self.rpc_endpoint}")
        logger.info(f"  Features: Adaptive thresholds, smart funding graph, feature-based scoring")

    async def start(self, run_hours: float):
        """Start monitoring and classification"""
        if self.running:
            logger.warning("Analyzer already running")
            return

        self.running = True
        self.start_time = datetime.now()

        logger.info(f"\n{'='*80}")
        logger.info("ANTI-BOT WALLET CLASSIFIER STARTING (Production-Ready v2)")
        logger.info(f"{'='*80}")
        logger.info(f"Run duration: {run_hours} hours")
        logger.info(f"Monitoring: ALL Pump.fun transactions")
        logger.info(f"Goal: Classify EVERY wallet and detect bot patterns")
        logger.info(f"\n📊 Production Features Enabled:")
        logger.info(f"  ✓ Adaptive burst thresholds (rolling p95 per cohort)")
        logger.info(f"  ✓ Feature-based confidence scoring (not len(reasons) * constant)")
        logger.info(f"  ✓ Inter-arrival CV for bot timing detection")
        logger.info(f"  ✓ Smallest-k mass for randomized sybil detection")
        logger.info(f"  ✓ SMART funding graph (trigger-based RPC lookups)")
        logger.info(f"\n💡 Smart RPC Optimization:")
        logger.info(f"  • Only queries RPC for wallets showing suspicious patterns")
        logger.info(f"  • Triggers: micro-buys, coordinated bursts, synchronized sells")
        logger.info(f"  • Estimated: ~500-2000 RPC calls over 6h (vs 216,000 brute-force)")
        logger.info(f"{'='*80}\n")

        # Start RPC manager
        await self.rpc_manager.start()
        logger.info("✅ RPC manager started")

        # Start funding graph worker
        await self.classifier.start_funding_graph_worker()
        logger.info("✅ Funding graph worker started\n")

        # Start LaserStream monitoring
        monitor_task = asyncio.create_task(self._connect_and_stream())

        # Periodic stats reporting
        stats_task = asyncio.create_task(self._report_stats_loop())

        # Periodic checkpointing (save every 15 min)
        checkpoint_task = asyncio.create_task(self._checkpoint_loop())

        # Wait for specified time
        try:
            await asyncio.sleep(run_hours * 3600)
        except KeyboardInterrupt:
            logger.info("\n\n⚠️ Interrupted by user")
        finally:
            # ALWAYS save results, even if interrupted
            logger.info("\n\n⏱️ Stopping and saving results...")
            await self.stop()

            # Cancel tasks
            monitor_task.cancel()
            stats_task.cancel()
            checkpoint_task.cancel()

            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            try:
                await stats_task
            except asyncio.CancelledError:
                pass
            try:
                await checkpoint_task
            except asyncio.CancelledError:
                pass

            # Print final statistics (also saves results)
            self._print_final_statistics()

    async def _connect_and_stream(self):
        """Connect to LaserStream with auto-reconnect on failure"""
        max_retries = 10
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            if not self.running:
                break

            try:
                logger.info(f"Connecting to LaserStream at {self.grpc_endpoint}... (attempt {attempt + 1}/{max_retries})")

                # Setup authentication
                auth = HeliusAuthMetadata(self.grpc_token)
                call_creds = grpc.metadata_call_credentials(auth)
                ssl_creds = grpc.ssl_channel_credentials()
                combined_creds = grpc.composite_channel_credentials(ssl_creds, call_creds)

                # Create secure channel
                self.channel = grpc.aio.secure_channel(
                    self.grpc_endpoint,
                    credentials=combined_creds,
                    options=[
                        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                        ('grpc.keepalive_time_ms', 30000),
                        ('grpc.keepalive_timeout_ms', 10000),
                    ]
                )

                self.stub = geyser_pb2_grpc.GeyserStub(self.channel)

                # Build subscription request
                subscription_request = self._build_subscription_request()

                logger.info("✅ LaserStream connection established")
                logger.info("📡 Observing ALL Pump.fun transactions...\n")

                async def request_generator():
                    yield subscription_request
                    while self.running:
                        await asyncio.sleep(30)

                async for message in self.stub.Subscribe(request_generator()):
                    if not self.running:
                        break

                    try:
                        await self._process_message(message)
                    except Exception as e:
                        logger.error(f"Message processing error: {e}")

                # If we get here, stream ended normally
                if self.running:
                    logger.warning("Stream ended unexpectedly, reconnecting...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    break

            except asyncio.CancelledError:
                logger.info("LaserStream monitoring cancelled")
                break
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    logger.warning(f"⚠️ Server restart detected, reconnecting in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"LaserStream RPC error: {e}")
                    await asyncio.sleep(retry_delay)
                    continue
            except Exception as e:
                logger.error(f"LaserStream error: {e}")
                await asyncio.sleep(retry_delay)
                continue
            finally:
                if self.channel:
                    try:
                        await self.channel.close()
                    except:
                        pass

        if attempt == max_retries - 1:
            logger.error("❌ Max reconnection attempts reached, giving up")

    def _build_subscription_request(self) -> geyser_pb2.SubscribeRequest:
        """Build subscription request for ALL Pump.fun transactions"""
        commitment = geyser_pb2.CommitmentLevel.PROCESSED

        tx_filter = geyser_pb2.SubscribeRequestFilterTransactions(
            account_include=[PUMP_FUN_PROGRAM_ID],
            vote=False,
            failed=False,
        )

        request = geyser_pb2.SubscribeRequest(
            transactions={
                "pump_fun_all_txs": tx_filter
            },
            commitment=commitment
        )

        return request

    async def _process_message(self, message: geyser_pb2.SubscribeUpdate):
        """Process incoming LaserStream message"""
        if not message.HasField('transaction'):
            return

        self.total_txs_seen += 1

        tx_update = message.transaction
        transaction = tx_update.transaction

        # Extract features
        features = extract_features_from_geyser_tx(tx_update, transaction)

        if features and features.mint:
            self.total_trades_seen += 1

            # Process through classifier
            self.classifier.process_transaction(features)

            # Log interesting classifications
            label, confidence, reasons = self.classifier.get_wallet_label(features.signer)
            if label != WalletLabel.UNKNOWN and label != WalletLabel.ORGANIC and confidence >= 0.7:
                logger.info(f"🎯 {label.value.upper()}: {features.signer[:8]}... "
                          f"(confidence: {confidence:.2f}) - {', '.join(reasons)}")

    async def _report_stats_loop(self):
        """Periodically report statistics"""
        try:
            while self.running:
                await asyncio.sleep(60)  # Every minute
                self._print_live_statistics()
        except asyncio.CancelledError:
            pass

    async def _checkpoint_loop(self):
        """Periodically save checkpoint to prevent data loss"""
        try:
            while self.running:
                await asyncio.sleep(900)  # Every 15 minutes
                logger.info("\n💾 Saving checkpoint...")
                self._save_checkpoint()
                logger.info("✅ Checkpoint saved\n")
        except asyncio.CancelledError:
            pass

    def _print_live_statistics(self):
        """Print live statistics"""
        runtime = (datetime.now() - self.start_time).total_seconds() / 60  # minutes

        # Trigger periodic re-classification for wallets with new funding data
        self.classifier.periodic_reclassification()

        metrics = self.classifier.get_statistics()

        logger.info(f"\n{'='*60}")
        logger.info(f"LIVE STATS ({runtime:.1f} min)")
        logger.info(f"{'='*60}")
        logger.info(f"Transactions seen: {self.total_txs_seen:,}")
        logger.info(f"Trades seen: {self.total_trades_seen:,}")
        logger.info(f"Wallets tracked: {metrics.total_wallets_tracked:,}")
        logger.info(f"Mints tracked: {metrics.total_mints_tracked:,}")

        logger.info(f"\nLabel Distribution:")
        for label, count in sorted(metrics.label_counts.items(), key=lambda x: -x[1]):
            pct = (count / metrics.total_wallets_tracked * 100) if metrics.total_wallets_tracked > 0 else 0
            logger.info(f"  {label.value:20s}: {count:6d} ({pct:5.1f}%)")

        logger.info(f"\nDetections:")
        logger.info(f"  Sybil clusters: {metrics.sybil_clusters_found}")
        logger.info(f"  Metric bots: {metrics.metric_bots_found}")
        logger.info(f"  Bait clusters: {metrics.bait_clusters_found}")

        # Funding graph stats
        fg_stats = self.classifier.funding_graph.get_stats()
        logger.info(f"\nFunding Graph (Smart Trigger Mode):")
        logger.info(f"  Suspicious wallets flagged: {fg_stats['suspicious_flagged']}")
        logger.info(f"  RPC calls made: {fg_stats['rpc_calls']}")
        logger.info(f"  RPC calls avoided: {fg_stats['rpc_calls_avoided']}")
        logger.info(f"  Efficiency: {fg_stats['efficiency']}")
        logger.info(f"  Cached relationships: {fg_stats['cached_wallets']}")
        logger.info(f"  Parent nodes discovered: {fg_stats['parent_nodes']}")

        logger.info(f"{'='*60}\n")

    def _print_final_statistics(self):
        """Print final comprehensive statistics"""
        runtime = (datetime.now() - self.start_time).total_seconds() / 3600
        metrics = self.classifier.get_statistics()

        logger.info(f"\n\n{'='*80}")
        logger.info("ANTI-BOT ANALYSIS - FINAL REPORT")
        logger.info(f"{'='*80}")
        logger.info(f"Runtime: {runtime:.2f} hours")
        logger.info(f"Total transactions: {self.total_txs_seen:,}")
        logger.info(f"Total trades: {self.total_trades_seen:,}")
        logger.info(f"Wallets classified: {metrics.total_wallets_tracked:,}")
        logger.info(f"Mints observed: {metrics.total_mints_tracked:,}")

        logger.info(f"\n{'='*80}")
        logger.info("WALLET CLASSIFICATION BREAKDOWN")
        logger.info(f"{'='*80}")

        total = metrics.total_wallets_tracked
        for label, count in sorted(metrics.label_counts.items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            logger.info(f"{label.value:20s}: {count:6d} ({pct:5.1f}%)")

        logger.info(f"\n{'='*80}")
        logger.info("CRITICAL INSIGHTS")
        logger.info(f"{'='*80}")

        # Calculate bot percentage
        bot_labels = {WalletLabel.SYBIL, WalletLabel.METRIC_BOT, WalletLabel.COPY_BOT,
                     WalletLabel.BAIT_CLUSTER}
        bot_count = sum(metrics.label_counts.get(label, 0) for label in bot_labels)
        bot_pct = (bot_count / total * 100) if total > 0 else 0

        organic_count = metrics.label_counts.get(WalletLabel.ORGANIC, 0)
        organic_pct = (organic_count / total * 100) if total > 0 else 0

        logger.info(f"\n🤖 BOT ACTIVITY: {bot_count:,} wallets ({bot_pct:.1f}%)")
        logger.info(f"👤 ORGANIC ACTIVITY: {organic_count:,} wallets ({organic_pct:.1f}%)")

        if bot_pct > 50:
            logger.info(f"\n⚠️ WARNING: Environment is HEAVILY botted ({bot_pct:.0f}%)")
            logger.info(f"   → Most 'organic' patterns are likely sophisticated bots")
            logger.info(f"   → Standard quant/metrics strategies will fail")
            logger.info(f"   → Edge must come from exploiting bot behaviors")
        elif bot_pct > 30:
            logger.info(f"\n⚠️ CAUTION: Significant bot presence ({bot_pct:.0f}%)")
            logger.info(f"   → Mix of organic and bot activity")
            logger.info(f"   → Filter aggressively before trading")
        else:
            logger.info(f"\n✅ Relatively organic environment ({organic_pct:.0f}% organic)")
            logger.info(f"   → Metrics-based strategies may work")

        # Save detailed results
        self._save_results()

        logger.info(f"\n{'='*80}")
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"{'='*80}\n")

    def _save_checkpoint(self):
        """Save checkpoint (same as _save_results but labeled as checkpoint)"""
        output_dir = Path("anti-bot/data/checkpoints")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"checkpoint_{timestamp}.json"

        # Build summary
        metrics = self.classifier.get_statistics()

        # Sample high-confidence classifications
        samples_by_label = defaultdict(list)
        for wallet, profile in self.classifier.wallet_profiles.items():
            if profile.confidence >= 0.5:  # Lower threshold for checkpoints
                samples_by_label[profile.label].append({
                    'wallet': wallet,  # Full wallet for recovery
                    'confidence': profile.confidence,
                    'reasons': profile.label_reasons,
                    'observations': profile.observations,
                    'trades': profile.trades_count,
                })

        # Limit samples
        for label in samples_by_label:
            samples_by_label[label] = sorted(samples_by_label[label],
                                            key=lambda x: -x['confidence'])[:50]

        # Add funding graph stats
        fg_stats = self.classifier.funding_graph.get_stats()

        data = {
            'metadata': {
                'checkpoint_time': datetime.now().isoformat(),
                'start_time': self.start_time.isoformat(),
                'runtime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'transactions_seen': self.total_txs_seen,
                'trades_seen': self.total_trades_seen,
                'wallets_tracked': metrics.total_wallets_tracked,
                'mints_tracked': metrics.total_mints_tracked,
            },
            'label_distribution': {label.value: count for label, count in metrics.label_counts.items()},
            'detection_stats': {
                'sybil_clusters': metrics.sybil_clusters_found,
                'metric_bots': metrics.metric_bots_found,
                'bait_clusters': metrics.bait_clusters_found,
            },
            'funding_graph': fg_stats,
            'sample_classifications': {label.value: samples for label, samples in samples_by_label.items()},
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_results(self):
        """Save detailed classification results with bot profiles, strategies, and blacklist"""
        output_dir = Path("anti-bot/data")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"classification_results_{timestamp}.json"
        blacklist_file = output_dir / f"wallet_blacklist_{timestamp}.txt"
        strategy_file = output_dir / f"bot_strategies_{timestamp}.md"

        # Build summary
        metrics = self.classifier.get_statistics()

        # Generate detailed bot profiles for high-confidence detections
        logger.info("\n📊 Generating detailed bot behavior profiles...")
        bot_profiles = []
        for wallet, profile in self.classifier.wallet_profiles.items():
            # Only profile confident bot detections (not organic/unknown)
            if profile.confidence >= 0.50 and profile.label not in [WalletLabel.UNKNOWN, WalletLabel.ORGANIC]:
                # Gather additional classifier data
                classifier_data = {
                    'hold_times': self.classifier._calculate_hold_times(wallet),
                    'cluster_size': self.classifier.funding_graph.get_cluster_size(wallet),
                }

                # Check for metric bot specific data
                trades = list(self.classifier.wallet_trades.get(wallet, []))
                if trades:
                    burst_follows = 0
                    total_buys = sum(1 for t in trades if t['is_buy'])
                    for trade in trades:
                        if trade['is_buy']:
                            mint = trade['mint']
                            if mint in self.classifier.bursts:
                                for burst in self.classifier.bursts[mint]:
                                    time_after = (trade['timestamp'] - burst.end_time).total_seconds()
                                    if 0.5 <= time_after <= 5.0:
                                        burst_follows += 1
                                        break

                    classifier_data['burst_follows'] = burst_follows
                    classifier_data['total_buys'] = total_buys

                    # Early trades count
                    early_trades = sum(1 for t in trades if t.get('entry_index', 100) <= 6)
                    classifier_data['early_trades'] = early_trades

                    # First seller count
                    first_seller_count = 0
                    for trade in trades:
                        if not trade['is_buy']:
                            mint = trade['mint']
                            if mint in self.classifier.mint_trades:
                                sells = [t for t in self.classifier.mint_trades[mint] if not t['is_buy']]
                                sell_index = sum(1 for s in sells if s['timestamp'] < trade['timestamp'])
                                if sell_index < 3:
                                    first_seller_count += 1
                    classifier_data['first_seller_count'] = first_seller_count

                bot_profile = self.profiler.generate_profile(profile, classifier_data)
                bot_profiles.append(bot_profile)

        logger.info(f"✅ Generated {len(bot_profiles)} detailed bot profiles")

        # Sample high-confidence classifications
        samples_by_label = defaultdict(list)
        for wallet, profile in self.classifier.wallet_profiles.items():
            if profile.confidence >= 0.60:  # Lower threshold for samples
                samples_by_label[profile.label].append({
                    'wallet': wallet[:16] + '...',
                    'confidence': profile.confidence,
                    'reasons': profile.label_reasons,
                    'observations': profile.observations,
                    'trades': profile.trades_count,
                })

        # Limit samples
        for label in samples_by_label:
            samples_by_label[label] = sorted(samples_by_label[label],
                                            key=lambda x: -x['confidence'])[:30]

        # Generate bot summary report
        bot_summary = self.profiler.generate_summary_report()

        # Build comprehensive JSON report
        data = {
            'metadata': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'runtime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'transactions_seen': self.total_txs_seen,
                'trades_seen': self.total_trades_seen,
                'wallets_tracked': metrics.total_wallets_tracked,
                'mints_tracked': metrics.total_mints_tracked,
            },
            'label_distribution': {label.value: count for label, count in metrics.label_counts.items()},
            'detection_stats': {
                'sybil_clusters': metrics.sybil_clusters_found,
                'metric_bots': metrics.metric_bots_found,
                'bait_clusters': metrics.bait_clusters_found,
            },
            'bot_summary': {
                'total_bots_profiled': bot_summary['total_bots_profiled'],
                'bots_by_type': bot_summary['by_type'],
                'top_threats': [
                    {
                        'wallet': p.wallet[:16] + '...',
                        'type': p.label.value,
                        'confidence': p.confidence,
                        'signature': p.signature,
                    }
                    for p in bot_summary['top_threats']
                ],
            },
            'sample_classifications': {label.value: samples for label, samples in samples_by_label.items()},
            'bot_behavior_profiles': [
                {
                    'wallet': p.wallet[:16] + '...',
                    'type': p.label.value,
                    'confidence': p.confidence,
                    'signature': p.signature,
                    'trades': p.total_trades,
                    'mints_traded': p.mints_traded,
                    'median_hold_sec': p.median_hold_time_sec,
                    'mean_buy_sol': p.mean_buy_size_sol,
                    'estimated_win_rate': p.estimated_win_rate,
                }
                for p in bot_profiles[:100]  # Top 100 profiles
            ],
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"✅ Detailed results saved to: {output_file}")

        # Export blacklist
        logger.info("\n🚫 Generating wallet blacklist...")
        blacklist = self.profiler.export_blacklist(
            min_confidence=0.50,  # Lowered from 0.60 to match detection thresholds
            exclude_types={WalletLabel.DEPLOYER, WalletLabel.UNKNOWN, WalletLabel.ORGANIC}
        )

        with open(blacklist_file, 'w') as f:
            f.write(f"# Anti-Bot Wallet Blacklist\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total blacklisted: {len(blacklist)}\n")
            f.write(f"# Minimum confidence: 0.50\n")
            f.write(f"# Excluded types: DEPLOYER, UNKNOWN, ORGANIC\n\n")
            for wallet in blacklist:
                f.write(f"{wallet}\n")

        logger.info(f"✅ Blacklist saved ({len(blacklist)} wallets): {blacklist_file}")

        # Export strategy guide
        logger.info("\n📚 Generating bot strategy guide...")
        with open(strategy_file, 'w', encoding='utf-8') as f:
            f.write(f"# Anti-Bot Strategy Guide\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            f.write(f"**Analysis Duration:** {(datetime.now() - self.start_time).total_seconds() / 3600:.2f} hours\n\n")
            f.write(f"**Wallets Analyzed:** {metrics.total_wallets_tracked:,}\n\n")
            f.write(f"---\n\n")

            # Generate strategy for each detected bot type
            for label in [WalletLabel.SYBIL, WalletLabel.METRIC_BOT, WalletLabel.PRIORITY_RACER,
                         WalletLabel.EARLY_EXIT, WalletLabel.DEPLOYER]:
                count = metrics.label_counts.get(label, 0)
                if count == 0:
                    continue

                strategy = self.profiler.get_strategy_guide(label)
                if not strategy:
                    continue

                f.write(f"## {label.value.upper()} ({count:,} detected)\n\n")
                f.write(f"**Threat Level:** {strategy.threat_level}\n\n")

                f.write(f"### How They Operate\n\n")
                f.write(f"{strategy.operation_description}\n\n")

                f.write(f"**Typical Timing:** {strategy.typical_timing}\n\n")

                f.write(f"### Common Characteristics\n\n")
                for char in strategy.common_characteristics:
                    f.write(f"- {char}\n")
                f.write("\n")

                f.write(f"### How to Avoid Getting Exploited\n\n")
                for avoid in strategy.avoidance_strategies:
                    f.write(f"- {avoid}\n")
                f.write("\n")

                f.write(f"### How to Exploit Their Behavior\n\n")
                for exploit in strategy.exploitation_strategies:
                    f.write(f"- {exploit}\n")
                f.write("\n")

                f.write(f"### Detection Signals\n\n")
                for signal in strategy.detection_signals:
                    f.write(f"- {signal}\n")
                f.write("\n")

                f.write(f"---\n\n")

        logger.info(f"✅ Strategy guide saved: {strategy_file}")

        # Export full bot database (CSV for easy analysis)
        logger.info("\n🗄️ Generating full bot database...")
        bot_db_file = output_dir / f"bot_database_{timestamp}.csv"

        import csv
        with open(bot_db_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'wallet_address',
                'bot_type',
                'confidence',
                'trades',
                'buys',
                'sells',
                'mints_traded',
                'mean_buy_sol',
                'median_hold_sec',
                'mean_cu_price',
                'cluster_size',
                'burst_follow_rate',
                'signature',
                'reasons'
            ])

            # Write all classified bots (not just samples)
            for wallet, profile in self.classifier.wallet_profiles.items():
                # Only export bots, not unknown/organic
                if profile.label in [WalletLabel.UNKNOWN, WalletLabel.ORGANIC]:
                    continue

                if profile.confidence < 0.50:
                    continue

                # Get detailed metrics
                hold_times = self.classifier._calculate_hold_times(wallet)
                median_hold = statistics.median(hold_times) if hold_times else 0.0
                cluster_size = self.classifier.funding_graph.get_cluster_size(wallet)

                # Calculate burst follow rate for metric bots
                burst_follow_rate = 0.0
                trades = list(self.classifier.wallet_trades.get(wallet, []))
                if profile.label == WalletLabel.METRIC_BOT and trades:
                    burst_follows = 0
                    total_buys = sum(1 for t in trades if t['is_buy'])
                    for trade in trades:
                        if trade['is_buy']:
                            mint = trade['mint']
                            if mint in self.classifier.bursts:
                                for burst in self.classifier.bursts[mint]:
                                    time_after = (trade['timestamp'] - burst.end_time).total_seconds()
                                    if 0.5 <= time_after <= 5.0:
                                        burst_follows += 1
                                        break
                    if total_buys > 0:
                        burst_follow_rate = burst_follows / total_buys

                # Build signature
                signature = ""
                if profile.label == WalletLabel.SYBIL:
                    signature = f"Sybil cluster ({cluster_size} wallets) | {profile.mean_buy_size_sol:.4f} SOL avg"
                elif profile.label == WalletLabel.METRIC_BOT:
                    signature = f"Metric bot ({burst_follow_rate:.0%} burst-triggered) | {median_hold:.0f}s hold"
                elif profile.label == WalletLabel.DEPLOYER:
                    signature = f"Deployer ({len(profile.mints_traded)} tokens)"
                elif profile.label == WalletLabel.EARLY_EXIT:
                    signature = f"Early exit | {median_hold:.0f}s median hold"

                writer.writerow([
                    wallet,  # FULL ADDRESS
                    profile.label.value,
                    f"{profile.confidence:.3f}",
                    profile.trades_count,
                    profile.buys_count,
                    profile.sells_count,
                    len(profile.mints_traded),
                    f"{profile.mean_buy_size_sol:.4f}",
                    f"{median_hold:.1f}",
                    int(profile.mean_cu_price),
                    cluster_size,
                    f"{burst_follow_rate:.3f}",
                    signature,
                    "; ".join(profile.label_reasons[:3])  # Top 3 reasons
                ])

        # Count entries
        bot_count = sum(1 for p in self.classifier.wallet_profiles.values()
                       if p.label not in [WalletLabel.UNKNOWN, WalletLabel.ORGANIC]
                       and p.confidence >= 0.50)

        logger.info(f"✅ Bot database saved ({bot_count} bots): {bot_db_file}")

    async def stop(self):
        """Stop analyzer"""
        self.running = False

        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None

        # Stop RPC manager
        if self.rpc_manager:
            await self.rpc_manager.stop()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Anti-Bot Wallet Classifier")
    parser.add_argument('--hours', type=float, default=6.0, help='How many hours to analyze (default: 6)')
    parser.add_argument('--config', type=str, default=None, help='Config file to use')

    args = parser.parse_args()

    # Setup logging
    setup_logging("INFO", "logs/anti_bot_analyzer.log")

    # Load config
    config = None
    config_file = args.config

    if config_file:
        try:
            config = load_config(config_file)
            logger.info(f"Loaded config from: {config_file}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            return
    else:
        # Auto-detect: try config_fast.yml first, then config.yml
        for config_name in ['config_fast.yml', 'config.yml']:
            try:
                config = load_config(config_name)
                logger.info(f"Loaded config from: {config_name}")
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.error(f"Error loading {config_name}: {e}")
                continue

        if not config:
            logger.error("Could not find config.yml or config_fast.yml")
            return

    # Get LaserStream config
    fast_config = config.get('fast_execution', {})
    analyzer_config = {
        'helius_grpc_endpoint': fast_config.get('helius_grpc_endpoint'),
        'helius_grpc_token': fast_config.get('helius_grpc_token'),
        # Note: rpc_endpoint auto-constructed from helius credentials in __init__
    }

    if not analyzer_config['helius_grpc_endpoint'] or not analyzer_config['helius_grpc_token']:
        logger.error("Missing helius_grpc_endpoint or helius_grpc_token in config")
        return

    # Create analyzer
    try:
        analyzer = AntiBotAnalyzer(analyzer_config)
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return

    # Run analysis
    try:
        await analyzer.start(run_hours=args.hours)
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Interrupted by user")
        await analyzer.stop()
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
