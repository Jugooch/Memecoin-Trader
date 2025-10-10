"""
Phase 1: Monitoring Mode Script
Collects data on Pump.fun initial dev buys WITHOUT trading

Purpose:
- Understand opportunity volume (how many per day?)
- Analyze competition (what fees are competitors paying?)
- Measure latency (can we detect and react fast enough?)
- Simulate PnL (what would we have made?)

This script runs for 1 week collecting data before any real trading.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, List
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import ConfigurationManager
from core.logger import setup_logging, get_logger
from services.mempool_monitor import (
    LaserStreamMonitor,
    LaserStreamConfig,
    PendingTransaction
)
from services.dev_detector import DevWalletDetector, DetectorConfig


logger = get_logger(__name__)


@dataclass
class OpportunityRecord:
    """Record of a detected frontrun opportunity"""
    timestamp: datetime
    signature: str
    buy_amount_sol: float
    detected_fee: int
    would_frontrun: bool
    reason: str

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MonitoringStats:
    """Statistics from monitoring session"""
    start_time: datetime
    end_time: datetime
    total_pump_fun_txs: int
    total_opportunities: int
    opportunities_by_hour: Dict[int, int]
    avg_buy_size_sol: float
    max_buy_size_sol: float
    avg_competitor_fee: float
    max_competitor_fee: int

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat()
        }


class Phase1Monitor:
    """
    Phase 1 Monitoring Mode

    Collects data on Pump.fun opportunities without trading.

    Features:
    - Tracks all Pump.fun transactions
    - Identifies frontrun opportunities
    - Records competitor fees
    - Simulates PnL (without trading)
    - Exports data for analysis

    Usage:
        python scripts/phase1_monitor.py --duration-hours 168  # 1 week
    """

    def __init__(
        self,
        config_path: str,
        output_dir: str = "data/monitoring"
    ):
        """
        Initialize Phase 1 monitor

        Args:
            config_path: Path to config.yml
            output_dir: Directory for output data
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config_manager = ConfigurationManager(config_path)
        self.bot_config = self.config_manager.load_config()

        # Initialize components
        helius_endpoint = self.bot_config.rpc_config.endpoints[0]

        if not helius_endpoint.laserstream_url or not helius_endpoint.laserstream_api_key:
            raise ValueError("Helius LaserStream configuration required")

        # Extract endpoint (remove https://)
        endpoint = helius_endpoint.laserstream_url.replace("https://", "").replace("http://", "")

        # LaserStream monitor
        laserstream_config = LaserStreamConfig(
            endpoint=endpoint,
            api_key=helius_endpoint.laserstream_api_key,
            commitment="processed"  # Fastest for frontrunning
        )
        self.monitor = LaserStreamMonitor(laserstream_config)

        # Dev detector (for identifying opportunities)
        detector_config = DetectorConfig(
            min_buy_amount_sol=0.5  # Minimum size worth tracking
        )
        self.detector = DevWalletDetector(detector_config)

        # Data collection
        self.opportunities: List[OpportunityRecord] = []
        self.start_time = None
        self.end_time = None

    async def run(self, duration_hours: float = 168.0):
        """
        Run monitoring for specified duration

        Args:
            duration_hours: How long to monitor (default: 168 hours = 1 week)
        """
        self.start_time = datetime.now()
        target_end_time = self.start_time + timedelta(hours=duration_hours)

        logger.info(
            "phase1_monitoring_started",
            duration_hours=duration_hours,
            start_time=self.start_time.isoformat(),
            target_end_time=target_end_time.isoformat()
        )

        # Start monitoring
        monitoring_task = asyncio.create_task(
            self.monitor.start_monitoring(self._on_pending_transaction)
        )

        # Report progress every hour
        report_task = asyncio.create_task(
            self._periodic_report(interval_s=3600)  # 1 hour
        )

        # Wait for duration
        try:
            await asyncio.sleep(duration_hours * 3600)
        except KeyboardInterrupt:
            logger.info("monitoring_interrupted_by_user")

        # Stop monitoring
        self.end_time = datetime.now()
        await self.monitor.stop_monitoring()

        # Cancel tasks
        monitoring_task.cancel()
        report_task.cancel()

        # Generate final report
        await self._generate_final_report()

    async def _on_pending_transaction(self, tx: PendingTransaction):
        """
        Callback for each pending transaction

        Args:
            tx: Pending transaction from Geyser
        """
        # Analyze if this is a frontrun opportunity
        opportunity = self.detector.analyze_transaction(
            program_id=tx.program_id,
            buy_amount_sol=tx.buy_amount_sol,
            sender=tx.accounts[0] if tx.accounts else None
        )

        # Record opportunity
        record = OpportunityRecord(
            timestamp=datetime.now(),
            signature=tx.signature,
            buy_amount_sol=tx.buy_amount_sol,
            detected_fee=tx.fee,
            would_frontrun=opportunity.should_frontrun,
            reason=opportunity.reason
        )

        self.opportunities.append(record)

        # Log interesting opportunities
        if opportunity.should_frontrun:
            logger.info(
                "frontrun_opportunity_detected",
                buy_amount_sol=tx.buy_amount_sol,
                fee=tx.fee,
                signature=tx.signature[:16] + "..."
            )

    async def _periodic_report(self, interval_s: int):
        """
        Generate periodic progress reports and export data

        Args:
            interval_s: Interval in seconds
        """
        while True:
            await asyncio.sleep(interval_s)

            # Calculate stats
            total_opps = len(self.opportunities)
            frontrun_opps = sum(1 for o in self.opportunities if o.would_frontrun)

            if total_opps > 0:
                avg_buy_size = sum(o.buy_amount_sol for o in self.opportunities) / total_opps
                max_buy_size = max(o.buy_amount_sol for o in self.opportunities)
                avg_fee = sum(o.detected_fee for o in self.opportunities) / total_opps
            else:
                avg_buy_size = 0.0
                max_buy_size = 0.0
                avg_fee = 0.0

            # Get mempool stats
            mempool_stats = self.monitor.get_mempool_stats()

            logger.info(
                "periodic_report",
                elapsed_hours=(datetime.now() - self.start_time).total_seconds() / 3600,
                total_pump_fun_txs=mempool_stats.pump_fun_txs_seen,
                total_opportunities=total_opps,
                frontrun_opportunities=frontrun_opps,
                avg_buy_size_sol=round(avg_buy_size, 3),
                max_buy_size_sol=round(max_buy_size, 3),
                avg_competitor_fee=int(avg_fee),
                reconnections=mempool_stats.reconnections,
                errors=mempool_stats.errors
            )

            # Export data periodically (every hour)
            await self._export_periodic_data()

    async def _export_periodic_data(self):
        """Export data periodically (called every hour)"""
        if not self.opportunities:
            return

        # Export opportunities incrementally
        opps_file = self.output_dir / f"opportunities_latest.json"
        with open(opps_file, 'w') as f:
            json.dump([o.to_dict() for o in self.opportunities], f, indent=2)

        logger.info(f"Periodic export: {len(self.opportunities)} opportunities saved to {opps_file}")

    async def _generate_final_report(self):
        """Generate final monitoring report and export data"""
        logger.info("generating_final_report")

        # Calculate statistics
        total_opps = len(self.opportunities)
        frontrun_opps = [o for o in self.opportunities if o.would_frontrun]

        if total_opps > 0:
            avg_buy_size = sum(o.buy_amount_sol for o in self.opportunities) / total_opps
            max_buy_size = max(o.buy_amount_sol for o in self.opportunities)
            avg_fee = sum(o.detected_fee for o in self.opportunities) / total_opps
            max_fee = max(o.detected_fee for o in self.opportunities)
        else:
            avg_buy_size = 0.0
            max_buy_size = 0.0
            avg_fee = 0.0
            max_fee = 0

        # Opportunities by hour
        opportunities_by_hour = {}
        for opp in self.opportunities:
            hour = opp.timestamp.hour
            opportunities_by_hour[hour] = opportunities_by_hour.get(hour, 0) + 1

        # Get mempool stats
        mempool_stats = self.monitor.get_mempool_stats()

        stats = MonitoringStats(
            start_time=self.start_time,
            end_time=self.end_time,
            total_pump_fun_txs=mempool_stats.pump_fun_txs_seen,
            total_opportunities=total_opps,
            opportunities_by_hour=opportunities_by_hour,
            avg_buy_size_sol=avg_buy_size,
            max_buy_size_sol=max_buy_size,
            avg_competitor_fee=avg_fee,
            max_competitor_fee=max_fee
        )

        # Export stats
        stats_file = self.output_dir / f"monitoring_stats_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)

        logger.info(f"Stats exported to {stats_file}")

        # Export opportunities
        opps_file = self.output_dir / f"opportunities_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(opps_file, 'w') as f:
            json.dump([o.to_dict() for o in self.opportunities], f, indent=2)

        logger.info(f"Opportunities exported to {opps_file}")

        # Print summary
        duration_hours = (self.end_time - self.start_time).total_seconds() / 3600

        logger.info(
            "===== PHASE 1 MONITORING COMPLETE =====",
            duration_hours=round(duration_hours, 2),
            total_pump_fun_txs=stats.total_pump_fun_txs,
            total_opportunities=stats.total_opportunities,
            frontrun_opportunities=len(frontrun_opps),
            avg_buy_size_sol=round(stats.avg_buy_size_sol, 3),
            max_buy_size_sol=round(stats.max_buy_size_sol, 3),
            avg_competitor_fee_lamports=int(stats.avg_competitor_fee),
            max_competitor_fee_lamports=stats.max_competitor_fee,
            stats_file=str(stats_file),
            opportunities_file=str(opps_file)
        )

        # Analysis recommendations
        if len(frontrun_opps) > 0:
            opps_per_day = len(frontrun_opps) / max(duration_hours / 24, 1)
            logger.info(
                "analysis_recommendation",
                message=f"Found {len(frontrun_opps)} frontrun opportunities ({opps_per_day:.1f} per day)",
                recommendation="Proceed to Phase 2 (Micro-Trading)" if opps_per_day > 5 else "Low opportunity volume - review strategy"
            )
        else:
            logger.warning(
                "no_opportunities_detected",
                recommendation="Review detector configuration or wait longer"
            )


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: Monitoring Mode")
    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="Path to config.yml"
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=168.0,
        help="Monitoring duration in hours (default: 168 = 1 week)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/monitoring",
        help="Output directory for data"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO", format="json", output_file="logs/phase1_monitor.log")

    logger.info(
        "phase1_monitor_starting",
        config=args.config,
        duration_hours=args.duration_hours,
        output_dir=args.output_dir
    )

    # Create and run monitor
    monitor = Phase1Monitor(
        config_path=args.config,
        output_dir=args.output_dir
    )

    await monitor.run(duration_hours=args.duration_hours)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("phase1_monitor_failed", error=str(e), error_type=type(e).__name__)
        raise
