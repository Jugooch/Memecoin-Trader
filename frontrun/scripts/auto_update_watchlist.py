"""
Auto-Update Dev Watchlist Scheduler

Runs the dev watchlist builder every X hours to continuously grow your database.

Usage:
    # Run every 8 hours (default)
    python scripts/auto_update_watchlist.py

    # Run every 6 hours
    python scripts/auto_update_watchlist.py --interval 6

    # Custom settings
    python scripts/auto_update_watchlist.py --interval 12 --max-pages 300 --min-buy-rate 75

Features:
    - Runs in background
    - Logs all updates
    - Graceful shutdown (Ctrl+C)
    - Error recovery (continues on failure)
    - Shows progress and stats
"""

import asyncio
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import json
import signal

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import setup_logging, get_logger

logger = get_logger(__name__)


class WatchlistScheduler:
    """
    Automatically update dev watchlist on a schedule
    """

    def __init__(
        self,
        interval_hours: float = 8.0,
        max_pages: int = 500,
        min_buy_rate: float = 70.0,
        min_tokens: int = 2,
        lookback_days: int = 7
    ):
        """
        Initialize scheduler

        Args:
            interval_hours: Hours between updates (default: 8)
            max_pages: Max pages per update (default: 500)
            min_buy_rate: Minimum buy rate % (default: 70)
            min_tokens: Minimum tokens created (default: 2)
            lookback_days: Days to look back (default: 7)
        """
        self.interval_hours = interval_hours
        self.max_pages = max_pages
        self.min_buy_rate = min_buy_rate
        self.min_tokens = min_tokens
        self.lookback_days = lookback_days

        self.running = False
        self.update_count = 0
        self.last_update_time = None
        self.next_update_time = None

        # Path to the builder script
        self.builder_script = Path(__file__).parent / "build_dev_watchlist.py"
        self.watchlist_path = Path(__file__).parent.parent / "data" / "dev_watchlist.json"

        logger.info(
            "scheduler_initialized",
            interval_hours=interval_hours,
            max_pages=max_pages,
            min_buy_rate=min_buy_rate
        )

    def _get_watchlist_stats(self) -> dict:
        """Get current watchlist statistics"""
        try:
            if not self.watchlist_path.exists():
                return {"total_devs": 0, "avg_buy_rate": 0}

            with open(self.watchlist_path, 'r') as f:
                data = json.load(f)

            if not data:
                return {"total_devs": 0, "avg_buy_rate": 0}

            avg_buy_rate = sum(d['buy_rate'] for d in data) / len(data)

            return {
                "total_devs": len(data),
                "avg_buy_rate": round(avg_buy_rate, 1)
            }

        except Exception as e:
            logger.error(f"error_reading_watchlist_stats", error=str(e))
            return {"total_devs": 0, "avg_buy_rate": 0}

    async def run_update(self) -> bool:
        """
        Run a single watchlist update

        Returns:
            True if successful, False otherwise
        """
        logger.info("=== STARTING WATCHLIST UPDATE ===")

        # Get stats before update
        stats_before = self._get_watchlist_stats()
        logger.info(
            "watchlist_before",
            devs=stats_before['total_devs'],
            avg_buy_rate=stats_before['avg_buy_rate']
        )

        # Build command
        cmd = [
            sys.executable,
            str(self.builder_script),
            "--max-pages", str(self.max_pages),
            "--min-buy-rate", str(self.min_buy_rate),
            "--min-tokens", str(self.min_tokens),
            "--lookback-days", str(self.lookback_days)
        ]

        logger.info(f"executing_command", cmd=" ".join(cmd))

        try:
            # Run the builder script
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info("✅ UPDATE SUCCESSFUL")

                # Get stats after update
                stats_after = self._get_watchlist_stats()

                devs_added = stats_after['total_devs'] - stats_before['total_devs']

                logger.info(
                    "watchlist_after",
                    devs=stats_after['total_devs'],
                    avg_buy_rate=stats_after['avg_buy_rate'],
                    devs_added=devs_added
                )

                # Log stdout for debugging
                if stdout:
                    logger.debug("builder_stdout", output=stdout.decode())

                self.update_count += 1
                self.last_update_time = datetime.now()

                return True

            else:
                logger.error(
                    "update_failed",
                    returncode=process.returncode,
                    stderr=stderr.decode() if stderr else None
                )
                return False

        except Exception as e:
            logger.error("update_exception", error=str(e), exc_info=True)
            return False

    async def run_forever(self):
        """Run scheduler forever (until stopped)"""
        self.running = True

        logger.info("=== AUTO-UPDATE SCHEDULER STARTED ===")
        logger.info(f"Update interval: Every {self.interval_hours} hours")
        logger.info(f"Pages per update: {self.max_pages}")
        logger.info(f"Lookback window: {self.lookback_days} days")
        logger.info(f"Min buy rate: {self.min_buy_rate}%")
        logger.info("Press Ctrl+C to stop\n")

        # Show initial stats
        initial_stats = self._get_watchlist_stats()
        logger.info(
            "initial_watchlist_stats",
            devs=initial_stats['total_devs'],
            avg_buy_rate=initial_stats['avg_buy_rate']
        )

        # Run first update immediately
        logger.info("Running first update immediately...")
        await self.run_update()

        # Calculate next update time
        self.next_update_time = datetime.now() + timedelta(hours=self.interval_hours)

        while self.running:
            try:
                # Calculate time until next update
                now = datetime.now()
                time_until_next = (self.next_update_time - now).total_seconds()

                if time_until_next > 0:
                    # Log countdown
                    hours = int(time_until_next / 3600)
                    minutes = int((time_until_next % 3600) / 60)

                    logger.info(
                        "waiting_for_next_update",
                        next_update=self.next_update_time.strftime("%Y-%m-%d %H:%M:%S"),
                        time_remaining=f"{hours}h {minutes}m",
                        updates_completed=self.update_count
                    )

                    # Sleep in smaller chunks so we can check self.running
                    sleep_interval = min(time_until_next, 300)  # Max 5 min chunks
                    await asyncio.sleep(sleep_interval)

                else:
                    # Time to update!
                    await self.run_update()

                    # Schedule next update
                    self.next_update_time = datetime.now() + timedelta(hours=self.interval_hours)

            except asyncio.CancelledError:
                logger.info("Scheduler cancelled")
                break

            except Exception as e:
                logger.error(
                    "scheduler_error",
                    error=str(e),
                    exc_info=True
                )

                # Wait a bit before retrying
                logger.info("Waiting 5 minutes before retry...")
                await asyncio.sleep(300)

        logger.info("=== SCHEDULER STOPPED ===")
        logger.info(f"Total updates completed: {self.update_count}")

        # Show final stats
        final_stats = self._get_watchlist_stats()
        logger.info(
            "final_watchlist_stats",
            devs=final_stats['total_devs'],
            avg_buy_rate=final_stats['avg_buy_rate']
        )

    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping scheduler...")
        self.running = False


async def main():
    parser = argparse.ArgumentParser(
        description="Auto-update dev watchlist on a schedule",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run every 8 hours (default)
  python scripts/auto_update_watchlist.py

  # Run every 6 hours with 300 pages
  python scripts/auto_update_watchlist.py --interval 6 --max-pages 300

  # Run every 12 hours with looser filters
  python scripts/auto_update_watchlist.py --interval 12 --min-buy-rate 60

Strategy:
  - Start with 8-hour intervals to build database quickly
  - After you have 5000+ devs, switch to 24-hour intervals
  - Lower min-buy-rate over time as your watchlist grows
  - The script runs continuously - leave it running in background

Background Usage (Linux/Mac):
  # Run in background with nohup
  nohup python scripts/auto_update_watchlist.py > logs/auto_update.log 2>&1 &

  # View logs
  tail -f logs/auto_update.log

Background Usage (Windows):
  # Run in a separate terminal window
  # Or use Task Scheduler to run at startup
        """
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=8.0,
        help="Hours between updates (default: 8)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=500,
        help="Max pages per update (default: 500)"
    )
    parser.add_argument(
        "--min-buy-rate",
        type=float,
        default=70.0,
        help="Minimum buy rate %% (default: 70)"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=2,
        help="Minimum tokens created (default: 2)"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Days to look back (default: 7)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO", format="console")

    # Create scheduler
    scheduler = WatchlistScheduler(
        interval_hours=args.interval,
        max_pages=args.max_pages,
        min_buy_rate=args.min_buy_rate,
        min_tokens=args.min_tokens,
        lookback_days=args.lookback_days
    )

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\nReceived shutdown signal")
        scheduler.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run forever
    await scheduler.run_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Scheduler stopped by user")
    except Exception as e:
        logger.error("fatal_error", error=str(e), exc_info=True)
        sys.exit(1)
