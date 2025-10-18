"""
Wallet Rotation Manager
Handles smart rotation of alpha wallets every 2 hours
"""

import asyncio
import logging
import time
import yaml
from typing import List, Dict, Set
from src.core.wallet_tracker import WalletTracker
from src.discovery.alpha_discovery_v2 import ProvenAlphaFinder
from src.clients.bitquery_client import BitqueryClient
from src.clients.moralis_client import MoralisClient
from src.core.database import Database

class WalletRotationManager:
    def __init__(self, wallet_tracker: WalletTracker, bitquery: BitqueryClient,
                 moralis: MoralisClient, database: Database, config_path: str = "config/config.yml",
                 discord_notifier=None, realtime_client=None, laserstream_monitor=None):
        self.wallet_tracker = wallet_tracker
        self.bitquery = bitquery
        self.moralis = moralis
        self.database = database
        self.config_path = config_path
        self.discord_notifier = discord_notifier
        self.realtime_client = realtime_client  # Add realtime client for updating subscriptions
        self.laserstream_monitor = laserstream_monitor  # Add LaserStream monitor for fast wallet detection
        self.logger = logging.getLogger(__name__)
        
        # Load config from file
        from src.utils.config_loader import load_config
        # Extract just the filename from path (e.g., "config/config_fast.yml" -> "config_fast.yml")
        self.config_filename = config_path.split('/')[-1] if '/' in config_path else config_path
        self.config = load_config(self.config_filename)

        self.logger.info(f"WalletRotationManager loaded config from: {self.config_filename}")

        # Rotation settings - read from config or use defaults
        rotation_config = self.config.get('wallet_rotation', {})
        self.rotation_interval = rotation_config.get('interval_hours', 2) * 3600  # Convert hours to seconds
        self.min_wallets = rotation_config.get('min_wallets', 50)
        self.max_wallets = rotation_config.get('max_wallets', 100)
        self.retention_threshold = rotation_config.get('retention_threshold', 50.0)
        self.high_performer_threshold = rotation_config.get('high_performer_threshold', 70.0)
        
        self.logger.info(f"Wallet rotation config: interval={self.rotation_interval/3600}h, "
                        f"wallets={self.min_wallets}-{self.max_wallets}, "
                        f"thresholds={self.retention_threshold}/{self.high_performer_threshold}")
        
        # Track last rotation
        self.last_rotation = 0
        self.running = False
    
    async def start_rotation_loop(self):
        """Start the 2-hour rotation loop"""
        self.logger.info("start_rotation_loop() called")
        self.running = True
        self.logger.info("Starting wallet rotation manager (2-hour intervals)")
        
        # Perform initial rotation immediately
        try:
            self.logger.info("Performing initial wallet rotation...")
            await self.perform_rotation()
            self.last_rotation = time.time()
            self.logger.info("Initial wallet rotation completed")
        except Exception as e:
            self.logger.error(f"Error in initial rotation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        self.logger.info("Entering rotation loop...")
        while self.running:
            try:
                # Wait before checking for rotation
                await asyncio.sleep(600)  # Check every 10 minutes
                
                # Check if it's time for rotation
                current_time = time.time()
                if current_time - self.last_rotation >= self.rotation_interval:
                    self.logger.info("Time for scheduled rotation")
                    await self.perform_rotation()
                    self.last_rotation = current_time
                
            except Exception as e:
                self.logger.error(f"Error in rotation loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def stop_rotation(self):
        """Stop the rotation loop"""
        self.running = False
        self.logger.info("Stopping wallet rotation manager")
    
    async def perform_rotation(self):
        """Perform a complete wallet rotation cycle"""
        self.logger.info("=" * 60)
        self.logger.info("Starting wallet rotation cycle")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Evaluate current wallets
        rotation_analysis = self.wallet_tracker.get_wallets_for_rotation()
        
        keep_wallets = set(rotation_analysis["keep"])
        candidates = set(rotation_analysis["candidates"])
        replace_wallets = set(rotation_analysis["replace"])
        
        self.logger.info(f"Wallet evaluation: {len(keep_wallets)} keep, "
                        f"{len(candidates)} candidates, {len(replace_wallets)} replace")
        
        # Step 2: Determine how many new wallets we need
        current_total = len(self.wallet_tracker.watched_wallets)
        wallets_to_remove = len(replace_wallets)
        
        # Calculate target: aim for max_wallets but ensure we have enough good ones
        target_new_wallets = wallets_to_remove + max(0, self.max_wallets - len(keep_wallets) - len(candidates))
        
        self.logger.info(f"Planning to discover {target_new_wallets} new wallets")
        
        # Step 3: Discover new alpha wallets
        new_wallets = []

        # Check if discovery is enabled
        discover_new_wallets = self.config.get('api_optimization', {}).get('discover_new_wallets', True)

        if target_new_wallets > 0 and discover_new_wallets:
            discovery_start = time.time()
            try:
                self.logger.info("Discovery enabled - searching for new alpha wallets...")
                finder = ProvenAlphaFinder(self.bitquery, self.moralis, self.database, self.config)
                discovered_wallets = await finder.discover_alpha_wallets()

                if discovered_wallets:
                    # Filter out wallets we already have
                    existing_wallets = self.wallet_tracker.watched_wallets
                    filtered_wallets = [w for w in discovered_wallets if w not in existing_wallets]
                    new_wallets = filtered_wallets[:target_new_wallets]

                    discovery_time = time.time() - discovery_start
                    self.logger.info(f"Discovery completed in {discovery_time:.1f}s: "
                                   f"discovered {len(discovered_wallets)} total, "
                                   f"{len(filtered_wallets)} new, "
                                   f"taking {len(new_wallets)} for rotation")
                else:
                    self.logger.warning("Alpha discovery returned no new wallets")

            except Exception as e:
                self.logger.error(f"Error during alpha discovery: {e}")
        elif target_new_wallets > 0 and not discover_new_wallets:
            self.logger.info(f"Discovery disabled - selecting best {target_new_wallets} wallets from existing cache")

            # SMART SELECTION: Score ALL existing wallets and take the top performers
            # This ensures we get active, high-scoring wallets even without discovery
            all_existing_wallets = list(self.wallet_tracker.watched_wallets)

            # Score each wallet
            scored_wallets = []
            for wallet in all_existing_wallets:
                score = self.wallet_tracker.performance_tracker.evaluate_wallet(wallet)
                scored_wallets.append((wallet, score))

            # Sort by score (highest first)
            scored_wallets.sort(key=lambda x: x[1], reverse=True)

            # Log top scorers for visibility
            top_5 = scored_wallets[:5]
            self.logger.info(f"Top 5 existing wallets by score:")
            for wallet, score in top_5:
                self.logger.info(f"  {wallet[:8]}... = {score:.1f} points")

            # Take the top N wallets to fill our target
            top_wallets = [w for w, s in scored_wallets[:target_new_wallets]]

            # Add them as "new" wallets (they're not really new, just best from cache)
            new_wallets = top_wallets

            # Move low-scoring wallets from "replace" to "candidates" (keep them as backup)
            candidates.update(replace_wallets)
            replace_wallets.clear()

            self.logger.info(f"Selected {len(new_wallets)} top-scoring wallets from existing cache (no discovery)")
        elif target_new_wallets == 0:
            self.logger.info("No new wallets needed for rotation")
        
        # Step 4: Build final wallet list
        final_wallets = set()
        
        # Always keep high performers
        final_wallets.update(keep_wallets)
        
        # Add candidates if we have room
        remaining_slots = self.max_wallets - len(final_wallets)
        if remaining_slots > 0:
            candidates_to_add = list(candidates)[:remaining_slots]
            final_wallets.update(candidates_to_add)
            remaining_slots -= len(candidates_to_add)
        
        # Add new wallets if we have room
        if remaining_slots > 0 and new_wallets:
            new_to_add = new_wallets[:remaining_slots]
            final_wallets.update(new_to_add)
        
        # Ensure we have minimum wallets
        if len(final_wallets) < self.min_wallets:
            # Keep some candidates even if they scored low
            additional_needed = self.min_wallets - len(final_wallets)
            additional_candidates = [w for w in candidates if w not in final_wallets][:additional_needed]
            final_wallets.update(additional_candidates)
        
        # Step 5: Update the wallet tracker
        old_wallets = set(self.wallet_tracker.watched_wallets)
        added_wallets = final_wallets - old_wallets
        removed_wallets = old_wallets - final_wallets
        kept_wallets = final_wallets & old_wallets
        
        # Update wallet tracker
        self.wallet_tracker.watched_wallets = final_wallets
        
        # Initialize new wallets in performance tracker
        for wallet in added_wallets:
            self.wallet_tracker.add_watched_wallet(wallet)
        
        # Step 6: Update config file
        await self._update_config_file(list(final_wallets))
        
        # Step 7: Update PumpPortal subscriptions with new wallet list
        if self.realtime_client:
            try:
                self.logger.info("Updating PumpPortal subscriptions with new wallet list...")
                await self.realtime_client.update_wallet_subscriptions(list(final_wallets))
                self.logger.info("PumpPortal subscriptions updated successfully")
            except Exception as e:
                self.logger.error(f"Failed to update PumpPortal subscriptions: {e}")

        # Step 7b: Update LaserStream monitor with new wallet list (for fast execution)
        if self.laserstream_monitor:
            try:
                self.logger.info("Updating LaserStream monitor with new wallet list...")
                self.laserstream_monitor.update_wallets(list(final_wallets))
                self.logger.info(f"LaserStream monitor updated successfully with {len(final_wallets)} wallets")
            except Exception as e:
                self.logger.error(f"Failed to update LaserStream monitor: {e}")
        
        # Step 8: Record the rotation
        self.wallet_tracker.performance_tracker.record_rotation(
            list(kept_wallets), list(added_wallets), list(removed_wallets)
        )
        
        # Step 9: Log results
        total_time = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info(f"Wallet rotation completed in {total_time:.1f}s:")
        self.logger.info(f"  Kept: {len(kept_wallets)} wallets")
        self.logger.info(f"  Added: {len(added_wallets)} new wallets")
        self.logger.info(f"  Removed: {len(removed_wallets)} wallets")
        self.logger.info(f"  Total: {len(final_wallets)} wallets")
        
        if added_wallets:
            sample_new = list(added_wallets)[:3]
            self.logger.info(f"  New wallets (sample): {[w[:8]+'...' for w in sample_new]}")
        
        if removed_wallets:
            sample_removed = list(removed_wallets)[:3]
            self.logger.info(f"  Removed wallets (sample): {[w[:8]+'...' for w in sample_removed]}")
        
        self.logger.info("=" * 60)
        
        # Step 10: Get performance summary and tier stats for Discord notification
        performance_summary = self.wallet_tracker.get_performance_summary()
        tier_stats = self.wallet_tracker.get_tier_performance_stats()
        
        self.logger.info(f"Performance summary: {performance_summary['active_wallets']}/{performance_summary['total_wallets']} active, "
                        f"win rate: {performance_summary['overall_win_rate']:.1%}")
        
        # Send Discord notification about rotation
        if self.discord_notifier and (len(added_wallets) > 0 or len(removed_wallets) > 0):
            # Build tier performance breakdown
            tier_breakdown = []
            for tier in ['S', 'A', 'B', 'C', 'Unknown']:
                stats = tier_stats[tier]
                if stats['count'] > 0:
                    tier_breakdown.append(
                        f"**{tier}**: {stats['count']} wallets, {stats['trades']} trades, "
                        f"{stats['win_rate']:.1%} WR, {stats['avg_pnl']:.1f}% avg P/L"
                    )
            
            tier_section = "\n".join(tier_breakdown) if tier_breakdown else "No wallet performance data yet"
            
            rotation_message = f"🔄 **Wallet Rotation Completed** ({total_time:.0f}s)\n" \
                             f"• Kept: {len(kept_wallets)} high-performing wallets\n" \
                             f"• Added: {len(added_wallets)} new alpha wallets\n" \
                             f"• Removed: {len(removed_wallets)} underperforming wallets\n" \
                             f"• **Total: {len(final_wallets)} wallets active**\n" \
                             f"• Performance: {performance_summary['active_wallets']}/{performance_summary['total_wallets']} active, " \
                             f"{performance_summary['overall_win_rate']:.1%} win rate\n\n" \
                             f"**📊 Wallet Tier Breakdown:**\n{tier_section}"
            
            try:
                await self.discord_notifier.send_text(rotation_message)
            except Exception as e:
                self.logger.warning(f"Failed to send rotation notification to Discord: {e}")
    
    async def _update_config_file(self, wallet_list: List[str]):
        """Update the config file with new wallet list using safe atomic write"""
        try:
            from src.utils.config_loader import safe_update_config
            from pathlib import Path

            # Build correct config path using the same config file we loaded from
            project_root = Path(__file__).parent.parent.parent
            config_file_path = str(project_root / "config" / self.config_filename)

            # Use safe atomic update with correct config path
            updates = {'watched_wallets': wallet_list}
            safe_update_config(updates, config_path=config_file_path)

            self.logger.info(f"Updated {self.config_filename} with {len(wallet_list)} wallets (safe atomic write)")

        except Exception as e:
            self.logger.error(f"Failed to update config file: {e}")
    
    def force_rotation(self):
        """Force an immediate rotation (for testing/manual trigger)"""
        self.last_rotation = 0  # Reset timer to trigger rotation
        self.logger.info("Forced rotation scheduled for next check")
    
    def get_rotation_status(self) -> Dict:
        """Get current rotation status"""
        current_time = time.time()
        time_since_last = current_time - self.last_rotation
        time_until_next = max(0, self.rotation_interval - time_since_last)
        
        return {
            "running": self.running,
            "last_rotation": self.last_rotation,
            "time_since_last_rotation": time_since_last,
            "time_until_next_rotation": time_until_next,
            "rotation_interval": self.rotation_interval,
            "total_wallets": len(self.wallet_tracker.watched_wallets),
            "performance_summary": self.wallet_tracker.get_performance_summary()
        }