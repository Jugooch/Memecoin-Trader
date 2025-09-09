#!/usr/bin/env python3
"""
Sniper Bot CLI - Management interface for operations
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from sniper.storage import Store
from src.utils.config_loader import load_config


class SniperCLI:
    """CLI interface for sniper bot management"""
    
    def __init__(self):
        # Load configuration
        self.config = load_config('config_sniper.yml')
        self.store = Store(self.config['storage'])
    
    def add_blacklist(self, dev_wallet: str, reason: str):
        """Add developer to blacklist"""
        try:
            self.store.add_to_blacklist(dev_wallet, reason)
            print(f"✅ Added {dev_wallet[:8]}... to blacklist")
            print(f"   Reason: {reason}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def remove_blacklist(self, dev_wallet: str):
        """Remove developer from blacklist"""
        try:
            if not self.store.is_blacklisted(dev_wallet):
                print(f"⚠️  {dev_wallet[:8]}... is not blacklisted")
                return
            
            self.store.remove_from_blacklist(dev_wallet)
            print(f"✅ Removed {dev_wallet[:8]}... from blacklist")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def add_whitelist(self, dev_wallet: str, score: float = None):
        """Add developer to whitelist"""
        try:
            if score is None:
                score = 60.0  # Default score
            
            self.store.add_to_whitelist(dev_wallet, score)
            print(f"✅ Added {dev_wallet[:8]}... to whitelist")
            print(f"   Score: {score}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def remove_whitelist(self, dev_wallet: str):
        """Remove developer from whitelist"""
        try:
            if not self.store.is_whitelisted(dev_wallet):
                print(f"⚠️  {dev_wallet[:8]}... is not whitelisted")
                return
            
            self.store.remove_from_whitelist(dev_wallet)
            print(f"✅ Removed {dev_wallet[:8]}... from whitelist")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def show_dev(self, dev_wallet: str):
        """Show developer information"""
        try:
            print(f"\n{'='*50}")
            print(f"Developer: {dev_wallet}")
            print(f"{'='*50}")
            
            # Check lists
            is_whitelisted = self.store.is_whitelisted(dev_wallet)
            is_blacklisted = self.store.is_blacklisted(dev_wallet)
            
            print(f"Status:")
            if is_whitelisted:
                print("  ✅ WHITELISTED")
            elif is_blacklisted:
                print("  🚫 BLACKLISTED")
            else:
                print("  ⚪ NEUTRAL")
            
            # Get profile if available
            profile = self.store.get_dev_profile(dev_wallet)
            if profile:
                print(f"\nProfile:")
                print(f"  Last updated: {profile.get('updated_at', 'Unknown')}")
                print(f"  Peak MC: ${profile.get('best_peak_mc_usd', 0):,.0f}")
                print(f"  Tokens launched: {profile.get('num_tokens_launched', 0)}")
                print(f"  Rugs (90d): {profile.get('num_rugs_90d', 0)}")
                print(f"  Recent launches (7d): {profile.get('tokens_launched_7d', 0)}")
                print(f"  Avg holders (30m): {profile.get('avg_holder_count_30m', 0)}")
            else:
                print(f"\nProfile: Not cached")
            
            # Check last attempt
            last_attempt = self.store.get_dev_last_attempt(dev_wallet)
            if last_attempt:
                print(f"\nLast attempt: {last_attempt}")
            else:
                print(f"\nLast attempt: Never")
            
            print(f"{'='*50}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def list_positions(self, status: str = 'all'):
        """List positions"""
        try:
            if status == 'open':
                positions = self.store.get_open_positions()
                title = "OPEN POSITIONS"
            elif status == 'closed':
                positions = [p for p in self.store.positions_cache.values() if p.get('status') == 'closed']
                title = "CLOSED POSITIONS"
            else:
                positions = list(self.store.positions_cache.values())
                title = "ALL POSITIONS"
            
            print(f"\n{'='*80}")
            print(f"{title} ({len(positions)} total)")
            print(f"{'='*80}")
            
            if not positions:
                print("No positions found")
                print(f"{'='*80}\n")
                return
            
            # Table header
            print(f"{'Token':<12} {'Status':<8} {'Entry':<8} {'PnL':<10} {'Time':<20} {'Reason':<15}")
            print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*20} {'-'*15}")
            
            for pos in positions:
                ticker = pos.get('ticker', 'UNKNOWN')[:11]
                status = pos.get('status', 'unknown')
                entry_sol = pos.get('entry_sol', 0)
                pnl = pos.get('realized_pnl', 0) if status == 'closed' else 0
                created_at = pos.get('created_at', '')[:19]
                reason = pos.get('exit_reason', '')[:14] if status == 'closed' else ''
                
                print(f"{ticker:<12} {status:<8} {entry_sol:<8.3f} {pnl:<+10.4f} {created_at:<20} {reason:<15}")
            
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def show_stats(self):
        """Show statistics"""
        try:
            stats = self.store.get_stats()
            
            print(f"\n{'='*50}")
            print(f"SNIPER BOT STATISTICS")
            print(f"{'='*50}")
            
            print(f"Storage:")
            print(f"  Path: {stats['storage_path']}")
            print(f"  Total positions: {stats['total_positions']}")
            print(f"  Open positions: {stats['open_positions']}")
            print(f"  Total events: {stats['total_events']}")
            
            print(f"\nDeveloper Management:")
            print(f"  Whitelisted devs: {stats['whitelisted_devs']}")
            print(f"  Blacklisted devs: {stats['blacklisted_devs']}")
            print(f"  Cached profiles: {stats['cached_profiles']}")
            
            # Calculate basic PnL
            closed_positions = [p for p in self.store.positions_cache.values() if p.get('status') == 'closed']
            if closed_positions:
                total_pnl = sum(p.get('realized_pnl', 0) for p in closed_positions)
                winning_trades = len([p for p in closed_positions if p.get('realized_pnl', 0) > 0])
                losing_trades = len([p for p in closed_positions if p.get('realized_pnl', 0) <= 0])
                win_rate = (winning_trades / len(closed_positions)) * 100 if closed_positions else 0
                
                print(f"\nPerformance:")
                print(f"  Total PnL: {total_pnl:+.4f} SOL")
                print(f"  Winning trades: {winning_trades}")
                print(f"  Losing trades: {losing_trades}")
                print(f"  Win rate: {win_rate:.1f}%")
            
            print(f"{'='*50}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def export_data(self, output_file: str):
        """Export data to JSON file"""
        try:
            export_data = {
                'positions': list(self.store.positions_cache.values()),
                'dev_profiles': dict(self.store.dev_profiles_cache),
                'whitelist': list(self.store.whitelist_cache),
                'blacklist': list(self.store.blacklist_cache),
                'exported_at': datetime.now().isoformat()
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Data exported to {output_file}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def tail_events(self, lines: int = 50):
        """Show recent events"""
        try:
            events_file = self.store.events_file
            
            if not events_file.exists():
                print("No events file found")
                return
            
            print(f"\n{'='*80}")
            print(f"RECENT EVENTS (last {lines} lines)")
            print(f"{'='*80}")
            
            # Read last N lines
            with open(events_file, 'r') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            for line in recent_lines:
                try:
                    event = json.loads(line.strip())
                    timestamp = event.get('timestamp', '')[:19]
                    event_type = event.get('type', 'UNKNOWN')
                    data = event.get('data', {})
                    reason = event.get('reason', '')
                    
                    ticker = data.get('ticker', 'UNKNOWN')
                    print(f"{timestamp} {event_type:<15} {ticker:<10} {reason}")
                except:
                    print(line.strip())
            
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Sniper Bot Management CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Blacklist commands
    blacklist_add = subparsers.add_parser('add-blacklist', help='Add developer to blacklist')
    blacklist_add.add_argument('dev_wallet', help='Developer wallet address')
    blacklist_add.add_argument('reason', help='Reason for blacklisting')
    
    blacklist_remove = subparsers.add_parser('remove-blacklist', help='Remove developer from blacklist')
    blacklist_remove.add_argument('dev_wallet', help='Developer wallet address')
    
    # Whitelist commands
    whitelist_add = subparsers.add_parser('add-whitelist', help='Add developer to whitelist')
    whitelist_add.add_argument('dev_wallet', help='Developer wallet address')
    whitelist_add.add_argument('--score', type=float, help='Risk score (default: 60.0)')
    
    whitelist_remove = subparsers.add_parser('remove-whitelist', help='Remove developer from whitelist')
    whitelist_remove.add_argument('dev_wallet', help='Developer wallet address')
    
    # Info commands
    show_dev = subparsers.add_parser('show-dev', help='Show developer information')
    show_dev.add_argument('dev_wallet', help='Developer wallet address')
    
    positions = subparsers.add_parser('positions', help='List positions')
    positions.add_argument('--status', choices=['open', 'closed', 'all'], default='all',
                          help='Position status filter')
    
    stats = subparsers.add_parser('stats', help='Show statistics')
    
    export = subparsers.add_parser('export', help='Export data to JSON')
    export.add_argument('output_file', help='Output file path')
    
    events = subparsers.add_parser('events', help='Show recent events')
    events.add_argument('--lines', type=int, default=50, help='Number of recent events to show')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    cli = SniperCLI()
    
    try:
        if args.command == 'add-blacklist':
            cli.add_blacklist(args.dev_wallet, args.reason)
        elif args.command == 'remove-blacklist':
            cli.remove_blacklist(args.dev_wallet)
        elif args.command == 'add-whitelist':
            cli.add_whitelist(args.dev_wallet, args.score)
        elif args.command == 'remove-whitelist':
            cli.remove_whitelist(args.dev_wallet)
        elif args.command == 'show-dev':
            cli.show_dev(args.dev_wallet)
        elif args.command == 'positions':
            cli.list_positions(args.status)
        elif args.command == 'stats':
            cli.show_stats()
        elif args.command == 'export':
            cli.export_data(args.output_file)
        elif args.command == 'events':
            cli.tail_events(args.lines)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()