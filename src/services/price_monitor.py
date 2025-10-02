"""
Periodic Price Update Service
Monitors top movers and specific wallet token creations
Completely isolated from main memecoin bot
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import aiohttp
from src.clients.moralis_client import MoralisClient
from src.clients.bitquery_client import BitqueryClient
from src.utils.config_loader import load_config
import json
from pathlib import Path

class PriceMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Webhook configuration
        self.webhook_url = config['price_monitor']['webhook_url']
        self.update_interval = config['price_monitor'].get('update_interval_minutes', 30) * 60

        # Manual tokens from config
        self.manual_token_mints = config['price_monitor'].get('manual_tokens', [])
        self.top_movers_count = config['price_monitor'].get('top_movers_count', 10)

        # Moralis client
        self.moralis = MoralisClient(
            api_keys=config.get('moralis_keys', []),
            api_optimization_config=config.get('api_optimization', {})
        )

        # Persistent token tracking
        self.tokens_file = Path(__file__).parent.parent.parent / 'data' / 'tracked_tokens.json'
        self.tracked_tokens = self.load_tracked_tokens()

        # Cache for tracking price changes
        self.price_cache = {}

        # Session for webhook posts
        self.session = None

        self.logger.info(f"Price Monitor initialized - Updates every {self.update_interval/60} minutes")
        self.logger.info(f"Manual tokens from config: {len(self.manual_token_mints)}")

    async def _get_session(self):
        """Get or create aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self.session

    def load_tracked_tokens(self) -> Dict:
        """Load tracked tokens from JSON file"""
        try:
            if self.tokens_file.exists():
                with open(self.tokens_file, 'r') as f:
                    return json.load(f)
            else:
                # Create data directory if it doesn't exist
                self.tokens_file.parent.mkdir(parents=True, exist_ok=True)
                return {}
        except Exception as e:
            self.logger.error(f"Error loading tracked tokens: {e}")
            return {}

    def save_tracked_tokens(self):
        """Save tracked tokens to JSON file"""
        try:
            with open(self.tokens_file, 'w') as f:
                json.dump(self.tracked_tokens, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving tracked tokens: {e}")

    async def discover_new_tokens(self):
        """Load manual tokens from config and persistent storage"""
        try:
            new_tokens_count = 0

            # Process manual tokens from config
            for mint in self.manual_token_mints:
                # Skip if already tracked
                if mint in self.tracked_tokens:
                    continue

                try:
                    # Get token metadata from Moralis
                    metadata = await self.moralis.get_token_metadata(mint)
                    price_details = await self.moralis.get_current_price_with_details(mint, fresh=True)

                    symbol = price_details.get('symbol') or metadata.get('symbol', 'UNKNOWN')
                    name = price_details.get('name') or metadata.get('name', 'Unknown')

                    # Add to tracked tokens
                    self.tracked_tokens[mint] = {
                        'symbol': symbol,
                        'name': name,
                        'decimals': metadata.get('decimals', 9),
                        'source': 'manual_config',
                        'discovered_at': datetime.utcnow().isoformat()
                    }

                    new_tokens_count += 1
                    self.logger.info(f"Added manual token: {symbol} ({mint[:8]}...)")

                except Exception as token_error:
                    self.logger.error(f"Error processing manual token {mint[:8]}...: {token_error}")
                    continue

            # Save updated tracked tokens
            if new_tokens_count > 0:
                self.save_tracked_tokens()
                self.logger.info(f"Added {new_tokens_count} new tokens. Now tracking {len(self.tracked_tokens)} total tokens")
            else:
                self.logger.info(f"All manual tokens already tracked. Total: {len(self.tracked_tokens)} tokens")

        except Exception as e:
            self.logger.error(f"Error loading manual tokens: {e}")

    async def get_all_tracked_tokens(self) -> List[Dict]:
        """Get price updates for ALL tracked tokens"""
        try:
            token_updates = []

            # Get price for each tracked token
            for mint, token_info in self.tracked_tokens.items():
                try:
                    # Get current price
                    price_details = await self.moralis.get_current_price_with_details(mint, fresh=True)
                    price = price_details.get('price', 0)

                    if price == 0:
                        self.logger.warning(f"No price found for {token_info['symbol']} ({mint})")
                        continue

                    # Calculate market cap
                    # Try to get supply from price_details first (more reliable for pump.fun tokens)
                    metadata = await self.moralis.get_token_metadata(mint)

                    # Check multiple sources for supply data
                    supply_val = metadata.get('supply', 0)

                    # For pump.fun tokens, if no supply data, use standard supply (1 billion with 6 decimals)
                    decimals = metadata.get('decimals', token_info.get('decimals', 6))

                    # Ensure decimals is an integer
                    if isinstance(decimals, str):
                        decimals = int(decimals)
                    elif decimals is None:
                        decimals = 6

                    if supply_val == 0:
                        # Pump.fun standard: 1,000,000,000 tokens with 6 decimals
                        # This gives actual supply of 1 billion
                        self.logger.debug(f"Using standard pump.fun supply for {token_info['symbol']}")
                        supply_val = 1_000_000_000 * (10 ** 6)  # 1 billion tokens at 6 decimals

                    if supply_val > 0:
                        actual_supply = supply_val / (10 ** decimals)
                        market_cap = actual_supply * price
                    else:
                        market_cap = 0

                    # Calculate price change
                    price_change_pct = 0
                    if mint in self.price_cache:
                        prev_price = self.price_cache[mint]
                        if prev_price > 0:
                            price_change_pct = ((price - prev_price) / prev_price) * 100

                    # Update cache
                    self.price_cache[mint] = price

                    token_update = {
                        'mint': mint,
                        'symbol': token_info['symbol'],
                        'name': token_info['name'],
                        'price': price,
                        'logo': price_details.get('logo'),
                        'market_cap': market_cap,
                        'price_change_pct': price_change_pct
                    }

                    token_updates.append(token_update)

                except Exception as e:
                    self.logger.error(f"Error getting price for {token_info.get('symbol')}: {e}")
                    continue

            # Sort by market cap (highest first)
            token_updates.sort(key=lambda x: x['market_cap'], reverse=True)

            return token_updates

        except Exception as e:
            self.logger.error(f"Error getting tracked tokens: {e}")
            return []

    async def get_top_movers(self) -> Tuple[List[Dict], List[Dict]]:
        """Get top gainers and losers from recent swaps activity"""
        try:
            # We'll analyze recent high-volume tokens
            # Get recent swaps to identify active tokens
            top_gainers = []
            top_losers = []

            # This would need to be expanded with more sophisticated logic
            # For now, returning empty as Moralis doesn't directly provide top movers
            # You'd need to track multiple tokens over time

            return top_gainers, top_losers

        except Exception as e:
            self.logger.error(f"Error getting top movers: {e}")
            return [], []

    async def format_discord_message(self, all_tokens: List[Dict]) -> Dict:
        """Format the update message for Discord - showing top 3 tokens by market cap"""

        # Timestamp for the update
        timestamp = datetime.utcnow().isoformat()

        # Get top 3 tokens by market cap
        top_tokens = all_tokens[:3]

        # Count gainers/losers from all tokens
        gainers = [t for t in all_tokens if t.get('price_change_pct', 0) > 0]
        losers = [t for t in all_tokens if t.get('price_change_pct', 0) < 0]

        # Build description with top 3 tokens
        description_lines = ["\u200b"]  # Add invisible character for spacing after title

        for i, token in enumerate(top_tokens, 1):
            price_emoji = "📈" if token.get('price_change_pct', 0) >= 0 else "📉"
            change_pct = token.get('price_change_pct', 0)

            # Format market cap - fix the calculation
            mc = token.get('market_cap', 0)
            if mc >= 1_000_000:
                mc_str = f"${mc/1_000_000:.2f}M"
            elif mc >= 1_000:
                mc_str = f"${mc/1_000:.2f}K"
            elif mc > 0:
                mc_str = f"${mc:.2f}"
            else:
                mc_str = "$0"

            # Format price - show fewer decimals for readability
            price = token['price']
            if price >= 0.01:
                price_str = f"${price:.4f}"
            else:
                price_str = f"${price:.8f}".rstrip('0').rstrip('.')

            # Format: Name - $SYMBOL, address below (tighter), then price info
            description_lines.append(
                f"**{i}. {token['name']} - ${token['symbol']}**"
                f"`{token['mint']}`\n"
                f"{price_emoji} {price_str} ({change_pct:+.1f}%) • MC: {mc_str}\n"  # Newline for spacing between tokens
            )

        # Add spacing before bottom section (one newline already exists from last token)
        description_lines.append("\u200b")  # Just invisible character, no extra newline

        # Create single comprehensive embed
        embed = {
            "title": "💰 AZ Coin Bros - Top 3 Tokens",
            "description": "\n".join(description_lines),
            "color": 0x3498DB,
            "fields": [
                {
                    "name": "\u200b",  # Invisible field name
                    "value": (
                        f"**Total Tracked:** {len(all_tokens)} tokens\n"
                        f"**Gainers:** {len(gainers)} 📈 | **Losers:** {len(losers)} 📉"
                    ),
                    "inline": False
                }
            ],
            "footer": {
                "text": f"Next update in {self.update_interval//60} minutes • Showing top 3 by market cap"
            },
            "timestamp": timestamp
        }

        return {
            "username": "AZ Coin Bros Price Updates",
            "embeds": [embed]
        }

    async def send_update(self):
        """Send periodic price update to Discord"""
        try:
            # First, discover any new tokens from creator wallets
            await self.discover_new_tokens()

            # Get ALL tracked tokens
            all_tokens = await self.get_all_tracked_tokens()

            if not all_tokens:
                self.logger.info("No tokens found to report")
                return

            # Format message
            message = await self.format_discord_message(all_tokens)

            # Send to Discord
            session = await self._get_session()
            async with session.post(self.webhook_url, json=message) as response:
                if response.status == 204:
                    self.logger.info(f"Price update sent successfully - {len(all_tokens)} tokens")
                else:
                    self.logger.error(f"Failed to send update: {response.status} - {await response.text()}")

        except Exception as e:
            self.logger.error(f"Error sending price update: {e}")

    async def run_forever(self):
        """Main loop for periodic updates"""
        self.logger.info("Starting Price Monitor service...")

        # Initial update
        await self.send_update()

        while True:
            try:
                # Wait for next update interval
                await asyncio.sleep(self.update_interval)

                # Send update
                await self.send_update()

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    async def close(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        await self.moralis.close()
        # BitQuery client doesn't have a close method, but we can clean up if needed
        # The transport will be closed automatically when the program exits


async def main():
    """Main entry point for the price monitor service"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config
    config = load_config()

    # Create and run monitor
    monitor = PriceMonitor(config)

    try:
        await monitor.run_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down Price Monitor...")
    finally:
        await monitor.close()


if __name__ == "__main__":
    asyncio.run(main())