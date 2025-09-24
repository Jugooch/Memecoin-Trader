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
from src.utils.config_loader import load_config
import json

class PriceMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Webhook configuration
        self.webhook_url = config['price_monitor']['webhook_url']
        self.update_interval = config['price_monitor'].get('update_interval_minutes', 30) * 60
        self.tracked_wallet = config['price_monitor'].get('tracked_wallet', '4bmuhbVQPbVmXuqPHysyqGVq3UBou8NL9ukL4MwshGob')
        self.top_movers_count = config['price_monitor'].get('top_movers_count', 10)

        # Moralis client
        self.moralis = MoralisClient(
            api_keys=config.get('moralis_keys', []),
            api_optimization_config=config.get('api_optimization', {})
        )

        # Cache for tracking price changes
        self.price_cache = {}
        self.wallet_tokens = {}  # Tokens created by tracked wallet

        # Session for webhook posts
        self.session = None

        self.logger.info(f"Price Monitor initialized - Updates every {self.update_interval/60} minutes")

    async def _get_session(self):
        """Get or create aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self.session

    async def get_wallet_created_tokens(self) -> List[Dict]:
        """Get tokens created by the tracked wallet"""
        try:
            # Get wallet portfolio to find tokens they hold
            portfolio = await self.moralis.get_wallet_portfolio(self.tracked_wallet)

            if not portfolio:
                return []

            wallet_tokens = []

            # Check each token in portfolio
            for token in portfolio.get('tokens', []):
                mint = token.get('mint')
                if not mint or token.get('possible_spam'):
                    continue

                # Get token metadata and price with details
                metadata = await self.moralis.get_token_metadata(mint)
                price_details = await self.moralis.get_current_price_with_details(mint, fresh=True)
                price = price_details.get('price', 0)

                if price == 0:
                    continue

                # Get liquidity info
                liquidity = await self.moralis.get_token_liquidity(mint)

                # Calculate market cap
                supply = metadata.get('supply', 0)
                decimals = metadata.get('decimals', 9)
                if supply > 0:
                    actual_supply = supply / (10 ** decimals)
                    market_cap = actual_supply * price
                else:
                    market_cap = 0

                token_info = {
                    'mint': mint,
                    'symbol': price_details.get('symbol') or metadata.get('symbol', 'UNKNOWN'),
                    'name': price_details.get('name') or metadata.get('name', 'Unknown'),
                    'price': price,
                    'logo': price_details.get('logo'),
                    'liquidity_usd': liquidity.get('total_liquidity_usd', 0),
                    'market_cap': market_cap,
                    'amount_held': token.get('amount', 0),
                    'value_usd': token.get('amount', 0) * price
                }

                # Track price change if we have previous data
                if mint in self.price_cache:
                    prev_price = self.price_cache[mint]
                    if prev_price > 0:
                        token_info['price_change_pct'] = ((price - prev_price) / prev_price) * 100
                    else:
                        token_info['price_change_pct'] = 0
                else:
                    token_info['price_change_pct'] = 0

                # Update cache
                self.price_cache[mint] = price

                wallet_tokens.append(token_info)

            # Sort by value held
            wallet_tokens.sort(key=lambda x: x['value_usd'], reverse=True)

            return wallet_tokens[:20]  # Top 20 tokens from wallet

        except Exception as e:
            self.logger.error(f"Error getting wallet tokens: {e}")
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

    async def format_discord_message(self, wallet_tokens: List[Dict]) -> Dict:
        """Format the update message for Discord"""

        # Create embeds for Discord
        embeds = []

        # Timestamp for the update
        timestamp = datetime.utcnow().isoformat()

        # Create individual embeds for each token (so each can have its own thumbnail/logo)
        if wallet_tokens:
            # Single header embed
            header_embed = {
                "title": f"📊 AZ Coin Bros Price Updates",
                "description": f"Tracking {len(wallet_tokens)} tokens from wallet",
                "color": 0x3498DB,
                "timestamp": timestamp,
                "footer": {
                    "text": "AZ Coin Bros"
                }
            }
            embeds.append(header_embed)

            # Create embed for each token (limit to 8 to stay under Discord's 10 embed limit)
            for i, token in enumerate(wallet_tokens[:8], 1):
                price_emoji = "📈" if token.get('price_change_pct', 0) >= 0 else "📉"
                change_pct = token.get('price_change_pct', 0)

                # Format market cap
                mc = token.get('market_cap', 0)
                if mc >= 1_000_000:
                    mc_str = f"${mc/1_000_000:.1f}M"
                elif mc >= 1_000:
                    mc_str = f"${mc/1_000:.1f}K"
                else:
                    mc_str = f"${mc:.0f}"

                # Determine color based on price change
                if change_pct > 10:
                    color = 0x00FF00  # Green
                elif change_pct > 0:
                    color = 0x90EE90  # Light green
                elif change_pct < -10:
                    color = 0xFF0000  # Red
                else:
                    color = 0xFFA500  # Orange

                token_embed = {
                    "title": f"{i}. {token['symbol']}",
                    "description": (
                        f"**Price:** ${token['price']:.8f}\n"
                        f"**Change:** {price_emoji} {change_pct:+.1f}%\n"
                        f"**Market Cap:** {mc_str}"
                    ),
                    "color": color,
                    "footer": {"text": token['name'][:30]}
                }

                # Add logo as thumbnail if available
                if token.get('logo'):
                    token_embed["thumbnail"] = {"url": token['logo']}

                embeds.append(token_embed)

        # Remove the significant movers section and simplify the summary
        # Market summary is now optional - only if we have room in embeds
        if wallet_tokens and len(embeds) < 9:  # Leave room for summary
            gainers = [t for t in wallet_tokens if t.get('price_change_pct', 0) > 0]
            losers = [t for t in wallet_tokens if t.get('price_change_pct', 0) < 0]

            summary_embed = {
                "title": "📈 Summary",
                "description": (
                    f"**Gainers:** {len(gainers)} 📈 | **Losers:** {len(losers)} 📉\n"
                    f"**Next Update:** <t:{int((datetime.utcnow() + timedelta(seconds=self.update_interval)).timestamp())}:R>"
                ),
                "color": 0x9B59B6,
                "timestamp": timestamp
            }
            embeds.append(summary_embed)

        return {
            "username": "AZ Coin Bros Price Updates",
            "avatar_url": "https://cdn.discordapp.com/attachments/1234567890/1234567890/chart.png",
            "embeds": embeds[:10]  # Discord limit
        }

    async def send_update(self):
        """Send periodic price update to Discord"""
        try:
            # Get wallet tokens
            wallet_tokens = await self.get_wallet_created_tokens()

            if not wallet_tokens:
                self.logger.info("No tokens found to report")
                return

            # Format message
            message = await self.format_discord_message(wallet_tokens)

            # Send to Discord
            session = await self._get_session()
            async with session.post(self.webhook_url, json=message) as response:
                if response.status == 204:
                    self.logger.info(f"Price update sent successfully - {len(wallet_tokens)} tokens")
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