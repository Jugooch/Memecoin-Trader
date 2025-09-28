"""
Discord Bot for Wallet Proof Commands
Completely isolated from main trading bot
Handles /wallet-proof command to show wallet holdings
"""

import discord
from discord import app_commands
from discord.ext import commands
import asyncio
import logging
from typing import Dict, Optional
import yaml
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.clients.moralis_client import MoralisClient
from src.utils.config_loader import load_config

class WalletProofBot(commands.Bot):
    def __init__(self, config: Dict):
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix='!', intents=intents)

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Get bot configuration
        bot_config = config.get('discord_bot', {})
        self.wallet_proofs_channel_id = bot_config.get('wallet_proofs_channel_id')
        self.tracked_wallet = config.get('price_monitor', {}).get('tracked_wallet', '4bmuhbVQPbVmXuqPHysyqGVq3UBou8NL9ukL4MwshGob')

        # Initialize Moralis client
        self.moralis = MoralisClient(
            api_keys=config.get('moralis_keys', []),
            api_optimization_config=config.get('api_optimization', {})
        )

        # Cache for our released tokens
        self.our_tokens = {}

        self.logger.info("WalletProofBot initialized")

    async def setup_hook(self):
        """Setup hook for the bot"""
        await self.tree.sync()
        self.logger.info(f"Synced {len(self.tree.get_commands())} command(s)")

    async def on_ready(self):
        """Event triggered when bot is ready"""
        self.logger.info(f'Logged in as {self.user} (ID: {self.user.id})')

        # Load our released tokens on startup
        await self.load_our_tokens()

    async def load_our_tokens(self):
        """Load tokens created by our tracked wallet"""
        try:
            portfolio = await self.moralis.get_wallet_portfolio(self.tracked_wallet)

            if portfolio and portfolio.get('tokens'):
                for token in portfolio['tokens']:
                    mint = token.get('mint')
                    if not mint or token.get('possible_spam'):
                        continue

                    # Get token metadata
                    metadata = await self.moralis.get_token_metadata(mint)
                    price_details = await self.moralis.get_current_price_with_details(mint, fresh=True)

                    symbol = price_details.get('symbol') or metadata.get('symbol', 'UNKNOWN')
                    name = price_details.get('name') or metadata.get('name', 'Unknown')
                    logo = price_details.get('logo')

                    self.our_tokens[symbol.upper()] = {
                        'mint': mint,
                        'symbol': symbol,
                        'name': name,
                        'logo': logo,
                        'decimals': metadata.get('decimals', 9)
                    }

                self.logger.info(f"Loaded {len(self.our_tokens)} tokens: {list(self.our_tokens.keys())}")

        except Exception as e:
            self.logger.error(f"Error loading our tokens: {e}")

    async def get_wallet_holdings(self, wallet_address: str) -> Dict:
        """Get wallet holdings for SOL and our released tokens"""
        try:
            holdings = {
                'sol_balance': 0,
                'tokens': {},
                'total_value_usd': 0
            }

            # Get wallet portfolio (includes SOL balance)
            portfolio = await self.moralis.get_wallet_portfolio(wallet_address)

            # Get SOL balance from portfolio
            if portfolio:
                native_balance = portfolio.get('native_balance', {})
                sol_balance = float(native_balance.get('solana', 0))
                holdings['sol_balance'] = sol_balance

                # Get SOL price using wrapped SOL address
                sol_mint = "So11111111111111111111111111111111111111112"
                sol_price_data = await self.moralis.get_current_price_with_details(sol_mint, fresh=True)
                sol_price = sol_price_data.get('price', 0)

                if sol_price == 0:
                    # If Moralis fails, we can't calculate USD value
                    self.logger.error("Failed to fetch SOL price")
                    holdings['sol_value_usd'] = 0
                else:
                    holdings['sol_value_usd'] = sol_balance * sol_price

                holdings['total_value_usd'] += holdings['sol_value_usd']

            # Process tokens from portfolio
            if portfolio and portfolio.get('tokens'):
                for token in portfolio['tokens']:
                    mint = token.get('mint')

                    # Check if this is one of our tokens
                    for symbol, token_info in self.our_tokens.items():
                        if token_info['mint'] == mint:
                            # Get current price
                            price_details = await self.moralis.get_current_price_with_details(mint, fresh=True)
                            price = price_details.get('price', 0)

                            # Calculate holdings
                            # Moralis already provides the amount as a converted float, not raw
                            actual_amount = float(token.get('amount', 0))
                            value_usd = actual_amount * price if price > 0 else 0

                            holdings['tokens'][symbol] = {
                                'symbol': symbol,
                                'name': token_info['name'],
                                'amount': actual_amount,
                                'price': price,
                                'value_usd': value_usd,
                                'logo': token_info.get('logo')
                            }

                            holdings['total_value_usd'] += value_usd
                            break

            return holdings

        except Exception as e:
            self.logger.error(f"Error getting wallet holdings: {e}")
            raise

    def format_holdings_embed(self, wallet_address: str, holdings: Dict) -> discord.Embed:
        """Format holdings into a Discord embed"""
        # Create main embed
        embed = discord.Embed(
            title="🔐 Wallet Proof",
            description=f"**Address:** `{wallet_address[:8]}...{wallet_address[-8:]}`",
            color=0x3498DB
        )

        # Add SOL balance
        sol_balance = holdings['sol_balance']
        sol_value = holdings.get('sol_value_usd', 0)
        embed.add_field(
            name="💎 SOL",
            value=f"**Amount:** {sol_balance:.4f} SOL\n**Value:** ${sol_value:.2f}",
            inline=False
        )

        # Add token holdings
        if holdings['tokens']:
            for symbol, token_data in holdings['tokens'].items():
                amount = token_data['amount']
                value = token_data['value_usd']
                price = token_data['price']

                # Format amount based on size
                if amount >= 1_000_000:
                    amount_str = f"{amount/1_000_000:.2f}M"
                elif amount >= 1_000:
                    amount_str = f"{amount/1_000:.2f}K"
                else:
                    amount_str = f"{amount:.4f}"

                embed.add_field(
                    name=f"🪙 {symbol}",
                    value=f"**Amount:** {amount_str}\n**Value:** ${value:.2f}\n**Price:** ${price:.8f}",
                    inline=True
                )

        # Add total value
        total_value = holdings['total_value_usd']
        embed.add_field(
            name="💰 Total Portfolio Value",
            value=f"**${total_value:,.2f} USD**",
            inline=False
        )

        # Add footer
        embed.set_footer(text="AZ Coin Bros • Wallet Verification")
        embed.timestamp = discord.utils.utcnow()

        return embed


# Create the bot instance
bot = None

async def setup_bot():
    """Setup and return the bot instance"""
    global bot

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config = load_config()

    # Get Discord bot token
    bot_token = config.get('discord_bot', {}).get('bot_token')
    if not bot_token:
        raise ValueError("Discord bot token not found in config! Add discord_bot.bot_token to your config.yml")

    # Create bot instance
    bot = WalletProofBot(config)

    # Register the slash command
    @bot.tree.command(name='wallet-proof', description='Verify wallet holdings for SOL and AZ Coin Bros tokens')
    async def wallet_proof(interaction: discord.Interaction, address: str):
        """Handle /wallet-proof command"""

        # Defer response as this might take a moment
        await interaction.response.defer()

        try:
            # Validate address format (basic check)
            if len(address) < 32 or len(address) > 44:
                await interaction.followup.send(
                    embed=discord.Embed(
                        title="❌ Invalid Address",
                        description="Please provide a valid Solana wallet address",
                        color=0xFF0000
                    )
                )
                return

            # Check if we should only respond in wallet-proofs channel
            if bot.wallet_proofs_channel_id and interaction.channel_id != bot.wallet_proofs_channel_id:
                channel_mention = f"<#{bot.wallet_proofs_channel_id}>"
                await interaction.followup.send(
                    embed=discord.Embed(
                        title="⚠️ Wrong Channel",
                        description=f"Please use the {channel_mention} channel for wallet proofs",
                        color=0xFFA500
                    ),
                    ephemeral=True
                )
                return

            # Get wallet holdings
            holdings = await bot.get_wallet_holdings(address)

            # Create and send embed
            embed = bot.format_holdings_embed(address, holdings)
            await interaction.followup.send(embed=embed)

            bot.logger.info(f"Wallet proof requested for {address[:8]}... by {interaction.user}")

        except Exception as e:
            bot.logger.error(f"Error processing wallet-proof command: {e}")
            await interaction.followup.send(
                embed=discord.Embed(
                    title="❌ Error",
                    description="Failed to retrieve wallet information. Please try again later.",
                    color=0xFF0000
                )
            )

    return bot, bot_token


async def main():
    """Main entry point for the Discord bot"""
    try:
        bot, token = await setup_bot()
        await bot.start(token)
    except KeyboardInterrupt:
        logging.info("Shutting down Discord Wallet Bot...")
        if bot:
            await bot.close()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        if bot:
            await bot.close()


if __name__ == "__main__":
    asyncio.run(main())