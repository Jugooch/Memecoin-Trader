"""
Discord Bot for Wallet Proof Commands
Completely isolated from main trading bot
Handles /wallet-proof command to show wallet holdings
Includes leveling system and price lookup
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
import json
import math
from datetime import datetime, timedelta

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.clients.moralis_client import MoralisClient
from src.clients.bitquery_client import BitqueryClient
from src.utils.config_loader import load_config

class WalletProofBot(commands.Bot):
    def __init__(self, config: Dict):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        super().__init__(command_prefix='!', intents=intents)

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Get bot configuration
        bot_config = config.get('discord_bot', {})
        self.wallet_proofs_channel_id = bot_config.get('wallet_proofs_channel_id')

        # Initialize Moralis client
        self.moralis = MoralisClient(
            api_keys=config.get('moralis_keys', []),
            api_optimization_config=config.get('api_optimization', {})
        )

        # Persistent token tracking (shared with price monitor)
        self.tokens_file = Path(__file__).parent.parent.parent / 'data' / 'tracked_tokens.json'
        self.our_tokens = self.load_tracked_tokens()

        # Leveling system
        self.xp_file = Path(__file__).parent.parent.parent / 'data' / 'discord_xp.json'
        self.xp_data = self.load_xp_data()
        self.xp_cooldowns = {}  # Track XP cooldowns per user

        self.logger.info("WalletProofBot initialized")

    async def setup_hook(self):
        """Setup hook for the bot"""
        await self.tree.sync()
        self.logger.info(f"Synced {len(self.tree.get_commands())} command(s)")

    async def on_ready(self):
        """Event triggered when bot is ready"""
        self.logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
        self.logger.info(f'Loaded {len(self.our_tokens)} tokens: {list(self.our_tokens.keys())}')

    async def on_message(self, message):
        """Handle message events for XP tracking"""
        # Ignore bot messages
        if message.author.bot:
            return

        self.logger.debug(f"Message received from {message.author} in guild {message.guild.id if message.guild else 'DM'}")

        # Process commands first
        await self.process_commands(message)

        # Award XP for messages
        await self.award_xp(message.author.id, message.guild.id if message.guild else None)

    def load_tracked_tokens(self) -> Dict:
        """Load tracked tokens from JSON file (shared with price monitor)"""
        try:
            if self.tokens_file.exists():
                with open(self.tokens_file, 'r') as f:
                    data = json.load(f)
                    # Convert from price monitor format to discord bot format
                    tokens = {}
                    for mint, info in data.items():
                        symbol = info.get('symbol', 'UNKNOWN')
                        tokens[symbol.upper()] = {
                            'mint': mint,
                            'symbol': symbol,
                            'name': info.get('name', 'Unknown'),
                            'decimals': info.get('decimals', 9),
                            'logo': None  # Will be fetched on demand
                        }
                    return tokens
            else:
                self.tokens_file.parent.mkdir(parents=True, exist_ok=True)
                return {}
        except Exception as e:
            self.logger.error(f"Error loading tracked tokens: {e}")
            return {}

    def save_tracked_tokens(self):
        """Save tracked tokens to JSON file (shared with price monitor)"""
        try:
            # Convert back to price monitor format
            data = {}
            for symbol, token_info in self.our_tokens.items():
                mint = token_info['mint']
                data[mint] = {
                    'symbol': token_info['symbol'],
                    'name': token_info['name'],
                    'decimals': token_info['decimals'],
                    'creator_wallet': token_info.get('creator_wallet', self.creator_wallets[0]),
                    'discovered_at': token_info.get('discovered_at', datetime.utcnow().isoformat())
                }
            with open(self.tokens_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving tracked tokens: {e}")

    def load_xp_data(self) -> Dict:
        """Load XP data from JSON file"""
        try:
            if self.xp_file.exists():
                with open(self.xp_file, 'r') as f:
                    return json.load(f)
            else:
                # Create data directory if it doesn't exist
                self.xp_file.parent.mkdir(parents=True, exist_ok=True)
                return {}
        except Exception as e:
            self.logger.error(f"Error loading XP data: {e}")
            return {}

    def save_xp_data(self):
        """Save XP data to JSON file"""
        try:
            with open(self.xp_file, 'w') as f:
                json.dump(self.xp_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving XP data: {e}")

    def get_user_key(self, user_id: int, guild_id: Optional[int] = None) -> str:
        """Generate a unique key for user in a guild"""
        if guild_id:
            return f"{guild_id}:{user_id}"
        return str(user_id)

    async def award_xp(self, user_id: int, guild_id: Optional[int] = None):
        """Award XP to a user for sending a message"""
        user_key = self.get_user_key(user_id, guild_id)

        # Check cooldown (60 seconds between XP awards)
        now = datetime.utcnow()
        if user_key in self.xp_cooldowns:
            time_since_last = now - self.xp_cooldowns[user_key]
            if time_since_last < timedelta(seconds=60):
                self.logger.debug(f"User {user_id} on cooldown ({60 - time_since_last.seconds}s remaining)")
                return

        self.xp_cooldowns[user_key] = now

        # Initialize user data if needed
        if user_key not in self.xp_data:
            self.xp_data[user_key] = {
                'xp': 0,
                'level': 0,
                'user_id': user_id,
                'guild_id': guild_id
            }
            self.logger.info(f"Initialized XP data for user {user_id}")

        # Award random XP between 15-25
        import random
        xp_gain = random.randint(15, 25)
        old_level = self.xp_data[user_key]['level']
        self.xp_data[user_key]['xp'] += xp_gain

        # Calculate new level
        new_level = self.calculate_level(self.xp_data[user_key]['xp'])
        self.xp_data[user_key]['level'] = new_level

        self.logger.info(f"Awarded {xp_gain} XP to user {user_id} (Total: {self.xp_data[user_key]['xp']} XP, Level: {new_level})")

        # Save data
        self.save_xp_data()
        self.logger.debug(f"Saved XP data to {self.xp_file}")

    def calculate_level(self, xp: int) -> int:
        """Calculate level from XP (level = sqrt(xp / 100))"""
        return int(math.sqrt(xp / 100))

    def xp_for_level(self, level: int) -> int:
        """Calculate XP required for a level"""
        return level * level * 100

    async def add_token_to_tracking(self, mint_address: str) -> tuple:
        """
        Add a token to the tracking list
        Returns: (success: bool, message: str, token_info: dict)
        """
        try:
            # Check if already tracked
            for token_data in self.our_tokens.values():
                if token_data['mint'] == mint_address:
                    return False, f"Token {mint_address[:8]}... is already being tracked", None

            # Get token metadata
            metadata = await self.moralis.get_token_metadata(mint_address)
            price_details = await self.moralis.get_current_price_with_details(mint_address, fresh=True)

            symbol = price_details.get('symbol') or metadata.get('symbol', 'UNKNOWN')
            name = price_details.get('name') or metadata.get('name', 'Unknown')
            logo = price_details.get('logo')

            # Add to tracked tokens
            self.our_tokens[symbol.upper()] = {
                'mint': mint_address,
                'symbol': symbol,
                'name': name,
                'logo': logo,
                'decimals': metadata.get('decimals', 9),
                'source': 'discord_command',
                'added_at': datetime.utcnow().isoformat()
            }

            # Save to persistent storage
            self.save_tracked_tokens()

            self.logger.info(f"Added token via Discord: {symbol} ({mint_address[:8]}...)")

            return True, f"Successfully added {symbol} ({name})", {
                'symbol': symbol,
                'name': name,
                'mint': mint_address
            }

        except Exception as e:
            self.logger.error(f"Error adding token {mint_address[:8]}...: {e}")
            return False, f"Error adding token: {str(e)}", None

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

    def format_price_embed(self, token_address: str, price_data: Dict, metadata: Dict) -> discord.Embed:
        """Format token price info into a Discord embed"""
        symbol = price_data.get('symbol') or metadata.get('symbol', 'UNKNOWN')
        name = price_data.get('name') or metadata.get('name', 'Unknown')
        price = price_data.get('price', 0)
        logo = price_data.get('logo')

        embed = discord.Embed(
            title=f"💰 {symbol} Price",
            description=f"**{name}**",
            color=0x3498DB
        )

        # Add price
        embed.add_field(
            name="💵 Current Price",
            value=f"${price:.10f}".rstrip('0').rstrip('.'),
            inline=False
        )

        # Calculate market cap if we have supply data
        supply = metadata.get('supply', 0)
        decimals = metadata.get('decimals', 9)
        if supply > 0 and price > 0:
            actual_supply = supply / (10 ** decimals)
            market_cap = actual_supply * price

            if market_cap >= 1_000_000:
                mc_str = f"${market_cap/1_000_000:.2f}M"
            elif market_cap >= 1_000:
                mc_str = f"${market_cap/1_000:.2f}K"
            else:
                mc_str = f"${market_cap:.2f}"

            embed.add_field(
                name="📊 Market Cap",
                value=mc_str,
                inline=True
            )

        # Add token address
        embed.add_field(
            name="🔗 Address",
            value=f"`{token_address[:8]}...{token_address[-8:]}`",
            inline=False
        )

        # Add logo as thumbnail if available
        if logo:
            embed.set_thumbnail(url=logo)

        embed.set_footer(text="AZ Coin Bros • Price Lookup")
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

    # Register the slash commands
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

    @bot.tree.command(name='price', description='Look up the current price of a token')
    async def price_lookup(interaction: discord.Interaction, address: str):
        """Handle /price command"""

        # Defer response as this might take a moment
        await interaction.response.defer()

        try:
            # Validate address format (basic check)
            if len(address) < 32 or len(address) > 44:
                await interaction.followup.send(
                    embed=discord.Embed(
                        title="❌ Invalid Address",
                        description="Please provide a valid Solana token address",
                        color=0xFF0000
                    )
                )
                return

            # Get token price and metadata
            price_data = await bot.moralis.get_current_price_with_details(address, fresh=True)
            metadata = await bot.moralis.get_token_metadata(address)

            if not price_data or price_data.get('price', 0) == 0:
                await interaction.followup.send(
                    embed=discord.Embed(
                        title="❌ Price Not Found",
                        description="Could not retrieve price data for this token. It may not be traded yet.",
                        color=0xFF0000
                    )
                )
                return

            # Create and send embed
            embed = bot.format_price_embed(address, price_data, metadata)
            await interaction.followup.send(embed=embed)

            bot.logger.info(f"Price lookup for {address[:8]}... by {interaction.user}")

        except Exception as e:
            bot.logger.error(f"Error processing price command: {e}")
            await interaction.followup.send(
                embed=discord.Embed(
                    title="❌ Error",
                    description="Failed to retrieve token price. Please try again later.",
                    color=0xFF0000
                )
            )

    @bot.tree.command(name='level', description='Check your current level and XP')
    async def check_level(interaction: discord.Interaction):
        """Handle /level command"""
        try:
            user_key = bot.get_user_key(interaction.user.id, interaction.guild.id if interaction.guild else None)

            if user_key not in bot.xp_data:
                await interaction.response.send_message(
                    embed=discord.Embed(
                        title="📊 Your Level",
                        description="You haven't earned any XP yet! Start chatting to earn XP and level up.",
                        color=0x3498DB
                    ),
                    ephemeral=True
                )
                return

            user_data = bot.xp_data[user_key]
            current_xp = user_data['xp']
            current_level = user_data['level']
            next_level = current_level + 1
            xp_needed = bot.xp_for_level(next_level)
            xp_progress = current_xp - bot.xp_for_level(current_level)
            xp_for_next = xp_needed - bot.xp_for_level(current_level)

            embed = discord.Embed(
                title=f"📊 {interaction.user.display_name}'s Level",
                color=0x3498DB
            )
            embed.add_field(
                name="🎯 Current Level",
                value=f"**Level {current_level}**",
                inline=True
            )
            embed.add_field(
                name="⭐ Total XP",
                value=f"**{current_xp:,} XP**",
                inline=True
            )
            embed.add_field(
                name="📈 Progress to Next Level",
                value=f"**{xp_progress}/{xp_for_next} XP** ({int(xp_progress/xp_for_next*100)}%)",
                inline=False
            )
            embed.set_thumbnail(url=interaction.user.display_avatar.url)
            embed.set_footer(text="Keep chatting to earn more XP!")

            await interaction.response.send_message(embed=embed, ephemeral=True)

        except Exception as e:
            bot.logger.error(f"Error processing level command: {e}")
            await interaction.response.send_message(
                embed=discord.Embed(
                    title="❌ Error",
                    description="Failed to retrieve level data. Please try again later.",
                    color=0xFF0000
                ),
                ephemeral=True
            )

    @bot.tree.command(name='rank', description='Check the level and rank of a user')
    async def check_rank(interaction: discord.Interaction, user: discord.User):
        """Handle /rank command"""
        try:
            guild_id = interaction.guild.id if interaction.guild else None
            user_key = bot.get_user_key(user.id, guild_id)

            if user_key not in bot.xp_data:
                await interaction.response.send_message(
                    embed=discord.Embed(
                        title=f"📊 {user.display_name}'s Rank",
                        description="This user hasn't earned any XP yet!",
                        color=0x3498DB
                    ),
                    ephemeral=True
                )
                return

            # Calculate rank
            guild_users = [(k, v) for k, v in bot.xp_data.items()
                          if v.get('guild_id') == guild_id]
            guild_users.sort(key=lambda x: x[1]['xp'], reverse=True)
            rank = next((i+1 for i, (k, v) in enumerate(guild_users) if k == user_key), None)

            user_data = bot.xp_data[user_key]
            current_xp = user_data['xp']
            current_level = user_data['level']

            embed = discord.Embed(
                title=f"📊 {user.display_name}'s Rank",
                color=0x3498DB
            )
            embed.add_field(
                name="🏆 Server Rank",
                value=f"**#{rank}**",
                inline=True
            )
            embed.add_field(
                name="🎯 Level",
                value=f"**{current_level}**",
                inline=True
            )
            embed.add_field(
                name="⭐ Total XP",
                value=f"**{current_xp:,} XP**",
                inline=True
            )
            embed.set_thumbnail(url=user.display_avatar.url)

            await interaction.response.send_message(embed=embed, ephemeral=True)

        except Exception as e:
            bot.logger.error(f"Error processing rank command: {e}")
            await interaction.response.send_message(
                embed=discord.Embed(
                    title="❌ Error",
                    description="Failed to retrieve rank data. Please try again later.",
                    color=0xFF0000
                ),
                ephemeral=True
            )

    @bot.tree.command(name='leaderboard', description='View the top 10 users by XP')
    async def leaderboard(interaction: discord.Interaction):
        """Handle /leaderboard command"""
        try:
            guild_id = interaction.guild.id if interaction.guild else None

            # Get users from this guild
            guild_users = [(k, v) for k, v in bot.xp_data.items()
                          if v.get('guild_id') == guild_id]

            if not guild_users:
                await interaction.response.send_message(
                    embed=discord.Embed(
                        title="🏆 Leaderboard",
                        description="No users have earned XP yet! Start chatting to be the first on the leaderboard.",
                        color=0x3498DB
                    ),
                    ephemeral=True
                )
                return

            # Sort by XP
            guild_users.sort(key=lambda x: x[1]['xp'], reverse=True)

            embed = discord.Embed(
                title="🏆 XP Leaderboard - Top 10",
                color=0xFFD700
            )

            medals = ["🥇", "🥈", "🥉"]

            for i, (user_key, user_data) in enumerate(guild_users[:10], 1):
                user_id = user_data['user_id']
                try:
                    user = await bot.fetch_user(user_id)
                    username = user.display_name
                except:
                    username = f"User {user_id}"

                medal = medals[i-1] if i <= 3 else f"#{i}"

                embed.add_field(
                    name=f"{medal} {username}",
                    value=f"**Level {user_data['level']}** • {user_data['xp']:,} XP",
                    inline=False
                )

            embed.set_footer(text="Keep chatting to climb the ranks!")
            await interaction.response.send_message(embed=embed, ephemeral=True)

        except Exception as e:
            bot.logger.error(f"Error processing leaderboard command: {e}")
            await interaction.response.send_message(
                embed=discord.Embed(
                    title="❌ Error",
                    description="Failed to retrieve leaderboard data. Please try again later.",
                    color=0xFF0000
                ),
                ephemeral=True
            )

    @bot.tree.command(name='add-token', description='[Moderator] Add a token to the tracking list')
    async def add_token(interaction: discord.Interaction, mint_address: str):
        """Handle /add-token command - moderator only"""
        try:
            # Check if user has moderator permissions
            if not interaction.user.guild_permissions.manage_messages:
                await interaction.response.send_message(
                    embed=discord.Embed(
                        title="❌ Permission Denied",
                        description="This command is only available to moderators.",
                        color=0xFF0000
                    ),
                    ephemeral=True
                )
                return

            # Defer response as this might take a moment
            await interaction.response.defer(ephemeral=True)

            # Add token
            success, message, token_info = await bot.add_token_to_tracking(mint_address)

            if success:
                embed = discord.Embed(
                    title="✅ Token Added",
                    description=message,
                    color=0x00FF00
                )

                if token_info:
                    embed.add_field(
                        name="Token Details",
                        value=f"**Symbol:** {token_info['symbol']}\n**Name:** {token_info['name']}\n**Mint:** `{token_info['mint']}`",
                        inline=False
                    )

                embed.add_field(
                    name="📊 Total Tracked",
                    value=f"Now tracking **{len(bot.our_tokens)}** tokens",
                    inline=False
                )

                bot.logger.info(f"Token added by {interaction.user}: {mint_address}")
            else:
                embed = discord.Embed(
                    title="⚠️ Token Not Added",
                    description=message,
                    color=0xFFA500
                )

            await interaction.followup.send(embed=embed, ephemeral=True)

        except Exception as e:
            bot.logger.error(f"Error processing add-token command: {e}")
            await interaction.followup.send(
                embed=discord.Embed(
                    title="❌ Error",
                    description=f"Failed to add token: {str(e)}",
                    color=0xFF0000
                ),
                ephemeral=True
            )

    @bot.tree.command(name='list-tokens', description='View all tracked tokens')
    async def list_tokens(interaction: discord.Interaction):
        """Handle /list-tokens command"""
        try:
            if not bot.our_tokens:
                await interaction.response.send_message(
                    embed=discord.Embed(
                        title="📋 Tracked Tokens",
                        description="No tokens are currently being tracked.",
                        color=0x3498DB
                    ),
                    ephemeral=True
                )
                return

            embed = discord.Embed(
                title="📋 Tracked Tokens",
                description=f"Tracking **{len(bot.our_tokens)}** tokens",
                color=0x3498DB
            )

            # List all tokens (limit to 25 for Discord embed field limit)
            token_list = []
            for i, (symbol, data) in enumerate(list(bot.our_tokens.items())[:25], 1):
                token_list.append(f"{i}. **{symbol}** - {data['name']}")

            embed.add_field(
                name="Tokens",
                value="\n".join(token_list) + ("\n*...and more*" if len(bot.our_tokens) > 25 else ""),
                inline=False
            )

            await interaction.response.send_message(embed=embed, ephemeral=True)

        except Exception as e:
            bot.logger.error(f"Error processing list-tokens command: {e}")
            await interaction.response.send_message(
                embed=discord.Embed(
                    title="❌ Error",
                    description="Failed to retrieve token list.",
                    color=0xFF0000
                ),
                ephemeral=True
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