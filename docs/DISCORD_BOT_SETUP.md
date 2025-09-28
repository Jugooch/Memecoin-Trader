# Discord Wallet Proof Bot Setup

This guide will help you set up the Discord bot for `/wallet-proof` commands that allow users to verify their wallet holdings.

## Features

- `/wallet-proof ADDRESS` - Shows SOL balance and holdings of all AZ Coin Bros tokens
- Automatically detects and tracks tokens from your configured wallet
- Beautiful embedded messages with token logos
- Optional channel restriction (can limit to #wallet-proofs channel only)

## Setup Steps

### 1. Create a Discord Application & Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **New Application**
3. Give it a name (e.g., "AZ Coin Bros Wallet Bot")
4. Go to the **Bot** section in the left sidebar
5. Click **Reset Token** and copy the bot token (you'll need this)
6. Under **Privileged Gateway Intents**, enable:
   - Message Content Intent (required for slash commands)

### 2. Invite the Bot to Your Server

1. In the Discord Developer Portal, go to **OAuth2** → **URL Generator**
2. Under **Scopes**, select:
   - `bot`
   - `applications.commands`
3. Under **Bot Permissions**, select:
   - Send Messages
   - Embed Links
   - Use Slash Commands
4. Copy the generated URL and open it in your browser
5. Select your server and authorize the bot

### 3. Get Your Channel ID (Optional)

If you want to restrict the command to a specific channel:

1. Enable Developer Mode in Discord (User Settings → Advanced → Developer Mode)
2. Right-click on your #wallet-proofs channel
3. Click **Copy Channel ID**

### 4. Configure the Bot

Add to your `config/config.yml`:

```yaml
# Discord Bot Configuration (for /wallet-proof command)
discord_bot:
  bot_token: "YOUR_BOT_TOKEN_HERE"          # From Step 1
  wallet_proofs_channel_id: 123456789012345  # Optional: From Step 3 (or use null for any channel)
```

### 5. Install Dependencies

```bash
# Activate your virtual environment
source bot_env/bin/activate  # or venv/bin/activate

# Install discord.py
pip install discord.py>=2.3.2
```

### 6. Run the Bot

```bash
# Using the startup script
./scripts/start_discord_bot.sh

# Or run directly
python -m src.services.discord_wallet_bot
```

## Using the Bot

Once running, users can type:

```
/wallet-proof ADDRESS
```

Example:
```
/wallet-proof 4bmuhbVQPbVmXuqPHysyqGVq3UBou8NL9ukL4MwshGob
```

The bot will respond with:
- SOL balance and USD value
- Holdings of all your released tokens (R6, Tylenol, etc.)
- Total portfolio value in USD

## Response Format

The bot sends an embedded message showing:

```
🔐 Wallet Proof
Address: 4bmuhb...shGob

💎 SOL
Amount: 12.3456 SOL
Value: $1,234.56

🪙 R6
Amount: 1.23M
Value: $456.78
Price: $0.00000123

🪙 TYLENOL
Amount: 456.78K
Value: $123.45
Price: $0.00000456

💰 Total Portfolio Value
$1,814.79 USD
```

## Running as a Service (Optional)

To keep the bot running 24/7 on your server:

### Using systemd (Linux)

Create `/etc/systemd/system/discord-wallet-bot.service`:

```ini
[Unit]
Description=Discord Wallet Proof Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/Memecoin-Trader
ExecStart=/path/to/venv/bin/python -m src.services.discord_wallet_bot
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable discord-wallet-bot
sudo systemctl start discord-wallet-bot
```

### Using PM2 (Node.js Process Manager)

```bash
# Install PM2
npm install -g pm2

# Start the bot
pm2 start "python -m src.services.discord_wallet_bot" --name discord-wallet-bot

# Save PM2 configuration
pm2 save
pm2 startup
```

## Architecture Notes

- **Completely Isolated**: The Discord bot runs separately from your main trading bot
- **Shared Infrastructure**: Uses the same Moralis client and configuration
- **Read-Only**: Only reads wallet data, doesn't perform any trades
- **Cached Token List**: Loads your released tokens on startup for efficient lookups

## Security Considerations

- **Never share your bot token** - Keep it secret
- **Use environment variables** for sensitive data in production
- **Channel restriction** - Consider limiting to specific channel to prevent spam
- **Rate limiting** - Discord handles this automatically for slash commands

## Troubleshooting

### Bot Not Responding to Commands

1. Check that the bot is online in your server (should show as online in member list)
2. Ensure slash commands are synced (may take up to 1 hour initially)
3. Try kicking and re-inviting the bot with proper permissions

### "Application did not respond" Error

This means the bot took too long to fetch wallet data:
- Check your Moralis API keys are valid
- Ensure the bot has internet access
- Check logs for specific errors

### Commands Not Showing Up

Slash commands can take up to 1 hour to sync globally. For instant testing:
1. Use a test server with fewer members
2. The bot automatically syncs on startup

## Monitoring

Check bot logs for:
```bash
tail -f discord_wallet_bot.log  # If logging to file
```

Or if running in terminal, you'll see:
```
2024-01-15 10:30:00 - INFO - Logged in as WalletBot#1234
2024-01-15 10:30:01 - INFO - Synced 1 command(s)
2024-01-15 10:30:02 - INFO - Loaded 2 tokens: ['R6', 'TYLENOL']
```

## Future Enhancements

Potential additions:
- `/leaderboard` - Show top holders
- `/token-info SYMBOL` - Detailed token statistics
- `/whale-alert` - Monitor large transactions
- DM notifications for price alerts

The bot provides a simple, secure way for your community to verify their holdings without needing to connect wallets or use third-party services!