# Discord Notifications Setup

This guide will help you set up Discord notifications for your memecoin trading bot so you can track trades and bot status without having to check the Oracle VM every time.

## Why Discord Instead of SQLite?

The bot now uses Discord webhooks for notifications instead of relying solely on SQLite database tracking because:

- **No VM Access Needed**: See trades instantly on any device with Discord
- **Reliable**: No database corruption or permission issues 
- **Real-time**: Get notifications immediately when trades happen
- **Persistent**: Messages saved in Discord channel history
- **Mobile Friendly**: Monitor your bot from your phone

## What You'll Get

The bot will send Discord notifications for:

- 🚀 **Bot startup/shutdown** with current capital
- 🟢 **BUY trades** with token, price, confidence score, and current equity  
- 🟠 **SELL trades** with profit/loss and updated equity
- ⚠️ **Errors** like API key exhaustion or failed trades
- 📊 **Daily summaries** with P&L, win rate, and trade count
- 💓 **Heartbeat** status updates

## Setup Steps

### 1. Create Discord Webhook

1. Go to your Discord server
2. Right-click on the channel where you want notifications
3. Select **Edit Channel** → **Integrations** → **Webhooks**
4. Click **New Webhook**
5. Give it a name (e.g., "Trading Bot")
6. **Copy the Webhook URL** (you'll need this)

### 2. Update Bot Configuration

Add the webhook URL to your `config/config.yml`:

```yaml
# Add this section to your config.yml
notifications:
  discord_webhook_url: "https://discord.com/api/webhooks/YOUR_WEBHOOK_URL_HERE"
  enabled: true
```

**⚠️ IMPORTANT: Never commit your webhook URL to git! Keep it private.**

### 3. Install Required Dependencies

The bot needs `httpx` for Discord notifications:

```bash
# Activate your bot environment
source bot_env/bin/activate

# Install httpx (already added to requirements.txt)
pip install httpx>=0.24.0
```

### 4. Test Your Setup

Run the test script to verify everything works:

```bash
python test_discord.py
```

This will send several test messages to your Discord channel to confirm the setup is working.

## P&L Tracking Without SQLite

The bot now uses a lightweight JSON file (`data/pnl_state.json`) to track:

- Current equity (starting capital + realized P&L)
- Trade history and statistics
- Daily P&L breakdown
- Win/loss rates

This file is automatically created and updated with each trade, providing persistent tracking without SQLite complexity.

## Message Types

### Trade Notifications

**BUY Example:**
```
🟢 BUY: PEPE
Token: `So111111...`
Quantity: 1,000,000
Price: $0.00001234
USD Value: $12.34
Confidence: 85%
Equity: 💰 $112.34
📝 Paper Trading
```

**SELL Example:**
```
🟠 SELL: PEPE  
Token: `So111111...`
Quantity: 500,000
Price: $0.00002468
USD Value: $12.34
Realized P&L: 📈 $2.34
Equity: 💰 $114.68
📝 Paper Trading
```

### Error Notifications
```
⚠️ Error Detected
Error: All Moralis API keys exhausted!
Module: moralis_client
Action: skipping_token
```

### Daily Summary
```
📊 Daily Summary
Equity: $124.68
Daily P&L: 📈 $24.68
Total Trades: 5
Win Rate: 80.0%
Active Positions: 2
```

## Troubleshooting

### No Messages Appearing

1. **Check webhook URL**: Make sure it's correctly added to config.yml
2. **Test connectivity**: Run `python test_discord.py`
3. **Check permissions**: Ensure the webhook has permission to post in the channel
4. **Review logs**: Check bot logs for Discord-related errors

### Rate Limiting

Discord webhooks allow ~5 requests per 5 seconds. The bot handles this automatically with retry logic.

### Webhook Security

- Keep your webhook URL private (don't share or commit to git)
- If compromised, delete the webhook and create a new one
- Consider using environment variables for the webhook URL

## Benefits for Oracle VM Users

This setup is especially useful for Oracle VM users because:

1. **No SSH needed**: Monitor trades from anywhere
2. **No database issues**: Avoid SQLite permission problems
3. **Always accessible**: Discord works on all devices
4. **Historical tracking**: All notifications preserved in channel
5. **Quick debugging**: See errors immediately

## Configuration Examples

### Minimal Setup
```yaml
notifications:
  discord_webhook_url: "https://discord.com/api/webhooks/..."
  enabled: true
```

### Disable Notifications
```yaml  
notifications:
  discord_webhook_url: ""
  enabled: false
```

Your bot will now keep you informed of all trading activity through Discord, making it much easier to track performance without constantly checking the VM!