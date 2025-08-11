#!/usr/bin/env python3
"""
Test script to verify Discord webhook is working
Usage: python test_discord.py
"""

import asyncio
import sys
import yaml
from src.utils.discord_notifier import DiscordNotifier
from src.utils.config_loader import load_config


async def test_discord_notifications():
    """Test Discord notifications functionality"""
    
    try:
        # Load config
        config = load_config("config/config.yml")
        
        # Get webhook URL from config
        webhook_url = ""
        if "notifications" in config:
            webhook_url = config["notifications"].get("discord_webhook_url", "")
        
        if not webhook_url:
            print("❌ No Discord webhook URL found in config/config.yml")
            print("Please add your webhook URL to config/config.yml under:")
            print("notifications:")
            print("  discord_webhook_url: \"https://discord.com/api/webhooks/...\"")
            return False
        
        print(f"🧪 Testing Discord webhook: {webhook_url[:50]}...")
        
        # Initialize notifier
        notifier = DiscordNotifier(webhook_url)
        
        # Test 1: Simple text message
        print("📤 Test 1: Sending simple text message...")
        success = await notifier.send_text("✅ Discord webhook test successful! Bot is ready to send notifications.")
        if success:
            print("✅ Text message sent successfully")
        else:
            print("❌ Failed to send text message")
            return False
        
        # Test 2: Trade notification (mock buy)
        print("📤 Test 2: Sending mock BUY trade notification...")
        success = await notifier.send_trade_notification(
            side="BUY",
            symbol="TESTCOIN",
            mint_address="So11111111111111111111111111111111111111112",
            quantity=1000000,
            price=0.00001234,
            usd_amount=12.34,
            equity=112.34,
            confidence_score=85.0,
            paper_mode=True
        )
        if success:
            print("✅ BUY trade notification sent successfully")
        else:
            print("❌ Failed to send BUY trade notification")
        
        # Test 3: Trade notification (mock sell)
        print("📤 Test 3: Sending mock SELL trade notification...")
        success = await notifier.send_trade_notification(
            side="SELL",
            symbol="TESTCOIN",
            mint_address="So11111111111111111111111111111111111111112",
            quantity=500000,
            price=0.00002468,
            usd_amount=12.34,
            equity=124.68,
            realized_pnl=2.34,
            paper_mode=True
        )
        if success:
            print("✅ SELL trade notification sent successfully")
        else:
            print("❌ Failed to send SELL trade notification")
        
        # Test 4: Error notification
        print("📤 Test 4: Sending error notification...")
        success = await notifier.send_error_notification(
            "Test error message - this is just a test",
            {"test_context": "webhook_testing", "severity": "low"}
        )
        if success:
            print("✅ Error notification sent successfully")
        else:
            print("❌ Failed to send error notification")
        
        # Test 5: Portfolio summary
        print("📤 Test 5: Sending portfolio summary...")
        success = await notifier.send_summary(
            equity=124.68,
            daily_pnl=24.68,
            total_trades=5,
            win_rate=80.0,
            active_positions=2
        )
        if success:
            print("✅ Portfolio summary sent successfully")
        else:
            print("❌ Failed to send portfolio summary")
        
        # Test 6: API exhaustion alert
        print("📤 Test 6: Sending API exhaustion alert...")
        success = await notifier.send_api_exhausted_alert("Moralis", remaining_keys=2)
        if success:
            print("✅ API exhaustion alert sent successfully")
        else:
            print("❌ Failed to send API exhaustion alert")
        
        # Test 7: Heartbeat
        print("📤 Test 7: Sending heartbeat...")
        success = await notifier.send_heartbeat(
            status="Testing",
            uptime_hours=0.1,
            tokens_scanned=50,
            alpha_checks=10
        )
        if success:
            print("✅ Heartbeat sent successfully")
        else:
            print("❌ Failed to send heartbeat")
        
        # Test 8: Alpha wallet discovery notification
        print("📤 Test 8: Sending alpha wallet discovery notification...")
        success = await notifier.send_alpha_wallet_update(
            new_wallets_found=12,
            total_added=8,
            total_watching=45,
            discovery_time=127.5,
            trigger_reason="low_count"
        )
        if success:
            print("✅ Alpha wallet discovery notification sent successfully")
        else:
            print("❌ Failed to send alpha wallet discovery notification")
        
        # Cleanup
        await notifier.close()
        
        print("\n🎉 All Discord webhook tests completed successfully!")
        print("Check your Discord channel to see all the test messages.")
        print("Your bot is ready to send live trading notifications!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during Discord testing: {e}")
        return False


async def main():
    """Main test function"""
    
    print("🚀 Discord Webhook Test Script")
    print("=" * 50)
    print()
    print("This script will test your Discord webhook configuration")
    print("and send several test messages to your Discord channel.")
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Test cancelled.")
        return
    
    print("\nStarting Discord webhook tests...")
    print()
    
    success = await test_discord_notifications()
    
    if success:
        print("\n✅ SUCCESS: Discord webhook is working correctly!")
        print("Your memecoin trading bot will now send notifications to Discord.")
        sys.exit(0)
    else:
        print("\n❌ FAILED: Discord webhook test failed.")
        print("Please check your configuration and try again.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())