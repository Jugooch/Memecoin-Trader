#!/usr/bin/env python3
"""
Manual buy script for tokens using pump.fun infrastructure
Uses the EXACT same client as the bot
"""

import asyncio
import sys
import os
import json
import aiohttp
import base64
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Set up logging to see what's happening
import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

# Import the actual PumpFun client the bot uses
from clients.pumpfun_client import PumpFunClient

async def manual_buy():
    """Manually buy a token using pump.fun"""
    
    # MANUAL CONFIG - Replace these values with your actual config
    MINT_ADDRESS = "3UrMErt68aBZAgq2m9YVng1KnmJEjiyxwGHpaLditQM3"  # Your pump.fun token
    USD_AMOUNT = 20.0  # Amount in USD to spend
    
    WALLET_PRIVATE_KEY = ""  # From your config
    WALLET_PUBLIC_KEY = ""   # From your config  
    QUICKNODE_ENDPOINT = "https://thrumming-lingering-shard.solana-mainnet.quiknode.pro/877f3b04f53a38f4cbedcfdb51705e772b7c45ac/"  # From your config
    
    print(f"🚀 Manual Token Buy")
    print(f"📍 Mint: {MINT_ADDRESS}")
    print(f"💵 Amount: ${USD_AMOUNT}")
    
    if "YOUR_" in MINT_ADDRESS or "YOUR_" in WALLET_PRIVATE_KEY or "YOUR_" in QUICKNODE_ENDPOINT:
        print("❌ Please update the script with your actual configuration!")
        print("Edit manual_buy.py and replace:")
        print("- MINT_ADDRESS with the token mint address")
        print("- USD_AMOUNT with the amount in USD to spend")
        print("- WALLET_PRIVATE_KEY with your private key")
        print("- WALLET_PUBLIC_KEY with your public key") 
        print("- QUICKNODE_ENDPOINT with your QuickNode RPC endpoint")
        return
    
    try:
        print(f"💼 Wallet: {WALLET_PUBLIC_KEY[:8]}...")
        
        # Load keypair
        try:
            if len(WALLET_PRIVATE_KEY.split(',')) > 1:
                # Array format [1,2,3,...]
                private_key_bytes = bytes([int(x.strip()) for x in WALLET_PRIVATE_KEY.strip('[]').split(',')])
            else:
                # Base58 format
                from solders.keypair import Keypair
                keypair_from_b58 = Keypair.from_base58_string(WALLET_PRIVATE_KEY)
                private_key_bytes = bytes(keypair_from_b58)
            
            keypair = Keypair.from_bytes(private_key_bytes)
            print("✅ Wallet loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load wallet: {e}")
            return
        
        # Get SOL price for conversion
        print("🔍 Getting current SOL price...")
        sol_price = await get_sol_price()
        if not sol_price:
            print("❌ Failed to get SOL price")
            return
            
        sol_amount = USD_AMOUNT / sol_price
        print(f"💰 Will spend: ${USD_AMOUNT} = {sol_amount:.4f} SOL (at ${sol_price:.2f}/SOL)")
        
        # Check wallet SOL balance
        print("🔍 Checking wallet balance...")
        wallet_balance = await get_wallet_balance(QUICKNODE_ENDPOINT, WALLET_PUBLIC_KEY)
        
        if wallet_balance is None:
            print("❌ Failed to get wallet balance")
            return
            
        print(f"💳 Current balance: {wallet_balance:.4f} SOL")
        
        if wallet_balance < sol_amount + 0.01:  # Extra 0.01 SOL for fees
            print(f"❌ Insufficient balance: {wallet_balance:.4f} SOL < {sol_amount + 0.01:.4f} SOL needed (including fees)")
            return
        
        # Get token info (optional, for display)
        print("🔍 Getting token info...")
        token_info = await get_token_info(MINT_ADDRESS)
        if token_info:
            print(f"📊 Token: {token_info.get('symbol', 'UNKNOWN')} - {token_info.get('name', 'Unknown Token')}")
            if 'market_cap' in token_info:
                print(f"💎 Market Cap: ${token_info['market_cap']:,.0f}")
        
        # Confirm buy
        response = input(f"\n🤔 Buy ${USD_AMOUNT} ({sol_amount:.4f} SOL) of this token? (y/N): ")
        if response.lower() != 'y':
            print("❌ Buy cancelled")
            return
            
        print(f"\n📤 Creating buy transaction for ${USD_AMOUNT}...")
        
        # Use the EXACT same client as the bot
        # Note: api_key is not actually used for local trading API, but required for init
        pumpfun_client = PumpFunClient(
            quicknode_endpoint=QUICKNODE_ENDPOINT,
            api_key="not_used_for_local_api"  # Placeholder - not used for pump portal local API
        )
        
        try:
            # Create buy transaction using the bot's actual method
            tx_result = await pumpfun_client.create_buy_transaction(
                wallet_pubkey=WALLET_PUBLIC_KEY,
                mint_address=MINT_ADDRESS,
                sol_amount=sol_amount,
                slippage_bps=3000  # 30% slippage for volatile tokens (same as bot)
            )
        finally:
            # Clean up the client
            await pumpfun_client.close()
        
        if not tx_result.get("success"):
            print(f"❌ Failed to create transaction: {tx_result.get('error')}")
            return
            
        # Get the transaction
        transaction_b64 = tx_result.get("transaction")
        if not transaction_b64:
            print("❌ No transaction returned from pump.fun")
            return
            
        print("✅ Transaction created successfully")
        
        # Optional: Simulate transaction first to see expected tokens
        print("📝 Simulating transaction...")
        sim_result = await simulate_transaction(transaction_b64, QUICKNODE_ENDPOINT)
        if sim_result.get("success"):
            print(f"✅ Simulation successful - Expected to receive tokens")
        else:
            print(f"⚠️ Simulation warning: {sim_result.get('error', 'Unknown error')}")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("❌ Buy cancelled")
                return
        
        print("📝 Signing and sending transaction...")
        
        # Sign and send transaction (exact same as bot)
        send_result = await sign_and_send_transaction(transaction_b64, keypair, QUICKNODE_ENDPOINT)
        
        if send_result.get("success"):
            tx_signature = send_result.get("signature")
            print(f"🎉 SUCCESS! Transaction sent")
            print(f"📋 TX: {tx_signature}")
            print(f"🔗 View: https://solscan.io/tx/{tx_signature}")
            
            # Wait and check new balance
            print("\n⏳ Waiting 10 seconds to check results...")
            await asyncio.sleep(10)
            
            # Check token balance
            token_balance = await get_token_balance(QUICKNODE_ENDPOINT, WALLET_PUBLIC_KEY, MINT_ADDRESS)
            if token_balance and token_balance > 0:
                print(f"💰 Token balance: {token_balance:,.0f} tokens")
                print(f"✅ Successfully bought tokens!")
            else:
                print("⚠️ Token balance not yet showing - transaction may still be processing")
                print("Check your wallet or Solscan for confirmation")
                
        else:
            print(f"❌ Failed to send transaction: {send_result.get('error')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def get_sol_price():
    """Get current SOL price in USD"""
    try:
        async with aiohttp.ClientSession() as session:
            # Try CoinGecko API
            async with session.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd",
                timeout=5
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('solana', {}).get('usd')
    except:
        pass
    
    # Fallback to a reasonable estimate
    return 140.0  # Update this with current SOL price if API fails

async def get_wallet_balance(rpc_endpoint, wallet_address):
    """Get SOL balance for a wallet"""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getBalance",
        "params": [wallet_address]
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_endpoint, json=payload) as response:
                result = await response.json()
                if 'result' in result and 'value' in result['result']:
                    lamports = result['result']['value']
                    return lamports / 1e9  # Convert lamports to SOL
    except Exception as e:
        print(f"Error getting balance: {e}")
    
    return None

async def get_token_balance(rpc_endpoint, wallet_address, mint_address):
    """Get token balance for a wallet"""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenAccountsByOwner",
        "params": [
            wallet_address,
            {"mint": mint_address},
            {"encoding": "jsonParsed"}
        ]
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_endpoint, json=payload) as response:
                result = await response.json()
                if result and 'result' in result:
                    for account in result['result']['value']:
                        if account['account']['data']['parsed']['info']['mint'] == mint_address:
                            amount = float(account['account']['data']['parsed']['info']['tokenAmount']['amount'])
                            decimals = account['account']['data']['parsed']['info']['tokenAmount']['decimals']
                            return amount / (10 ** decimals)
    except Exception as e:
        print(f"Error getting token balance: {e}")
    
    return 0

async def get_token_info(mint_address):
    """Get basic token info from pump.fun"""
    try:
        async with aiohttp.ClientSession() as session:
            # Try to get token info from pump.fun
            async with session.get(
                f"https://frontend-api.pump.fun/coins/{mint_address}",
                timeout=5
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'symbol': data.get('symbol'),
                        'name': data.get('name'),
                        'market_cap': data.get('usd_market_cap')
                    }
    except:
        pass
    return None


async def simulate_transaction(transaction_b64, rpc_endpoint):
    """Simulate transaction to check if it will succeed"""
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "simulateTransaction",
            "params": [
                transaction_b64,
                {
                    "encoding": "base64",
                    "commitment": "processed",
                    "replaceRecentBlockhash": False,
                    "accounts": {
                        "encoding": "base64"
                    }
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_endpoint, json=payload) as response:
                result = await response.json()
                
                if 'result' in result and 'value' in result['result']:
                    value = result['result']['value']
                    if value.get('err') is None:
                        return {"success": True}
                    else:
                        return {"success": False, "error": str(value.get('err'))}
                else:
                    return {"success": False, "error": result.get('error', 'Unknown error')}
                    
    except Exception as e:
        return {"success": False, "error": str(e)}

async def sign_and_send_transaction(transaction_b64, keypair, rpc_endpoint):
    """Sign and send transaction - EXACT COPY from bot flow"""
    try:
        # Decode transaction 
        transaction_bytes = base64.b64decode(transaction_b64)
        
        # Try to deserialize as VersionedTransaction first (Pump Portal format)
        try:
            from solders.transaction import VersionedTransaction, Transaction
            transaction = VersionedTransaction.from_bytes(transaction_bytes)
            # For VersionedTransaction, we need to sign differently  
            signed_tx = VersionedTransaction(transaction.message, [keypair])
            signed_tx_bytes = bytes(signed_tx)
            print("✅ Signed as VersionedTransaction")
        except Exception as versioned_error:
            # Fallback to regular Transaction
            try:
                transaction = Transaction.from_bytes(transaction_bytes)
                transaction.sign(keypair)
                signed_tx_bytes = bytes(transaction)
                print("✅ Signed as regular Transaction")
            except Exception as regular_error:
                print(f"❌ Failed to parse as VersionedTransaction: {versioned_error}")
                print(f"❌ Failed to parse as regular Transaction: {regular_error}")
                return {
                    "success": False,
                    "error": f"Transaction parsing failed: {regular_error}"
                }
        
        # Serialize signed transaction back to base64
        signed_tx_b64 = base64.b64encode(signed_tx_bytes).decode('utf-8')
        
        # Send transaction via QuickNode RPC (exact same as bot)
        payload = {
            "jsonrpc": "2.0", 
            "id": 1,
            "method": "sendTransaction",
            "params": [
                signed_tx_b64,
                {
                    "encoding": "base64",
                    "skipPreflight": False,
                    "preflightCommitment": "processed"
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_endpoint, json=payload) as response:
                result = await response.json()
                
                if 'result' in result:
                    return {
                        "success": True,
                        "signature": result['result']
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get('error', 'Unknown error')
                    }
                    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
        
if __name__ == "__main__":
    print("=" * 60)
    print("MANUAL BUY SCRIPT - Pump.fun Token Purchase")
    print("=" * 60)
    print("\nThis script allows you to manually buy tokens using the")
    print("exact same flow as the trading bot:")
    print("1. Creates transaction via PumpPortal local API")
    print("2. Signs transaction with your wallet")
    print("3. Sends transaction via QuickNode RPC")
    print("\nMAKE SURE TO UPDATE THE CONFIGURATION VALUES FIRST!")
    print("=" * 60)
    print()
    
    asyncio.run(manual_buy())