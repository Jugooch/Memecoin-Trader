#!/usr/bin/env python3
"""
Manual sell script for GDP token using pump.fun infrastructure
"""

import asyncio
import sys
import os
import aiohttp
import base64
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders.rpc.requests import SendVersionedTransaction
from solders.rpc.config import RpcSendTransactionConfig
from solders.commitment_config import CommitmentLevel

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def manual_sell_gdp():
    """Manually sell GDP token using pump.fun"""
    
    # GDP token details
    mint_address = "Exe5MzDztYznAJaAS7GqA9MCzBNC5nZkmLELThRSpump"
    
    # MANUAL CONFIG - Replace these values with your actual config
    WALLET_PRIVATE_KEY = "51sxhdpjcX15bVsJTSfTC4MJQymhSwDJyegcJyDm8vd6xPTkM1qwrhZrUgMEYgM7GWbE5aVWNyUUGy36NAvhkL1S"  # From your config
    WALLET_PUBLIC_KEY = "FosoFG8fAN4GEAW5GDLf6P2XjFiV2L2RD2yjSrWBedsr"   # From your config  
    QUICKNODE_ENDPOINT = "https://thrumming-lingering-shard.solana-mainnet.quiknode.pro/877f3b04f53a38f4cbedcfdb51705e772b7c45ac/"  # From your config
    
    print(f"🚀 Manual GDP Token Sell")
    print(f"📍 Mint: {mint_address}")
    
    if "YOUR_" in WALLET_PRIVATE_KEY or "YOUR_" in QUICKNODE_ENDPOINT:
        print("❌ Please update the script with your actual wallet and QuickNode details!")
        print("Edit manual_sell_gdp.py and replace:")
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
        
        # Get token balance via RPC
        print("🔍 Checking current token balance...")
        token_accounts_response = await get_token_accounts(QUICKNODE_ENDPOINT, WALLET_PUBLIC_KEY, mint_address)
        
        current_balance = 0
        if token_accounts_response and 'result' in token_accounts_response:
            for account in token_accounts_response['result']['value']:
                if account['account']['data']['parsed']['info']['mint'] == mint_address:
                    current_balance = float(account['account']['data']['parsed']['info']['tokenAmount']['amount'])
                    decimals = account['account']['data']['parsed']['info']['tokenAmount']['decimals']
                    current_balance = current_balance / (10 ** decimals)
                    break
        
        if current_balance <= 0:
            print(f"❌ No tokens found in wallet (balance: {current_balance})")
            return
            
        print(f"💰 Current balance: {current_balance:,.0f} GDP tokens")
        
        # Confirm sell
        response = input(f"\n🤔 Sell ALL {current_balance:,.0f} GDP tokens? (y/N): ")
        if response.lower() != 'y':
            print("❌ Sell cancelled")
            return
            
        print(f"\n📤 Creating sell transaction for {current_balance:,.0f} tokens...")
        
        # Create sell transaction via pump.fun
        tx_result = await create_sell_transaction(
            wallet_pubkey=WALLET_PUBLIC_KEY,
            mint_address=mint_address,
            token_amount=current_balance,
            slippage_bps=200  # 2% slippage for safety
        )
        
        if not tx_result.get("success"):
            print(f"❌ Failed to create transaction: {tx_result.get('error')}")
            return
            
        # Get the transaction
        transaction_b64 = tx_result.get("transaction")
        if not transaction_b64:
            print("❌ No transaction returned from pump.fun")
            return
            
        print("✅ Transaction created successfully")
        print("📝 Signing and sending transaction...")
        
        # Sign and send transaction
        send_result = await sign_and_send_transaction(transaction_b64, keypair, QUICKNODE_ENDPOINT)
        
        if send_result.get("success"):
            tx_signature = send_result.get("signature")
            print(f"🎉 SUCCESS! Transaction sent")
            print(f"📋 TX: {tx_signature}")
            print(f"🔗 View: https://solscan.io/tx/{tx_signature}")
            
            # Wait a bit then check new balance
            print("\n⏳ Waiting 10 seconds to check results...")
            await asyncio.sleep(10)
            
            # Check new balance
            new_token_accounts_response = await get_token_accounts(QUICKNODE_ENDPOINT, WALLET_PUBLIC_KEY, mint_address)
            new_balance = 0
            if new_token_accounts_response and 'result' in new_token_accounts_response:
                for account in new_token_accounts_response['result']['value']:
                    if account['account']['data']['parsed']['info']['mint'] == mint_address:
                        new_balance = float(account['account']['data']['parsed']['info']['tokenAmount']['amount'])
                        decimals = account['account']['data']['parsed']['info']['tokenAmount']['decimals']
                        new_balance = new_balance / (10 ** decimals)
                        break
            
            print(f"💰 New balance: {new_balance:,.0f} tokens")
            
            if new_balance < current_balance:
                tokens_sold = current_balance - new_balance
                print(f"✅ Successfully sold {tokens_sold:,.0f} GDP tokens!")
            else:
                print("⚠️  Balance unchanged - transaction may still be processing")
                
        else:
            print(f"❌ Failed to send transaction: {send_result.get('error')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

async def get_token_accounts(rpc_endpoint, wallet_address, mint_address):
    """Get token accounts for a wallet"""
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
    
    async with aiohttp.ClientSession() as session:
        async with session.post(rpc_endpoint, json=payload) as response:
            return await response.json()

async def create_sell_transaction(wallet_pubkey, mint_address, token_amount, slippage_bps=200):
    """Create sell transaction via pump.fun"""
    trade_data = {
        'publicKey': wallet_pubkey,
        'action': 'sell', 
        'mint': mint_address,
        'amount': token_amount,
        'denominatedInSol': 'false',  # Amount is in tokens
        'slippage': slippage_bps / 100,  # Convert bps to percentage
        'priorityFee': 0.003,
        'pool': 'auto'
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://pumpportal.fun/api/trade-local",
                json=trade_data,
                timeout=30
            ) as response:
                if response.status == 200:
                    # Pump.fun returns binary transaction data
                    response_bytes = await response.read()
                    # Convert to base64 for consistency with our signing function
                    response_b64 = base64.b64encode(response_bytes).decode('utf-8')
                    return {
                        "success": True,
                        "transaction": response_b64
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

async def sign_and_send_transaction(transaction_b64, keypair, rpc_endpoint):
    """Sign and send transaction - EXACT COPY from working bot"""
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
        
        # Send transaction via QuickNode RPC
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
    asyncio.run(manual_sell_gdp())