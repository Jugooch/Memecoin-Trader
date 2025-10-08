# Integration Tests - Setup Guide

Integration tests run against **real Solana devnet** to validate the frontrun bot works with actual blockchain.

## Quick Start

### 1. Generate Test Wallet

```bash
# Generate new keypair
solana-keygen new --outfile devnet-test-wallet.json --no-bip39-passphrase

# This will output:
# pubkey: <YOUR_PUBLIC_KEY>
# Save this address!
```

### 2. Fund Test Wallet

Visit **https://faucet.solana.com** and request devnet SOL for your wallet address.

Alternatively, use the Solana CLI:
```bash
# Set to devnet
solana config set --url devnet

# Request airdrop (may be rate-limited)
solana airdrop 2 <YOUR_PUBLIC_KEY>

# Check balance
solana balance <YOUR_PUBLIC_KEY>
```

### 3. Set Environment Variable

#### Linux/Mac:
```bash
export DEVNET_TEST_WALLET_PRIVATE_KEY=$(cat devnet-test-wallet.json)
```

#### Windows PowerShell:
```powershell
$env:DEVNET_TEST_WALLET_PRIVATE_KEY = Get-Content devnet-test-wallet.json -Raw
```

#### Windows CMD:
```cmd
set /p DEVNET_TEST_WALLET_PRIVATE_KEY=<devnet-test-wallet.json
```

### 4. Run Tests

```bash
# Run all integration tests
python -m pytest tests/integration -v

# Run specific test file
python -m pytest tests/integration/test_devnet_rpc_connection.py -v

# Stop at first failure
python -m pytest tests/integration -v -x
```

## Test Categories

### ✅ No Funding Required
- **RPC Connection Tests** (`test_devnet_rpc_connection.py`) - Tests reading blockchain data
- **Wallet Creation Tests** - Tests keypair generation

### 💰 Requires Funded Wallet
- **Balance Tests** - Tests reading wallet balances
- **Transfer Tests** - Tests submitting real transactions
- **Bonding Curve Tests** - Tests reading Pump.fun program accounts
- **Full Trade Flow Tests** - Tests complete buy/sell flows

## Troubleshooting

### "No funded wallet available"
Set the `DEVNET_TEST_WALLET_PRIVATE_KEY` environment variable (see step 3 above).

### "Funded wallet has insufficient balance"
Your test wallet needs at least 0.1 SOL. Fund it at https://faucet.solana.com

### "RPC error: rate limit" or "Internal error"
Devnet can be flaky. Try:
1. Using different devnet RPC (update `conftest.py`)
2. Waiting a few minutes and retrying
3. Using local test validator (`solana-test-validator`)

## Production vs Test Wallets

⚠️ **IMPORTANT**:
- Test wallets are for **devnet only**
- Never use production wallets in tests
- Never commit wallet private keys to git
- The devnet test wallet in `.gitignore`

## Security Notes

- Devnet test wallets have NO real value
- It's acceptable to share devnet private keys in testing documentation
- **NEVER** use mainnet wallets for testing
- **NEVER** commit wallet files to git

## Local Test Validator (Advanced)

For unlimited SOL and faster tests:

```bash
# Install Solana CLI tools
sh -c "$(curl -sSfL https://release.solana.com/stable/install)"

# Start local validator
solana-test-validator

# Update conftest.py to use localhost:8899
# Unlimited SOL available instantly!
```
