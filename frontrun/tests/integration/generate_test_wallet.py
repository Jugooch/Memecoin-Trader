"""
Generate a devnet test wallet for integration tests

This script creates a new keypair and saves it in the format needed for testing.
"""

import json
from solders.keypair import Keypair


def generate_test_wallet():
    """Generate new keypair and save to file"""
    # Generate new keypair
    keypair = Keypair()

    # Get public key
    pubkey = str(keypair.pubkey())

    # Get private key bytes as list
    private_key_bytes = list(bytes(keypair))

    # Save to JSON file (Solana standard format)
    output_file = "devnet-test-wallet.json"
    with open(output_file, 'w') as f:
        json.dump(private_key_bytes, f)

    print(f"✅ Wallet generated successfully!")
    print(f"")
    print(f"📁 Private key saved to: {output_file}")
    print(f"🔑 Public key (address): {pubkey}")
    print(f"")
    print(f"Next steps:")
    print(f"1. Fund this wallet at: https://faucet.solana.com")
    print(f"   Address: {pubkey}")
    print(f"")
    print(f"2. Set environment variable:")
    print(f"   PowerShell:")
    print(f"   $env:DEVNET_TEST_WALLET_PRIVATE_KEY = Get-Content {output_file} -Raw")
    print(f"")
    print(f"3. Run integration tests:")
    print(f"   python -m pytest tests/integration -v")
    print(f"")
    print(f"⚠️  Keep this file secure! Add to .gitignore")

    return pubkey, private_key_bytes


if __name__ == "__main__":
    generate_test_wallet()
