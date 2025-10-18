#!/usr/bin/env python3
"""
Decode Pump.fun TradeEvent from Program data log
"""

import base64
import struct

# The base64 data from line 45
program_data = "vdt/007mYe7I8IfZQ6+jBAoAmmv/1o3+u9m+TUw2Uv1rcm+hcT3+3+APlwAAAAAAXQDhxzAAAAABg64azHvFh4FI1PaNOySDuVixmy3+JhT4oavmR+Ht/XJx1O9oAAAAAHrn7BQJAAAAnCgmYJLuAgB6O8kYAgAAAJyQExQB8AEArRHmpPwpRKT6glG++BVCbhv7KMa2ZGZ3YHxq2fVmpkZfAAAAAAAAAGJvAQAAAAAAfWmiKn7ojhsEOf8c903/n0tVSKxyDUYWM4BEbO9/UcQeAAAAAAAAAAR0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"

# Expected tokens from bonding curve
expected_tokens = 209_511_841_885

print("="*80)
print("DECODING PUMP.FUN TRADE EVENT")
print("="*80)
print(f"\nExpected tokens: {expected_tokens:,} (0x{expected_tokens:X})\n")

# Decode base64
data = base64.b64decode(program_data)
print(f"Raw data length: {len(data)} bytes\n")

# Print hex dump
print("HEX DUMP:")
print("-"*80)
for i in range(0, min(len(data), 200), 16):
    hex_str = ' '.join(f'{b:02x}' for b in data[i:i+16])
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
    print(f"{i:04x}: {hex_str:<48} {ascii_str}")

print("\n" + "="*80)
print("SEARCHING FOR TOKEN AMOUNT")
print("="*80)

# Search for the expected token amount as little-endian u64
expected_bytes = struct.pack('<Q', expected_tokens)
print(f"\nLooking for bytes: {' '.join(f'{b:02x}' for b in expected_bytes)}")

if expected_bytes in data:
    offset = data.index(expected_bytes)
    print(f"✅ Found at offset {offset} (0x{offset:x})")

    # Extract as u64
    token_amount = struct.unpack('<Q', data[offset:offset+8])[0]
    print(f"Token amount: {token_amount:,}")
else:
    print("❌ Not found with exact match")

    # Try to find similar values (off by a few percent)
    print("\nSearching for similar u64 values in data...")
    for i in range(0, len(data) - 8, 8):
        try:
            val = struct.unpack('<Q', data[i:i+8])[0]
            if val > 1e11 and val < 1e15:  # Reasonable token amount range
                diff_pct = abs(val - expected_tokens) / expected_tokens * 100
                if diff_pct < 10:  # Within 10%
                    print(f"  Offset {i:3d} (0x{i:02x}): {val:,} ({diff_pct:+.2f}% diff)")
        except:
            pass

print("\n" + "="*80)
print("KNOWN ANCHOR EVENT STRUCTURE")
print("="*80)
print("""
Typical Anchor event structure:
  Bytes 0-7:   Event discriminator (8 bytes)
  Bytes 8+:    Event data fields

Pump.fun TradeEvent likely contains:
  - Mint address (32 bytes)
  - Trader address (32 bytes)
  - SOL amount (8 bytes, u64)
  - Token amount (8 bytes, u64)
  - Is buy (1 byte)
  - Timestamp (8 bytes, i64)
  - Virtual SOL reserves (8 bytes, u64)
  - Virtual token reserves (8 bytes, u64)
  - Real SOL reserves (8 bytes, u64)
  - Real token reserves (8 bytes, u64)

Let me try parsing with this structure...
""")

print("\n" + "="*80)
print("ATTEMPTING TO PARSE AS TRADE EVENT")
print("="*80)

try:
    offset = 0

    # Skip discriminator
    discriminator = data[offset:offset+8]
    print(f"\nDiscriminator: {discriminator.hex()}")
    offset += 8

    # Try to find token amount by scanning for large u64 values
    print("\nScanning for large u64 values (potential token amounts):")
    for i in range(8, len(data) - 8):
        val = struct.unpack('<Q', data[i:i+8])[0]

        # Token amounts are typically 1e11 to 1e15 (billions to trillions of base units)
        if 1e11 < val < 1e15:
            diff = abs(val - expected_tokens)
            diff_pct = diff / expected_tokens * 100
            marker = "✅ MATCH!" if diff_pct < 1 else "⚠️ Close" if diff_pct < 5 else ""
            print(f"  Offset {i:3d}: {val:,} ({diff_pct:+.2f}% diff) {marker}")

except Exception as e:
    print(f"Error parsing: {e}")

print("\n" + "="*80)
