"""
Extract features from Laserstream transaction events
"""

import struct
import base58
import base64
from typing import Optional, Dict, List
from datetime import datetime

from anti_bot.core.types import TxFeatures


# Known Jito tip accounts
JITO_TIP_ACCOUNTS = {
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
}

COMPUTE_BUDGET_PROGRAM = "ComputeBudget111111111111111111111111111111"
PUMP_FUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# TradeEvent discriminator
TRADE_EVENT_DISCRIMINATOR = bytes([189, 219, 127, 211, 78, 230, 97, 238])


def parse_trade_event_from_logs(logs: List[str]) -> Optional[Dict]:
    """Parse TradeEvent data from transaction logs"""
    for log in logs:
        if "Program data:" in log:
            try:
                encoded_data = log.split("Program data: ")[1].strip()
                decoded_data = base64.b64decode(encoded_data)

                if len(decoded_data) >= 8:
                    discriminator = decoded_data[:8]
                    if discriminator == TRADE_EVENT_DISCRIMINATOR:
                        return _decode_trade_event(decoded_data[8:])
            except Exception:
                continue
    return None


def _decode_trade_event(data: bytes) -> Optional[Dict]:
    """Decode TradeEvent structure from raw bytes"""
    if len(data) < 105:
        return None

    offset = 0

    # Parse mint (32 bytes)
    mint_bytes = data[offset:offset + 32]
    mint = base58.b58encode(mint_bytes).decode('utf-8')
    offset += 32

    # Parse sol_amount (u64)
    sol_amount = struct.unpack('<Q', data[offset:offset + 8])[0]
    offset += 8

    # Parse token_amount (u64)
    token_amount = struct.unpack('<Q', data[offset:offset + 8])[0]
    offset += 8

    # Parse is_buy (bool)
    is_buy = bool(data[offset])
    offset += 1

    # Parse user (32 bytes)
    user_bytes = data[offset:offset + 32]
    user = base58.b58encode(user_bytes).decode('utf-8')
    offset += 32

    # Parse timestamp (i64)
    timestamp = struct.unpack('<q', data[offset:offset + 8])[0]
    offset += 8

    # Parse virtual_sol_reserves (u64)
    virtual_sol_reserves = struct.unpack('<Q', data[offset:offset + 8])[0]
    offset += 8

    # Parse virtual_token_reserves (u64)
    virtual_token_reserves = struct.unpack('<Q', data[offset:offset + 8])[0]

    return {
        'mint': mint,
        'sol_amount': sol_amount,
        'token_amount': token_amount,
        'is_buy': is_buy,
        'user': user,
        'timestamp': timestamp,
        'virtual_sol_reserves': virtual_sol_reserves,
        'virtual_token_reserves': virtual_token_reserves,
        'sol_amount_ui': sol_amount / 1e9,
        'token_amount_ui': token_amount / 1e6,
    }


def extract_features_from_geyser_tx(tx_update, transaction) -> Optional[TxFeatures]:
    """
    Extract TxFeatures from Geyser transaction update

    Args:
        tx_update: The Geyser SubscribeUpdateTransaction message
        transaction: The inner transaction object

    Returns:
        TxFeatures object with all extracted features
    """
    try:
        # Basic identifiers
        signature = base58.b58encode(bytes(transaction.signature)).decode('utf-8')
        slot = tx_update.slot

        # Get transaction inner structure
        tx_inner = transaction.transaction
        msg = getattr(tx_inner, "message", None)
        if msg is None:
            return None

        # Get signer (first account key)
        account_keys = list(msg.account_keys)
        if not account_keys:
            return None

        signer_bytes = account_keys[0]
        signer = base58.b58encode(bytes(signer_bytes)).decode('utf-8')

        # Extract metadata
        meta = transaction.meta if hasattr(transaction, 'meta') else None

        # Extract logs
        logs = []
        if meta:
            if hasattr(meta, 'log_messages'):
                logs = list(meta.log_messages)
            elif hasattr(meta, 'logs'):
                logs = list(meta.logs)

        # Parse TradeEvent from logs
        trade_event = parse_trade_event_from_logs(logs)

        # Extract fee info
        base_fee = 0
        if meta and hasattr(meta, 'fee'):
            base_fee = meta.fee

        # Parse instructions
        programs = []
        ix_types = []
        cu_requested = 0
        cu_price = 0
        jito_tip = 0

        for ix in msg.instructions:
            # Get program ID
            if ix.program_id_index < len(account_keys):
                program_bytes = account_keys[ix.program_id_index]
                program_id = base58.b58encode(bytes(program_bytes)).decode('utf-8')
                programs.append(program_id)

                # Parse ComputeBudget instructions
                if program_id == COMPUTE_BUDGET_PROGRAM:
                    ix_data = bytes(ix.data)
                    if len(ix_data) >= 1:
                        discriminator = ix_data[0]
                        # SetComputeUnitLimit (0x02)
                        if discriminator == 2 and len(ix_data) >= 5:
                            cu_requested = struct.unpack('<I', ix_data[1:5])[0]
                        # SetComputeUnitPrice (0x03)
                        elif discriminator == 3 and len(ix_data) >= 9:
                            cu_price = struct.unpack('<Q', ix_data[1:9])[0]

                # Detect Jito tips (SOL transfers to known tip accounts)
                # This requires checking if any account in the instruction is a Jito tip account
                for acc_idx in ix.accounts:
                    if acc_idx < len(account_keys):
                        acc_bytes = account_keys[acc_idx]
                        acc_str = base58.b58encode(bytes(acc_bytes)).decode('utf-8')
                        if acc_str in JITO_TIP_ACCOUNTS:
                            # Estimate tip amount from instruction data if it's a transfer
                            # System program transfer has 4 bytes discriminator + 8 bytes amount
                            ix_data = bytes(ix.data)
                            if len(ix_data) >= 12:
                                # Transfer instruction: first 4 bytes are discriminator, next 8 are lamports
                                try:
                                    jito_tip = struct.unpack('<Q', ix_data[4:12])[0]
                                except:
                                    pass

                # Decode pump.fun instruction types
                if program_id == PUMP_FUN_PROGRAM:
                    ix_data = bytes(ix.data)
                    if ix_data[:8] == struct.pack("<Q", 8576854823835016728):
                        ix_types.append("Create")
                    elif ix_data[:8] == bytes.fromhex("66063d1201daebea"):
                        ix_types.append("Buy")
                    elif ix_data[:8] == bytes.fromhex("33e685a4017f83ad"):
                        ix_types.append("Sell")

        # Get CU consumed
        cu_consumed = 0
        if meta and hasattr(meta, 'compute_units_consumed'):
            cu_consumed = meta.compute_units_consumed

        # Build TxFeatures
        features = TxFeatures(
            slot=slot,
            entry_index=0,  # Geyser doesn't expose this easily
            signature=signature,
            signer=signer,
            timestamp=datetime.now(),
            programs=programs,
            ix_types=ix_types,
            cu_price=cu_price,
            cu_consumed=cu_consumed,
            cu_requested=cu_requested,
            base_fee_lamports=base_fee,
            jito_tip_lamports=jito_tip,
            logs_sample=logs[:10],  # First 10 logs
            tx_size_bytes=len(bytes(tx_inner.SerializeToString())) if hasattr(tx_inner, 'SerializeToString') else 0,
        )

        # Add trade event data if available
        if trade_event:
            features.mint = trade_event['mint']
            features.is_buy = trade_event['is_buy']
            features.sol_amount = trade_event['sol_amount_ui']
            features.token_amount = trade_event['token_amount_ui']
            features.sol_delta = trade_event['sol_amount_ui'] * (-1 if trade_event['is_buy'] else 1)
            features.token_delta = trade_event['token_amount_ui'] * (1 if trade_event['is_buy'] else -1)
            features.virtual_sol_reserves = trade_event['virtual_sol_reserves']
            features.virtual_token_reserves = trade_event['virtual_token_reserves']

        return features

    except Exception as e:
        # Silently fail - Geyser sends many transaction types
        return None


def calculate_gini_coefficient(values: List[float]) -> float:
    """Calculate Gini coefficient for buy size distribution"""
    if not values or len(values) < 2:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)
    cumsum = 0
    for i, val in enumerate(sorted_values):
        cumsum += (i + 1) * val

    total = sum(sorted_values)
    if total == 0:
        return 0.0

    return (2 * cumsum) / (n * total) - (n + 1) / n
