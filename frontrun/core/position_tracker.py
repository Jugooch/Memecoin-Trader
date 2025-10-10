"""
Position Tracker
Tracks open and closed positions with SQLite persistence
"""

import uuid
import aiosqlite
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from solders.pubkey import Pubkey

from core.logger import get_logger
from core.metrics import get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


class PositionStatus(Enum):
    """Position status"""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"  # Partially closed


@dataclass
class Position:
    """Open trading position"""
    position_id: str  # UUID
    wallet: Pubkey
    mint: Pubkey
    amount_tokens: int
    entry_price_sol: float
    entry_slot: int
    entry_timestamp: datetime
    strategy: str  # "frontrun" or "copytrading"
    status: PositionStatus = PositionStatus.OPEN

    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            "position_id": self.position_id,
            "wallet": str(self.wallet),
            "mint": str(self.mint),
            "amount_tokens": self.amount_tokens,
            "entry_price_sol": self.entry_price_sol,
            "entry_slot": self.entry_slot,
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "strategy": self.strategy,
            "status": self.status.value
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        """Create from dictionary"""
        return cls(
            position_id=data["position_id"],
            wallet=Pubkey.from_string(data["wallet"]),
            mint=Pubkey.from_string(data["mint"]),
            amount_tokens=data["amount_tokens"],
            entry_price_sol=data["entry_price_sol"],
            entry_slot=data["entry_slot"],
            entry_timestamp=datetime.fromisoformat(data["entry_timestamp"]),
            strategy=data["strategy"],
            status=PositionStatus(data["status"])
        )


@dataclass
class ClosedPosition(Position):
    """Closed trading position with PnL"""
    exit_price_sol: float = 0.0
    exit_slot: int = 0
    exit_timestamp: Optional[datetime] = None
    pnl_sol: float = 0.0
    pnl_pct: float = 0.0
    holding_time_seconds: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        base = super().to_dict()
        base.update({
            "exit_price_sol": self.exit_price_sol,
            "exit_slot": self.exit_slot,
            "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            "pnl_sol": self.pnl_sol,
            "pnl_pct": self.pnl_pct,
            "holding_time_seconds": self.holding_time_seconds
        })
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "ClosedPosition":
        """Create from dictionary"""
        return cls(
            position_id=data["position_id"],
            wallet=Pubkey.from_string(data["wallet"]),
            mint=Pubkey.from_string(data["mint"]),
            amount_tokens=data["amount_tokens"],
            entry_price_sol=data["entry_price_sol"],
            entry_slot=data["entry_slot"],
            entry_timestamp=datetime.fromisoformat(data["entry_timestamp"]),
            strategy=data["strategy"],
            status=PositionStatus(data["status"]),
            exit_price_sol=data.get("exit_price_sol", 0.0),
            exit_slot=data.get("exit_slot", 0),
            exit_timestamp=datetime.fromisoformat(data["exit_timestamp"]) if data.get("exit_timestamp") else None,
            pnl_sol=data.get("pnl_sol", 0.0),
            pnl_pct=data.get("pnl_pct", 0.0),
            holding_time_seconds=data.get("holding_time_seconds", 0)
        )


class PositionStorage:
    """
    SQLite storage for positions

    Schema:
    - positions: All positions (open and closed)
    - Indexes on wallet, mint, status for fast queries
    """

    def __init__(self, db_path: str = "data/positions.db"):
        """
        Initialize position storage

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None

        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info("position_storage_initialized", db_path=db_path)

    async def connect(self):
        """Connect to database and create tables"""
        self._connection = await aiosqlite.connect(self.db_path)
        await self._create_tables()
        logger.info("position_storage_connected")

    async def _create_tables(self):
        """Create database tables if they don't exist"""
        # Phase 4: Enable WAL mode for better concurrency and crash recovery
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA synchronous=NORMAL")
        await self._connection.execute("PRAGMA busy_timeout=5000")

        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                wallet TEXT NOT NULL,
                mint TEXT NOT NULL,
                amount_tokens INTEGER NOT NULL,
                entry_price_sol REAL NOT NULL,
                entry_slot INTEGER NOT NULL,
                entry_timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                status TEXT NOT NULL,
                exit_price_sol REAL,
                exit_slot INTEGER,
                exit_timestamp TEXT,
                pnl_sol REAL,
                pnl_pct REAL,
                holding_time_seconds INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for fast queries
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_wallet ON positions(wallet)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_mint ON positions(mint)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON positions(status)
        """)
        await self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_strategy ON positions(strategy)
        """)

        await self._connection.commit()

        logger.info("position_tables_created")

    async def close(self):
        """Close database connection"""
        if self._connection:
            await self._connection.close()
            logger.info("position_storage_closed")

    async def insert_position(self, position: Position):
        """Insert new position"""
        data = position.to_dict()
        await self._connection.execute("""
            INSERT INTO positions (
                position_id, wallet, mint, amount_tokens, entry_price_sol,
                entry_slot, entry_timestamp, strategy, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["position_id"],
            data["wallet"],
            data["mint"],
            data["amount_tokens"],
            data["entry_price_sol"],
            data["entry_slot"],
            data["entry_timestamp"],
            data["strategy"],
            data["status"]
        ))
        await self._connection.commit()

    async def update_position(self, position: ClosedPosition):
        """Update position (for closing)"""
        data = position.to_dict()
        await self._connection.execute("""
            UPDATE positions SET
                status = ?,
                exit_price_sol = ?,
                exit_slot = ?,
                exit_timestamp = ?,
                pnl_sol = ?,
                pnl_pct = ?,
                holding_time_seconds = ?
            WHERE position_id = ?
        """, (
            data["status"],
            data.get("exit_price_sol"),
            data.get("exit_slot"),
            data.get("exit_timestamp"),
            data.get("pnl_sol"),
            data.get("pnl_pct"),
            data.get("holding_time_seconds"),
            data["position_id"]
        ))
        await self._connection.commit()

    async def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID"""
        cursor = await self._connection.execute("""
            SELECT * FROM positions WHERE position_id = ?
        """, (position_id,))
        row = await cursor.fetchone()

        if not row:
            return None

        # Convert row to dict
        columns = [desc[0] for desc in cursor.description]
        data = dict(zip(columns, row))

        # Return appropriate class based on status
        if data["status"] == PositionStatus.CLOSED.value:
            return ClosedPosition.from_dict(data)
        return Position.from_dict(data)

    async def get_open_positions(
        self,
        wallet: Optional[Pubkey] = None,
        mint: Optional[Pubkey] = None
    ) -> List[Position]:
        """Get open positions with optional filters"""
        query = "SELECT * FROM positions WHERE status = ?"
        params = [PositionStatus.OPEN.value]

        if wallet:
            query += " AND wallet = ?"
            params.append(str(wallet))

        if mint:
            query += " AND mint = ?"
            params.append(str(mint))

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()

        columns = [desc[0] for desc in cursor.description]
        return [Position.from_dict(dict(zip(columns, row))) for row in rows]

    async def get_closed_positions(
        self,
        wallet: Optional[Pubkey] = None,
        limit: int = 100
    ) -> List[ClosedPosition]:
        """Get closed positions"""
        query = "SELECT * FROM positions WHERE status = ?"
        params = [PositionStatus.CLOSED.value]

        if wallet:
            query += " AND wallet = ?"
            params.append(str(wallet))

        query += " ORDER BY exit_timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()

        columns = [desc[0] for desc in cursor.description]
        return [ClosedPosition.from_dict(dict(zip(columns, row))) for row in rows]


class PositionTracker:
    """
    Tracks trading positions with persistence

    Features:
    - Open/close position tracking
    - Position size updates (partial exits)
    - Query open/closed positions
    - Automatic PnL calculation
    - SQLite persistence

    Usage:
        tracker = PositionTracker(storage)
        await tracker.start()

        # Open position
        position = await tracker.open_position(
            wallet=wallet.pubkey(),
            mint=mint_pubkey,
            amount_tokens=1_000_000_000,
            entry_price_sol=0.5,
            entry_slot=12345,
            strategy="frontrun"
        )

        # Close position
        closed = await tracker.close_position(
            position_id=position.position_id,
            exit_price_sol=0.75,
            exit_slot=12350
        )
    """

    def __init__(self, storage: PositionStorage):
        """
        Initialize position tracker

        Args:
            storage: Position storage backend
        """
        self.storage = storage
        self._running = False

        logger.info("position_tracker_initialized")

    async def start(self):
        """Start position tracker"""
        await self.storage.connect()
        self._running = True
        logger.info("position_tracker_started")

    async def stop(self):
        """Stop position tracker"""
        await self.storage.close()
        self._running = False
        logger.info("position_tracker_stopped")

    async def open_position(
        self,
        wallet: Pubkey,
        mint: Pubkey,
        amount_tokens: int,
        entry_price_sol: float,
        entry_slot: int,
        strategy: str = "frontrun"
    ) -> Position:
        """
        Open new position

        Args:
            wallet: Wallet pubkey
            mint: Token mint pubkey
            amount_tokens: Amount of tokens bought
            entry_price_sol: Entry price in SOL
            entry_slot: Entry slot
            strategy: Trading strategy ("frontrun" or "copytrading")

        Returns:
            Created position

        Example:
            position = await tracker.open_position(
                wallet=wallet.pubkey(),
                mint=mint_pubkey,
                amount_tokens=1_000_000_000,
                entry_price_sol=0.5,
                entry_slot=12345,
                strategy="frontrun"
            )
        """
        position = Position(
            position_id=str(uuid.uuid4()),
            wallet=wallet,
            mint=mint,
            amount_tokens=amount_tokens,
            entry_price_sol=entry_price_sol,
            entry_slot=entry_slot,
            entry_timestamp=datetime.now(timezone.utc),
            strategy=strategy,
            status=PositionStatus.OPEN
        )

        await self.storage.insert_position(position)

        logger.info(
            "position_opened",
            position_id=position.position_id,
            wallet=str(wallet),
            mint=str(mint),
            amount_tokens=amount_tokens,
            entry_price_sol=entry_price_sol,
            strategy=strategy
        )

        metrics.increment_counter("positions_opened", labels={"strategy": strategy})

        return position

    async def close_position(
        self,
        position_id: str,
        exit_price_sol: float,
        exit_slot: int
    ) -> ClosedPosition:
        """
        Close position and calculate PnL

        Args:
            position_id: Position ID to close
            exit_price_sol: Exit price in SOL
            exit_slot: Exit slot

        Returns:
            Closed position with PnL

        Raises:
            ValueError: If position not found or already closed

        Example:
            closed = await tracker.close_position(
                position_id="abc-123",
                exit_price_sol=0.75,
                exit_slot=12350
            )
        """
        # Fetch position
        position = await self.storage.get_position(position_id)
        if not position:
            raise ValueError(f"Position {position_id} not found")

        if position.status == PositionStatus.CLOSED:
            raise ValueError(f"Position {position_id} already closed")

        # Calculate PnL
        exit_timestamp = datetime.now(timezone.utc)
        holding_time_seconds = int((exit_timestamp - position.entry_timestamp).total_seconds())

        pnl_sol = exit_price_sol - position.entry_price_sol
        pnl_pct = (pnl_sol / position.entry_price_sol) * 100 if position.entry_price_sol > 0 else 0

        # Create closed position
        closed_position = ClosedPosition(
            position_id=position.position_id,
            wallet=position.wallet,
            mint=position.mint,
            amount_tokens=position.amount_tokens,
            entry_price_sol=position.entry_price_sol,
            entry_slot=position.entry_slot,
            entry_timestamp=position.entry_timestamp,
            strategy=position.strategy,
            status=PositionStatus.CLOSED,
            exit_price_sol=exit_price_sol,
            exit_slot=exit_slot,
            exit_timestamp=exit_timestamp,
            pnl_sol=pnl_sol,
            pnl_pct=pnl_pct,
            holding_time_seconds=holding_time_seconds
        )

        await self.storage.update_position(closed_position)

        logger.info(
            "position_closed",
            position_id=position_id,
            pnl_sol=pnl_sol,
            pnl_pct=pnl_pct,
            holding_time_seconds=holding_time_seconds,
            strategy=position.strategy
        )

        metrics.increment_counter(
            "positions_closed",
            labels={
                "strategy": position.strategy,
                "profitable": str(pnl_sol > 0)
            }
        )

        return closed_position

    async def get_open_positions(
        self,
        wallet: Optional[Pubkey] = None,
        mint: Optional[Pubkey] = None
    ) -> List[Position]:
        """
        Get open positions with optional filters

        Args:
            wallet: Filter by wallet (optional)
            mint: Filter by mint (optional)

        Returns:
            List of open positions

        Example:
            # All open positions
            all_open = await tracker.get_open_positions()

            # Open positions for specific wallet
            wallet_open = await tracker.get_open_positions(wallet=wallet.pubkey())

            # Open positions for specific mint
            mint_open = await tracker.get_open_positions(mint=mint_pubkey)
        """
        return await self.storage.get_open_positions(wallet=wallet, mint=mint)

    async def get_closed_positions(
        self,
        wallet: Optional[Pubkey] = None,
        limit: int = 100
    ) -> List[ClosedPosition]:
        """
        Get closed positions

        Args:
            wallet: Filter by wallet (optional)
            limit: Maximum number of positions to return

        Returns:
            List of closed positions (most recent first)

        Example:
            # Recent closed positions
            recent = await tracker.get_closed_positions(limit=10)
        """
        return await self.storage.get_closed_positions(wallet=wallet, limit=limit)

    async def get_position(self, position_id: str) -> Optional[Position]:
        """
        Get position by ID

        Args:
            position_id: Position ID

        Returns:
            Position or None if not found

        Example:
            position = await tracker.get_position("abc-123")
        """
        return await self.storage.get_position(position_id)

    async def update_position_size(
        self,
        position_id: str,
        new_amount: int
    ) -> Position:
        """
        Update position size (for partial exits)

        Args:
            position_id: Position ID
            new_amount: New token amount

        Returns:
            Updated position

        Raises:
            ValueError: If position not found or closed

        Example:
            # Partial exit - reduce position size
            updated = await tracker.update_position_size(
                position_id="abc-123",
                new_amount=500_000_000  # Reduced from 1B to 500M
            )
        """
        position = await self.storage.get_position(position_id)
        if not position:
            raise ValueError(f"Position {position_id} not found")

        if position.status == PositionStatus.CLOSED:
            raise ValueError(f"Cannot update closed position {position_id}")

        position.amount_tokens = new_amount
        position.status = PositionStatus.PARTIAL if new_amount < position.amount_tokens else PositionStatus.OPEN

        await self.storage.update_position(position)

        logger.info(
            "position_size_updated",
            position_id=position_id,
            new_amount=new_amount,
            status=position.status.value
        )

        return position


# Example usage
if __name__ == "__main__":
    from core.logger import setup_logging
    import asyncio

    async def main():
        # Setup logging
        setup_logging(level="INFO", format="console")

        # Create storage and tracker
        storage = PositionStorage("data/test_positions.db")
        tracker = PositionTracker(storage)
        await tracker.start()

        try:
            # Example wallet and mint
            example_wallet = Pubkey.from_string("11111111111111111111111111111111")
            example_mint = Pubkey.from_string("22222222222222222222222222222222")

            # Open position
            position = await tracker.open_position(
                wallet=example_wallet,
                mint=example_mint,
                amount_tokens=1_000_000_000,
                entry_price_sol=0.5,
                entry_slot=12345,
                strategy="frontrun"
            )
            logger.info("position_opened", position_id=position.position_id)

            # Get open positions
            open_positions = await tracker.get_open_positions()
            logger.info("open_positions", count=len(open_positions))

            # Close position
            closed = await tracker.close_position(
                position_id=position.position_id,
                exit_price_sol=0.75,
                exit_slot=12350
            )
            logger.info(
                "position_closed",
                pnl_sol=closed.pnl_sol,
                pnl_pct=closed.pnl_pct
            )

            # Get closed positions
            closed_positions = await tracker.get_closed_positions()
            logger.info("closed_positions", count=len(closed_positions))

        finally:
            await tracker.stop()

    asyncio.run(main())
