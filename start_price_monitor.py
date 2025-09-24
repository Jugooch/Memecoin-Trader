#!/usr/bin/env python3
"""
Start script for the Price Monitor service
Runs independently from the main memecoin bot
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.services.price_monitor import main

if __name__ == "__main__":
    # Run the price monitor
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPrice Monitor stopped by user")
    except Exception as e:
        print(f"Price Monitor error: {e}")
        sys.exit(1)