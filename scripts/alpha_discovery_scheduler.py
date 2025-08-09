#!/usr/bin/env python3
"""
Alpha Discovery Scheduler
Runs alpha wallet discovery every 6 hours to continuously find new profitable wallets
"""

import asyncio
import sys
import os
import logging
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.clients.bitquery_client import BitqueryClient
from src.clients.moralis_client import MoralisClient
from src.core.database import Database
from src.discovery.alpha_discovery_v2 import ProvenAlphaFinder
from src.utils.config_loader import load_config, get_database_path


async def run_discovery():
    """Run single discovery cycle"""
    logging.info("="*60)
    logging.info(f"Starting alpha discovery cycle at {datetime.now()}")
    logging.info("="*60)
    
    try:
        # Load config using shared loader
        config = load_config('config.yml')
        
        # Initialize clients
        bitquery = BitqueryClient(config['bitquery_token'])
        moralis = MoralisClient(config.get('moralis_keys', config.get('moralis_key')))
        database = Database(get_database_path(config))
        
        await database.initialize()
        
        # Run alpha finder
        finder = ProvenAlphaFinder(bitquery, moralis, database)
        alpha_wallets = await finder.discover_alpha_wallets()
        
        if alpha_wallets:
            logging.info(f"Discovered {len(alpha_wallets)} new alpha wallets")
            
            # Save to database
            await finder._save_discovered_wallets(alpha_wallets)
            
            # Update bot config
            await finder._update_bot_config(alpha_wallets)
            
            logging.info("Successfully updated wallet list")
        else:
            logging.info("No new alpha wallets found this cycle")
            
        # Cleanup
        await moralis.close()
        if hasattr(bitquery, 'client') and bitquery.client:
            await bitquery.client.transport.close()
            
    except Exception as e:
        logging.error(f"Error in discovery cycle: {e}")


async def main():
    """Main scheduler loop"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/alpha_discovery_scheduler.log', mode='a')
        ]
    )
    
    logging.info("Alpha Discovery Scheduler Started")
    logging.info("Will run discovery every 2 hours (reduced from 6 for fresher wallets)")
    
    # Run immediately on startup
    await run_discovery()
    
    # Then run every 2 hours (reduced from 6 for more frequent updates)
    while True:
        try:
            # Wait 2 hours
            await asyncio.sleep(2 * 3600)
            
            # Run discovery
            await run_discovery()
            
        except KeyboardInterrupt:
            logging.info("Scheduler stopped by user")
            break
        except Exception as e:
            logging.error(f"Scheduler error: {e}")
            # Wait 1 hour on error before retrying
            await asyncio.sleep(3600)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScheduler stopped")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)