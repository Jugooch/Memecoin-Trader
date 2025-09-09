"""
JSON-based storage with caching for VM performance optimization
"""

import json
import os
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging
from threading import Lock


class Store:
    """High-performance JSON storage with in-memory caching"""
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Storage paths
        self.base_path = Path(config.get('path', 'data'))
        self.base_path.mkdir(exist_ok=True)
        
        # File paths
        self.positions_file = self.base_path / 'positions.json'
        self.dev_profiles_file = self.base_path / 'dev_profiles.json'
        self.whitelist_file = self.base_path / 'whitelist.json'
        self.blacklist_file = self.base_path / 'blacklist.json'
        self.events_file = self.base_path / 'events.jsonl'
        self.fills_file = self.base_path / 'fills.json'
        
        # In-memory caches
        self.positions_cache = {}
        self.dev_profiles_cache = {}
        self.whitelist_cache = set()
        self.blacklist_cache = set()
        self.fills_cache = defaultdict(list)
        
        # Cache timestamps
        self.cache_timestamps = {}
        self.cache_ttl = 3600  # 1 hour default TTL
        
        # Thread safety
        self.lock = Lock()
        
        # Rotation settings
        self.max_file_size_mb = config.get('rotate_jsonl_mb', 128)
        self.events_counter = 0
        
        # Load existing data
        self._load_all()
    
    def _load_all(self):
        """Load all JSON files into memory caches"""
        self._load_json_cache(self.positions_file, self.positions_cache)
        self._load_json_cache(self.dev_profiles_file, self.dev_profiles_cache)
        self._load_json_cache(self.whitelist_file, self.whitelist_cache, is_set=True)
        self._load_json_cache(self.blacklist_file, self.blacklist_cache, is_set=True)
        self._load_json_cache(self.fills_file, self.fills_cache)
    
    def _load_json_cache(self, file_path: Path, cache: Any, is_set: bool = False):
        """Load JSON file into cache"""
        if not file_path.exists():
            return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if is_set:
                    cache.clear()
                    cache.update(data)
                else:
                    cache.clear()
                    cache.update(data)
            self.logger.debug(f"Loaded {len(data)} items from {file_path.name}")
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
    
    def _save_json_cache(self, cache: Any, file_path: Path, is_set: bool = False):
        """Save cache to JSON file"""
        try:
            with self.lock:
                data = list(cache) if is_set else cache
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving {file_path}: {e}")
    
    # Position Management
    def add_position(self, position: Dict) -> str:
        """Add new position"""
        position_id = f"{position['token']}_{int(time.time())}"
        position['id'] = position_id
        position['status'] = 'open'
        position['created_at'] = datetime.now().isoformat()
        
        self.positions_cache[position_id] = position
        self._save_json_cache(self.positions_cache, self.positions_file)
        
        self.logger.info(f"Added position {position_id}")
        return position_id
    
    def update_position(self, position_id: str, updates: Dict):
        """Update existing position"""
        if position_id in self.positions_cache:
            self.positions_cache[position_id].update(updates)
            self.positions_cache[position_id]['updated_at'] = datetime.now().isoformat()
            self._save_json_cache(self.positions_cache, self.positions_file)
    
    def get_position(self, position_id: str) -> Optional[Dict]:
        """Get position by ID"""
        return self.positions_cache.get(position_id)
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        return [p for p in self.positions_cache.values() if p.get('status') == 'open']
    
    def close_position(self, position_id: str, exit_reason: str, pnl: float):
        """Close a position"""
        if position_id in self.positions_cache:
            self.positions_cache[position_id].update({
                'status': 'closed',
                'exit_reason': exit_reason,
                'realized_pnl': pnl,
                'closed_at': datetime.now().isoformat()
            })
            self._save_json_cache(self.positions_cache, self.positions_file)
    
    # Dev Profile Management with Smart Caching
    def get_dev_profile(self, dev_wallet: str) -> Optional[Dict]:
        """Get cached dev profile if fresh, None if stale/missing"""
        profile = self.dev_profiles_cache.get(dev_wallet)
        if not profile:
            return None
        
        # Check if cache is fresh (24 hours)
        last_updated = profile.get('updated_at')
        if last_updated:
            last_updated_dt = datetime.fromisoformat(last_updated)
            if datetime.now() - last_updated_dt > timedelta(hours=24):
                return None  # Cache is stale
        
        return profile
    
    def update_dev_profile(self, dev_wallet: str, profile: Dict):
        """Update dev profile with timestamp"""
        profile['dev_wallet'] = dev_wallet
        profile['updated_at'] = datetime.now().isoformat()
        
        self.dev_profiles_cache[dev_wallet] = profile
        self._save_json_cache(self.dev_profiles_cache, self.dev_profiles_file)
    
    def get_stale_dev_profiles(self, max_age_hours: int = 24) -> List[str]:
        """Get list of dev wallets with stale profiles"""
        stale_devs = []
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for dev_wallet, profile in self.dev_profiles_cache.items():
            last_updated = profile.get('updated_at')
            if not last_updated:
                stale_devs.append(dev_wallet)
            else:
                last_updated_dt = datetime.fromisoformat(last_updated)
                if last_updated_dt < cutoff_time:
                    stale_devs.append(dev_wallet)
        
        return stale_devs
    
    # Whitelist/Blacklist Management
    def add_to_whitelist(self, dev_wallet: str, score: float = 0):
        """Add dev to whitelist"""
        entry = {'wallet': dev_wallet, 'score': score, 'added_at': datetime.now().isoformat()}
        self.whitelist_cache.add(dev_wallet)
        
        # Also save with metadata
        whitelist_data = self._load_whitelist_with_metadata()
        whitelist_data[dev_wallet] = entry
        with open(self.whitelist_file, 'w') as f:
            json.dump(whitelist_data, f, indent=2)
    
    def remove_from_whitelist(self, dev_wallet: str):
        """Remove dev from whitelist"""
        self.whitelist_cache.discard(dev_wallet)
        
        whitelist_data = self._load_whitelist_with_metadata()
        whitelist_data.pop(dev_wallet, None)
        with open(self.whitelist_file, 'w') as f:
            json.dump(whitelist_data, f, indent=2)
    
    def is_whitelisted(self, dev_wallet: str) -> bool:
        """Check if dev is whitelisted"""
        return dev_wallet in self.whitelist_cache
    
    def add_to_blacklist(self, dev_wallet: str, reason: str):
        """Add dev to blacklist"""
        entry = {'wallet': dev_wallet, 'reason': reason, 'added_at': datetime.now().isoformat()}
        self.blacklist_cache.add(dev_wallet)
        
        # Also save with metadata
        blacklist_data = self._load_blacklist_with_metadata()
        blacklist_data[dev_wallet] = entry
        with open(self.blacklist_file, 'w') as f:
            json.dump(blacklist_data, f, indent=2)
    
    def remove_from_blacklist(self, dev_wallet: str):
        """Remove dev from blacklist"""
        self.blacklist_cache.discard(dev_wallet)
        
        blacklist_data = self._load_blacklist_with_metadata()
        blacklist_data.pop(dev_wallet, None)
        with open(self.blacklist_file, 'w') as f:
            json.dump(blacklist_data, f, indent=2)
    
    def is_blacklisted(self, dev_wallet: str) -> bool:
        """Check if dev is blacklisted"""
        return dev_wallet in self.blacklist_cache
    
    def is_blacklisted_dev(self, dev_wallet: str) -> bool:
        """Alias for is_blacklisted"""
        return self.is_blacklisted(dev_wallet)
    
    def _load_whitelist_with_metadata(self) -> Dict:
        """Load whitelist with full metadata"""
        if not self.whitelist_file.exists():
            return {}
        try:
            with open(self.whitelist_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _load_blacklist_with_metadata(self) -> Dict:
        """Load blacklist with full metadata"""
        if not self.blacklist_file.exists():
            return {}
        try:
            with open(self.blacklist_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    # Event Logging
    def log_event(self, event_type: str, data: Dict, reason: str = None):
        """Log event to JSONL file with rotation"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'data': data,
            'reason': reason
        }
        
        # Check file size for rotation
        if self.events_file.exists():
            size_mb = self.events_file.stat().st_size / (1024 * 1024)
            if size_mb >= self.max_file_size_mb:
                # Rotate file
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                rotated_file = self.base_path / f'events_{timestamp}.jsonl'
                self.events_file.rename(rotated_file)
                self.logger.info(f"Rotated events file to {rotated_file.name}")
        
        # Append event with datetime serialization
        with open(self.events_file, 'a') as f:
            f.write(json.dumps(event, default=self._json_serializer) + '\n')
        
        self.events_counter += 1
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for datetime objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    # Fill Tracking
    def add_fill(self, position_id: str, side: str, qty: float, price: float, fee_sol: float):
        """Add fill record"""
        fill = {
            'position_id': position_id,
            'side': side,
            'qty': qty,
            'price': price,
            'fee_sol': fee_sol,
            'timestamp': datetime.now().isoformat()
        }
        
        self.fills_cache[position_id].append(fill)
        self._save_json_cache(dict(self.fills_cache), self.fills_file)
    
    def get_fills(self, position_id: str) -> List[Dict]:
        """Get fills for position"""
        return self.fills_cache.get(position_id, [])
    
    # Dev History
    def get_dev_last_attempt(self, dev_wallet: str) -> Optional[datetime]:
        """Get last attempt timestamp for dev"""
        profile = self.dev_profiles_cache.get(dev_wallet, {})
        last_attempt = profile.get('last_attempt')
        if last_attempt:
            return datetime.fromisoformat(last_attempt)
        return None
    
    def update_dev_last_attempt(self, dev_wallet: str):
        """Update last attempt timestamp for dev"""
        if dev_wallet not in self.dev_profiles_cache:
            self.dev_profiles_cache[dev_wallet] = {}
        
        self.dev_profiles_cache[dev_wallet]['last_attempt'] = datetime.now().isoformat()
        self._save_json_cache(self.dev_profiles_cache, self.dev_profiles_file)
    
    # Statistics
    def get_stats(self) -> Dict:
        """Get storage statistics"""
        return {
            'total_positions': len(self.positions_cache),
            'open_positions': len(self.get_open_positions()),
            'whitelisted_devs': len(self.whitelist_cache),
            'blacklisted_devs': len(self.blacklist_cache),
            'cached_profiles': len(self.dev_profiles_cache),
            'total_events': self.events_counter,
            'storage_path': str(self.base_path)
        }