#!/usr/bin/env python3
"""
Historical Strategy Backtester
Extracts data from your actual trading logs and tests strategies on historical alpha signals
"""

import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sims.strategy_ab_tester import StrategyConfig, SimulatedPosition, MarketReality, TradeExecution
from src.utils.config_loader import load_config


@dataclass
class HistoricalAlphaSignal:
    """Historical alpha wallet signal extracted from logs"""
    timestamp: datetime
    mint_address: str
    wallet_address: str
    symbol: str
    price: float
    source: str  # "log" or "database"


@dataclass
class HistoricalTokenData:
    """Historical token price movements"""
    mint_address: str
    symbol: str
    launch_time: datetime
    alpha_signals: List[HistoricalAlphaSignal]
    price_history: List[Tuple[datetime, float]]  # (timestamp, price)
    max_price: float
    max_price_time: datetime
    final_price: Optional[float] = None


class HistoricalDataExtractor:
    """Extract historical alpha signals and price data from your logs/database"""
    
    def __init__(self, config_path: str = "config.yml"):
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Setup paths
        self.project_root = Path(__file__).parent.parent
        self.logs_dir = self.project_root / "logs"
        self.database_path = self.project_root / self.config.get('database', {}).get('file', 'trades.db')
        
    def extract_historical_signals(self, days_back: int = 30) -> List[HistoricalTokenData]:
        """Extract historical alpha signals from logs and database"""
        self.logger.info(f"📚 Extracting historical data from last {days_back} days")
        
        # Extract from multiple sources
        log_signals = self._extract_from_logs(days_back)
        db_signals = self._extract_from_database(days_back)
        
        # Combine and deduplicate
        all_signals = log_signals + db_signals
        token_data = self._group_signals_by_token(all_signals)
        
        self.logger.info(f"📈 Found {len(token_data)} tokens with alpha signals")
        return token_data
    
    def _extract_from_logs(self, days_back: int) -> List[HistoricalAlphaSignal]:
        """Extract alpha signals from trading logs"""
        signals = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Find log files
        log_files = []
        if self.logs_dir.exists():
            log_files.extend(self.logs_dir.glob("*.log"))
            log_files.extend(self.logs_dir.glob("trading*.log"))
            log_files.extend(self.logs_dir.glob("*aggressive*.log"))
        
        self.logger.info(f"🔍 Scanning {len(log_files)} log files")
        
        for log_file in log_files:
            try:
                signals.extend(self._parse_log_file(log_file, cutoff_date))
            except Exception as e:
                self.logger.warning(f"⚠️ Error parsing {log_file}: {e}")
        
        self.logger.info(f"📡 Found {len(signals)} alpha signals from logs")
        return signals
    
    def _parse_log_file(self, log_file: Path, cutoff_date: datetime) -> List[HistoricalAlphaSignal]:
        """Parse individual log file for alpha signals"""
        signals = []
        
        # Patterns to match alpha wallet detections
        alpha_patterns = [
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*ALPHA WALLET DETECTED: (\w+).*bought (\w+)',
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*REALTIME ALPHA: (\w+).*bought (\w+)',
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Alpha signal: (\w+).*bought (\w+).*at \$([0-9.e-]+)',
        ]
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        for pattern in alpha_patterns:
                            match = re.search(pattern, line)
                            if match:
                                groups = match.groups()
                                
                                # Parse timestamp
                                timestamp_str = groups[0]
                                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                                
                                if timestamp < cutoff_date:
                                    continue
                                
                                # Extract wallet and mint
                                wallet = groups[1]
                                mint = groups[2] if len(groups) > 2 else 'unknown'
                                price = float(groups[3]) if len(groups) > 3 else 0.0
                                
                                # Extract symbol if present in line
                                symbol_match = re.search(r'symbol[:\s]*([A-Z0-9]+)', line, re.IGNORECASE)
                                symbol = symbol_match.group(1) if symbol_match else 'UNKNOWN'
                                
                                signals.append(HistoricalAlphaSignal(
                                    timestamp=timestamp,
                                    mint_address=mint,
                                    wallet_address=wallet,
                                    symbol=symbol,
                                    price=price,
                                    source='log'
                                ))
                                break
                    except Exception as e:
                        # Skip malformed lines
                        continue
        except Exception as e:
            self.logger.error(f"❌ Error reading {log_file}: {e}")
        
        return signals
    
    def _extract_from_database(self, days_back: int) -> List[HistoricalAlphaSignal]:
        """Extract alpha signals from trading database"""
        signals = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        if not self.database_path.exists():
            self.logger.warning(f"⚠️ Database not found: {self.database_path}")
            return signals
        
        try:
            with sqlite3.connect(str(self.database_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Extract from trades table if it exists
                try:
                    cursor.execute("""
                        SELECT timestamp, mint_address, symbol, entry_price, action
                        FROM trades 
                        WHERE timestamp > ? AND action = 'BUY'
                        ORDER BY timestamp DESC
                    """, (cutoff_date.timestamp(),))
                    
                    for row in cursor.fetchall():
                        # Create synthetic alpha signal for our own trades
                        # (These represent tokens we actually traded)
                        signals.append(HistoricalAlphaSignal(
                            timestamp=datetime.fromtimestamp(row['timestamp']),
                            mint_address=row['mint_address'],
                            wallet_address='our_bot',  # Mark as our bot's trade
                            symbol=row['symbol'],
                            price=row['entry_price'],
                            source='database'
                        ))
                except sqlite3.OperationalError:
                    # Table might not exist
                    pass
                
                # Try to extract from any alpha_signals table if it exists
                try:
                    cursor.execute("""
                        SELECT timestamp, mint_address, wallet_address, price
                        FROM alpha_signals 
                        WHERE timestamp > ?
                        ORDER BY timestamp DESC
                    """, (cutoff_date.timestamp(),))
                    
                    for row in cursor.fetchall():
                        signals.append(HistoricalAlphaSignal(
                            timestamp=datetime.fromtimestamp(row['timestamp']),
                            mint_address=row['mint_address'],
                            wallet_address=row['wallet_address'],
                            symbol='UNKNOWN',
                            price=row.get('price', 0.0),
                            source='database'
                        ))
                except sqlite3.OperationalError:
                    # Table might not exist
                    pass
        
        except Exception as e:
            self.logger.error(f"❌ Error accessing database: {e}")
        
        self.logger.info(f"🗄️ Found {len(signals)} signals from database")
        return signals
    
    def _group_signals_by_token(self, signals: List[HistoricalAlphaSignal]) -> List[HistoricalTokenData]:
        """Group signals by token and create token data objects"""
        token_groups = {}
        
        for signal in signals:
            mint = signal.mint_address
            if mint not in token_groups:
                token_groups[mint] = []
            token_groups[mint].append(signal)
        
        token_data = []
        for mint, mint_signals in token_groups.items():
            # Sort by timestamp
            mint_signals.sort(key=lambda s: s.timestamp)
            
            # Use first signal for launch time and symbol
            first_signal = mint_signals[0]
            
            # Create token data
            token = HistoricalTokenData(
                mint_address=mint,
                symbol=first_signal.symbol,
                launch_time=first_signal.timestamp,
                alpha_signals=mint_signals,
                price_history=[],  # Will be populated later
                max_price=0.0,
                max_price_time=first_signal.timestamp
            )
            
            token_data.append(token)
        
        return token_data


class HistoricalBacktester:
    """Backtest strategies on historical data"""
    
    def __init__(self, config_path: str = "config.yml"):
        self.config = load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.market = MarketReality()
        
        # Initialize extractor
        self.extractor = HistoricalDataExtractor(config_path)
        
    def backtest_strategy(self, strategy: StrategyConfig, historical_data: List[HistoricalTokenData], 
                         capital_usd: float = 1000) -> List[SimulatedPosition]:
        """Backtest a strategy on historical data"""
        self.logger.info(f"🔬 Backtesting {strategy.name} on {len(historical_data)} historical tokens")
        
        positions = []
        available_capital = capital_usd
        
        for token_data in historical_data:
            try:
                position = self._simulate_token_trade(token_data, strategy, available_capital)
                if position:
                    positions.append(position)
                    # Update available capital (simplified)
                    if position.final_pnl_usd:
                        available_capital += position.final_pnl_usd
            except Exception as e:
                self.logger.error(f"❌ Error simulating {token_data.mint_address[:8]}...: {e}")
        
        self.logger.info(f"📊 Backtested {len(positions)} trades for {strategy.name}")
        return positions
    
    def _simulate_token_trade(self, token_data: HistoricalTokenData, strategy: StrategyConfig, 
                             capital_usd: float) -> Optional[SimulatedPosition]:
        """Simulate a single token trade"""
        
        # Check if strategy would have entered
        alpha_count = len(token_data.alpha_signals)
        if alpha_count < strategy.min_alpha_wallets:
            return None
        
        # Use first alpha signal as entry point
        entry_signal = token_data.alpha_signals[0]
        
        # Check independence if required
        if strategy.independence_check and alpha_count >= 2:
            # Simplified independence check - different wallets
            unique_wallets = set(s.wallet_address for s in token_data.alpha_signals[:2])
            if len(unique_wallets) < 2:
                return None
        
        # Simulate entry execution
        entry_price = entry_signal.price
        position_size_usd = min(strategy.position_size_pct * capital_usd, capital_usd * 0.1)
        
        if position_size_usd < 10:  # Minimum position size
            return None
        
        # Add realistic slippage and fees
        slippage_pct = strategy.base_slippage_pct
        execution_price = entry_price * (1 + slippage_pct)
        fees_usd = 2.0  # Simplified fee
        tokens_bought = position_size_usd / execution_price
        cost_basis = position_size_usd + fees_usd
        
        # Create position
        position = SimulatedPosition(
            mint_address=token_data.mint_address,
            symbol=token_data.symbol,
            strategy_name=strategy.name,
            entry_signal_time=entry_signal.timestamp,
            entry_execution=TradeExecution(
                signal_time=entry_signal.timestamp,
                attempted_time=entry_signal.timestamp,
                executed_time=entry_signal.timestamp,
                target_price=entry_price,
                executed_price=execution_price,
                slippage_tolerance=slippage_pct,
                actual_slippage=slippage_pct,
                success=True,
                gas_cost_usd=1.0,
                priority_fee_usd=1.0
            ),
            alpha_wallets=[s.wallet_address for s in token_data.alpha_signals],
            alpha_entry_price=entry_price,
            leader_price_delta=(execution_price - entry_price) / entry_price,
            tokens_bought=tokens_bought,
            usd_invested=position_size_usd,
            cost_basis_with_fees=cost_basis,
            peak_price=execution_price,
            is_active=False  # Will be marked complete
        )
        
        # Simulate price movements and exits (simplified)
        # For backtesting, we need to make assumptions about price movements
        # This is a limitation - we don't have full price history
        
        # Simulate exit based on strategy parameters
        max_hold_seconds = strategy.max_hold_seconds
        hold_time = min(max_hold_seconds, 3600)  # Cap at 1 hour for simplicity
        
        # Estimate final price (this is the weak point in backtesting)
        # Without real price data, we simulate based on probabilities
        final_price = self._estimate_final_price(entry_price, hold_time)
        
        # Calculate P&L
        exit_price = final_price * (1 - 0.02)  # 2% exit slippage
        usd_received = tokens_bought * exit_price - fees_usd  # Exit fees
        
        position.final_pnl_usd = usd_received - cost_basis
        position.final_pnl_pct = (position.final_pnl_usd / cost_basis) * 100
        position.hold_duration_seconds = hold_time
        
        # Determine exit reason based on P&L
        if position.final_pnl_pct <= strategy.stop_loss_pct * 100:
            position.exit_reason = "stop_loss"
        elif position.final_pnl_pct >= 20:  # TP1 level
            position.exit_reason = "take_profit"
            position.tp_levels_hit = [0.2]
        else:
            position.exit_reason = "time_based"
        
        return position
    
    def _estimate_final_price(self, entry_price: float, hold_seconds: int) -> float:
        """Estimate final price - simplified model"""
        # This is where historical backtesting gets tricky
        # Without full price history, we simulate based on memecoin statistics
        
        # Rough memecoin outcome probabilities (based on market data)
        import random
        
        outcome = random.random()
        
        if outcome < 0.15:  # 15% chance of big pump
            multiplier = random.uniform(2.0, 10.0)
        elif outcome < 0.35:  # 20% chance of moderate pump  
            multiplier = random.uniform(1.2, 2.0)
        elif outcome < 0.65:  # 30% chance of slight loss
            multiplier = random.uniform(0.8, 1.1)
        else:  # 35% chance of significant loss
            multiplier = random.uniform(0.3, 0.8)
        
        return entry_price * multiplier


def run_historical_backtest(days_back: int = 30, strategies: List[str] = None) -> Dict:
    """Run historical backtest on multiple strategies"""
    
    # Default strategies
    if strategies is None:
        strategies = ['current_aggressive', 'gpt5_recommended']
    
    # Initialize backtester
    backtester = HistoricalBacktester()
    
    # Extract historical data
    print(f"📚 Extracting {days_back} days of historical data...")
    historical_data = backtester.extractor.extract_historical_signals(days_back)
    
    if not historical_data:
        print("❌ No historical data found. Check your logs and database.")
        return {}
    
    print(f"📈 Found {len(historical_data)} tokens with alpha signals")
    
    # Load strategy configs
    strategy_configs = {}
    for strategy_name in strategies:
        if strategy_name == 'current_aggressive':
            from sims.strategy_ab_tester import StrategyABTester
            tester = StrategyABTester()
            strategy_configs[strategy_name] = tester.strategies['current_aggressive']
        elif strategy_name == 'gpt5_recommended':
            from sims.strategy_ab_tester import StrategyABTester  
            tester = StrategyABTester()
            strategy_configs[strategy_name] = tester.strategies['gpt5_recommended']
    
    # Run backtests
    results = {}
    for strategy_name, strategy_config in strategy_configs.items():
        print(f"🔬 Backtesting {strategy_name}...")
        positions = backtester.backtest_strategy(strategy_config, historical_data)
        
        # Calculate metrics
        if positions:
            completed = [p for p in positions if p.final_pnl_pct is not None]
            winners = [p for p in completed if p.final_pnl_pct > 0]
            
            win_rate = len(winners) / len(completed) * 100 if completed else 0
            avg_pnl = sum(p.final_pnl_pct for p in completed) / len(completed) if completed else 0
            total_pnl = sum(p.final_pnl_usd for p in completed) if completed else 0
            
            results[strategy_name] = {
                'total_trades': len(completed),
                'win_rate': win_rate,
                'avg_pnl_pct': avg_pnl,
                'total_pnl_usd': total_pnl,
                'positions': positions
            }
            
            print(f"📊 {strategy_name}: {len(completed)} trades, {win_rate:.1f}% win rate, {avg_pnl:+.1f}% avg P&L")
        else:
            results[strategy_name] = {
                'total_trades': 0,
                'win_rate': 0,
                'avg_pnl_pct': 0,
                'total_pnl_usd': 0,
                'positions': []
            }
    
    return results


if __name__ == "__main__":
    # Run quick historical backtest
    results = run_historical_backtest(days_back=14)
    
    print("\n🏁 Backtest Results Summary:")
    for strategy, metrics in results.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%") 
        print(f"  Avg P&L: {metrics['avg_pnl_pct']:+.1f}%")
        print(f"  Total P&L: ${metrics['total_pnl_usd']:+.2f}")