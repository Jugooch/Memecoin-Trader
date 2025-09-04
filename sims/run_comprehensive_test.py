#!/usr/bin/env python3
"""
Comprehensive Strategy Testing Suite
Runs both historical backtesting and live A/B testing with GPT5's required metrics
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sims.historical_backtester import run_historical_backtest
from sims.strategy_ab_tester import StrategyABTester
from src.utils.logger_setup import setup_logging


class ComprehensiveStrategyTester:
    """Complete strategy testing with both historical and live components"""
    
    def __init__(self):
        self.logger = setup_logging('INFO', 'sims/logs/comprehensive_test.log')
        self.results_dir = Path('sims/results')
        self.results_dir.mkdir(exist_ok=True)
        
    async def run_full_test_suite(self, historical_days: int = 14, live_hours: float = 6.0) -> Dict[str, Any]:
        """Run complete test suite"""
        print("🚀 Starting Comprehensive Strategy Testing Suite")
        print("="*60)
        
        test_results = {
            'test_info': {
                'start_time': datetime.now().isoformat(),
                'historical_days': historical_days,
                'live_hours': live_hours
            },
            'historical_backtest': {},
            'live_ab_test': {},
            'analysis': {},
            'recommendations': {}
        }
        
        # Phase 1: Historical Backtesting
        print(f"\n📚 Phase 1: Historical Backtesting ({historical_days} days)")
        print("-" * 50)
        
        try:
            historical_results = run_historical_backtest(
                days_back=historical_days,
                strategies=['current_aggressive', 'gpt5_recommended']
            )
            test_results['historical_backtest'] = historical_results
            
            # Display historical results
            self._display_historical_results(historical_results)
            
        except Exception as e:
            print(f"❌ Historical backtesting failed: {e}")
            self.logger.error(f"Historical backtesting error: {e}")
            test_results['historical_backtest']['error'] = str(e)
        
        # Phase 2: Live A/B Testing
        print(f"\n📡 Phase 2: Live A/B Testing ({live_hours} hours)")
        print("-" * 50)
        
        try:
            ab_tester = StrategyABTester()
            live_results = await ab_tester.run_ab_test(duration_hours=live_hours)
            test_results['live_ab_test'] = live_results
            
        except Exception as e:
            print(f"❌ Live A/B testing failed: {e}")
            self.logger.error(f"Live A/B testing error: {e}")
            test_results['live_ab_test']['error'] = str(e)
        
        # Phase 3: Analysis and Recommendations
        print(f"\n📊 Phase 3: Analysis & Recommendations")
        print("-" * 50)
        
        analysis = self._generate_comprehensive_analysis(test_results)
        test_results['analysis'] = analysis
        test_results['recommendations'] = self._generate_recommendations(analysis)
        
        # Save results
        results_file = self._save_results(test_results)
        
        # Display final summary
        self._display_final_summary(test_results)
        
        print(f"\n📄 Complete results saved to: {results_file}")
        return test_results
    
    def _display_historical_results(self, results: Dict):
        """Display historical backtest results"""
        if not results:
            print("⚠️ No historical results available")
            return
        
        print(f"{'Strategy':<20} {'Trades':<8} {'Win Rate':<10} {'Avg P&L':<10} {'Total P&L':<12}")
        print("-" * 70)
        
        for strategy, metrics in results.items():
            if isinstance(metrics, dict):
                print(f"{strategy:<20} {metrics['total_trades']:<8} "
                      f"{metrics['win_rate']:<10.1f}% {metrics['avg_pnl_pct']:<+10.1f}% "
                      f"${metrics['total_pnl_usd']:<+12.2f}")
        
        # Determine historical winner
        if len(results) >= 2:
            sorted_strategies = sorted(
                [(k, v) for k, v in results.items() if isinstance(v, dict)],
                key=lambda x: x[1]['win_rate'],
                reverse=True
            )
            winner = sorted_strategies[0]
            print(f"\n🏆 Historical Winner: {winner[0]} ({winner[1]['win_rate']:.1f}% win rate)")
    
    def _generate_comprehensive_analysis(self, results: Dict) -> Dict[str, Any]:
        """Generate comprehensive analysis comparing all results"""
        analysis = {
            'data_quality': self._assess_data_quality(results),
            'strategy_comparison': self._compare_strategies(results),
            'gpt5_metrics': self._calculate_gpt5_metrics(results),
            'statistical_significance': self._assess_significance(results),
            'key_insights': []
        }
        
        # Generate insights
        insights = self._extract_key_insights(results, analysis)
        analysis['key_insights'] = insights
        
        return analysis
    
    def _assess_data_quality(self, results: Dict) -> Dict:
        """Assess the quality and reliability of test data"""
        quality = {
            'historical_data_points': 0,
            'live_signals_detected': 0,
            'live_positions_opened': 0,
            'data_reliability': 'unknown'
        }
        
        # Historical data quality
        hist_results = results.get('historical_backtest', {})
        if hist_results:
            total_trades = sum(
                metrics.get('total_trades', 0) 
                for metrics in hist_results.values() 
                if isinstance(metrics, dict)
            )
            quality['historical_data_points'] = total_trades
        
        # Live data quality
        live_results = results.get('live_ab_test', {})
        if live_results and 'test_summary' in live_results:
            quality['live_signals_detected'] = live_results['test_summary'].get('signals_detected', 0)
            quality['live_positions_opened'] = live_results['test_summary'].get('positions_opened', 0)
        
        # Overall reliability assessment
        if quality['historical_data_points'] >= 20 and quality['live_signals_detected'] >= 5:
            quality['data_reliability'] = 'high'
        elif quality['historical_data_points'] >= 10 and quality['live_signals_detected'] >= 2:
            quality['data_reliability'] = 'medium'
        else:
            quality['data_reliability'] = 'low'
        
        return quality
    
    def _compare_strategies(self, results: Dict) -> Dict:
        """Compare strategies across both historical and live tests"""
        comparison = {
            'current_aggressive': {'historical': {}, 'live': {}},
            'gpt5_recommended': {'historical': {}, 'live': {}},
            'winner': {'historical': None, 'live': None, 'overall': None}
        }
        
        # Historical comparison
        hist_results = results.get('historical_backtest', {})
        for strategy in ['current_aggressive', 'gpt5_recommended']:
            if strategy in hist_results:
                comparison[strategy]['historical'] = hist_results[strategy]
        
        # Live comparison
        live_results = results.get('live_ab_test', {})
        if 'strategies' in live_results:
            for strategy in ['current_aggressive', 'gpt5_recommended']:
                if strategy in live_results['strategies']:
                    comparison[strategy]['live'] = live_results['strategies'][strategy]
        
        # Determine winners
        comparison['winner'] = self._determine_winners(comparison)
        
        return comparison
    
    def _calculate_gpt5_metrics(self, results: Dict) -> Dict:
        """Calculate the specific metrics GPT5 demanded"""
        gpt5_metrics = {
            'fill_rate': {},
            'median_leader_delta': {},
            'tp_hit_rates': {},
            'loss_distribution': {},
            'early_recovery_rate': {},
            'fee_burden': {}
        }
        
        # Extract from live results (more accurate than historical simulation)
        live_results = results.get('live_ab_test', {})
        if 'strategies' in live_results:
            for strategy, strategy_data in live_results['strategies'].items():
                if strategy in ['current_aggressive', 'gpt5_recommended']:
                    gpt5_metrics['fill_rate'][strategy] = strategy_data.get('fill_rate', 'N/A')
                    gpt5_metrics['median_leader_delta'][strategy] = strategy_data.get('median_leader_delta', 'N/A')
                    gpt5_metrics['tp_hit_rates'][strategy] = strategy_data.get('tp1_hit_rate', 'N/A')
                    gpt5_metrics['loss_distribution'][strategy] = strategy_data.get('loss_distribution', {})
                    gpt5_metrics['early_recovery_rate'][strategy] = strategy_data.get('early_recovery_rate', 'N/A')
                    gpt5_metrics['fee_burden'][strategy] = strategy_data.get('median_fee_burden', 'N/A')
        
        return gpt5_metrics
    
    def _assess_significance(self, results: Dict) -> Dict:
        """Assess statistical significance of results"""
        significance = {
            'sample_size_adequate': False,
            'win_rate_difference': 0,
            'confidence_level': 'low',
            'recommendation_confidence': 'low'
        }
        
        # Check sample sizes
        hist_results = results.get('historical_backtest', {})
        live_results = results.get('live_ab_test', {})
        
        total_samples = 0
        if hist_results:
            total_samples += sum(
                metrics.get('total_trades', 0) 
                for metrics in hist_results.values() 
                if isinstance(metrics, dict)
            )
        
        if live_results and 'test_summary' in live_results:
            total_samples += live_results['test_summary'].get('positions_opened', 0)
        
        significance['sample_size_adequate'] = total_samples >= 30
        
        # Calculate win rate differences
        current_wr = 0
        gpt5_wr = 0
        
        if 'strategies' in live_results:
            current_wr = live_results['strategies'].get('current_aggressive', {}).get('win_rate', 0)
            gpt5_wr = live_results['strategies'].get('gpt5_recommended', {}).get('win_rate', 0)
        
        significance['win_rate_difference'] = gpt5_wr - current_wr
        
        # Determine confidence level
        if significance['sample_size_adequate'] and abs(significance['win_rate_difference']) > 10:
            significance['confidence_level'] = 'high'
        elif total_samples >= 15 and abs(significance['win_rate_difference']) > 5:
            significance['confidence_level'] = 'medium'
        else:
            significance['confidence_level'] = 'low'
        
        return significance
    
    def _determine_winners(self, comparison: Dict) -> Dict:
        """Determine winners across different test phases"""
        winners = {'historical': None, 'live': None, 'overall': None}
        
        # Historical winner (by win rate)
        hist_current = comparison['current_aggressive']['historical']
        hist_gpt5 = comparison['gpt5_recommended']['historical']
        
        if hist_current and hist_gpt5:
            if hist_current.get('win_rate', 0) > hist_gpt5.get('win_rate', 0):
                winners['historical'] = 'current_aggressive'
            else:
                winners['historical'] = 'gpt5_recommended'
        
        # Live winner (by win rate)
        live_current = comparison['current_aggressive']['live']
        live_gpt5 = comparison['gpt5_recommended']['live']
        
        if live_current and live_gpt5:
            if live_current.get('win_rate', 0) > live_gpt5.get('win_rate', 0):
                winners['live'] = 'current_aggressive'
            else:
                winners['live'] = 'gpt5_recommended'
        
        # Overall winner (prioritize live results, fall back to historical)
        if winners['live']:
            winners['overall'] = winners['live']
        elif winners['historical']:
            winners['overall'] = winners['historical']
        
        return winners
    
    def _extract_key_insights(self, results: Dict, analysis: Dict) -> List[str]:
        """Extract key insights from the analysis"""
        insights = []
        
        quality = analysis['data_quality']
        comparison = analysis['strategy_comparison']
        significance = analysis['statistical_significance']
        
        # Data quality insights
        if quality['data_reliability'] == 'low':
            insights.append("⚠️ Low data reliability - results should be interpreted cautiously")
        
        # Win rate insights
        win_rate_diff = significance['win_rate_difference']
        if abs(win_rate_diff) > 15:
            better_strategy = 'GPT5' if win_rate_diff > 0 else 'Current'
            insights.append(f"🎯 {better_strategy} strategy shows {abs(win_rate_diff):.1f}% higher win rate")
        
        # Mathematical viability
        live_results = results.get('live_ab_test', {})
        if 'strategies' in live_results:
            for strategy, data in live_results['strategies'].items():
                win_rate = data.get('win_rate', 0)
                required_wr = data.get('required_win_rate', 50)
                
                if win_rate > required_wr:
                    insights.append(f"✅ {strategy} is mathematically profitable ({win_rate:.1f}% > {required_wr:.1f}% required)")
                else:
                    insights.append(f"❌ {strategy} is mathematically unprofitable ({win_rate:.1f}% < {required_wr:.1f}% required)")
        
        # Execution insights
        gpt5_metrics = analysis['gpt5_metrics']
        if gpt5_metrics['fill_rate']:
            for strategy, fill_rate in gpt5_metrics['fill_rate'].items():
                if isinstance(fill_rate, (int, float)) and fill_rate < 80:
                    insights.append(f"📉 {strategy} has low fill rate ({fill_rate:.1f}%) - missing winners")
        
        return insights
    
    def _generate_recommendations(self, analysis: Dict) -> Dict:
        """Generate actionable recommendations based on analysis"""
        recommendations = {
            'immediate_actions': [],
            'configuration_changes': {},
            'further_testing_needed': [],
            'confidence_level': analysis['statistical_significance']['confidence_level']
        }
        
        comparison = analysis['strategy_comparison']
        significance = analysis['statistical_significance']
        insights = analysis['key_insights']
        
        # Immediate actions based on results
        if significance['win_rate_difference'] > 10 and significance['confidence_level'] in ['medium', 'high']:
            if significance['win_rate_difference'] > 0:
                recommendations['immediate_actions'].append("Switch to GPT5 recommended configuration")
            else:
                recommendations['immediate_actions'].append("Keep current aggressive configuration")
        else:
            recommendations['immediate_actions'].append("Continue testing - results inconclusive")
        
        # Configuration changes
        if any("fill rate" in insight for insight in insights):
            recommendations['configuration_changes']['slippage'] = "Increase slippage tolerance to 5-8%"
        
        if any("unprofitable" in insight for insight in insights):
            recommendations['configuration_changes']['stop_loss'] = "Tighten stop losses to -20% or -15%"
            recommendations['configuration_changes']['position_sizing'] = "Reduce position size to 1-2%"
        
        # Further testing needed
        if analysis['data_quality']['data_reliability'] == 'low':
            recommendations['further_testing_needed'].append("Run longer test period (24-48 hours)")
        
        if significance['sample_size_adequate'] == False:
            recommendations['further_testing_needed'].append("Collect more data points (minimum 30 trades)")
        
        return recommendations
    
    def _save_results(self, results: Dict) -> str:
        """Save comprehensive results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"comprehensive_test_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return str(filepath)
    
    def _display_final_summary(self, results: Dict):
        """Display final comprehensive summary"""
        print("\n" + "="*60)
        print("🏁 FINAL SUMMARY & RECOMMENDATIONS")
        print("="*60)
        
        analysis = results.get('analysis', {})
        recommendations = results.get('recommendations', {})
        
        # Display key insights
        insights = analysis.get('key_insights', [])
        if insights:
            print("\n📋 Key Insights:")
            for insight in insights[:5]:  # Top 5 insights
                print(f"  • {insight}")
        
        # Display winner
        comparison = analysis.get('strategy_comparison', {})
        winner = comparison.get('winner', {}).get('overall')
        if winner:
            print(f"\n🏆 Overall Winner: {winner.upper()}")
        
        # Display recommendations
        immediate_actions = recommendations.get('immediate_actions', [])
        if immediate_actions:
            print(f"\n⚡ Immediate Actions:")
            for action in immediate_actions:
                print(f"  • {action}")
        
        config_changes = recommendations.get('configuration_changes', {})
        if config_changes:
            print(f"\n🔧 Configuration Changes:")
            for key, value in config_changes.items():
                print(f"  • {key}: {value}")
        
        # Display confidence
        confidence = recommendations.get('confidence_level', 'low')
        print(f"\n📊 Recommendation Confidence: {confidence.upper()}")
        
        # The bottom line for your specific situation
        print(f"\n" + "-"*60)
        print("💰 THE BOTTOM LINE:")
        
        live_results = results.get('live_ab_test', {})
        if 'strategies' in live_results:
            current_wr = live_results['strategies'].get('current_aggressive', {}).get('win_rate', 0)
            gpt5_wr = live_results['strategies'].get('gpt5_recommended', {}).get('win_rate', 0)
            
            if max(current_wr, gpt5_wr) > 40:
                print("✅ You have a profitable strategy! Deploy it carefully.")
            elif max(current_wr, gpt5_wr) > 35:
                print("🟡 Close to profitability. Fine-tune and test more.")
            else:
                print("❌ Neither strategy is profitable yet. Major changes needed.")
        else:
            print("⚠️ Insufficient data to make definitive recommendation.")


async def main():
    """Run the comprehensive test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Strategy Testing Suite')
    parser.add_argument('--historical-days', type=int, default=14, help='Days of historical data to analyze')
    parser.add_argument('--live-hours', type=float, default=6.0, help='Hours of live testing')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test (2 hours live)')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.live_hours = 2.0
        args.historical_days = 7
    
    tester = ComprehensiveStrategyTester()
    results = await tester.run_full_test_suite(
        historical_days=args.historical_days,
        live_hours=args.live_hours
    )
    
    return results


if __name__ == "__main__":
    try:
        results = asyncio.run(main())
        print(f"\n🎉 Test suite completed successfully!")
    except KeyboardInterrupt:
        print(f"\n⚠️ Test suite interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()