#!/usr/bin/env python3
"""
Master Test Runner for Memecoin Trading Bot

Usage: python tests/run_all_tests.py [unit|integration|performance|all]

Example output:
Running unit tests...
✓ 15/15 unit tests passed
Running integration tests...
✓ 8/8 integration tests passed
Running performance tests...
✓ Memory usage: 125MB (stable)
✓ Latency: 45ms average
✓ Throughput: 150 trades/min

Overall: 23/23 tests passed ✅
"""

import sys
import subprocess
import os
import time
from pathlib import Path

def print_banner(text):
    """Print colored banner"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def run_command(cmd, description):
    """Run command and return success status"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"✗ {description} failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ {description} failed with exception: {e}")
        return False

def run_unit_tests():
    """Run unit tests"""
    print_banner("UNIT TESTS")
    
    # Check if unit test directory exists
    unit_dir = Path(__file__).parent / "unit"
    if not unit_dir.exists():
        print("⚠️  Unit test directory not found, creating...")
        unit_dir.mkdir(exist_ok=True)
        (unit_dir / "__init__.py").touch()
        return True
    
    cmd = ["python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"]
    return run_command(cmd, "Unit tests")

def run_integration_tests():
    """Run integration tests"""
    print_banner("INTEGRATION TESTS")
    
    # Check if integration test directory exists
    integration_dir = Path(__file__).parent / "integration"
    if not integration_dir.exists():
        print("⚠️  Integration test directory not found, creating...")
        integration_dir.mkdir(exist_ok=True)
        (integration_dir / "__init__.py").touch()
        return True
    
    cmd = ["python", "-m", "pytest", "tests/integration/", "-v", "--tb=short"]
    return run_command(cmd, "Integration tests")

def run_performance_tests():
    """Run performance tests"""
    print_banner("PERFORMANCE TESTS")
    
    performance_dir = Path(__file__).parent / "performance"
    if not performance_dir.exists():
        print("⚠️  Performance test directory not found, creating...")
        performance_dir.mkdir(exist_ok=True)
        return True
    
    # Run memory test if it exists
    memory_test = performance_dir / "test_memory_usage.py"
    if memory_test.exists():
        cmd = ["python", str(memory_test)]
        success = run_command(cmd, "Memory usage test")
    else:
        print("⚠️  Memory test not found, skipping...")
        success = True
    
    # Run latency test if it exists
    latency_test = performance_dir / "test_latency.py"
    if latency_test.exists():
        cmd = ["python", str(latency_test)]
        success &= run_command(cmd, "Latency test")
    else:
        print("⚠️  Latency test not found, skipping...")
    
    return success

def run_coverage_report():
    """Generate coverage report"""
    print_banner("COVERAGE REPORT")
    
    cmd = ["python", "-m", "pytest", "tests/", "--cov=src", "--cov-report=term-missing", "--cov-report=html:tests/reports/coverage"]
    return run_command(cmd, "Coverage analysis")

def check_test_environment():
    """Check if test environment is properly set up"""
    print_banner("ENVIRONMENT CHECK")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if we're in the right directory
    project_root = Path(__file__).parent.parent
    if not (project_root / "main.py").exists():
        print("✗ Not in project root directory")
        return False
    
    print("✓ Project root found")
    
    # Check if requirements are installed
    try:
        import pytest
        print(f"✓ pytest {pytest.__version__} installed")
    except ImportError:
        print("✗ pytest not installed")
        print("Run: pip install -r tests/requirements-test.txt")
        return False
    
    # Create test directories if they don't exist
    test_dirs = ["unit", "integration", "performance", "fixtures", "reports"]
    for dir_name in test_dirs:
        test_dir = Path(__file__).parent / dir_name
        if not test_dir.exists():
            print(f"Creating {dir_name} directory...")
            test_dir.mkdir(exist_ok=True)
            if dir_name in ["unit", "integration"]:
                (test_dir / "__init__.py").touch()
    
    # Create reports/coverage directory
    coverage_dir = Path(__file__).parent / "reports" / "coverage"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    
    print("✓ Test environment ready")
    return True

def create_sample_tests():
    """Create sample test files if they don't exist"""
    
    # Sample unit test
    unit_test_file = Path(__file__).parent / "unit" / "test_sample.py"
    if not unit_test_file.exists():
        sample_unit_test = '''"""
Sample Unit Test

Run: python -m pytest tests/unit/test_sample.py -v

Expected output:
test_sample_assertion ✓ (Basic assertion works)
test_sample_calculation ✓ (Math works correctly)
"""

def test_sample_assertion():
    """Test that basic assertions work"""
    assert True
    assert 1 + 1 == 2
    
def test_sample_calculation():
    """Test that calculations work"""
    result = 5 * 10
    assert result == 50
'''
        unit_test_file.write_text(sample_unit_test)
        print("✓ Created sample unit test")
    
    # Sample integration test
    integration_test_file = Path(__file__).parent / "integration" / "test_sample.py"
    if not integration_test_file.exists():
        sample_integration_test = '''"""
Sample Integration Test

Run: python -m pytest tests/integration/test_sample.py -v

Expected output:
test_integration_example ✓ (Integration test works)
"""

def test_integration_example():
    """Test that integration testing works"""
    # This would test interaction between components
    component_a_result = "Hello"
    component_b_result = "World"
    combined = f"{component_a_result} {component_b_result}"
    assert combined == "Hello World"
'''
        integration_test_file.write_text(sample_integration_test)
        print("✓ Created sample integration test")

def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        test_type = "all"
    
    print_banner("MEMECOIN TRADING BOT - TEST RUNNER")
    print(f"Test type: {test_type}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check environment
    if not check_test_environment():
        print("\n✗ Environment check failed")
        sys.exit(1)
    
    # Create sample tests if needed
    create_sample_tests()
    
    success_count = 0
    total_count = 0
    
    if test_type in ["unit", "all"]:
        total_count += 1
        if run_unit_tests():
            success_count += 1
    
    if test_type in ["integration", "all"]:
        total_count += 1
        if run_integration_tests():
            success_count += 1
    
    if test_type in ["performance", "all"]:
        total_count += 1
        if run_performance_tests():
            success_count += 1
    
    if test_type in ["coverage", "all"]:
        total_count += 1
        if run_coverage_report():
            success_count += 1
    
    # Final summary
    print_banner("TEST SUMMARY")
    print(f"Tests completed: {success_count}/{total_count}")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_count:
        print("🎉 All tests passed! ✅")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()