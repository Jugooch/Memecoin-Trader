#!/usr/bin/env python3
"""
Frontrun Bot - Test Runner

Quick test execution with common scenarios
"""

import sys
import subprocess


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and print results"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    if len(sys.argv) < 2:
        print("Frontrun Bot Test Runner\n")
        print("Usage:")
        print("  python run_tests.py all          # Run all tests")
        print("  python run_tests.py unit         # Run unit tests only")
        print("  python run_tests.py integration  # Run integration tests")
        print("  python run_tests.py fast         # Run unit tests (fast)")
        print("  python run_tests.py coverage     # Run with coverage report")
        print("  python run_tests.py phase1       # Run Phase 1 tests only")
        print("  python run_tests.py phase2       # Run Phase 2 tests only")
        return 1

    mode = sys.argv[1].lower()

    if mode == "all":
        return run_command(
            [sys.executable, "-m", "pytest", ".", "-v"],
            "Running All Tests"
        )

    elif mode == "unit":
        return run_command(
            [sys.executable, "-m", "pytest", "unit/", "-v"],
            "Running Unit Tests Only"
        )

    elif mode == "integration":
        return run_command(
            [sys.executable, "-m", "pytest", "integration/", "-v"],
            "Running Integration Tests"
        )

    elif mode == "fast":
        return run_command(
            [sys.executable, "-m", "pytest", "unit/", "-v", "-m", "not slow"],
            "Running Fast Unit Tests"
        )

    elif mode == "coverage":
        returncode = run_command(
            [sys.executable, "-m", "pytest", ".", "--cov=core", "--cov-report=html", "--cov-report=term"],
            "Running Tests with Coverage"
        )

        if returncode == 0:
            print("\n✅ Coverage report generated: htmlcov/index.html")

        return returncode

    elif mode == "phase1":
        print("\n" + "="*60)
        print("  Phase 1 Regression Tests")
        print("="*60 + "\n")

        # Unit tests
        print("\n📦 Running Unit Tests...")
        returncode = subprocess.run([
            sys.executable, "-m", "pytest",
            "unit/test_config.py",
            "unit/test_metrics.py",
            "-v"
        ]).returncode

        if returncode != 0:
            return returncode

        # Integration tests
        print("\n🌐 Running Integration Tests...")
        returncode = subprocess.run([
            sys.executable, "-m", "pytest",
            "integration/test_phase1_integration.py",
            "-v",
            "-m", "integration and not slow"
        ]).returncode

        return returncode

    elif mode == "phase2":
        print("\n" + "="*60)
        print("  Phase 2 Complete Test Suite")
        print("="*60 + "\n")

        # Unit tests
        print("\n📦 Running Phase 2 Unit Tests...")
        returncode = subprocess.run([
            sys.executable, "-m", "pytest",
            "unit/test_tx_builder.py",
            "unit/test_tx_signer.py",
            "unit/test_tx_submitter.py",
            "unit/test_priority_fees.py",
            "-v"
        ]).returncode

        if returncode != 0:
            return returncode

        # Integration tests
        print("\n🌐 Running Phase 2 Integration Tests...")
        returncode = subprocess.run([
            sys.executable, "-m", "pytest",
            "integration/test_phase2_integration.py",
            "-v",
            "-m", "integration and not slow"
        ]).returncode

        return returncode

    else:
        print(f"❌ Unknown mode: {mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
