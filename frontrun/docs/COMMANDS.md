# Command Cheat Sheet

Quick reference for all commonly used commands in the Frontrun Bot project.

**Last Updated**: Phase 3 Complete (316 tests)

---

## Table of Contents

- [Setup & Installation](#setup--installation)
- [Testing Commands](#testing-commands)
- [Development Workflow](#development-workflow)
- [Solana Devnet Commands](#solana-devnet-commands)
- [Database Management](#database-management)
- [Code Quality](#code-quality)
- [Git Operations](#git-operations)
- [Debugging & Troubleshooting](#debugging--troubleshooting)

---

## Setup & Installation

### Initial Setup

```bash
# Navigate to project
cd frontrun

# Linux/Mac - Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Windows - Run setup script
.\scripts\setup.bat

# Manual setup
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate.bat  # Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Configuration

```bash
# Copy example config
cp config/config.yml.example config/config.yml

# Edit config (Linux/Mac)
nano config/config.yml

# Edit config (Windows)
notepad config/config.yml
```

### Environment Activation

```bash
# Linux/Mac
source venv/bin/activate

# Windows (CMD)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Deactivate
deactivate
```

---

## Testing Commands

### Run All Tests

```bash
# All tests (316 tests: 309 unit + 7 integration)
python -m pytest tests/ -v --tb=short

# All tests with timing
python -m pytest tests/ -v --durations=10

# Stop on first failure
python -m pytest tests/ -v -x
```

### Unit Tests (309 tests - Fast, No Network)

```bash
# All unit tests
python -m pytest tests/unit -v --tb=short

# Specific phase
python -m pytest tests/unit/test_config.py tests/unit/test_metrics.py -v  # Phase 1
python -m pytest tests/unit/test_tx_*.py tests/unit/test_priority_fees.py tests/unit/test_wallet_manager.py -v  # Phase 2
python -m pytest tests/unit/test_pumpfun_client.py tests/unit/test_bonding_curve.py tests/unit/test_slippage.py tests/unit/test_pnl.py tests/unit/test_position_tracker.py -v  # Phase 3

# Specific component
python -m pytest tests/unit/test_bonding_curve.py -v
python -m pytest tests/unit/test_position_tracker.py -v
python -m pytest tests/unit/test_pumpfun_client.py -v

# Specific test function
python -m pytest tests/unit/test_bonding_curve.py::TestBondingCurveCalculator::test_calculate_buy_price -v

# With print statements visible
python -m pytest tests/unit -v -s
```

### Integration Tests (7 tests - Requires 2 SOL Devnet)

```bash
# All integration tests (requires funded wallet)
python -m pytest tests/integration -v --tb=short

# Specific integration test
python -m pytest tests/integration/test_devnet_rpc_connection.py -v
python -m pytest tests/integration/test_devnet_wallet_operations.py -v
python -m pytest tests/integration/test_devnet_bonding_curve_read.py -v
python -m pytest tests/integration/test_devnet_transaction_submission.py -v
python -m pytest tests/integration/test_devnet_full_trade_flow.py -v
python -m pytest tests/integration/test_phase2_integration.py -v
python -m pytest tests/integration/test_phase3_complete_trade_flow.py -v
```

### Coverage Reports

```bash
# Generate HTML coverage report
python -m pytest tests/unit --cov=core --cov=clients --cov-report=html --cov-report=term

# View coverage report (Linux/Mac)
open htmlcov/index.html

# View coverage report (Windows)
start htmlcov/index.html

# Coverage with missing lines
python -m pytest tests/unit --cov=core --cov=clients --cov-report=term-missing

# Minimum coverage threshold (fail if below 80%)
python -m pytest tests/unit --cov=core --cov=clients --cov-fail-under=80
```

### Test Filtering

```bash
# Run tests by marker
python -m pytest -m integration -v  # Only integration tests
python -m pytest -m "not integration" -v  # Skip integration tests
python -m pytest -m slow -v  # Only slow tests

# Run tests by name pattern
python -m pytest tests/ -k "bonding_curve" -v  # All bonding curve tests
python -m pytest tests/ -k "test_buy or test_sell" -v  # Tests matching pattern

# Run last failed tests
python -m pytest tests/ --lf -v

# Run failed tests first
python -m pytest tests/ --ff -v
```

---

## Development Workflow

### Running Components Directly

```bash
# Test RPC connection
python -m core.rpc_manager

# Test health monitoring
python -m core.health_monitor

# Test bonding curve calculations
python -m core.bonding_curve

# Test logger
python -m core.logger

# Test metrics
python -m core.metrics

# Test configuration loading
python -m core.config
```

### Python REPL (Interactive Testing)

```bash
# Start IPython (enhanced REPL)
ipython

# In IPython:
from core.bonding_curve import BondingCurveCalculator, BondingCurveState
calculator = BondingCurveCalculator()
curve_state = BondingCurveState(
    virtual_token_reserves=1_000_000_000_000,
    virtual_sol_reserves=30_000_000_000,
    real_token_reserves=800_000_000_000,
    real_sol_reserves=20_000_000_000,
    token_total_supply=1_000_000_000_000,
    complete=False
)
quote = calculator.calculate_buy_price(curve_state, 1_000_000_000)
print(f"Tokens out: {quote.tokens_out:,}")
```

### Package Management

```bash
# List installed packages
pip list

# Show package info
pip show solders
pip show aiosqlite

# Check for outdated packages
pip list --outdated

# Upgrade package
pip install --upgrade solders

# Install new package
pip install package-name

# Freeze requirements
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## Solana Devnet Commands

### Wallet Management

```bash
# Generate test wallet (already done - creates tests/integration/devnet-test-wallet.json)
python tests/integration/generate_test_wallet.py

# Check wallet balance
solana balance tests/integration/devnet-test-wallet.json --url devnet

# Get wallet address
solana address -k tests/integration/devnet-test-wallet.json
```

### Airdrop SOL

```bash
# Method 1: Solana CLI (recommended)
solana airdrop 2 <WALLET_ADDRESS> --url devnet

# Method 2: Check balance after airdrop
solana balance tests/integration/devnet-test-wallet.json --url devnet

# Method 3: Web faucet
# Go to: https://faucet.solana.com/
# Select "Devnet", enter address, request 2 SOL
```

### Network Info

```bash
# Check cluster status
solana cluster-version --url devnet

# Get current slot
solana slot --url devnet

# Get epoch info
solana epoch-info --url devnet

# Get block time
solana block-time <SLOT> --url devnet

# Get transaction details
solana confirm <SIGNATURE> --url devnet
```

---

## Database Management

### SQLite Position Database

```bash
# View database (requires sqlite3)
sqlite3 data/positions.db

# In sqlite3:
.tables                              # List tables
.schema positions                    # Show schema
SELECT * FROM positions;             # View all positions
SELECT * FROM closed_positions;      # View closed positions
.quit                                # Exit

# One-liner queries
sqlite3 data/positions.db "SELECT COUNT(*) FROM positions;"
sqlite3 data/positions.db "SELECT * FROM positions WHERE status='open';"

# Delete test databases
rm data/positions.db
rm data/positions.db-journal
rm tests/unit/.test_data/*.db
```

### Database Backup

```bash
# Backup database
cp data/positions.db data/positions.db.backup

# Restore from backup
cp data/positions.db.backup data/positions.db

# Export to SQL
sqlite3 data/positions.db .dump > positions_backup.sql

# Import from SQL
sqlite3 data/positions.db < positions_backup.sql
```

---

## Code Quality

### Formatting

```bash
# Format code with black
black core/ clients/ tests/

# Check what would be formatted (dry run)
black --check core/ clients/ tests/

# Format specific file
black core/bonding_curve.py

# Format with line length
black --line-length 100 core/
```

### Linting

```bash
# Run flake8
flake8 core/ clients/ tests/

# Flake8 with specific rules
flake8 --max-line-length=100 --ignore=E501,W503 core/

# Run mypy (type checking)
mypy core/ clients/

# Run pylint
pylint core/ clients/
```

### Imports

```bash
# Sort imports with isort
isort core/ clients/ tests/

# Check imports (dry run)
isort --check-only core/ clients/ tests/

# Sort specific file
isort core/bonding_curve.py
```

### All at Once

```bash
# Format, sort imports, and lint
black core/ clients/ tests/ && isort core/ clients/ tests/ && flake8 core/ clients/ tests/
```

---

## Git Operations

### Common Git Commands

```bash
# Check status
git status

# View diff
git diff
git diff core/bonding_curve.py

# Stage changes
git add .
git add core/bonding_curve.py

# Commit
git commit -m "Add bonding curve calculator"

# Push
git push origin main

# Pull latest
git pull origin main

# View log
git log --oneline -10

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard changes
git checkout -- core/bonding_curve.py
```

### Branch Management

```bash
# Create new branch
git checkout -b feature/phase4-mempool

# Switch branch
git checkout main

# List branches
git branch -a

# Delete branch
git branch -d feature/old-feature

# Merge branch
git checkout main
git merge feature/phase4-mempool
```

### Stash Changes

```bash
# Stash changes
git stash

# List stashes
git stash list

# Apply stash
git stash apply

# Pop stash
git stash pop

# Clear stashes
git stash clear
```

---

## Debugging & Troubleshooting

### Python Debugging

```bash
# Run with debugger (ipdb)
python -m ipdb core/bonding_curve.py

# Add breakpoint in code:
# import ipdb; ipdb.set_trace()

# Run tests with debugger
python -m pytest tests/unit/test_bonding_curve.py --pdb

# Drop into debugger on failure
python -m pytest tests/unit/test_bonding_curve.py --pdb -x
```

### Verbose Output

```bash
# Run tests with verbose output
python -m pytest tests/ -v -s --tb=long

# Show local variables on failure
python -m pytest tests/ -v --tb=long --showlocals

# Show full diff on assertion errors
python -m pytest tests/ -v --tb=long -vv
```

### Performance Profiling

```bash
# Profile test execution time
python -m pytest tests/ --durations=20

# Profile with cProfile
python -m cProfile -o profile.stats core/bonding_curve.py

# View profile stats
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

### Logging & Output

```bash
# Run with debug logging
LOG_LEVEL=DEBUG python -m core.rpc_manager

# Capture logs during tests
python -m pytest tests/ -v --log-cli-level=DEBUG

# Save test output to file
python -m pytest tests/ -v > test_output.txt 2>&1
```

### Cleanup

```bash
# Remove Python cache
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Remove test artifacts
rm -rf .pytest_cache
rm -rf htmlcov
rm -rf .coverage
rm -f .test_wallets_cache.json

# Remove all temp files
rm -rf data/*.db data/*.db-journal
rm -rf tests/unit/.test_data/*.db
rm -rf logs/*.log
```

---

## Quick Reference Cards

### Daily Development Workflow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Pull latest changes
git pull origin main

# 3. Run tests
python -m pytest tests/unit -v

# 4. Make changes
# ... edit files ...

# 5. Format code
black core/ clients/

# 6. Run tests again
python -m pytest tests/unit -v

# 7. Commit
git add .
git commit -m "Description"
git push origin main
```

### Pre-Commit Checklist

```bash
# Run these before every commit:
python -m pytest tests/unit -v --tb=short          # All unit tests pass
black --check core/ clients/ tests/                # Code formatted
flake8 core/ clients/                              # No linting errors
python -m pytest tests/unit --cov=core --cov-fail-under=80  # Coverage >80%
```

### Pre-Phase 4 Validation

```bash
# Follow TEST_EXECUTION_CHECKLIST.md
# 1. Fund wallet
solana airdrop 2 <WALLET_ADDRESS> --url devnet

# 2. Verify balance
solana balance tests/integration/devnet-test-wallet.json --url devnet

# 3. Run all unit tests
python -m pytest tests/unit -v --tb=short

# 4. Run all integration tests
python -m pytest tests/integration -v --tb=short

# 5. Verify results
# Expected: 316/316 tests passing ✅
```

---

## Environment Variables

```bash
# Set environment variables (Linux/Mac)
export HELIUS_API_KEY="your-key-here"
export LOG_LEVEL="DEBUG"

# Set environment variables (Windows CMD)
set HELIUS_API_KEY=your-key-here
set LOG_LEVEL=DEBUG

# Set environment variables (Windows PowerShell)
$env:HELIUS_API_KEY="your-key-here"
$env:LOG_LEVEL="DEBUG"

# Use in config.yml
# url: "https://api.helius.com/?api-key=${HELIUS_API_KEY}"
```

---

## Useful Aliases (Optional)

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Navigate to project
alias cdfront='cd /path/to/frontrun'

# Activate environment
alias venv-front='source /path/to/frontrun/venv/bin/activate'

# Run tests
alias test-unit='python -m pytest tests/unit -v --tb=short'
alias test-all='python -m pytest tests/ -v --tb=short'
alias test-cov='python -m pytest tests/unit --cov=core --cov=clients --cov-report=html'

# Format code
alias fmt='black core/ clients/ tests/ && isort core/ clients/ tests/'

# Git shortcuts
alias gs='git status'
alias gd='git diff'
alias gc='git commit -m'
alias gp='git push origin main'
```

After adding, reload shell:
```bash
source ~/.bashrc  # or ~/.zshrc
```

---

## Help & Documentation

```bash
# Pytest help
python -m pytest --help

# Python module help
python -c "from core.bonding_curve import BondingCurveCalculator; help(BondingCurveCalculator)"

# Package documentation
pip show solders

# List all available pytest markers
python -m pytest --markers
```

---

## Notes

- **Always activate virtual environment** before running commands
- **Run unit tests frequently** during development (fast feedback)
- **Run integration tests** before commits (requires devnet SOL)
- **Check coverage** to ensure new code is tested
- **Format code** before committing (black + isort)
- **See `docs/TEST_EXECUTION_CHECKLIST.md`** for detailed test execution guide
- **See `tests/README.md`** for testing framework documentation

---

**Quick Navigation**:
- 📖 [Development Roadmap](DevelopmentRoadmap.md)
- ✅ [Test Execution Checklist](TEST_EXECUTION_CHECKLIST.md)
- 🧪 [Testing Guide](../tests/README.md)
- 📚 [Phase 1 Docs](phases/Phase1.md) | [Phase 2 Docs](phases/Phase2.md) | [Phase 3 Docs](phases/Phase3.md)
