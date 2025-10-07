# Frontrun Bot - Low-Latency Solana Trading Bot

Production-ready infrastructure for sub-100ms Solana blockchain interaction with automatic failover, health monitoring, and comprehensive testing.

**Current Status**: Phase 1 Complete ✅ (Core RPC Infrastructure)

---

## Quick Start

### Prerequisites

- **Python 3.11+** (3.12.3 recommended)
- **pip** (Python package manager)
- **Solana RPC access** (Helius, QuickNode, or Solana Labs)

### Installation

#### Windows (PowerShell)

```powershell
# Clone or navigate to project
cd frontrun

# Run setup script
.\setup.bat

# Activate environment
.\venv\Scripts\activate.bat

# Configure (copy example and add your API keys)
copy config\config.yml.example config\config.yml
notepad config\config.yml
```

#### Linux/Mac/WSL

```bash
# Clone or navigate to project
cd frontrun

# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh

# Activate environment
source venv/bin/activate

# Configure (copy example and add your API keys)
cp config/config.yml.example config/config.yml
nano config/config.yml
```

#### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate.bat  # Windows

# Install dependencies
pip install -r requirements.txt

# For development (testing, linting)
pip install -r requirements-dev.txt
```

---

## Verify Installation

```bash
# Run unit tests (fast, no network required)
python run_tests.py unit

# Test RPC connectivity
python -m core.rpc_manager

# Run all tests
python run_tests.py all
```

**Expected Output**:
- ✅ 34/34 unit tests pass
- ✅ Connects to 3 RPC endpoints
- ✅ Receives real-time slot updates

---

## Project Structure

```
frontrun/
├── core/                   # Core infrastructure
│   ├── config.py           # Configuration management
│   ├── logger.py           # Structured logging
│   ├── metrics.py          # Metrics collection
│   └── rpc_manager.py      # Multi-RPC WebSocket manager
├── config/
│   └── config.yml          # Runtime configuration (add your API keys here)
├── tests/
│   ├── unit/               # Unit tests (34 tests)
│   ├── integration/        # Integration tests (8 tests)
│   └── README.md           # Testing documentation
├── docs/
│   └── phases/
│       └── Phase1.md       # Comprehensive Phase 1 docs
├── requirements.txt        # Core dependencies
├── requirements-dev.txt    # Development dependencies
├── setup.sh / setup.bat    # Setup scripts
└── run_tests.py            # Test runner
```

---

## Dependencies

### Core (Phase 1)

| Package | Version | Purpose |
|---------|---------|---------|
| `websockets` | 12.0 | WebSocket client for Solana RPC |
| `aiohttp` | 3.9.5 | Async HTTP client |
| `PyYAML` | 6.0.1 | Configuration file parsing |
| `structlog` | 24.1.0 | Structured logging |
| `psutil` | 5.9.8 | System monitoring |
| `prometheus-client` | 0.20.0 | Metrics export |

**Total size**: ~15 MB installed

**See `requirements.txt` for detailed explanation of each dependency.**

### Development

Testing, linting, formatting (see `requirements-dev.txt`):
- pytest, pytest-asyncio, pytest-cov
- black, flake8, mypy, isort
- ipython, ipdb

---

## Configuration

### 1. Copy Example Config

```bash
cp config/config.yml.example config/config.yml
```

### 2. Add Your API Keys

Edit `config/config.yml`:

```yaml
rpc:
  endpoints:
    - url: "https://devnet.helius-rpc.com/?api-key=YOUR_KEY_HERE"
      websocket_url: "wss://devnet.helius-rpc.com/?api-key=YOUR_KEY_HERE"
      priority: 0
      label: "helius_devnet"
```

**Environment Variables** (recommended for production):

```yaml
# In config.yml
url: "https://api.helius.com/?api-key=${HELIUS_API_KEY}"

# In .env file (don't commit!)
HELIUS_API_KEY=your-actual-key-here
```

**Supported patterns**:
- Full value: `"${API_KEY}"` → `"abc123"`
- Embedded: `"https://api.com/?key=${API_KEY}"` → `"https://api.com/?key=abc123"`

---

## Testing

### Quick Commands

```bash
# All tests
python run_tests.py all

# Unit tests only (fast, no network)
python run_tests.py unit

# Integration tests (requires RPC access)
python run_tests.py integration

# With coverage report
python run_tests.py coverage

# Phase 1 regression (for future phases)
python run_tests.py phase1
```

### Manual pytest

```bash
# All tests with verbose output
pytest tests/ -v

# Specific test file
pytest tests/unit/test_config.py -v

# Skip slow/integration tests
pytest tests/ -m "not slow" -v
pytest tests/ --ignore=tests/integration/ -v

# Coverage report
pytest tests/ --cov=core --cov-report=html
```

**See `tests/README.md` for comprehensive testing guide.**

---

## Usage Examples

### Basic RPC Connection Test

```python
from core.config import ConfigurationManager
from core.rpc_manager import RPCManager

# Load config
config_manager = ConfigurationManager("config/config.yml")
bot_config = config_manager.load_config()

# Create RPC manager
rpc_manager = RPCManager(bot_config.rpc_config)

# Start connections
await rpc_manager.start()

# Get health stats
health = rpc_manager.get_health_stats()
for endpoint, stats in health.items():
    print(f"{endpoint}: healthy={stats.is_healthy}")

# Subscribe to slot updates
async for notification in rpc_manager.subscribe("slotSubscribe", []):
    slot = notification["result"]["slot"]
    print(f"New slot: {slot}")

# Clean shutdown
await rpc_manager.stop()
```

### Using Structured Logging

```python
from core.logger import get_logger, setup_logging

# Setup logging
setup_logging(level="INFO", format="json")

# Get logger
logger = get_logger(__name__)

# Log with context
logger.info("trade_executed",
    mint="ABC123",
    amount_sol=0.5,
    pnl_sol=0.15
)

# Output (JSON):
# {"timestamp": "2025-10-06T...", "level": "info", "event": "trade_executed",
#  "mint": "ABC123", "amount_sol": 0.5, "pnl_sol": 0.15}
```

### Collecting Metrics

```python
from core.metrics import get_metrics, LatencyTimer

metrics = get_metrics()

# Record latency with context manager
with LatencyTimer(metrics, "rpc_call", labels={"endpoint": "helius"}):
    result = await rpc_manager.call_rpc("getSlot", [])

# Get stats
stats = metrics.get_histogram_stats("rpc_call")
print(f"p99 latency: {stats.p99:.2f}ms")

# Export all metrics
exported = metrics.export_metrics()
```

---

## What's Implemented (Phase 1)

### ✅ Core Infrastructure

- [x] Multi-RPC WebSocket connections (3+ endpoints)
- [x] Automatic failover (<1 second)
- [x] Health monitoring (every 10 seconds)
- [x] Exponential backoff reconnection
- [x] Real-time subscriptions (slots, accounts, programs)
- [x] Structured JSON logging
- [x] Metrics collection (latency, counters, gauges)
- [x] Type-safe configuration with environment variables
- [x] Comprehensive test suite (42 tests, 85% coverage)

### Performance

| Metric | Target | Actual |
|--------|--------|--------|
| RPC Connection | <500ms | 195-350ms ✅ |
| Reconnection | <1s | <1s ✅ |
| Message Processing | <10ms | ~1ms ✅ |
| Slot Update Latency | <500ms | ~400ms ✅ |

---

## What's Next (Phase 2)

**Transaction Infrastructure** (Week 3)

Will add:
- Transaction Builder (versioned transactions)
- Transaction Signer (Ed25519)
- Transaction Submitter (HTTP RPC)
- Priority Fee Calculator

**New dependencies**:
- `solders` - Solana SDK
- `base58` - Base58 encoding
- `PyNaCl` - Ed25519 signing

---

## Documentation

- **[Phase 1 Technical Docs](docs/phases/Phase1.md)** - 850 lines of comprehensive documentation
  - Every file explained
  - Every dependency explained
  - Architecture diagrams
  - Performance benchmarks
  - Lessons learned

- **[Testing Guide](tests/README.md)** - Complete testing documentation
  - How to run tests
  - How to write tests
  - Fixtures and markers
  - Troubleshooting

- **[Phase 1 Complete Summary](PHASE1_COMPLETE.md)** - Quick reference

---

## Troubleshooting

### Tests Failing

```bash
# Ensure you're in frontrun directory
cd frontrun

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate.bat  # Windows

# Verify dependencies installed
pip list | grep websockets
pip list | grep pytest

# Run with verbose output
pytest tests/unit/test_config.py -v -s
```

### RPC Connection Issues

```bash
# Check config file
cat config/config.yml

# Verify API keys set correctly
# Test with public Solana Labs endpoint first:
# url: "https://api.devnet.solana.com"
# websocket_url: "wss://api.devnet.solana.com"

# Test connection
python -m core.rpc_manager
```

### Import Errors

```bash
# Make sure you're running from frontrun/ directory
pwd  # Should end in /frontrun

# Use module syntax
python -m core.rpc_manager  # ✓ Correct
python core/rpc_manager.py  # ✗ Wrong (import errors)
```

---

## Deployment (Future - Phase 6)

**Current**: Development on local machine (Phase 1-3)

**Production** (Phase 6):
- VPS/bare-metal hosting (2 regions)
- Managed Prometheus + Grafana
- Automated deployments
- 24/7 monitoring

**Cost estimates**: See [Infrastructure.md](docs/Infrastructure.md)

---

## Contributing

When adding new features:

1. **Write tests first** (TDD)
2. **Update documentation**
3. **Run regression tests**:
   ```bash
   python run_tests.py phase1  # Ensure Phase 1 still works
   ```
4. **Check coverage**:
   ```bash
   python run_tests.py coverage  # Target: >80%
   ```

---

## License

Private project - All rights reserved

---

## Support

Questions? Check documentation:
- `docs/phases/Phase1.md` - Technical reference
- `tests/README.md` - Testing guide
- `PHASE1_COMPLETE.md` - Quick summary

---

**Built with:** Python 3.12 • WebSockets • Async/Await • Solana

**Status:** Phase 1 Complete ✅ - Ready for Phase 2
