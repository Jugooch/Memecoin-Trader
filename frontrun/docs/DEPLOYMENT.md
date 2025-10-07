# Frontrun Bot - Deployment Checklist

This document explains how to deploy the frontrun bot on a **new system** (VPS, server, or another developer's machine).

---

## Prerequisites

- ✅ Python 3.11+ installed
- ✅ Git installed
- ✅ Internet access
- ✅ RPC API keys (Helius, QuickNode, etc.)

---

## Quick Deployment (5 Minutes)

### 1. Clone/Copy Repository

```bash
# If using git
git clone <repo-url>
cd frontrun

# Or copy files manually
# Make sure you have all files from frontrun/ directory
```

### 2. Run Setup Script

**Windows:**
```powershell
.\setup.bat
```

**Linux/Mac/WSL:**
```bash
chmod +x setup.sh
./setup.sh
```

The script will:
- ✅ Check Python version (3.11+ required)
- ✅ Create virtual environment
- ✅ Install all dependencies from `requirements.txt`
- ✅ Optionally install dev dependencies
- ✅ Create logs/ and data/ directories
- ✅ Run tests to verify installation

### 3. Configure RPC Endpoints

```bash
# Copy example config
cp config/config.yml.example config/config.yml

# Edit with your API keys
nano config/config.yml  # Linux/Mac
notepad config/config.yml  # Windows
```

**Add your API keys:**
```yaml
rpc:
  endpoints:
    - url: "https://devnet.helius-rpc.com/?api-key=YOUR_KEY_HERE"
      websocket_url: "wss://devnet.helius-rpc.com/?api-key=YOUR_KEY_HERE"
      priority: 0
      label: "helius_devnet"
```

**OR use environment variables (recommended for production):**

```bash
# Create .env file (don't commit!)
echo "HELIUS_API_KEY=your-key-here" > .env
echo "QUICKNODE_API_KEY=your-key-here" >> .env

# In config.yml
url: "https://api.helius.com/?api-key=${HELIUS_API_KEY}"
```

### 4. Verify Installation

```bash
# Activate environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate.bat  # Windows

# Run tests
python run_tests.py unit

# Test RPC connection
python -m core.rpc_manager
```

**Expected output:**
- ✅ All 34 unit tests pass
- ✅ Connects to RPC endpoints
- ✅ Receives real-time slot updates

---

## Manual Deployment (Step-by-Step)

If the setup script doesn't work, follow these steps:

### 1. Create Virtual Environment

```bash
cd frontrun
python3 -m venv venv
```

### 2. Activate Environment

```bash
# Linux/Mac/WSL
source venv/bin/activate

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows Command Prompt
venv\Scripts\activate.bat
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install Dependencies

```bash
# Core dependencies (required)
pip install -r requirements.txt

# Development dependencies (optional, for testing/development)
pip install -r requirements-dev.txt
```

**Expected installation:**
```
✓ websockets==12.0
✓ aiohttp==3.9.5
✓ PyYAML==6.0.1
✓ structlog==24.1.0
✓ psutil==5.9.8
✓ prometheus-client==0.20.0
```

### 5. Create Directories

```bash
mkdir -p logs data
```

### 6. Configure

```bash
cp config/config.yml.example config/config.yml
# Edit config.yml with your API keys
```

### 7. Test

```bash
python run_tests.py unit
```

---

## Requirements Files Explained

### `requirements.txt` - Core Dependencies

**What it contains:**
- WebSocket client (`websockets`)
- HTTP client (`aiohttp`)
- Configuration (`PyYAML`)
- Logging (`structlog`)
- Metrics (`prometheus-client`, `psutil`)

**When to use:**
- Production deployments
- Minimal installations
- CI/CD pipelines

**Install with:**
```bash
pip install -r requirements.txt
```

### `requirements-dev.txt` - Development Dependencies

**What it contains:**
- Testing framework (`pytest`, `pytest-asyncio`, `pytest-cov`)
- Code quality (`black`, `flake8`, `mypy`, `isort`)
- Development tools (`ipython`, `ipdb`)

**When to use:**
- Local development
- Running tests
- Code contributions

**Install with:**
```bash
pip install -r requirements-dev.txt
```

**Note:** This includes `requirements.txt` automatically (via `-r requirements.txt`)

---

## Dependency Management

### Adding New Dependencies

When Phase 2 adds new dependencies:

```bash
# 1. Install new package
pip install solders==0.21.0

# 2. Update requirements.txt
echo "solders==0.21.0  # Solana SDK for transaction building" >> requirements.txt

# 3. Test on clean environment
deactivate
rm -rf venv
./setup.sh  # Verify everything installs correctly
```

### Freezing Exact Versions

To capture **exact** versions of all dependencies:

```bash
# Generate full freeze (includes sub-dependencies)
pip freeze > requirements-frozen.txt

# Use for reproducible deployments
pip install -r requirements-frozen.txt
```

**When to use frozen requirements:**
- Production deployments (exact reproducibility)
- Debugging version conflicts
- Long-term archival

**Current approach (requirements.txt):**
- Specifies only direct dependencies
- Uses `==` for exact versions
- More maintainable
- Good for Phase 1

---

## Verification Checklist

After deployment, verify everything works:

### Core Functionality

```bash
# 1. Python version
python --version
# Expected: Python 3.11+ or 3.12+

# 2. Dependencies installed
pip list | grep websockets
pip list | grep structlog
# Expected: All packages listed

# 3. Unit tests pass
python run_tests.py unit
# Expected: 34/34 tests pass

# 4. RPC connectivity
python -m core.rpc_manager
# Expected: Connects to endpoints, receives slot updates
```

### File Structure

Ensure these exist:

```bash
frontrun/
├── venv/                 # Virtual environment
├── core/                 # 4 Python files
├── config/
│   └── config.yml        # Your config with API keys
├── logs/                 # Log directory
├── data/                 # Data directory
├── requirements.txt      # Dependencies
└── README.md
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'websockets'"

**Cause**: Virtual environment not activated or dependencies not installed

**Fix:**
```bash
source venv/bin/activate  # Activate environment
pip install -r requirements.txt  # Install dependencies
```

### "pytest: command not found"

**Cause**: Development dependencies not installed

**Fix:**
```bash
pip install -r requirements-dev.txt

# Or run as module
python -m pytest tests/unit/ -v
```

### Tests Fail with "Connection refused"

**Cause**: RPC endpoints down or incorrect API keys

**Fix:**
```bash
# 1. Check config
cat config/config.yml

# 2. Test with public endpoint first
# In config.yml, use:
url: "https://api.devnet.solana.com"
websocket_url: "wss://api.devnet.solana.com"

# 3. Skip integration tests
python run_tests.py unit  # Unit tests don't need network
```

### "Python version too old"

**Cause**: Python <3.11

**Fix:**
```bash
# Check version
python --version

# Install Python 3.12 (recommended)
# Ubuntu:
sudo apt install python3.12 python3.12-venv

# macOS (Homebrew):
brew install python@3.12

# Windows: Download from python.org
```

---

## Production Deployment (Phase 6)

**Current**: Phase 1 is for development only

**Future** (Phase 6 will add):
- Docker containerization
- Systemd service files
- Nginx reverse proxy
- PM2/supervisor process management
- Automated deployments
- Health check endpoints
- Prometheus metrics export

**For now**: Run manually in screen/tmux session

```bash
# Example (not recommended for production yet)
screen -S frontrun
source venv/bin/activate
python -m core.rpc_manager
# Ctrl+A, D to detach
```

---

## CI/CD Integration (Future)

**GitHub Actions Example** (add when ready):

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pip install -r requirements-dev.txt
      - run: python run_tests.py unit
```

---

## Summary

**To deploy on a new system:**

1. ✅ Clone repository
2. ✅ Run `./setup.sh` (Linux/Mac) or `.\setup.bat` (Windows)
3. ✅ Configure `config/config.yml` with API keys
4. ✅ Test with `python run_tests.py unit`
5. ✅ Verify with `python -m core.rpc_manager`

**Total time**: 5-10 minutes

**Dependencies are tracked in:**
- `requirements.txt` - Core dependencies (production)
- `requirements-dev.txt` - Development dependencies (testing)
- Both files have exact versions (`package==version`)
- All dependencies documented with purpose and usage

**Everything is version-controlled and reproducible!** 🎯
