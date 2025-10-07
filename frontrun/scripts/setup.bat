@echo off
REM Frontrun Bot - Setup Script (Windows)
REM Sets up development environment on Windows

echo ========================================
echo   Frontrun Bot - Environment Setup
echo ========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    exit /b 1
)
echo OK: Python found
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo OK: Virtual environment created
) else (
    echo OK: Virtual environment exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo OK: Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo OK: pip upgraded
echo.

REM Install core dependencies
echo Installing core dependencies...
pip install -r requirements.txt
echo OK: Core dependencies installed
echo.

REM Ask about development dependencies
set /p INSTALL_DEV="Install development dependencies (pytest, black, mypy)? [y/N] "
if /i "%INSTALL_DEV%"=="y" (
    pip install -r requirements-dev.txt
    echo OK: Development dependencies installed
    echo.
)

REM Create necessary directories
echo Creating directory structure...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
echo OK: Directories created
echo.

REM Check config file
if exist "config\config.yml" (
    echo OK: Configuration file exists
) else (
    echo WARNING: config\config.yml not found
    echo   Please copy config\config.yml.example and fill in your API keys
)
echo.

REM Ask to run tests
set /p RUN_TESTS="Run tests to verify setup? [y/N] "
if /i "%RUN_TESTS%"=="y" (
    echo.
    echo Running unit tests...
    python run_tests.py unit
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Activate environment: venv\Scripts\activate.bat
echo   2. Configure: copy config\config.yml.example config\config.yml
echo   3. Edit config.yml with your API keys
echo   4. Test RPC: python -m core.rpc_manager
echo   5. Run tests: python run_tests.py unit
echo.
pause
