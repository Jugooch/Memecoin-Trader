@echo off
REM Start Alpha Bait Analyzer (Windows version)

echo =================================================
echo     Alpha Bait Strategy Analyzer
echo =================================================
echo.
echo This script will monitor ALL Pump.fun token launches
echo for several hours to collect bot swarm data.
echo.
echo Select configuration:
echo   1) Quick test (1 hour, 0.3 SOL minimum)
echo   2) Standard analysis (3 hours, 0.3 SOL minimum) [RECOMMENDED]
echo   3) Extended analysis (6 hours, 0.3 SOL minimum)
echo   4) Large buys only (3 hours, 0.5 SOL minimum)
echo   5) Custom configuration
echo.

set /p choice="Enter choice [1-5]: "

if "%choice%"=="1" (
    set HOURS=1
    set MIN_BUY=0.3
    echo.
    echo Running: Quick test (1 hour)
) else if "%choice%"=="2" (
    set HOURS=3
    set MIN_BUY=0.3
    echo.
    echo Running: Standard analysis (3 hours)
) else if "%choice%"=="3" (
    set HOURS=6
    set MIN_BUY=0.3
    echo.
    echo Running: Extended analysis (6 hours)
) else if "%choice%"=="4" (
    set HOURS=3
    set MIN_BUY=0.5
    echo.
    echo Running: Large buys only (3 hours, 0.5 SOL)
) else if "%choice%"=="5" (
    set /p HOURS="Enter hours to run: "
    set /p MIN_BUY="Enter minimum SOL for first buy: "
    echo.
    echo Running: Custom (%HOURS% hours, %MIN_BUY% SOL)
) else (
    echo.
    echo Invalid choice, using standard (3 hours)
    set HOURS=3
    set MIN_BUY=0.3
)

REM Navigate to project root
cd /d "%~dp0\.."

REM Check if virtual environment exists
if not exist "venv" (
    echo.
    echo Warning: Virtual environment not found
    echo Run this first: python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if config exists
if not exist "config\config.yml" (
    echo.
    echo Error: config\config.yml not found
    echo Copy config\config.yml.example and fill in your Helius credentials
    pause
    exit /b 1
)

REM Create output directory
if not exist "data\alpha_bait_analysis" mkdir data\alpha_bait_analysis

echo =================================================
echo Starting analyzer...
echo =================================================
echo.
echo Duration: %HOURS% hours
echo Minimum first buy: %MIN_BUY% SOL
echo Output: data\alpha_bait_analysis\
echo.
echo Press Ctrl+C to stop early
echo.

REM Run analyzer
python scripts\alpha_bait_analyzer.py --hours %HOURS% --min-buy %MIN_BUY% --output data\alpha_bait_analysis

echo.
echo =================================================
echo Analysis complete!
echo =================================================
echo.
echo Results saved to: data\alpha_bait_analysis\
echo.
echo Next steps:
echo   1. Review the JSON output file
echo   2. Analyze patterns in bot swarms
echo   3. See ALPHA_BAIT_STRATEGY.md for analysis guide
echo.
pause
