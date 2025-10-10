@echo off
REM Start Auto-Update Watchlist Scheduler (Windows)
REM
REM This script runs the auto-updater in the background
REM Press Ctrl+C in the terminal to stop it

echo ====================================
echo   Dev Watchlist Auto-Updater
echo ====================================
echo.
echo This will update your dev watchlist every 8 hours
echo Press Ctrl+C to stop at any time
echo.

REM Activate virtual environment if it exists
if exist "..\bot_env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call ..\bot_env\Scripts\activate.bat
)

REM Create logs directory if it doesn't exist
if not exist "..\logs" mkdir "..\logs"

REM Run the auto-updater
echo Starting auto-updater...
echo.
python auto_update_watchlist.py --interval 8 --max-pages 500 --min-buy-rate 70

pause
