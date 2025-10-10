# Auto-Update Watchlist Guide

This guide shows you how to automatically grow your dev watchlist by running updates every 8 hours.

## Quick Start

### Windows

```bash
cd frontrun\scripts
start_auto_update.bat
```

### Linux/Mac

```bash
cd frontrun/scripts
chmod +x start_auto_update.sh
./start_auto_update.sh
```

### Manual Python

```bash
cd frontrun
python scripts/auto_update_watchlist.py
```

## What It Does

The auto-updater:
1. ✅ Runs immediately on start
2. ✅ Waits 8 hours
3. ✅ Runs another update
4. ✅ Repeats forever (until stopped)

Each update:
- Scans 500 pages (~12 hours of data)
- Finds new devs with 70%+ buy rate
- Merges with existing watchlist
- Shows before/after stats

## Example Output

```
=== AUTO-UPDATE SCHEDULER STARTED ===
Update interval: Every 8.0 hours
Pages per update: 500
Lookback window: 7 days
Min buy rate: 70.0%

initial_watchlist_stats devs=1945 avg_buy_rate=93.4

Running first update immediately...
=== STARTING WATCHLIST UPDATE ===
watchlist_before devs=1945 avg_buy_rate=93.4

✅ UPDATE SUCCESSFUL
watchlist_after devs=2145 avg_buy_rate=92.8 devs_added=200

waiting_for_next_update next_update=2025-10-10 21:30:00 time_remaining=7h 55m
```

## Configuration Options

### Change Update Interval

```bash
# Every 6 hours (faster growth)
python scripts/auto_update_watchlist.py --interval 6

# Every 12 hours (slower growth)
python scripts/auto_update_watchlist.py --interval 12

# Every 24 hours (maintenance mode)
python scripts/auto_update_watchlist.py --interval 24
```

### Change Pages Per Update

```bash
# Light updates (300 pages = ~6 hours of data)
python scripts/auto_update_watchlist.py --max-pages 300

# Heavy updates (1000 pages = ~2 days of data)
python scripts/auto_update_watchlist.py --max-pages 1000
```

### Change Quality Filters

```bash
# Stricter (fewer, higher quality devs)
python scripts/auto_update_watchlist.py --min-buy-rate 80

# Looser (more devs, lower quality)
python scripts/auto_update_watchlist.py --min-buy-rate 60
```

## Running in Background

### Windows

**Option 1: Separate Terminal**
1. Open a new PowerShell/CMD window
2. Run `start_auto_update.bat`
3. Minimize the window
4. Leave it running

**Option 2: Windows Task Scheduler**
1. Open Task Scheduler
2. Create Basic Task
3. Trigger: At startup
4. Action: Start a program
5. Program: `C:\path\to\python.exe`
6. Arguments: `scripts/auto_update_watchlist.py`
7. Start in: `C:\path\to\Memecoin-Trader\frontrun`

### Linux/Mac

**Option 1: nohup (Simple)**
```bash
cd frontrun/scripts
nohup ./start_auto_update.sh > ../logs/auto_update.log 2>&1 &

# View logs
tail -f ../logs/auto_update.log

# Stop it later
ps aux | grep auto_update_watchlist.py
kill <PID>
```

**Option 2: tmux/screen (Better)**
```bash
# Install tmux (if needed)
sudo apt install tmux  # Ubuntu/Debian
brew install tmux      # Mac

# Start tmux session
tmux new -s watchlist

# Run the updater
cd frontrun/scripts
./start_auto_update.sh

# Detach: Press Ctrl+B, then D

# Reattach later
tmux attach -t watchlist
```

**Option 3: systemd Service (Best)**

Create `/etc/systemd/system/watchlist-updater.service`:
```ini
[Unit]
Description=Dev Watchlist Auto-Updater
After=network.target

[Service]
Type=simple
User=yourusername
WorkingDirectory=/path/to/Memecoin-Trader/frontrun
Environment="PATH=/path/to/bot_env/bin"
ExecStart=/path/to/bot_env/bin/python scripts/auto_update_watchlist.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable watchlist-updater
sudo systemctl start watchlist-updater

# Check status
sudo systemctl status watchlist-updater

# View logs
sudo journalctl -u watchlist-updater -f
```

## Monitoring

### Check Current Stats

```bash
cd frontrun
python -c "
from core.dev_watchlist_loader import DevWatchlistLoader
loader = DevWatchlistLoader('data/dev_watchlist.json')
stats = loader.get_stats()
print(f'Total Devs: {stats[\"total_devs\"]:,}')
print(f'Avg Buy Rate: {stats[\"avg_buy_rate\"]:.1f}%')
"
```

### View Logs

```bash
# If running in terminal, logs show live

# If using nohup:
tail -f frontrun/logs/auto_update.log

# If using systemd:
sudo journalctl -u watchlist-updater -f
```

## Growth Estimates

Based on typical Pump.fun activity:

| Interval | Pages/Update | Est. Growth/Day | Time to 10k Devs |
|----------|--------------|-----------------|------------------|
| 6 hours | 500 | ~600-800 | ~14 days |
| 8 hours | 500 | ~450-600 | ~18 days |
| 12 hours | 500 | ~300-400 | ~28 days |
| 24 hours | 300 | ~150-200 | ~56 days |

**Note:** Growth slows over time as you find duplicate devs.

## Recommended Strategy

### Phase 1: Rapid Growth (Weeks 1-2)
```bash
# Update every 6 hours with 500 pages
python scripts/auto_update_watchlist.py --interval 6 --max-pages 500
```

Target: 5,000+ devs

### Phase 2: Steady Growth (Weeks 3-4)
```bash
# Update every 8 hours with 500 pages
python scripts/auto_update_watchlist.py --interval 8 --max-pages 500
```

Target: 10,000+ devs

### Phase 3: Maintenance (Month 2+)
```bash
# Update every 24 hours with 300 pages
python scripts/auto_update_watchlist.py --interval 24 --max-pages 300
```

Target: Keep catching new devs as they appear

## Troubleshooting

### "Bitquery API credits exhausted"

- Your API is hitting rate limits
- Solution: Increase interval or decrease pages
- Example: `--interval 12 --max-pages 300`

### "Script stopped unexpectedly"

- Check logs for errors
- Common causes: Network issues, Bitquery API issues
- Script will auto-retry after 5 minutes

### "File not found error"

- Make sure you're in the `frontrun` directory
- Check that `scripts/build_dev_watchlist.py` exists

### "Too slow / Not finding new devs"

- You may have already found most active devs
- Lower `--min-buy-rate` to catch more devs
- Example: `--min-buy-rate 60`

## Performance Tips

1. **Start aggressive, slow down later**
   - Week 1: Every 6 hours
   - Week 2: Every 8 hours
   - Week 3+: Every 12-24 hours

2. **Monitor API usage**
   - 500 pages = 500 API calls
   - Every 8 hours = 1,500 calls/day
   - Make sure your Bitquery plan supports this

3. **Quality over quantity**
   - 5,000 high-quality devs (90%+) > 20,000 low-quality devs (60%+)
   - Use `--min-buy-rate 80` for better results

4. **Check for duplicates**
   - The script automatically deduplicates
   - Growth will slow as your database gets larger

## Stopping the Scheduler

### If running in foreground
- Press `Ctrl+C`

### If running with nohup
```bash
ps aux | grep auto_update_watchlist.py
kill <PID>
```

### If running with systemd
```bash
sudo systemctl stop watchlist-updater
```

### If running with tmux
```bash
tmux attach -t watchlist
# Then press Ctrl+C
```

## Next Steps

1. Start the auto-updater
2. Let it run for 1-2 weeks
3. Monitor your watchlist growth
4. Once you have 5,000+ devs, you're ready for production trading!

Your watchlist will grow automatically while you focus on other things! 🚀
