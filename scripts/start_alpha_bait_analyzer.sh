#!/bin/bash
# Start Alpha Bait Analyzer with common configurations

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}    Alpha Bait Strategy Analyzer${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""
echo -e "${YELLOW}This script will monitor ALL Pump.fun token launches${NC}"
echo -e "${YELLOW}for several hours to collect bot swarm data.${NC}"
echo ""
echo -e "Select configuration:"
echo -e "  ${GREEN}1)${NC} Quick test (1 hour, 0.3 SOL minimum)"
echo -e "  ${GREEN}2)${NC} Standard analysis (3 hours, 0.3 SOL minimum) [RECOMMENDED]"
echo -e "  ${GREEN}3)${NC} Extended analysis (6 hours, 0.3 SOL minimum)"
echo -e "  ${GREEN}4)${NC} Large buys only (3 hours, 0.5 SOL minimum)"
echo -e "  ${GREEN}5)${NC} Custom configuration"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        HOURS=1
        MIN_BUY=0.3
        echo -e "\n${GREEN}Running: Quick test (1 hour)${NC}"
        ;;
    2)
        HOURS=3
        MIN_BUY=0.3
        echo -e "\n${GREEN}Running: Standard analysis (3 hours)${NC}"
        ;;
    3)
        HOURS=6
        MIN_BUY=0.3
        echo -e "\n${GREEN}Running: Extended analysis (6 hours)${NC}"
        ;;
    4)
        HOURS=3
        MIN_BUY=0.5
        echo -e "\n${GREEN}Running: Large buys only (3 hours, 0.5 SOL)${NC}"
        ;;
    5)
        read -p "Enter hours to run: " HOURS
        read -p "Enter minimum SOL for first buy: " MIN_BUY
        echo -e "\n${GREEN}Running: Custom (${HOURS} hours, ${MIN_BUY} SOL)${NC}"
        ;;
    *)
        echo -e "\n${YELLOW}Invalid choice, using standard (3 hours)${NC}"
        HOURS=3
        MIN_BUY=0.3
        ;;
esac

# Navigate to project root
cd "$(dirname "$0")/.." || exit

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "\n${YELLOW}Warning: Virtual environment not found${NC}"
    echo -e "Run this first: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo -e "\n${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if config exists
if [ ! -f "config/config.yml" ]; then
    echo -e "\n${YELLOW}Error: config/config.yml not found${NC}"
    echo -e "Copy config/config.yml.example and fill in your Helius credentials"
    exit 1
fi

# Create output directory
mkdir -p data/alpha_bait_analysis

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}Starting analyzer...${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""
echo -e "Duration: ${HOURS} hours"
echo -e "Minimum first buy: ${MIN_BUY} SOL"
echo -e "Output: data/alpha_bait_analysis/"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop early${NC}"
echo ""

# Run analyzer
python scripts/alpha_bait_analyzer.py \
    --hours "$HOURS" \
    --min-buy "$MIN_BUY" \
    --output data/alpha_bait_analysis

echo -e "\n${GREEN}=================================================${NC}"
echo -e "${GREEN}Analysis complete!${NC}"
echo -e "${GREEN}=================================================${NC}"
echo ""
echo -e "Results saved to: ${BLUE}data/alpha_bait_analysis/${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. Review the JSON output file"
echo -e "  2. Analyze patterns in bot swarms"
echo -e "  3. See ${BLUE}ALPHA_BAIT_STRATEGY.md${NC} for analysis guide"
echo ""
