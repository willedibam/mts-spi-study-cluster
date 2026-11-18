#!/bin/zsh
cd "$(dirname $0)"
cat late_jobs.txt | xargs -P 3 -L 1 zsh -c '[ -f "$1" ] && echo "skip $1" || { /Users/wedi0306/Code/mts-spi-study-cluster/.venv/bin/python extract_claar.py "$0" "$1.tmp.npz" late > "$1.log" 2>&1; grep -q "^done" "$1.log" && mv "$1.tmp.npz" "$1" && echo "ok $1" || echo "FAIL $1 $(tail -2 $1.log)"; }'
echo ALLDONE
