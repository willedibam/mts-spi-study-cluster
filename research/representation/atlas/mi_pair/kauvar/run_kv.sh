#!/bin/zsh
cd "$(dirname $0)"
cat kv_jobs_primary.txt | xargs -P 3 -L 1 zsh -c '[ -f "$1" ] && echo "skip $1" || { /Users/wedi0306/Code/mts-spi-study-cluster/.venv/bin/python extract_kv.py "$0" "$1.tmp.npz" > "$1.log" 2>&1; grep -q "^done" "$1.log" && mv "$1.tmp.npz" "$1" && echo "ok $1 $(grep ^done $1.log | cut -c1-200)" || echo "FAIL $1 $(tail -2 $1.log)"; }'
echo ALLDONE
