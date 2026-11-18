#!/bin/zsh
# extract each pending session (skip existing outputs), 3 at a time
cd "$(dirname $0)"
cat claar_jobs.txt | xargs -P 3 -L 1 zsh -c '[ -f "$1" ] && echo "skip $1" || { /Users/wedi0306/Code/mts-spi-study-cluster/.venv/bin/python extract_claar.py "$0" "$1.tmp.npz" > "$1.log" 2>&1 && mv "$1.tmp.npz" "$1" && echo "ok $1 $(tail -1 $1.log)" || echo "FAIL $1 $(tail -3 $1.log)"; }'
echo ALLDONE
