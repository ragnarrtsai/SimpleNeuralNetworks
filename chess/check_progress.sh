#!/bin/bash
# Read-only AlphaZero training monitor. Prints process status + latest progress.
if pgrep -f "alphazero.train" > /dev/null; then echo "RUNNING"; else echo "ENDED"; fi
echo "---"
grep -E "Iteration|Promoting" training.log | tail -4
echo "---DONE-CHECK---"
grep -c "Training complete" training.log || true
exit 0
