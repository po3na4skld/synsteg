#!/bin/sh
# one ML job at a time, memory rules of the session: run1.sh <log> <args...>; waits for exit
log=$1; shift
until [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -ge 30 ]; do sleep 30; done
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 DET_BS=2 "$@" > "$log" 2>&1
echo "exit $? $log"
