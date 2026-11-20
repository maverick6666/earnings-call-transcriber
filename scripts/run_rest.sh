#!/usr/bin/env bash
# run_all.sh 뒤 남은 것: large-v3 이어 하기 · 무음 2초 기준 V1 · V2 · 관례 채점 · 요약
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
step() { echo "=== $(date '+%H:%M:%S') $*"; "$PY" -B "$@" || echo "!!! 실패: $*"; }

step scripts/eval_earnings21.py --model large-v3 --subset eval10 --strategies B0,B1,O1,O2,V2
step scripts/eval_zeroth.py --model large-v3-turbo --strategies V1,V2 --min-silence 2 --tag sil2
step scripts/eval_earnings21.py --model large-v3-turbo --subset eval10 --strategies V1,V2 --min-silence 2 --tag sil2
step scripts/rescore.py
step scripts/summarize.py
echo "=== $(date '+%H:%M:%S') 끝"
