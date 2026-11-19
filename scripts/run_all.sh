#!/usr/bin/env bash
# 실제 평가를 순서대로(그래픽카드 하나). 멈췄다 다시 돌리면 끝난 결과 · 청크는 건너뛴다.
#   bash scripts/run_all.sh > runs_all.log 2>&1
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8

step() { echo "=== $(date '+%H:%M:%S') $*"; "$PY" -B "$@" || echo "!!! 실패: $*"; }

# 0. 한국어 turbo 의 V1 · V2 를 5초 무음 기준으로 다시(이전 결과는 지움)
step scripts/eval_zeroth.py --model large-v3-turbo --strategies V1,V2
# 1. Earnings-21 Eval-10 · large-v3-turbo · 여섯 방식 전부
step scripts/eval_earnings21.py --model large-v3-turbo --subset eval10 --strategies B0,B1,O1,O2,V1,V2
# 2. +P 용어집(상한 · 잡음 섞임) — 겹쳐 자르기 위에
step scripts/eval_earnings21.py --model large-v3-turbo --subset eval10 --strategies O1,O2 --hotwords oracle --tag oracle
step scripts/eval_earnings21.py --model large-v3-turbo --subset eval10 --strategies O1,O2 --hotwords noisy --tag noisy
# 3. small(지금 wspr 모델)
step scripts/eval_earnings21.py --model small --subset eval10 --strategies B0,O1,O2,V2
# 4. large-v3
step scripts/eval_earnings21.py --model large-v3 --subset eval10 --strategies B0,B1,O1,O2,V2
# 5. 한국어 large-v3
step scripts/eval_zeroth.py --model large-v3 --strategies B0,B1,O1,O2,V1,V2
step scripts/summarize.py
echo "=== $(date '+%H:%M:%S') 끝"
