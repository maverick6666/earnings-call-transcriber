"""reports/correction_*.json → 검수 층 보고서에 넣는 표(마크다운).

    python scripts/review_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = (("-noshape-nocommon", "기본(용어집 · 유사도 · 대소문자)"), ("-nocommon", "+ 규칙 6 모양만 바꾸는 편집"),
         ("-common3", "+ 규칙 7 흔한 낱말은 위험 근거"))


def main() -> None:
    rows = []
    for path in sorted((ROOT / "reports").glob("correction_*.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        rule = next((name for suffix, name in RULES if s["tag"].endswith(suffix)), s["tag"])
        glossary = "oracle" if "-oracle" in s["tag"] else "noisy"
        rows.append((s["hyp"], s["subset"], glossary, list(dict(RULES).values()).index(rule)
                     if rule in dict(RULES).values() else 9, rule, s))
    rows.sort(key=lambda r: (r[0], r[1] != "eval10", r[2], r[3]))
    print("| 받아쓰기 | 통화 | 용어집 | 규칙 | 바꾼 자리 | WER | 고유명사 오류 | 숫자 오류 | 고침 · 망침 |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|")
    for hyp, subset, glossary, _, rule, s in rows:
        o = s["outcome"]
        print(f"| {hyp} | {subset} ({len(s['calls'])}) | {glossary} | {rule} | {o.get('applied', 0):,} | "
              f"{s['wer'][0]:.2%} → {s['wer'][1]:.2%} | {s['entity_error'][0]:.2%} → {s['entity_error'][1]:.2%} | "
              f"{s['number_error'][0]:.2%} → {s['number_error'][1]:.2%} | {o.get('fixed', 0)} · {o.get('broke', 0)} |")


if __name__ == "__main__":
    main()
