"""reports/real_summary.json → 보고서 · README 에 넣을 마크다운 표(손으로 옮기다 틀리지 않게).

    .venv/Scripts/python scripts/make_tables.py > reports/real_tables.md
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"B0": "B0 60초 고정", "B1": "B1 모델 자체 긴 오디오 처리", "O1": "**O1 겹쳐 자르기**",
         "O2": "O2 겹쳐 자르기 + 투표", "V1": "V1 무음 건너뛰기", "V2": "V2 무음 건너뛰기 + 1초 겹침"}


def pct(x) -> str:
    return "–" if x is None else f"{x * 100:.1f}%"


def table(rows: dict, conventional: bool) -> list[str]:
    head = ("| 방식 | 오류율 | 자른 곳 ±2초 안 | 그 밖 | 고유명사 | 숫자 | 지어낸 문장(시간당) "
            "| 단어 확률 AUC | 계산(배) |")
    out = [head, "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in ("B0", "B1", "O1", "O2", "V1", "V2"):
        if s not in rows:
            continue
        v = rows[s]
        c = v.get("conventional") if conventional else v
        if c is None:
            continue
        out.append(f"| {NAMES[s]} | {pct(c['error_rate']['rate'])} | {pct(c['near_cut_edges']['rate'])} | "
                   f"{pct(c['away_from_cut_edges']['rate'])} | {pct(c['entity']['rate'])} | "
                   f"{pct(c['number']['rate'])} | "
                   f"{c['hallucination_runs_per_hour']} | {c['confidence_auc'] or '–'} | {c['audio_transcribed_x']} |")
    return out


def main() -> None:
    s = json.loads((ROOT / "reports" / "real_summary.json").read_text(encoding="utf-8"))
    corpora = (("earnings21", "Earnings-21 Eval-10 (영어, WER)", "군말 · 줄임말 그대로"),
               ("zeroth", "Zeroth-Korean (한국어, CER)", "숫자 표기 그대로"))
    print("# 전체 결과 표\n")
    print("`reports/real_summary.json` 에서 `scripts/make_tables.py` 로 만든 표. "
          "설명은 [real_results.md](real_results.md).")
    print("오류율 · 자른 곳 · 그 밖 · 고유명사 · 숫자는 정답 낱말(한국어는 글자) 대비, 지어낸 문장은 녹음 시간당,")
    print("계산은 받아 적은 소리 ÷ 녹음 길이. 모델 이름 뒤 -oracle · -noisy 는 용어집, -sil2 는 무음 기준 2초.")
    for corpus, label, raw_note in corpora:
        for model, rows in s.get(corpus, {}).items():
            conv = all("conventional" in v for v in rows.values())
            files = max(v["files"] for v in rows.values())
            hours = max(v["hours"] for v in rows.values())
            note = ""
            if len({v["files"] for v in rows.values()}) > 1:  # 아직 다 돌지 않은 방식이 있으면 방식별 건수를 밝힌다
                note = " · 진행 중(" + ", ".join(f"{k} {v['files']}건" for k, v in rows.items()) + ")"
            print(f"\n### {label} · {model} · {files}건 {hours}시간" + (" · 관례 정규화" if conv else "") + note + "\n")
            print("\n".join(table(rows, conv)))
            if conv:
                print(f"\n<details><summary>원래 정규화({raw_note})</summary>\n")
                print("\n".join(table(rows, False)))
                print("\n</details>")


if __name__ == "__main__":
    main()
