"""runs/ 의 파일별 결과를 모델 × 방식으로 묶어 셈한다 → reports/real_summary.json

    .venv/Scripts/python scripts/summarize.py

오류율은 파일별 비율의 평균이 아니라 오류 수 ÷ 정답 낱말 수를 모두 더해 낸다(긴 통화가 더 무겁게).
단어 확률 AUC 는 파일별 값을 틀린 낱말 수로 가중 평균한다.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from callscribe.runner import STRATEGIES

ROOT = Path(__file__).resolve().parents[1]


def pooled(records: list[dict], field: str = "eval") -> dict:
    def rate(part: str):
        ref = sum(r[field][part]["ref"] for r in records)
        if part == "overall":
            err = sum(r[field][part]["S"] + r[field][part]["D"] + r[field][part]["I"] for r in records)
        else:
            err = sum(r[field][part]["errors"] for r in records)
        return {"rate": round(err / ref, 4) if ref else None, "ref": ref}

    def tag(group: str):
        ref = sum(r[field]["tagged"][group]["ref"] for r in records)
        err = sum(r[field]["tagged"][group]["errors"] for r in records)
        return {"rate": round(err / ref, 4) if ref else None, "ref": ref}

    hours = sum(r["duration"] for r in records) / 3600
    o = [r[field]["overall"] for r in records]
    conf = [r[field]["confidence"] for r in records if r[field]["confidence"]["auc"] is not None]
    w = sum(c["wrong"] for c in conf)
    return {
        "files": len(records), "hours": round(hours, 2),
        "error_rate": rate("overall"), "S": sum(x["S"] for x in o), "D": sum(x["D"] for x in o),
        "I": sum(x["I"] for x in o),
        "near_cut_edges": rate("near_cut_edges"), "away_from_cut_edges": rate("away_from_cut_edges"),
        "near_seams": rate("near_seams"), "number": tag("number"), "entity": tag("entity"),
        "hallucination_runs_per_hour": round(sum(r[field]["hallucination"]["runs"] for r in records) / hours, 2),
        "confidence_auc": round(sum(c["auc"] * c["wrong"] for c in conf) / w, 4) if w else None,
        "audio_transcribed_x": round(sum(r["audio_transcribed_sec"] for r in records)
                                     / sum(r["duration"] for r in records), 3),
    }


def main() -> None:
    out = {}
    for corpus in ("earnings21", "zeroth"):
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for p in (ROOT / "runs" / corpus).glob("*/*/*.json"):
            if p.stem not in STRATEGIES:  # 실행 기록만(.words · .risk · +L 교정 결과는 뺀다)
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            groups[(p.parent.parent.name, r["strategy"])].append(r)
        table: dict[str, dict] = defaultdict(dict)
        for (model, strategy), records in sorted(groups.items()):
            table[model][strategy] = pooled(records)
            if all("eval_conventional" in r for r in records):  # 영어: 관례 정규화(군말 · 줄임말 · %)로 다시 센 것
                table[model][strategy]["conventional"] = pooled(records, "eval_conventional")
        out[corpus] = table
    (ROOT / "reports" / "real_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                        encoding="utf-8")
    for corpus, table in out.items():
        for model, rows in table.items():
            print(f"== {corpus} · {model}")
            for s, v in rows.items():
                print(f"  {s}: {v['files']} files {v['hours']}h  ER {v['error_rate']['rate']}  "
                      f"cut {v['near_cut_edges']['rate']}  away {v['away_from_cut_edges']['rate']}  "
                      f"num {v['number']['rate']}  ent {v['entity']['rate']}  "
                      f"hall/h {v['hallucination_runs_per_hour']}  auc {v['confidence_auc']}  "
                      f"x{v['audio_transcribed_x']}")
                if "conventional" in v:
                    c = v["conventional"]
                    print(f"      관례: ER {c['error_rate']['rate']}  cut {c['near_cut_edges']['rate']}  "
                          f"away {c['away_from_cut_edges']['rate']}  num {c['number']['rate']}  "
                          f"ent {c['entity']['rate']}")


if __name__ == "__main__":
    main()
