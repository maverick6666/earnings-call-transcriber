"""모델 없이: 가짜 모델(FakeASR)로 방식들의 작동 원리를 견준다. 진짜 모델 성능이 아니다.

    PYTHONPATH=src python scripts/simulate_compare.py

가정 두 개를 바꿔 가며 네 판을 돌린다 — 결과가 가정에 얼마나 기대는지 보이기 위해서다.
- pause_edge_factor: 쉼에서 자른 가장자리의 불이익(1.0 = 말 한가운데서 자른 것과 같음, 0.3 = 30% 로 줄어듦)
- informative_prob: 단어 확률이 맞고 틀림을 알려 주는가(O2 투표가 기대는 가정)
결과: reports/simulation_summary.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from callscribe import chunking, metrics, runner, simulate
from callscribe.normalize import normalize_en

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("B0", "O1", "O2", "V1", "V2")
CALLS = 8
SETTINGS = [
    {"pause_edge_factor": 1.0, "informative_prob": True},
    {"pause_edge_factor": 0.3, "informative_prob": True},
    {"pause_edge_factor": 1.0, "informative_prob": False},
    {"pause_edge_factor": 0.3, "informative_prob": False},
]


def pooled(parts: list[dict]) -> dict:
    def rate(key):
        ref = sum(p[key]["ref"] for p in parts)
        err = sum(p[key].get("errors", p[key].get("S", 0) + p[key].get("D", 0) + p[key].get("I", 0)) for p in parts)
        return round(err / ref, 4) if ref else None

    def tag_rate(group):
        ref = sum(p["tagged"][group]["ref"] for p in parts)
        return round(sum(p["tagged"][group]["errors"] for p in parts) / ref, 4) if ref else None

    hours = sum(p["_duration"] for p in parts) / 3600
    return {
        "wer": rate("overall"), "near_cut_edges": rate("near_cut_edges"),
        "away_from_cut_edges": rate("away_from_cut_edges"), "near_seams": rate("near_seams"),
        "number": tag_rate("number"), "entity": tag_rate("entity"),
        "hallucination_runs_per_hour": round(sum(p["hallucination"]["runs"] for p in parts) / hours, 2),
        "audio_transcribed_x": round(sum(p["_transcribed"] for p in parts) / sum(p["_duration"] for p in parts), 2),
        "chunks_per_hour": round(sum(p["_chunks"] for p in parts) / hours, 1),
    }


def main() -> None:
    t0 = time.time()
    calls = [simulate.make_call(100 + k, n_words=2500, lead_silence=45.0 if k % 2 == 0 else 0.0) for k in range(CALLS)]
    out = {"calls": CALLS, "hours": round(sum(c.duration for c in calls) / 3600, 2),
           "words": sum(len(c.words) for c in calls), "settings": []}
    for setting in SETTINGS:
        rows = {}
        for s in STRATEGIES:
            parts = []
            for k, tl in enumerate(calls):
                asr = simulate.FakeASR(tl, seed=100 + k, **setting)
                res = runner.run(runner.RunConfig(s), asr, tl.duration, f"sim{k}", energy=tl.energy())
                ev = metrics.evaluate(tl.words, res.words, cut_edges=chunking.cut_edges(res.chunks, tl.duration),
                                      seams=chunking.seams(res.chunks), normalize=normalize_en)
                ev["_duration"] = tl.duration
                ev["_transcribed"] = sum(c.end - c.start for c in res.chunks if not c.skip)
                ev["_chunks"] = len(res.chunks)
                parts.append(ev)
            rows[s] = pooled(parts)
        out["settings"].append({"assumptions": setting, "strategies": rows})
        print(setting, {s: rows[s]["wer"] for s in STRATEGIES}, flush=True)
    out["elapsed_sec"] = round(time.time() - t0, 1)
    (ROOT / "reports" / "simulation_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")


if __name__ == "__main__":
    main()
