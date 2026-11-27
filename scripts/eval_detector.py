"""검수 0층 평가: Earnings-21 O1 결과로 '틀렸을 위험' 점수를 학습 · 시험한다(통화 하나씩 빼고).

    python scripts/eval_detector.py --model large-v3-turbo
    python scripts/eval_detector.py --model large-v3-turbo --alt small          # 다른 모델 O1 을 두 번째 의견으로
    python scripts/eval_detector.py --model large-v3-turbo --glossary oracle    # 용어집 특징까지

필요한 것: runs/earnings21/<모델>/<파일>/O1.json · O1.words.json (eval_earnings21.py 결과)와 cache/ 의 창별 받아쓰기
(두 번째 창 일치 — 저장본이 없으면 그 특징만 비고 경고한다). 모델 · 그래픽카드는 필요 없다.

결과
- reports/detector_<모델>.json · .md : 특징 묶음별 AUC · 낱말 예산 · 듣는 시간 예산별로 잡는 오류 비율 · 가중치
- runs/earnings21/<모델>/<파일>/O1.risk.json : 낱말마다 위험 점수(그 통화를 빼고 학습)
  — eval_correction.py --risky-top 이 쓴다
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_earnings21  # noqa: E402
from eval_earnings21 import call_terms, file_ids  # noqa: E402
from rescore import chunks_of  # noqa: E402

from callscribe.chunking import Chunk  # noqa: E402
from callscribe.datasets import read_nlp  # noqa: E402
from callscribe.detect import (  # noqa: E402
    FEATURES,
    CallData,
    RiskModel,
    alt_support,
    features,
    leave_one_call_out,
    second_window_support,
    summarize,
    word_labels,
)
from callscribe.metrics import ENTITY_TAGS, NUMBER_TAGS  # noqa: E402
from callscribe.normalize import normalize_en_conventional  # noqa: E402
from callscribe.words import Word  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

FEATURE_SETS = {
    "확률만": ["risk_prob"],
    "확률 + 앞뒤 확률": ["risk_prob", "risk_prob_nb"],
    "두 번째 창만": ["disagree", "no_second"],
    "확률 + 두 번째 창": ["risk_prob", "risk_prob_nb", "disagree", "no_second"],
    "어휘 · 시간만": ["has_digit", "capital_mid", "all_caps", "n_chars", "log_dur_per_char", "repeat"],
}
BASE_ALL = [f for f in FEATURES if f not in ("alt_disagree", "near_glossary", "in_glossary", "has_latin")]


def load_words(path: Path) -> list[Word]:
    return [Word(**w) for w in json.loads(path.read_text(encoding="utf-8"))]


def chunk_results(rec: dict, cache: Path) -> list[tuple[Chunk, list[Word]]] | None:
    """기록의 청크 계획(초기 기록에 없으면 같은 설정으로 다시 만든다)대로 창별 저장본을 읽는다. 하나라도 없으면 None."""
    store = cache / rec["manifest"]["cache_key"]
    out = []
    for c in chunks_of(rec):
        if c.skip:
            out.append((c, []))
            continue
        path = store / f"{c.start:.3f}-{c.end:.3f}.json"
        if not path.exists():
            return None
        out.append((c, load_words(path)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--strategy", default="O1")
    ap.add_argument("--subset", default="eval10")
    ap.add_argument("--alt", default=None, help="두 번째 의견으로 쓸 다른 모델 결과 폴더 이름(예: small, large-v3)")
    ap.add_argument("--glossary", choices=("oracle", "noisy"), default=None)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "earnings21")
    ap.add_argument("--l2", type=float, default=1.0)
    args = ap.parse_args()
    eval_earnings21.DATA = args.data  # file_ids 가 이 폴더의 eval10 목록을 읽게

    runs = ROOT / "runs" / "earnings21"
    calls: list[CallData] = []
    missing_cache = []
    for fid in file_ids(args.subset):
        rec_path = runs / args.model / fid / f"{args.strategy}.json"
        if not rec_path.exists():
            print(f"{fid}: {rec_path} 없음 — 건너뜀", flush=True)
            continue
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        words = load_words(rec_path.with_name(f"{args.strategy}.words.json"))
        ref = read_nlp(args.data / "transcripts" / "nlp_references" / f"{fid}.nlp",
                       args.data / "transcripts" / "wer_tags" / f"{fid}.wer_tag.json")
        results = chunk_results(rec, ROOT / "cache")
        support = second_window_support(words, results) if results else None
        if results is None:
            missing_cache.append(fid)
        alt = None
        if args.alt:
            alt_path = runs / args.alt / fid / f"{args.strategy}.words.json"
            alt = alt_support(words, load_words(alt_path)) if alt_path.exists() else None
        terms = call_terms(ref, args.glossary, seed=int(fid), data=args.data) if args.glossary else ()
        x = features(words, support=support, alt=alt, glossary=terms)
        lab = word_labels(ref, words, normalize_en_conventional, ENTITY_TAGS | NUMBER_TAGS)
        calls.append(CallData(fid, words, x, lab, rec["duration"]))
        print(f"{fid}: 낱말 {len(words)} · 틀림 {int((lab.wrong == 1).sum())} · 빠짐 {lab.deletions}", flush=True)
    if len(calls) < 2:
        sys.exit("통화가 둘 이상 있어야 한다(eval_earnings21.py 를 먼저)")
    if missing_cache:
        print(f"경고: 창별 저장본이 없는 통화 {missing_cache} — 두 번째 창 특징이 비어 있다", flush=True)

    sets = dict(FEATURE_SETS)
    full = list(BASE_ALL)
    if args.alt:
        sets["확률 + 다른 모델"] = ["risk_prob", "risk_prob_nb", "alt_disagree"]
        full.append("alt_disagree")
    if args.glossary:
        full += ["near_glossary", "in_glossary"]
    if full != BASE_ALL:
        sets["전부(다른 모델 · 용어집 빼고)"] = BASE_ALL
    sets["전부"] = full

    results = {}
    best_scores = None
    for name, cols in sets.items():
        scores = leave_one_call_out(calls, cols, l2=args.l2)
        results[name] = {"features": cols, **summarize(calls, scores)}
        if name == "전부":
            best_scores = scores
        r = results[name]
        print(f"{name:24s} AUC {r['auc']}  시간 10/20/30% → {r['catch_at_time_budget']}", flush=True)

    model = RiskModel.fit(np.vstack([c.x[c.mask][:, [FEATURES.index(f) for f in full]] for c in calls]),
                          np.concatenate([c.labels.wrong[c.mask] for c in calls]).astype(float), full, l2=args.l2)
    for c in calls:
        out = runs / args.model / c.call_id / f"{args.strategy}.risk.json"
        out.write_text(json.dumps({"features": full, "note": "그 통화를 빼고 학습한 모델의 점수",
                                   "risk": [round(float(v), 5) for v in best_scores[c.call_id]]}), encoding="utf-8")

    tag = args.model + (f"-alt-{args.alt}" if args.alt else "") + (f"-{args.glossary}" if args.glossary else "")
    summary = {"model": args.model, "strategy": args.strategy, "alt": args.alt, "glossary": args.glossary,
               "calls": [c.call_id for c in calls], "missing_cache": missing_cache, "sets": results,
               "weights_all_calls": model.weights()}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"detector_{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")
    (ROOT / "reports" / f"detector_{tag}.md").write_text(to_markdown(summary), encoding="utf-8")
    print(f"→ reports/detector_{tag}.md", flush=True)


def to_markdown(s: dict) -> str:
    any_set = next(iter(s["sets"].values()))
    lines = [
        f"# 검수 0층: 위험 점수 — {s['model']} {s['strategy']}",
        "",
        f"Earnings-21 {len(s['calls'])}통화 · 낱말 {any_set['words']:,} · "
        f"틀린 낱말(바꿈 · 끼어듦) {any_set['errors']:,} · 빠진 정답 낱말 {any_set['deletions']:,}"
        "(이 점수로 직접 못 잡음). 통화 하나씩 빼고 학습해 그 통화를 점수 매김.",
        f"두 번째 의견: {s['alt'] or '없음'} · 용어집: {s['glossary'] or '없음'}"
        + (f" · 창별 저장본 없음: {s['missing_cache']}" if s["missing_cache"] else ""),
        "",
        "| 특징 묶음 | AUC | 낱말 상위 5% · 10% · 20% | 듣는 시간 10% · 20% · 30% "
        "| 고유명사 · 숫자 오류(시간 10 · 20 · 30%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in s["sets"].items():
        wb = " · ".join(f"{v:.0%}" for v in r["catch_at_word_budget"].values())
        tb = " · ".join(f"{v:.0%}" for v in r["catch_at_time_budget"].values())
        gb = " · ".join(f"{v:.0%}" if v is not None else "–" for v in r["tagged_catch_at_time_budget"].values())
        lines.append(f"| {name} | {r['auc']} | {wb} | {tb} | {gb} |")
    lines += ["", "**읽는 법.** '듣는 시간 20%' = 사람이 녹음 길이의 20% 만(위험 높은 낱말 앞뒤 1초씩, 2초 안으로 붙은",
              "구간은 하나로) 들을 때 틀린 낱말 중 그 안에 든 비율. 아무 데나 들으면 대략 시간 비율만큼 잡힌다.", "",
              "**가중치(전부, 모든 통화로 학습, 표준화 기준 — 클수록 위험을 올림)**", ""]
    lines += [f"- {k}: {v:+.3f}" for k, v in sorted(s["weights_all_calls"].items(), key=lambda kv: -abs(kv[1]))]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
