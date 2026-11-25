"""검수 1층 평가(+L): Earnings-21 O1 받아쓰기에 고유명사 제한 교정을 걸고 정답과 견준다.

    python scripts/eval_correction.py --model large-v3-turbo --proposer fuzzy --glossary oracle
    ANTHROPIC_API_KEY=... python scripts/eval_correction.py --model large-v3-turbo --proposer llm --glossary noisy
    python scripts/eval_correction.py --model large-v3-turbo --proposer llm --glossary noisy --risky-top 0.2
    python scripts/eval_correction.py --hyp reference --proposer fuzzy --glossary noisy   # 받아쓰기 없이 과교정만
    python scripts/eval_correction.py --model large-v3-turbo --proposer search --limit 2   # claude -p + 웹 검색

- 용어집은 +P 와 같다(eval_earnings21.call_terms): oracle = 그 통화 정답에 나오는 말(상한) · noisy = 같은 수의 엉뚱한
  이름을 섞음. 그래서 '디코딩 때 hotwords(+P)' 와 '받아 적은 뒤 교정(+L)' 을 같은 용어집으로 견줄 수 있다.
- 흔한 낱말(규칙 7): 이 통화를 뺀 Earnings-21 정답 전사문 중 --common-min-docs 개 넘게 나오는 낱말. 실제로 쓸 때는
  지난 통화 전사문 묶음이 이 자리를 채운다. 흔한 낱말이 걸린 바꿈은 0층 위험 표시가 있어야 받아들인다.
- --risky-top f: 0층 위험 점수(eval_detector.py 가 남긴 O1.risk.json) 상위 f 낱말을 위험으로 표시한다.
  --ask-only-risky 를 함께 주면 위험 낱말이 든 후보만 제안 받는다(LLM 호출 수를 줄임).
- --hyp reference: 정답 글 자체를 받아쓰기로 넣는다(완벽한 받아쓰기). 여기서 바뀐 것은 전부 과교정이다 — 모델 ·
  그래픽카드 · 녹음 없이 '교정기가 맞는 글을 얼마나 망치나' 를 잰다.
- LLM 답은 cache/llm/ 에 저장한다(같은 입력 → 호출 없이 같은 답).
- --proposer search: 용어집 없이 claude -p(구독 로그인, 도구는 WebSearch · WebFetch 만)가 통화 전체에서 잘못 들은
  듯한 이름을 골라 검색하고, 검색 결과에 그 철자가 실제로 있는 것만 용어집으로 삼는다(search_fix.verify).
  세션 기록은 cache/llm_search/ 에 저장한다. --claude-bin 으로 claude 실행 파일을 고른다.

결과: runs/earnings21/<모델>/<파일>/O1+L-<제안자>-<용어집>.json(.words.json) · reports/correction_<…>.json · .md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_earnings21  # noqa: E402
from eval_earnings21 import call_terms, file_ids  # noqa: E402

from callscribe import metrics  # noqa: E402
from callscribe.datasets import read_nlp  # noqa: E402
from callscribe.entity_fix import (  # noqa: E402
    Edit,
    FuzzyProposer,
    LLMProposer,
    common_vocabulary,
    correct_entities,
    edit_outcomes,
)
from callscribe.glossary import term_recall  # noqa: E402
from callscribe.llm import AnthropicLLM, CachedLLM  # noqa: E402
from callscribe.normalize import normalize_en_conventional  # noqa: E402
from callscribe.search_fix import (  # noqa: E402
    ClaudeSearch,
    FixedEditsProposer,
    build_prompt,
    parse_proposals,
    transcript_urls,
    verify,
)
from callscribe.words import RefToken, Word  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def reference_as_words(ref: list[RefToken], nlp_path: Path) -> list[Word]:
    """정답 .nlp 를 받아쓰기 모양으로(문장부호 붙임, 시각은 0.3초 간격 가짜)."""
    rows = nlp_path.read_text(encoding="utf-8").splitlines()[1:]
    out = []
    for k, (r, line) in enumerate(zip(ref, [x for x in rows if x.strip()], strict=True)):
        punct = line.split("|")[4]
        t = r.start if r.start is not None else k * 0.3
        out.append(Word(r.text + punct, t, (r.end if r.end is not None else t + 0.25), None, k // 100))
    return out


def scores(ref, words) -> dict:
    ev = metrics.evaluate(ref, words, normalize=normalize_en_conventional, join_percent=True)
    return {"overall": ev["overall"], "tagged": ev["tagged"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3-turbo", help="받아쓰기 결과 폴더(+P 결과는 large-v3-turbo-oracle)")
    ap.add_argument("--strategy", default="O1")
    ap.add_argument("--hyp", choices=("asr", "reference"), default="asr")
    ap.add_argument("--subset", default="eval10")
    ap.add_argument("--proposer", choices=("fuzzy", "llm", "search"), default="fuzzy")
    ap.add_argument("--fuzzy-min", type=float, default=0.8)
    ap.add_argument("--glossary", choices=("oracle", "noisy"), default="oracle")
    ap.add_argument("--max-terms", type=int, default=40)
    ap.add_argument("--llm-model", default="claude-sonnet-5-5")
    ap.add_argument("--claude-bin", default=os.environ.get("CLAUDE_BIN", "claude"),
                    help="--proposer search 가 부를 claude 실행 파일")
    ap.add_argument("--risky-top", type=float, default=None)
    ap.add_argument("--ask-only-risky", action="store_true")
    ap.add_argument("--risk-for-all", action="store_true", help="모든 바꿈에 0층 위험 표시를 요구")
    ap.add_argument("--no-shape-rules", action="store_true", help="규칙 6(모양만 바꾸는 편집 거르기)을 끔 — 견주기용")
    ap.add_argument("--subset-name", default=None, help="결과 파일 이름에 넣을 통화 묶음 이름(기본: --subset)")
    ap.add_argument("--common-min-docs", type=int, default=3, help="0 이면 규칙 7 을 쓰지 않음")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "earnings21")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 몇 통화만(시험용)")
    args = ap.parse_args()
    eval_earnings21.DATA = args.data  # file_ids 가 이 폴더의 eval10 목록을 읽게

    llm = search = None
    if args.proposer == "llm":
        llm = CachedLLM(AnthropicLLM(model=args.llm_model), ROOT / "cache" / "llm")
        ptag = f"llm-{args.llm_model}"
    elif args.proposer == "search":
        search = ClaudeSearch(binary=args.claude_bin, model=args.llm_model, cache_dir=ROOT / "cache" / "llm_search")
        ptag = f"search-{args.llm_model}"
    else:
        ptag = f"fuzzy{args.fuzzy_min:g}"
    glossary_tag = "web" if search else args.glossary  # 검색 제안자는 미리 만든 용어집을 쓰지 않는다
    tag = (f"{args.strategy}+L-{ptag}-{glossary_tag}" + (f"-risky{args.risky_top:g}" if args.risky_top else "")
           + ("-askrisky" if args.ask_only_risky else "") + ("-riskall" if args.risk_for_all else "")
           + ("-noshape" if args.no_shape_rules else "")
           + (f"-common{args.common_min_docs}" if args.common_min_docs else "-nocommon"))
    ref_dir = args.data / "transcripts" / "nlp_references"
    corpus = {p.stem: [line.split("|", 1)[0] for line in p.read_text(encoding="utf-8").splitlines()[1:]]
              for p in sorted(ref_dir.glob("*.nlp"))} if args.common_min_docs else {}
    model_dir = "reference" if args.hyp == "reference" else args.model
    runs = ROOT / "runs" / "earnings21" / model_dir

    per_call, totals = [], {}
    for fid in file_ids(args.subset)[: args.limit]:
        nlp = args.data / "transcripts" / "nlp_references" / f"{fid}.nlp"
        ref = read_nlp(nlp, args.data / "transcripts" / "wer_tags" / f"{fid}.wer_tag.json")
        if args.hyp == "reference":
            words = reference_as_words(ref, nlp)
        else:
            wpath = runs / fid / f"{args.strategy}.words.json"
            if not wpath.exists():
                print(f"{fid}: {wpath} 없음 — 건너뜀", flush=True)
                continue
            words = [Word(**w) for w in json.loads(wpath.read_text(encoding="utf-8"))]
        risky = None
        if args.risky_top:
            rpath = runs / fid / f"{args.strategy}.risk.json"
            if not rpath.exists():
                sys.exit(f"{rpath} 없음 — eval_detector.py 를 먼저")
            r = np.array(json.loads(rpath.read_text(encoding="utf-8"))["risk"])
            risky = r >= np.quantile(r, 1 - args.risky_top)
        meta = next((m for m in company_rows(args.data) if m["file_id"] == fid), {})
        search_rec = None
        if search:
            trace = search.run(build_prompt(words, meta.get("company_name", ""), meta.get("financial_quarter", "")))
            checked = verify(parse_proposals(trace.text), trace, words)
            ok = [p for p, why in checked if why == "ok"]
            terms = list(dict.fromkeys(p.new for p in ok))  # 검색으로 확정한 이름 = 이 통화의 용어집
            proposer = FixedEditsProposer([Edit(p.old, p.new, p.url) for p in ok])
            search_rec = {"proposals": [{"old": p.old, "new": p.new, "url": p.url, "evidence": p.evidence,
                                         "check": why} for p, why in checked],
                          "checks": _count(why for _, why in checked),
                          "tool_calls": _count(c["name"] for c in trace.calls),
                          "transcript_urls": transcript_urls(trace), "turns": trace.turns,
                          "cost_usd": trace.cost_usd, "usage": trace.usage}
        else:
            terms = call_terms(ref, args.glossary, seed=int(fid), max_terms=args.max_terms, data=args.data)
            proposer = (LLMProposer(llm, context=f"{meta.get('company_name', '')} earnings call".strip())
                        if llm else FuzzyProposer(min_score=args.fuzzy_min))
        common = (common_vocabulary([d for k, d in corpus.items() if k != fid], args.common_min_docs)
                  if args.common_min_docs else frozenset())
        after, rep = correct_entities(words, terms, proposer, risky=risky, ask_only_risky=args.ask_only_risky,
                                      common=common, risk_for_all=args.risk_for_all,
                                      shape_rules=not args.no_shape_rules)
        before_s, after_s = scores(ref, words), scores(ref, after)
        outcome = edit_outcomes(ref, words, after, rep["mapping"], normalize_en_conventional)
        oracle_terms = call_terms(ref, "oracle", seed=0, max_terms=args.max_terms, data=args.data)
        rec = {"file": fid, "tag": tag, "glossary": terms, "before": before_s, "after": after_s, "outcome": outcome,
               "term_recall_before": term_recall(ref, words, oracle_terms),
               "term_recall_after": term_recall(ref, after, oracle_terms),
               "correction": {k: v for k, v in rep.items() if k != "mapping"}, "search": search_rec}
        (runs / fid).mkdir(parents=True, exist_ok=True)
        (runs / fid / f"{tag}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        (runs / fid / f"{tag}.words.json").write_text(json.dumps([w.to_dict() for w in after], ensure_ascii=False),
                                                     encoding="utf-8")
        per_call.append(rec)
        for k, v in outcome.items():
            if isinstance(v, int):
                totals[k] = totals.get(k, 0) + v
        print(f"{fid}: WER {before_s['overall']['rate']} → {after_s['overall']['rate']} · 고유명사 "
              f"{before_s['tagged']['entity']['rate']} → {after_s['tagged']['entity']['rate']} · "
              f"바꿈 {outcome['applied']} (고침 {outcome['fixed']} · 망침 {outcome['broke']}) · 제안 받은 구간 "
              f"{rep['asked']}/{rep['segments']}"
              + (f" · 검색 제안 {len(search_rec['proposals'])} → 확정 {len(terms)} · 도구 {search_rec['tool_calls']}"
                 f" · 전사문 주소 {len(search_rec['transcript_urls'])}" if search_rec else ""), flush=True)
    if not per_call:
        sys.exit("평가한 통화가 없다")

    def pooled(side: str, group: str | None) -> float:
        get = (lambda r: r[side]["overall"]) if group is None else (lambda r: r[side]["tagged"][group])
        errors = sum(get(r).get("errors", get(r).get("S", 0) + get(r).get("D", 0) + get(r).get("I", 0))
                     for r in per_call)
        return round(errors / sum(get(r)["ref"] for r in per_call), 5)

    subset = args.subset_name or (args.subset if len(args.subset) <= 16 else f"{len(per_call)}calls")
    if args.limit and not args.subset_name:
        subset += f"-first{len(per_call)}"  # 시험 삼아 앞 몇 통화만 돌린 결과가 전체 결과 이름을 덮지 않게
    summary = {
        "tag": tag, "subset": subset, "hyp": args.hyp, "model": args.model,
        "calls": [r["file"] for r in per_call],
        "wer": [pooled("before", None), pooled("after", None)],
        "entity_error": [pooled("before", "entity"), pooled("after", "entity")],
        "number_error": [pooled("before", "number"), pooled("after", "number")],
        "outcome": {**totals, "over_correction_rate": round(totals["broke"] / totals["applied"], 4)
                    if totals.get("applied") else None},
        "asked_segments": sum(r["correction"]["asked"] for r in per_call),
        "segments": sum(r["correction"]["segments"] for r in per_call),
        "decisions": _sum_dicts(r["correction"]["decisions"] for r in per_call),
        "llm": {"calls": llm.calls, "cache_hits": llm.hits} if llm else None,
        "search": _search_summary(per_call, search) if search else None,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    name = f"correction_{model_dir}_{subset}_{tag}"
    (ROOT / "reports" / f"{name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def _search_summary(per_call: list[dict], search: ClaudeSearch) -> dict:
    """검색 제안자 묶음 숫자(이름 · 주소는 빼고 — 통화별 기록에만 남긴다)."""
    recs = [r["search"] for r in per_call]
    return {"model": search.name, "sessions": search.calls, "cache_hits": search.hits,
            "proposals": sum(len(s["proposals"]) for s in recs), "checks": _sum_dicts(s["checks"] for s in recs),
            "tool_calls": _sum_dicts(s["tool_calls"] for s in recs),
            "calls_that_saw_transcripts": sum(bool(s["transcript_urls"]) for s in recs),
            "cost_usd": round(sum(s["cost_usd"] or 0 for s in recs), 2)}


def _sum_dicts(ds) -> dict:
    out: dict[str, int] = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


_rows_cache: dict[Path, list[dict]] = {}


def company_rows(data: Path) -> list[dict]:
    import csv

    if data not in _rows_cache:
        path = data / "earnings21-file-metadata.csv"
        _rows_cache[data] = list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []
    return _rows_cache[data]


if __name__ == "__main__":
    main()
