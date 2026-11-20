"""Earnings-21 실제 평가: 방식 × 모델.

    .venv/Scripts/python scripts/eval_earnings21.py --model large-v3 --subset eval10 --strategies B0,B1,O1,O2,V1,V2

- 청크 받아쓰기는 cache/ 에 저장해 방식끼리 같은 구간을 함께 쓰고(O1 · O2), 멈췄다 다시 돌리면 이어 한다.
- 결과: runs/earnings21/<모델>/<파일>/<방식>.json (점수 · 실행 기록 · 걸린 시간), 낱말은 같은 이름 .words.json
- 점수는 정답과 받아쓰기에 같은 정규화(소문자 · 문장부호 제거)를 쓴다.
  숫자 표기(4.2 ↔ four point two) 정규화는 하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from callscribe import chunking, metrics, runner
from callscribe.backends import SAMPLE_RATE, FasterWhisperASR, load_audio, speech_energy
from callscribe.datasets import read_nlp
from callscribe.glossary import hotwords, term_recall
from callscribe.normalize import normalize_en

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "earnings21"
RUNS = ROOT / "runs" / "earnings21"
CACHE = ROOT / "cache"
MODELS = ROOT / "models"


def file_ids(subset: str) -> list[str]:
    if subset == "all":
        return sorted(p.stem for p in (DATA / "media").glob("*.mp3"))
    if subset == "eval10":
        with open(DATA / "eval10-file-metadata.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        key = next(k for k in rows[0] if "id" in k.lower())
        return sorted(str(r[key]).strip() for r in rows)
    return subset.split(",")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--subset", default="eval10", help="eval10 · all · 쉼표로 이은 파일 번호")
    ap.add_argument("--strategies", default="B0,B1,O1,O2,V1,V2")
    ap.add_argument("--min-silence", type=float, default=runner.RunConfig("V1").min_silence,
                    help="V1 · V2: 이보다 긴 말소리 없는 구간만 건너뜀(초)")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--hotwords", choices=("oracle", "noisy"), default=None,
                    help="+P: oracle = 그 통화 정답에 나오는 용어만(상한) · noisy = 같은 수의 엉뚱한 이름 섞음")
    ap.add_argument("--tag", default="", help="결과 폴더 이름에 붙일 꼬리표")
    args = ap.parse_args()

    asr = FasterWhisperASR(model_size=args.model, language="en", compute_type=args.compute_type,
                           download_root=str(MODELS))
    out_root = RUNS / (args.model + (f"-{args.tag}" if args.tag else ""))
    for fid in file_ids(args.subset):
        todo = [s for s in args.strategies.split(",") if not (out_root / fid / f"{s}.json").exists()]
        if not todo:
            continue
        audio = load_audio(DATA / "media" / f"{fid}.mp3")
        duration = len(audio) / SAMPLE_RATE
        ref = read_nlp(DATA / "transcripts" / "nlp_references" / f"{fid}.nlp",
                       DATA / "transcripts" / "wer_tags" / f"{fid}.wer_tag.json")
        energy = speech_energy(audio)
        asr.audio = audio
        terms = call_terms(ref, args.hotwords, seed=int(fid)) if args.hotwords else []
        asr.hotwords = hotwords(terms) if terms else None
        for s in todo:
            t0, n_retry = time.time(), len(asr.padded_retries)
            cfg = runner.RunConfig(s, min_silence=args.min_silence)
            res = runner.run(cfg, asr, duration, f"earnings21-{fid}", cache_dir=CACHE, energy=energy)
            elapsed = time.time() - t0
            cut = chunking.cut_edges(res.chunks, duration) if s != "B1" else []
            ev = metrics.evaluate(ref, res.words, cut_edges=cut, seams=chunking.seams(res.chunks),
                                  normalize=normalize_en)
            (out_root / fid).mkdir(parents=True, exist_ok=True)
            if args.hotwords:
                ev["term_recall"] = term_recall(ref, res.words, call_terms(ref, "oracle", seed=0))
            record = {"file": fid, "strategy": s, "model": args.model, "duration": duration, "elapsed_sec": elapsed,
                      "hotwords": asr.hotwords,
                      "transcribed_chunks": res.transcribed, "cached_chunks": res.from_cache,
                      "audio_transcribed_sec": sum(c.end - c.start for c in res.chunks if not c.skip),
                      "chunks": [[c.start, c.end, c.own_start, c.own_end, c.skip] for c in res.chunks],
                      "padded_retries": asr.padded_retries[n_retry:],
                      "eval": ev, "manifest": res.manifest}
            (out_root / fid / f"{s}.words.json").write_text(
                json.dumps([w.to_dict() for w in res.words], ensure_ascii=False), encoding="utf-8")
            (out_root / fid / f"{s}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1),
                                                      encoding="utf-8")
            o = ev["overall"]
            print(f"{fid} {s} WER {o['rate']} cut {ev['near_cut_edges']['rate']} "
                  f"auc {ev['confidence']['auc']} {elapsed:.0f}s ({res.transcribed} new, {res.from_cache} cached)",
                  flush=True)


def _bias_list(name: str) -> list[str]:
    path = DATA / "bias_lists" / f"{name}_list.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def call_terms(ref, mode: str, seed: int, max_terms: int = 40) -> list[str]:
    """+P 용어집. Earnings-21 bias_lists 는 통화별이 아니라 전체 목록이라 통화마다 골라 쓴다.

    oracle: 전체 정답에 나오는 말(oracle_list) 중 이 통화 정답에 실제로 나오는 것 — 완벽한 용어집일 때의 상한.
    noisy : oracle 만큼의 엉뚱한 이름(distractor_list 에만 있는 Fortune 500 회사명 · CEO)을 섞음 — 잡음 섞인 용어집.
    """
    text = " " + " ".join(normalize_en(" ".join(t.text for t in ref))) + " "
    oracle = [t for t in _bias_list("oracle") if f" {' '.join(normalize_en(t))} " in text]
    oracle = sorted(oracle, key=lambda t: (-text.count(f" {' '.join(normalize_en(t))} "), t))[:max_terms]
    if mode == "oracle":
        return [_pretty(t) for t in oracle]
    extra = sorted(set(_bias_list("distractor")) - set(_bias_list("oracle")))
    rng = random.Random(seed)
    mixed = oracle + rng.sample(extra, min(len(oracle), len(extra)))
    rng.shuffle(mixed)
    return [_pretty(t) for t in mixed]


def _pretty(term: str) -> str:
    """목록은 모두 대문자다. 4글자 이하 약어(BMO)는 그대로, 나머지는 'Monro Forward' 처럼."""
    return " ".join(w if len(w) <= 4 and w.isalpha() and len(term.split()) == 1 else w.capitalize()
                    for w in term.split())


if __name__ == "__main__":
    main()
