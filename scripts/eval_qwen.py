"""맥락을 넣을 수 있는 음성 LLM(Qwen3-ASR-1.7B)을 Whisper 와 같은 평가 틀 · 같은 O1 창으로 잰다.

    # torch · qwen-asr 가 든 별도 환경에서(모델은 처음 실행 때 Hugging Face 에서 받음)
    PYTHONPATH=src .venv-qwen/Scripts/python -B scripts/eval_qwen.py --corpus earnings21 --context none
    PYTHONPATH=src .venv-qwen/Scripts/python -B scripts/eval_qwen.py --corpus earnings21 --context oracle
    PYTHONPATH=src .venv-qwen/Scripts/python -B scripts/eval_qwen.py --corpus earnings21 --context noisy
    PYTHONPATH=src .venv-qwen/Scripts/python -B scripts/eval_qwen.py --corpus zeroth

- 창 계획 · 잇기 · 채점은 Whisper 와 똑같다(runner.plan · merge · metrics). 창 받아쓰기는 여러 개를 한 번에 넣고
  cache/ 에 같은 이름으로 저장해, 멈췄다 다시 돌리면 이어 한다.
- context: none = 맥락 없음 · oracle / noisy = +P 와 같은 용어집(eval_earnings21.call_terms)을 시스템 프롬프트로.
  Whisper 의 hotwords(+P)와 같은 용어집이라 '맥락을 넣는 방법' 끼리 견줄 수 있다.
- 결과는 eval_earnings21 · eval_zeroth 와 같은 모양으로 runs/<자료>/qwen3-asr-1.7b[-oracle|-noisy]/<파일>/O1.json
  에 쓴다(summarize · make_tables · per_file 이 그대로 읽는다). 영어는 관례 정규화 점수도 함께.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_earnings21 import call_terms, file_ids  # noqa: E402

from callscribe import chunking, metrics, runner  # noqa: E402
from callscribe.datasets import read_nlp  # noqa: E402
from callscribe.glossary import term_recall  # noqa: E402
from callscribe.normalize import normalize_en, normalize_en_conventional, normalize_ko_chars  # noqa: E402
from callscribe.qwen import SAMPLE_RATE, QwenASR  # noqa: E402
from callscribe.words import RefToken  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
E21 = ROOT / "data" / "earnings21"
LONG = ROOT / "data" / "zeroth_long"


def calls(corpus: str) -> list[tuple[str, str]]:
    """(파일 번호, 녹음 식별값)."""
    if corpus == "earnings21":
        return [(fid, f"earnings21-{fid}") for fid in file_ids("eval10")]
    return [(p.stem, f"zeroth-{p.stem}") for p in sorted(LONG.glob("*.npy"))]


def load(corpus: str, fid: str) -> tuple[np.ndarray, list[RefToken]]:
    if corpus == "earnings21":
        import librosa

        audio, _ = librosa.load(str(E21 / "media" / f"{fid}.mp3"), sr=SAMPLE_RATE, mono=True)
        ref = read_nlp(E21 / "transcripts" / "nlp_references" / f"{fid}.nlp",
                       E21 / "transcripts" / "wer_tags" / f"{fid}.wer_tag.json")
        return audio.astype(np.float32), ref
    meta = json.loads((LONG / f"{fid}.json").read_text(encoding="utf-8"))
    return np.load(LONG / f"{fid}.npy"), [RefToken(t, a, b) for t, a, b in meta["ref"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("earnings21", "zeroth"), default="earnings21")
    ap.add_argument("--context", choices=("none", "oracle", "noisy"), default="none")
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="앞에서 몇 통화만(시험용)")
    args = ap.parse_args()
    if args.corpus == "zeroth" and args.context != "none":
        sys.exit("한국어 자료에는 용어집이 없다")

    language = "English" if args.corpus == "earnings21" else "Korean"
    asr = QwenASR(model_name=args.model, language=language, batch_size=args.batch)
    folder = args.model.split("/")[-1].lower() + ("" if args.context == "none" else f"-{args.context}")
    out_root = ROOT / "runs" / args.corpus / folder
    cfg = runner.RunConfig("O1")
    for fid, audio_id in calls(args.corpus)[: args.limit]:
        if (out_root / fid / "O1.json").exists():
            continue
        audio, ref = load(args.corpus, fid)
        duration = len(audio) / SAMPLE_RATE
        terms = call_terms(ref, args.context, seed=int(fid)) if args.context != "none" else []
        asr.audio = audio
        asr.context = ", ".join(terms)
        t0 = time.time()
        # 1) 아직 없는 창만 여러 개씩 받아 적어 runner 와 같은 저장본 자리에 쓴다
        chunks = runner.plan(cfg, duration)
        key = runner.cache_key(audio_id, asr)
        store = CACHE / key
        store.mkdir(parents=True, exist_ok=True)
        todo = [c for c in chunks if not c.skip and not (store / f"{c.start:.3f}-{c.end:.3f}.json").exists()]
        for k in range(0, len(todo), args.batch * 4):
            part = todo[k: k + args.batch * 4]
            for c, words in zip(part, asr.transcribe_many([(c.start, c.end) for c in part]), strict=True):
                path = store / f"{c.start:.3f}-{c.end:.3f}.json"
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps([w.to_dict() for w in words], ensure_ascii=False), encoding="utf-8")
                runner._replace(tmp, path)
            print(f"  {fid}: 창 {min(k + len(part), len(todo))}/{len(todo)} · {time.time() - t0:.0f}s", flush=True)
        # 2) 잇기 · 검사는 Whisper 와 같은 runner 로(이제 모든 창이 저장본에 있다)
        res = runner.run(cfg, asr, duration, audio_id, cache_dir=CACHE)
        elapsed = time.time() - t0
        cut, seams = chunking.cut_edges(res.chunks, duration), chunking.seams(res.chunks)
        record = {"file": fid, "strategy": "O1", "model": folder, "duration": duration, "elapsed_sec": elapsed,
                  "context": asr.context, "transcribed_chunks": len(todo), "cached_chunks": len(chunks) - len(todo),
                  "audio_transcribed_sec": sum(c.end - c.start for c in res.chunks if not c.skip),
                  "chunks": [[c.start, c.end, c.own_start, c.own_end, c.skip] for c in res.chunks],
                  "manifest": res.manifest}
        if args.corpus == "earnings21":
            record["eval"] = metrics.evaluate(ref, res.words, cut_edges=cut, seams=seams, normalize=normalize_en)
            record["eval_conventional"] = metrics.evaluate(ref, res.words, cut_edges=cut, seams=seams,
                                                           normalize=normalize_en_conventional, join_percent=True)
            record["eval"]["term_recall"] = term_recall(ref, res.words, call_terms(ref, "oracle", seed=0))
            rate = record["eval_conventional"]["overall"]["rate"]
        else:
            record["eval"] = metrics.evaluate(ref, res.words, cut_edges=cut, seams=seams, normalize=normalize_ko_chars)
            rate = record["eval"]["overall"]["rate"]
        (out_root / fid).mkdir(parents=True, exist_ok=True)
        (out_root / fid / "O1.words.json").write_text(json.dumps([w.to_dict() for w in res.words],
                                                                 ensure_ascii=False), encoding="utf-8")
        (out_root / fid / "O1.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{fid} O1 오류율 {rate} · 자른 곳 {record['eval']['near_cut_edges']['rate']} · {elapsed:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
