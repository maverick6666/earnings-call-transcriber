"""한국어 평가: Zeroth-Korean 시험 묶음(457문장 · 1.2시간, CC BY 4.0)을 화자별로 이어 붙인 긴 오디오 10개.

    .venv/Scripts/python scripts/eval_zeroth.py --model large-v3-turbo --strategies B0,B1,O1,O2,V1,V2

- 문장 사이 0.3~1.2초 쉼, 5% 는 3~6초 긴 쉼, 짝수 번째 녹음은 처음 30초 무음(대기 시나리오).
- 정답은 문장 단위 시각이 정확하다(글자 시각은 문장 안에서 고르게 나눈 근사).
  점수는 글자 오류율(CER, 띄어쓰기 · 문장부호 뺌).
- 자료는 Hugging Face 의 OpenSLR 사본(kresnik/zeroth_korean, test 파일 하나)을 쓴다 — OpenSLR 미러가 너무 느려서.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from callscribe import chunking, metrics, runner
from callscribe.backends import SAMPLE_RATE, FasterWhisperASR, speech_energy
from callscribe.datasets import build_long_audio
from callscribe.normalize import normalize_ko_chars
from callscribe.words import RefToken

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "zeroth_hf" / "test.parquet"
LONG = ROOT / "data" / "zeroth_long"
RUNS = ROOT / "runs" / "zeroth"
CACHE = ROOT / "cache"
MODELS = ROOT / "models"


def build() -> list[str]:
    """화자별 긴 오디오(16 kHz npy) · 정답(json). 이미 있으면 다시 만들지 않는다."""
    LONG.mkdir(parents=True, exist_ok=True)
    done = sorted(p.stem for p in LONG.glob("*.npy"))
    if done:
        return done
    rows = pq.read_table(SRC).to_pylist()
    by_speaker: dict[int, list[dict]] = {}
    for r in rows:
        by_speaker.setdefault(r["speaker_id"], []).append(r)
    names = []
    for k, (spk, utts) in enumerate(sorted(by_speaker.items())):
        items = []
        for r in sorted(utts, key=lambda r: r["id"]):
            audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            if sr != SAMPLE_RATE:
                raise ValueError(f"{r['id']}: {sr} Hz")
            items.append((r["id"], audio if audio.ndim == 1 else audio.mean(axis=1), r["text"]))
        audio, ref, spans = build_long_audio(items, SAMPLE_RATE, seed=k, lead_silence=30.0 if k % 2 == 0 else 0.0)
        name = f"spk{spk}"
        np.save(LONG / f"{name}.npy", audio)
        (LONG / f"{name}.json").write_text(json.dumps({"ref": [[t.text, t.start, t.end] for t in ref],
                                                       "spans": spans}, ensure_ascii=False), encoding="utf-8")
        names.append(name)
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--tag", default="", help="결과 폴더 이름에 붙일 꼬리표")
    ap.add_argument("--strategies", default="B0,B1,O1,O2,V1,V2")
    ap.add_argument("--min-silence", type=float, default=runner.RunConfig("V1").min_silence,
                    help="V1 · V2: 이보다 긴 말소리 없는 구간만 건너뜀(초)")
    args = ap.parse_args()
    names = build()
    asr = FasterWhisperASR(model_size=args.model, language="ko", download_root=str(MODELS))
    out_root = RUNS / (args.model + (f"-{args.tag}" if args.tag else ""))
    for name in names:
        todo = [s for s in args.strategies.split(",") if not (out_root / name / f"{s}.json").exists()]
        if not todo:
            continue
        audio = np.load(LONG / f"{name}.npy")
        meta = json.loads((LONG / f"{name}.json").read_text(encoding="utf-8"))
        ref = [RefToken(t, a, b) for t, a, b in meta["ref"]]
        duration = len(audio) / SAMPLE_RATE
        energy = speech_energy(audio)
        asr.audio = audio
        for s in todo:
            t0 = time.time()
            cfg = runner.RunConfig(s, min_silence=args.min_silence)
            res = runner.run(cfg, asr, duration, f"zeroth-{name}", cache_dir=CACHE, energy=energy)
            elapsed = time.time() - t0
            cut = chunking.cut_edges(res.chunks, duration) if s != "B1" else []
            ev = metrics.evaluate(ref, res.words, cut_edges=cut, seams=chunking.seams(res.chunks),
                                  normalize=normalize_ko_chars)
            out = out_root / name
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{s}.words.json").write_text(json.dumps([w.to_dict() for w in res.words], ensure_ascii=False),
                                                 encoding="utf-8")
            (out / f"{s}.json").write_text(json.dumps(
                {"file": name, "strategy": s, "model": args.model, "duration": duration, "elapsed_sec": elapsed,
                 "audio_transcribed_sec": sum(c.end - c.start for c in res.chunks if not c.skip),
                 "chunks": [[c.start, c.end, c.own_start, c.own_end, c.skip] for c in res.chunks],
                 "eval": ev, "manifest": res.manifest}, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{name} {s} CER {ev['overall']['rate']} cut {ev['near_cut_edges']['rate']} {elapsed:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
