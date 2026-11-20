"""Earnings-21 결과를 '관례 정규화'(군말 지우기 · 구어 줄임말 풀기 · 숫자 뒤 % 붙이기)로 다시 채점한다.

    .venv/Scripts/python scripts/rescore.py          # 아직 안 한 기록만
    .venv/Scripts/python scripts/rescore.py --all    # 모두 다시

저장된 낱말(.words.json)만 쓴다(모델 · 그래픽카드 없이). 결과는 각 기록의 eval_conventional 에.
원래 채점(eval)은 그대로 둔다 — 보고서는 둘 다 싣는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from callscribe import chunking, metrics
from callscribe.chunking import Chunk
from callscribe.datasets import read_nlp
from callscribe.normalize import normalize_en_conventional
from callscribe.words import Word

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "earnings21"


def main() -> None:
    again = "--all" in sys.argv
    refs = {}
    done = 0
    for path in sorted((ROOT / "runs" / "earnings21").glob("*/*/*.json")):
        if path.name.endswith(".words.json"):
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("eval_conventional") and not again:  # 이미 채점한 기록은 건너뜀(--all 이면 모두 다시)
            continue
        words = [Word(**w) for w in json.loads(path.with_name(f"{path.stem}.words.json").read_text("utf-8"))]
        fid = rec["file"]
        if fid not in refs:
            refs[fid] = read_nlp(DATA / "transcripts" / "nlp_references" / f"{fid}.nlp",
                                 DATA / "transcripts" / "wer_tags" / f"{fid}.wer_tag.json")
        # 자른 곳 · 이음매는 원래 채점과 같게: 실행 기록에 남은 청크 계획으로 다시 만든다
        cut, seams = rec_cut_points(rec)
        rec["eval_conventional"] = metrics.evaluate(refs[fid], words, cut_edges=cut, seams=seams,
                                                    normalize=normalize_en_conventional, join_percent=True)
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        done += 1
    print(f"{done} 기록 다시 채점")


_energy_cache: dict[str, tuple] = {}


def chunks_of(rec: dict, audio=None) -> list[Chunk]:
    """기록의 청크 계획. 기록에 있으면 그것, 없으면(초기 실행) 같은 설정으로 다시 만들고 청크 수를 대조한다.

    audio: 녹음을 읽는 함수(파일 번호 → 16 kHz 배열). V1 · V2 를 다시 만들 때만 쓴다. 없으면 Earnings-21 mp3.
    """
    from callscribe import runner

    if rec.get("chunks"):
        return [Chunk(k, a, b, oa, ob, bool(sk)) for k, (a, b, oa, ob, sk) in enumerate(rec["chunks"])]
    cfg = runner.RunConfig(**rec["manifest"]["config"])
    energy = None
    if cfg.strategy in ("V1", "V2"):  # 소리 크기가 필요하다: 녹음을 다시 읽어 계산(CPU 로 몇 초)
        if rec["file"] not in _energy_cache:
            from callscribe.backends import load_audio, speech_energy

            read = audio or (lambda fid: load_audio(DATA / "media" / f"{fid}.mp3"))
            _energy_cache[rec["file"]] = speech_energy(read(rec["file"]))
        energy = _energy_cache[rec["file"]]
    chunks = runner.plan(cfg, rec["duration"], energy)
    if len(chunks) != rec["manifest"]["chunks"]:
        raise AssertionError(f"{rec['file']} {cfg.strategy}: 다시 만든 청크 수가 기록과 다르다")
    return chunks


def rec_cut_points(rec: dict) -> tuple[list[float], list[float]]:
    """원래 채점과 같은 자른 곳 · 이음매."""
    chunks = chunks_of(rec)
    cut = chunking.cut_edges(chunks, rec["duration"]) if rec["strategy"] != "B1" else []
    return cut, chunking.seams(chunks)


if __name__ == "__main__":
    main()
