"""저장된 청크 받아쓰기로 다시 잇고 다시 채점한다(모델 · 그래픽카드 없이). 잇는 방법을 고친 뒤에 쓴다.

    .venv/Scripts/python scripts/remerge.py --corpus zeroth --strategies O2

저장본에 없는 청크가 있으면 멈춘다(새로 받아 적지 않는다).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from callscribe import chunking, metrics, runner
from callscribe.normalize import normalize_en, normalize_ko_chars
from callscribe.words import RefToken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
CACHE = ROOT / "cache"


class CachedOnly:
    def __init__(self, tid: dict):
        self.name, self.version, self.settings = tid["name"], tid["version"], tid["settings"]

    def transcribe(self, start: float, end: float):
        raise RuntimeError(f"저장본에 없는 청크 {start:.3f}-{end:.3f}")


def reference(corpus: str, fid: str) -> list[RefToken]:
    if corpus == "earnings21":
        from eval_earnings21 import DATA

        from callscribe.datasets import read_nlp

        return read_nlp(DATA / "transcripts" / "nlp_references" / f"{fid}.nlp",
                        DATA / "transcripts" / "wer_tags" / f"{fid}.wer_tag.json")
    meta = json.loads((ROOT / "data" / "zeroth_long" / f"{fid}.json").read_text(encoding="utf-8"))
    return [RefToken(t, a, b) for t, a, b in meta["ref"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("earnings21", "zeroth"), required=True)
    ap.add_argument("--strategies", default="O2")
    args = ap.parse_args()
    wanted = set(args.strategies.split(","))
    norm = normalize_en if args.corpus == "earnings21" else normalize_ko_chars
    refs: dict[str, list[RefToken]] = {}
    for path in sorted((ROOT / "runs" / args.corpus).glob("*/*/*.json")):
        if path.name.endswith(".words.json") or path.stem not in wanted:
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec["strategy"].startswith("V"):
            raise SystemExit("V1 · V2 는 소리 크기가 필요해 여기서 다시 잇지 않는다")
        man = rec["manifest"]
        cfg = runner.RunConfig(**man["config"])
        res = runner.run(cfg, CachedOnly(man["transcriber"]), rec["duration"], man["audio_id"], cache_dir=CACHE)
        ref = refs.setdefault(rec["file"], reference(args.corpus, rec["file"]))
        cut = chunking.cut_edges(res.chunks, rec["duration"]) if rec["strategy"] != "B1" else []
        term_recall = rec["eval"].get("term_recall")
        rec["eval"] = metrics.evaluate(ref, res.words, cut_edges=cut, seams=chunking.seams(res.chunks),
                                       normalize=norm)
        if term_recall is not None:
            from eval_earnings21 import call_terms

            from callscribe.glossary import term_recall as recall

            rec["eval"]["term_recall"] = recall(ref, res.words, call_terms(ref, "oracle", seed=0))
        rec["manifest"] = res.manifest
        path.with_name(f"{path.stem}.words.json").write_text(
            json.dumps([w.to_dict() for w in res.words], ensure_ascii=False), encoding="utf-8")
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        rate = rec["eval"]["overall"]["rate"]
        print(f"{path.parent.parent.name}/{rec['file']} {rec['strategy']} → {rate}", flush=True)


if __name__ == "__main__":
    main()
