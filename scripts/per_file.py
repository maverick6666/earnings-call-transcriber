"""파일별로 본 결과: 묶어 센 오류율이 한두 통화에 끌려간 것은 아닌지, 무음 건너뛰기는 왜 졌는지.

    .venv/Scripts/python scripts/per_file.py > reports/per_file.md

- 파일마다 O1 이 다른 방식보다 낮은가(이긴 통화 수 / 전체).
- V1 · V2 가 건너뛴 구간에서 O1 은 낱말을 몇 개 받아 적었나(= 말소리 감지가 놓친 말의 대략).
  정답에는 시각이 없어 O1 받아쓰기를 대신 쓴다.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("B0", "B1", "O1", "O2", "V1", "V2")


def load(corpus: str) -> dict[str, dict[str, dict[str, dict]]]:
    """모델 → 파일 → 방식 → 기록."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for p in (ROOT / "runs" / corpus).glob("*/*/*.json"):
        if p.stem in STRATEGIES:  # 실행 기록만(.words · .risk · +L 교정 결과는 뺀다)
            rec = json.loads(p.read_text(encoding="utf-8"))
            out[p.parent.parent.name][rec["file"]][rec["strategy"]] = rec
    return out


def rate(rec: dict, conventional: bool) -> float:
    ev = rec.get("eval_conventional") if conventional else None
    return (ev or rec["eval"])["overall"]["rate"]


def wins(files: dict, a: str, b: str, conventional: bool) -> str:
    pairs = [(rate(f[a], conventional), rate(f[b], conventional)) for f in files.values() if a in f and b in f]
    if not pairs:
        return "–"
    return f"{sum(x < y for x, y in pairs)}/{len(pairs)}"


def skipped_share(v1: dict, o1_words: list[dict], corpus: str) -> tuple[float, float]:
    """(V1 이 건너뛴 시간 비율, 그 구간에 든 O1 낱말 비율)."""
    import numpy as np
    from rescore import chunks_of  # 청크 계획 복원 규칙을 채점과 같이 쓴다

    audio = (lambda fid: np.load(ROOT / "data" / "zeroth_long" / f"{fid}.npy")) if corpus == "zeroth" else None
    spans = [(c.start, c.end) for c in chunks_of(v1, audio) if c.skip]
    dur = v1["duration"]
    inside = sum(any(a <= (w["start"] + w["end"]) / 2 < b for a, b in spans) for w in o1_words)
    return sum(b - a for a, b in spans) / dur, inside / max(len(o1_words), 1)


def other_family(model: str, files: dict, data: dict, conv: bool) -> None:
    """O1 만 돌린 다른 계열 모델(Qwen3-ASR)은 같은 용어집 조건의 turbo O1 과 통화별로 견준다."""
    suffix = next((s for s in ("-oracle", "-noisy") if model.endswith(s)), "")
    base = data.get("large-v3-turbo" + suffix, {})
    print(f"\n### {model} (견줄 것: large-v3-turbo{suffix} O1)\n")
    print("| 파일 | 길이(분) | 이 모델 O1 | turbo O1 |")
    print("|---|---:|---:|---:|")
    pairs = []
    for fid in sorted(files):
        mine, theirs = rate(files[fid]["O1"], conv), base.get(fid, {}).get("O1")
        other = rate(theirs, conv) if theirs else None
        if other is not None:
            pairs.append((mine, other))
        print(f"| {fid} | {files[fid]['O1']['duration'] / 60:.0f} | {mine * 100:.1f}% | "
              + (f"{other * 100:.1f}% |" if other is not None else "– |"))
    print(f"\n통화별로 turbo O1 보다 낮은 수 — {sum(a < b for a, b in pairs)}/{len(pairs)}")


def main() -> None:
    print("# 통화별 결과\n")
    print("`scripts/per_file.py` 로 만든 표. 묶어 센 오류율이 한두 통화에 끌려간 것은 아닌지, "
          "무음 건너뛰기(V)가 어디서 졌는지 본다.")
    print("'V 가 건너뛴 시간' 은 V1 · V2 가 받아 적지 않은 녹음 비율, '그 안의 O1 낱말' 은 같은 구간에서 O1 이 "
          "받아 적은 낱말 비율(정답에 시각이 없어 대신 씀).")
    for corpus, unit in (("earnings21", "WER(관례 정규화)"), ("zeroth", "CER(관례 정규화 · 숫자를 한글 읽기로)")):
        conv = True  # 관례 점수가 없는 기록은 rate() 가 원래 점수로 대신한다
        data = load(corpus)
        print(f"\n## {corpus} — 파일별 {unit}\n")
        for model in sorted(data):
            files = data[model]
            have = [s for s in STRATEGIES if any(s in f for f in files.values())]
            if have == ["O1"] and not model.startswith(("large-v3", "small")):
                other_family(model, files, data, conv)
                continue
            base = model if "O1" in have else model.rsplit("-", 1)[0]  # 무음 2초 실행은 같은 모델의 O1 과 견준다
            if base != model:
                for fid, f in files.items():
                    if "O1" in data[base].get(fid, {}):
                        f["O1"] = data[base][fid]["O1"]
                have = ["O1"] + have
            print(f"\n### {model}\n")
            print("| 파일 | 길이(분) | " + " | ".join(have) + " | V 가 건너뛴 시간 | 그 안의 O1 낱말 |")
            print("|---|---:|" + "---:|" * len(have) + "---:|---:|")
            for fid in sorted(files):
                f = files[fid]
                cells = [f"{rate(f[s], conv) * 100:.1f}%" if s in f else "" for s in have]
                skip = ""
                if "O1" not in f:
                    continue
                v = f.get("V1") or f.get("V2")  # 건너뛰는 구간은 V1 · V2 가 같다
                if v:
                    o1_words = json.loads((ROOT / "runs" / corpus / base / fid / "O1.words.json")
                                          .read_text(encoding="utf-8"))
                    t, w = skipped_share(v, o1_words, corpus)
                    skip = f"{t * 100:.0f}% | {w * 100:.1f}%"
                else:
                    skip = " | "
                print(f"| {fid} | {f['O1']['duration'] / 60:.0f} | " + " | ".join(cells) + f" | {skip} |")
            line = [f"O1 < {s}: {wins(files, 'O1', s, conv)}" for s in have if s != "O1"]
            print("\n통화별로 O1 이 더 낮은 수 — " + " · ".join(line))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
