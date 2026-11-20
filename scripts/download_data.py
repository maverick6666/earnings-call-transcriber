"""평가 자료 · 모델 내려받기(저장소 안 data/ · models/, 깃에는 올리지 않음).

    .venv/Scripts/python scripts/download_data.py earnings21   # Rev Earnings-21 mp3 44개 + 정답 (CC BY-SA 4.0)
    .venv/Scripts/python scripts/download_data.py zeroth       # Zeroth-Korean 시험 묶음 60MB (CC BY 4.0, HF 사본)
    .venv/Scripts/python scripts/download_data.py models       # faster-whisper small · large-v3-turbo · large-v3

받은 파일은 데이터로만 쓴다(실행하지 않음).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
E21_API = "https://api.github.com/repos/revdotcom/speech-datasets/contents/earnings21/{}"
E21_DIRS = ("media", "transcripts/nlp_references", "transcripts/wer_tags", "bias_lists")
E21_FILES = ("earnings21-file-metadata.csv", "eval10-file-metadata.csv", "speaker-metadata.csv", "LICENSE.md")
# OpenSLR 40 원본(10GB)은 미러가 초당 10~30KB 라, 같은 자료의 Hugging Face 사본에서 시험 묶음만 받는다.
ZEROTH_HF = "https://huggingface.co/datasets/kresnik/zeroth_korean/resolve/main/{}"
MODEL_SIZES = ("small", "large-v3-turbo", "large-v3")
UA = {"User-Agent": "earnings-call-transcriber-eval"}


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def earnings21() -> None:
    out = DATA / "earnings21"
    for d in E21_DIRS:
        with urllib.request.urlopen(urllib.request.Request(E21_API.format(d), headers=UA), timeout=60) as r:
            items = json.load(r)
        files = [i for i in items if i["type"] == "file"]
        for k, item in enumerate(files, 1):
            fetch(item["download_url"], out / d / item["name"])
            print(f"{d}: {k}/{len(files)} {item['name']}", flush=True)
    for name in E21_FILES:
        fetch(f"https://raw.githubusercontent.com/revdotcom/speech-datasets/main/earnings21/{name}", out / name)
    print("earnings21 끝", flush=True)


def zeroth() -> None:
    out = DATA / "zeroth_hf"
    fetch(ZEROTH_HF.format("data/test-00000-of-00001.parquet"), out / "test.parquet")
    fetch(ZEROTH_HF.format("README.md"), out / "README.md")  # 출처 · 사용 조건
    print("zeroth 끝", flush=True)


def models() -> None:
    from faster_whisper.utils import download_model

    for size in MODEL_SIZES:
        path = download_model(size, cache_dir=str(MODELS))
        print(f"{size}: {path}", flush=True)


if __name__ == "__main__":
    {"earnings21": earnings21, "zeroth": zeroth, "models": models}[sys.argv[1]]()
