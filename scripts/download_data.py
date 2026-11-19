"""평가 자료 · 모델 내려받기(저장소 안 data/ · models/, 깃에는 올리지 않음).

    .venv/Scripts/python scripts/download_data.py earnings21   # Rev Earnings-21 mp3 44개 + 정답 (CC BY-SA 4.0)
    .venv/Scripts/python scripts/download_data.py zeroth       # OpenSLR 40 Zeroth-Korean 10GB (CC BY 4.0)
    .venv/Scripts/python scripts/download_data.py models       # faster-whisper small · large-v3-turbo · large-v3

받은 파일은 데이터로만 쓴다(실행하지 않음). 압축 풀기는 경로 밖으로 나가는 항목을 막는 'data' 필터로 한다.
"""

from __future__ import annotations

import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
E21_API = "https://api.github.com/repos/revdotcom/speech-datasets/contents/earnings21/{}"
E21_DIRS = ("media", "transcripts/nlp_references", "transcripts/wer_tags", "bias_lists")
E21_FILES = ("earnings21-file-metadata.csv", "eval10-file-metadata.csv", "speaker-metadata.csv", "LICENSE.md")
ZEROTH_URL = "https://www.openslr.org/resources/40/zeroth_korean.tar.gz"
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
    out = DATA / "zeroth"
    archive = out / "zeroth_korean.tar.gz"
    fetch(ZEROTH_URL, archive)
    print("압축 풀기", flush=True)
    with tarfile.open(archive) as t:
        t.extractall(out, filter="data")
    print("zeroth 끝", flush=True)


def models() -> None:
    from faster_whisper.utils import download_model

    for size in MODEL_SIZES:
        path = download_model(size, cache_dir=str(MODELS))
        print(f"{size}: {path}", flush=True)


if __name__ == "__main__":
    {"earnings21": earnings21, "zeroth": zeroth, "models": models}[sys.argv[1]]()
