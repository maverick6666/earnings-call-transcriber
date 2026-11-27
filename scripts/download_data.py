"""평가 자료 · 모델 내려받기(저장소 안 data/ · models/, 깃에는 올리지 않음).

    .venv/Scripts/python scripts/download_data.py earnings21   # Rev Earnings-21 mp3 44개 + 정답 (CC BY-SA 4.0)
    .venv/Scripts/python scripts/download_data.py zeroth       # Zeroth-Korean 시험 묶음 60MB (CC BY 4.0, HF 사본)
    .venv/Scripts/python scripts/download_data.py models       # faster-whisper small · large-v3-turbo · large-v3
    .venv/Scripts/python scripts/download_data.py krcalls 2025Q3 2025Q4   # 삼성SDS 한국어 실적 콜

받은 파일은 데이터로만 쓴다(실행하지 않음).
삼성SDS 실적 콜(녹음 · 스크립트 · 발표자료)은 회사 IR 페이지에 공개된 것을 평가용으로만 받는다.
녹음 · 스크립트는 저장소에 넣지 않고 숫자 결과만 싣는다.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import quote, urljoin

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


SDS_IR = "https://www.samsungsds.com/kr/investor/ir_events/earnings-release.html"
SDS_HLS = "https://replay.teletogether.com/aod/_definst/mp4:samsungsds/samsungsds{}_kor.m4a/playlist.m3u8"
SDS_REPLAY = "https://irsvc.teletogether.com/samsungsds/samsungsds{}_kor.php"


def sds_links() -> dict[str, dict[str, str]]:
    """IR 페이지의 분기별 링크 {'2025Q3': {'deck': …, 'script': …}}.

    줄마다 '다시 듣기'(주소에 연도 · 분기가 있다) → '발표자료' → '스크립트' 순이라,
    다시 듣기 주소로 분기를 잡고 뒤 링크를 붙인다.
    """
    with urllib.request.urlopen(urllib.request.Request(SDS_IR, headers=UA), timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
    out: dict[str, dict[str, str]] = {}
    quarter = ""
    for href, inner in re.findall(r'<a\b[^>]*?href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
        label = re.sub(r"<[^>]+>|\s+", "", inner)
        if m := re.search(r"samsungsds(\d{4}Q\d)_kor\.php", href):
            quarter = m.group(1)
            out[quarter] = {}
        elif quarter and label in ("발표자료", "스크립트"):
            out[quarter]["deck" if label == "발표자료" else "script"] = quote(urljoin(SDS_IR, href), safe=":/%?=&")
    return out


def save_stream_audio(url: str, dest: Path) -> float:
    """다시 듣기 스트림(HLS)을 16 kHz 모노 FLAC 로 저장하고 길이(초)를 돌려준다."""
    import av
    import soundfile as sf

    if dest.exists() and dest.stat().st_size > 0:
        return sf.info(str(dest)).duration
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    with av.open(url, options={"user_agent": UA["User-Agent"]}) as src, \
            sf.SoundFile(str(tmp), "w", 16000, 1, "PCM_16", format="FLAC") as out:
        for frame in src.decode(src.streams.audio[0]):
            for f in resampler.resample(frame):
                out.write(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):
            out.write(f.to_ndarray().reshape(-1))
    tmp.replace(dest)
    return sf.info(str(dest)).duration


def replay_schedule(quarter: str) -> str:
    """다시 듣기 페이지의 일정 · 발표자 · 참석 임원(글만). 첫 페이지가 쿠키를 심고 실제 페이지로 보낸다."""
    import http.cookiejar
    from html import unescape

    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    first = opener.open(urllib.request.Request(SDS_REPLAY.format(quarter), headers=UA), timeout=60).read()
    m = re.search(r'href = "(/samsungsds/samsungsds\.php\?c=samsungsds&y=\d+)"', first.decode("utf-8", "replace"))
    if not m:
        raise SystemExit(f"{quarter}: 다시 듣기 페이지 모양이 바뀌었다")
    html = opener.open(urllib.request.Request(urljoin(SDS_REPLAY, m.group(1)), headers=UA), timeout=60).read()
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html.decode("utf-8", "replace"), flags=re.S)
    text = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", text)).replace("\u3000", " ").replace("\u2003", " "))
    a, b = text.find("Detailed Schedule"), text.find("MEDIA PLAYER")
    return text[a:b if b > a else None].strip() if a >= 0 else text.strip()


def _previous(quarter: str) -> str:
    year, q = int(quarter[:4]), int(quarter[-1])
    return f"{year - 1}Q4" if q == 1 else f"{year}Q{q - 1}"


def krcalls(*quarters: str) -> None:
    """삼성SDS 실적 콜: data/krcalls/samsungsds/<분기>/ 에 audio.flac · script.pdf(정답) · deck.pdf 와
    materials/ — 통화 전에 공개된 자료(발표자료와 같은 날 · 다시 듣기 페이지의 참석자 명단 · 지난 분기 스크립트).
    교정기는 materials/ 와 deck.pdf 만 본다. script.pdf 는 채점에만 쓴다."""
    links = sds_links()
    scripted = sorted(q for q, v in links.items() if set(v) == {"deck", "script"})  # 분기를 안 주면 이것 전부
    for q in quarters or scripted:
        if q not in links or set(links[q]) != {"deck", "script"}:
            raise SystemExit(f"{q}: IR 페이지에 녹음 · 발표자료 · 스크립트가 다 있지 않다 ({sorted(links.get(q, {}))})")
        out = DATA / "krcalls" / "samsungsds" / q
        fetch(links[q]["script"], out / "script.pdf")
        fetch(links[q]["deck"], out / "deck.pdf")
        sec = save_stream_audio(SDS_HLS.format(q), out / "audio.flac")
        materials = out / "materials"
        materials.mkdir(exist_ok=True)
        if not (materials / "schedule.txt").exists():
            (materials / "schedule.txt").write_text(replay_schedule(q), encoding="utf-8")
        if links.get(_previous(q), {}).get("script"):
            fetch(links[_previous(q)]["script"], materials / "previous_script.pdf")
        (out / "source.json").write_text(json.dumps({"company": "Samsung SDS", "quarter": q, "language": "ko",
                                                     "audio": SDS_HLS.format(q), **links[q]}, indent=1),
                                         encoding="utf-8")
        print(f"{q}: 녹음 {sec / 60:.1f}분 · 스크립트 · 발표자료", flush=True)


if __name__ == "__main__":
    {"earnings21": earnings21, "zeroth": zeroth, "models": models, "krcalls": krcalls}[sys.argv[1]](*sys.argv[2:])
