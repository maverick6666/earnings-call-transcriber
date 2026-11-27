"""LLM 호출. 어떤 LLM · 키를 쓸지는 사용자 결정이라 얇게 둔다(표준 라이브러리만, 추가 설치 없음).

- AnthropicLLM: Messages API(POST /v1/messages). 키는 ANTHROPIC_API_KEY 환경 변수. 429 · 5xx 는 잠깐 쉬고 다시.
  최신 모델은 생각(thinking) 블록을 함께 돌려줄 수 있어 text 블록만 모은다. temperature 는 보내지 않는다
  (모델마다 받는 설정이 달라서) — 같은 입력에 같은 결과는 아래 저장본이 보장한다.
- CachedLLM: (모델 이름, 지시문, 입력)의 해시로 답을 저장한다. 다시 돌리면 호출 없이 같은 답(불변조건 4 · 5).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


@dataclass
class AnthropicLLM:
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 4096
    api_key: str | None = None
    version: str = "2023-06-01"
    url: str = "https://api.anthropic.com/v1/messages"
    timeout: float = 180.0
    retries: int = 4

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def complete(self, system: str, user: str) -> str:
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY 가 없다")
        body = json.dumps({"model": self.model, "max_tokens": self.max_tokens, "system": system,
                           "messages": [{"role": "user", "content": user}]}).encode()
        headers = {"x-api-key": key, "anthropic-version": self.version, "content-type": "application/json"}
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.load(r)
                return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 529) and attempt < self.retries:
                    time.sleep(min(60.0, 2.0 ** attempt * 2))
                    continue
                raise RuntimeError(f"LLM 호출 실패 {e.code}: {e.read()[:300]!r}") from e
            except (urllib.error.URLError, TimeoutError):
                if attempt < self.retries:
                    time.sleep(min(60.0, 2.0 ** attempt * 2))
                    continue
                raise
        raise RuntimeError("LLM 호출 실패")


@dataclass
class CachedLLM:
    inner: LLM
    cache_dir: Path
    hits: int = 0
    calls: int = 0

    @property
    def name(self) -> str:
        return self.inner.name

    def complete(self, system: str, user: str) -> str:
        h = hashlib.sha256(json.dumps([self.inner.name, system, user], ensure_ascii=False).encode()).hexdigest()
        path = Path(self.cache_dir) / h[:2] / f"{h}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        text = self.inner.complete(system, user)
        self.calls += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"model": self.inner.name, "text": text}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return text
