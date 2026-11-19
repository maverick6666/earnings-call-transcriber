"""단어 하나 = 글자 + 녹음 시각. 결과의 모든 단어는 시각을 가진다(불변조건 2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Word:
    text: str
    start: float  # 초
    end: float
    prob: float | None = None  # 모델이 준 단어 확률(없으면 None)
    chunk: int | None = None  # 어느 청크에서 나왔나
    agreed: bool | None = None  # 겹친 구간에서 두 청크가 같게 받아 적었나(O2)
    original: str | None = None  # 교정(+L)으로 바뀐 낱말이면 원래 받아 적은 글(불변조건 2)

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RefToken:
    """정답 낱말. 시각은 정답에 있을 때만(Earnings-21 은 비어 있을 수 있다), 꼬리표는 고유명사 · 숫자 종류."""
    text: str
    start: float | None = None
    end: float | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)  # 예: ("ORG",), ("CARDINAL",)
