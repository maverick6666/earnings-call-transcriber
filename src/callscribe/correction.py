"""+L: LLM 이 고친 글을 받아들이기 전에 검사한다(불변조건 2 · 3).

LLM 사후 교정은 단어 오류를 크게 줄일 수 있지만(HyPoradise, NeurIPS 2023) 없는 말을 지어낼 수 있다. 그래서
LLM 에게서 받은 글을 원래 받아쓰기와 정렬해
- 바꾼 낱말(S)은 받아들이되 시각은 원래 낱말 것을 쓰고, 원래 글을 `original` 에 남긴다(교정됨 표시).
- 더한 낱말(I)은 짧게(MAX_INSERT_RUN 개 이하) 바꾼 낱말 곁에 붙은 것만 받아들인다 — 'HBM3E' → 'HBM 3E' 같은 쪼개기.
  그보다 긴 덧붙임은 문장 단위 추가로 보고 버린다.
- 지운 낱말(D)은 받아들이되 몇 개를 지웠는지 보고한다(되돌릴 수 있게 원래 결과는 따로 남는다).
LLM 호출 자체는 이 모듈 밖에 둔다(어떤 LLM 을 쓸지 · 키는 사용자 결정).
"""

from __future__ import annotations

from collections.abc import Sequence

from .align import align
from .words import Word

MAX_INSERT_RUN = 2


def _key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def apply_correction(words: Sequence[Word], corrected: str) -> tuple[list[Word], dict]:
    new_tokens = corrected.split()
    ops = align([_key(w.text) for w in words], [_key(t) for t in new_tokens])
    out: list[Word] = []
    report = {"kept": 0, "replaced": 0, "deleted": 0, "inserted": 0, "rejected_insertions": 0, "rejected_runs": []}
    k = 0
    while k < len(ops):
        op, i, j = ops[k]
        if op == "C":
            out.append(words[i])
            report["kept"] += 1
        elif op == "S":
            w = words[i]
            out.append(Word(new_tokens[j], w.start, w.end, w.prob, w.chunk, w.agreed, w.original or w.text))
            report["replaced"] += 1
        elif op == "D":
            report["deleted"] += 1
        else:  # 더한 낱말 덩어리
            run = []
            while k < len(ops) and ops[k][0] == "I":
                run.append(ops[k][2])
                k += 1
            beside_change = (out and out[-1].original is not None) or (k < len(ops) and ops[k][0] == "S")
            if len(run) <= MAX_INSERT_RUN and beside_change:
                prev_end = out[-1].end if out else (words[0].start if words else 0.0)
                nxt = next((words[ops[m][1]].start for m in range(k, len(ops)) if ops[m][1] is not None), prev_end)
                step = max(nxt - prev_end, 0.0) / (len(run) + 1)
                for n, jj in enumerate(run, 1):
                    t = prev_end + step * n
                    out.append(Word(new_tokens[jj], t, t, None, None, None, ""))  # 원래 낱말 없음 = 빈 글
                report["inserted"] += len(run)
            else:
                report["rejected_insertions"] += len(run)
                report["rejected_runs"].append(" ".join(new_tokens[jj] for jj in run))
            continue
        k += 1
    return out, report
