"""한국어 실적 콜 평가: 삼성SDS 가 IR 페이지에 올린 녹음 · 스크립트(download_data.py krcalls).

    .venv/Scripts/python scripts/eval_krcalls.py transcribe --model large-v3-turbo
    .venv/Scripts/python scripts/eval_krcalls.py score --model large-v3-turbo
    .venv/Scripts/python scripts/eval_krcalls.py score --model large-v3-turbo --proposer search --claude-bin <claude>
    .venv/Scripts/python scripts/eval_krcalls.py score --strategy O1R --proposer both --claude-bin <claude>
    ... --quarters test --set test8   # 방법을 만들며 보지 않은 분기만(DEV_QUARTERS 를 뺀 것)

- 정답은 회사가 낸 스크립트다. 발표 부분은 거의 말한 그대로지만 질의응답은 다듬어 줄인 글이고, 사회자 안내 ·
  화자 꼬리표는 없다(datasets.read_kr_script). 그래서 글자 오류율(CER)은 발표 부분과 질의응답을 나눠 싣는다.
  고유명사 · 숫자 오류는 정답 글자가 틀리거나 빠진 것만 센다(끼어든 말은 세지 않는다).
- 숫자 꼬리표: 정답에서 숫자 + 붙은 단위(datasets.KR_NUMBER). 고유명사 꼬리표: claude -p(도구 없음)가 정답
  스크립트에서 뽑은 사람 · 회사 · 기관 · 제품 · 지명 중 스크립트에 그대로 있는 것(cache/llm_tag/ 에 저장).
  정답 쪽에만 쓰는 표시라 교정기는 보지 못한다.
- 점수 정규화: 숫자를 한자어 읽기로(normalize_ko_conventional), '%p' 는 '퍼센트포인트', 라틴 글자는 소문자.
  Whisper 는 한국어 숫자를 '3' ',913억' · '5' '.0' '%' 처럼 조각으로 내고 낱말 앞 띄어쓰기를 지워 저장한다 — 조각마다
  읽으면 '삼구백십삼억' 이 되므로 채점 전에 붙인다(glue_number_pieces). 받아쓰기 자체는 바꾸지 않는다.
- --proposer search: 영어와 같은 검색 확인 교정(search_fix)을 그대로 건다. 회사가 올린 스크립트 · 녹취록은 근거에서
  막는다(search_fix.transcript_like). 흔한 낱말 규칙(7)은 지난 통화 묶음이 없어 끈다.
- --proposer materials: 통화 전에 공개된 회사 자료(그 분기 발표자료 · 다시 듣기 페이지의 참석자 명단 · 지난 분기
  스크립트)에 적힌 표기로만 고친다(materials_fix). both: 자료 → 검색 순(같은 old 는 자료 쪽).

결과: runs/krcalls/<모델>/<통화>/O1.json(받아쓰기 기록) · O1.words.json · O1.score.json(채점) · O1+L-<…>.json(교정)
· reports/krcalls_<모델>_<…>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import soundfile as sf

from callscribe import align as align_module
from callscribe import chunking, materials_fix, metrics, runner
from callscribe.align import align
from callscribe.backends import SAMPLE_RATE, FasterWhisperASR, speech_energy
from callscribe.datasets import kr_reference, read_kr_script
from callscribe.entity_fix import Edit, correct_entities, edit_outcomes, key
from callscribe.normalize import normalize_ko_conventional
from callscribe.search_fix import ClaudeSearch, FixedEditsProposer, Proposal, search_and_verify, transcript_urls
from callscribe.words import RefToken, Word

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "krcalls"
RUNS = ROOT / "runs" / "krcalls"
CACHE = ROOT / "cache"
MODELS = ROOT / "models"
COMPANY = {"samsungsds": "Samsung SDS (삼성SDS)"}
COMPANY_KO = {"samsungsds": "삼성SDS"}


def call_hint(folder: Path) -> str:
    """O1R 이 뭉개진 창에 마지막으로 주는 통화 설명. 회사 이름과 분기만(스크립트 내용은 쓰지 않는다)."""
    q = folder.name  # 2025Q3
    return f"{COMPANY_KO.get(folder.parent.name, folder.parent.name)} {q[:4]}년 {q[-1]}분기 실적발표 컨퍼런스콜입니다."


TAG_SYSTEM = "You extract named entities from Korean business text. Follow the user message. Do not use tools."
TAG_PROMPT = """아래는 한국 회사의 실적발표 컨퍼런스콜 스크립트다. 이 글에 나오는 고유명사를 빠짐없이, 글에 적힌 철자 \
그대로 뽑아라.
- 넣을 것: 사람 이름(PERSON), 회사 · 기관 · 정부 부처 · 지자체 · 단체(ORG),
  제품 · 서비스 · 플랫폼 · 브랜드 이름(PRODUCT), 나라 · 도시 · 지명(GPE).
- 뺄 것: 직함과 회사 안 부서 이름(상무, 전략마케팅실, 클라우드서비스사업부), 일반 용어와 일반 약어(클라우드, 생성형 AI,
  매출, AI, MSP, CSP, GPU, ERP).
- 글에 여러 번 나와도 한 번만 적는다. 띄어쓰기 · 대소문자 · 문장부호를 글과 똑같이 적는다. 같은 이름이 글에서 다르게
  적혔으면 둘 다 적는다. 조사는 떼고 이름만 적는다.

JSON 만 답한다: {"entities": [{"text": "...", "type": "PERSON|ORG|PRODUCT|GPE"}]}

스크립트:
"""


# 방법을 만들며 본 통화(개발용). 시험 분기는 이 둘을 빼고, 방법 코드를 고정한 커밋으로 한 번만 돌린다.
DEV_QUARTERS = ("2025Q3", "2025Q4")
_QUARTERS: tuple[str, ...] = ()  # --quarters 로 고른 분기(비면 받아 둔 것 전부)


def calls() -> list[Path]:
    """받아 둔 통화 폴더(회사/분기). 녹음 · 스크립트가 다 있는 것만, --quarters 로 고른 분기만."""
    return sorted(p for p in DATA.glob("*/*") if (p / "audio.flac").exists() and (p / "script.pdf").exists()
                  and (not _QUARTERS or p.name in _QUARTERS))


def call_name(folder: Path) -> str:
    return f"{folder.parent.name}-{folder.name}"


def ko_norm(text: str) -> list[str]:
    return normalize_ko_conventional(text.lower().replace("%p", "%포인트"))


# ---------- 받아쓰기 ----------

def transcribe(args: argparse.Namespace) -> None:
    asr = FasterWhisperASR(model_size=args.model, language="ko", download_root=str(MODELS))
    for folder in calls():
        name = call_name(folder)
        out = RUNS / args.model / name
        if (out / f"{args.strategy}.words.json").exists():
            continue
        audio, sr = sf.read(str(folder / "audio.flac"), dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{name}: {sr} Hz")
        duration = len(audio) / SAMPLE_RATE
        asr.audio = audio
        t0 = time.time()
        res = runner.run(runner.RunConfig(args.strategy), asr, duration, f"krcalls-{name}", cache_dir=CACHE,
                         energy=speech_energy(audio), hint=call_hint(folder))
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{args.strategy}.words.json").write_text(
            json.dumps([w.to_dict() for w in res.words], ensure_ascii=False), encoding="utf-8")
        (out / f"{args.strategy}.json").write_text(json.dumps(
            {"file": name, "strategy": args.strategy, "model": args.model, "duration": duration,
             "elapsed_sec": time.time() - t0, "seams": chunking.seams(res.chunks),
             "chunks": [[c.start, c.end, c.own_start, c.own_end, c.skip] for c in res.chunks],
             "manifest": res.manifest}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name} {args.strategy}: {duration / 60:.1f}분 · {time.time() - t0:.0f}초", flush=True)


# ---------- 정답 ----------

def script_text(folder: Path) -> str:
    return pdf_text(folder / "script.pdf")


def pdf_text(path: Path) -> str:
    from pypdf import PdfReader  # 한국 콜 평가에만 쓰는 선택 의존성

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def materials_text(folder: Path) -> str:
    """교정기가 보는 회사 자료: 그 분기 발표자료 · 참석자 명단 · 지난 분기 스크립트(모두 통화 전에 공개)."""
    parts = [pdf_text(folder / "deck.pdf")]
    for name in ("schedule.txt", "previous_script.pdf"):
        path = folder / "materials" / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8") if path.suffix == ".txt" else pdf_text(path))
    return "\n\n".join(parts)


def script_paragraphs(folder: Path) -> list[tuple[str, str]]:
    return read_kr_script(script_text(folder))


def _squash(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def edits_in_script(edit_log: list[dict], raw_script: str) -> dict:
    """적용한 바꿈마다 '고친 말이 회사 스크립트(화자 꼬리표 포함) 어딘가에 그대로 있나'. 질의응답은 스크립트가 줄인
    글이고 사회자 말이 없어 자리별 채점(edit_outcomes)이 고친 곳을 대부분 '여전히 틀림' 으로 센다 — 그 빈틈을 메우는
    보조 숫자. 띄어쓰기 · 대소문자 · 문장부호는 무시한다. 회사가 라틴 글자로 적은 이름을 한글로 고친 것은
    없다고 나온다."""
    raw = _squash(raw_script)
    applied = list(dict.fromkeys((e["old"], e["new"]) for e in edit_log if e["decision"] == "ok"))
    return {"applied_pairs": len(applied), "new_in_script": sum(_squash(n) in raw for _, n in applied),
            "old_in_script": sum(_squash(o) in raw for o, _ in applied)}


def script_entities(paragraphs: list[tuple[str, str]], tagger: ClaudeSearch) -> dict[str, str]:
    """정답 스크립트의 고유명사 {글: 종류}. 스크립트에 글자 그대로 없는 답은 버린다."""
    body = "\n".join(text for _, text in paragraphs)
    trace = tagger.run(TAG_PROMPT + body)
    raw = trace.text.strip()
    start, end = raw.find("{"), raw.rfind("}")
    items = json.loads(raw[start:end + 1]).get("entities", []) if start >= 0 else []
    out: dict[str, str] = {}
    for item in items:
        text, kind = str(item.get("text", "")).strip(), str(item.get("type", "")).upper()
        if text and text in body and kind in {"PERSON", "ORG", "PRODUCT", "GPE"}:
            out.setdefault(text, kind)
    return out


# ---------- 점수 ----------

_NUMBER_PIECE = re.compile(r"[,.]\d|%")


def glue_number_pieces(words: list[Word]) -> list[Word]:
    """'3' ',913억' → '3,913억', '5' '.0' '%' → '5.0%' (숫자로 끝난 낱말 뒤의 ',숫자' · '.숫자' · '%' 조각)."""
    out: list[Word] = []
    for w in words:
        if out and _NUMBER_PIECE.match(w.text) and out[-1].text[-1:].isdigit():
            out[-1] = replace(out[-1], text=out[-1].text + w.text, end=w.end)
        else:
            out.append(w)
    return out


def sections(ref: list[RefToken], words: list[Word]) -> dict:
    """발표 · 질의응답 부분별 CER · 고유명사 · 숫자 오류. CER 은 끼어든 글자를 바로 앞 정답 글자의 부분으로 세고,
    고유명사 · 숫자는 꼬리표 달린 정답 글자가 틀리거나 빠진 것만 센다(metrics.tagged_error 와 같은 셈)."""
    ref_t = [(ch, "qa" if "QA" in r.tags else "talk", set(r.tags)) for r in ref for ch in ko_norm(r.text)]
    hyp_t = [ch for w in words for ch in ko_norm(w.text)]
    count = {part: {g: [0, 0] for g in ("cer", "entity", "number")} for part in ("talk", "qa")}
    part = "talk"
    for op, i, _ in align([c for c, _, _ in ref_t], hyp_t):
        if i is not None:
            part, tags = ref_t[i][1], ref_t[i][2]
            count[part]["cer"][0] += 1
            for g, hit in (("entity", tags & metrics.ENTITY_TAGS), ("number", tags & metrics.NUMBER_TAGS)):
                if hit:
                    count[part][g][0] += 1
                    count[part][g][1] += op in ("S", "D")
        count[part]["cer"][1] += op != "C"
    return {part: {g: {"ref": n, "errors": e, "rate": round(e / n, 5) if n else None} for g, (n, e) in gs.items()}
            for part, gs in count.items()}


def scores(ref: list[RefToken], words: list[Word]) -> dict:
    words = glue_number_pieces(words)
    ev = metrics.evaluate(ref, words, normalize=ko_norm)
    return {"overall": ev["overall"], "tagged": ev["tagged"], "sections": sections(ref, words)}


def score(args: argparse.Namespace) -> None:
    # 정답(회사 스크립트)과 받아쓰기 길이 차이가 커서(사회자 말 · 줄인 질의응답) 대각선 띠 정렬이 엇나갈 수 있다
    # (B1 2025Q3: 띠 오류 9,381 · 전체 6,479). 글자 2만 개 안팎이라 전체 정렬로 센다(통화당 수십 초 · 수백 MB).
    align_module.FULL_LIMIT = 10**9
    runs = RUNS / args.model
    claude = {"binary": args.claude_bin, "model": args.llm_model}
    tagger = ClaudeSearch(**claude, tools="", system_prompt=TAG_SYSTEM, cache_dir=CACHE / "llm_tag")
    use_search, use_materials = args.proposer in ("search", "both"), args.proposer in ("materials", "both")
    search = ClaudeSearch(**claude, cache_dir=CACHE / "llm_search") if use_search else None
    namer = ClaudeSearch(**claude, tools="", system_prompt=materials_fix.NAMES_SYSTEM,
                         cache_dir=CACHE / "llm_materials") if use_materials else None
    fixer = ClaudeSearch(**claude, tools="", system_prompt=materials_fix.FIX_SYSTEM,
                         cache_dir=CACHE / "llm_materials") if use_materials else None
    ptag = {"search": f"search-{args.llm_model}-web", "materials": f"materials-{args.llm_model}",
            "both": f"materials+search-{args.llm_model}-web"}.get(args.proposer)
    tag = f"{args.strategy}+L-{ptag}-nocommon" if ptag else args.strategy
    per_call = []
    for folder in calls():
        name = call_name(folder)
        wpath = runs / name / f"{args.strategy}.words.json"
        if not wpath.exists():
            print(f"{name}: 받아쓰기 없음 — transcribe 먼저", flush=True)
            continue
        words = [Word(**w) for w in json.loads(wpath.read_text(encoding="utf-8"))]
        raw_script = script_text(folder)
        paragraphs = read_kr_script(raw_script)
        entities = script_entities(paragraphs, tagger)
        ref = kr_reference(paragraphs, entities)
        rec = {"file": name, "tag": tag, "entities": len(entities),
               "entity_types": _count(entities.values()), "before": scores(ref, words)}
        if ptag:
            edits: list[Proposal] = []  # 확정한 바꿈. 같은 old 는 자료 쪽이 먼저(회사 표기 우선)
            if use_materials:
                materials = materials_text(folder)
                names = materials_fix.names_from_materials(materials, namer)
                checked_m, trace_m = materials_fix.propose(words, names, materials,
                                                           COMPANY_KO.get(folder.parent.name, ""), fixer)
                edits += [p for p, why in checked_m if why == "ok"]
                rec["materials"] = {"names": len(names), "confirmed": len(edits),
                                    "proposals": [{"old": p.old, "new": p.new, "check": why} for p, why in checked_m],
                                    "checks": _count(why for _, why in checked_m),
                                    "cost_usd": (trace_m.cost_usd or 0) if trace_m else 0}
            if use_search:
                rec["search"], ok = _search(search, words, folder)
                taken = {key(p.old) for p in edits}
                edits += [p for p in ok if key(p.old) not in taken]
            edits = list(dict.fromkeys(edits))
            proposer = FixedEditsProposer(list(dict.fromkeys(Edit(p.old, p.new) for p in edits)))
            after, rep = correct_entities(words, list(dict.fromkeys(p.new for p in edits)), proposer)
            rec |= {"after": scores(ref, after),
                    "outcome": edit_outcomes(ref, words, after, rep["mapping"], ko_norm),
                    "edits_in_script": edits_in_script(rep["edits"], raw_script),
                    "correction": {k: v for k, v in rep.items() if k != "mapping"}}
            (runs / name / f"{tag}.words.json").write_text(
                json.dumps([w.to_dict() for w in after], ensure_ascii=False), encoding="utf-8")
        # 교정 없는 채점은 O1.score.json — O1.json 은 받아쓰기 기록(창 · 이음매)이라 덮지 않는다
        out = runs / name / (f"{tag}.json" if ptag else f"{tag}.score.json")
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        per_call.append(rec)
        print(_line(name, rec), flush=True)
    if not per_call:
        sys.exit("평가한 통화가 없다")
    _report(args, tag, per_call, search, use_materials)


def _search(search: ClaudeSearch, words: list[Word], folder: Path) -> tuple[dict, list[Proposal]]:
    company = COMPANY.get(folder.parent.name, folder.parent.name)
    rounds = search_and_verify(search, words, company, when=f"{folder.name} earnings conference call, spoken in Korean")
    checked = [(n, p, why) for n, rd in enumerate(rounds, 1) for p, why in rd.checked]
    ok = list(dict.fromkeys(p for _, p, why in checked if why == "ok"))
    traces = [rd.trace for rd in rounds]
    return {"proposals": [{"round": n, "old": p.old, "new": p.new, "url": p.url, "evidence": p.evidence,
                           "check": why} for n, p, why in checked],
            "checks": _count(why for n, _, why in checked if n == 1),
            "retry_checks": _count(why for n, _, why in checked if n == 2),
            "confirmed": len(ok),
            "tool_calls": _count(c["name"] for t in traces for c in t.calls),
            "transcript_urls": list(dict.fromkeys(u for t in traces for u in transcript_urls(t))),
            "cost_usd": sum(t.cost_usd or 0 for t in traces)}, ok


def _line(name: str, rec: dict) -> str:
    b = rec["before"]
    talk = b["sections"]["talk"]
    line = (f"{name}: 발표 부분 CER {talk['cer']['rate']} · 고유명사 {talk['entity']['rate']} · 숫자 "
            f"{talk['number']['rate']} (전체 고유명사 {b['tagged']['entity']['rate']} · 숫자 "
            f"{b['tagged']['number']['rate']})")
    if "after" in rec:
        a, o = rec["after"], rec["outcome"]
        line += (f" → 교정 뒤 발표 CER {a['sections']['talk']['cer']['rate']} · 고유명사 "
                 f"{a['sections']['talk']['entity']['rate']} (전체 {a['tagged']['entity']['rate']}) · "
                 f"바꿈 {o['applied']} (고침 {o['fixed']} · 망침 {o['broke']})")
        for src, label in (("materials", "자료"), ("search", "검색")):
            if src in rec:
                line += f" · {label} 제안 {len(rec[src]['proposals'])} → 확정 {rec[src]['confirmed']}"
    return line


def _report(args: argparse.Namespace, tag: str, per_call: list[dict], search: ClaudeSearch | None,
            use_materials: bool) -> None:
    """묶음 숫자(이름 · 주소 · 글은 빼고 — 통화별 기록에만 남긴다)."""
    def pooled(side: str, get) -> float | None:
        recs = [get(r[side]) for r in per_call if side in r]
        n = sum(x["ref"] for x in recs)
        return round(sum(x.get("errors", x.get("S", 0) + x.get("D", 0) + x.get("I", 0)) for x in recs) / n, 5) \
            if n else None

    corrected = all("after" in r for r in per_call)
    sides = ("before", "after") if corrected else ("before",)
    summary = {
        "tag": tag, "model": args.model, "calls": [r["file"] for r in per_call],
        "cer": [pooled(s, lambda x: x["overall"]) for s in sides],
        "cer_talk": [pooled(s, lambda x: x["sections"]["talk"]["cer"]) for s in sides],
        "cer_qa": [pooled(s, lambda x: x["sections"]["qa"]["cer"]) for s in sides],
        "entity_error_talk": [pooled(s, lambda x: x["sections"]["talk"]["entity"]) for s in sides],
        "number_error_talk": [pooled(s, lambda x: x["sections"]["talk"]["number"]) for s in sides],
        "entity_error": [pooled(s, lambda x: x["tagged"]["entity"]) for s in sides],
        "number_error": [pooled(s, lambda x: x["tagged"]["number"]) for s in sides],
        "reference_entities": sum(r["entities"] for r in per_call),
    }
    if corrected:
        outs = [r["outcome"] for r in per_call]
        summary["outcome"] = {k: sum(o[k] for o in outs) for k in ("fixed", "broke", "still_wrong", "neutral")}
        summary["outcome"]["applied"] = sum(o["applied"] for o in outs)
        summary["edits_in_script"] = _sum_dicts(r["edits_in_script"] for r in per_call)
        summary["decisions"] = _sum_dicts(r["correction"]["decisions"] for r in per_call)
    if use_materials:
        summary["materials"] = {"names": sum(r["materials"]["names"] for r in per_call),
                                "proposals": sum(len(r["materials"]["proposals"]) for r in per_call),
                                "confirmed": sum(r["materials"]["confirmed"] for r in per_call),
                                "checks": _sum_dicts(r["materials"]["checks"] for r in per_call),
                                "cost_usd": round(sum(r["materials"]["cost_usd"] for r in per_call), 2)}
    if search:
        summary["search"] = {"model": search.name, "sessions": search.calls, "cache_hits": search.hits,
                             "proposals": sum(sum(r["search"]["checks"].values()) for r in per_call),
                             "confirmed": sum(r["search"]["confirmed"] for r in per_call),
                             "checks": _sum_dicts(r["search"]["checks"] for r in per_call),
                             "retry_checks": _sum_dicts(r["search"]["retry_checks"] for r in per_call),
                             "calls_that_saw_scripts": sum(bool(r["search"]["transcript_urls"]) for r in per_call),
                             "cost_usd": round(sum(r["search"]["cost_usd"] for r in per_call), 2)}
    (ROOT / "reports").mkdir(exist_ok=True)
    group = args.set or f"{len(per_call)}calls"
    summary["set"] = group
    path = ROOT / "reports" / f"krcalls_{args.model}_{group}_{tag}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def _sum_dicts(ds) -> dict:
    out: dict[str, int] = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("transcribe", "score"))
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--strategy", default="O1")
    ap.add_argument("--proposer", choices=("none", "search", "materials", "both"), default="none")
    ap.add_argument("--llm-model", default="claude-sonnet-5-5")
    ap.add_argument("--claude-bin", default=os.environ.get("CLAUDE_BIN", "claude"))
    ap.add_argument("--quarters", default="", help="쉼표로 이은 분기 · 'dev'(방법을 만들며 본 2통화) · 'test'(나머지)")
    ap.add_argument("--set", default="", help="결과 파일 이름에 넣을 묶음 이름(기본: 'N calls')")
    args = ap.parse_args()
    global _QUARTERS
    every = sorted(p.name for p in DATA.glob("*/*") if (p / "audio.flac").exists())
    _QUARTERS = {"dev": DEV_QUARTERS, "test": tuple(q for q in every if q not in DEV_QUARTERS)}.get(
        args.quarters, tuple(q for q in args.quarters.split(",") if q))
    (transcribe if args.step == "transcribe" else score)(args)


if __name__ == "__main__":
    main()
