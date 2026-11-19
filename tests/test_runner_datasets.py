"""실행(재현 · 이어 하기) · 자료 읽기 · 가짜 모델."""

import json

import numpy as np
import pytest

from callscribe import runner, simulate
from callscribe.datasets import build_long_audio, read_nlp, read_trans
from callscribe.words import Word


class CountingASR:
    """시험용: 청크마다 낱말 하나를 내고 몇 번 불렸는지 센다."""
    name, version = "counting", "1"

    def __init__(self):
        self.calls = 0

    def transcribe(self, start, end):
        self.calls += 1
        return [Word(f"t{int(start)}", start + 0.1, start + 0.5, 0.9)]


def test_resume_skips_finished_chunks(tmp_path):
    cfg = runner.RunConfig("B0", fixed_size=10.0)
    first = CountingASR()
    r1 = runner.run(cfg, first, 35.0, "call", cache_dir=tmp_path)
    assert (r1.transcribed, r1.from_cache, first.calls) == (4, 0, 4)
    # 중간에 멈춘 것처럼 저장본 하나를 지우고 다시 돌리면 그 청크만 다시 받아 적는다
    store = next(tmp_path.iterdir())
    sorted(store.glob("*.json"))[1].unlink()
    again = CountingASR()
    r2 = runner.run(cfg, again, 35.0, "call", cache_dir=tmp_path)
    assert (r2.transcribed, r2.from_cache, again.calls) == (1, 3, 1)
    assert r2.manifest["output_hash"] == r1.manifest["output_hash"]


def test_config_change_does_not_reuse_cache(tmp_path):
    runner.run(runner.RunConfig("B0", fixed_size=10.0), CountingASR(), 35.0, "call", cache_dir=tmp_path)
    other = CountingASR()
    runner.run(runner.RunConfig("B0", fixed_size=12.0), other, 35.0, "call", cache_dir=tmp_path)
    assert other.calls == 3


def test_same_input_same_config_same_output():
    tl = simulate.make_call(5, n_words=400, lead_silence=10)
    outs = [runner.run(runner.RunConfig("O2"), simulate.FakeASR(tl, seed=5), tl.duration, "c").manifest
            for _ in range(2)]
    assert outs[0]["output_hash"] == outs[1]["output_hash"] and outs[0]["config_hash"] == outs[1]["config_hash"]


def test_leading_silence_stays_empty_with_vad():
    tl = simulate.make_call(6, n_words=400, lead_silence=40)
    res = runner.run(runner.RunConfig("V1"), simulate.FakeASR(tl, seed=6, p_hallucinate=1.0), tl.duration, "c",
                     energy=tl.energy())
    assert res.chunks[0].skip and all(w.mid > 39.0 for w in res.words)
    fixed = runner.run(runner.RunConfig("B0"), simulate.FakeASR(tl, seed=6, p_hallucinate=1.0), tl.duration, "c")
    assert any(w.text == "thank" and w.mid < 40 for w in fixed.words)  # 자르기만 하면 대기음에서 지어낸 말이 남는다


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        runner.RunConfig("X9")


def test_read_nlp_with_tags_and_sidecar(tmp_path):
    nlp = tmp_path / "1.nlp"
    nlp.write_text("token|speaker|ts|endTs|punctuation|case|tags|wer_tags\n"
                   "Good|0||||UC|[]|[]\n"
                   "morning|0||||LC|['5:TIME']|['5']\n"
                   "2020|0|3.5|3.9||CA|[]|['0', '1']\n", encoding="utf-8")
    tags = tmp_path / "1.wer_tag.json"
    tags.write_text(json.dumps({"0": {"entity_type": "YEAR"}, "1": {"entity_type": "CARDINAL"}}), encoding="utf-8")
    ref = read_nlp(nlp, tags)
    assert [r.text for r in ref] == ["Good", "morning", "2020"]
    assert ref[1].tags == ("TIME",) and ref[2].tags == ("CARDINAL", "YEAR")
    assert ref[0].start is None and (ref[2].start, ref[2].end) == (3.5, 3.9)


def test_read_nlp_rejects_wrong_header(tmp_path):
    p = tmp_path / "x.nlp"
    p.write_text("token|speaker\nGood|0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_nlp(p)


def test_long_audio_has_exact_sentence_times(tmp_path):
    (tmp_path / "1_1.trans.txt").write_text("1_1_0001 매출이 늘었다\n1_1_0002 이익은 줄었다\n", encoding="utf-8")
    texts = read_trans(tmp_path.glob("*.trans.txt"))
    sr = 1000
    utts = [(uid, np.ones(2 * sr, dtype=np.float32), text) for uid, text in sorted(texts.items())]
    audio, ref, spans = build_long_audio(utts, sr, seed=0, lead_silence=1.0)
    assert spans[0][:2] == (1.0, 3.0) and spans[1][0] > 3.0
    assert "".join(r.text for r in ref) == "매출이늘었다이익은줄었다"
    assert ref[0].start == 1.0 and len(audio) >= int(5 * sr)


def test_fake_asr_is_deterministic_and_loses_words_cut_by_edges():
    tl = simulate.make_call(1, n_words=300)
    a = simulate.FakeASR(tl, seed=1, p_fragment=0.0)
    w1, w2 = a.transcribe(10.0, 40.0), a.transcribe(10.0, 40.0)
    assert w1 == w2
    cut = [w for w in tl.words if w.start < 40.0 < w.end]
    if cut:
        assert not any(abs(w.start - cut[0].start) < 1e-9 for w in w1)
