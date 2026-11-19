"""청크 계획(녹음을 정확히 한 번 덮기) · 무음 건너뛰기 · 이어 붙이기."""

import numpy as np
import pytest

from callscribe import chunking
from callscribe.merge import dedupe_seams, merge_own, merge_vote
from callscribe.runner import check_output
from callscribe.words import Word


@pytest.mark.parametrize("duration", [10.0, 59.9, 60.0, 61.0, 1234.5])
def test_fixed_and_overlapped_plans_cover_recording_exactly_once(duration):
    plans = (chunking.fixed(duration, 60), chunking.overlapped(duration, 30), chunking.overlapped(duration, 30, 10))
    for plan in plans:
        assert chunking.check_partition(plan, duration) == []


def test_overlap_seams_are_in_the_middle_of_overlaps():
    plan = chunking.overlapped(100.0, 30.0)
    assert [(c.start, c.end) for c in plan[:3]] == [(0.0, 30.0), (15.0, 45.0), (30.0, 60.0)]
    assert chunking.seams(plan)[:2] == [22.5, 37.5]
    assert plan[-1].end == 100.0  # 마지막 창은 녹음 끝에 맞춰 당긴다


def test_partition_check_catches_gaps_and_overlaps():
    bad = [chunking.Chunk(0, 0, 10, 0, 10), chunking.Chunk(1, 9, 20, 11, 20)]
    assert any("빈틈" in p for p in chunking.check_partition(bad, 20))


def energy(duration, speech, frame=0.03):
    t = (np.arange(int(duration / frame)) + 0.5) * frame
    db = np.full(len(t), -60.0)
    for a, b in speech:
        db[(t >= a) & (t < b)] = -20.0
    return t, db


def test_vad_skips_long_silence_and_cuts_speech_in_pauses():
    # 0~5초 대기음, 5~70초 말(0.2초 쉼이 1초마다), 70~75초 무음, 75~80초 말
    speech = [(5 + k, 5 + k + 0.8) for k in range(65)] + [(75.0, 80.0)]
    t, db = energy(80.0, speech)
    plan = chunking.vad(80.0, t, db, max_len=30.0)
    assert chunking.check_partition(plan, 80.0) == []
    skips = [(round(c.own_start, 1), round(c.own_end, 1)) for c in plan if c.skip]
    assert skips[0][0] == 0.0 and skips[0][1] > 4.0 and any(a >= 69 and b <= 75.5 for a, b in skips[1:])
    for c in plan:
        if not c.skip:
            assert c.end - c.start <= 30.0 + 1e-6
    cuts = [c.own_end for c in plan[:-1]]
    assert all(not any(a < x < b for a, b in speech) for x in cuts)  # 말소리 한가운데서 자르지 않는다


def test_vad_padding_widens_only_transcribed_span():
    speech = [(k * 1.0, k * 1.0 + 0.8) for k in range(70)]
    t, db = energy(70.0, speech)
    v1 = chunking.vad(70.0, t, db, max_len=30.0)
    v2 = chunking.vad(70.0, t, db, max_len=30.0, pad=1.0)
    assert [(c.own_start, c.own_end) for c in v1] == [(c.own_start, c.own_end) for c in v2]
    assert v2[1].start == pytest.approx(v1[1].start - 1.0)


def words(texts_times, chunk=None, prob=0.9):
    return [Word(t, s, s + 0.4, prob, chunk) for t, s in texts_times]


def test_overlap_recovers_word_cut_at_window_edge():
    plan = chunking.overlapped(60.0, 30.0)  # [0,30] [15,45] [30,60], 이음매 22.5 · 37.5
    truth = [(f"w{k}", float(k)) for k in range(60)]
    res = []
    for c in plan:
        # 창 끝 1초 안의 낱말은 그 창에서 사라진다(경계 손실)
        res.append((c, words([(t, s) for t, s in truth if c.start <= s and s + 0.4 <= c.end - 1.0
                              or (c.end == 60.0 and c.start <= s)])))
    merged = merge_own(res)
    assert [w.text for w in merged] == [t for t, _ in truth]


def test_vote_keeps_agreed_words_and_trusts_the_more_central_window():
    a = chunking.Chunk(0, 0.0, 30.0, 0.0, 22.5)
    b = chunking.Chunk(1, 15.0, 45.0, 22.5, 45.0)
    wa = words([("revenue", 16.0), ("rose", 20.0), ("stro", 28.5)], prob=0.6)  # 28.5 는 A 의 끝 가장자리
    wb = words([("revenue", 16.1), ("rose", 20.0), ("strongly", 28.6), ("again", 40.0)], prob=0.9)
    merged = merge_vote([(a, wa), (b, wb)])
    assert [w.text for w in merged] == ["revenue", "rose", "strongly", "again"]
    assert merged[0].agreed and merged[1].agreed


def test_seam_duplicates_are_removed():
    ws = [Word("growth", 22.3, 22.6, 0.8, 0), Word("growth", 22.4, 22.7, 0.9, 1), Word("next", 23.0, 23.3, 0.9, 1)]
    out = dedupe_seams(ws, [22.5])
    assert [w.text for w in out] == ["growth", "next"] and out[0].prob == 0.9


def test_output_check_rejects_words_in_skipped_silence():
    plan = [chunking.Chunk(0, 0, 10, 0, 10, skip=True), chunking.Chunk(1, 10, 20, 10, 20)]
    with pytest.raises(AssertionError):
        check_output([Word("thank", 3.0, 3.4)], plan)
