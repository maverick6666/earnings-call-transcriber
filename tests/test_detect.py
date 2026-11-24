"""검수 0층: 두 번째 창 일치 · 특징 · 라벨 · 점수 모델 · 예산별로 잡는 비율."""

import numpy as np

from callscribe import chunking, merge
from callscribe.detect import (
    FEATURES,
    CallData,
    RiskModel,
    alt_support,
    catch_at_time_budget,
    catch_at_word_budget,
    features,
    leave_one_call_out,
    rank_auc,
    review_seconds,
    second_window_support,
    summarize,
    word_labels,
)
from callscribe.normalize import normalize_en
from callscribe.simulate import FakeASR, make_call
from callscribe.words import RefToken, Word


def test_second_window_support_on_overlapped_windows():
    chunks = chunking.overlapped(60.0, 30.0)  # 0-30, 15-45, 30-60
    a = [Word("revenue", 20.0, 20.4, 0.9), Word("grew", 25.0, 25.3, 0.9)]
    b = [Word("revenue", 20.2, 20.6, 0.8), Word("crew", 25.0, 25.3, 0.6), Word("strongly", 40.0, 40.5, 0.9)]
    c = [Word("strongly", 40.1, 40.5, 0.9)]
    results = [(chunks[0], a), (chunks[1], b), (chunks[2], c)]
    merged = merge.merge_own(results)
    sup = dict(zip([w.text for w in merged], second_window_support(merged, results), strict=True))
    # 20초 · 25초는 창 0(가운데 쪽)과 창 1이 함께 받아 적었다: revenue 는 같고 grew/crew 는 다르다
    assert sup["revenue"] is True and sup["crew" if "crew" in sup else "grew"] is False
    assert sup["strongly"] is True


def test_second_window_missing_at_recording_start():
    chunks = chunking.overlapped(60.0, 30.0)
    results = [(chunks[0], [Word("good", 1.0, 1.3)]), (chunks[1], []), (chunks[2], [])]
    merged = merge.merge_own(results)
    assert second_window_support(merged, results) == [None]  # 처음 15초는 창 하나만 받아 적는다


def test_alt_support_by_time_and_word():
    words = [Word("Nvidia", 1.0, 1.5), Word("grew", 2.0, 2.3)]
    alt = [Word("in", 0.9, 1.1), Word("video", 1.1, 1.5), Word("grew", 2.1, 2.4)]
    assert alt_support(words, alt) == [False, True]
    assert alt_support(words, None) == [None, None]


def test_features_shape_and_lexical_signals():
    words = [Word("Our", 0, 0.3, 0.9), Word("HBM3E", 0.3, 0.8, 0.4), Word("sales.", 0.8, 1.2, 0.95),
             Word("Hynics", 1.3, 1.7, 0.5), Word("Hynics", 1.7, 2.0, 0.5)]
    x = features(words, glossary=["SK Hynix", "HBM3E"])
    col = {n: k for k, n in enumerate(FEATURES)}
    assert x.shape == (5, len(FEATURES))
    assert x[1, col["has_digit"]] == 1 and x[1, col["in_glossary"]] == 1 and x[1, col["capital_mid"]] == 1
    assert x[0, col["capital_mid"]] == 0  # 문장 첫 낱말
    assert x[3, col["capital_mid"]] == 0  # 앞 낱말이 마침표로 끝남
    assert x[3, col["near_glossary"]] == 1 and x[3, col["in_glossary"]] == 0
    assert x[4, col["repeat"]] == 1
    assert abs(x[1, col["risk_prob"]] - 0.6) < 1e-9 and abs(x[2, col["risk_prob_nb"]] - 0.6) < 1e-9


def test_word_labels_mark_substitutions_and_insertions():
    ref = [RefToken(t) for t in ["nvidia", "revenue", "grew"]]
    ref[0] = RefToken("Nvidia", tags=("ORG",))
    hyp = [Word("in", 0, 1), Word("video", 1, 2), Word("revenue", 2, 3), Word("--", 3, 3.1), Word("grew", 3, 4)]
    lab = word_labels(ref, hyp, normalize_en, {"ORG"})
    assert list(lab.wrong) == [1, 1, 0, -1, 0]  # '--' 는 정규화하면 사라진다 → 평가에서 뺌
    assert lab.tagged[0] or lab.tagged[1]
    assert lab.deletions == 0 and lab.ref_words == 3


def test_rank_auc_matches_definition():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.4, 0.4, 0.2, 0.1])
    # 틀린 낱말-맞은 낱말 짝 6개 중: 0.9 는 셋 다 이김(3), 0.4 는 0.2·0.1 이기고 0.4 와 비김(2.5)
    assert rank_auc(y, s) == round(5.5 / 6, 4)
    assert rank_auc(np.array([1, 1]), np.array([0.1, 0.2])) is None


def test_model_learns_separable_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, len(FEATURES)))
    y = (x[:, 2] + 0.3 * rng.normal(size=2000) > 0.5).astype(float)
    m = RiskModel.fit(x, y)
    assert rank_auc(y, m.score(x)) > 0.95
    assert max(m.weights(), key=lambda k: abs(m.weights()[k])) == FEATURES[2]
    again = RiskModel.from_dict(m.to_dict())
    assert np.allclose(again.score(x), m.score(x))


def test_budgets_and_review_time():
    words = [Word(str(k), float(k), k + 0.5) for k in range(10)]
    y = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
    s = np.array([0, 0, 0.9, 0, 0, 0, 0, 0.8, 0, 0], dtype=float)
    assert catch_at_word_budget(y, s, (0.1, 0.2)) == {"0.1": 0.5, "0.2": 1.0}
    picked = np.zeros(10, dtype=bool)
    picked[[2, 7]] = True
    assert review_seconds(words, picked, pad=1.0, join=0.0) == 2 * 2.5  # 낱말 0.5초 + 앞뒤 1초씩
    # 녹음 10초의 30% = 3초 → 한 구간(2.5초)만 들을 수 있다 → 둘 중 하나
    assert catch_at_time_budget(words, y, s, 10.0, (0.3, 0.6), pad=1.0, join=0.0) == {"0.3": 0.5, "0.6": 1.0}


def test_leave_one_call_out_on_simulated_calls():
    """가짜 모델: 틀린 낱말 확률이 낮다(informative_prob) → 확률만으로도 잘 가르고, 모든 특징을 써도 무너지지 않는다."""
    calls = []
    for seed in range(3):
        tl = make_call(seed, n_words=600)
        asr = FakeASR(tl, seed=seed)
        chunks = chunking.overlapped(tl.duration, 30.0)
        results = [(c, asr.transcribe(c.start, c.end)) for c in chunks]
        words = merge.merge_own(results)
        x = features(words, support=second_window_support(words, results))
        lab = word_labels(tl.words, words, normalize_en, {"ORG", "CARDINAL"})
        calls.append(CallData(str(seed), words, x, lab, tl.duration))
    scores = leave_one_call_out(calls)
    rep = summarize(calls, scores)
    prob_only = summarize(calls, leave_one_call_out(calls, ["risk_prob"]))
    assert rep["auc"] > 0.9 and prob_only["auc"] > 0.9
    assert set(rep["catch_at_time_budget"]) == {"0.1", "0.2", "0.3"}
    assert rep["errors"] > 0 and len(rep["auc_per_call"]) == 3
