# 🎧 어닝콜 받아쓰기 — 청크 경계에서 사라지는 낱말을 숫자로 잡고 없애기

[![ci](https://github.com/maverick6666/earnings-call-transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/maverick6666/earnings-call-transcriber/actions/workflows/ci.yml)

긴 녹음을 청크로 잘라 받아 적으면 **잘린 자리의 낱말이 사라지거나 틀린다.** 강의 영상 받아쓰기 도구(wspr)를 만들며 겪은 문제다.
이 저장소는 그 손실을 **직접 재는 평가 틀**을 만들고, 줄이는 방법 여섯 가지를 **실제 모델 3개 · 실제 어닝콜 · 한국어**에서
같은 틀로 비교했다.

## 결과

**겹쳐 자르기(O1)** — 30초 창을 15초씩 겹쳐 1+2, 2+3 … 으로 받아 적고, 겹친 구간의 가운데에서 잇는 방식 — 가 경계 손실을 없앴다.

| 방식 | 영어 어닝콜 WER | 자른 곳 ±2초 안 | 그 밖 | 한국어 CER | 받아 적은 소리 |
|---|---:|---:|---:|---:|---:|
| B0 60초 고정(원래 방식) | 12.2% | **38.4%** | 10.4% | 9.1% | 1.0배 |
| B1 모델에 통째로 맡기기 | 10.7% | – | – | 6.9% | 1.0배 |
| **O1 겹쳐 자르기** | **10.35%** | **10.0%** | 10.5% | **5.9%** | 2.0배 |
| O2 겹쳐 자르기 + 투표 | 10.35% | 10.2% | 10.4% | 6.5% | 2.0배 |
| V1 무음 건너뛰기 | 17.3% | 29.5% | 13.9% | 8.8% | 0.9배 |
| V2 무음 건너뛰기 + 1초 겹침 | 17.6% | 24.5% | 14.4% | 7.3% | 1.0배 |

large-v3-turbo · Earnings-21 Eval-10(실제 어닝콜 11건 · 10.3시간) · Zeroth-Korean(1.35시간). 모델별 · 칸별 전체 표와 설명은
[**평가 보고서**](reports/real_results.md).

- **경계 손실은 실제로 크다.** 60초 고정은 자른 곳 ±2초 안 오류가 그 밖의 3.7배(한국어 8배)다.
- **겹쳐 자르기는 모든 모델 · 언어에서 이 차이를 없앤다.** 60초 고정보다 영어 33통화 중 32, 한국어 20개 중 18에서 낮았다(모델 3개 · 2개).
- **큰 모델에 통째로 맡기면 되풀이 고리에 빠진다.** large-v3 는 한 통화에서 16분 동안 같은 문장을 37번 적었다(WER 17.0%).
  겹쳐 자르기는 창마다 따로 받아 적어 고리가 이어지지 않는다(10.3%, 지어낸 문장 시간당 27.6 → 1.0).
- **투표는 이득이 없었다.** 실제 단어 확률이 맞고 틀림을 충분히 가르지 못한다(AUC 0.72~0.82).
- **무음 건너뛰기는 말을 조용히 놓친다.** 11통화 중 3통화에서 말소리 감지가 실제 말을 건너뛰었다(최대 녹음의 45%).
- **용어집(hotwords)은 정확할 때만 효과가 있다.** 고유명사 오류 18.7% → 16.1%, 엉뚱한 이름이 섞이면 18.1%.

→ 파이프라인 기본값은 **large-v3-turbo + O1**. large-v3 와 정확도가 같고 시간은 3분의 1 이다.

## 무엇을 재는가

| 숫자 | 뜻 |
|---|---|
| WER / CER | 전체 오류율(영어 낱말 / 한국어 글자). 영어는 Whisper 관례대로 군말(uh · um)을 지우고 센다 — 원래 정규화 숫자도 함께 싣는다 |
| **경계 오류율** | 청크를 자른 시각 ±2초 안 정답 낱말의 오류율 — 이 문제를 직접 재는 숫자 |
| 고유명사 · 숫자 오류율 | Earnings-21 꼬리표(ORG · PRODUCT · CARDINAL · MONEY …) 낱말 중 틀린 비율 — 어닝콜에서 가장 비싼 오류 |
| 지어낸 문장 | 정답에 없는 낱말이 4개 이상 이어진 덩어리(시간당) |
| 단어 확률 AUC | 모델이 낸 단어 확률이 맞은 낱말과 틀린 낱말을 얼마나 가르나 — 투표가 통할지 미리 알려 주는 숫자 |
| 받아 적은 소리 | 녹음 길이 대비 몇 배를 모델에 넣었나(겹치면 계산이 는다) |

모든 평가는 먼저 **가짜 모델 시뮬레이션**([보고서](reports/simulation.md))으로 점수 계산기가 경계 손실을 잡는지 확인한 뒤
진짜 모델로 돌렸다.

## 지키는 규칙

1. 녹음의 모든 시간 구간은 결과에 **정확히 한 번** 반영된다 — 청크 계획마다 검사(`chunking.check_partition`).
2. 결과의 모든 낱말은 녹음 시각을 가진다. 교정된 낱말은 원래 글을 `original` 에 남긴다.
3. 교정은 정렬 검사를 통과해야 한다. 원래 받아쓰기에 없는 3낱말 넘는 덧붙임은 버린다(`correction.apply_correction`).
4. 같은 녹음 + 같은 설정 → 같은 결과. 모델 버전 · 설정 · 청크 계획을 실행 기록에 남긴다.
5. 중간에 멈췄다 다시 돌리면 끝난 청크는 다시 받아 적지 않는다. 저장본 열쇠는 녹음 + 모델 버전 + 설정이라 O1 · O2 가
   같은 받아쓰기를 함께 쓴다.

## 구조

```text
src/callscribe/
  words.py       낱말 = 글 + 시각 (+ 확률 · 청크 · 투표 일치 · 교정 전 글)
  normalize.py   영어 낱말(원래 · 관례) / 한국어 글자 정규화
  align.py       편집 거리 정렬(긴 글은 띠 정렬, 경로가 띠 가장자리에 닿으면 넓혀 다시)
  metrics.py     WER · 경계 오류율 · 고유명사/숫자 · 지어낸 문장 · 단어 확률 AUC
  chunking.py    B0 · B1 · O1/O2 · V1/V2 청크 계획 + '정확히 한 번 덮기' 검사
  merge.py       가운데에서 잇기 · 겹친 구간 투표(같은 자리 · 같은 낱말만 짝으로)
  runner.py      실행 · 저장본으로 이어 하기 · 실행 기록
  correction.py  +L 교정 검사(지어낸 문장 거르기)
  glossary.py    +P 용어집 · hotwords · 용어 재현율
  datasets.py    Earnings-21(.nlp · wer_tag.json) · Zeroth-Korean 읽기, 긴 한국어 오디오 만들기
  backends.py    faster-whisper 연결 · 말소리 감지
  simulate.py    가짜 통화 · 가짜 모델
scripts/
  download_data.py   평가 자료 · 모델 내려받기
  eval_earnings21.py eval_zeroth.py   실제 평가(방식 × 모델)
  rescore.py summarize.py per_file.py make_tables.py   다시 채점 · 묶어 세기 · 통화별 표 · 보고서 표
  simulate_compare.py   가짜 모델 비교
tests/   정렬 · 점수 · 청크 · 잇기 · 실행 · 자료 읽기 · 교정 · 용어집 (45개, 모델 없이 돈다)
```

## 실행

```bash
pip install -e ".[dev]"
python -m pytest                                  # 모델 · 녹음 없이

pip install -e ".[asr]"                           # Windows 에서 CUDA 가 따로 없으면 ".[asr,cuda]"
python scripts/download_data.py earnings21        # 실제 어닝콜 44건 · 734MB
python scripts/download_data.py zeroth            # 한국어 시험 묶음 60MB
python scripts/download_data.py models            # small · large-v3-turbo · large-v3
python scripts/eval_earnings21.py --model large-v3-turbo --strategies B0,B1,O1,O2,V1,V2
python scripts/eval_zeroth.py --model large-v3-turbo --strategies B0,B1,O1,O2,V1,V2
python scripts/rescore.py && python scripts/summarize.py
```

평가는 faster-whisper 1.2.1 · RTX 5070 으로 했다. 평가 자료 · 녹음 · 받아쓰기 결과 · 모델은 저장소에 넣지 않는다
(Earnings-21 은 CC BY-SA 4.0, Zeroth-Korean 은 CC BY 4.0 — 숫자 결과만 싣는다).

## 한계

- 한국어는 낭독 문장을 이어 붙인 자료다. 공개된 한국어 어닝콜 정답 자료가 없다.
- +L(LLM 교정)은 검사 코드만 있고 LLM 으로 돌리지 않았다.
- 창 길이 · 겹침은 고르지 않고 처음 정한 값(30초 · 15초)을 썼다. 시험 자료를 보고 고친 것은 투표(O2) 하나이고 보고서에 적었다.
