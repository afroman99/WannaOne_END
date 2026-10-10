# 연속소둔로(CAL) 이상 구간 탐지

## 1. 프로젝트 개요

### 주제
**강종이 바뀌어도 정상의 기준은 같은가?  
공정 조건 변화와 설비 이상을 구분할 수 있는가?**

연속소둔로(CAL)의 온도 센서 데이터를 기반으로 강종, LOT, 반복 공정 Cycle, 정기수리 등 공정 Context를 고려하여 정상 패턴을 정의하고, 정상적인 공정 변화로 설명되지 않는 이상 구간 후보를 탐지하는 것을 목표로 한다.

본 데이터는 KAMP 공개 제조AI 데이터셋을 원본으로 제철 공정 맥락에 맞게 재구성한 교육용 재현 데이터이며, 특정 기업의 실제 조업 데이터가 아니다.

---

## 2. 사용 데이터

현재 분석에는 다음 3개 데이터를 사용하였다.

- `T-CR1-CAL01_온도.csv`
  - 연속소둔로(CAL) 22개 센서 시계열
  - `MEAS_DT`, `LOT_NO` 및 온도/조업 관련 센서 데이터

- `G-02_조업이벤트.csv`
  - 강종변경 등 조업 이벤트
  - LOT와 강종을 연결하기 위한 Context 데이터

- `G-01_정기수리캘린더.csv`
  - 연속소둔로 및 냉연라인 정기수리 일정
  - 계획정비 전후의 공정 변화를 설비 이상으로 오인하지 않기 위한 Context 데이터

현재 단계에서는 실제 고장 여부를 나타내는 정답 Label을 사용하지 않았다.

---

## 3. 분석 흐름

### 3.1 DDA

#### `analysis/dda.py`
데이터 크기, 컬럼, 자료형, 결측치, 중복 데이터 및 기본 구조를 확인하였다.

#### `analysis/dda_time_structure.py`
LOT별 시간 구조를 분석하였다.

- 총 14개 LOT 확인
- LOT 내부는 대부분 1초 간격
- LOT 사이에는 긴 시간 공백 존재
- 전체 데이터를 하나의 연속 시계열로 처리하지 않고 LOT 단위로 분석

#### `analysis/dda_lot_grade.py`
LOT 시작시각과 CR1 강종변경 이벤트를 비교하였다.

14개 LOT의 시작시각이 모두 강종변경 이벤트 시각과 일치하였으며, 조업이벤트의 `AFT_VAL`을 각 LOT의 `STEEL_GRADE`로 연결하였다.

---

## 4. EDA 및 이상 탐지

### 4.1 강종별 정상 수준 분석

`analysis/eda_grade.py`

강종에 따라 일부 센서의 절대 Level이 다르게 나타남을 확인하였다.

따라서 모든 강종에 동일한 고정 임계값을 적용하는 방식은 적절하지 않다고 판단하였다.

---

### 4.2 LOT 간 변동성 분석

`analysis/eda_grade_variability.py`  
`analysis/eda_grade_visual.py`

동일 강종에서도 LOT별 센서 수준 차이가 존재함을 확인하였다.

이를 통해 정상 상태가 단순히 강종 하나만으로 결정되지 않으며, LOT 및 운전조건의 영향도 존재한다고 판단하였다.

---

### 4.3 반복 Cycle 탐색

`analysis/eda_transition_analysis.py`  
`analysis/eda_change_point.py`  
`analysis/eda_cycle_analysis.py`

실제 시간축에서 `TC-OUT1`, `TC-OUT2` 센서의 반복적인 파형을 확인하였다.

Autocorrelation 및 Peak 분석 결과:

- `TC-OUT1` : 약 43~44분의 반복 Cycle
- `TC-OUT2` : 약 47~49분의 반복 Cycle

특히 `TC-OUT1`은 여러 강종에서 약 42~44분의 비교적 안정적인 Cycle이 반복되었다.

---

### 4.4 Cycle Phase 정규화

`analysis/eda_cycle_phase.py`

각 Peak-to-Peak 구간을 하나의 Cycle로 정의하고 0~100% Phase로 정규화하였다.

강종별 절대 Level은 다르게 나타났으나, 각 Cycle의 평균값을 제거한 뒤 비교하면 강종별 Cycle Shape가 유사하게 나타났다.

이를 통해 다음과 같은 정상 구조를 가정하였다.

```text
정상 센서값
≈ 강종별 기준 Level
+ 공통 Cycle Shape
+ LOT/운전조건 영향