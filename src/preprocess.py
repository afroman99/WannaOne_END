"""연속소둔로(CAL01) EDA에서 결론이 확정된 전처리 함수 모음.

탐색 노트북의 [확정] 셀을 옮긴 것이다. 근거는 analysis/01_eda.py에서
다시 계산할 수 있다.

    from src.preprocess import (load_data, attach_grade, label_operating_state,
                                link_maintenance, find_missing_segments)

    df_cal, df_run_evt_cal, df_m_cal = load_data()
    lot_grade = attach_grade(df_cal, df_run_evt_cal)
    df_cal_labeled = label_operating_state(df_cal, lot_grade)
    lot_maint_summary = link_maintenance(lot_grade, df_m_cal)
    missing_runs = find_missing_segments(df_cal)

데이터 전제(데이터사전): 재구성된 교육용 데이터다. 존 배정과 파일 간
연결은 [가정]이다. 결과는 설비 특성이 아니라 데이터상 패턴으로 읽는다.
"""

from pathlib import Path

import pandas as pd

# CAL01 설비 코드 (정비이력의 EQP_TAG)
CAL_EQP_TAG = 'P1-CR1-CAL01'

# 승온·감온 로트의 비교용 경계 (EDA 8절)
# 1분 온도 흐름을 검토해 승온·감온이 비교 구간에 섞이지 않도록 정했다
#   승온: 경계 이전이 승온 구간
#   감온: 경계 이후가 감온 구간
TRANSITION_BOUNDARIES = {
    240058: ('감온', pd.Timestamp('2024-06-09 07:55:00')),
    240061: ('승온', pd.Timestamp('2024-06-15 02:45:00')),
    240114: ('감온', pd.Timestamp('2024-11-04 20:00:00')),
    240117: ('승온', pd.Timestamp('2024-11-13 08:45:00')),
}

# 감온 경계 앞의 여러 구역 변동 구간 시작 시각 (EDA 8절, 2026-10-07 보완)
# 감온 30~40분 전부터 온도 유지 상태와 다른 변화가 있어 비교 후보에서 뺀다
PRE_COOLING_STARTS = {
    240058: pd.Timestamp('2024-06-09 07:25:00'),
    240114: pd.Timestamp('2024-11-04 19:15:00'),
}

# 결측 구간을 찾을 센서 컬럼의 접두어
SENSOR_PREFIXES = ('TC-', 'OP-', 'CP-', 'CLN-')


def load_data(data_dir='data'):
    """온도 센서, 강종변경 이벤트, CAL 정비이력을 불러온다. (EDA 1절)

    날짜 컬럼은 datetime으로 바꾼다. 센서 데이터는 시간순으로 정렬한다.
    로트 사이나 정기수리 기간은 측정이 없는 구간이다. 채우거나 보간하지 않는다.

    Returns
    -------
    df_cal : 14개 로트 센서 측정값
    df_run_evt_cal : 조업 이벤트 중 강종변경 기록
    df_m_cal : CAL01 설비의 정비이력
    """
    data_dir = Path(data_dir)

    df_cal = pd.read_csv(data_dir / 'T-CR1-CAL01_온도.csv')
    df_cal['MEAS_DT'] = pd.to_datetime(df_cal['MEAS_DT'])
    df_cal = df_cal.sort_values('MEAS_DT')

    df_run_evt = pd.read_csv(data_dir / 'G-02_조업이벤트.csv')
    df_run_evt_cal = df_run_evt.loc[
        df_run_evt['EVT_TYPE'] == '강종변경'
    ].copy()
    df_run_evt_cal['EVT_DT'] = pd.to_datetime(df_run_evt_cal['EVT_DT'])

    df_m = pd.read_csv(data_dir / 'G-02_정비이력.csv')
    df_m_cal = df_m.loc[df_m['EQP_TAG'] == CAL_EQP_TAG].copy()
    for col in ['DETC_DT', 'STRT_DT', 'END_DT']:
        df_m_cal[col] = pd.to_datetime(df_m_cal[col])

    return df_cal, df_run_evt_cal, df_m_cal


def attach_grade(df_cal, df_run_evt_cal):
    """로트별 시작·종료 시각과 강종을 연결한다. (EDA 3절)

    로트 시작 시각과 같거나 그 이전인 가장 최근 CR1 강종변경 기록을 붙인다.
    측정 기간 안에 추가 강종변경이 있는지도 센다. 현재 데이터에서는 14개
    로트 모두 0건이라 로트당 강종 하나로 볼 수 있다.

    Returns
    -------
    lot_grade : 로트당 한 행
        START_GRADE 컬럼이 그 로트의 강종이다.
    """
    cal = df_cal.copy()
    cal['MEAS_DT'] = pd.to_datetime(cal['MEAS_DT'])

    # 로트별 시작·종료 시각
    lot_info = (
        cal.groupby('LOT_NO', as_index=False)
        .agg(
            START_DT=('MEAS_DT', 'min'),
            END_DT=('MEAS_DT', 'max'),
            ROW_COUNT=('MEAS_DT', 'size')
        )
        .sort_values('START_DT')
    )

    # CR1(냉연) 강종변경 기록만 사용
    grade_events = df_run_evt_cal.loc[
        df_run_evt_cal['PLANT_CD'] == 'CR1',
        ['EVT_DT', 'BEF_VAL', 'AFT_VAL']
    ].copy()
    grade_events['EVT_DT'] = pd.to_datetime(grade_events['EVT_DT'])
    grade_events = grade_events.sort_values('EVT_DT')

    # 로트 시작 시각 이전의 가장 최근 강종변경 연결
    lot_grade = pd.merge_asof(
        lot_info,
        grade_events,
        left_on='START_DT',
        right_on='EVT_DT',
        direction='backward'
    ).rename(columns={
        'EVT_DT': 'GRADE_CHANGE_DT',
        'AFT_VAL': 'START_GRADE'
    })

    # 로트 시작 후~종료 사이의 추가 강종변경 수
    lot_grade['CHANGE_COUNT_INSIDE'] = [
        int(
            (
                (grade_events['EVT_DT'] > start)
                & (grade_events['EVT_DT'] <= end)
            ).sum()
        )
        for start, end in zip(lot_grade['START_DT'], lot_grade['END_DT'])
    ]

    return lot_grade


def label_operating_state(df_cal, lot_grade):
    """각 측정 행에 강종과 운전 상태를 붙인다. (EDA 8절)

    OP_STATE 값:
        '온도 유지 중심·비교 후보'
        '승온 구간 추정'        승온 로트의 경계 이전
        '감온 구간 추정'        감온 로트의 경계 이후
        '감온 전 변동 구간'      감온 경계 앞의 여러 구역 변동
        '정보 부족·비교 보류'    강종이나 필요한 경계가 없음

    COMPARE_CANDIDATE=True는 온도 유지 중심의 **비교 후보**라는 뜻이다.
    설비가 정상이라는 판정이 아니다.

    Returns
    -------
    df_cal_labeled : df_cal에 아래 컬럼을 더한 것
        GRADE, TRANSITION_TYPE, REVIEW_BOUNDARY_DT, BOUNDARY_SOURCE,
        PRE_COOLING_START, OP_STATE, COMPARE_CANDIDATE
    """
    # 로트별 강종과 경계 정보
    lot_meta = lot_grade[['LOT_NO', 'START_GRADE']].rename(
        columns={'START_GRADE': 'GRADE'}
    ).copy()

    lot_meta['TRANSITION_TYPE'] = lot_meta['LOT_NO'].map(
        {lot_no: kind for lot_no, (kind, _) in TRANSITION_BOUNDARIES.items()}
    )
    lot_meta['REVIEW_BOUNDARY_DT'] = lot_meta['LOT_NO'].map(
        {lot_no: dt for lot_no, (_, dt) in TRANSITION_BOUNDARIES.items()}
    )
    lot_meta['BOUNDARY_SOURCE'] = '큰 승온·감온 미관찰'
    lot_meta.loc[
        lot_meta['LOT_NO'].isin(TRANSITION_BOUNDARIES), 'BOUNDARY_SOURCE'
    ] = '1분 온도 흐름 검토 후 정한 비교용 경계'
    lot_meta['PRE_COOLING_START'] = lot_meta['LOT_NO'].map(PRE_COOLING_STARTS)

    # 각 측정 행에 연결
    df_cal_labeled = df_cal.copy()
    df_cal_labeled['MEAS_DT'] = pd.to_datetime(df_cal_labeled['MEAS_DT'])
    df_cal_labeled = df_cal_labeled.merge(
        lot_meta, on='LOT_NO', how='left', validate='many_to_one'
    )

    # 기본값: 온도 유지 중심의 비교 후보
    df_cal_labeled['OP_STATE'] = '온도 유지 중심·비교 후보'
    df_cal_labeled['COMPARE_CANDIDATE'] = True

    warming = df_cal_labeled['TRANSITION_TYPE'].eq('승온')
    cooling = df_cal_labeled['TRANSITION_TYPE'].eq('감온')
    time = df_cal_labeled['MEAS_DT']
    boundary = df_cal_labeled['REVIEW_BOUNDARY_DT']

    warming_period = warming & time.lt(boundary)
    cooling_period = cooling & time.ge(boundary)
    # 감온 전 변동 구간: 시작 시각 이상, 감온 경계 미만
    pre_cooling_period = (
        cooling
        & time.ge(df_cal_labeled['PRE_COOLING_START'])
        & time.lt(boundary)
    )

    df_cal_labeled.loc[warming_period, 'OP_STATE'] = '승온 구간 추정'
    df_cal_labeled.loc[cooling_period, 'OP_STATE'] = '감온 구간 추정'
    df_cal_labeled.loc[pre_cooling_period, 'OP_STATE'] = '감온 전 변동 구간'
    df_cal_labeled.loc[
        warming_period | cooling_period | pre_cooling_period,
        'COMPARE_CANDIDATE'
    ] = False

    # 강종이 없거나, 승온·감온 로트인데 경계가 없으면 비교 보류
    unresolved = (
        df_cal_labeled['GRADE'].isna()
        | (warming | cooling) & boundary.isna()
    )
    df_cal_labeled.loc[unresolved, 'OP_STATE'] = '정보 부족·비교 보류'
    df_cal_labeled.loc[unresolved, 'COMPARE_CANDIDATE'] = False

    return df_cal_labeled


def link_maintenance(lot_grade, df_m_cal):
    """로트마다 앞뒤의 정비 기록을 연결한다. (EDA 6절)

    로트당 한 행으로 다음을 정리한다.
        PREV_*      로트 시작 시점까지 완료된 가장 최근 정비
        OVERLAP_*   로트 측정 기간과 작업 시간이 겹치는 정비
                    (경계에서 끝나거나 시작하기만 하는 작업은 제외)
        NEXT_*      로트 종료 이후 가장 먼저 나온 이상 발견 기록
        DETC_INSIDE_*  로트 측정 기간 안의 이상 발견 기록

    주의
    - DETC_DT는 발견·기록 시각이지 문제가 시작된 시각이 아니다.
    - 데이터사전상 고장모드는 배정한 값([가정])이다. 이 결과를 센서 이상의
      정답 라벨로 쓰지 않는다.
    - 부품교체이력을 붙이면 작업 하나에 행이 여러 개 생긴다. 그래서 여기서는
      부품을 붙이기 전의 정비이력을 쓴다.
    """
    lots = lot_grade[['LOT_NO', 'START_DT', 'END_DT', 'START_GRADE']].copy()
    for col in ['START_DT', 'END_DT']:
        lots[col] = pd.to_datetime(lots[col])
    lots = lots.sort_values('START_DT')

    maint = df_m_cal.copy()
    for col in ['DETC_DT', 'STRT_DT', 'END_DT']:
        maint[col] = pd.to_datetime(maint[col])

    # 작업번호가 비거나 중복이면 임의로 합치지 않고 멈춘다
    if maint['WO_NO'].isna().any() or maint['WO_NO'].duplicated().any():
        raise ValueError('df_m_cal의 WO_NO 결측 또는 중복을 먼저 확인하세요.')

    results = []

    for _, lot in lots.iterrows():
        start = lot['START_DT']
        end = lot['END_DT']

        # 로트 시작 시점까지 완료된 정비
        previous = maint.loc[
            maint['END_DT'].notna()
            & (maint['END_DT'] <= start)
        ].sort_values(['END_DT', 'WO_NO'])

        # 로트 측정 구간과 작업 시간이 겹치는 정비
        overlapping = maint.loc[
            maint['STRT_DT'].notna()
            & maint['END_DT'].notna()
            & (maint['STRT_DT'] < end)
            & (maint['END_DT'] > start)
        ].sort_values(['STRT_DT', 'WO_NO'])

        # 로트 종료 이후의 발견 기록
        following = maint.loc[
            maint['DETC_DT'].notna()
            & (maint['DETC_DT'] > end)
        ].sort_values(['DETC_DT', 'WO_NO'])

        # 로트 측정 기간 안의 발견 기록
        detected = maint.loc[
            maint['DETC_DT'].between(start, end)
        ].sort_values('DETC_DT')

        row = {
            'LOT_NO': lot['LOT_NO'],
            'GRADE': lot['START_GRADE'],
            'LOT_START': start,
            'LOT_END': end,

            'PREV_WO_NO': pd.NA,
            'PREV_MNT_TYPE': pd.NA,
            'PREV_MNT_END': pd.NaT,
            'HOURS_SINCE_MNT': float('nan'),

            'OVERLAP_COUNT': len(overlapping),
            'OVERLAP_WO_NO': ', '.join(overlapping['WO_NO'].astype(str)),

            'NEXT_WO_NO': pd.NA,
            'NEXT_DETC_DT': pd.NaT,
            'NEXT_MNT_TYPE': pd.NA,
            'NEXT_FAIL_MODE': pd.NA,
            'NEXT_REMARK': pd.NA,
            'HOURS_TO_NEXT_DETC': float('nan'),

            'DETC_INSIDE_COUNT': len(detected),
            'DETC_INSIDE_WO_NO': ', '.join(detected['WO_NO'].astype(str)),
        }

        if not previous.empty:
            prev = previous.iloc[-1]
            row.update({
                'PREV_WO_NO': prev['WO_NO'],
                'PREV_MNT_TYPE': prev['MNT_TYPE'],
                'PREV_MNT_END': prev['END_DT'],
                'HOURS_SINCE_MNT':
                    (start - prev['END_DT']).total_seconds() / 3600
            })

        if not following.empty:
            nxt = following.iloc[0]
            row.update({
                'NEXT_WO_NO': nxt['WO_NO'],
                'NEXT_DETC_DT': nxt['DETC_DT'],
                'NEXT_MNT_TYPE': nxt['MNT_TYPE'],
                'NEXT_FAIL_MODE': nxt['FAIL_MODE'],
                'NEXT_REMARK': nxt['REMARK'],
                'HOURS_TO_NEXT_DETC':
                    (nxt['DETC_DT'] - end).total_seconds() / 3600
            })

        results.append(row)

    lot_maint_summary = pd.DataFrame(results)

    hour_cols = ['HOURS_SINCE_MNT', 'HOURS_TO_NEXT_DETC']
    lot_maint_summary[hour_cols] = lot_maint_summary[hour_cols].round(2)

    return lot_maint_summary


def find_missing_segments(df_cal):
    """로트·센서별로 연속된 결측을 한 구간으로 묶는다. (EDA 11절)

    현재 데이터에서는 모든 결측 구간이 로트 시작에서 시작한다.
    OP=0은 결측이 아니라 실제 값이므로 여기서 세지 않는다.
    결측 행 수를 초로 바꾸지 않는다. 측정 간격이 항상 1초는 아니다.

    Returns
    -------
    missing_runs : 결측 구간당 한 행
        LOT_NO, SENSOR, MISSING_ROWS, START_DT, END_DT,
        AT_LOT_START, AT_LOT_END
    """
    data = df_cal.copy()
    data['MEAS_DT'] = pd.to_datetime(data['MEAS_DT'])
    data = data.sort_values(['LOT_NO', 'MEAS_DT']).reset_index(drop=True)

    sensor_cols = [
        col for col in data.columns if col.startswith(SENSOR_PREFIXES)
    ]

    run_rows = []

    for lot_no, lot in data.groupby('LOT_NO', sort=False):
        lot = lot.reset_index(drop=True)

        for sensor in sensor_cols:
            missing = lot[sensor].isna()

            # 결측이 새로 시작되는 위치마다 번호를 붙여 연속 결측을 묶는다
            run_start = missing & ~missing.shift(fill_value=False)
            run_id = run_start.cumsum()

            for _, run in lot.loc[missing].groupby(
                run_id.loc[missing], sort=False
            ):
                first_row = run.index[0]
                last_row = run.index[-1]

                run_rows.append({
                    'LOT_NO': lot_no,
                    'SENSOR': sensor,
                    'MISSING_ROWS': len(run),
                    'START_DT': lot.loc[first_row, 'MEAS_DT'],
                    'END_DT': lot.loc[last_row, 'MEAS_DT'],
                    'AT_LOT_START': first_row == 0,
                    'AT_LOT_END': last_row == len(lot) - 1
                })

    return pd.DataFrame(run_rows)
