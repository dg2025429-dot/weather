import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
BASE_END_YEAR = 2025      # 분석 기준 기간의 끝 연도
MIN_OBS_DAYS = 300        # 연도별 최소 관측일수


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    return df


@st.cache_data
def compute_yearly(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        df.groupby("연도")
        .agg(연평균기온=("평균기온", "mean"), 관측일수=("평균기온", "count"))
        .reset_index()
    )
    # 기준 기간(2025년까지) & 관측일수 300일 이상만 사용
    yearly = yearly[
        (yearly["연도"] <= BASE_END_YEAR) & (yearly["관측일수"] >= MIN_OBS_DAYS)
    ]
    yearly = yearly.sort_values("연도").reset_index(drop=True)
    return yearly


# ------------------------------------------------------------
# 데이터 로드 & 집계
# ------------------------------------------------------------
st.title("🌡️ 서울 기온 예측기")
st.caption(
    "서울의 일별 관측 데이터를 바탕으로 연도별 평균기온의 추세를 계산하고, "
    "원하는 연도의 예상 평균기온을 보여줍니다."
)

try:
    with st.spinner("데이터를 불러오는 중입니다..."):
        raw_df = load_data()
        yearly = compute_yearly(raw_df)
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 직선을 만들기에 충분한 연도 데이터가 없습니다.")
    st.stop()

years = yearly["연도"].to_numpy(dtype=float)
temps = yearly["연평균기온"].to_numpy(dtype=float)

# 선형 회귀 (최소제곱법) & 상관계수
slope, intercept = np.polyfit(years, temps, 1)
corr = float(np.corrcoef(years, temps)[0, 1])

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# 100년당 기온 상승 폭 (전체 기간)
rate_100_full = slope * 100

# 최근 20년만으로 구한 회귀 (비교용)
RECENT_WINDOW = 20
recent_yearly = yearly[yearly["연도"] >= end_year - (RECENT_WINDOW - 1)]

has_recent = len(recent_yearly) >= 2
if has_recent:
    recent_years = recent_yearly["연도"].to_numpy(dtype=float)
    recent_temps = recent_yearly["연평균기온"].to_numpy(dtype=float)
    recent_slope, recent_intercept = np.polyfit(recent_years, recent_temps, 1)
    recent_corr = float(np.corrcoef(recent_years, recent_temps)[0, 1])
    recent_n = len(recent_yearly)
    recent_start = int(recent_yearly["연도"].min())
    recent_end = int(recent_yearly["연도"].max())
    rate_100_recent = recent_slope * 100
else:
    recent_slope = recent_intercept = recent_corr = None
    recent_n = len(recent_yearly)
    recent_start = recent_end = None
    rate_100_recent = None

# ------------------------------------------------------------
# 요약 지표
# ------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("회귀선에 사용된 연도 수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
c4.metric("상관계수 (r)", f"{corr:.3f}")

st.divider()

# ------------------------------------------------------------
# 100년당 상승폭 비교 (전체 기간 vs 최근 20년)
# ------------------------------------------------------------
st.subheader("100년당 기온 상승폭 비교")

if has_recent:
    recent_box = f"""
        <div style="text-align:center; padding:20px; background-color:#fff4ec;
                    border-radius:16px; border:2px solid #e4572e;">
            <div style="font-size:16px; color:#555;">최근 {RECENT_WINDOW}년 ({recent_start}~{recent_end}년)</div>
            <div style="font-size:52px; font-weight:800; color:#e4572e; line-height:1.2;">
                {rate_100_recent:+.2f}°C
            </div>
            <div style="font-size:14px; color:#888;">/ 100년 · r = {recent_corr:.3f} (연도 {recent_n}개)</div>
        </div>
    """
else:
    recent_box = f"""
        <div style="text-align:center; padding:20px; background-color:#fff4ec;
                    border-radius:16px; border:2px solid #e4572e;">
            <div style="font-size:16px; color:#555;">최근 {RECENT_WINDOW}년</div>
            <div style="font-size:20px; color:#888; padding-top:16px;">데이터가 부족합니다</div>
        </div>
    """

col_full, col_recent = st.columns(2)
with col_full:
    st.markdown(
        f"""
        <div style="text-align:center; padding:20px; background-color:#eef3fb;
                    border-radius:16px; border:2px solid #1f77b4;">
            <div style="font-size:16px; color:#555;">전체 기간 ({start_year}~{end_year}년)</div>
            <div style="font-size:52px; font-weight:800; color:#1f77b4; line-height:1.2;">
                {rate_100_full:+.2f}°C
            </div>
            <div style="font-size:14px; color:#888;">/ 100년 · r = {corr:.3f} (연도 {n_years}개)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_recent:
    st.markdown(recent_box, unsafe_allow_html=True)

st.divider()

# ------------------------------------------------------------
# 슬라이더 & 예측값 크게 표시
# ------------------------------------------------------------
st.subheader("연도를 선택하면 예상 평균기온을 보여줍니다")
selected_year = st.slider(
    "연도 선택", min_value=1900, max_value=2100, value=BASE_END_YEAR, step=1
)
predicted_temp = slope * selected_year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:24px; background-color:#f0f2f6;
                border-radius:16px; margin-bottom:8px;">
        <div style="font-size:20px; color:#555;">{selected_year}년 예상 평균기온 (전체 기간 회귀 기준)</div>
        <div style="font-size:72px; font-weight:800; color:#e4572e; line-height:1.2;">
            {predicted_temp:.2f}°C
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ------------------------------------------------------------
# 산점도 + 회귀 직선 (Plotly)
# ------------------------------------------------------------
st.subheader("연도별 평균기온 산점도와 회귀 직선")

x_min = min(start_year, selected_year) - 2
x_max = max(end_year, selected_year) + 2
line_x = np.linspace(x_min, x_max, 300)
line_y = slope * line_x + intercept

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=years,
        y=temps,
        mode="markers",
        name="연평균기온 (관측)",
        marker=dict(size=8, color="#1f77b4"),
        hovertemplate="%{x}년<br>%{y:.2f}°C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선 (전체 기간)",
        line=dict(color="#1f77b4", width=2),
    )
)
if has_recent:
    recent_line_y = recent_slope * line_x + recent_intercept
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=recent_line_y,
            mode="lines",
            name=f"회귀 직선 (최근 {RECENT_WINDOW}년)",
            line=dict(color="#e4572e", width=2, dash="dash"),
        )
    )
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예측값",
        marker=dict(
            size=18, color="#e4572e", symbol="star", line=dict(width=2, color="black")
        ),
        hovertemplate=f"{selected_year}년 예측<br>%{{y:.2f}}°C<extra></extra>",
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    template="plotly_white",
    height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60),
)

st.plotly_chart(fig, use_container_width=True)

recent_caption = (
    f" · 최근 {RECENT_WINDOW}년({recent_start}~{recent_end}) 회귀식: "
    f"평균기온 = {recent_slope:.5f} × 연도 + {recent_intercept:.3f} "
    f"(100년당 {rate_100_recent:+.2f}°C, r = {recent_corr:.3f})"
    if has_recent
    else " · 최근 20년 자료 부족으로 비교 회귀식 생략"
)

st.caption(
    f"※ 분석 대상: {start_year}년 ~ {end_year}년 (총 {n_years}개 연도, "
    f"연 관측일수 {MIN_OBS_DAYS}일 이상, {BASE_END_YEAR}년까지의 자료만 사용) · "
    f"전체 기간 회귀식: 평균기온 = {slope:.5f} × 연도 + {intercept:.3f} "
    f"(100년당 {rate_100_full:+.2f}°C, r = {corr:.3f})"
    + recent_caption
)

with st.expander("연도별 집계 데이터 보기"):
    st.dataframe(yearly, use_container_width=True)
