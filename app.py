from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config import LITERATURE_DB_PATH
from ai_assistant import evidence_dataframe, generate_ai_answer, search_literature
from db_loader import DBValidationError, load_literature_db
from literature import build_presets_from_db, closest_literature_case, exact_literature_match
from plotter import draw_profile
from predictor import (
    calc_cd_top_with_erosion,
    calc_profile_geometry,
    calc_pr_etch_rate_A_s,
    calc_selectivity,
    calc_sidewall_angle,
    calc_tin_etch_rate_nm_min,
)
from references import reference_for_source, reference_rows, short_source_label
from utils import nm_min_to_A_s
from validation import render_validation_section

st.set_page_config(
    page_title="TiN 蝕刻文獻探索工具",
    page_icon="🧪",
    layout="wide",
)


# 鎖定淺色主題，避免 Dark Reader / 瀏覽器強制深色模式造成白底白字。
components.html(
    """<script>
    try {
      const d = window.parent.document;
      if (!d.querySelector('meta[name="darkreader-lock"]')) {
        const m = d.createElement('meta'); m.name='darkreader-lock'; d.head.appendChild(m);
      }
      d.documentElement.lang = 'zh-Hant';
      d.documentElement.style.colorScheme = 'light';
    } catch(e) {}
    </script>""", height=0, width=0
)

st.markdown(
    """
    <style>
      :root, html, body { color-scheme: light !important; }
      .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background:#f7f8fa !important; color:#111827 !important;}
      [data-testid="stHeader"] {background:#f7f8fa !important;}
      .block-container {max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem;}
      p, span, label, li, h1, h2, h3, h4, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {color:#111827 !important;}
      section[data-testid="stSidebar"] {background:#ffffff !important; color:#111827 !important;}
      section[data-testid="stSidebar"] * {color:#111827 !important;}
      input, textarea, [data-baseweb="select"] > div, [data-testid="stNumberInput"] div[data-baseweb="input"] {background:#ffffff !important; color:#111827 !important;}
      [data-testid="stMetric"], [data-testid="stMetric"] * {background:#ffffff !important; color:#111827 !important;}
      [data-testid="stAlert"] {color:#111827 !important;}
      [data-testid="stAlert"] * {color:#111827 !important;}
      h1, h2, h3 {letter-spacing: -0.02em;}
      .eyebrow {font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; color:#7a685a; font-weight:700;}
      .subtle {color:#666; font-size:.95rem; line-height:1.65;}
      .prototype {border-left:4px solid #9a6841; padding:.8rem 1rem; background:#faf7f2; border-radius:4px;}
      div[data-testid="stMetric"] {border:1px solid #e6e0d8; padding:14px 16px; border-radius:8px; background:white;}
      .architecture {border:1px solid #e6e0d8; padding:14px 16px; border-radius:8px; background:#fff; font-family:monospace; line-height:1.7;}
      .reference-card {border:1px solid #e6e0d8; padding:.75rem .85rem; border-radius:8px; background:#faf7f2; margin:.35rem 0 .8rem 0; line-height:1.45;}
      .reference-title {font-size:.82rem; color:#4b5563; margin-top:.2rem;}
      .quick-note {padding:.65rem .85rem; border:1px solid #e6e0d8; border-radius:8px; background:#fff;}
      .ai-hero {border:1px solid #e6e0d8; background:#ffffff; border-radius:12px; padding:18px 20px; margin:.4rem 0 1rem 0;}
      .ai-hero-title {font-size:1.05rem; font-weight:700; margin-bottom:.35rem;}
      .ai-hero-sub {font-size:.92rem; color:#5b6472; line-height:1.6;}
      .tier-card {border:1px solid #e6e0d8; background:#fff; border-radius:10px; padding:12px 14px;}
      .tier-card strong {font-size:1.2rem;}
      .tier-note {font-size:.82rem; color:#6b7280; margin-top:.2rem;}
      div[data-testid="stDataFrame"] {border:1px solid #e6e0d8; border-radius:10px; overflow:hidden;}
      div[data-testid="stMetric"] [data-testid="stMetricValue"] {font-size:1.8rem;}
      @media (max-width: 768px) {
        .block-container {padding-top: 1rem; padding-left: .9rem; padding-right: .9rem; padding-bottom: 2rem;}
        h1 {font-size: 1.75rem !important;}
        h2 {font-size: 1.35rem !important;}
        h3 {font-size: 1.15rem !important;}
        div[data-testid="stMetric"] {padding: 10px 12px;}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {font-size: 1.45rem;}
        [data-testid="stTabs"] button {font-size: .88rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Data
# -----------------------------
try:
    LITERATURE_DB = load_literature_db(LITERATURE_DB_PATH)
except (DBValidationError, FileNotFoundError) as exc:
    st.error(f"我整理的文獻資料載入失敗：{exc}")
    st.stop()

PRESETS = build_presets_from_db(LITERATURE_DB)
PRESET_LABELS = list(PRESETS.keys())
PRESET_META = {PRESET_LABELS[0]: None}
for label, item in zip(PRESET_LABELS[1:], LITERATURE_DB):
    PRESET_META[label] = item

DEFAULTS = {
    "BCl3": 0,
    "Cl2": 40,
    "Ar": 60,
    "N2": 0,
    "pressure": 10.0,
    "bias": 100,
    "icp": 500,
    "etch_time_slider": 60,
    "tin_thick": 1000,
    "pr_thick": 6000,
    "manual_sel": 0.60,
    "cd_top": 1.00,
    "lat_erosion": 0.15,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def load_preset(label: str):
    values = PRESETS[label]
    for key, value in values.items():
        st.session_state[key] = value


def preset_display_label(label: str) -> str:
    item = PRESET_META.get(label)
    if item is None:
        return "手動輸入"
    ref = reference_for_source(item.get("source"))
    if ref is None:
        return label
    case = str(item.get("case", "")).strip()
    return f"{ref['id']}｜{ref['short']}｜{case}" if case else f"{ref['id']}｜{ref['short']}"


def case_display(item) -> str:
    if item is None:
        return ""
    ref = reference_for_source(item.get("source"))
    case = str(item.get("case", "")).strip()
    if ref is not None:
        return f"{ref['id']}｜{ref['short']}｜{case}" if case else f"{ref['id']}｜{ref['short']}"
    return f"{item.get('source', '')} / {case}"


def source_reference_card(item):
    if item is None:
        return
    ref = reference_for_source(item.get("source"))
    if ref is None:
        return
    st.markdown(
        f"""
        <div class="reference-card">
          <strong>{ref['id']}｜{ref['short']}</strong><br>
          <div class="reference-title">{ref['title']}</div>
          <div class="reference-title">用途：{ref['role']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="eyebrow">TiN 蝕刻研究整理工具</div>', unsafe_allow_html=True)
st.title("TiN 蝕刻文獻探索工具")
st.markdown(
    "這個工具是從大學專題延伸出來的。我把查到的 TiN 蝕刻文獻整理在一起，讓不同氣體比例、壓力和功率可以直接切換比較，也比較容易看出蝕刻速率、側壁角度和選擇比的變化。"
)
st.caption("目前還是研究整理用的原型。我主要拿它比對文獻、觀察趨勢和找下一步要驗證的條件，不會直接用這些結果決定實際製程 recipe。")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("輸入製程條件")
    preset_label = st.selectbox(
        "先選一組參考文獻",
        PRESET_LABELS,
        format_func=preset_display_label,
    )
    source_reference_card(PRESET_META.get(preset_label))

    if st.button("套用這組條件", use_container_width=True):
        load_preset(preset_label)
        st.rerun()

    st.divider()
    st.subheader("氣體流量（sccm）")
    c1, c2 = st.columns(2)
    BCl3 = c1.number_input("BCl₃", 0, 80, key="BCl3")
    Cl2 = c2.number_input("Cl₂", 0, 120, key="Cl2")
    c3, c4 = st.columns(2)
    Ar = c3.number_input("Ar", 0, 200, key="Ar")
    N2 = c4.number_input("N₂", 0, 80, key="N2")

    total = BCl3 + Cl2 + Ar + N2
    if total > 0:
        comps = [("BCl₃", BCl3), ("Cl₂", Cl2), ("Ar", Ar), ("N₂", N2)]
        ratio = " · ".join(f"{name} {value/total*100:.0f}%" for name, value in comps if value > 0)
        st.caption(f"總流量 {total} sccm · {ratio}")

    st.subheader("其他製程參數")
    pressure = st.number_input("壓力（mTorr）", 1.0, 100.0, step=0.5, key="pressure")
    bias_power = st.number_input("Bias / Chuck（V·W）", 0, 500, step=1, key="bias")
    icp_power = st.number_input("ICP / Source（W）", 100, 1200, step=10, key="icp")
    etch_time = st.number_input("蝕刻時間（秒）", 1, 300, step=1, key="etch_time_slider")

    with st.expander("幾何與光阻設定"):
        tin_thickness = st.number_input("TiN 厚度（Å）", 100, 3000, step=10, key="tin_thick")
        pr_thickness = st.number_input("光阻厚度（Å）", 500, 20000, step=10, key="pr_thick")
        cd_top_um = st.number_input("初始上方線寬（µm）", 0.05, 5.00, step=0.05, key="cd_top")
        lateral_erosion_ratio = st.number_input(
            "光阻側向侵蝕比例", 0.00, 0.50, step=0.01, key="lat_erosion",
            help="這個值只影響截面示意圖，用來看幾何變化，不是文獻實測值。",
        )

# -----------------------------
# Calculation
# -----------------------------
exact_case = exact_literature_match(
    LITERATURE_DB,
    BCl3=BCl3, Cl2=Cl2, Ar=Ar, N2=N2,
    pressure=pressure, bias_power=bias_power, icp_power=icp_power,
)

selectivity = calc_selectivity(LITERATURE_DB, exact_case, BCl3, Cl2, Ar, N2)
angle, base_angle, angle_low, angle_high = calc_sidewall_angle(
    LITERATURE_DB, exact_case, BCl3, Cl2, Ar, N2,
    pressure=pressure, bias_power=bias_power, icp_power=icp_power,
    selectivity=selectivity, is_manual_selectivity=False,
)
tin_rate_nm_min, base_tin_rate_nm_min = calc_tin_etch_rate_nm_min(
    LITERATURE_DB, exact_case, BCl3, Cl2, Ar, N2,
    pressure=pressure, bias_power=bias_power, icp_power=icp_power,
)

tin_rate_A_s = nm_min_to_A_s(tin_rate_nm_min)
pr_rate_A_s = calc_pr_etch_rate_A_s(tin_rate_A_s, selectivity)
etched_depth_A = tin_rate_A_s * etch_time
etched_depth_nm = etched_depth_A / 10
pr_loss_A = pr_rate_A_s * etch_time
pr_loss_nm = pr_loss_A / 10
remaining_pr_A = max(pr_thickness - pr_loss_A, 0)
is_tin_breakthrough = etched_depth_A >= tin_thickness
geometry_depth_nm = min(etched_depth_nm, tin_thickness / 10)
cd_top_actual_um = calc_cd_top_with_erosion(cd_top_um, pr_loss_nm, lateral_erosion_ratio)
cd_bottom_um = calc_profile_geometry(cd_top_actual_um, geometry_depth_nm, angle)
closest_case, closest_dist = closest_literature_case(LITERATURE_DB, BCl3, Cl2, Ar, N2)

# -----------------------------
# Tabs
# -----------------------------
overview_tab, ai_tab, literature_tab, validation_tab, about_tab = st.tabs(
    ["操作與結果", "AI 文獻助理", "文獻資料", "目前模型差多少", "這個工具怎麼開始的"]
)

with overview_tab:
    st.markdown("### 操作指南")
    st.markdown(
        "可以先從左邊套用一組參考文獻，再改氣體流量、壓力或功率看看差異。參數一改，下面的結果就會一起更新。我通常會先看側壁角度、蝕刻速率和選擇比，再回頭看截面圖和右邊的提醒。"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("側壁角度", f"{angle:.1f}°")
    m2.metric("TiN 蝕刻速率", f"{tin_rate_nm_min:.1f} nm/min")
    m3.metric("TiN:PR 選擇比", f"{selectivity:.2f}")

    if exact_case is not None:
        st.success(f"目前這組條件有找到相同的文獻案例：{case_display(exact_case)}")
        source_reference_card(exact_case)
    else:
        st.info(f"資料庫裡沒有完全一樣的條件，所以先抓最接近的文獻案例來估：{case_display(closest_case)}")
        source_reference_card(closest_case)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### 側壁截面示意")
        fig = draw_profile(
            cd_top_um=cd_top_actual_um,
            cd_bottom_um=cd_bottom_um,
            depth_nm=geometry_depth_nm,
            pr_loss_nm=pr_loss_nm,
            angle_deg=angle,
            remaining_pr_A=remaining_pr_A,
            pr_thickness=pr_thickness,
            angle_low=angle_low,
            angle_high=angle_high,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("這是依目前計算結果畫出的簡化截面，只用來比較不同條件下的幾何變化，不是 SEM 實拍。")

    with right:
        st.markdown("#### 這組條件我會先注意這些")
        if is_tin_breakthrough:
            st.success("TiN：照目前的估算，這個蝕刻時間應該足以蝕穿。")
        else:
            st.warning("TiN：照目前的估算，這個蝕刻時間可能還不夠。")

        if selectivity < 1:
            st.warning("光阻：選擇比小於 1，代表光阻可能比 TiN 消耗得更快，這組條件要特別小心。")
        elif selectivity < 2:
            st.info("光阻：選擇比不算高，我會再確認光阻剩餘厚度夠不夠。")
        else:
            st.success("光阻：目前的選擇比相對有利，遮罩不太容易比 TiN 更早被消耗完。")

        if angle_high - angle_low > 1.0:
            st.warning(f"角度：這組條件的估算範圍比較寬，目前大約落在 {angle_low:.1f}°–{angle_high:.1f}°。")

        st.markdown("#### 我怎麼得到這些結果？")
        st.write("1. 先找資料庫裡有沒有相同條件；沒有的話就找附近的文獻資料。")
        st.write("2. 再把壓力、Bias、ICP 等條件的差異帶進計算。")
        st.write("3. 最後把結果整理成數值、截面圖和幾個需要注意的地方。")

    with st.expander("看詳細計算"):
        c1, c2, c3 = st.columns(3)
        c1.metric("文獻角度基準", f"{base_angle:.1f}°")
        c2.metric("文獻蝕刻速率基準", f"{base_tin_rate_nm_min:.1f} nm/min")
        c3.metric("估算底部線寬", f"{cd_bottom_um:.2f} µm")
        st.write(f"估算 TiN 蝕刻深度：**{etched_depth_nm:.1f} nm**")
        st.write(f"估算光阻損耗：**{pr_loss_nm:.1f} nm**")

with ai_tab:
    st.markdown(
        """
        <div class="ai-hero">
          <div class="ai-hero-title">AI 文獻助理</div>
          <div class="ai-hero-sub">
            先從目前整理的 TiN 蝕刻資料庫找出可比較證據，再由生成式 AI 協助整理。
            系統會區分證據層級，資料不足時不強行下結論，也不直接產生最佳 recipe。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 1. 提出研究問題")
    question = st.text_area(
        "製程問題",
        height=105,
        placeholder="例如：Ar 增加對 TiN 側壁角度有什麼影響？",
        key="ai_question",
        label_visibility="collapsed",
    )

    evidence_limit = 5
    st.caption("系統會自動挑選最多 5 筆最相關的證據，優先保留可比較性較高的案例。")
    run_ai = st.button("開始分析", type="primary", use_container_width=True)

    if run_ai:
        if not question.strip():
            st.warning("請先輸入一個研究問題。")
        else:
            evidence = search_literature(question, LITERATURE_DB, limit=evidence_limit)

            st.markdown("### 2. 證據篩選")
            tier_counts = {"A": 0, "B": 0, "C": 0}
            for item in evidence:
                tier = item.get("_evidence_tier", "C")
                if tier in tier_counts:
                    tier_counts[tier] += 1

            t1, t2, t3 = st.columns(3)
            t1.markdown(
                f'<div class="tier-card"><strong>A　{tier_counts["A"]}</strong><div class="tier-note">高可比｜接近單一變因對照</div></div>',
                unsafe_allow_html=True,
            )
            t2.markdown(
                f'<div class="tier-card"><strong>B　{tier_counts["B"]}</strong><div class="tier-note">可參考｜仍有來源或條件差異</div></div>',
                unsafe_allow_html=True,
            )
            t3.markdown(
                f'<div class="tier-card"><strong>C　{tier_counts["C"]}</strong><div class="tier-note">背景資料｜不支持因果結論</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### 本次檢索到的文獻證據")
            st.dataframe(
                evidence_dataframe(evidence),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "A/B/C 是系統對『這個問題能不能用這些資料回答』的可比較性分級，"
                "不是對論文本身品質的評分。不同研究的機台、樣品與條件仍可能不同。"
            )

            try:
                api_key = st.secrets.get("GEMINI_API_KEY", "")
                model_name = st.secrets.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
                fallback_model_name = st.secrets.get("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash")
            except Exception:
                api_key = ""
                model_name = "gemini-3.5-flash-lite"
                fallback_model_name = "gemini-3.8-flash"

            st.markdown("### 3. AI 研究整理")
            if not api_key:
                st.info("文獻檢索可以正常使用，但目前尚未設定 Gemini API Key，因此生成式整理暫時關閉。")
            else:
                try:
                    with st.spinner("正在依證據層級整理回答……"):
                        answer = generate_ai_answer(
                            question=question,
                            evidence=evidence,
                            api_key=api_key,
                            model=model_name,
                            fallback_model=fallback_model_name,
                        )
                    st.markdown(answer)
                except Exception as exc:
                    if "AI_SERVICE_BUSY" in str(exc):
                        st.warning(
                            "AI 服務目前較忙碌，系統已保留本次文獻檢索結果。"
                            "請稍後再按一次「開始分析」。"
                        )
                        st.caption("系統已自動重試並嘗試備援模型。")
                    else:
                        st.error("AI 整理暫時無法完成。文獻檢索結果仍可正常使用。")
                        st.caption("請稍後重試；若持續發生，再檢查 API Key 或模型設定。")

    with st.expander("分析原則與目前限制"):
        st.markdown(
            """
            **證據層級**
            - **A｜高可比：** 目標輸出有實測值，目標變因有改變，其他條件盡量固定。
            - **B｜可參考：** 目標輸出完整，但仍存在來源或其他製程條件差異。
            - **C｜背景資料：** 只能協助理解脈絡，不拿來支持因果或方向性結論。

            **目前範圍**
            - 只使用本地整理的 TiN 蝕刻文獻資料。
            - 不自動搜尋網路、不自動下載論文。
            - 不直接產生最佳製程 recipe。
            - 資料不足時，優先說明缺口，而不是勉強生成結論。
            """
        )


with literature_tab:
    st.subheader("參考文獻對照")
    st.write("P001～P019 是我整理資料時使用的內部編號。這裡把編號、簡稱和正式論文名稱放在一起，方便回頭確認每一筆資料的來源。")
    st.dataframe(pd.DataFrame(reference_rows()), use_container_width=True, hide_index=True)

    st.markdown("#### 我整理進資料庫的製程資料")
    db = pd.DataFrame(LITERATURE_DB)
    compact = pd.DataFrame({
        "文獻來源": [short_source_label(value) for value in db["source"]],
        "案例": db["case"],
        "氣體 BCl₃/Cl₂/Ar/N₂": [f"{r.BCl3:g}/{r.Cl2:g}/{r.Ar:g}/{r.N2:g}" for r in db.itertuples()],
        "壓力（mTorr）": db["pressure"],
        "ICP / Source 功率（W）": db["source_power"],
        "Bias / Chuck": db["bias"],
        "側壁角度（°）": db["angle"],
        "蝕刻速率（nm/min）": db["etch_rate_nm_min"],
        "選擇比": db["selectivity"],
    })
    st.dataframe(compact, use_container_width=True, hide_index=True)
    st.caption(
        "氣體欄位依序是 BCl₃ / Cl₂ / Ar / N₂。因為這些資料來自不同研究，使用的機台、樣品和製程條件並不完全一致，所以我把它們當成參考資料看趨勢，不會把它們當成同一套 DOE 的實驗結果。"
    )
    st.markdown("#### 這些資料怎麼進到工具裡")
    with st.expander("看資料怎麼一路算到結果"):
        st.markdown(
            """
            <div class="architecture">
            我整理的文獻資料（Excel）<br>
            ↓<br>
            先檢查資料欄位<br>
            ↓<br>
            找相同條件；找不到時用鄰近資料插值<br>
            ↓<br>
            依輸入條件做修正與計算<br>
            ↓<br>
            畫出側壁截面，並列出需要注意的地方<br>
            ↓<br>
            最後用 LOOCV 檢查誤差
            </div>
            """,
            unsafe_allow_html=True,
        )

with validation_tab:
    st.subheader("目前模型差多少")
    st.write("我用留一法交叉驗證（LOOCV）做一個簡單的自我檢查：每次先拿掉一筆文獻，再用剩下的資料去估它，看看結果會差多少。現在資料量還不大，所以這頁主要是讓我知道哪些輸出還不能太相信，不是拿來證明模型已經可以做正式製程預測。")
    render_validation_section(LITERATURE_DB)

with about_tab:
    st.subheader("這個工具怎麼開始的")
    st.markdown(
        """
        最初做這個工具，是想協助學姊整理 TiN 蝕刻條件，也希望不同文獻裡的製程參數可以更容易放在一起比較。實際開始整理後，我發現 Cl₂、Ar、N₂、壓力、功率和蝕刻時間等條件散在不同論文裡，每次遇到新的條件都要重新翻資料，所以才有了把這些資訊集中到同一個介面的想法。

        一開始的功能其實很簡單，只是把文獻資料整理起來，再讓製程條件可以直接輸入。後來在測試過程中，我才慢慢加上文獻比對、插值、側壁截面示意和模型檢查，希望它不只是把資料集中在一起，也能幫我比較不同條件可能帶來的變化。

        現在我比較把它當成一個 **研究整理與條件探索工具**。它可以幫我把分散的資料放在一起看，也能提醒我哪些條件值得再回頭查文獻或做實驗，但目前還不能取代真正的製程驗證。

        **我主要做了什麼**
        - 先決定這個工具到底要解決哪一個問題，而不是一開始就把它當成完整預測模型。
        - 把文獻中常見的氣體流量、壓力、功率、蝕刻時間、側壁角度等資料整理成可以比較的欄位。
        - 規劃操作流程，包括要輸入哪些參數、哪些結果先顯示，以及什麼情況需要跳出提醒。
        - 用不同製程條件反覆測試，檢查輸出有沒有出現明顯不合理或互相矛盾的結果。
        - 如果結果和文獻或製程概念對不上，就回頭檢查資料、計算邏輯或功能設計，再決定要修改，還是直接把限制標出來。

        **AI 在這個專案裡扮演的角色**

        我本身不是以程式開發為主要專長，所以實際程式碼大多是在生成式 AI 協助下完成。我主要負責的是把研究需求說清楚、規劃功能架構、整理文獻資料，以及反覆測試程式跑出來的結果。

        對我來說，AI 比較像開發工具。我還是要先知道自己想解決什麼問題，也要能判斷它產生的程式和結果合不合理。這個專案真正讓我累積的，不是單純把程式碼寫出來，而是把研究問題拆開，再一步一步轉成可以測試的工具。
        """
    )
    st.markdown("#### 我目前怎麼看這個工具")
    st.write(
        "目前最大的限制還是資料量。不同文獻使用的機台、樣品和條件差很多，側壁角度的實測資料又特別少；另外壓力、Bias、ICP 的部分修正也還沒有用自己的 DOE 做過完整驗證。所以我現在把它定位成研究整理與條件探索工具，用來比文獻、看趨勢和找下一步值得做的實驗，不會直接拿它決定製程 recipe。"
    )
