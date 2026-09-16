"""
頁面視覺主題：CSS、等高線／網格底圖產生器、
以及幾個共用的小型 HTML 渲染 helper（警示框、表格等）。

對外主要入口是 inject_theme()，app.py 在 st.set_page_config()
之後呼叫一次即可；其餘 helper（glass_alert / term_head /
barcode_strip / render_html_table）在頁面各處視需要呼叫。
"""

import urllib.parse as _urlparse

import numpy as np
import streamlit as st
import streamlit.components.v1 as _components


def _lock_light_theme():
    """
    讓 Dark Reader / Night Eye 等「瀏覽器端深色模式擴充功能」不要再介入改色。

    原理：Dark Reader 官方支援一個 opt-out 機制——只要偵測到頁面 <head> 裡有
    <meta name="darkreader-lock">，它就會直接跳過整個頁面，完全不做顏色反轉/
    重上色，改用網站自己設計的配色（也就是我們自己寫的白底淺色主題 CSS）。

    Streamlit 的 st.markdown 不會執行 <script>，所以這裡改用 components.html，
    透過 window.parent.document 把 meta tag 插入到「外層」真正的網頁 <head>
    （components.html 本身是跑在一個獨立 iframe 裡，必須用 window.parent
    才能碰到外層 DOM）。
    """
    _components.html(
        """
        <script>
        (function () {
            try {
                var parentDoc = window.parent.document;
                if (!parentDoc.querySelector('meta[name="darkreader-lock"]')) {
                    var meta = parentDoc.createElement('meta');
                    meta.name = 'darkreader-lock';
                    parentDoc.head.appendChild(meta);
                }
            } catch (e) {
                // 瀏覽器安全性限制導致無法存取 parent document 時，靜默失敗，
                // 不影響其餘頁面功能。
            }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


# =========================================================
# 全頁等高線背景產生器（給 .stApp / sidebar 用，藍圖風格淺灰等高線）
# =========================================================

def _contour_svg(width=1400, height=1400, n=9, opacity=0.05, stroke_w=1, seed=1, cx_ratio=0.1, cy_ratio=0.05, stroke_color="#2A2722"):
    _rng = np.random.default_rng(seed)
    cx, cy = width * cx_ratio, height * cy_ratio
    paths = []
    for i in range(n):
        r = 90 + i * (max(width, height) * 1.1 / n)
        squish = 0.55 + 0.1 * _rng.random()
        paths.append(
            f'<path d="M {cx-r:.0f},{cy:.0f} A {r:.0f},{r*squish:.0f} 0 0,1 {cx+r:.0f},{cy:.0f}" '
            f'stroke="{stroke_color}" stroke-width="{stroke_w}" fill="none" opacity="{opacity}"/>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{"".join(paths)}</svg>'
    )
    return "data:image/svg+xml," + _urlparse.quote(svg)

def _grid_svg(width=640, height=640, step=64, opacity=0.05, stroke_color="#2A2722"):
    lines = []
    for x in range(0, width + 1, step):
        lines.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{height}" stroke="{stroke_color}" stroke-width="1" opacity="{opacity}"/>')
    for y in range(0, height + 1, step):
        lines.append(f'<line x1="0" y1="{y}" x2="{width}" y2="{y}" stroke="{stroke_color}" stroke-width="1" opacity="{opacity}"/>')
    # 角落十字準星，呼應測繪儀器圖紙的座標標記
    for cx, cy in [(step, step), (width - step, height - step)]:
        lines.append(f'<line x1="{cx-10}" y1="{cy}" x2="{cx+10}" y2="{cy}" stroke="{stroke_color}" stroke-width="1.2" opacity="{opacity*2.2}"/>')
        lines.append(f'<line x1="{cx}" y1="{cy-10}" x2="{cx}" y2="{cy+10}" stroke="{stroke_color}" stroke-width="1.2" opacity="{opacity*2.2}"/>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{"".join(lines)}</svg>'
    )
    return "data:image/svg+xml," + _urlparse.quote(svg)

_contour_bg_main = _contour_svg(opacity=0.5, stroke_w=1, seed=3, cx_ratio=0.08, cy_ratio=0.0, stroke_color="#B9CCE3")
_contour_bg_sidebar = _contour_svg(width=700, height=1400, n=11, opacity=0.5, stroke_w=1, seed=7, cx_ratio=0.05, cy_ratio=0.5, stroke_color="#B9CCE3")
_grid_bg_main = _grid_svg(opacity=0.35, stroke_color="#D0D9E5")



# =========================================================
# 現代簡約風格 CSS
# =========================================================

_app_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        font-weight: 400;
    }

    /* ---------- ARCHIVAL CHART // 暖米白測繪儀器風 ----------
       背景：暖米白底，佈滿極細淡灰等高線（如舊氣象圖紙的地圖紋理）
       面板：半透明米白毛玻璃，邊框與陰影皆收斂、安靜
       強調色：極低彩度陶土橘，僅作為單一資料光暈使用，不作為系統發光色
    ----------------------------------------------- */
    :root {
        /* 【新增】明確聲明本頁是淺色主題（light）。
           沒有這行時，若使用者的 Windows／瀏覽器開啟了「強制深色模式」
           (Force Dark Mode / 網頁自動變暗)，瀏覽器會在畫面繪製階段
           把整頁顏色重新反轉上色：白色輸入框被轉成接近黑色、
           深色文字(--ink: #111827)因為背景也一起變暗而失去對比，
           結果就是「輸入框看起來全黑、數字看不見」，
           滑桿的藍色 (--cyan) 也可能被重新上色成偏紅色。
           加上 color-scheme:light 後，支援此屬性的瀏覽器會直接跳過
           自動變暗，維持程式原本指定的淺色配色。 */
        color-scheme: light !important;
        --bg: #F7F8FA;
        --bg-grid: #EEF1F5;
        --surface: #FFFFFF;
        --surface-2: #FFFFFF;
        --surface-solid: #FFFFFF;
        --line: #C7D2E0;
        --line-bright: #0078FF;
        --ink: #111827;
        --muted: #8B9BB4;
        --cyan: #0078FF;
        --cyan-soft: rgba(0, 120, 255, 0.06);
        --cyan-line: rgba(0, 120, 255, 0.45);
        --ok: #0078FF;
        --warn: #E26D5C;
        --err: #E26D5C;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: var(--bg) !important;
        background-image: url("__CONTOUR_MAIN__"), url("__GRID_MAIN__") !important;
        background-size: cover, 64px 64px !important;
        background-position: center, -1px -1px !important;
        background-repeat: no-repeat, repeat !important;
        background-attachment: fixed, fixed !important;
    }

    [data-testid="stHeader"] {
        background-color: var(--bg) !important;
        background-image: none !important;
    }

    html, body { background-color: var(--bg) !important; color-scheme: light !important; }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
        color: var(--ink) !important;
        text-transform: none;
    }
    h1 { font-size: 2.1rem !important; font-weight: 300 !important; }
    h2, h3 { font-size: 1.05rem !important; }

    p, label, .stMarkdown, li { color: var(--ink); font-weight: 400; font-size: 0.95rem; }
    [data-testid="stCaptionContainer"] {
        color: var(--muted) !important; font-size: 0.82rem; letter-spacing: 0.03em;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--surface);
        background-image: url("__CONTOUR_SIDE__");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.08em;
        color: var(--muted) !important;
        margin-top: 0.6rem;
        border-bottom: none;
        padding-bottom: 6px;
        text-transform: uppercase;
    }

    hr { border-color: var(--line) !important; }

    /* metric：透明底 + 藍色極細邊框 + 邊角十字準星標記 (bounding box) */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--cyan);
        border-radius: 0;
        padding: 14px 16px 16px 16px;
        position: relative;
        box-shadow: none;
    }
    [data-testid="stMetric"]::before,
    [data-testid="stMetric"]::after {
        content: "";
        position: absolute;
        width: 9px; height: 9px;
        border-top: 1.4px solid var(--cyan);
        border-left: 1.4px solid var(--cyan);
        top: -1px; left: -1px;
    }
    [data-testid="stMetric"]::after {
        top: auto; left: auto; bottom: -1px; right: -1px;
        border-top: none; border-left: none;
        border-bottom: 1.4px solid var(--cyan);
        border-right: 1.4px solid var(--cyan);
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        font-size: 0.7rem !important;
        letter-spacing: 0.1em;
        padding-left: 2px;
    }
    [data-testid="stMetricLabel"] p::before { content: none; }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: var(--ink) !important;
        font-weight: 600 !important;
        font-size: 1.9rem !important;
        line-height: 1.15 !important;
        border-bottom: none;
        padding-bottom: 0;
        padding-left: 2px;
        display: inline-block;
    }

    .stSlider [data-baseweb="slider"] > div > div { background: var(--line); height: 1px !important; }
    .stSlider [data-baseweb="slider"] > div > div > div { background: var(--cyan) !important; height: 1px !important; }
    .stSlider [role="slider"] {
        background-color: var(--cyan) !important;
        border: none !important;
        border-radius: 0 !important;
        width: 10px !important; height: 10px !important;
        box-shadow: none !important;
    }
    .stSlider label p {
        text-transform: uppercase;
        font-size: 0.78rem !important;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace !important;
        color: var(--muted) !important;
        letter-spacing: 0.06em;
    }

    /* =========================================================
       表單控件統一風格：滑桿化（Slider-ized）
       設計語言：直角邊框 + 頂部科技藍裝飾線（呼應滑桿的 cyan 進度線）
                 + 聚焦時四邊轉為科技藍高亮
       關鍵修正：只保留「單一」不透明背景層（外層容器），
                 內層 <input> / 顯示區一律 background: transparent，
                 從結構上根除「數值被內層背景蓋住」的問題，
                 而不是疊加多層背景再用 z-index 補洞。
    ========================================================= */

    /* Number Input */
[data-testid="stNumberInput"] * {
    color-scheme: light !important;
}

[data-testid="stNumberInput"] div[data-baseweb="base-input"],
[data-testid="stNumberInput"] div[data-baseweb="input"]{
    background:#ffffff !important;
    border:1px solid #C7D2E0 !important;
    border-top:2px solid #0078FF !important;
    border-radius:0 !important;
    box-shadow:none !important;
}

[data-testid="stNumberInput"] div[data-baseweb="base-input"] > div{
    background:#ffffff !important;
}

[data-testid="stNumberInput"] input{
    background:#ffffff !important;
    color:#111827 !important;
    -webkit-text-fill-color:#111827 !important;
    caret-color:#111827 !important;
}

[data-testid="stNumberInput"] button{
    background:#ffffff !important;
    color:#111827 !important;
    border-left:1px solid #C7D2E0 !important;
}

[data-testid="stNumberInput"] svg{
    fill:#111827 !important;
}
    /* ---------- 拉桿數值氣泡（拖曳時彈出的 thumb value tooltip） ----------
       同樣容易被自訂底色/邊框壓在下層，強制拉到最上層並給予清楚對比色。
    ----------------------------------------------- */
    .stSlider [data-baseweb="tooltip"],
    div[data-baseweb="popover"][data-testid="stTooltipContent"],
    .stSlider [data-testid="stThumbValue"] {
        background-color: var(--ink) !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border-radius: 0 !important;
        z-index: 999 !important;
        position: relative !important;
    }
    /* 拉桿左右兩側最小/最大值文字，避免被淡色背景吃掉對比 */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: var(--muted) !important;
        -webkit-text-fill-color: var(--muted) !important;
        z-index: 5 !important;
        position: relative !important;
    }

    /* ---------- 下拉選單：與數字輸入框同一套直角＋頂部藍線語言 ---------- */
    .stSelectbox div[data-baseweb="select"] > div {
        background: var(--surface-solid) !important;
        border: 1px solid var(--line) !important;
        border-top: 2px solid var(--cyan) !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        transition: border-color 0.15s ease;
    }
    .stSelectbox div[data-baseweb="select"]:focus-within > div {
        border-color: var(--cyan) !important;
        box-shadow: inset 0 0 0 1px var(--cyan) !important;
    }
    .stSelectbox div[data-baseweb="select"] * {
        background: transparent !important;
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 400 !important;
    }
    /* 下拉選單彈出清單也強制使用淺底深字，避免繼承系統深色佈景 */
    div[data-baseweb="popover"] li, div[data-baseweb="popover"] ul {
        background-color: var(--surface-solid) !important;
        color: var(--ink) !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: var(--cyan-soft) !important;
    }

    /* ---------- HTML 表格外框（取代 st.dataframe，直接輸出 <table>） ----------
       用普通 DOM 表格元素取代 canvas 畫的互動表格，CSS 可以完全控制表頭底色、
       格線、hover 效果；overflow-x: auto 保留原本橫向捲動的能力。
    ----------------------------------------------- */
    .custom-table-frame {
        background: var(--surface);
        border: 1px solid var(--line);
        border-top: 2px solid var(--cyan);
        border-radius: 0;
        overflow-x: auto;
        overflow-y: hidden;
        margin: 4px 0 18px 0;
    }
    .custom-table-frame table {
        border-collapse: collapse;
        width: 100%;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        white-space: nowrap;
    }
    .custom-table-frame thead th {
        background: var(--cyan-soft);
        color: var(--ink);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 0.7rem;
        text-align: left;
        padding: 9px 14px;
        border-bottom: 1.4px solid var(--cyan);
        white-space: nowrap;
    }
    .custom-table-frame tbody td {
        padding: 7px 14px;
        border-bottom: 1px solid var(--line);
        color: var(--ink);
    }
    .custom-table-frame tbody tr:last-child td { border-bottom: none; }
    .custom-table-frame tbody tr:hover td { background: rgba(0, 120, 255, 0.04); }
    /* pandas Styler 在 hide(axis="index") 後有時仍留一個空白角落格，隱藏它 */
    .custom-table-frame thead th.blank { display: none; }

    .stCheckbox label p { color: var(--ink) !important; font-weight: 400; }

    /* ---------- dataframe：同樣採用直角＋頂部藍線外框 ---------- */
    .stDataFrame {
        border: 1px solid var(--line);
        border-top: 2px solid var(--cyan);
        border-radius: 0;
        background: var(--surface);
        overflow: hidden;
        padding: 0;
    }
    .stDataFrame * { font-family: 'JetBrains Mono', monospace !important; font-weight: 400 !important; }



    div[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 0;
        background: var(--surface);
        padding: 2px 6px;
    }
    div[data-testid="stExpander"] summary {
        color: var(--muted) !important;
        font-weight: 400;
        text-transform: none;
        font-size: 0.82rem;
        letter-spacing: 0.01em;
    }

    div[data-testid="stAlertContainer"] {
        border-radius: 0;
        border: 1px solid var(--line);
        border-left: 2px solid var(--cyan);
        background: var(--surface);
    }

    /* 圖表外框：白底 + 藍色細邊框 + 角落準星，直角 */
    .chart-glass-frame {
        background: var(--surface);
        border: 1px solid var(--cyan);
        border-radius: 0;
        padding: 14px;
        position: relative;
        box-shadow: none;
    }
    .chart-glass-frame::before,
    .chart-glass-frame::after {
        content: "";
        position: absolute;
        width: 12px; height: 12px;
        border-top: 1.4px solid var(--cyan);
        border-left: 1.4px solid var(--cyan);
        top: -1px; left: -1px;
    }
    .chart-glass-frame::after {
        top: auto; left: auto; bottom: -1px; right: -1px;
        border-top: none; border-left: none;
        border-bottom: 1.4px solid var(--cyan);
        border-right: 1.4px solid var(--cyan);
    }
    .chart-glass-frame [data-testid="stImage"],
    .chart-glass-frame .stPyplot {
        border-radius: 0;
        overflow: hidden;
    }

    .param-label {
        font-size: 0.78rem;
        color: var(--ink);
        font-weight: 400;
        margin-bottom: -0.6rem;
    }

    /* 章節標頭：[ SEC.0X ] 黑白幾何切割色塊 + cyan 延伸線 */
    .term-head {
        display: flex; align-items: stretch; gap: 0;
        margin: 46px 0 18px 0;
        height: 30px;
    }
    .term-head .idx {
        font-size: 0.72rem; color: #FFFFFF; background: var(--ink);
        border: none;
        padding: 0 12px; font-weight: 600; letter-spacing: 0.1em;
        font-family: 'JetBrains Mono', monospace;
        display: flex; align-items: center;
    }
    .term-head .idx::before { content: none; }
    .term-head .label {
        font-size: 0.92rem; color: var(--ink); font-weight: 600;
        letter-spacing: 0.02em; text-transform: uppercase;
        background: var(--surface);
        border: 1px solid var(--ink);
        border-left: none;
        padding: 0 14px;
        display: flex; align-items: center;
    }
    .term-head .rule {
        flex: 1; height: 1px; align-self: center;
        background: var(--cyan);
        position: relative;
    }
    .term-head .rule::after {
        content: "";
        position: absolute; right: 0; top: 50%; transform: translateY(-50%);
        width: 6px; height: 6px;
        background: var(--cyan);
    }

    /* 診斷提示框：白底 + 藍色細邊框，左側依狀態著色細線，直角 */
    .glass-alert {
        display: flex;
        gap: 13px;
        align-items: flex-start;
        padding: 10px 14px 10px 16px;
        border-radius: 0;
        margin: 12px 0;
        font-size: 0.92rem;
        font-weight: 400;
        line-height: 1.6;
        color: var(--ink);
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 2px solid var(--line-bright);
    }
    .glass-alert.success { border-left-color: var(--ok); }
    .glass-alert.info    { border-left-color: var(--cyan); }
    .glass-alert.warning { border-left-color: var(--warn); }
    .glass-alert.error   { border-left-color: var(--err); }
    .glass-alert .dot {
        flex-shrink: 0;
        font-size: 0.74rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        white-space: nowrap;
        padding-top: 1px;
        color: var(--muted);
        font-family: 'JetBrains Mono', monospace;
    }
    .glass-alert.success .dot { color: var(--ok); }
    .glass-alert.info    .dot { color: var(--cyan); }
    .glass-alert.warning .dot { color: var(--warn); }
    .glass-alert.error   .dot { color: var(--err); }
    .glass-alert .dot::before { margin-right: 7px; display: inline-block; font-weight: 300; }
    .glass-alert.success .dot::before { content: "●"; }
    .glass-alert.info    .dot::before { content: "○"; }
    .glass-alert.warning .dot::before { content: "△"; }
    .glass-alert.error   .dot::before { content: "✕"; }

    /* barcode 裝飾：方點，藍色/灰藍交錯 */
    .barcode {
        display: flex; align-items: center; gap: 5px;
        height: 14px; margin: 10px 0 4px 0;
        opacity: 0.9;
    }
    .barcode span {
        display: inline-block;
        width: 2.5px !important; height: 6px !important;
        border-radius: 0;
        background: var(--cyan);
    }
    .barcode-id {
        font-size: 0.66rem; color: var(--muted); letter-spacing: 0.1em; margin-bottom: 4px;
        font-family: 'JetBrains Mono', monospace; font-weight: 300;
    }
    </style>
"""

_app_css = _app_css.replace("__CONTOUR_MAIN__", _contour_bg_main).replace("__CONTOUR_SIDE__", _contour_bg_sidebar).replace("__GRID_MAIN__", _grid_bg_main)


def inject_theme():
    """
    套用整站視覺主題：鎖定淺色模式 + 注入全站 CSS。

    務必在 st.set_page_config() 之後、其餘頁面內容之前呼叫一次。
    """
    _lock_light_theme()
    st.markdown(_app_css, unsafe_allow_html=True)


def glass_alert(kind, text):
    """終端風診斷訊息卡，取代 st.success/info/warning/error。"""
    tag = {"success": "STATUS — OK", "info": "STATUS — NOTE",
           "warning": "STATUS — CAUTION", "error": "STATUS — ERROR"}[kind]
    st.markdown(
        f'<div class="glass-alert {kind}"><div class="dot">{tag}</div><div>{text}</div></div>',
        unsafe_allow_html=True,
    )

def term_head(idx, label):
    """SEC. 0X — LABEL 風格的章節標頭。"""
    st.markdown(
        f'<div class="term-head"><span class="idx">SEC. {idx}</span>'
        f'<span class="label">{label}</span><span class="rule"></span></div>',
        unsafe_allow_html=True,
    )

def barcode_strip(seed_text):
    import hashlib
    h = hashlib.md5(seed_text.encode()).hexdigest()
    bars = "".join(
        f'<span style="width:{2 + (int(c, 16) % 4)}px; opacity:{0.4 + (int(c,16)%6)/10};"></span>'
        for c in h[:38]
    )
    st.markdown(f'<div class="barcode">{bars}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="barcode-id">SCAN_ID // {h[:16].upper()}</div>', unsafe_allow_html=True)

def render_html_table(styler_or_df):
    """
    用 Pandas 原生 to_html() 輸出取代 st.dataframe。

    原因：st.dataframe 是用 canvas 畫的互動表格元件，表頭底色／表格外框
    這類細節無法被 CSS 或 Styler 覆蓋（前面才另外寫了 config.toml 去補這塊）。
    改成直接輸出 <table> HTML 後，就是一般 DOM 元素，CSS 選擇器完全吃得到，
    不再需要額外設定檔。外層包一個 overflow-x: auto 的容器，欄位多時依然
    可以水平滑動，跟原本 st.dataframe 的體驗一致。
    """
    if hasattr(styler_or_df, "to_html"):
        # Styler 物件：隱藏索引欄，並保留 Styler 已套用的儲存格色彩
        try:
            table_html = styler_or_df.hide(axis="index").to_html()
        except AttributeError:
            # 相容較舊版本 pandas（hide_index 已於新版棄用）
            table_html = styler_or_df.hide_index().to_html()
    else:
        table_html = styler_or_df.to_html(index=False, border=0)
    st.markdown(f'<div class="custom-table-frame">{table_html}</div>', unsafe_allow_html=True)


