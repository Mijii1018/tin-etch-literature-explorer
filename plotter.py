"""
繪製 TiN 蝕刻側壁 profile 圖（matplotlib）。

不碰 st.session_state，輸入純數字，回傳一個 matplotlib Figure，
呼叫端（app.py）自己決定要用 st.pyplot() 還是存檔。
"""

import numpy as np
import matplotlib.pyplot as plt


# 不確定帶顯示門檻：跟 app.py / predictor.py 裡既有的
# angle_uncertainty_half_width > 0.5 判斷用同一個量級，避免同一件事在
# 不同地方用不同標準，導致文字警示跟圖上的視覺表現對不起來。
ANGLE_BAND_DISPLAY_THRESHOLD_DEG = 0.5

# ---------------------------------------------------------------
# 【修正 bug】TOP/BOTTOM WIDTH 與角度標註文字，原本用 matplotlib 的
# textcoords="offset points"（螢幕像素距離）定位。這在大多數 profile 下
# 沒問題，但套用某些文獻（例如 top width ~1000nm、蝕刻深度被 TiN 厚度夾限
# 在 ~100nm 的寬扁 profile）時，ax.set_aspect("equal", adjustable="box")
# 會因為 x 範圍遠大於 y 範圍而把整個繪圖框壓得很扁（letterbox），使
# px-per-data-unit 大幅下降；同樣 42pt 的螢幕位移換算回資料座標後會變成
# 遠超過原本 bottom_pad 預留空間的偏移量，導致 BOTTOM WIDTH 的數值文字
# 被推到 ylim 範圍外，跟 X 軸刻度重疊。
#
# 修法：改用「固定的資料座標（nm）位移」取代「螢幕像素位移」。因為
# aspect 是 1:1（1 資料單位 = 1 資料單位，無論 x 或 y），資料座標位移
# 不會受 letterbox 效應影響，永遠跟下面 top_pad/bottom_pad 的計算用
# 同一套單位、同一套邏輯，天然保證不會跑到留白範圍之外。
# ---------------------------------------------------------------
DIM_LABEL_OFFSET_NM = 40     # TOP/BOTTOM WIDTH 小標籤，離量測線的距離
DIM_VALUE_OFFSET_NM = 100    # TOP/BOTTOM WIDTH 數值，離量測線的距離（含小標籤本身的間距）
ANGLE_TEXT_X_PAD_NM = 15     # 角度標註文字，離引線終點的水平間距
ANGLE_LABEL_Y_OFFSET_NM = 45   # "SIDEWALL ANGLE" 標籤，離引線終點的垂直間距（向上）
ANGLE_VALUE_Y_OFFSET_NM = 45   # 角度數值，離引線終點的垂直間距（向下）
ANGLE_RANGE_Y_OFFSET_NM = 95   # 不確定帶範圍文字，離引線終點的垂直間距（向下，在角度數值之下）


def _cd_bottom_nm_for_angle(cd_top_nm, depth_nm, angle_deg):
    """
    跟 predictor.calc_profile_geometry() 同一套幾何公式（單一角度線性
    近似），只是回傳單位改成 nm 並且不做 um 轉換，方便在 plotter.py
    內部直接畫不確定帶的側壁線，不需要讓 plotter 依賴 predictor 模組。
    """
    if depth_nm <= 0:
        return cd_top_nm
    theta = np.radians(angle_deg)
    lateral_loss_each_side = depth_nm / np.tan(theta)
    return max(cd_top_nm - 2 * lateral_loss_each_side, 0)


def draw_profile(cd_top_um, cd_bottom_um, depth_nm, pr_loss_nm, angle_deg,
                  remaining_pr_A=None, pr_thickness=None,
                  angle_low=None, angle_high=None):
    """
    angle_low / angle_high（可選）：與 predictor.calc_sidewall_angle() 回傳的
    不確定帶對應。只有在 (angle_high - angle_low) > ANGLE_BAND_DISPLAY_THRESHOLD_DEG
    時才會被視為「有意義的不確定帶」並畫出來——這跟文獻精準命中時
    angle_low == angle_high == angle_deg 的情況一致（差值為 0，不畫）。

    此時主側壁線也會改成虛線，搭配不確定帶一起表示「這是模型估算值，
    不是文獻實測值」；未提供 angle_low/angle_high，或差值低於門檻時，
    維持原本的實線畫法。
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(6, 5.6))

    # [MOD 1] 背景色：Figure / Axes 一律改為純白，取代原本的深黑底 (#050505)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # ---- 淺色模式配色（取代原本「去網格化虛空風」的深色配色） ----
    # [MOD 2] 文字（標題／標籤／刻度）改為深灰／黑色，取代原本接近白色的 ink
    text_dark = "#1F2328"
    # 次要標籤（小字說明文字）用中灰，維持主／次文字的層級對比，但仍在白底上清晰可讀
    label_grey = "#5B6270"
    # [MOD 3] 蝕刻 Profile 的線條改為標準藍色，取代原本近白色的線條
    profile_blue = "#1565C0"
    # 量測輔助線／corner marker 用較深的灰，白底上才看得清楚（原本的淺灰在深底才成立）
    line_soft = "#8A9099"
    # [MOD 4] 網格線顏色：淺灰
    grid_color = "#D9DCE1"
    # [MOD 5] 圖表邊框（spines）顏色：淺灰
    spine_color = "#B7BCC4"

    tin_fill = (0.08, 0.36, 0.74, 0.08)     # TiN 溝槽：極淡藍色平面（呼應藍色主線）
    pr_fill = (0.66, 0.68, 0.72, 0.28)      # PR 遮罩：淺灰平面，白底上仍可辨識輪廓

    cd_top_nm = cd_top_um * 1000
    cd_bottom_nm = cd_bottom_um * 1000

    top_left = -cd_top_nm / 2
    top_right = cd_top_nm / 2
    bot_left = -cd_bottom_nm / 2
    bot_right = cd_bottom_nm / 2

    # 是否有「有意義」的不確定帶：文獻精準命中時 angle_low==angle_high==angle_deg，
    # 差值為 0，不畫；只有模型估算（IDW+pressure/bias/ICP修正）且修正量夠大時才畫。
    has_uncertainty_band = (
        angle_low is not None and angle_high is not None
        and (angle_high - angle_low) > ANGLE_BAND_DISPLAY_THRESHOLD_DEG
        and depth_nm > 0
    )
    main_sidewall_linestyle = (0, (5, 2.5)) if has_uncertainty_band else "solid"

    margin = max(cd_top_nm, cd_bottom_nm, 500) * 0.95
    y_bottom = -depth_nm - 150

    # 【修法一】原本 max(80 - pr_loss_nm, 10) 是拿絕對奈米數去扣一個跟使用者
    # 實際設定的光阻厚度（500Å ~ 20000Å）完全無關的固定畫布像素預算(80)，
    # 導致不同厚度下視覺消耗比例失真。改用「剩餘比例」，讓畫面比例
    # 精準對應「剩多少成」。
    MAX_VISUAL_HEIGHT = 90.0
    if remaining_pr_A is not None and pr_thickness is not None and pr_thickness > 0:
        remaining_fraction = np.clip(remaining_pr_A / pr_thickness, 0.0, 1.0)
    else:
        # 沒有提供真實厚度資訊時，退回舊有近似（僅供向後相容）
        remaining_fraction = np.clip(1.0 - pr_loss_nm / 80.0, 0.0, 1.0)

    is_collapsed = remaining_fraction <= 0.0
    pr_display_height = MAX_VISUAL_HEIGHT * remaining_fraction
    y_top = MAX_VISUAL_HEIGHT + 110

    # 基板下緣：單一極細線暗示空間錨點
    ax.plot([-margin * 0.7, margin * 0.7], [-depth_nm, -depth_nm],
            color=line_soft, linewidth=0.6, zorder=1)

    # ---- TiN 溝槽：淡藍色半透明平面 + 藍色線條輪廓 ----
    trench_x = [top_left, top_right, bot_right, bot_left, top_left]
    trench_y = [0, 0, -depth_nm, -depth_nm, 0]

    ax.fill(trench_x, trench_y, color=tin_fill, zorder=2)
    # [MOD 3] 這裡把原本的 color=ink（近白）改為 color=profile_blue（標準藍）
    # 【新增】主側壁線改用 main_sidewall_linestyle：有不確定帶時虛線（代表模型
    # 估算值），文獻精準命中或未提供 angle_low/high 時維持實線（代表文獻實測值）。
    ax.plot(trench_x, trench_y, linewidth=1.2, color=profile_blue,
            linestyle=main_sidewall_linestyle, solid_joinstyle="round", zorder=3)

    # ---- 【新增】角度不確定帶：用 angle_low / angle_high 各畫一條淺色虛線側壁，
    # 呈現「這個角度可能落在的範圍」，而不是假裝只有一個篤定答案。
    # 只畫側壁線（不重複畫底邊/填色），避免視覺過度擁擠。
    if has_uncertainty_band:
        band_color = "#8FA8C7"
        for a_deg in (angle_low, angle_high):
            band_bottom_nm = _cd_bottom_nm_for_angle(cd_top_nm, depth_nm, a_deg)
            band_bot_left = -band_bottom_nm / 2
            band_bot_right = band_bottom_nm / 2
            ax.plot([top_left, band_bot_left], [0, -depth_nm],
                    linewidth=0.9, color=band_color, linestyle=(0, (1, 2)), zorder=2.5)
            ax.plot([top_right, band_bot_right], [0, -depth_nm],
                    linewidth=0.9, color=band_color, linestyle=(0, (1, 2)), zorder=2.5)

    mask_left_x = top_left - 260
    mask_right_x = top_right + 260

    if is_collapsed:
        # 光阻已完全耗盡（remaining_pr_A <= 0，ERROR 已跳出）：
        # 改用虛線輪廓 + 淡紅色警示色畫一個扁平殘影，取代原本無論如何
        # 都會有下限高度(10)的實心灰色方塊，讓「文字說崩塌」跟「圖看起來崩塌」一致。
        collapse_height = 4.0
        collapse_fill = (0.89, 0.29, 0.24, 0.18)
        collapse_edge = "#C0392B"
        for xs in ([mask_left_x, top_left], [top_right, mask_right_x]):
            ax.fill_between(xs, 0, collapse_height, color=collapse_fill, zorder=2)
            ax.plot(xs, [collapse_height, collapse_height], color=collapse_edge,
                    linewidth=0.9, linestyle=(0, (3, 2)), zorder=3)
            ax.plot([xs[0], xs[0]], [0, collapse_height], color=collapse_edge,
                    linewidth=0.7, linestyle=(0, (3, 2)), zorder=3)
            ax.plot([xs[1], xs[1]], [0, collapse_height], color=collapse_edge,
                    linewidth=0.7, linestyle=(0, (3, 2)), zorder=3)
        pr_display_height = collapse_height  # 讓下方標註線改貼著殘影高度，而非原本的 0 高度
    else:
        for xs in ([mask_left_x, top_left], [top_right, mask_right_x]):
            ax.fill_between(xs, 0, pr_display_height, color=pr_fill, zorder=2)
            ax.plot(xs, [pr_display_height, pr_display_height], color=line_soft, linewidth=0.7, zorder=3)
            ax.plot([xs[0], xs[0]], [0, pr_display_height], color=line_soft, linewidth=0.6, zorder=3)
            ax.plot([xs[1], xs[1]], [0, pr_display_height], color=line_soft, linewidth=0.6, zorder=3)

    # ---- 無框尺寸標註：細線 + 兩端小圓點 + 懸浮資料塊（小字標籤＋大字粗體數值）----
    # 【修正 bug】label / value 原本用 annotate(..., textcoords="offset points")
    # 以「螢幕像素」距離拉開間距；改用 ax.text() 搭配固定的「資料座標（nm）」
    # 位移（DIM_LABEL_OFFSET_NM / DIM_VALUE_OFFSET_NM），理由見檔案開頭常數
    # 定義處的說明——避免寬扁 profile 下 letterbox 效應把文字推出繪圖範圍。
    def dim_line_h(x0, x1, y, label, value, above=True):
        ax.plot([x0, x1], [y, y], color=label_grey, linewidth=0.6, zorder=4)
        ax.scatter([x0, x1], [y, y], s=4, color=label_grey, zorder=4, linewidths=0)
        va = "bottom" if above else "top"
        direction = 1 if above else -1
        mid_x = (x0 + x1) / 2

        # 小標籤（label）：緊貼量測線，偏移量較小
        ax.text(
            mid_x, y + direction * DIM_LABEL_OFFSET_NM, label,
            ha="center", va=va, color=label_grey, fontsize=9.5,
            family="sans-serif", fontweight="normal", zorder=6,
        )
        # [MOD 2] 大數值（value）文字顏色從白色改為深色 text_dark，白底上才可讀
        ax.text(
            mid_x, y + direction * DIM_VALUE_OFFSET_NM, value,
            ha="center", va=va, color=text_dark, fontsize=15,
            family="sans-serif", fontweight="bold", zorder=6,
        )

    dim_line_h(top_left, top_right, pr_display_height + 40, "TOP WIDTH", f"{cd_top_um:.2f} \u00b5m", above=True)
    dim_line_h(bot_left, bot_right, -depth_nm - 40, "BOTTOM WIDTH", f"{cd_bottom_um:.2f} \u00b5m", above=False)

    # ---- 角度標註：細引線指向側壁角，label／value 同樣改用資料座標位移錯開
    # （原因同上：offset points 在極端 aspect 下會失真，見檔案開頭常數說明）----
    angle_anchor_x = bot_right + 90
    angle_anchor_y = -depth_nm * 0.5
    angle_text_x = angle_anchor_x + ANGLE_TEXT_X_PAD_NM

    ax.annotate(
        "", xy=(bot_right, angle_anchor_y), xytext=(angle_anchor_x, angle_anchor_y),
        arrowprops=dict(arrowstyle="-", color=label_grey, linewidth=0.6, connectionstyle="arc3,rad=0.15"),
        zorder=5,
    )
    ax.text(
        angle_text_x, angle_anchor_y + ANGLE_LABEL_Y_OFFSET_NM, "SIDEWALL ANGLE",
        color=label_grey, fontsize=9.5, family="sans-serif", va="center", ha="left", zorder=6,
    )
    # [MOD 2] 角度數值文字顏色從白色改為深色 text_dark
    ax.text(
        angle_text_x, angle_anchor_y - ANGLE_VALUE_Y_OFFSET_NM, f"{angle_deg:.1f}\u00b0",
        color=text_dark, fontsize=17, family="sans-serif", fontweight="bold",
        va="center", ha="left", zorder=6,
    )
    # 【新增】有不確定帶時，在角度數值下方補一行範圍註記，跟虛線側壁互相對應，
    # 不是新增數據，只是把已經算出來的 angle_low/angle_high 標在圖上。
    if has_uncertainty_band:
        ax.text(
            angle_text_x, angle_anchor_y - ANGLE_RANGE_Y_OFFSET_NM,
            f"range {angle_low:.1f}\u00b0\u2013{angle_high:.1f}\u00b0",
            color=label_grey, fontsize=8.5, family="sans-serif", fontstyle="italic",
            va="center", ha="left", zorder=6,
        )

    # 側壁底角／頂角用懸浮小圓點標記
    for cx, cy in [(bot_left, -depth_nm), (bot_right, -depth_nm), (top_left, 0), (top_right, 0)]:
        ax.scatter([cx], [cy], s=10, facecolors="none", edgecolors=line_soft, linewidths=0.9, zorder=4)

    ax.axhline(0, linestyle=(0, (1, 4)), linewidth=0.6, color=line_soft, zorder=1)

    ax.set_aspect("equal", adjustable="box")
    # [MOD 2] 標題顏色改為深色 text_dark
    ax.set_title("Simulated TiN Etch Profile", color=text_dark, fontsize=14,
                 family="sans-serif", fontweight="bold", pad=20, loc="left")

    # 上／下邊界留白：固定量 + 比例量的加法緩衝，避免小尺寸參數時標註超出範圍
    total_span = (y_top) - (y_bottom)
    top_pad = max(total_span * 0.24, 190)
    bottom_pad = max(total_span * 0.08, 80)

    # 右側留白需同時容納 PR mask 區塊與側壁角度標註，避免被畫布邊界裁切
    x_right = max(margin, mask_right_x + 40, angle_anchor_x + 60)
    ax.set_xlim(-margin, x_right)
    ax.set_ylim(y_bottom - bottom_pad, y_top + top_pad)

    # [MOD 2] 新增 X / Y 軸標籤，並統一使用深色文字
    ax.set_xlabel("Lateral Position (nm)", color=text_dark, fontsize=10, family="sans-serif", labelpad=8)
    ax.set_ylabel("Depth (nm)", color=text_dark, fontsize=10, family="sans-serif", labelpad=8)

    # [MOD 2] 刻度文字（ticks）改為深灰／黑色；原本用 set_xticks([]) / set_yticks([]) 完全隱藏，
    # 淺色模式改為保留標準刻度，方便讀出實際 nm 數值
    ax.tick_params(axis="both", colors=text_dark, labelsize=8.5)

    # [MOD 4] 保留網格，並設定為淺灰色；set_axisbelow(True) 確保網格畫在資料圖層下方，不會蓋住線條/標註
    ax.set_axisbelow(True)
    ax.grid(True, color=grid_color, linewidth=0.7, zorder=0)

    # [MOD 5] 圖表邊框（spines）改為可見，顏色設為淺灰
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(spine_color)
        spine.set_linewidth(0.8)

    # 【新增】圖說明確標註：側壁是單一角度的線性理想化重建，不是真實 SEM
    # 輪廓（沒有 bowing、footing、microtrenching 等局部變化）——資料庫裡
    # 沒有任何 TiN 文獻報告完整的深度-角度輪廓曲線，只有單一平均 taper 角度，
    # 畫成弧形/局部變化形狀會是編造，因此圖本身也如實反映這個限制。
    # （文字用英文，跟圖上其餘標籤 TOP WIDTH / SIDEWALL ANGLE 等同一套字型
    # 系統（Arial/Helvetica/DejaVu Sans）一致，避免中文字在無 CJK 字型的
    # 環境下顯示成方框亂碼。）
    footnote = "Idealized linear sidewall (single angle); not an actual SEM profile — no bowing/footing/microtrenching"
    if has_uncertainty_band:
        footnote += "\ndashed = model-estimated angle · light dotted = uncertainty band"
    fig.text(0.01, 0.005, footnote, color="#9AA3AF", fontsize=7.0,
              family="sans-serif", fontstyle="italic", ha="left", va="bottom")

    fig.tight_layout()
    return fig