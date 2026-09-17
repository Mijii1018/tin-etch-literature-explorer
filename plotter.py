"""繪製 TiN 蝕刻側壁 profile 示意圖。

這個檔案只負責畫圖，不碰 Streamlit session state。
為避免切換不同文獻時畫面忽大忽小，這裡使用固定 viewport：
所有實際幾何尺寸先以同一比例縮放後放進固定畫布，因此 x / y 不會被
不同 recipe 的線寬或蝕刻深度撐開；同時因 x / y 使用相同縮放倍率，
側壁角度仍維持正確。
"""

import numpy as np
import matplotlib.pyplot as plt


ANGLE_BAND_DISPLAY_THRESHOLD_DEG = 0.5

# 固定顯示視窗。這些是「畫面座標」，不是 nm 刻度；真正尺寸直接標在圖上。
VIEW_X_MIN = -420
VIEW_X_MAX = 420
VIEW_Y_MIN = -430
VIEW_Y_MAX = 250
TARGET_GEOMETRY_WIDTH = 520
TARGET_GEOMETRY_HEIGHT = 360


def _cd_bottom_nm_for_angle(cd_top_nm, depth_nm, angle_deg):
    if depth_nm <= 0:
        return cd_top_nm
    theta = np.radians(angle_deg)
    lateral_loss_each_side = depth_nm / np.tan(theta)
    return max(cd_top_nm - 2 * lateral_loss_each_side, 0)


def draw_profile(
    cd_top_um,
    cd_bottom_um,
    depth_nm,
    pr_loss_nm,
    angle_deg,
    remaining_pr_A=None,
    pr_thickness=None,
    angle_low=None,
    angle_high=None,
):
    """回傳固定大小的 TiN 側壁截面示意圖。

    尺寸數字仍顯示真實值；圖形本身會等比例縮放到固定 viewport 中。
    因 x / y 使用同一縮放倍率，側壁角度不會因版面調整而變形。
    """

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    # Figure 尺寸固定，Streamlit 在 use_container_width=True 時只做整體等比縮放。
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=110)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    text_dark = "#1F2328"
    label_grey = "#5B6270"
    profile_blue = "#1565C0"
    band_blue = "#8FA8C7"
    line_soft = "#8A9099"
    spine_color = "#D1D5DB"
    tin_fill = (0.08, 0.36, 0.74, 0.08)
    pr_fill = (0.66, 0.68, 0.72, 0.28)

    cd_top_nm = max(float(cd_top_um) * 1000.0, 1.0)
    cd_bottom_nm = max(float(cd_bottom_um) * 1000.0, 0.0)
    depth_nm = max(float(depth_nm), 1.0)

    if remaining_pr_A is not None and pr_thickness is not None and pr_thickness > 0:
        remaining_fraction = float(np.clip(remaining_pr_A / pr_thickness, 0.0, 1.0))
    else:
        remaining_fraction = float(np.clip(1.0 - pr_loss_nm / 80.0, 0.0, 1.0))

    is_collapsed = remaining_fraction <= 0.0

    # ------------------------------------------------------------------
    # 固定 viewport 的核心：
    # 先以真實 nm 計算「需要容納的幾何範圍」，再用單一 scale 同時縮放 x/y。
    # 這樣不同文獻的 0.1 µm / 1 µm 線寬不會改變 Axes 的尺寸，角度也不會失真。
    # ------------------------------------------------------------------
    mask_extension_nm = max(cd_top_nm * 0.30, 180.0)
    natural_width_nm = max(cd_top_nm, cd_bottom_nm) + 2 * mask_extension_nm
    pr_visual_nm = max(depth_nm * 0.28, 70.0) * max(remaining_fraction, 0.06)
    natural_height_nm = depth_nm + pr_visual_nm + max(depth_nm * 0.25, 80.0)

    scale = min(
        TARGET_GEOMETRY_WIDTH / max(natural_width_nm, 1.0),
        TARGET_GEOMETRY_HEIGHT / max(natural_height_nm, 1.0),
    )

    # 保留合理下限與上限，避免極端條件放大成誇張尺寸。
    scale = float(np.clip(scale, 0.12, 4.0))

    top_half = cd_top_nm * scale / 2.0
    bottom_half = cd_bottom_nm * scale / 2.0
    depth_d = depth_nm * scale
    mask_ext_d = mask_extension_nm * scale

    # 幾何中心固定在畫布中間偏上，讓標註與 footnote 都有固定空間。
    y_surface = 35.0
    y_bottom = y_surface - depth_d

    if is_collapsed:
        pr_height = 5.0
    else:
        pr_height = max(12.0, pr_visual_nm * scale)

    top_left, top_right = -top_half, top_half
    bot_left, bot_right = -bottom_half, bottom_half

    has_uncertainty_band = (
        angle_low is not None
        and angle_high is not None
        and (angle_high - angle_low) > ANGLE_BAND_DISPLAY_THRESHOLD_DEG
        and depth_nm > 0
    )
    main_ls = (0, (5, 2.5)) if has_uncertainty_band else "solid"

    # 基板基準線
    ax.plot([-330, 330], [y_bottom, y_bottom], color=line_soft, linewidth=0.7, zorder=1)

    # TiN 溝槽
    trench_x = [top_left, top_right, bot_right, bot_left, top_left]
    trench_y = [y_surface, y_surface, y_bottom, y_bottom, y_surface]
    ax.fill(trench_x, trench_y, color=tin_fill, zorder=2)
    ax.plot(
        trench_x,
        trench_y,
        linewidth=1.4,
        color=profile_blue,
        linestyle=main_ls,
        solid_joinstyle="round",
        zorder=3,
    )

    # 不確定帶仍用同一個縮放倍率，所以角度關係不變。
    if has_uncertainty_band:
        for a_deg in (angle_low, angle_high):
            band_bottom_nm = _cd_bottom_nm_for_angle(cd_top_nm, depth_nm, a_deg)
            band_half = band_bottom_nm * scale / 2.0
            ax.plot(
                [top_left, -band_half], [y_surface, y_bottom],
                linewidth=0.9, color=band_blue, linestyle=(0, (1, 2)), zorder=2.5,
            )
            ax.plot(
                [top_right, band_half], [y_surface, y_bottom],
                linewidth=0.9, color=band_blue, linestyle=(0, (1, 2)), zorder=2.5,
            )

    # 光阻遮罩
    mask_left = top_left - mask_ext_d
    mask_right = top_right + mask_ext_d
    if is_collapsed:
        collapse_fill = (0.89, 0.29, 0.24, 0.18)
        collapse_edge = "#C0392B"
        for x0, x1 in ((mask_left, top_left), (top_right, mask_right)):
            ax.fill_between([x0, x1], y_surface, y_surface + pr_height, color=collapse_fill, zorder=2)
            ax.plot([x0, x1], [y_surface + pr_height] * 2, color=collapse_edge,
                    linewidth=0.9, linestyle=(0, (3, 2)), zorder=3)
    else:
        for x0, x1 in ((mask_left, top_left), (top_right, mask_right)):
            ax.fill_between([x0, x1], y_surface, y_surface + pr_height, color=pr_fill, zorder=2)
            ax.plot([x0, x1], [y_surface + pr_height] * 2, color=line_soft, linewidth=0.7, zorder=3)
            ax.plot([x0, x0], [y_surface, y_surface + pr_height], color=line_soft, linewidth=0.6, zorder=3)
            ax.plot([x1, x1], [y_surface, y_surface + pr_height], color=line_soft, linewidth=0.6, zorder=3)

    # -------------------------
    # 尺寸標註（固定畫面座標）
    # -------------------------
    def h_dimension(x0, x1, y, label, value, above=True):
        ax.plot([x0, x1], [y, y], color=label_grey, linewidth=0.7, zorder=4)
        ax.scatter([x0, x1], [y, y], s=6, color=label_grey, zorder=4, linewidths=0)
        dy1, dy2 = (15, 38) if above else (-15, -38)
        va = "bottom" if above else "top"
        mid = (x0 + x1) / 2.0
        ax.text(mid, y + dy1, label, ha="center", va=va,
                color=label_grey, fontsize=9.0, zorder=6)
        ax.text(mid, y + dy2, value, ha="center", va=va,
                color=text_dark, fontsize=14.5, fontweight="bold", zorder=6)

    top_dim_y = y_surface + pr_height + 22
    bottom_dim_y = y_bottom - 22
    h_dimension(top_left, top_right, top_dim_y, "TOP WIDTH", f"{cd_top_um:.2f} µm", above=True)
    h_dimension(bot_left, bot_right, bottom_dim_y, "BOTTOM WIDTH", f"{cd_bottom_um:.2f} µm", above=False)

    # 蝕刻深度：改成固定左側尺寸線，不再依 y-axis label / tick 留白。
    depth_x = -315
    ax.plot([depth_x, depth_x], [y_bottom, y_surface], color=label_grey, linewidth=0.7, zorder=4)
    ax.scatter([depth_x, depth_x], [y_bottom, y_surface], s=6, color=label_grey, zorder=4, linewidths=0)
    ax.text(depth_x - 18, (y_surface + y_bottom) / 2,
            f"ETCH DEPTH\n{depth_nm:.0f} nm",
            rotation=90, ha="center", va="center", color=text_dark,
            fontsize=9.3, fontweight="bold", zorder=6)

    # 側壁角度標註固定在右側；不再讓長線寬把文字推到畫布外。
    angle_anchor_y = y_surface - depth_d * 0.52
    angle_x0 = min(max(bot_right + 28, 95), 185)
    angle_text_x = min(angle_x0 + 45, 255)
    ax.annotate(
        "",
        xy=(bot_right, angle_anchor_y),
        xytext=(angle_x0, angle_anchor_y),
        arrowprops=dict(arrowstyle="-", color=label_grey, linewidth=0.7,
                        connectionstyle="arc3,rad=0.12"),
        zorder=5,
    )
    ax.text(angle_text_x, angle_anchor_y + 25, "SIDEWALL ANGLE",
            color=label_grey, fontsize=9.2, va="center", ha="left", zorder=6)
    ax.text(angle_text_x, angle_anchor_y - 12, f"{angle_deg:.1f}°",
            color=text_dark, fontsize=17, fontweight="bold",
            va="center", ha="left", zorder=6)
    if has_uncertainty_band:
        ax.text(angle_text_x, angle_anchor_y - 42,
                f"range {angle_low:.1f}°–{angle_high:.1f}°",
                color=label_grey, fontsize=8.2, fontstyle="italic",
                va="center", ha="left", zorder=6)

    # 角點
    for cx, cy in ((bot_left, y_bottom), (bot_right, y_bottom), (top_left, y_surface), (top_right, y_surface)):
        ax.scatter([cx], [cy], s=11, facecolors="none", edgecolors=line_soft,
                   linewidths=0.9, zorder=4)

    ax.axhline(y_surface, linestyle=(0, (1, 4)), linewidth=0.6, color=line_soft, zorder=1)

    # -------------------------
    # 固定 viewport：不再依文獻條件改 xlim / ylim。
    # -------------------------
    ax.set_xlim(VIEW_X_MIN, VIEW_X_MAX)
    ax.set_ylim(VIEW_Y_MIN, VIEW_Y_MAX)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        "Simulated TiN Etch Profile",
        color=text_dark,
        fontsize=14,
        fontweight="bold",
        pad=14,
        loc="left",
    )

    # 這張圖是正規化的示意 viewport，因此不顯示會被誤解成實際 nm 座標的 ticks。
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(spine_color)
        spine.set_linewidth(0.8)

    footnote = "Normalized viewport · dimensions are labels, not axis scale · idealized linear sidewall"
    if has_uncertainty_band:
        footnote += "\ndashed = model estimate · light dotted = uncertainty band"
    fig.text(
        0.055, 0.018, footnote,
        color="#9AA3AF", fontsize=7.0, fontstyle="italic",
        ha="left", va="bottom",
    )

    # 固定 subplot 邊界，避免不同文字內容觸發 tight_layout 後再次改變 axes 尺寸。
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.10)
    return fig
