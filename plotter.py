"""繪製 TiN 蝕刻側壁 profile 示意圖。

這張圖的用途是「比較不同條件」，不是精密幾何量測，因此版面固定、
所有標註位置固定。實際數值以文字顯示；截面輪廓只用來表達 taper 趨勢，
不再用真實 x/y 比例把圖壓扁或撐開。
"""

import numpy as np
import matplotlib.pyplot as plt


ANGLE_BAND_DISPLAY_THRESHOLD_DEG = 0.5


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
    """回傳固定版面的側壁截面示意圖。

    注意：這是非等比例示意圖。線寬、深度與角度的真實值都直接標在圖上；
    圖形本身只用來顯示「較垂直 / 較傾斜」的相對趨勢。
    """

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    fig = plt.figure(figsize=(6.2, 4.6), dpi=110, facecolor="#FFFFFF")
    ax = fig.add_axes([0.06, 0.10, 0.88, 0.84])
    ax.set_facecolor("#FFFFFF")

    text_dark = "#1F2328"
    label_grey = "#5B6270"
    line_soft = "#8A9099"
    profile_blue = "#1565C0"
    band_blue = "#8FA8C7"
    pr_fill = (0.66, 0.68, 0.72, 0.28)
    tin_fill = (0.08, 0.36, 0.74, 0.08)

    top_um = max(float(cd_top_um), 0.001)
    bottom_um = max(float(cd_bottom_um), 0.0)
    depth_nm = max(float(depth_nm), 0.0)

    # 光阻剩餘比例只影響灰色遮罩高度，不影響整體版面。
    if remaining_pr_A is not None and pr_thickness is not None and pr_thickness > 0:
        remaining_fraction = float(np.clip(remaining_pr_A / pr_thickness, 0.0, 1.0))
    else:
        remaining_fraction = float(np.clip(1.0 - pr_loss_nm / 80.0, 0.0, 1.0))

    # -------------------------------------------------------------
    # 固定 viewport 背景格線。
    # 這些格線只是視覺參考，不代表實際 nm / µm 座標；因此固定在 0~1 畫布。
    # 用非常淡的灰線補足技術圖感，同時避免搶過截面與標註。
    # -------------------------------------------------------------
    minor_grid = "#F1F3F5"
    major_grid = "#E5E7EB"
    for v in np.arange(0.1, 1.0, 0.1):
        is_major = abs((v * 10) % 2) < 1e-9
        color = major_grid if is_major else minor_grid
        lw = 0.75 if is_major else 0.45
        ax.plot([v, v], [0.08, 0.93], color=color, linewidth=lw, zorder=0)
        ax.plot([0.05, 0.95], [v, v], color=color, linewidth=lw, zorder=0)

    # -------------------------------------------------------------
    # 固定幾何區：每一篇文獻都畫在同一塊區域。
    # top width 固定視覺寬度；bottom width 只用比例表達 taper。
    # 這樣切換文獻時卡片大小、標註位置完全不會跑。
    # -------------------------------------------------------------
    cx = 0.50
    y_surface = 0.60
    y_bottom = 0.34
    top_visual_w = 0.30

    ratio = bottom_um / top_um if top_um > 0 else 1.0
    # 避免極端數據把圖形壓成一條線；這只是示意圖，不是等比例圖。
    ratio = float(np.clip(ratio, 0.18, 1.00))
    bottom_visual_w = top_visual_w * ratio

    tl = cx - top_visual_w / 2
    tr = cx + top_visual_w / 2
    bl = cx - bottom_visual_w / 2
    br = cx + bottom_visual_w / 2

    # PR 遮罩寬度、位置固定。
    pr_h = 0.045 + 0.07 * max(remaining_fraction, 0.0)
    left_outer = 0.20
    right_outer = 0.80

    # 基板線
    ax.plot([0.16, 0.84], [y_bottom, y_bottom], color=line_soft, linewidth=0.9, zorder=1)

    # TiN 截面
    tx = [tl, tr, br, bl, tl]
    ty = [y_surface, y_surface, y_bottom, y_bottom, y_surface]
    ax.fill(tx, ty, color=tin_fill, zorder=2)
    ax.plot(tx, ty, color=profile_blue, linewidth=2.0, zorder=3)

    # 有不確定範圍時，用外側兩條淡虛線表示，但不改變主要版面。
    has_uncertainty_band = (
        angle_low is not None
        and angle_high is not None
        and (angle_high - angle_low) > ANGLE_BAND_DISPLAY_THRESHOLD_DEG
    )
    if has_uncertainty_band:
        spread = min(0.055, 0.012 + (angle_high - angle_low) / 180.0)
        ax.plot([tl, bl - spread], [y_surface, y_bottom], color=band_blue,
                linewidth=1.0, linestyle=(0, (1, 2)), zorder=2.5)
        ax.plot([tr, br + spread], [y_surface, y_bottom], color=band_blue,
                linewidth=1.0, linestyle=(0, (1, 2)), zorder=2.5)

    # 光阻遮罩
    if remaining_fraction <= 0:
        edge = "#C0392B"
        fill = (0.89, 0.29, 0.24, 0.16)
        ph = 0.012
        for x0, x1 in ((left_outer, tl), (tr, right_outer)):
            ax.fill_between([x0, x1], y_surface, y_surface + ph, color=fill, zorder=2)
            ax.plot([x0, x1], [y_surface + ph, y_surface + ph], color=edge,
                    linewidth=0.9, linestyle=(0, (3, 2)), zorder=3)
    else:
        for x0, x1 in ((left_outer, tl), (tr, right_outer)):
            ax.fill_between([x0, x1], y_surface, y_surface + pr_h, color=pr_fill, zorder=2)
            ax.plot([x0, x1], [y_surface + pr_h, y_surface + pr_h], color=line_soft, linewidth=0.8)
            ax.plot([x0, x0], [y_surface, y_surface + pr_h], color=line_soft, linewidth=0.7)
            ax.plot([x1, x1], [y_surface, y_surface + pr_h], color=line_soft, linewidth=0.7)

    ax.axhline(y_surface, xmin=0.15, xmax=0.85, linestyle=(0, (1, 4)),
               linewidth=0.7, color=line_soft, zorder=1)

    # ---------------------------
    # 固定位置標註
    # ---------------------------
    # Top width
    ax.plot([0.35, 0.65], [0.82, 0.82], color=label_grey, linewidth=0.8)
    ax.scatter([0.35, 0.65], [0.82, 0.82], s=8, color=label_grey, linewidths=0)
    ax.text(0.50, 0.875, f"{top_um:.2f} µm", ha="center", va="center",
            color=text_dark, fontsize=16, fontweight="bold")
    ax.text(0.50, 0.84, "TOP WIDTH", ha="center", va="center",
            color=label_grey, fontsize=9.5)

    # Bottom width
    ax.plot([0.35, 0.65], [0.20, 0.20], color=label_grey, linewidth=0.8)
    ax.scatter([0.35, 0.65], [0.20, 0.20], s=8, color=label_grey, linewidths=0)
    ax.text(0.50, 0.125, f"{bottom_um:.2f} µm", ha="center", va="center",
            color=text_dark, fontsize=16, fontweight="bold")
    ax.text(0.50, 0.165, "BOTTOM WIDTH", ha="center", va="center",
            color=label_grey, fontsize=9.5)

    # Etch depth
    ax.plot([0.15, 0.15], [y_bottom, y_surface], color=label_grey, linewidth=0.8)
    ax.scatter([0.15, 0.15], [y_bottom, y_surface], s=8, color=label_grey, linewidths=0)
    ax.text(0.07, 0.50, "ETCH DEPTH", ha="left", va="center",
            color=label_grey, fontsize=9.3, fontweight="bold")
    ax.text(0.07, 0.455, f"{depth_nm:.0f} nm", ha="left", va="center",
            color=text_dark, fontsize=13.5, fontweight="bold")

    # Sidewall angle
    anchor_y = 0.47
    ax.annotate(
        "", xy=(br, anchor_y), xytext=(0.70, anchor_y),
        arrowprops=dict(arrowstyle="-", color=label_grey, linewidth=0.8,
                        connectionstyle="arc3,rad=0.10"),
    )
    ax.text(0.70, 0.525, "SIDEWALL ANGLE", ha="left", va="center",
            color=label_grey, fontsize=9.3)
    ax.text(0.70, 0.472, f"{angle_deg:.1f}°", ha="left", va="center",
            color=text_dark, fontsize=18, fontweight="bold")
    if has_uncertainty_band:
        ax.text(0.70, 0.425, f"range {angle_low:.1f}°–{angle_high:.1f}°",
                ha="left", va="center", color=label_grey,
                fontsize=8.1, fontstyle="italic")

    for x, y in ((tl, y_surface), (tr, y_surface), (bl, y_bottom), (br, y_bottom)):
        ax.scatter([x], [y], s=16, facecolors="none", edgecolors=line_soft,
                   linewidths=1.0, zorder=4)

    # 完全固定的 viewport
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 明確告知不是等比例圖，避免教授把視覺角度當量測值。
    fig.text(
        0.06, 0.035,
        "Schematic only · grid is visual reference only · not to scale",
        color="#9AA3AF", fontsize=7.2, fontstyle="italic", ha="left", va="bottom",
    )

    return fig
