"""繪製 TiN 蝕刻側壁 profile 示意圖。

這個檔案只負責畫圖，不碰 Streamlit session state。
圖框尺寸、標註位置與留白固定；不同 recipe 只改變截面幾何和數值，
避免切換文獻時出現標題、深度標籤或圖框跳動。
"""

import numpy as np
import matplotlib.pyplot as plt


ANGLE_BAND_DISPLAY_THRESHOLD_DEG = 0.5


def _cd_bottom_nm_for_angle(cd_top_nm, depth_nm, angle_deg):
    if depth_nm <= 0:
        return cd_top_nm
    theta = np.radians(angle_deg)
    lateral_loss_each_side = depth_nm / np.tan(theta)
    return max(cd_top_nm - 2 * lateral_loss_each_side, 0.0)


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
    """回傳固定版面的 TiN 側壁截面示意圖。

    實際尺寸以文字顯示。幾何圖會用同一個 x/y 縮放倍率放進固定視窗，
    因此角度關係不會因版面縮放而失真。
    """

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    # 固定 Figure 尺寸。不要使用 tight_layout，避免每次文字長度不同又重算版面。
    fig = plt.figure(figsize=(6.4, 4.9), dpi=110, facecolor="#FFFFFF")
    ax = fig.add_axes([0.08, 0.13, 0.84, 0.78])
    ax.set_facecolor("#FFFFFF")

    text_dark = "#1F2328"
    label_grey = "#5B6270"
    profile_blue = "#1565C0"
    band_blue = "#8FA8C7"
    line_soft = "#8A9099"
    pr_fill = (0.66, 0.68, 0.72, 0.28)
    tin_fill = (0.08, 0.36, 0.74, 0.08)

    cd_top_nm = max(float(cd_top_um) * 1000.0, 1.0)
    cd_bottom_nm = max(float(cd_bottom_um) * 1000.0, 0.0)
    depth_nm = max(float(depth_nm), 1.0)

    if remaining_pr_A is not None and pr_thickness is not None and pr_thickness > 0:
        remaining_fraction = float(np.clip(remaining_pr_A / pr_thickness, 0.0, 1.0))
    else:
        remaining_fraction = float(np.clip(1.0 - pr_loss_nm / 80.0, 0.0, 1.0))
    is_collapsed = remaining_fraction <= 0.0

    # -------------------------------------------------------------
    # 固定 viewport：axes 永遠是 0~1 的正方形資料座標。
    # profile 只在中央 46% 寬、34% 高的區域中等比例縮放。
    # -------------------------------------------------------------
    max_profile_w = 0.46
    max_profile_h = 0.34
    physical_w = max(cd_top_nm, cd_bottom_nm, 1.0)
    physical_h = max(depth_nm, 1.0)
    scale = min(max_profile_w / physical_w, max_profile_h / physical_h)

    top_w = cd_top_nm * scale
    bottom_w = cd_bottom_nm * scale
    depth_h = depth_nm * scale

    x_center = 0.49
    y_surface = 0.58
    y_bottom = y_surface - depth_h

    top_left = x_center - top_w / 2.0
    top_right = x_center + top_w / 2.0
    bot_left = x_center - bottom_w / 2.0
    bot_right = x_center + bottom_w / 2.0

    # 光阻在畫面上的延伸量固定，不再使用固定 260 nm 去撐大資料範圍。
    mask_extension = 0.13
    max_pr_height = 0.10
    pr_height = 0.012 if is_collapsed else max(0.025, max_pr_height * max(remaining_fraction, 0.15))

    has_uncertainty_band = (
        angle_low is not None
        and angle_high is not None
        and (angle_high - angle_low) > ANGLE_BAND_DISPLAY_THRESHOLD_DEG
    )
    main_ls = (0, (5, 2.5)) if has_uncertainty_band else "solid"

    # 基板基準線
    ax.plot([0.18, 0.82], [y_bottom, y_bottom], color=line_soft, linewidth=0.8, zorder=1)

    # TiN 截面
    trench_x = [top_left, top_right, bot_right, bot_left, top_left]
    trench_y = [y_surface, y_surface, y_bottom, y_bottom, y_surface]
    ax.fill(trench_x, trench_y, color=tin_fill, zorder=2)
    ax.plot(
        trench_x,
        trench_y,
        color=profile_blue,
        linewidth=1.8,
        linestyle=main_ls,
        solid_joinstyle="round",
        zorder=3,
    )

    # 模型角度不確定帶
    if has_uncertainty_band:
        for a_deg in (angle_low, angle_high):
            band_bottom_nm = _cd_bottom_nm_for_angle(cd_top_nm, depth_nm, a_deg)
            band_w = band_bottom_nm * scale
            band_left = x_center - band_w / 2.0
            band_right = x_center + band_w / 2.0
            ax.plot([top_left, band_left], [y_surface, y_bottom], color=band_blue,
                    linewidth=1.0, linestyle=(0, (1, 2)), zorder=2.5)
            ax.plot([top_right, band_right], [y_surface, y_bottom], color=band_blue,
                    linewidth=1.0, linestyle=(0, (1, 2)), zorder=2.5)

    # 光阻遮罩
    left_outer = max(0.12, top_left - mask_extension)
    right_outer = min(0.86, top_right + mask_extension)
    if is_collapsed:
        collapse_fill = (0.89, 0.29, 0.24, 0.18)
        collapse_edge = "#C0392B"
        for x0, x1 in ((left_outer, top_left), (top_right, right_outer)):
            ax.fill_between([x0, x1], y_surface, y_surface + pr_height,
                            color=collapse_fill, zorder=2)
            ax.plot([x0, x1], [y_surface + pr_height] * 2,
                    color=collapse_edge, linewidth=0.9, linestyle=(0, (3, 2)), zorder=3)
    else:
        for x0, x1 in ((left_outer, top_left), (top_right, right_outer)):
            ax.fill_between([x0, x1], y_surface, y_surface + pr_height,
                            color=pr_fill, zorder=2)
            ax.plot([x0, x1], [y_surface + pr_height] * 2,
                    color=line_soft, linewidth=0.8, zorder=3)
            ax.plot([x0, x0], [y_surface, y_surface + pr_height],
                    color=line_soft, linewidth=0.7, zorder=3)
            ax.plot([x1, x1], [y_surface, y_surface + pr_height],
                    color=line_soft, linewidth=0.7, zorder=3)

    # 表面虛線
    ax.axhline(y_surface, xmin=0.16, xmax=0.84, linestyle=(0, (1, 4)),
               linewidth=0.7, color=line_soft, zorder=1)

    # ----------------------
    # 固定位置的尺寸標註
    # ----------------------
    top_dim_y = 0.78
    ax.plot([top_left, top_right], [top_dim_y, top_dim_y], color=label_grey, linewidth=0.8)
    ax.scatter([top_left, top_right], [top_dim_y, top_dim_y], s=8, color=label_grey, linewidths=0)
    ax.text(x_center, 0.835, f"{cd_top_um:.2f} µm", ha="center", va="center",
            color=text_dark, fontsize=16, fontweight="bold")
    ax.text(x_center, 0.795, "TOP WIDTH", ha="center", va="center",
            color=label_grey, fontsize=9.5)

    bottom_dim_y = 0.18
    ax.plot([bot_left, bot_right], [bottom_dim_y, bottom_dim_y], color=label_grey, linewidth=0.8)
    ax.scatter([bot_left, bot_right], [bottom_dim_y, bottom_dim_y], s=8, color=label_grey, linewidths=0)
    ax.text(x_center, 0.105, f"{cd_bottom_um:.2f} µm", ha="center", va="center",
            color=text_dark, fontsize=16, fontweight="bold")
    ax.text(x_center, 0.145, "BOTTOM WIDTH", ha="center", va="center",
            color=label_grey, fontsize=9.5)

    # 深度不再使用旋轉的大標籤，避免跟圖框撞在一起。
    ax.text(0.11, 0.50, "ETCH DEPTH", ha="left", va="center",
            color=label_grey, fontsize=9.5, fontweight="bold")
    ax.text(0.11, 0.455, f"{depth_nm:.0f} nm", ha="left", va="center",
            color=text_dark, fontsize=13.5, fontweight="bold")
    ax.plot([0.16, 0.16], [y_bottom, y_surface], color=label_grey, linewidth=0.8)
    ax.scatter([0.16, 0.16], [y_bottom, y_surface], s=8, color=label_grey, linewidths=0)

    # 角度標註固定在右側。
    anchor_y = y_surface - depth_h * 0.50
    text_x = 0.70
    ax.annotate(
        "",
        xy=(bot_right, anchor_y),
        xytext=(text_x - 0.02, anchor_y),
        arrowprops=dict(arrowstyle="-", color=label_grey, linewidth=0.8,
                        connectionstyle="arc3,rad=0.10"),
    )
    ax.text(text_x, anchor_y + 0.045, "SIDEWALL ANGLE", ha="left", va="center",
            color=label_grey, fontsize=9.5)
    ax.text(text_x, anchor_y - 0.005, f"{angle_deg:.1f}°", ha="left", va="center",
            color=text_dark, fontsize=18, fontweight="bold")
    if has_uncertainty_band:
        ax.text(text_x, anchor_y - 0.055,
                f"range {angle_low:.1f}°–{angle_high:.1f}°",
                ha="left", va="center", color=label_grey,
                fontsize=8.2, fontstyle="italic")

    # 角點
    for cx, cy in ((bot_left, y_bottom), (bot_right, y_bottom), (top_left, y_surface), (top_right, y_surface)):
        ax.scatter([cx], [cy], s=14, facecolors="none", edgecolors=line_soft,
                   linewidths=1.0, zorder=4)

    # 固定畫布；關掉所有會依資料範圍變化的軸元素。
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 不在 Matplotlib 裡重複放標題，外層 Streamlit 已經有「側壁截面示意」。
    # 這樣可以避免標題在不同瀏覽器縮放比例下被裁切。
    footnote = "Normalized viewport · dimensions shown as labels · idealized linear sidewall"
    if has_uncertainty_band:
        footnote += " · dashed=model estimate"
    fig.text(0.08, 0.035, footnote, color="#9AA3AF", fontsize=7.2,
             fontstyle="italic", ha="left", va="bottom")

    return fig
