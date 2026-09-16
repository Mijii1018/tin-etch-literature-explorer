# -*- coding: utf-8 -*-
"""
modules/validation.py

Leave-One-Out Cross-Validation（LOOCV）模型驗證面板。

背景：exact_literature_match() 只要輸入條件對到某篇文獻就直接回傳文獻值，
完全繞過 IDW 插值與 predictor.py 的修正公式。這代表「把滑桿調到文獻條件」
測不出模型好壞，只測得出使用者會不會抄數字。

這支模組刻意獨立於 app.py 之外，只對外暴露一個進入點
render_validation_section(literature_db)，方便日後：
  - 覺得這個驗證面板不需要了，直接刪掉這支檔案 + app.py 裡呼叫它的
    那一兩行就好，不會動到任何既有邏輯。
  - 之後想在別的頁面或別的專案重用同一套 LOOCV 邏輯，也可以直接搬走整支檔案。

跟 predictor.py / literature.py 一樣不碰 st.session_state，
但因為要輸出到畫面上，這支檔案 import streamlit（跟 widgets.py 同一類）。
"""

import numpy as np
import pandas as pd
import streamlit as st

from literature import database_predict_value
from predictor import calc_sidewall_angle, calc_tin_etch_rate_nm_min
from theme import term_head, glass_alert, render_html_table


# =========================================================
# 純計算部分：跟畫面無關，理論上可以獨立寫單元測試
# =========================================================

def loocv_etch_rate(literature_db):
    """對每一筆文獻，假裝它不存在於資料庫，用剩下的資料 IDW+修正公式預測它的 ER，
    再跟論文實際值比較。回傳 list[dict]。"""
    rows = []
    for i, held_out in enumerate(literature_db):
        subset = literature_db[:i] + literature_db[i + 1:]
        pred, _ = calc_tin_etch_rate_nm_min(
            subset, None,
            held_out["BCl3"], held_out["Cl2"], held_out["Ar"], held_out["N2"],
            pressure=held_out["pressure"], bias_power=held_out["bias"],
            icp_power=held_out["source_power"],
        )
        actual = held_out["etch_rate_nm_min"]
        err = pred - actual
        pct = 100 * err / actual if actual else float("nan")
        rows.append({
            "source": held_out["source"], "case": held_out["case"],
            "actual": actual, "pred": pred, "err": err, "pct": pct,
        })
    return rows


def loocv_selectivity(literature_db):
    rows = []
    for i, held_out in enumerate(literature_db):
        subset = literature_db[:i] + literature_db[i + 1:]
        target_material = held_out["selectivity_target"]
        pred = database_predict_value(
            subset, held_out["BCl3"], held_out["Cl2"], held_out["Ar"], held_out["N2"],
            "selectivity", required_target_material=target_material,
        )
        actual = held_out["selectivity"]
        err = pred - actual
        pct = 100 * err / actual if actual else float("nan")
        rows.append({
            "source": held_out["source"], "case": held_out["case"],
            "target": target_material, "actual": actual, "pred": pred,
            "err": err, "pct": pct,
        })
    return rows


def loocv_angle(literature_db):
    """只有 angle 有實測值的條目才能參與（Min/Woo 系列沒有角度，天生排除）。"""
    rows = []
    for i, held_out in enumerate(literature_db):
        if held_out.get("angle") is None:
            continue
        subset = literature_db[:i] + literature_db[i + 1:]
        angle, base_angle, lo, hi = calc_sidewall_angle(
            subset, None,
            held_out["BCl3"], held_out["Cl2"], held_out["Ar"], held_out["N2"],
            pressure=held_out["pressure"], bias_power=held_out["bias"],
            icp_power=held_out["source_power"],
            selectivity=held_out["selectivity"], is_manual_selectivity=False,
        )
        actual = held_out["angle"]
        err = angle - actual
        rows.append({
            "source": held_out["source"], "case": held_out["case"],
            "actual": actual, "pred": angle, "base_idw": base_angle,
            "band_lo": lo, "band_hi": hi, "err": err,
        })
    return rows


def _summarize(rows, err_key="err", pct_key=None):
    errs = np.array([r[err_key] for r in rows], dtype=float)
    out = {
        "n": len(rows),
        "mae": float(np.mean(np.abs(errs))),
        "rmse": float(np.sqrt(np.mean(errs ** 2))),
    }
    if pct_key is not None:
        pcts = np.array([r[pct_key] for r in rows], dtype=float)
        pcts = pcts[~np.isnan(pcts)]
        out["mean_abs_pct"] = float(np.mean(np.abs(pcts))) if len(pcts) else float("nan")
        out["median_abs_pct"] = float(np.median(np.abs(pcts))) if len(pcts) else float("nan")
    return out


# =========================================================
# 畫面呈現部分
# =========================================================

def _rows_to_display_df(rows, columns, formatters):
    df = pd.DataFrame(rows)[list(columns.keys())].rename(columns=columns)
    for col, fmt in formatters.items():
        display_col = columns.get(col, col)
        df[display_col] = df[display_col].apply(fmt)
    return df


def render_validation_section(literature_db):
    """
    在頁面上渲染一個「模型驗證（LOOCV）」的可折疊區塊。

    唯一對外進入點——app.py 只需要：
        from validation import render_validation_section
        ...
        render_validation_section(LITERATURE_DB)
    要移除這個功能，刪掉這支檔案 + app.py 那兩行即可，不影響其餘任何邏輯。
    """
    st.divider()
    term_head("10", "模型目前差多少 // 留一法交叉驗證（LOOCV）")

    glass_alert(
        "info",
        "這裡不會把參數直接調回原本那筆文獻再算一次，因為那樣只是把答案重新讀回來，看不出模型到底會不會估。"
        "做法是先拿掉其中一筆資料，只用剩下的文獻去估它，再把結果和原本的實測值相比。"
        "每一筆都輪流做一次，這樣比較看得出來模型遇到沒有直接看過的條件時，大概會差多少。",
    )

    with st.expander("看 LOOCV 結果", expanded=False):

        rate_rows = loocv_etch_rate(literature_db)
        sel_rows = loocv_selectivity(literature_db)
        angle_rows = loocv_angle(literature_db)

        rate_summary = _summarize(rate_rows, pct_key="pct")
        sel_summary = _summarize(sel_rows, pct_key="pct")
        angle_summary = _summarize(angle_rows) if angle_rows else None

        st.markdown("**先看目前大概會差多少**")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "TiN 蝕刻速率｜誤差中位數",
            f"{rate_summary['median_abs_pct']:.0f}%",
            help=f"MAE={rate_summary['mae']:.1f} nm/min，n={rate_summary['n']}。"
                 f"有幾筆資料的落差特別大，所以這裡除了 MAE 之外也看中位數，避免結果被少數極端案例拉走。",
        )
        col2.metric(
            "選擇比｜誤差中位數",
            f"{sel_summary['median_abs_pct']:.0f}%",
            help=f"MAE={sel_summary['mae']:.2f}，n={sel_summary['n']}。",
        )
        if angle_summary:
            col3.metric(
                "側壁角度｜MAE",
                f"{angle_summary['mae']:.1f}\u00b0",
                help=f"n={angle_summary['n']}。有側壁角度實測值的資料只有約 6 筆，"
                     f"所以這個 MAE 目前只能當成粗略參考。",
            )
        else:
            col3.metric("側壁角度｜MAE", "N/A")

        glass_alert(
            "warning",
            "目前整個資料庫約 18 筆，側壁角度只有約 6 筆，而且資料來自不同機台和不同尺度。"
            "這些誤差比較適合拿來提醒我現在模型大概有多不準，不能當成嚴謹的信賴區間。",
        )

        st.markdown("---")
        st.markdown("**TiN 蝕刻速率（nm/min）**")
        rate_df = _rows_to_display_df(
            rate_rows,
            columns={"source": "文獻來源", "case": "案例", "actual": "實測值",
                     "pred": "LOOCV 估算", "pct": "誤差 %"},
            formatters={
                "actual": lambda v: f"{v:.1f}",
                "pred": lambda v: f"{v:.1f}",
                "pct": lambda v: f"{v:+.0f}%",
            },
        )
        render_html_table(rate_df.style.hide(axis="index"))

        st.markdown("**選擇比**")
        sel_df = _rows_to_display_df(
            sel_rows,
            columns={"source": "文獻來源", "case": "案例", "target": "目標材料",
                     "actual": "實測值", "pred": "LOOCV 估算", "pct": "誤差 %"},
            formatters={
                "actual": lambda v: f"{v:.2f}",
                "pred": lambda v: f"{v:.2f}",
                "pct": lambda v: f"{v:+.0f}%",
            },
        )
        render_html_table(sel_df.style.hide(axis="index"))

        if angle_rows:
            st.markdown("**側壁角度（\u00b0）**")
            angle_df = _rows_to_display_df(
                angle_rows,
                columns={"source": "文獻來源", "case": "案例", "actual": "實測值",
                         "pred": "LOOCV 估算", "err": "角度誤差（°）"},
                formatters={
                    "actual": lambda v: f"{v:.1f}",
                    "pred": lambda v: f"{v:.1f}",
                    "err": lambda v: f"{v:+.1f}",
                },
            )
            render_html_table(angle_df.style.hide(axis="index"))

        st.caption(
            "MAE 是平均絕對誤差；RMSE 對少數很大的誤差會更敏感；"
            "誤差 % =（LOOCV 估算值 − 文獻實測值）/ 文獻實測值 × 100。"
        )
