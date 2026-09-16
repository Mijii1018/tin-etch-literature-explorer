"""
文獻資料庫的比對／插值／預設集邏輯。

跟 predictor.py 一樣刻意不 import streamlit，純函式輸入輸出，
不碰 st.session_state。UI 接線邏輯（on_case_change / reset_to_manual）
留在 app.py，因為那兩個函式本質是 sidebar widget 的接線，不是領域邏輯。
"""

import numpy as np


def build_case_label(item, idx):
    chemistry_str = "/".join(item["chemistry"].split("/"))
    angle_str = f"{item['angle']}\u00b0" if item.get("angle") is not None else "angle N/A"
    return (
        f"{idx:02d} \u2014 {chemistry_str} \u00b7 {angle_str}"
        f" ({item['source'][:18]})"
    )

def build_presets_from_db(literature_db):
    presets = {
        "00 — Manual override": {
            "BCl3": 0, "Cl2": 40, "Ar": 60, "N2": 0,
            "pressure": 10.0, "bias": 100, "icp": 500,
            "manual_sel": 0.60, "cd_top": 0.15,
            "etch_time_slider": 60,
        }
    }
    for idx, item in enumerate(literature_db, start=1):
        label = build_case_label(item, idx)
        preset_etch_time = item["etch_time_s"] if item["etch_time_s"] else 60

        # 【修正】比例型文獻（Total_sccm=None）的 BCl3/Cl2/Ar/N2 欄位是百分比，
        # 不能直接當 sccm 帶入滑桿。這裡把它縮放到一個「代表性總流量」
        # （RATIO_TARGET_SCCM = 100 sccm），讓比例關係等比重現，
        # 同時讓滑桿數值落在有意義的工程範圍內。
        # Total_sccm 有值的文獻直接使用原始數值，不需要縮放。
        RATIO_TARGET_SCCM = 100
        total = item.get("Total_sccm")
        if total is None:
            raw_sum = item["BCl3"] + item["Cl2"] + item["Ar"] + item["N2"]
            if raw_sum > 0:
                scale = RATIO_TARGET_SCCM / raw_sum
            else:
                scale = 1.0
        else:
            scale = 1.0

        preset_entry = {
            "BCl3": round(item["BCl3"] * scale),
            "Cl2": round(item["Cl2"] * scale),
            "Ar": round(item["Ar"] * scale),
            "N2": round(item["N2"] * scale),
            "pressure": float(item["pressure"]),
            "bias": item["bias"],
            "icp": item["source_power"],
            "manual_sel": item["selectivity"],
            "cd_top": item["top_width_um"],
            "etch_time_slider": int(preset_etch_time),
        }

        # 【新增】只有在文獻資料裡確實記錄了真實 TiN/PR 膜厚時，才在選定該文獻
        # 時一併同步 tin_thick / pr_thick 滑桿。這是修正「選文獻後蝕穿/光阻耗盡」
        # 問題的關鍵：先前版本完全不會覆寫這兩個滑桿，導致膜厚永遠停在使用者
        # 上次手動設定的值（或滑桿最小值），跟該文獻的真實蝕刻時間對不上。
        # 沒有實測膜厚數據的文獻條目（tin_thick_A/pr_thick_A 未提供）維持原樣，
        # 不覆寫使用者當前的厚度設定，避免用未經查證的數字誤導使用者。
        if item.get("tin_thick_A") is not None:
            preset_entry["tin_thick"] = int(item["tin_thick_A"])
        if item.get("pr_thick_A") is not None:
            preset_entry["pr_thick"] = int(item["pr_thick_A"])

        presets[label] = preset_entry
    return presets


# =========================================================
# Utility functions
# =========================================================

def normalized_gas_vector(BCl3, Cl2, Ar, N2=0):
    # 【特徵解耦】這裡的正規化只依賴 BCl3/Cl2/Ar/N2 這四個「原始欄位值」互相求和取比例，
    # 完全不需要知道絕對總 sccm 是多少。也因此，不管來源是「真實絕對流量」還是
    # 「文獻只給的百分比」，只要四個欄位彼此的相對大小正確，算出來的比例向量就會一致。
    # 這是讓 Total_sccm=None 的文獻可以直接混用的關鍵。
    total = BCl3 + Cl2 + Ar + N2
    if total <= 0:
        return np.array([0.0, 0.0, 0.0, 0.0])
    return np.array([BCl3 / total, Cl2 / total, Ar / total, N2 / total])

def literature_vector(item):
    # 【特徵解耦】不讀取 item["Total_sccm"]，只用 BCl3/Cl2/Ar/N2 四個欄位算比例向量。
    # 無論該筆文獻的 Total_sccm 是實際數字還是 None，這裡的行為完全一致。
    return normalized_gas_vector(item["BCl3"], item["Cl2"], item["Ar"], item["N2"])

def exact_literature_match(
    literature_db, BCl3, Cl2, Ar, N2, pressure, bias_power, icp_power,
    tol_flow=0.01, tol_process=0.01,
):
    """以 np.isclose 進行浮點數比對，找出輸入條件是否完全等於某筆文獻資料。"""
    for item in literature_db:
        # 【新增防呆】exact match 比的是「絕對 sccm」數值是否相等，
        # 但 Total_sccm=None 的文獻，其 BCl3/Cl2/Ar/N2 欄位存的是「百分比」，
        # 跟使用者從拉桿輸入的「絕對 sccm」單位不同、不可比較。
        # 若不排除，會出現「使用者剛好把拉桿調到跟某篇文獻的百分比數字一樣」
        # 就被誤判成完全命中的荒謬情況。因此只有 Total_sccm 有值的文獻才參與 exact match。
        if item.get("Total_sccm") is None:
            continue

        same_gas = (
            np.isclose(BCl3, item["BCl3"], atol=tol_flow) and
            np.isclose(Cl2, item["Cl2"], atol=tol_flow) and
            np.isclose(Ar, item["Ar"], atol=tol_flow) and
            np.isclose(N2, item["N2"], atol=tol_flow)
        )

        same_process = (
            np.isclose(pressure, float(item["pressure"]), atol=tol_process) and
            np.isclose(bias_power, float(item["bias"]), atol=tol_process) and
            np.isclose(icp_power, float(item["source_power"]), atol=tol_process)
        )

        if same_gas and same_process:
            return item

    return None

def database_predict_value(literature_db, BCl3, Cl2, Ar, N2, target_key, required_target_material=None):
    """
    IDW（反距離加權）預測函式 —— 特徵解耦版。

    背景：部分文獻只提供氣體「比例」而沒有絕對 sccm（LITERATURE_DB 中
    Total_sccm=None 的條目，BCl3/Cl2/Ar/N2 欄位直接就是百分比，例如
    BCl3=25, Ar=75 代表 25% / 75%，兩者相加不一定等於使用者輸入的總 sccm）。
    如果直接拿使用者的「絕對 sccm」去跟這種「相對比例」算歐式距離，
    數值尺度（scale）完全對不上，距離會毫無意義。

    解法：不管是使用者輸入還是文獻資料，一律先各自正規化成「百分比向量」
    （四個分量相加為 1），只比較「化學成分的相似度」，徹底跟絕對流量大小脫鉤。

    【修正】新增兩個防呆機制：
    1. 跳過 item.get(target_key) is None 的條目——例如 Min et al. 2008
       完全沒有報告角度，若不跳過，np.array(values, dtype=float) 會直接
       炸掉或把 None 轉成 nan 汙染整個加權平均。
    2. required_target_material：只在查詢 selectivity 時使用。Woo 的
       selectivity 量的是 TiN 對 Al2O3 或 SiO2，跟 Tonotani/Chene/Izsák/
       Lee Sang/Min 量的 TiN:PR 物理意義不同，不能互相加權平均。指定這個
       參數後，只有 selectivity_target 相符的條目會參與插值。
    """

    # ---- 步驟 1：使用者輸入 → 計算總和（total sccm） ----
    user_total_sccm = BCl3 + Cl2 + Ar + N2
    if user_total_sccm <= 0:
        return 0.0

    # ---- 步驟 2：使用者輸入正規化為「百分比向量（比例）」 ----
    # current_ratio 四個分量相加 = 1，只代表「氣體組成的形狀」，與總流量大小無關。
    current_ratio = np.array([BCl3, Cl2, Ar, N2]) / user_total_sccm

    values = []
    weights = []
    exact_match_value = None

    for item in literature_db:
        # 跳過這個目標量根本沒有數據的條目（例如 Min et al. 2008 沒有 angle）
        if item.get(target_key) is None:
            continue
        # 只在查 selectivity 時套用材料別過濾，避免 TiN:Al2O3/SiO2 混進 TiN:PR
        if required_target_material is not None and item.get("selectivity_target") != required_target_material:
            continue

        # ---- 步驟 3：文獻資料同樣正規化為百分比向量 ----
        # 不論該文獻的 Total_sccm 是實際數字還是 None，這裡都只用
        # BCl3/Cl2/Ar/N2 四個欄位彼此的相對大小換算比例，行為完全一致，
        # 不需要對 Total_sccm=None 的條目另外寫特例判斷。
        ref_total = item["BCl3"] + item["Cl2"] + item["Ar"] + item["N2"]
        if ref_total <= 0:
            continue
        ref_ratio = np.array([item["BCl3"], item["Cl2"], item["Ar"], item["N2"]]) / ref_total

        # ---- 步驟 4：改用「正規化後的比例」計算歐氏距離，而非絕對 sccm ----
        dist = np.linalg.norm(current_ratio - ref_ratio)
        if dist < 1e-8:
            exact_match_value = item[target_key]
            continue

        weight = 1 / ((dist + 1e-6) ** 1.0)
        values.append(item[target_key])
        weights.append(weight)

    # 完全氣體比例相同、且該條目這個目標量有數據 → 直接回傳（沿用原本的短路行為）
    if exact_match_value is not None:
        return exact_match_value

    if not weights:
        return 0.0

    values = np.array(values, dtype=float)
    weights = np.array(weights, dtype=float)

    return np.sum(values * weights) / np.sum(weights)

def closest_literature_case(literature_db, BCl3, Cl2, Ar, N2):
    current_vec = normalized_gas_vector(BCl3, Cl2, Ar, N2)
    best_item = None
    best_dist = np.inf

    for item in literature_db:
        ref_vec = literature_vector(item)
        dist = np.linalg.norm(current_vec - ref_vec)
        if dist < best_dist:
            best_dist = dist
            best_item = item

    return best_item, best_dist