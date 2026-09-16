"""
TiN 蝕刻物理模型：修正函式 + 主要 calc_* 計算函式。

刻意不 import streamlit —— 這支檔案只吃數字、吐數字，
跟畫面完全無關，方便日後單獨寫單元測試驗證模型行為。
"""

import numpy as np

from literature import database_predict_value


# =========================================================
# Process correction models
# =========================================================

# ---------------------------------------------------------
# 【修正】pressure / bias 對側壁角度的影響，原本各自獨立、無交互作用的
# 線性公式已拆解如下，改用電漿物理裡兩個有實驗與理論基礎的機制：
#
#   (1) Child-Langmuir 型鞘層定律：離子能量隨 bias 呈次線性（約 sqrt）成長，
#       而不是線性——這部分是有物理依據的形狀，但係數 k_bias 本身仍是
#       工程估計值，並未針對這三批文獻資料回歸過。
#   (2) 離子平均自由徑近似與壓力成反比：壓力越高，離子在鞘層加速前
#       碰撞散射次數越多，角度分布（IADF）越寬，方向性越差。這個「壓力
#       造成的散射」效應會直接削弱 bias 想帶來的方向性提升——也就是說，
#       兩者不是加法疊加，而是「pressure 決定 bias 有多少效果能真正體現
#       在側壁角度上」的乘法交互。
#
# ⚠️ 這個交互作用「有沒有」是有物理依據的（機制成立），但交互作用「有多強」
# （k_bias、指向性倍率的上下限）仍是工程估計值，不是從 Tonotani/Woo/Chene/
# Izsák/Lee Sang 這幾批資料回歸出來的迴歸係數 —— 現有文獻資料庫每個 case
# 的 pressure/bias/ICP 都不同、且數量太少，不足以做 RSM (Response Surface
# Methodology) 迴歸。若未來蒐集到同一組氣體條件、掃 pressure×bias×ICP 的
# 完整 DOE 資料，這裡的係數應該用該資料重新擬合。
#
# 【引用來源】「pressure 越高、IADF 展寬、方向性越差」這個方向假設，具體對應
# Brcka (2013), US Patent 8,409,398 B2 (assignee: Tokyo Electron Limited) 的
# FIG.1B / background section 引用的 prior art：2 mTorr 時 IADF cone angle
# 明顯較窄，50 mTorr 時分布變寬。這篇只能佐證「方向」（且是專利文件裡的
# background 圖示，非本文實測數據，量化數字是撐開 claim 保護範圍用的參數
# 區間），不能佐證 k_bias=8.0 這類強度係數。詳見 vault 筆記
# P015_Brcka_2013_IADF_Control_Patent.md。
#
# ⚠️⚠️【已知文獻矛盾，尚未解決】Qian et al. (2024), Physics of Plasmas —
# LIF 直接量測 + HPEM 數值模擬（Fig.21）顯示的卻是相反方向：pressure 越高，
# 離子入射角反而越小（越垂直、越好）。這不只是「強度未驗證」的問題，而是
# 連「方向」本身都可能因製程 regime（本文機制：低壓力時天線對電漿的電容
# 耦合增強、離子能量分布變寬 bimodal；本函式假設的機制：平均自由徑↓/碰撞
# 散射↑）不同而反轉。兩篇的物理情境不同（Qian 2024 是低壓高密度電漿下的
# Ar 物理濺射 IEAD 量測，只在 0.5 mTorr 有實測、1/5/10 mTorr 僅有模擬結果；
# 本函式的假設對象是 TiN 文獻常見的化學反應主導蝕刻），不能直接視為「這個
# 函式錯了」，但這是資料庫中第一次出現與現有假設方向直接衝突、且證據等級
# 不算低的文獻，不應該被忽略或藏在註解裡不提——完整討論見 vault 筆記
# pressure-angle-direction-controversy.md 與 P016_Qian_2024_ionmotion_lif.md。
# ---------------------------------------------------------

def bias_pressure_interaction_effect(bias_power, pressure):
    bias_ref = 100.0
    pressure_ref = 10.0

    bias_power_safe = max(bias_power, 0.0)
    pressure_safe = max(pressure, 0.5)

    # 離子能量隨 bias 的縮放：sqrt 型（Child-Langmuir 鞘層電壓—電流關係的簡化近似），
    # 在 bias = bias_ref 時歸零，維持與舊版公式一致的錨點行為。
    ion_energy_ratio = np.sqrt(bias_power_safe / bias_ref)

    # 平均自由徑 ∝ 1/pressure：壓力越低，離子指向性倍率越高（散射越少）；
    # 壓力越高，倍率被壓低，代表同樣的 bias 在高壓下換不到同等的方向性。
    directionality_ratio = np.clip(pressure_ref / pressure_safe, 0.4, 2.5)

    k_bias = 8.0  # ⚠️ 工程估計值，非文獻回歸值
    effect = k_bias * (ion_energy_ratio - 1.0) * directionality_ratio
    return np.clip(effect, -10, 12)

def pressure_scattering_effect(pressure):
    """
    壓力對角度的「純散射展寬」效應：即使 bias = 0，壓力升高仍會讓
    中性粒子化學蝕刻的等向性成分增加，角度略微下降。
    這裡刻意不含 bias 交互項（交互作用已經放進 bias_pressure_interaction_effect），
    避免同一個物理機制被算兩次。
    """
    pressure_ref = 10.0
    effect = -0.18 * (pressure - pressure_ref)
    return np.clip(effect, -5, 3)

def icp_effect(icp_power):
    # ICP 主要透過電漿密度影響離子/自由基「通量」，較少直接影響角度分布，
    # 因此維持原本的獨立線性項；它跟 pressure 的交互作用（loading effect）
    # 已經反映在 calc_tin_etch_rate_nm_min() 的乘法式蝕刻率公式裡。
    icp_ref = 500.0
    effect = 0.002 * (icp_power - icp_ref)
    return np.clip(effect, -3, 3)

def n2_effect(N2):
    # 【補註, 原本完全沒有註解/引用來源】這個公式的方向（N2 增加→角度變差）
    # 只跟 P002_LeeSang_2016_N2ArCl2 一致；資料庫另外兩篇 N2 三元氣體文獻
    # （P004_Woo_2011_N2BCl3 是非單調、P014_Kim_2011_N2Cl2Ar_ACP 是單調上升）
    # 方向都跟這個公式不一致。三篇分屬不同基礎氣體化學/電漿源類型/量測尺度，
    # 目前沒有足夠證據判定哪篇更適用於 TSRI 製程。完整討論見 vault 筆記
    # N2-effect-controversy.md。
    effect = -0.035 * N2
    return np.clip(effect, -5, 0)

def _selectivity_angle_effect_raw(selectivity):
    if selectivity <= 0:
        return 0
    return 18.0 * np.log(selectivity / 1.2)

def selectivity_angle_effect(selectivity):
    effect = _selectivity_angle_effect_raw(selectivity)
    return np.clip(effect, -30, 5)


# =========================================================
# Main model functions
# =========================================================
# 【修正】原本 calc_selectivity / calc_sidewall_angle / calc_tin_etch_rate_nm_min
# 各自獨立呼叫 exact_literature_match()，三者輸入完全相同、結果必然一致，
# 後面的「資料不一致」偵測邏輯因此永遠不會被觸發（死碼）。
# 現在改成只比對一次，统一傳入下游函式，邏輯更清楚也省掉重複運算。

def calc_selectivity(literature_db, exact_item, BCl3, Cl2, Ar, N2):
    # 【修正 bug】原本呼叫 database_predict_value() 沒有傳入
    # required_target_material，導致 IDW 插值時把 selectivity_target="PR"
    # （TiN:PR，量級約 0.6~1.2）跟 "Al2O3"／"SiO2"（TiN:Al2O3、TiN:SiO2，
    # 量級可以到 20+，例如 P014 的 SiO2 選擇比 20.3）混在一起加權平均。
    # 這三種對照材料量的是完全不同的物理量，不該互相平均。
    #
    # 這裡固定用 "PR" 是因為 app.py 目前 UI 上展示、比較的都是 TiN:PR
    # selectivity（sidebar 的 manual_sel 滑桿、輸出 metric 標籤都寫
    # "TiN:PR 選擇比"），所以插值目標材料也必須固定為 PR，否則插值結果
    # 跟畫面上的標籤物理意義對不上。若未來 UI 想讓使用者選擇對照材料
    # （PR / Al2O3 / SiO2），這裡的 "PR" 需要改成對應的參數。
    if exact_item is not None:
        return exact_item["selectivity"]

    sel = database_predict_value(
        literature_db, BCl3, Cl2, Ar, N2, "selectivity",
        required_target_material="PR",
    )
    sel = max(sel, 0.10)
    return sel

# ---------------------------------------------------------
# 【新增】角度預測不確定帶（uncertainty band）
#
# pressure_scattering_effect / bias_pressure_interaction_effect / icp_effect
# 這三個修正函式的「方向」有電漿物理依據（IADF 隨 pressure/bias 展寬、
# Child-Langmuir 型鞘層近似），但「強度」（k_bias=8.0、指向性倍率上下限
# 0.4-2.5、icp_ref 附近斜率 0.002 等）並不是從本頁 TiN 文獻回歸出來的係數
# —— 現有文獻資料庫裡沒有任何一筆資料同時報告 sidewall angle、又完整掃過
# pressure x bias x ICP power 三個維度可供擬合（Min et al. 2008 掃了這三個
# 參數但完全沒報角度；Woo/Tonotani 有角度卻沒掃這三個參數）。
#
# 相對地，base_angle（IDW 插值出的資料庫基準角度）、n2_effect、
# selectivity_angle_effect 至少有 TiN 文獻的角度數字或化學機制描述可依循，
# 這裡不計入這個不確定帶。
#
# UNVALIDATED_CORRECTION_UNCERTAINTY_FRACTION：不確定帶半寬度 =
# 「pressure + bias/pressure交互作用 + ICP」三項修正量絕對值總和 x 這個比例。
# 0.5 本身也是工程判斷（"這三項修正方向大致可信，但強度可能有五成上下誤差"），
# 不是統計推導值。之後若拿到 TiN 的 pressure/bias/ICP DOE 實測資料（例如 TSRI
# 內部掃描，見 README 的建議方向），應該改用「模型預測 vs 實測」殘差的標準差
# 取代這個粗略比例。
# ---------------------------------------------------------
UNVALIDATED_CORRECTION_UNCERTAINTY_FRACTION = 0.5


def calc_sidewall_angle(
    literature_db, exact_item, BCl3, Cl2, Ar, N2, pressure, bias_power, icp_power,
    selectivity, is_manual_selectivity=False
):
    """
    回傳 (angle, base_angle, angle_low, angle_high)。

    exact_item 命中時：角度直接取自文獻實測值（或依 selectivity 差異做的
    net_sel_change 修正，這部分不屬於 pressure/bias/ICP 未驗證修正），
    angle_low = angle_high = angle —— 這不是模型外推，是文獻資料本身，
    不需要不確定帶。

    未命中、走 IDW + 修正公式時：angle_low/angle_high 只反映 pressure/bias/
    ICP 這三項「方向有依據、強度未驗證」修正項的不確定性，寬度隨這三項修正量
    的絕對值總和而變 —— 目前輸入離文獻資料庫的原始條件越遠、修正量越大，
    不確定帶也越寬，讓使用者一眼看出這次預測有多依賴未驗證的工程係數。
    """
    # 【修正】exact_item 命中氣體/製程條件，不代表該篇文獻一定有報告角度。
    # 例如 Woo et al. 2011 TEEM / TEEM v12n4（P003/P004）是 blanket film
    # 研究，論文本身沒有 SEM 輪廓/taper angle 數據，literature_db.xlsx 裡
    # 這兩篇的 angle 欄位已改回 None（見 build_literature_xlsx.py 校對紀錄），
    # 不再塞一個沒有標記來源的「估計角度」進資料庫。如果這裡不特別處理，
    # exact_item["angle"] 會是 None，往下 return 出去後 app.py 的
    # f"{angle:.1f}°" 會直接 TypeError。因此只有 exact_item 的 angle 有
    # 真實數據時才走「文獻角度直接鎖定」這條路；否則角度視同未精準命中，
    # 改走跟下方 else 分支相同的 IDW + pressure/bias/ICP 修正路徑（rate 與
    # selectivity 仍照舊由 exact_item 直接鎖定，不受影響）。
    if exact_item is not None and exact_item.get("angle") is not None:
        if not is_manual_selectivity:
            return exact_item["angle"], exact_item["angle"], exact_item["angle"], exact_item["angle"]
        else:
            base_angle = exact_item["angle"]
            original_sel_effect_raw = _selectivity_angle_effect_raw(exact_item["selectivity"])
            new_sel_effect_raw = _selectivity_angle_effect_raw(selectivity)
            net_sel_change = np.clip(new_sel_effect_raw - original_sel_effect_raw, -30, 5)

            angle = base_angle + net_sel_change
            angle = np.clip(angle, 25, 92)
            pure_base_angle = base_angle - original_sel_effect_raw
            return angle, pure_base_angle, angle, angle

    base_angle = database_predict_value(literature_db, BCl3, Cl2, Ar, N2, "angle")

    p_effect = pressure_scattering_effect(pressure)
    bp_effect = bias_pressure_interaction_effect(bias_power, pressure)
    icp_eff = icp_effect(icp_power)
    n2_eff = n2_effect(N2)
    sel_eff = selectivity_angle_effect(selectivity)

    angle = base_angle + p_effect + bp_effect + icp_eff + n2_eff + sel_eff
    angle = np.clip(angle, 25, 92)

    unvalidated_correction = abs(p_effect) + abs(bp_effect) + abs(icp_eff)
    half_width = unvalidated_correction * UNVALIDATED_CORRECTION_UNCERTAINTY_FRACTION

    angle_low = np.clip(angle - half_width, 25, 92)
    angle_high = np.clip(angle + half_width, 25, 92)

    return angle, base_angle, angle_low, angle_high

def calc_tin_etch_rate_nm_min(literature_db, exact_item, BCl3, Cl2, Ar, N2, pressure, bias_power, icp_power):
    if exact_item is not None:
        return exact_item["etch_rate_nm_min"], exact_item["etch_rate_nm_min"]

    base_rate = database_predict_value(literature_db, BCl3, Cl2, Ar, N2, "etch_rate_nm_min")

    # ---------------------------------------------------------
    # 【校正】以下三個係數用 Min et al. 2008 JIEC 的 Fig.3/4/5 實測掃描
    # （40% Cl2/60% Ar，TiN 蝕刻率）重新回歸，取代原本純猜測的係數。
    # ⚠️ 這是讀圖數據（無原始數據表）的簡單最小平方回歸，且只驗證於
    # Min 論文實際掃過的範圍：RF 500–900W、bias 200–400V、P 1–10mTorr、
    # 純 Cl2/Ar 化學。超出這個範圍（尤其 bias 外推到本模型 bias_ref=100V）
    # 是形狀延伸，不是實測驗證。
    #
    # ICP/RF：對 (500W,2450)/(700W,3050)/(900W,3180) 三點做 ln-ln 回歸，
    # 得到指數 ≈0.46——跟原本猜的 0.45 幾乎一樣，等於是被獨立驗證，不用改。
    rate = base_rate
    rate *= (icp_power / 500) ** 0.45

    # Bias：對 (200V,2180)/(300V,2980)/(400V,3350) 做線性回歸，斜率換算成
    # 「相對 bias_ref=100V 的倍率」約 0.0034/V，是原本 0.0018 的近 2 倍——
    # 原係數低估了 bias 對蝕刻率的影響。實測範圍是 200–400V，外推到
    # bias_ref=100V 只是線性形狀延伸，且已知真實關係在高 bias 端會飽和
    # （200→300V 漲幅比 300→400V 大），線性外推到很低 bias 時可能高估飽和，
    # 但比原本被低估兩倍的係數更接近實測。
    rate *= (1 + 0.0034 * (bias_power - 100))

    # Pressure：對 (1mTorr,2650)/(5mTorr,3050)/(10mTorr,3050) 讀圖，5→10mTorr
    # 幾乎打平，論文原文也直接寫「壓力對蝕刻率影響很小」。原本 0.012 這個
    # 係數偏大，且方向在 1→5mTorr 這段其實跟讀圖趨勢相反（讀圖是壓力越低
    # 蝕刻率越低，原公式假設壓力越低蝕刻率越高）。由於訊號本身很小、方向
    # 不穩定，這裡保守地把係數縮小到原本的 1/4，反映「蝕刻率對壓力不敏感、
    # 壓力真正的影響在側壁角度」這個論文結論，而不是硬擬合一個方向不穩的斜率。
    rate *= np.exp(-0.003 * (pressure - 10))

    rate *= (1 - 0.0022 * N2)
    rate = max(rate, 0)

    return rate, base_rate

def calc_pr_etch_rate_A_s(tin_rate_A_s, selectivity):
    if selectivity <= 0:
        return tin_rate_A_s
    return tin_rate_A_s / selectivity

def calc_cd_top_with_erosion(cd_top_um, pr_loss_nm, lateral_erosion_ratio):
    """
    【修法四】Top Width 原本從頭到尾是靜態值，跟蝕刻時間/光阻消耗量無關，
    導致「ERROR：遮罩崩塌造成幾何失真」的警告文字沒有對應的幾何量在變化。

    這裡用一個側向侵蝕係數（lateral_erosion_ratio，使用者可調，預設 0.15，
    ⚠️ 為工程估計值，不是從本頁文獻查證出來的數字）把光阻消耗量轉換成
    Top Width 的縮減量：側向侵蝕速率被假設為垂直侵蝕（pr_loss）速率的
    lateral_erosion_ratio 倍，兩側都會侵蝕，所以乘以 2。
    """
    cd_top_nm = cd_top_um * 1000
    shrink_nm = 2.0 * lateral_erosion_ratio * max(pr_loss_nm, 0.0)
    cd_top_actual_nm = max(cd_top_nm - shrink_nm, 20.0)  # 下限 20nm，避免數值歸零或負值
    return cd_top_actual_nm / 1000

def calc_profile_geometry(cd_top_um, depth_nm, angle_deg):
    cd_top_nm = cd_top_um * 1000
    if depth_nm <= 0:
        return cd_top_um

    theta = np.radians(angle_deg)
    lateral_loss_each_side = depth_nm / np.tan(theta)

    cd_bottom_nm = cd_top_nm - 2 * lateral_loss_each_side
    cd_bottom_nm = max(cd_bottom_nm, 0)

    return cd_bottom_nm / 1000