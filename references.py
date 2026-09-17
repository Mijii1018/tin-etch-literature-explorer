"""P001–P019 與正式文獻名稱的對照。

P 編號是本專案內部用來整理 literature_db.xlsx 的識別碼。
正式論文名稱集中放在這裡，避免散落在 UI 或模型程式裡。
"""

import re


REFERENCES = {
    "P001": {
        "short": "Cl₂/Ar ICP",
        "title": "Inductively coupled plasma reactive ion etching of titanium nitride thin films in a Cl2/Ar plasma",
        "role": "TiN 直接蝕刻資料",
    },
    "P002": {
        "short": "Cl₂/Ar/N₂ ICP",
        "title": "Selective dry etching of TiN nanostructures over SiO2 nanotrenches using a Cl2/Ar/N2 ICP",
        "role": "TiN 直接蝕刻資料",
    },
    "P003": {
        "short": "BCl₃-based plasma",
        "title": "Dry Etching Characteristics of TiN Thin Films in BCl3-Based Plasma",
        "role": "TiN 直接蝕刻資料",
    },
    "P004": {
        "short": "N₂/BCl₃/Ar ICP",
        "title": "The Dry Etching Properties on TiN Thin Film Using an N2/BCl3/Ar ICP",
        "role": "TiN 直接蝕刻資料",
    },
    "P005": {
        "short": "F / Cl RIE",
        "title": "Etch profile engineering of TiN thin films by dry RIE in fluorine- and chlorine-based gas chemistries",
        "role": "TiN 直接蝕刻資料",
    },
    "P006": {
        "short": "300 mm superconducting BEOL",
        "title": "TiN and TaN cointegration for 300 mm superconducting back end of line",
        "role": "超導 TiN 製程背景",
    },
    "P007": {
        "short": "Sputtered TiN CPW",
        "title": "Room temperature deposition of sputtered TiN films for superconducting coplanar waveguide resonators",
        "role": "超導 TiN 製程背景",
    },
    "P008": {
        "short": "Wafer-edge etch profile",
        "title": "Characterization of an Etch Profile at a Wafer Edge in Capacitively Coupled Plasma",
        "role": "晶圓邊緣／電漿機制參考",
    },
    "P009": {
        "short": "AlN / AlScN ICP",
        "title": "Characterization of AlN and AlScN film ICP etching for micro/nano fabrication",
        "role": "跨材料蝕刻與側壁參考",
    },
    "P010": {
        "short": "Ar/CHF₃・Cl₂・BCl₃ ICP",
        "title": "Dry etching characteristics of TiN film using Ar/CHF3, Ar/Cl2, and Ar/BCl3 gas chemistries in an ICP",
        "role": "TiN 直接蝕刻資料",
    },
    "P011": {
        "short": "TiN plasma-thermal ALE",
        "title": "Isotropic plasma-thermal atomic layer etching of superconducting TiN films using sequential exposures of molecular oxygen and SF6/H2 plasma",
        "role": "TiN 蝕刻／ALE 參考",
    },
    "P012": {
        "short": "AlN MEMS etching",
        "title": "Researching the Aluminum Nitride Etching Process for Application in MEMS Resonators",
        "role": "跨材料蝕刻與側壁參考",
    },
    "P013": {
        "short": "TiN/Al₂O₃ Cl₂/Ar ACP",
        "title": "Etch Characteristics of TiN/Al2O3 Thin Film by Using a Cl2/Ar Adaptive Coupled Plasma",
        "role": "TiN 直接蝕刻資料",
    },
    "P014": {
        "short": "N₂/Cl₂/Ar ACP",
        "title": "Dry Etching of TiN in N2/Cl2/Ar Adaptively Coupled Plasma",
        "role": "TiN 直接蝕刻資料",
    },
    "P015": {
        "short": "Ion angular distribution",
        "title": "Control of Ion Angular Distribution Function at Wafer Surface",
        "role": "晶圓邊緣／電漿機制參考",
    },
    "P016": {
        "short": "Biased-wafer ion motion",
        "title": "Ion motion above a biased wafer in a plasma etching reactor",
        "role": "晶圓邊緣／電漿機制參考",
    },
    "P017": {
        "short": "Cl₂ helicon-wave plasma",
        "title": "Characterization of titanium nitride etch rate and selectivity to silicon dioxide in a Cl2 helicon-wave plasma",
        "role": "TiN 直接蝕刻資料",
    },
    "P018": {
        "short": "GaN BCl₃/Cl₂ sidewall",
        "title": "Effect of BCl3 Concentration and Process Pressure on the GaN Mesa Sidewalls in BCl3/Cl2 Based Inductively Coupled Plasma Etching",
        "role": "跨材料蝕刻與側壁參考",
    },
    "P019": {
        "short": "GaAs ICP anisotropy",
        "title": "Inductively Coupled Plasma Etching of GaAs with High Anisotropy for Photonics Applications",
        "role": "跨材料蝕刻與側壁參考",
    },
}


def reference_id_from_text(value):
    """從 source、preset label 等文字中抓出 P001 形式的編號。"""
    if value is None:
        return None
    match = re.search(r"\bP\d{3}\b", str(value), flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def reference_for_source(source):
    ref_id = reference_id_from_text(source)
    if ref_id is None:
        return None
    item = REFERENCES.get(ref_id)
    if item is None:
        return None
    return {"id": ref_id, **item}


def short_source_label(source):
    ref = reference_for_source(source)
    if ref is None:
        return str(source)
    return f"{ref['id']}｜{ref['short']}"


def reference_rows():
    """給 Streamlit dataframe 用的扁平資料。"""
    return [
        {
            "編號": ref_id,
            "簡稱": item["short"],
            "正式論文名稱": item["title"],
            "用途": item["role"],
        }
        for ref_id, item in REFERENCES.items()
    ]
