# -*- coding: utf-8 -*-
"""
db_loader.py

把原本寫死在 prototype10.py 裡的 LITERATURE_DB(list[dict]) 改成從
literature_db.xlsx 讀取，並在讀取時做 schema 驗證，取代散落在
exact_literature_match() / database_predict_value() 各處的 ad-hoc
None 防呆判斷。

用法（在 prototype10.py 裡）：

    from db_loader import load_literature_db, DBValidationError

    try:
        LITERATURE_DB = load_literature_db("literature_db.xlsx")
    except DBValidationError as e:
        st.error(f"文獻資料庫驗證失敗，請修正 literature_db.xlsx：\\n{e}")
        st.stop()

load_literature_db() 回傳的 list[dict] 欄位型別跟原本手寫在 .py 裡的
LITERATURE_DB 完全相容（int/float 保持 int/float、缺值是 Python None
而不是 NaN、bool 是真正的 bool），下游的 exact_literature_match() /
database_predict_value() / calc_*() 等函式完全不需要修改。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


class DBValidationError(Exception):
    """literature_db.xlsx 內容不符合 schema 時拋出，訊息會列出全部錯誤，一次看完不用反覆重跑。"""


# ---------------------------------------------------------------
# Schema 定義
# ---------------------------------------------------------------
# required=True  → 這個 case 一定要有值，缺值視為資料錯誤（例如 source/case/chemistry）
# required=False → 允許缺值（NaN → None），代表「該篇文獻沒有報告這個量」，
#                  下游 database_predict_value() 本來就會 skip target_key 是 None 的條目。
#
# dtype 只接受："str" | "int" | "float" | "bool"
# int 欄位允許 Excel 存成 float（例如 120.0），讀入時只要是整數值就轉型，
# 因為 Excel 儲存格本質上沒有真正的 int/float 區分，這裡不強求輸入端多做一次型別轉換。

@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    required: bool
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: tuple | None = None


SCHEMA: list[ColumnSpec] = [
    ColumnSpec("source", "str", required=True),
    ColumnSpec("case", "str", required=True),
    ColumnSpec("chemistry", "str", required=True),
    ColumnSpec("BCl3", "float", required=True, min_value=0),
    ColumnSpec("Cl2", "float", required=True, min_value=0),
    ColumnSpec("Ar", "float", required=True, min_value=0),
    ColumnSpec("N2", "float", required=True, min_value=0),
    ColumnSpec("Total_sccm", "float", required=False, min_value=0),
    ColumnSpec("pressure", "float", required=True, min_value=0),
    ColumnSpec("source_power", "float", required=True, min_value=0),
    ColumnSpec("bias", "float", required=True, min_value=0),
    ColumnSpec(
        "bias_unit", "str", required=False,
        allowed_values=("V", "W"),
    ),
    ColumnSpec("etch_time_s", "float", required=False, min_value=0),
    ColumnSpec("top_width_um", "float", required=False, min_value=0),
    ColumnSpec("angle", "float", required=False, min_value=0, max_value=90),
    ColumnSpec("etch_rate_nm_min", "float", required=True, min_value=0),
    ColumnSpec("selectivity", "float", required=True, min_value=0),
    ColumnSpec(
        "selectivity_target", "str", required=True,
        allowed_values=("PR", "Al2O3", "SiO2"),
    ),
    ColumnSpec("tin_thick_A", "float", required=False, min_value=0),
    ColumnSpec("pr_thick_A", "float", required=False, min_value=0),
    ColumnSpec("intentional_overetch", "bool", required=False),
    ColumnSpec("note", "str", required=False),
]

REQUIRED_COLUMNS = [c.name for c in SCHEMA]

# int-like 欄位：BCl3/Cl2/Ar/N2/pressure/source_power/bias 這些在原本手寫的
# LITERATURE_DB 裡幾乎都是 int（少數 pressure 是 float，例如 15.0），
# 這裡統一交給 _coerce_numeric() 判斷「是不是整數值」，是的話轉成 Python int，
# 讓輸出跟原本手寫版本的型別盡量一致（雖然下游計算對 int/float 並不敏感）。
INT_LIKE_COLUMNS = {
    "BCl3", "Cl2", "Ar", "N2", "Total_sccm",
    "source_power", "bias", "etch_time_s",
    "tin_thick_A", "pr_thick_A",
}


def _coerce_numeric(value: Any, as_int_if_whole: bool):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if as_int_if_whole and float(value).is_integer():
        return int(value)
    return float(value)


def _validate_and_normalize(df: pd.DataFrame) -> list[dict]:
    errors: list[str] = []

    # ---- 1) 欄位存在性檢查：缺欄位直接視為 fatal error，不逐列往下驗證 ----
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DBValidationError(
            "literature_db.xlsx 缺少必要欄位：" + ", ".join(missing_cols) +
            f"\n目前欄位為：{list(df.columns)}"
        )

    warnings: list[str] = []
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra_cols:
        # 【修正 bug】這裡原本把「警告-不中斷」訊息塞進跟 fatal error 同一個
        # `errors` list，導致函式最後 `if errors: raise DBValidationError(...)`
        # 把它當成一般錯誤一起拋出——標籤寫著「不中斷」，實際上還是會讓整個
        # app 中斷。現在改成獨立的 `warnings` list，不參與是否 raise 的判斷，
        # 只在驗證通過後印出來提醒，真正做到「不中斷」。
        warnings.append(
            f"發現 schema 未定義的多餘欄位，將被忽略：{extra_cols}"
        )

    records: list[dict] = []

    # ---- 2) 逐列驗證 ----
    for row_idx, row in df.iterrows():
        excel_row_no = row_idx + 2  # +1 轉 1-based，+1 跳過表頭列，對應使用者在 Excel 看到的實際列號
        record: dict = {}

        for spec in SCHEMA:
            raw = row.get(spec.name)
            is_na = raw is None or (isinstance(raw, float) and math.isnan(raw))

            if is_na:
                if spec.required:
                    errors.append(
                        f"第 {excel_row_no} 列（{row.get('source', '?')} / "
                        f"{row.get('case', '?')}）：必要欄位 '{spec.name}' 為空值。"
                    )
                    record[spec.name] = None
                else:
                    record[spec.name] = None
                continue

            # ---- dtype 轉換 ----
            try:
                if spec.dtype == "str":
                    value = str(raw).strip()
                elif spec.dtype == "bool":
                    value = bool(raw)
                elif spec.dtype in ("int", "float"):
                    as_int = spec.name in INT_LIKE_COLUMNS
                    value = _coerce_numeric(raw, as_int_if_whole=as_int)
                else:
                    value = raw
            except (TypeError, ValueError) as exc:
                errors.append(
                    f"第 {excel_row_no} 列：欄位 '{spec.name}' 值 {raw!r} 無法轉型為 "
                    f"{spec.dtype}（{exc}）。"
                )
                record[spec.name] = None
                continue

            # ---- 數值範圍檢查 ----
            if spec.dtype in ("int", "float") and value is not None:
                if spec.min_value is not None and value < spec.min_value:
                    errors.append(
                        f"第 {excel_row_no} 列：欄位 '{spec.name}' = {value} "
                        f"小於允許下限 {spec.min_value}。"
                    )
                if spec.max_value is not None and value > spec.max_value:
                    errors.append(
                        f"第 {excel_row_no} 列：欄位 '{spec.name}' = {value} "
                        f"大於允許上限 {spec.max_value}（angle 上限 90 度，"
                        f"若是製程真的量到 >90° 的 undercut profile，"
                        f"schema 需要另外討論怎麼定義正負號）。"
                    )

            # ---- 列舉值檢查 ----
            if spec.allowed_values is not None and value is not None:
                if value not in spec.allowed_values:
                    errors.append(
                        f"第 {excel_row_no} 列：欄位 '{spec.name}' = {value!r} 不在允許清單 "
                        f"{spec.allowed_values} 內。"
                    )

            record[spec.name] = value

        # intentional_overetch 若未填，補預設值 False（跟原本手寫版本行為一致，
        # 原本沒寫這個 key 的條目在 dict.get("intentional_overetch") 就是 None，
        # 但下游用法都是 if item.get("intentional_overetch")，None 跟 False 效果相同，
        # 這裡明確補 False 只是讓輸出型別更乾淨，不影響任何既有邏輯)
        if record.get("intentional_overetch") is None:
            record["intentional_overetch"] = False

        # ---- 3) 氣體流量健全性檢查：至少要有一種氣體 > 0 ----
        gas_sum = sum(
            (record.get(g) or 0) for g in ("BCl3", "Cl2", "Ar", "N2")
        )
        if gas_sum <= 0:
            errors.append(
                f"第 {excel_row_no} 列（{record.get('source')} / {record.get('case')}）："
                f"BCl3+Cl2+Ar+N2 = 0，這筆資料無法參與 normalized_gas_vector() 正規化"
                f"（分母為零）。"
            )

        # ---- 4) Total_sccm 一致性檢查（只在 Total_sccm 有值時檢查） ----
        # Total_sccm=None 的條目（例如 Min et al. 系列）代表 BCl3/Cl2/Ar/N2 欄位存的是
        # 百分比而非絕對 sccm，這種情況不檢查總和，因為本來就不該相等。
        if record.get("Total_sccm") is not None:
            declared_total = record["Total_sccm"]
            if not math.isclose(declared_total, gas_sum, rel_tol=0.02, abs_tol=0.5):
                errors.append(
                    f"第 {excel_row_no} 列（{record.get('source')} / {record.get('case')}）："
                    f"Total_sccm={declared_total} 但 BCl3+Cl2+Ar+N2={gas_sum}，"
                    f"兩者不一致，可能是輸入時打錯數字。"
                )

        records.append(record)

    # ---- 5) 跨列檢查：source+case 是否重複 ----
    seen = {}
    for i, rec in enumerate(records):
        key = (rec.get("source"), rec.get("case"))
        if key in seen:
            errors.append(
                f"'{key[0]}' / '{key[1]}' 出現重複列（第 {seen[key]} 列與第 {i + 2} 列），"
                f"IDW 插值會把同一筆文獻資料算兩次權重。"
            )
        else:
            seen[key] = i + 2

    if errors:
        raise DBValidationError("\n".join(f"- {e}" for e in errors))

    for w in warnings:
        print(f"[literature_db 警告-不中斷] {w}")

    return records


def load_literature_db(path: str = "literature_db.xlsx", sheet_name: str = "literature_db") -> list[dict]:
    """
    讀取 literature_db.xlsx 並回傳 list[dict]，型別與內容通過 schema 驗證。

    驗證失敗時拋出 DBValidationError，訊息包含「所有」發現的問題
    （而不是遇到第一個錯誤就中斷），方便一次修完 Excel 檔案再重跑，
    不用反覆「修一個、跑一次、又報下一個」。
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
    except FileNotFoundError as exc:
        raise DBValidationError(f"找不到檔案：{path}") from exc
    except ValueError as exc:
        raise DBValidationError(f"找不到工作表 '{sheet_name}'：{exc}") from exc

    if df.empty:
        raise DBValidationError(f"{path} 的 '{sheet_name}' 工作表沒有任何資料列。")

    return _validate_and_normalize(df)


if __name__ == "__main__":
    # 手動執行這個檔案可以單獨驗證 literature_db.xlsx，
    # 不用啟動整個 Streamlit app 才能發現資料有沒有問題。
    import sys

    target_path = sys.argv[1] if len(sys.argv) > 1 else "literature_db.xlsx"
    try:
        db = load_literature_db(target_path)
    except DBValidationError as e:
        print("驗證失敗：\n" + str(e))
        raise SystemExit(1)
    print(f"驗證通過：共 {len(db)} 筆資料，欄位：{list(db[0].keys())}")