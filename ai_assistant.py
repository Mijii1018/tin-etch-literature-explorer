import math
import re
from typing import Any

import pandas as pd
from google import genai
from google.genai import types


FIELD_TERMS = {
    "sidewall": ["側壁", "角度", "垂直", "profile", "sidewall", "angle"],
    "etch_rate": ["蝕刻速率", "etch rate", "rate", "速率"],
    "selectivity": ["選擇比", "selectivity", "光阻", "photoresist", "pr"],
    "pressure": ["壓力", "pressure", "mtorr"],
    "bias": ["bias", "偏壓", "chuck"],
    "source_power": ["icp", "source power", "功率", "源功率"],
}

GAS_TERMS = {
    "BCl3": ["bcl3", "bcl₃"],
    "Cl2": ["cl2", "cl₂"],
    "Ar": [" ar ", "氬", "argon"],
    "N2": ["n2", "n₂", "氮"],
}

TARGET_ALIASES = {
    "Ar": GAS_TERMS["Ar"],
    "Cl2": GAS_TERMS["Cl2"],
    "BCl3": GAS_TERMS["BCl3"],
    "N2": GAS_TERMS["N2"],
    "pressure": FIELD_TERMS["pressure"],
    "bias": FIELD_TERMS["bias"],
    "source_power": FIELD_TERMS["source_power"],
}

CONTEXT_FIELDS = ["BCl3", "Cl2", "Ar", "N2", "pressure", "bias", "source_power"]


def _normalize(text: str) -> str:
    return " " + re.sub(r"\s+", " ", str(text).lower()).strip() + " "


def _record_text(item: dict[str, Any]) -> str:
    parts = [
        item.get("source", ""),
        item.get("case", ""),
        item.get("chemistry", ""),
        f"BCl3 {item.get('BCl3', '')}",
        f"Cl2 {item.get('Cl2', '')}",
        f"Ar {item.get('Ar', '')}",
        f"N2 {item.get('N2', '')}",
        f"pressure {item.get('pressure', '')}",
        f"bias {item.get('bias', '')}",
        f"source power {item.get('source_power', '')}",
        f"angle {item.get('angle', '')}",
        f"etch rate {item.get('etch_rate_nm_min', '')}",
        f"selectivity {item.get('selectivity', '')}",
    ]
    return _normalize(" ".join(map(str, parts)))


def detect_target_variable(question: str) -> str | None:
    q = _normalize(question)
    for field, aliases in TARGET_ALIASES.items():
        if any(alias in q for alias in aliases):
            return field
    return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_distance(a: Any, b: Any, scale: float = 1.0) -> float:
    av = _as_float(a)
    bv = _as_float(b)
    if av is None or bv is None:
        return 1.0
    return abs(av - bv) / max(scale, abs(av), abs(bv), 1.0)


def _pair_comparability(a: dict, b: dict, target: str | None) -> tuple[float, str]:
    if target is None:
        return 0.0, "一般相關案例"

    same_source = str(a.get("source", "")).strip() == str(b.get("source", "")).strip()
    score = 3.0 if same_source else 0.0

    target_diff = _normalized_distance(a.get(target), b.get(target))
    if target_diff <= 0.01:
        return -5.0, "目標變因沒有改變"

    score += min(target_diff, 2.0) * 1.5

    penalty = 0.0
    for field in CONTEXT_FIELDS:
        if field == target:
            continue
        scale = 10.0 if field in {"BCl3", "Cl2", "Ar", "N2"} else 100.0
        penalty += _normalized_distance(a.get(field), b.get(field), scale=scale)

    score -= penalty

    if same_source and penalty <= 0.30:
        label = "高可比：同來源且其他條件近似"
    elif same_source and penalty <= 0.80:
        label = "中可比：同來源但其他條件仍有差異"
    elif penalty <= 0.45:
        label = "中可比：跨來源但其他條件較接近"
    else:
        label = "低可比：僅能作相關案例參考"

    return score, label


def _base_relevance(question: str, item: dict[str, Any]) -> float:
    q = _normalize(question)
    text = _record_text(item)
    score = 0.0

    for gas, aliases in GAS_TERMS.items():
        if any(alias in q for alias in aliases):
            score += 3.0
            if float(item.get(gas, 0) or 0) > 0:
                score += 4.0

    for field, aliases in FIELD_TERMS.items():
        if any(alias in q for alias in aliases):
            score += 1.5
            value = item.get(field)
            if value not in (None, "", "N/A"):
                score += 1.0

    for token in re.findall(r"[a-z0-9_.-]{3,}", q):
        if token in text:
            score += 0.5

    return score


def search_literature(question: str, literature_db: list[dict], limit: int = 5) -> list[dict]:
    """Retrieve relevant evidence, prioritizing comparable cases for one-variable questions."""
    target = detect_target_variable(question)

    if target is not None and len(literature_db) >= 2:
        pair_rows = []
        for i, a in enumerate(literature_db):
            for j in range(i + 1, len(literature_db)):
                b = literature_db[j]
                pair_score, label = _pair_comparability(a, b, target)
                relevance = (_base_relevance(question, a) + _base_relevance(question, b)) / 2
                total = pair_score + relevance
                pair_rows.append((total, label, a, b))

        pair_rows.sort(key=lambda x: x[0], reverse=True)

        selected = []
        seen = set()
        for _, label, a, b in pair_rows:
            for item in (a, b):
                key = (str(item.get("source", "")), str(item.get("case", "")))
                if key in seen:
                    continue
                enriched = dict(item)
                enriched["_comparability"] = label
                enriched["_target_variable"] = target
                selected.append(enriched)
                seen.add(key)
                if len(selected) >= limit:
                    return selected

    scored = []
    for idx, item in enumerate(literature_db):
        score = _base_relevance(question, item)
        enriched = dict(item)
        enriched["_comparability"] = "一般相關案例"
        enriched["_target_variable"] = target
        scored.append((score, idx, enriched))

    scored.sort(key=lambda x: (-x[0], x[1]))
    positive = [item for score, _, item in scored if score > 0]
    if positive:
        return positive[:limit]
    return [item for _, _, item in scored[:limit]]


def evidence_dataframe(items: list[dict]) -> pd.DataFrame:
    rows = []
    for i, item in enumerate(items, start=1):
        rows.append({
            "證據": f"[{i}]",
            "可比較性": item.get("_comparability", "一般相關案例"),
            "來源": item.get("source", ""),
            "案例": item.get("case", ""),
            "BCl₃": item.get("BCl3"),
            "Cl₂": item.get("Cl2"),
            "Ar": item.get("Ar"),
            "N₂": item.get("N2"),
            "壓力(mTorr)": item.get("pressure"),
            "ICP/Source(W)": item.get("source_power"),
            "Bias/Chuck": item.get("bias"),
            "側壁角度(°)": item.get("angle"),
            "蝕刻速率(nm/min)": item.get("etch_rate_nm_min"),
            "選擇比": item.get("selectivity"),
        })
    return pd.DataFrame(rows)


def _evidence_text(items: list[dict]) -> str:
    blocks = []
    for i, item in enumerate(items, start=1):
        blocks.append(
            "\n".join([
                f"[{i}] comparability={item.get('_comparability', '一般相關案例')}",
                f"source={item.get('source', '')}; case={item.get('case', '')}",
                (
                    "gas(sccm): "
                    f"BCl3={item.get('BCl3')}, Cl2={item.get('Cl2')}, "
                    f"Ar={item.get('Ar')}, N2={item.get('N2')}"
                ),
                (
                    f"pressure={item.get('pressure')} mTorr; "
                    f"source_power={item.get('source_power')} W; bias={item.get('bias')}"
                ),
                (
                    f"angle={item.get('angle')} deg; "
                    f"etch_rate={item.get('etch_rate_nm_min')} nm/min; "
                    f"selectivity={item.get('selectivity')}"
                ),
            ])
        )
    return "\n\n".join(blocks)


def generate_ai_answer(
    question: str,
    evidence: list[dict],
    api_key: str,
    model: str = "gemini-3.5-flash-lite",
) -> str:
    if not evidence:
        return "目前資料庫沒有找到可用的文獻案例，因此不產生製程結論。"

    client = genai.Client(api_key=api_key)
    target = detect_target_variable(question)

    instructions = """你是半導體乾式蝕刻研究助理。請只根據提供的 evidence 回答，不得補造文獻數據或最佳 recipe。

規則：
1. 回答使用繁體中文。
2. 每一個重要結論句只在句尾引用一次 [1]、[2] 等證據編號，不要在句首和句尾重複 citation。
3. 先判斷 evidence 的「可比較性」。高可比資料可以用來討論較可信的趨勢；低可比資料只能當背景參考。
4. 如果使用者問的是某個變因增加/降低造成的影響，只有在其他條件近似時，才可以說「可能呈現某趨勢」；否則要明確說無法分離該變因本身的效果。
5. 不同文獻機台、樣品、功率或壓力不同時，要提醒不能直接當作單一 DOE 的因果結論。
6. 如果證據不足，直接說資料不足，並指出還缺什麼對照資料。
7. 可以提出「下一步值得驗證的變因」，但不要給出宣稱最佳的製程參數。
8. 結尾用一小段「資料限制」說明目前回答的可靠範圍。
"""

    target_note = (
        f"本題辨識的主要變因：{target}。"
        if target is not None
        else "本題沒有辨識到單一主要變因。"
    )

    prompt = f"""使用者問題：
{question}

{target_note}

目前從本地 TiN 蝕刻資料庫檢索到的 evidence：
{_evidence_text(evidence)}

請先區分「可以比較」與「僅相關」的資料，再整理回答。"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=instructions,
            temperature=0.2,
        ),
    )
    return response.text or "模型沒有回傳文字內容。"
