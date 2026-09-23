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


def search_literature(question: str, literature_db: list[dict], limit: int = 5) -> list[dict]:
    """Small, explainable retrieval layer for the competition MVP."""
    q = _normalize(question)
    scored = []

    for idx, item in enumerate(literature_db):
        score = 0.0
        text = _record_text(item)

        # Process-gas mentions are strong signals.
        for gas, aliases in GAS_TERMS.items():
            if any(alias in q for alias in aliases):
                score += 3.0
                if float(item.get(gas, 0) or 0) > 0:
                    score += 4.0

        # Research intent terms.
        for field, aliases in FIELD_TERMS.items():
            if any(alias in q for alias in aliases):
                score += 1.5
                value = item.get(field)
                if value not in (None, "", "N/A"):
                    score += 1.0

        # Source/case literal overlap.
        for token in re.findall(r"[a-z0-9_.-]{3,}", q):
            if token in text:
                score += 0.5

        scored.append((score, idx, item))

    scored.sort(key=lambda x: (-x[0], x[1]))
    positive = [item for score, _, item in scored if score > 0]
    if positive:
        return positive[:limit]

    # Generic questions still get a small representative set instead of failing.
    return [item for _, _, item in scored[:limit]]


def evidence_dataframe(items: list[dict]) -> pd.DataFrame:
    rows = []
    for i, item in enumerate(items, start=1):
        rows.append({
            "證據": f"[{i}]",
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
                f"[{i}] source={item.get('source', '')}; case={item.get('case', '')}",
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
    instructions = """你是半導體乾式蝕刻研究助理。請只根據提供的 evidence 回答，不得補造文獻數據或最佳 recipe。

規則：
1. 回答使用繁體中文。
2. 重要敘述用 [1]、[2] 這種編號對應 evidence。
3. 不同文獻機台、樣品、功率或壓力不同時，要明確提醒不能直接當作單一 DOE 的因果結論。
4. 如果證據不足，直接說資料不足，並指出還缺什麼資料。
5. 可以整理趨勢與提出「下一步值得驗證的變因」，但不要給出宣稱最佳的製程參數。
6. 結尾用一小段「資料限制」說明目前回答的可靠範圍。
"""
    prompt = f"""使用者問題：
{question}

目前從本地 TiN 蝕刻資料庫檢索到的 evidence：
{_evidence_text(evidence)}

請依規則整理回答。"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=instructions,
            temperature=0.2,
        ),
    )
    return response.text or "模型沒有回傳文字內容。"
