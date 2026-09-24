import re
import time
from typing import Any

import pandas as pd
from google import genai
from google.genai import types

from references import short_source_label


FIELD_TERMS = {
    "angle": ["側壁", "側壁角度", "角度", "垂直", "profile", "sidewall", "angle"],
    "etch_rate_nm_min": ["蝕刻速率", "etch rate", "rate", "速率"],
    "selectivity": ["選擇比", "selectivity"],
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

OUTCOME_ALIASES = {
    "angle": FIELD_TERMS["angle"],
    "etch_rate_nm_min": FIELD_TERMS["etch_rate_nm_min"],
    "selectivity": FIELD_TERMS["selectivity"],
}

OUTCOME_LABELS = {
    "angle": "側壁角度",
    "etch_rate_nm_min": "蝕刻速率",
    "selectivity": "選擇比",
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


PHOTORESIST_TERMS = ["光阻", "photoresist", "mask", "遮罩", "pr thickness", "pr etch", "mask erosion"]


def analyze_question_support(question: str) -> dict[str, Any]:
    """Judge whether the current structured database can directly support the question."""
    q = _normalize(question)

    # Generic photoresist-role questions need fields that are not currently stored.
    asks_photoresist = any(term in q for term in PHOTORESIST_TERMS)
    asks_selectivity = any(term in q for term in ["選擇比", "selectivity"])
    if asks_photoresist and not asks_selectivity:
        return {
            "supported": False,
            "topic": "光阻",
            "message": (
                "目前資料庫沒有光阻厚度、光阻蝕刻率或遮罩消耗等欄位；"
                "現有選擇比資料只能作背景，不能直接回答光阻在模擬器中的角色。"
            ),
        }

    return {"supported": True, "topic": None, "message": ""}


def detect_question_type(question: str) -> str:
    """Classify the research intent so retrieval can use the right logic."""
    q = _normalize(question)

    lookup_patterns = [
        "哪一組", "哪組", "最高", "最低", "最大", "最小",
        "最垂直", "最快", "最慢", "排名", "top", "highest", "lowest",
    ]
    if any(term in q for term in lookup_patterns):
        return "lookup"

    support = analyze_question_support(question)
    if not support["supported"]:
        return "unsupported"

    impact_patterns = [
        "影響", "增加", "降低", "提升", "下降", "變化", "關係",
        "effect", "impact", "increase", "decrease",
    ]
    if any(term in q for term in impact_patterns):
        return "impact"

    return "descriptive"


def _lookup_direction(question: str) -> str:
    q = _normalize(question)
    ascending_terms = ["最低", "最小", "最慢", "lowest", "minimum"]
    return "asc" if any(term in q for term in ascending_terms) else "desc"


def detect_target_variable(question: str) -> str | None:
    q = _normalize(question)
    for field, aliases in TARGET_ALIASES.items():
        if any(alias in q for alias in aliases):
            return field
    return None


def detect_outcome_variable(question: str) -> str | None:
    q = _normalize(question)
    for field, aliases in OUTCOME_ALIASES.items():
        if any(alias in q for alias in aliases):
            return field
    return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        value = float(value)
        if pd.isna(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def _has_value(item: dict, field: str | None) -> bool:
    if field is None:
        return True
    return _as_float(item.get(field)) is not None


def _normalized_distance(a: Any, b: Any, scale: float = 1.0) -> float:
    av = _as_float(a)
    bv = _as_float(b)
    if av is None or bv is None:
        return 1.0
    return abs(av - bv) / max(scale, abs(av), abs(bv), 1.0)


def _pair_comparability(a: dict, b: dict, target: str | None) -> tuple[float, str, str]:
    if target is None:
        return 0.0, "一般相關案例", "C"

    same_source = str(a.get("source", "")).strip() == str(b.get("source", "")).strip()
    score = 3.0 if same_source else 0.0

    target_diff = _normalized_distance(a.get(target), b.get(target))
    if target_diff <= 0.01:
        return -10.0, "目標變因沒有改變", "C"

    score += min(target_diff, 2.0) * 1.5

    penalty = 0.0
    changed_context = 0
    for field in CONTEXT_FIELDS:
        if field == target:
            continue
        scale = 10.0 if field in {"BCl3", "Cl2", "Ar", "N2"} else 100.0
        dist = _normalized_distance(a.get(field), b.get(field), scale=scale)
        penalty += dist
        if dist > 0.08:
            changed_context += 1

    score -= penalty

    if same_source and penalty <= 0.22 and changed_context <= 1:
        return score, "A｜高可比：同來源且其他條件近似", "A"
    if same_source and penalty <= 0.85 and changed_context <= 2:
        return score, "B｜可參考：同來源，但仍有其他條件差異", "B"
    if (not same_source) and penalty <= 0.35 and changed_context <= 1:
        return score, "B｜可參考：輸出完整，但仍有來源或條件差異", "B"
    return score, "C｜背景：僅能作相關案例參考", "C"


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
            if _has_value(item, field):
                score += 1.0

    for token in re.findall(r"[a-z0-9_.-]{3,}", q):
        if token in text:
            score += 0.5

    return score


def search_literature(question: str, literature_db: list[dict], limit: int = 5) -> list[dict]:
    """
    Evidence hierarchy:
      A: target outcome exists + target variable changes + other conditions nearly fixed.
      B: outcome exists + useful comparison, but confounders remain.
      C: relevant background only; not primary causal evidence.
    """
    target = detect_target_variable(question)
    outcome = detect_outcome_variable(question)
    support = analyze_question_support(question)
    question_type = detect_question_type(question)

    if question_type == "lookup" and outcome is not None:
        rows = []
        for idx, item in enumerate(literature_db):
            value = _as_float(item.get(outcome))
            if value is None:
                continue
            enriched = dict(item)
            enriched["_comparability"] = "查詢結果｜依目標欄位排序"
            enriched["_evidence_tier"] = "Q"
            enriched["_target_variable"] = target
            enriched["_outcome_variable"] = outcome
            enriched["_question_type"] = "lookup"
            rows.append((value, idx, enriched))

        reverse = _lookup_direction(question) == "desc"
        rows.sort(key=lambda x: (x[0], -x[1]) if not reverse else (-x[0], x[1]))

        if rows:
            best_value = rows[0][0]
            tied = [item for value, _, item in rows if abs(value - best_value) < 1e-9]
            if len(tied) >= limit:
                return tied[:limit]

            selected = tied[:]
            seen = {(str(x.get("source","")), str(x.get("case",""))) for x in selected}
            for _, _, item in rows:
                key = (str(item.get("source","")), str(item.get("case","")))
                if key in seen:
                    continue
                selected.append(item)
                seen.add(key)
                if len(selected) >= limit:
                    break
            return selected

    if not support["supported"]:
        # Show only related background records; none may be treated as A/B evidence.
        ranked = []
        for idx, item in enumerate(literature_db):
            score = _base_relevance(question, item)
            # For generic photoresist questions, selectivity-bearing rows are the most relevant background.
            if support.get("topic") == "光阻" and _has_value(item, "selectivity"):
                score += 3.0
            enriched = dict(item)
            enriched["_comparability"] = "C｜背景：目前資料庫缺少此問題所需的直接欄位"
            enriched["_evidence_tier"] = "C"
            enriched["_target_variable"] = target
            enriched["_outcome_variable"] = outcome
            enriched["_support_message"] = support["message"]
            ranked.append((score, idx, enriched))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        return [item for _, _, item in ranked[:limit]]

    usable = [item for item in literature_db if _has_value(item, outcome)]
    background = [item for item in literature_db if item not in usable]

    if target is not None and outcome is not None and len(usable) >= 2:
        pair_rows = []
        for i, a in enumerate(usable):
            for j in range(i + 1, len(usable)):
                b = usable[j]
                pair_score, label, tier = _pair_comparability(a, b, target)
                relevance = (_base_relevance(question, a) + _base_relevance(question, b)) / 2
                tier_bonus = {"A": 12.0, "B": 6.0, "C": 0.0}[tier]
                total = pair_score + relevance + tier_bonus
                pair_rows.append((total, tier, label, a, b))

        pair_rows.sort(key=lambda x: ({"A": 0, "B": 1, "C": 2}[x[1]], -x[0]))

        selected = []
        seen = set()

        # Primary evidence: A then B. C is added only if space remains.
        for wanted_tier in ("A", "B", "C"):
            for _, tier, label, a, b in pair_rows:
                if tier != wanted_tier:
                    continue
                for item in (a, b):
                    key = (str(item.get("source", "")), str(item.get("case", "")))
                    if key in seen:
                        continue
                    enriched = dict(item)
                    enriched["_comparability"] = label
                    enriched["_evidence_tier"] = tier
                    enriched["_target_variable"] = target
                    enriched["_outcome_variable"] = outcome
                    selected.append(enriched)
                    seen.add(key)
                    if len(selected) >= limit:
                        return selected

        if selected:
            return selected

    # Fallback for descriptive questions or when no pair can be formed.
    scored = []
    for idx, item in enumerate(usable):
        score = _base_relevance(question, item)
        enriched = dict(item)
        enriched["_comparability"] = (
            "B｜可參考：輸出完整，但缺乏單一變因對照"
            if outcome is not None
            else "一般相關案例"
        )
        enriched["_evidence_tier"] = "B" if outcome is not None else "C"
        enriched["_target_variable"] = target
        enriched["_outcome_variable"] = outcome
        scored.append((score, idx, enriched))

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = [item for score, _, item in scored if score > 0][:limit]

    # If no usable evidence exists, show only background records and mark them unusable.
    if not selected and background:
        for item in background[:limit]:
            enriched = dict(item)
            label = OUTCOME_LABELS.get(outcome, outcome or "目標輸出")
            enriched["_comparability"] = f"C｜背景：缺少{label}實測值"
            enriched["_evidence_tier"] = "C"
            enriched["_target_variable"] = target
            enriched["_outcome_variable"] = outcome
            selected.append(enriched)

    return selected


def evidence_dataframe(items: list[dict]) -> pd.DataFrame:
    rows = []
    for i, item in enumerate(items, start=1):
        rows.append({
            "證據": f"[{i}]",
            "證據層級": item.get("_evidence_tier", "C"),
            "可比較性": item.get("_comparability", "一般相關案例"),
            "來源": short_source_label(item.get("source", "")),
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


def _human_value(value: Any, suffix: str = "") -> str:
    number = _as_float(value)
    if number is None:
        return "未提供"
    if abs(number - round(number)) < 1e-9:
        text = str(int(round(number)))
    else:
        text = f"{number:.2f}".rstrip("0").rstrip(".")
    return f"{text}{suffix}"


def _human_case_summary(item: dict) -> str:
    return (
        f"來源：{short_source_label(item.get('source', ''))}\n"
        f"案例：{item.get('case', '')}\n"
        f"氣體流量：BCl₃ {_human_value(item.get('BCl3'), ' sccm')}、"
        f"Cl₂ {_human_value(item.get('Cl2'), ' sccm')}、"
        f"Ar {_human_value(item.get('Ar'), ' sccm')}、"
        f"N₂ {_human_value(item.get('N2'), ' sccm')}\n"
        f"製程壓力：{_human_value(item.get('pressure'), ' mTorr')}\n"
        f"ICP / Source Power：{_human_value(item.get('source_power'), ' W')}\n"
        f"Bias / Chuck：{_human_value(item.get('bias'))}\n"
        f"側壁角度：{_human_value(item.get('angle'), '°')}\n"
        f"TiN 蝕刻速率：{_human_value(item.get('etch_rate_nm_min'), ' nm/min')}\n"
        f"TiN:PR 選擇比：{_human_value(item.get('selectivity'))}"
    )


def _evidence_text(items: list[dict]) -> str:
    blocks = []
    for i, item in enumerate(items, start=1):
        blocks.append(
            "\n".join([
                f"[{i}] 證據層級：{item.get('_evidence_tier', 'C')}",
                f"可比較性：{item.get('_comparability', '一般相關案例')}",
                f"資料支援提醒：{item.get('_support_message', '') or '無'}",
                _human_case_summary(item),
            ])
        )
    return "\n\n".join(blocks)


def generate_ai_answer(
    question: str,
    evidence: list[dict],
    api_key: str,
    model: str = "gemini-3.5-flash-lite",
    fallback_model: str = "gemini-3.8-flash",
) -> str:
    if not evidence:
        return "目前資料庫沒有找到可用的文獻案例，因此不產生製程結論。"

    client = genai.Client(api_key=api_key)
    target = detect_target_variable(question)
    outcome = detect_outcome_variable(question)
    support = analyze_question_support(question)
    question_type = detect_question_type(question)

    instructions = """你是半導體乾式蝕刻研究助理。請只根據提供的 evidence 回答，不得補造文獻數據或最佳 recipe。

規則：
1. 回答使用繁體中文。
2. 每一個重要結論句只在句尾引用一次 [1]、[2] 等證據編號。
3. 證據層級 A = 最佳對照；B = 可參考但仍有混雜變因；C = 只能作背景。
4. 如果使用者問某個變因增加/降低對某個輸出的影響，只有 A 或 B 證據且該輸出有實測值時，才可以討論可能趨勢。
5. C 級資料不能拿來支持因果或方向性結論。
6. 如果沒有 A/B 證據，直接說目前資料庫不足以回答，不要勉強推論。
7. 如果 support_supported=False，必須明確說明資料庫缺少哪些欄位；所有 C 級資料只能作背景，不得把它們包裝成可回答該問題的證據。
8. 若其他條件同時改變，要明確指出混雜變因，不能把結果單獨歸因於目標變因。
9. 可以提出下一步值得做的單一變因對照實驗，但不要宣稱最佳 recipe。
10. 不要把內部欄位名稱如 tier=、comparability=、support_note=、source_power=、etch_rate=、angle= 原樣輸出給使用者。
11. 所有輸出要用一般研究者看得懂的中文名稱，例如：
- source_power → ICP / Source Power
- bias → Bias / Chuck
- angle → 側壁角度
- etch_rate_nm_min → TiN 蝕刻速率
- selectivity → TiN:PR 選擇比
來源請使用 P 編號＋簡稱，不要顯示完整內部檔名。
12. 若 question_type=lookup，請改用以下三個 Markdown 小節：
## 結論
直接指出目前資料庫中的最高/最低或符合查詢條件的案例；若有並列要全部列出。
## 條件摘要
簡潔列出主要案例的製程條件與目標數值。使用自然中文與單位，不要輸出程式欄位名稱；每個案例最多 4–5 行。
## 提醒
說明這只是目前資料庫中的紀錄，不代表最佳製程，也不代表因果關係。
13. 若 question_type 不是 lookup，請固定使用以下四個 Markdown 小節輸出：
## 結論
先用 1–2 句直接回答目前證據能否支持使用者的問題。
## 證據依據
只列最重要的 A/B 級 evidence；若只有 C 級，明確說只能作背景。
## 下一步建議
只提出 1–2 個可驗證、範圍小的實驗或資料補強方向。若只有一項建議，請直接寫成一小段，不要使用「1.」編號；只有兩項時才使用編號列表。
## 資料限制
簡短說明目前證據能支持到什麼程度。
"""

    target_note = (
        f"本題辨識的主要變因：{target}。"
        if target is not None
        else "本題沒有辨識到單一主要變因。"
    )
    outcome_note = (
        f"本題辨識的觀察輸出：{OUTCOME_LABELS.get(outcome, outcome)}。"
        if outcome is not None
        else "本題沒有辨識到單一觀察輸出。"
    )

    support_note = (
        f"資料支援判斷：不足。{support['message']}"
        if not support["supported"]
        else "資料支援判斷：可進一步依 A/B/C 證據層級分析。"
    )

    prompt = f"""使用者問題：
{question}

question_type={question_type}

{question}

{target_note}
{outcome_note}
{support_note}

目前從本地 TiN 蝕刻資料庫檢索到的 evidence：
{_evidence_text(evidence)}

若 question_type=lookup，請只根據排序後 evidence 做資料庫查詢摘要，不需要做 A/B/C 因果判斷。
其他問題則依 A/B/C 證據層級回答；若沒有足夠的 A/B 證據，直接說無法由現有資料判斷獨立影響。"""

    models_to_try = []
    for name in (model, fallback_model):
        if name and name not in models_to_try:
            models_to_try.append(name)

    last_error = None
    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=instructions,
                        temperature=0.2,
                    ),
                )
                return response.text or "模型沒有回傳文字內容。"
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                is_transient = (
                    "503" in message
                    or "unavailable" in message
                    or "high demand" in message
                    or "429" in message
                    or "resource_exhausted" in message
                )
                if is_transient and attempt == 0:
                    time.sleep(1.2)
                    continue
                break

    raise RuntimeError("AI_SERVICE_BUSY") from last_error
