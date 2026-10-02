"""Microsoft Foundry-backed research assistant helpers.

The module intentionally has no Streamlit dependency so evidence selection and
request construction can be tested independently from the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

import requests

from references import reference_for_source


SYSTEM_INSTRUCTIONS = """你是 TiN 乾式蝕刻文獻研究助理。請以繁體中文回答。
你只能根據系統提供的結構化證據回答，並遵守以下規則：
1. 每一個重要判斷都要標示資料來源編號，例如 [P010]。
2. 不同文獻若同時改變氣體、壓力、功率、樣品或設備，不得宣稱單一變因造成結果。
3. 資料不足時要直接指出缺少哪些欄位或對照，不得補造數值。
4. 先給結論，再列證據、限制與下一步建議。
5. 本系統只能協助文獻探索，不得把結果描述成可直接使用的製程配方。
"""


class FoundryError(RuntimeError):
    """Raised when the Foundry configuration or request is invalid."""


@dataclass(frozen=True)
class FoundrySettings:
    endpoint: str = ""
    api_key: str = ""
    deployment: str = ""

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "FoundrySettings":
        values = values or {}
        return cls(
            endpoint=str(values.get("endpoint", "")).strip(),
            api_key=str(values.get("api_key", "")).strip(),
            deployment=str(values.get("deployment", "")).strip(),
        )

    @property
    def configured(self) -> bool:
        return bool(self.endpoint and self.api_key and self.deployment)

    @property
    def chat_completions_url(self) -> str:
        endpoint = self.endpoint.rstrip("/")
        if endpoint.endswith("/openai/v1"):
            return f"{endpoint}/chat/completions"
        return f"{endpoint}/openai/v1/chat/completions"


QUESTION_TYPES = {
    "排序查詢": ("最高", "最低", "最大", "最小", "哪一組", "排行", "排名"),
    "製程影響": ("影響", "增加", "降低", "改變", "趨勢", "關係", "造成"),
}

METRIC_ALIASES = {
    "angle": ("角度", "側壁", "profile"),
    "etch_rate_nm_min": ("蝕刻速率", "速率", "etch rate"),
    "selectivity": ("選擇比", "selectivity"),
}

PROCESS_ALIASES = {
    "BCl3": ("bcl3", "bcl₃"),
    "Cl2": ("cl2", "cl₂"),
    "Ar": (" ar", "氬"),
    "N2": ("n2", "n₂", "氮"),
    "pressure": ("壓力", "pressure", "mtorr"),
    "source_power": ("icp", "source", "tcp"),
    "bias": ("bias", "chuck", "偏壓"),
}


def classify_question(question: str) -> str:
    normalized = question.casefold()
    for label, terms in QUESTION_TYPES.items():
        if any(term.casefold() in normalized for term in terms):
            return label
    return "文獻探索"


def _detect_metric(question: str) -> str | None:
    normalized = question.casefold()
    for field, aliases in METRIC_ALIASES.items():
        if any(alias.casefold() in normalized for alias in aliases):
            return field
    return None


def _record_score(record: Mapping[str, Any], question: str) -> float:
    normalized = f" {question.casefold()} "
    score = 0.0
    for field, aliases in PROCESS_ALIASES.items():
        if any(alias.casefold() in normalized for alias in aliases):
            value = record.get(field)
            if isinstance(value, (int, float)) and value > 0:
                score += 2.0
    note = str(record.get("note", "")).casefold()
    source = str(record.get("source", "")).casefold()
    for token in normalized.replace("？", " ").replace("?", " ").split():
        if len(token) >= 3 and (token in note or token in source):
            score += 0.25
    return score


def select_evidence(
    records: Sequence[Mapping[str, Any]], question: str, limit: int = 6
) -> list[dict[str, Any]]:
    """Select a compact, deterministic evidence set for the model prompt."""

    if limit < 1:
        raise ValueError("limit must be at least 1")

    metric = _detect_metric(question)
    normalized = question.casefold()
    reverse = not any(term in normalized for term in ("最低", "最小", "降低"))

    ranked = list(records)
    if metric and any(term in normalized for term in QUESTION_TYPES["排序查詢"]):
        ranked.sort(
            key=lambda item: (
                item.get(metric) is not None,
                float(item.get(metric) or 0),
            ),
            reverse=reverse,
        )
    else:
        ranked.sort(key=lambda item: _record_score(item, question), reverse=True)

    evidence: list[dict[str, Any]] = []
    for record in ranked[:limit]:
        ref = reference_for_source(record.get("source"))
        evidence.append(
            {
                "reference": ref["id"] if ref else str(record.get("source", "")),
                "case": record.get("case"),
                "chemistry": record.get("chemistry"),
                "BCl3_sccm": record.get("BCl3"),
                "Cl2_sccm": record.get("Cl2"),
                "Ar_sccm": record.get("Ar"),
                "N2_sccm": record.get("N2"),
                "pressure_mTorr": record.get("pressure"),
                "source_power_W": record.get("source_power"),
                "bias": record.get("bias"),
                "sidewall_angle_deg": record.get("angle"),
                "etch_rate_nm_min": record.get("etch_rate_nm_min"),
                "selectivity": record.get("selectivity"),
                "note": record.get("note"),
            }
        )
    return evidence


def ask_foundry(
    settings: FoundrySettings,
    question: str,
    evidence: Sequence[Mapping[str, Any]],
    process_context: Mapping[str, Any],
    history: Sequence[Mapping[str, str]] | None = None,
    timeout: int = 60,
) -> str:
    """Send a grounded question to a Microsoft Foundry model deployment."""

    if not settings.configured:
        raise FoundryError("Microsoft Foundry 尚未完成設定。")

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
    for item in list(history or [])[-6:]:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    grounded_prompt = {
        "question_type": classify_question(question),
        "question": question,
        "current_process_conditions": dict(process_context),
        "selected_literature_evidence": list(evidence),
    }
    messages.append(
        {
            "role": "user",
            "content": "請分析以下 JSON。資料之外的內容一律視為未知。\n"
            + json.dumps(grounded_prompt, ensure_ascii=False, default=str),
        }
    )

    try:
        response = requests.post(
            settings.chat_completions_url,
            headers={
                "Content-Type": "application/json",
                "api-key": settings.api_key,
            },
            json={
                "model": settings.deployment,
                "messages": messages,
                "max_tokens": 900,
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise FoundryError(f"無法連線至 Microsoft Foundry：{exc}") from exc

    if not response.ok:
        detail = response.text.strip().replace("\n", " ")[:400]
        raise FoundryError(f"Microsoft Foundry 回傳 HTTP {response.status_code}：{detail}")

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise FoundryError("Microsoft Foundry 回傳了無法辨識的資料格式。") from exc

    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) if isinstance(part, Mapping) else str(part)
            for part in content
        )
    answer = str(content).strip()
    if not answer:
        raise FoundryError("Microsoft Foundry 沒有產生回答。")
    return answer
