from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any

from safeguard_harness.core import SAFE, UNSAFE, Decision, SafetyCase


def load_jsonl_cases(path: str | Path) -> list[SafetyCase]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []

    if source.suffix.casefold() == ".json" or stripped.startswith("["):
        return _load_json_cases(source, stripped)

    if stripped.startswith("{") and "\n" not in stripped:
        try:
            return _cases_from_json_payload(json.loads(stripped), path=source)
        except json.JSONDecodeError:
            pass

    cases: list[SafetyCase] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped_line = line.strip()
        if not stripped_line:
            continue
        try:
            payload = json.loads(stripped_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        cases.append(_case_from_payload(payload, path=source, index=line_number))
    return cases


def _load_json_cases(path: Path, text: str) -> list[SafetyCase]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at {path}") from exc
    return _cases_from_json_payload(payload, path=path)


def _cases_from_json_payload(payload: Any, *, path: Path) -> list[SafetyCase]:
    if isinstance(payload, list):
        raw_cases = payload
    elif isinstance(payload, dict) and (payload.get("question") or "messages" in payload):
        raw_cases = [payload]
    elif isinstance(payload, dict):
        raw_cases = _first_case_list(payload)
    else:
        raise ValueError(f"JSON input at {path} must be a case object or a list of case objects")

    return [_case_from_payload(item, path=path, index=index) for index, item in enumerate(raw_cases, start=1)]


def _first_case_list(payload: dict[str, Any]) -> list[Any]:
    for key in ("cases", "data", "records", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise ValueError("JSON input object must contain question/messages or a cases/data/records/items list")


def _case_from_payload(payload: Any, *, path: Path, index: int) -> SafetyCase:
    if not isinstance(payload, dict):
        raise ValueError(f"case at {path}:{index} must be a JSON object")
    normalized = _normalize_case_payload(payload)
    if "id" in payload:
        metadata = dict(normalized.get("metadata") or {})
        metadata.setdefault("raw_id", payload["id"])
        normalized = {**normalized, "metadata": metadata}
    return SafetyCase.from_dict(normalized)


def _normalize_case_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("question") or "messages" not in payload:
        return payload

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload

    user_texts = _message_texts(messages, role="user")
    assistant_texts = _message_texts(messages, role="assistant")
    image_refs = _message_images(messages)
    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "source_format": "messages",
            "messages": messages,
        }
    )
    for key in ("type", "is_mt", "MT", "mt", "source"):
        if key in payload:
            metadata[key] = payload[key]
    if "is_mt" not in metadata:
        for alias in ("MT", "mt"):
            if alias in metadata:
                metadata["is_mt"] = metadata[alias]
                break

    normalized = dict(payload)
    normalized["question"] = "\n\n".join(user_texts).strip()
    normalized["answer"] = "\n\n".join(assistant_texts).strip() or payload.get("answer")
    if image_refs:
        normalized["attachments"] = list(payload.get("attachments") or []) + image_refs
        normalized["modality"] = "image"
        metadata["has_image"] = True
        metadata["image_attachments"] = image_refs
    normalized["metadata"] = metadata
    return normalized


def _message_texts(messages: list[Any], *, role: str) -> list[str]:
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != role:
            continue
        text = _content_text(message.get("content")).strip()
        if text:
            texts.append(text)
    return texts


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
            parts.append(str(item["text"]))
    return "\n".join(parts)


def _message_images(messages: list[Any]) -> list[str]:
    refs: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        refs.extend(_content_images(message.get("content")))
    return _dedupe_preserve_order(refs)


def _content_images(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []

    refs: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").casefold()
        if item_type in {"image", "input_image"}:
            refs.extend(_image_value_strings(item.get("image") or item.get("path") or item.get("url")))
        elif item_type in {"image_id", "input_image_id"}:
            refs.extend(_image_value_strings(item.get("image_id") or item.get("id")))
        elif item_type in {"image_url", "input_image_url"}:
            refs.extend(_image_value_strings(item.get("image_url") or item.get("url")))
    return refs


def _image_value_strings(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        refs: list[str] = []
        for item in value:
            refs.extend(_image_value_strings(item))
        return refs
    if isinstance(value, dict):
        for key in ("url", "path", "image", "image_path", "file"):
            if key in value:
                return _image_value_strings(value[key])
    return [str(value)]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def deliverable_result_row(case: SafetyCase, decision: Decision) -> dict[str, int | str]:
    if decision.label == UNSAFE:
        result = 1
    elif decision.label == SAFE:
        result = 0
    else:
        raise ValueError(f"deliverable result requires safe/unsafe decision, got {decision.label!r}")
    return {"id": case.metadata.get("raw_id", case.id), "result": result}
