from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SAFE = "safe"
UNSAFE = "unsafe"
UNKNOWN = "unknown"
VALID_LABELS = {SAFE, UNSAFE, UNKNOWN}
IMAGE_FIELD_KEYS = ("image", "image_path", "image_file", "img")
IMAGE_LIST_FIELD_KEYS = ("images", "image_paths", "image_files", "imgs")
IMAGE_MODALITIES = {"image", "images", "vision", "multimodal", "multi_modal"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def validate_label(label: str | None, *, allow_none: bool = False) -> str | None:
    if label is None and allow_none:
        return None
    if label not in VALID_LABELS:
        raise ValueError(f"label must be one of {sorted(VALID_LABELS)}, got {label!r}")
    return label


@dataclass(frozen=True)
class SafetyCase:
    id: str
    question: str
    answer: str | None = None
    label: str | None = None
    modality: str = "text"
    attachments: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SafetyCase":
        if not payload.get("question"):
            raise ValueError("case requires a non-empty question")
        label = validate_label(payload.get("label"), allow_none=True)
        metadata = dict(payload.get("metadata") or {})
        attachments = _coerce_string_list(payload.get("attachments") or [])
        image_refs = extract_image_references(payload)
        for image_ref in image_refs:
            if image_ref not in attachments:
                attachments.append(image_ref)

        modality = str(payload.get("modality") or "text")
        has_image = _payload_has_image(payload, image_refs=image_refs, attachments=attachments, modality=modality)
        if has_image:
            metadata.setdefault("has_image", True)
            metadata.setdefault("image_attachments", list(image_refs or attachments))
            if modality.casefold() == "text":
                modality = "image"

        return cls(
            id=str(payload.get("id") or "case"),
            question=str(payload["question"]),
            answer=payload.get("answer"),
            label=label,
            modality=modality,
            attachments=attachments,
            metadata=metadata,
        )

    def text_for_judging(self) -> str:
        parts = [self.question]
        if self.answer:
            parts.append(self.answer)
        return "\n".join(parts)

    def has_image(self) -> bool:
        if self.metadata.get("has_image") is True and self.attachments:
            return True
        if self.modality.casefold() in IMAGE_MODALITIES and self.attachments:
            return True
        return any(is_image_reference(attachment) for attachment in self.attachments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "label": self.label,
            "modality": self.modality,
            "attachments": list(self.attachments),
            "metadata": dict(self.metadata),
        }


@dataclass
class MethodResult:
    method_id: str
    label: str
    unsafe_score: float
    confidence: float
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False

    def __post_init__(self) -> None:
        validate_label(self.label)
        self.unsafe_score = _clamp01(self.unsafe_score)
        self.confidence = _clamp01(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "label": self.label,
            "unsafe_score": self.unsafe_score,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "metadata": dict(self.metadata),
            "skipped": self.skipped,
        }


@dataclass
class TraceStep:
    step_id: str
    method_id: str
    result: MethodResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "method_id": self.method_id,
            "result": self.result.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass
class RunTrace:
    case_id: str
    runner: str = "static"
    steps: list[TraceStep] = field(default_factory=list)
    stop_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: TraceStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "runner": self.runner,
            "steps": [step.to_dict() for step in self.steps],
            "stop_reason": self.stop_reason,
            "metadata": dict(self.metadata),
        }


@dataclass
class Decision:
    case_id: str
    label: str
    unsafe_score: float
    confidence: float
    reasons: list[str] = field(default_factory=list)
    trace: RunTrace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_label(self.label)
        self.unsafe_score = _clamp01(self.unsafe_score)
        self.confidence = _clamp01(self.confidence)

    def to_dict(self, *, include_trace: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "label": self.label,
            "unsafe_score": self.unsafe_score,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }
        if include_trace and self.trace is not None:
            payload["trace"] = self.trace.to_dict()
        return payload


@dataclass
class RunContext:
    run_id: str = ""
    llm_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def count_llm_call(self) -> None:
        self.llm_calls += 1


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def extract_image_references(payload: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in IMAGE_FIELD_KEYS:
        refs.extend(_image_reference_strings(payload.get(key)))
    for key in IMAGE_LIST_FIELD_KEYS:
        refs.extend(_image_reference_strings(payload.get(key)))
    return _dedupe_preserve_order(refs)


def is_image_reference(value: Any) -> bool:
    text = str(value).strip()
    if not text:
        return False
    lowered = text.casefold()
    if lowered.startswith("data:image/"):
        return True
    path_part = lowered.split("?", 1)[0].split("#", 1)[0]
    return any(path_part.endswith(extension) for extension in IMAGE_EXTENSIONS)


def _payload_has_image(
    payload: dict[str, Any],
    *,
    image_refs: list[str],
    attachments: list[str],
    modality: str,
) -> bool:
    if image_refs:
        return True
    payload_metadata = payload.get("metadata")
    if isinstance(payload_metadata, dict) and payload_metadata.get("has_image") is True and attachments:
        return True
    if modality.casefold() in IMAGE_MODALITIES and attachments:
        return True
    return any(is_image_reference(attachment) for attachment in attachments)


def _image_reference_strings(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        refs: list[str] = []
        for item in value:
            refs.extend(_image_reference_strings(item))
        return refs
    if isinstance(value, dict):
        for key in ("image", "image_path", "path", "url", "file", "image_url"):
            if key in value:
                return _image_reference_strings(value[key])
    return [str(value)]


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None and item != ""]
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
