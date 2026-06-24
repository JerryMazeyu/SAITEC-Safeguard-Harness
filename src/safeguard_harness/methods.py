from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from safeguard_harness.core import SAFE, UNKNOWN, UNSAFE, MethodResult, RunContext, SafetyCase
from safeguard_harness.internal_llm import InternalLlmJudge
from safeguard_harness.providers import BinaryModelOutput, parse_binary_label_from_text

DEFAULT_REFUSAL_MARKERS = [
    "cannot help",
    "can't help",
    "cannot provide",
    "unable to assist",
    "unsafe request",
    "must decline",
    "i decline",
    "拒绝",
    "不能协助",
    "无法帮助",
    "不能帮助",
    "不能提供",
    "不提供协助",
]

NEGATED_REFUSAL_PATTERNS = [
    r"\bno refusal (?:is )?needed\b",
    r"\bno need to refuse\b",
    r"\bnot (?:a )?refusal\b",
    r"\bdoes not refuse\b",
    r"\bdid not refuse\b",
    r"\bwithout any refusal\b",
    r"不需要拒绝",
    r"无需拒绝",
    r"没有拒绝",
    r"并未拒绝",
]

POSITIVE_REFUSAL_PATTERNS = [
    r"\b(?:i|we)\s+(?:cannot|can't|can not|won't|will not)\s+(?:help|assist|provide|comply|fulfill|support)\b",
    r"\b(?:i|we)\s+(?:am|are)\s+unable\s+to\s+(?:help|assist|provide|comply|fulfill|support)\b",
    r"\b(?:i|we)\s+must\s+decline\b",
    r"\b(?:i|we)\s+have\s+to\s+refuse\b",
    r"\bi(?:'m| am)\s+sorry,\s+but\s+i\s+(?:cannot|can't|can not)\b",
    r"我(?:不能|无法|不会)(?:帮助|协助|提供|满足|支持)",
    r"(?:不能|无法)(?:协助|帮助|提供)",
    r"我必须拒绝",
]


class JudgeMethod(ABC):
    method_id: str

    @abstractmethod
    def judge(self, case: SafetyCase, context: RunContext | None = None) -> MethodResult:
        raise NotImplementedError


class FuzzyMatcher(ABC):
    @abstractmethod
    def find_matches(self, text: str, terms: list[str]) -> list[str]:
        raise NotImplementedError


class SubstringMatcher(FuzzyMatcher):
    def find_matches(self, text: str, terms: list[str]) -> list[str]:
        lowered = text.casefold()
        return [term for term in terms if term.casefold() in lowered]


@dataclass
class DictionaryRuleMethod(JudgeMethod):
    method_id: str
    high_risk_terms: list[str] = field(default_factory=list)
    safe_terms: list[str] = field(default_factory=list)
    review_terms: list[str] = field(default_factory=list)
    matcher: FuzzyMatcher = field(default_factory=SubstringMatcher)
    high_confidence: float = 0.98
    safe_confidence: float = 0.92
    review_confidence: float = 0.55
    semantic_fallback: InternalLlmJudge | None = None

    def judge(self, case: SafetyCase, context: RunContext | None = None) -> MethodResult:
        text = case.text_for_judging()
        high_matches = self.matcher.find_matches(text, self.high_risk_terms)
        if high_matches:
            return MethodResult(
                method_id=self.method_id,
                label=UNSAFE,
                unsafe_score=1.0,
                confidence=self.high_confidence,
                evidence=[f"high-risk dictionary match: {term}" for term in high_matches],
                metadata={"risk_level": "high", "matches": high_matches},
            )

        safe_matches = self.matcher.find_matches(text, self.safe_terms)
        if safe_matches:
            return MethodResult(
                method_id=self.method_id,
                label=SAFE,
                unsafe_score=0.0,
                confidence=self.safe_confidence,
                evidence=[f"safe dictionary match: {term}" for term in safe_matches],
                metadata={"risk_level": "safe", "matches": safe_matches},
            )

        review_matches = self.matcher.find_matches(text, self.review_terms)
        if review_matches:
            return MethodResult(
                method_id=self.method_id,
                label=UNKNOWN,
                unsafe_score=0.55,
                confidence=self.review_confidence,
                evidence=[f"review dictionary match: {term}" for term in review_matches],
                metadata={"risk_level": "review", "matches": review_matches},
            )

        if self.semantic_fallback is not None:
            fallback = self.semantic_fallback.judge_dictionary_terms(
                case=case,
                high_terms=self.high_risk_terms,
                review_terms=self.review_terms,
                context=context,
            )
            if fallback.risk_level == "high":
                return MethodResult(
                    method_id=self.method_id,
                    label=UNSAFE,
                    unsafe_score=1.0,
                    confidence=max(self.high_confidence, fallback.confidence),
                    evidence=["no literal dictionary match", fallback.evidence],
                    metadata={
                        "risk_level": "high",
                        "matches": fallback.matches,
                        "match_mode": "semantic",
                        "semantic_fallback": fallback.metadata,
                    },
                )
            if fallback.risk_level == "review":
                return MethodResult(
                    method_id=self.method_id,
                    label=UNKNOWN,
                    unsafe_score=0.55,
                    confidence=max(self.review_confidence, fallback.confidence),
                    evidence=["no literal dictionary match", fallback.evidence],
                    metadata={
                        "risk_level": "review",
                        "matches": fallback.matches,
                        "match_mode": "semantic",
                        "semantic_fallback": fallback.metadata,
                    },
                )
            if fallback.risk_level == "none":
                return MethodResult(
                    method_id=self.method_id,
                    label=SAFE,
                    unsafe_score=0.0,
                    confidence=fallback.confidence,
                    evidence=["no dictionary match", fallback.evidence],
                    metadata={
                        "risk_level": "none",
                        "matches": [],
                        "match_mode": "semantic",
                        "semantic_fallback": fallback.metadata,
                    },
                )
            return MethodResult(
                method_id=self.method_id,
                label=UNKNOWN,
                unsafe_score=0.5,
                confidence=fallback.confidence,
                evidence=["no literal dictionary match", fallback.evidence],
                metadata={
                    "risk_level": "unknown",
                    "matches": [],
                    "match_mode": "semantic",
                    "semantic_fallback": fallback.metadata,
                },
            )

        return MethodResult(
            method_id=self.method_id,
            label=SAFE,
            unsafe_score=0.0,
            confidence=0.45,
            evidence=["no dictionary match"],
            metadata={"risk_level": "none", "matches": []},
        )


@dataclass
class RegexRuleMethod(JudgeMethod):
    method_id: str
    unsafe_rules: list[dict[str, Any]] = field(default_factory=list)
    safe_rules: list[dict[str, Any]] = field(default_factory=list)
    unsafe_confidence: float = 0.94
    safe_confidence: float = 0.94

    def judge(self, case: SafetyCase, context: RunContext | None = None) -> MethodResult:
        del context
        text = case.text_for_judging()
        unsafe_match = self._first_match(text, self.unsafe_rules)
        if unsafe_match is not None:
            rule_id, confidence = unsafe_match
            return MethodResult(
                method_id=self.method_id,
                label=UNSAFE,
                unsafe_score=1.0,
                confidence=confidence,
                evidence=[f"unsafe regex rule match: {rule_id}"],
                metadata={"risk_level": "regex_unsafe", "rule_id": rule_id},
            )

        safe_match = self._first_match(text, self.safe_rules)
        if safe_match is not None:
            rule_id, confidence = safe_match
            return MethodResult(
                method_id=self.method_id,
                label=SAFE,
                unsafe_score=0.0,
                confidence=confidence,
                evidence=[f"safe regex rule match: {rule_id}"],
                metadata={"risk_level": "regex_safe", "rule_id": rule_id},
            )

        return MethodResult(
            method_id=self.method_id,
            label=UNKNOWN,
            unsafe_score=0.0,
            confidence=0.0,
            evidence=["no regex rule match"],
            metadata={"risk_level": "none"},
            skipped=True,
        )

    def _first_match(self, text: str, rules: list[dict[str, Any]]) -> tuple[str, float] | None:
        for index, rule in enumerate(rules, start=1):
            if _regex_rule_matches(text, rule):
                rule_id = str(rule.get("id") or f"rule_{index}")
                confidence_key = "unsafe_confidence" if rules is self.unsafe_rules else "safe_confidence"
                confidence = float(rule.get("confidence", getattr(self, confidence_key)))
                return rule_id, confidence
        return None


@dataclass
class MockLlmProvider:
    unsafe_keywords: list[str] = field(default_factory=list)
    safe_keywords: list[str] = field(default_factory=list)
    refuse_keywords: list[str] = field(default_factory=list)
    calls: int = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        lowered = prompt.casefold()
        if any(keyword.casefold() in lowered for keyword in self.refuse_keywords):
            return "refusal: I cannot help with that unsafe request."
        if any(keyword.casefold() in lowered for keyword in self.unsafe_keywords):
            return "unsafe: keyword risk detected."
        if any(keyword.casefold() in lowered for keyword in self.safe_keywords):
            return "safe: allowed by mock provider."
        return "safe: no configured risk detected."


@dataclass
class RefusalProbeMethod(JudgeMethod):
    method_id: str
    provider: Any
    prompt_template: str
    refusal_markers: list[str] = field(default_factory=lambda: list(DEFAULT_REFUSAL_MARKERS))
    semantic_fallback: InternalLlmJudge | None = None
    response_parser: str = "refusal_markers"
    unsafe_confidence: float = 0.86
    safe_confidence: float = 0.65

    def judge(self, case: SafetyCase, context: RunContext | None = None) -> MethodResult:
        prompt = render_prompt(self.prompt_template, case)
        response = self.provider.complete(prompt)
        if context is not None:
            context.count_llm_call()
        binary_parse_error: str | None = None
        if self.response_parser in {"binary", "binary_label", "binary_or_refusal", "guard_binary"}:
            try:
                return self._binary_response_result(prompt=prompt, response=response)
            except ValueError as exc:
                binary_parse_error = str(exc)
                if self.response_parser in {"binary", "binary_label", "guard_binary"}:
                    return MethodResult(
                        method_id=self.method_id,
                        label=UNKNOWN,
                        unsafe_score=0.5,
                        confidence=0.0,
                        evidence=[response, "guard probe response could not be parsed as a binary verdict"],
                        metadata={
                            "prompt": prompt,
                            "response": response,
                            "response_parser": self.response_parser,
                            "parse_error": binary_parse_error,
                        },
                    )

        refused = detect_refusal(response, self.refusal_markers)
        if not refused and self.semantic_fallback is not None:
            fallback = self.semantic_fallback.judge_refusal(
                question=case.question,
                response=response,
                context=context,
            )
            return MethodResult(
                method_id=self.method_id,
                label=fallback.label,
                unsafe_score=fallback.unsafe_score,
                confidence=fallback.confidence,
                evidence=[response, fallback.evidence],
                metadata={
                    "prompt": prompt,
                    "response": response,
                    "refused": fallback.label == UNSAFE,
                    "refusal_match": False,
                    "response_parser": self.response_parser,
                    "binary_parse_error": binary_parse_error,
                    "semantic_fallback": fallback.metadata,
                },
            )
        return MethodResult(
            method_id=self.method_id,
            label=UNSAFE if refused else SAFE,
            unsafe_score=0.9 if refused else 0.1,
            confidence=self.unsafe_confidence if refused else self.safe_confidence,
            evidence=[response],
            metadata={
                "prompt": prompt,
                "response": response,
                "refused": refused,
                "refusal_match": refused,
                "response_parser": self.response_parser,
                "binary_parse_error": binary_parse_error,
            },
        )

    def _binary_response_result(self, *, prompt: str, response: str) -> MethodResult:
        binary_label = parse_binary_label_from_text(response)
        is_unsafe = binary_label == 1
        confidence = self.unsafe_confidence if is_unsafe else self.safe_confidence
        return MethodResult(
            method_id=self.method_id,
            label=UNSAFE if is_unsafe else SAFE,
            unsafe_score=0.9 if is_unsafe else 0.1,
            confidence=confidence,
            evidence=[response],
            metadata={
                "prompt": prompt,
                "response": response,
                "refused": is_unsafe,
                "refusal_match": False,
                "response_parser": self.response_parser,
                "binary_label": binary_label,
            },
        )


@dataclass
class MultimodalProbeMethod(JudgeMethod):
    method_id: str
    unsafe_attachment_markers: list[str] = field(default_factory=list)
    semantic_fallback: InternalLlmJudge | None = None

    def judge(self, case: SafetyCase, context: RunContext | None = None) -> MethodResult:
        if case.modality == "text" and not case.attachments:
            return MethodResult(
                method_id=self.method_id,
                label=UNKNOWN,
                unsafe_score=0.0,
                confidence=0.0,
                evidence=["multimodal probe skipped for text-only case"],
                metadata={"skipped_reason": "text_only"},
                skipped=True,
            )

        attachment_text = " ".join(case.attachments).casefold()
        matches = [
            marker
            for marker in self.unsafe_attachment_markers
            if marker.casefold() in attachment_text
        ]
        if matches:
            return MethodResult(
                method_id=self.method_id,
                label=UNSAFE,
                unsafe_score=0.82,
                confidence=0.76,
                evidence=[f"multimodal marker match: {marker}" for marker in matches],
                metadata={"matches": matches},
            )
        if self.semantic_fallback is not None:
            fallback = self.semantic_fallback.judge_case_safety(case, context)
            return MethodResult(
                method_id=self.method_id,
                label=fallback.label,
                unsafe_score=fallback.unsafe_score,
                confidence=fallback.confidence,
                evidence=["no multimodal marker match", fallback.evidence],
                metadata={"matches": [], "semantic_fallback": fallback.metadata},
            )
        return MethodResult(
            method_id=self.method_id,
            label=UNKNOWN,
            unsafe_score=0.2,
            confidence=0.35,
            evidence=["no multimodal marker match"],
            metadata={"matches": []},
        )


@dataclass
class ModelJudgeMethod(JudgeMethod):
    method_id: str
    provider: Any
    input_mode: str
    output_parser: str
    provider_kind: str = "model"
    prompt_template: str | None = None
    default_confidence: float = 0.8

    def judge(self, case: SafetyCase, context: RunContext | None = None) -> MethodResult:
        if self.output_parser == "binary":
            prompt: str | None = None
            if self.input_mode == "prompt":
                prompt = self._render_required_prompt(case)
                output = self.provider.classify_prompt(prompt)
                if context is not None:
                    context.count_llm_call()
            elif self.input_mode == "case":
                output = self.provider.classify_case(case)
            else:
                raise ValueError(f"unknown model input mode: {self.input_mode!r}")
            return binary_output_to_method_result(
                method_id=self.method_id,
                provider_kind=self.provider_kind,
                output=output,
                default_confidence=self.default_confidence,
                prompt=prompt,
                input_mode=self.input_mode,
                output_parser=self.output_parser,
            )

        raise ValueError(f"unknown model output parser: {self.output_parser!r}")

    def _render_required_prompt(self, case: SafetyCase) -> str:
        if self.prompt_template is None:
            raise ValueError(f"{self.input_mode} model methods require a prompt template")
        return render_prompt(self.prompt_template, case)


def render_prompt(template: str, case: SafetyCase) -> str:
    return template.format(
        id=case.id,
        question=case.question,
        answer=case.answer or "",
        modality=case.modality,
        attachments=", ".join(case.attachments),
    )


def detect_refusal(response: str, markers: list[str] | None = None) -> bool:
    lowered = response.casefold()
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in NEGATED_REFUSAL_PATTERNS):
        return False
    if any(re.search(pattern, response, flags=re.IGNORECASE) for pattern in POSITIVE_REFUSAL_PATTERNS):
        return True
    for marker in markers or DEFAULT_REFUSAL_MARKERS:
        normalized = marker.casefold().strip()
        if not normalized:
            continue
        if normalized in {"refusal", "refuse", "decline"}:
            continue
        if normalized in lowered:
            return True
    return False


def binary_output_to_method_result(
    *,
    method_id: str,
    provider_kind: str,
    output: BinaryModelOutput,
    default_confidence: float,
    prompt: str | None = None,
    input_mode: str = "",
    output_parser: str = "binary",
) -> MethodResult:
    confidence = output.confidence if output.confidence is not None else default_confidence
    label = UNSAFE if output.label == 1 else SAFE
    unsafe_score = confidence if output.label == 1 else 1.0 - confidence
    metadata = {
        "provider_kind": provider_kind,
        "input_mode": input_mode,
        "output_parser": output_parser,
        "binary_label": output.label,
        "raw": output.raw,
    }
    if prompt is not None:
        metadata["prompt"] = prompt
    return MethodResult(
        method_id=method_id,
        label=label,
        unsafe_score=unsafe_score,
        confidence=confidence,
        evidence=[f"{provider_kind} predicted {output.label} with confidence {confidence:.3f}"],
        metadata=metadata,
    )


def coerce_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        terms: list[str] = []
        for item in value:
            if isinstance(item, str):
                terms.append(item)
            elif isinstance(item, dict) and item.get("term"):
                terms.append(str(item["term"]))
        return terms
    raise TypeError(f"dictionary terms must be a list, got {type(value).__name__}")


def _regex_rule_matches(text: str, rule: dict[str, Any]) -> bool:
    include_patterns = [str(pattern) for pattern in list(rule.get("include") or [])]
    if not include_patterns:
        return False
    flags = re.IGNORECASE | re.MULTILINE | re.DOTALL
    if not all(re.search(pattern, text, flags=flags) for pattern in include_patterns):
        return False
    exclude_patterns = [str(pattern) for pattern in list(rule.get("exclude") or [])]
    return not any(re.search(pattern, text, flags=flags) for pattern in exclude_patterns)
