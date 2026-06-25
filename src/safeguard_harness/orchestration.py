from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from safeguard_harness.core import SAFE, UNSAFE, Decision, MethodResult, RunContext, RunTrace, SafetyCase, TraceStep
from safeguard_harness.methods import JudgeMethod


@dataclass
class Pipeline:
    runner: str
    methods: dict[str, JudgeMethod]
    aggregation: dict[str, Any] = field(default_factory=dict)
    raw_config: dict[str, Any] = field(default_factory=dict)

    def judge(self, case: SafetyCase) -> Decision:
        raise NotImplementedError

    def _judge_multi_turn_output_case(
        self,
        case: SafetyCase,
        judge_single: Callable[[SafetyCase], Decision],
    ) -> Decision | None:
        subcases = _multi_turn_output_subcases(case)
        if not subcases:
            return None

        pair_decisions = [
            (index, subcase, judge_single(subcase))
            for index, subcase in enumerate(subcases, start=1)
        ]
        return _union_multi_turn_output_decisions(
            parent_case=case,
            runner=self.runner,
            pair_decisions=pair_decisions,
        )

    def aggregate(self, case_id: str, trace: RunTrace) -> Decision:
        threshold = float(self.aggregation.get("unsafe_threshold", 0.5))
        strategy = str(self.aggregation.get("strategy", "max"))
        results = [
            step.result
            for step in trace.steps
            if not step.result.skipped and step.metadata.get("include_in_aggregation", True) is not False
        ]
        if not results:
            return Decision(
                case_id=case_id,
                label=SAFE,
                unsafe_score=0.0,
                confidence=0.0,
                reasons=["no method results"],
                trace=trace,
            )

        if strategy == "weighted_vote":
            unsafe_score, confidence = _weighted_vote_score(results)
        elif strategy in {"max", "max_unsafe_score"}:
            unsafe_score = max(result.unsafe_score for result in results)
            unsafe_results = [result for result in results if result.label == UNSAFE]
            if unsafe_results:
                confidence = max(result.confidence for result in unsafe_results)
            else:
                confidence = sum(result.confidence for result in results) / len(results)
        else:
            raise ValueError(f"unknown aggregation strategy: {strategy!r}")
        label = UNSAFE if unsafe_score >= threshold else SAFE
        reasons = [
            evidence
            for result in results
            for evidence in result.evidence
            if result.label == UNSAFE or result.unsafe_score > 0
        ]
        if not reasons:
            reasons = ["no unsafe evidence above threshold"]
        return Decision(
            case_id=case_id,
            label=label,
            unsafe_score=unsafe_score,
            confidence=confidence,
            reasons=reasons,
            trace=trace,
            metadata={"aggregation_strategy": strategy},
        )


@dataclass
class StaticPipeline(Pipeline):
    steps: list[dict[str, Any]] = field(default_factory=list)

    def judge(self, case: SafetyCase) -> Decision:
        multi_turn_decision = self._judge_multi_turn_output_case(case, self._judge_single)
        if multi_turn_decision is not None:
            return multi_turn_decision
        return self._judge_single(case)

    def _judge_single(self, case: SafetyCase) -> Decision:
        trace = RunTrace(case_id=case.id, runner="static")
        context = RunContext()
        for step in self.steps:
            if "method" in step:
                result = self._run_method(step["method"], case, context)
                step_metadata = _step_trace_metadata(step)
                trace.add_step(
                    TraceStep(
                        step_id=str(step.get("id") or step["method"]),
                        method_id=step["method"],
                        result=result,
                        metadata=step_metadata,
                    )
                )
                step_id = str(step.get("id") or step["method"])
                if result.label == UNSAFE and step.get("on_unsafe") == "stop" and _metadata_matches(
                    result.metadata, step.get("on_unsafe_metadata")
                ):
                    trace.stop_reason = f"short_circuit:{step.get('id') or step['method']}"
                    return Decision(
                        case_id=case.id,
                        label=UNSAFE,
                        unsafe_score=result.unsafe_score,
                        confidence=result.confidence,
                        reasons=list(result.evidence),
                        trace=trace,
                    )
                if result.label == SAFE and step.get("on_safe") == "stop" and _metadata_matches(
                    result.metadata, step.get("on_safe_metadata")
                ):
                    trace.stop_reason = f"short_circuit:{step_id}"
                    return Decision(
                        case_id=case.id,
                        label=SAFE,
                        unsafe_score=result.unsafe_score,
                        confidence=result.confidence,
                        reasons=list(result.evidence) or ["safe short-circuit"],
                        trace=trace,
                    )
                continue

            if "repeat" in step:
                self._run_repeat(step_id=str(step.get("id") or "repeat"), spec=step["repeat"], case=case, context=context, trace=trace)

        trace.stop_reason = trace.stop_reason or "completed"
        return self.aggregate(case.id, trace)

    def _run_repeat(
        self,
        *,
        step_id: str,
        spec: dict[str, Any],
        case: SafetyCase,
        context: RunContext,
        trace: RunTrace,
    ) -> None:
        max_rounds = int(spec.get("max_rounds", 1))
        method_ids = list(spec.get("methods") or [])
        for round_index in range(1, max_rounds + 1):
            decision = self.aggregate(case.id, trace)
            if not _should_repeat(decision, spec.get("when") or {}):
                return
            for method_id in method_ids:
                result = self._run_method(method_id, case, context)
                trace.add_step(
                    TraceStep(
                        step_id=f"{step_id}.round{round_index}.{method_id}",
                        method_id=method_id,
                        result=result,
                    )
                )

    def _run_method(self, method_id: str, case: SafetyCase, context: RunContext) -> MethodResult:
        try:
            method = self.methods[method_id]
        except KeyError as exc:
            raise KeyError(f"unknown method {method_id!r}") from exc
        return method.judge(case, context)


@dataclass
class ReactPipeline(Pipeline):
    loop: dict[str, Any] = field(default_factory=dict)

    def judge(self, case: SafetyCase) -> Decision:
        multi_turn_decision = self._judge_multi_turn_output_case(case, self._judge_single)
        if multi_turn_decision is not None:
            return multi_turn_decision
        return self._judge_single(case)

    def _judge_single(self, case: SafetyCase) -> Decision:
        trace = RunTrace(case_id=case.id, runner="react")
        context = RunContext()
        max_steps = int(self.loop.get("max_steps", 4))
        max_llm_calls = self.loop.get("max_llm_calls")
        allowed_actions = list(self.loop.get("allowed_actions") or self.methods.keys())
        stop_when = self.loop.get("stop_when") or {}

        for action_index, method_id in enumerate(allowed_actions[:max_steps], start=1):
            if max_llm_calls is not None and context.llm_calls >= int(max_llm_calls):
                trace.stop_reason = "llm_budget_exhausted"
                break
            result = self.methods[method_id].judge(case, context)
            trace.add_step(
                TraceStep(
                    step_id=f"react.step{action_index}.{method_id}",
                    method_id=method_id,
                    result=result,
                    metadata={"action": method_id},
                )
            )
            decision = self.aggregate(case.id, trace)
            if _stop_condition_met(decision, stop_when):
                trace.stop_reason = "stop_condition_met"
                return decision

        if not trace.stop_reason:
            trace.stop_reason = "budget_exhausted"
        if "fallback" in self.loop and trace.stop_reason in {"budget_exhausted", "llm_budget_exhausted"}:
            fallback = self.loop["fallback"]
            return Decision(
                case_id=case.id,
                label=str(fallback.get("label", SAFE)),
                unsafe_score=1.0 if fallback.get("label") == UNSAFE else 0.0,
                confidence=0.0,
                reasons=[str(fallback.get("reason", trace.stop_reason))],
                trace=trace,
            )
        return self.aggregate(case.id, trace)


def _multi_turn_output_subcases(case: SafetyCase) -> list[SafetyCase]:
    metadata = case.metadata or {}
    if not _truthy(_first_metadata_value(metadata, ("is_mt", "MT", "mt"))):
        return []
    if not _is_output_side(metadata.get("type")):
        return []

    messages = metadata.get("messages")
    if not isinstance(messages, list):
        return []

    pairs = _message_qa_pairs(messages)
    if len(pairs) <= 1:
        return []

    subcases: list[SafetyCase] = []
    pair_count = len(pairs)
    for index, (question, answer, pair_messages) in enumerate(pairs, start=1):
        sub_metadata = {
            key: value
            for key, value in metadata.items()
            if key not in {"messages", "is_mt", "MT", "mt"}
        }
        sub_metadata.update(
            {
                "source_format": "messages_pair",
                "messages": pair_messages,
                "is_mt": 0,
                "MT": 0,
                "mt": 0,
                "parent_is_mt": _first_metadata_value(metadata, ("is_mt", "MT", "mt")),
                "parent_case_id": case.id,
                "multi_turn_pair_index": index,
                "multi_turn_pair_count": pair_count,
            }
        )
        subcases.append(
            SafetyCase(
                id=f"{case.id}#mt{index}",
                question=question,
                answer=answer,
                label=case.label,
                modality=case.modality,
                attachments=list(case.attachments),
                metadata=sub_metadata,
            )
        )
    return subcases


def _union_multi_turn_output_decisions(
    *,
    parent_case: SafetyCase,
    runner: str,
    pair_decisions: list[tuple[int, SafetyCase, Decision]],
) -> Decision:
    trace = RunTrace(
        case_id=parent_case.id,
        runner=runner,
        metadata={
            "multi_turn_output_union": True,
            "pair_count": len(pair_decisions),
            "subcase_ids": [subcase.id for _, subcase, _ in pair_decisions],
        },
    )
    pair_results: list[dict[str, Any]] = []
    unsafe_pair_indices: list[int] = []
    unsafe_decisions: list[tuple[int, Decision]] = []

    for pair_index, subcase, decision in pair_decisions:
        pair_results.append(
            {
                "pair_index": pair_index,
                "case_id": subcase.id,
                "label": decision.label,
                "unsafe_score": decision.unsafe_score,
                "confidence": decision.confidence,
            }
        )
        if decision.label == UNSAFE:
            unsafe_pair_indices.append(pair_index)
            unsafe_decisions.append((pair_index, decision))
        if decision.trace is None:
            continue
        for step in decision.trace.steps:
            trace.add_step(
                TraceStep(
                    step_id=f"mt_pair{pair_index}.{step.step_id}",
                    method_id=step.method_id,
                    result=step.result,
                    metadata={
                        **step.metadata,
                        "multi_turn_pair_index": pair_index,
                        "subcase_id": subcase.id,
                    },
                )
            )

    trace.metadata.update(
        {
            "pair_results": pair_results,
            "unsafe_pair_indices": unsafe_pair_indices,
        }
    )
    if unsafe_decisions:
        trace.stop_reason = "multi_turn_output_union:unsafe"
        reasons = [
            f"multi-turn pair {pair_index}: {reason}"
            for pair_index, decision in unsafe_decisions
            for reason in decision.reasons
        ]
        if not reasons:
            reasons = ["at least one multi-turn output pair judged unsafe"]
        return Decision(
            case_id=parent_case.id,
            label=UNSAFE,
            unsafe_score=max(decision.unsafe_score for _, decision in unsafe_decisions),
            confidence=max(decision.confidence for _, decision in unsafe_decisions),
            reasons=reasons,
            trace=trace,
            metadata={
                "aggregation_strategy": "multi_turn_output_union",
                "pair_count": len(pair_decisions),
                "unsafe_pair_indices": unsafe_pair_indices,
                "pair_results": pair_results,
            },
        )

    trace.stop_reason = "multi_turn_output_union:safe"
    return Decision(
        case_id=parent_case.id,
        label=SAFE,
        unsafe_score=max((decision.unsafe_score for _, _, decision in pair_decisions), default=0.0),
        confidence=min((decision.confidence for _, _, decision in pair_decisions), default=0.0),
        reasons=["all multi-turn output pairs judged safe"],
        trace=trace,
        metadata={
            "aggregation_strategy": "multi_turn_output_union",
            "pair_count": len(pair_decisions),
            "unsafe_pair_indices": [],
            "pair_results": pair_results,
        },
    )


def _message_qa_pairs(messages: list[Any]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    pairs: list[tuple[str, str, list[dict[str, Any]]]] = []
    pending_user_texts: list[str] = []
    pending_user_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").casefold()
        text = _message_content_text(message.get("content")).strip()
        if role == "user":
            if text:
                pending_user_texts.append(text)
                pending_user_messages.append(message)
            continue
        if role != "assistant" or not pending_user_texts or not text:
            continue
        pairs.append(("\n\n".join(pending_user_texts), text, [*pending_user_messages, message]))
        pending_user_texts = []
        pending_user_messages = []
    return pairs


def _message_content_text(content: Any) -> str:
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


def _first_metadata_value(metadata: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _is_output_side(value: Any) -> bool:
    normalized = str(value or "").strip().casefold()
    return normalized in {"输出侧", "output", "output_side", "response", "assistant"}


def _should_repeat(decision: Decision, when: dict[str, Any]) -> bool:
    if "confidence_lt" in when and not decision.confidence < float(when["confidence_lt"]):
        return False
    if "unsafe_score_lt" in when and not decision.unsafe_score < float(when["unsafe_score_lt"]):
        return False
    return bool(when)


def _step_trace_metadata(step: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if step.get("include_in_aggregation") is False:
        metadata["include_in_aggregation"] = False
    return metadata


def _metadata_matches(metadata: dict[str, Any], expected: Any) -> bool:
    if not expected:
        return True
    if not isinstance(expected, dict):
        raise TypeError("short-circuit metadata filters must be mappings")
    for key, expected_value in expected.items():
        actual_value = metadata.get(str(key))
        if isinstance(expected_value, list):
            if actual_value not in expected_value:
                return False
        elif actual_value != expected_value:
            return False
    return True


def _stop_condition_met(decision: Decision, stop_when: dict[str, Any]) -> bool:
    if not stop_when:
        return False
    if "confidence_gte" in stop_when and decision.confidence >= float(stop_when["confidence_gte"]):
        return True
    if "unsafe_score_gte" in stop_when and decision.unsafe_score >= float(stop_when["unsafe_score_gte"]):
        return True
    return False


def _weighted_vote_score(results: list[MethodResult]) -> tuple[float, float]:
    has_explicit_unsafe = any(result.label == UNSAFE for result in results)
    unsafe_weight = 0.0
    safe_weight = 0.0
    neutral_weight = 0.0
    for result in results:
        weight = max(result.confidence, 0.0)
        if weight == 0.0:
            continue
        if result.label == UNSAFE or (has_explicit_unsafe and result.label != SAFE and result.unsafe_score > 0.5):
            unsafe_weight += weight
        elif result.label == SAFE:
            safe_weight += weight
        else:
            neutral_weight += weight

    total_weight = unsafe_weight + safe_weight
    if total_weight == 0.0:
        unsafe_score = sum(result.unsafe_score for result in results) / len(results)
        confidence = sum(result.confidence for result in results) / len(results)
        return unsafe_score, confidence

    unsafe_score = unsafe_weight / total_weight
    confidence = max(unsafe_weight, safe_weight) / (total_weight + neutral_weight)
    return unsafe_score, confidence
