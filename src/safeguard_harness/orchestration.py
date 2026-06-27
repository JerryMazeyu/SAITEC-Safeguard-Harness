from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from safeguard_harness.core import SAFE, UNSAFE, Decision, MethodResult, RunContext, RunTrace, SafetyCase, TraceStep
from safeguard_harness.methods import JudgeMethod
from safeguard_harness.progress import TerminalProgress


@dataclass
class Pipeline:
    runner: str
    methods: dict[str, JudgeMethod]
    aggregation: dict[str, Any] = field(default_factory=dict)
    raw_config: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None

    def judge(self, case: SafetyCase) -> Decision:
        raise NotImplementedError

    def judge_many(
        self,
        cases: Iterable[SafetyCase],
        *,
        intermediate_dir: str | Path | None = None,
    ) -> Iterator[Decision]:
        del intermediate_dir
        for case in cases:
            yield self.judge(case)

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

    def aggregate(self, case_id: str, trace: RunTrace, case: SafetyCase | None = None) -> Decision:
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

        if strategy == "side_branch_rules":
            return _side_branch_rules_decision(
                case_id=case_id,
                trace=trace,
                results=results,
                aggregation=self.aggregation,
                case=case,
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
    batch_scheduler: dict[str, Any] = field(default_factory=dict)

    def judge(self, case: SafetyCase) -> Decision:
        multi_turn_decision = self._judge_multi_turn_output_case(case, self._judge_single)
        if multi_turn_decision is not None:
            return multi_turn_decision
        return self._judge_single(case)

    def judge_many(
        self,
        cases: Iterable[SafetyCase],
        *,
        intermediate_dir: str | Path | None = None,
    ) -> Iterator[Decision]:
        case_list = list(cases)
        if not _batch_scheduler_enabled(self.batch_scheduler):
            yield from super().judge_many(case_list, intermediate_dir=intermediate_dir)
            return

        runner = _ResourceAwareBatchRunner(
            pipeline=self,
            scheduler=self.batch_scheduler,
            intermediate_dir=_resource_intermediate_dir(intermediate_dir),
        )
        yield from runner.judge_many(case_list)

    def _judge_single(self, case: SafetyCase) -> Decision:
        trace = RunTrace(case_id=case.id, runner="static")
        context = RunContext()
        for step in self.steps:
            if _step_skipped_for_case(step, case):
                continue

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
        return self.aggregate(case.id, trace, case=case)

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
            decision = self.aggregate(case.id, trace, case=case)
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


@dataclass(frozen=True)
class _TextExecutionUnit:
    original_index: int
    original_case: SafetyCase
    case: SafetyCase
    pair_index: int | None = None


class _ResourceAwareBatchRunner:
    def __init__(
        self,
        *,
        pipeline: StaticPipeline,
        scheduler: dict[str, Any],
        intermediate_dir: Path,
    ) -> None:
        self.pipeline = pipeline
        self.scheduler = scheduler
        self.intermediate_dir = intermediate_dir
        self.stage_dir = intermediate_dir / "stages"
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.step_by_method = self._step_by_method()

    def judge_many(self, cases: list[SafetyCase]) -> Iterator[Decision]:
        text_units, image_cases, multi_turn_units = self._prepare_units(cases)
        decisions_by_index: dict[int, Decision] = {}
        traces_by_unit_id = {
            unit.case.id: RunTrace(
                case_id=unit.case.id,
                runner=self.pipeline.runner,
                metadata={
                    "resource_scheduler": True,
                    "intermediate_dir": self.intermediate_dir.as_posix(),
                    "original_case_id": unit.original_case.id,
                },
            )
            for unit in text_units
        }

        next_stage_index = 0
        multimodal_pipeline = self._load_multimodal_pipeline()
        if multimodal_pipeline is not None:
            stage_id = str(self.scheduler.get("multimodal_stage_id", "multimodal_base"))
            decisions_by_index.update(
                self._run_multimodal_stage(
                    stage_index=next_stage_index,
                    stage_id=stage_id,
                    image_cases=image_cases,
                    multimodal_pipeline=multimodal_pipeline,
                )
            )
            next_stage_index += 1
        elif image_cases:
            raise ValueError("resource-aware batch scheduler requires multimodal_pipeline for image cases")

        for stage_offset, stage in enumerate(self.scheduler.get("stages") or [], start=next_stage_index):
            self._run_text_stage(
                stage_index=stage_offset,
                stage=stage,
                text_units=text_units,
                traces_by_unit_id=traces_by_unit_id,
            )

        unit_decisions = {
            unit.case.id: self._aggregate_unit_decision(unit.case, traces_by_unit_id[unit.case.id])
            for unit in text_units
        }
        for index, case in enumerate(cases):
            if index in decisions_by_index:
                yield decisions_by_index[index]
                continue

            parent_units = multi_turn_units.get(index)
            if parent_units:
                yield _union_multi_turn_output_decisions(
                    parent_case=case,
                    runner=self.pipeline.runner,
                    pair_decisions=[
                        (unit.pair_index or pair_index, unit.case, unit_decisions[unit.case.id])
                        for pair_index, unit in enumerate(parent_units, start=1)
                    ],
                )
                continue

            yield unit_decisions[case.id]

    def _prepare_units(
        self,
        cases: list[SafetyCase],
    ) -> tuple[list[_TextExecutionUnit], list[tuple[int, SafetyCase]], dict[int, list[_TextExecutionUnit]]]:
        text_units: list[_TextExecutionUnit] = []
        image_cases: list[tuple[int, SafetyCase]] = []
        multi_turn_units: dict[int, list[_TextExecutionUnit]] = {}
        for index, case in enumerate(cases):
            if case.has_image():
                image_cases.append((index, case))
                continue

            subcases = _multi_turn_output_subcases(case)
            if subcases:
                units = [
                    _TextExecutionUnit(
                        original_index=index,
                        original_case=case,
                        case=subcase,
                        pair_index=pair_index,
                    )
                    for pair_index, subcase in enumerate(subcases, start=1)
                ]
                multi_turn_units[index] = units
                text_units.extend(units)
                continue

            text_units.append(_TextExecutionUnit(original_index=index, original_case=case, case=case))
        return text_units, image_cases, multi_turn_units

    def _run_multimodal_stage(
        self,
        *,
        stage_index: int,
        stage_id: str,
        image_cases: list[tuple[int, SafetyCase]],
        multimodal_pipeline: Pipeline,
    ) -> dict[int, Decision]:
        stage_file = self._stage_file(stage_index, stage_id)
        decisions_by_index: dict[int, Decision] = {}
        progress = TerminalProgress(stage_id, len(image_cases))
        processed = 0
        if image_cases:
            progress.start()
        try:
            with stage_file.open("w", encoding="utf-8") as handle:
                if image_cases:
                    image_case_list = [case for _, case in image_cases]
                    image_decisions = multimodal_pipeline.judge_many(
                        image_case_list,
                        intermediate_dir=self.intermediate_dir / "multimodal",
                    )
                    for index, ((original_index, case), decision) in enumerate(
                        zip(image_cases, image_decisions),
                        start=1,
                    ):
                        self._mark_resource_route(decision, stage_id=stage_id, route="multimodal")
                        decisions_by_index[original_index] = decision
                        handle.write(
                            json.dumps(
                                {
                                    "stage": stage_id,
                                    "case_id": case.id,
                                    "route": "multimodal",
                                    "decision": decision.to_dict(),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        handle.flush()
                        processed = index
                        progress.update(processed, current=f"case={case.id}")
        except Exception as exc:
            progress.fail(processed=processed, error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            if image_cases:
                progress.finish()
        finally:
            self._release_resources(multimodal_pipeline)
        return decisions_by_index

    def _run_text_stage(
        self,
        *,
        stage_index: int,
        stage: dict[str, Any],
        text_units: list[_TextExecutionUnit],
        traces_by_unit_id: dict[str, RunTrace],
    ) -> None:
        stage_id = str(stage.get("id") or f"stage_{stage_index}")
        method_ids = [str(method_id) for method_id in (stage.get("methods") or [])]
        stage_filter = str(stage.get("case_filter", "text"))
        context = RunContext(metadata={"resource_stage": stage_id})
        stage_file = self._stage_file(stage_index, stage_id)
        work_items = self._stage_work_items(method_ids, stage_filter, text_units)
        progress = TerminalProgress(stage_id, len(work_items))
        processed = 0
        if work_items:
            progress.start()
        try:
            with stage_file.open("w", encoding="utf-8") as handle:
                for index, (method_id, step, unit) in enumerate(work_items, start=1):
                    result = self.pipeline._run_method(method_id, unit.case, context)
                    trace_step = TraceStep(
                        step_id=str(step.get("id") or method_id),
                        method_id=method_id,
                        result=result,
                        metadata={
                            **_step_trace_metadata(step),
                            "resource_stage": stage_id,
                            "original_case_id": unit.original_case.id,
                        },
                    )
                    if unit.pair_index is not None:
                        trace_step.metadata["multi_turn_pair_index"] = unit.pair_index
                    traces_by_unit_id[unit.case.id].add_step(trace_step)
                    handle.write(
                        json.dumps(
                            {
                                "stage": stage_id,
                                "case_id": unit.case.id,
                                "original_case_id": unit.original_case.id,
                                "step": trace_step.to_dict(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    handle.flush()
                    processed = index
                    progress.update(processed, current=f"{method_id}:case={unit.original_case.id}")
        except Exception as exc:
            progress.fail(processed=processed, error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            if work_items:
                progress.finish()
        finally:
            self._release_resources([self.pipeline.methods[method_id] for method_id in method_ids if method_id in self.pipeline.methods])

    def _aggregate_unit_decision(self, case: SafetyCase, trace: RunTrace) -> Decision:
        trace.stop_reason = trace.stop_reason or "completed"
        return self.pipeline.aggregate(case.id, trace, case=case)

    def _stage_work_items(
        self,
        method_ids: list[str],
        stage_filter: str,
        text_units: list[_TextExecutionUnit],
    ) -> list[tuple[str, dict[str, Any], _TextExecutionUnit]]:
        work_items: list[tuple[str, dict[str, Any], _TextExecutionUnit]] = []
        for method_id in method_ids:
            step = self.step_by_method.get(method_id, {"id": method_id, "method": method_id})
            for unit in text_units:
                if not _unit_matches_stage_filter(unit, stage_filter, self.pipeline.aggregation):
                    continue
                if _step_skipped_for_case(step, unit.case):
                    continue
                work_items.append((method_id, step, unit))
        return work_items

    def _step_by_method(self) -> dict[str, dict[str, Any]]:
        steps: dict[str, dict[str, Any]] = {}
        for step in self.pipeline.steps:
            if "method" not in step:
                continue
            steps.setdefault(str(step["method"]), step)
        return steps

    def _load_multimodal_pipeline(self) -> Pipeline | None:
        pipeline_path = self.scheduler.get("multimodal_pipeline")
        if not pipeline_path:
            return None
        from safeguard_harness.config import load_pipeline

        return load_pipeline(_resolve_scheduler_path(pipeline_path, self.pipeline.config_path))

    def _stage_file(self, index: int, stage_id: str) -> Path:
        return self.stage_dir / f"{index:02d}_{_safe_stage_id(stage_id)}.jsonl"

    def _mark_resource_route(self, decision: Decision, *, stage_id: str, route: str) -> None:
        decision.metadata.update(
            {
                "resource_scheduler": True,
                "resource_stage": stage_id,
                "resource_route": route,
                "intermediate_dir": self.intermediate_dir.as_posix(),
            }
        )
        if decision.trace is not None:
            decision.trace.metadata.update(
                {
                    "resource_scheduler": True,
                    "resource_stage": stage_id,
                    "resource_route": route,
                    "intermediate_dir": self.intermediate_dir.as_posix(),
                }
            )

    def _release_resources(self, owner: Any) -> None:
        from safeguard_harness.providers import release_cached_model_resources

        release_cached_model_resources(owner)


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
            decision = self.aggregate(case.id, trace, case=case)
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
        return self.aggregate(case.id, trace, case=case)


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


def _batch_scheduler_enabled(scheduler: dict[str, Any]) -> bool:
    if not scheduler:
        return False
    return bool(scheduler.get("enabled", True))


def _resource_intermediate_dir(intermediate_dir: str | Path | None) -> Path:
    if intermediate_dir is not None:
        path = Path(intermediate_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.mkdtemp(prefix="safeguard_resource_scheduler_"))


def _unit_matches_stage_filter(
    unit: _TextExecutionUnit,
    stage_filter: str,
    aggregation: dict[str, Any],
) -> bool:
    normalized = stage_filter.strip().casefold()
    if normalized in {"", "all", "text", "non_multimodal", "non-multimodal"}:
        return True
    side_key = str(aggregation.get("side_metadata_key", "type"))
    side_value = unit.case.metadata.get(side_key)
    if normalized in {"input", "text_input", "input_text"}:
        return not _is_output_side(side_value)
    if normalized in {"output", "text_output", "output_text"}:
        return _is_output_side(side_value)
    raise ValueError(f"unknown resource batch stage case_filter: {stage_filter!r}")


def _resolve_scheduler_path(value: Any, config_path: Path | None) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if config_path is not None:
        return config_path.parent / path
    return Path.cwd() / path


def _safe_stage_id(stage_id: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stage_id.strip())
    return cleaned or "stage"


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


def _step_skipped_for_case(step: dict[str, Any], case: SafetyCase) -> bool:
    when_metadata = step.get("when_metadata")
    if when_metadata is not None and not _case_metadata_matches(case.metadata, when_metadata):
        return True

    skip_when_metadata = step.get("skip_when_metadata")
    if skip_when_metadata is not None and _case_metadata_matches(case.metadata, skip_when_metadata):
        return True

    return False


def _step_trace_metadata(step: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if step.get("include_in_aggregation") is False:
        metadata["include_in_aggregation"] = False
    return metadata


def _case_metadata_matches(metadata: dict[str, Any], expected: Any) -> bool:
    if not expected:
        return True
    if not isinstance(expected, dict):
        raise TypeError("case metadata filters must be mappings")
    for key, expected_value in expected.items():
        if not _metadata_value_matches(metadata.get(str(key)), expected_value):
            return False
    return True


def _metadata_matches(metadata: dict[str, Any], expected: Any) -> bool:
    if not expected:
        return True
    if not isinstance(expected, dict):
        raise TypeError("short-circuit metadata filters must be mappings")
    for key, expected_value in expected.items():
        actual_value = metadata.get(str(key))
        if not _metadata_value_matches(actual_value, expected_value):
            return False
    return True


def _metadata_value_matches(actual_value: Any, expected_value: Any) -> bool:
    if isinstance(expected_value, list):
        return any(_metadata_value_matches(actual_value, item) for item in expected_value)
    if isinstance(actual_value, str) and isinstance(expected_value, str):
        return actual_value.strip().casefold() == expected_value.strip().casefold()
    return actual_value == expected_value


def _stop_condition_met(decision: Decision, stop_when: dict[str, Any]) -> bool:
    if not stop_when:
        return False
    if "confidence_gte" in stop_when and decision.confidence >= float(stop_when["confidence_gte"]):
        return True
    if "unsafe_score_gte" in stop_when and decision.unsafe_score >= float(stop_when["unsafe_score_gte"]):
        return True
    return False


def _side_branch_rules_decision(
    *,
    case_id: str,
    trace: RunTrace,
    results: list[MethodResult],
    aggregation: dict[str, Any],
    case: SafetyCase | None,
) -> Decision:
    branch = _side_branch_name(case, aggregation)
    rule_key = f"{branch}_rule"
    rule = aggregation.get(rule_key)
    if not isinstance(rule, dict):
        raise ValueError(f"side_branch_rules requires a mapping at aggregation.{rule_key}")

    label, unsafe_score, confidence, metadata, method_results = _apply_side_branch_rule(rule, results)
    reasons = _side_branch_reasons(label, rule, method_results)
    metadata.update(
        {
            "aggregation_strategy": "side_branch_rules",
            "side_branch": branch,
            "side_metadata_key": str(aggregation.get("side_metadata_key", "type")),
        }
    )
    return Decision(
        case_id=case_id,
        label=label,
        unsafe_score=unsafe_score,
        confidence=confidence,
        reasons=reasons,
        trace=trace,
        metadata=metadata,
    )


def _side_branch_name(case: SafetyCase | None, aggregation: dict[str, Any]) -> str:
    metadata = case.metadata if case is not None else {}
    side_key = str(aggregation.get("side_metadata_key", "type"))
    side_value = metadata.get(side_key)
    output_values = aggregation.get("output_values")
    input_values = aggregation.get("input_values")
    if output_values is not None and _metadata_value_matches(side_value, output_values):
        return "output"
    if input_values is not None and _metadata_value_matches(side_value, input_values):
        return "input"
    if output_values is None and _is_output_side(side_value):
        return "output"
    default_branch = str(aggregation.get("default_branch", "input")).strip().casefold()
    return "output" if default_branch == "output" else "input"


def _apply_side_branch_rule(
    rule: dict[str, Any],
    results: list[MethodResult],
) -> tuple[str, float, float, dict[str, Any], list[MethodResult]]:
    rule_type = str(rule.get("type") or "")
    method_results = _rule_method_results(rule, results)
    if rule_type == "weighted_score_threshold":
        return _apply_weighted_score_threshold(rule, method_results)
    if rule_type == "binary_truth_table":
        return _apply_binary_truth_table(rule, method_results)
    raise ValueError(f"unknown side branch rule type: {rule_type!r}")


def _rule_method_results(rule: dict[str, Any], results: list[MethodResult]) -> list[MethodResult]:
    by_method = {result.method_id: result for result in results}
    method_ids = [str(method_id) for method_id in (rule.get("methods") or [])]
    missing = [method_id for method_id in method_ids if method_id not in by_method]
    if missing:
        raise ValueError(f"side branch rule missing method result(s): {missing}")
    return [by_method[method_id] for method_id in method_ids]


def _apply_weighted_score_threshold(
    rule: dict[str, Any],
    method_results: list[MethodResult],
) -> tuple[str, float, float, dict[str, Any], list[MethodResult]]:
    weights = [float(weight) for weight in (rule.get("weights") or [1.0] * len(method_results))]
    if len(weights) != len(method_results):
        raise ValueError("weighted_score_threshold weights length must match methods length")
    threshold = float(rule.get("threshold", 0.5))
    raw_score = sum(weight * result.unsafe_score for weight, result in zip(weights, method_results))
    total_weight = sum(abs(weight) for weight in weights)
    unsafe_score = raw_score / total_weight if total_weight else 0.0
    label = UNSAFE if raw_score >= threshold else SAFE
    confidence = _rule_confidence(label, method_results)
    metadata = {
        "rule_type": "weighted_score_threshold",
        "rule_methods": [result.method_id for result in method_results],
        "rule_weights": weights,
        "raw_weighted_score": raw_score,
        "weighted_threshold": threshold,
    }
    return label, unsafe_score, confidence, metadata, method_results


def _apply_binary_truth_table(
    rule: dict[str, Any],
    method_results: list[MethodResult],
) -> tuple[str, float, float, dict[str, Any], list[MethodResult]]:
    table = rule.get("table") or {}
    if not isinstance(table, dict):
        raise ValueError("binary_truth_table requires a mapping table")
    pattern = "".join("1" if result.label == UNSAFE else "0" for result in method_results)
    if pattern not in table:
        raise ValueError(f"binary_truth_table missing pattern {pattern!r}")
    label = _coerce_rule_label(table[pattern])
    unsafe_score = 1.0 if label == UNSAFE else 0.0
    confidence = _rule_confidence(label, method_results)
    metadata = {
        "rule_type": "binary_truth_table",
        "rule_methods": [result.method_id for result in method_results],
        "truth_table_pattern": pattern,
        "truth_table_value": table[pattern],
    }
    return label, unsafe_score, confidence, metadata, method_results


def _coerce_rule_label(value: Any) -> str:
    if isinstance(value, bool):
        return UNSAFE if value else SAFE
    if isinstance(value, (int, float)):
        return UNSAFE if int(value) == 1 else SAFE
    normalized = str(value).strip().casefold()
    if normalized in {"1", "unsafe", UNSAFE}:
        return UNSAFE
    if normalized in {"0", "safe", SAFE}:
        return SAFE
    raise ValueError(f"rule label must be safe/unsafe or 0/1, got {value!r}")


def _rule_confidence(label: str, method_results: list[MethodResult]) -> float:
    matching = [result.confidence for result in method_results if result.label == label]
    if matching:
        return max(matching)
    if not method_results:
        return 0.0
    return sum(result.confidence for result in method_results) / len(method_results)


def _side_branch_reasons(label: str, rule: dict[str, Any], method_results: list[MethodResult]) -> list[str]:
    if label == UNSAFE:
        reasons = [
            evidence
            for result in method_results
            if result.label == UNSAFE or result.unsafe_score > 0.5
            for evidence in result.evidence
        ]
        if reasons:
            return reasons
    return [f"side_branch_rules {rule.get('type')} judged {label}"]


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
