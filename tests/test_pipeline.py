from pathlib import Path

import pytest

from safeguard_harness.config import load_pipeline
from safeguard_harness.core import MethodResult, RunTrace, SafetyCase, TraceStep
from safeguard_harness.orchestration import Pipeline


def test_static_pipeline_short_circuits_on_high_risk_rule(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
methods:
  rules:
    type: dictionary
    high_risk_terms: ["steal token"]
    review_terms: []
  llm:
    type: prompt_binary_model
    prompt_template: "Judge: {question}"
    unsafe_keywords: ["never reached"]
steps:
  - id: rules
    method: rules
    on_unsafe: stop
  - id: llm
    method: llm
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.5
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="How to steal token?"))

    assert decision.label == "unsafe"
    assert decision.trace.stop_reason == "short_circuit:rules"
    assert [step.method_id for step in decision.trace.steps] == ["rules"]


def test_static_pipeline_can_short_circuit_on_safe_dictionary_match(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
methods:
  rules:
    type: dictionary
    high_risk_terms: []
    safe_terms: ["benign research"]
  llm:
    type: prompt_binary_model
    prompt_template: "Judge: {question}"
    unsafe_keywords: ["benign research"]
steps:
  - id: rules
    method: rules
    on_safe: stop
    on_safe_metadata:
      risk_level: safe
  - id: llm
    method: llm
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.5
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="benign research"))

    assert decision.label == "safe"
    assert decision.trace.stop_reason == "short_circuit:rules"
    assert [step.method_id for step in decision.trace.steps] == ["rules"]


def test_static_pipeline_safe_short_circuit_respects_metadata_filter(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
methods:
  rules:
    type: dictionary
    high_risk_terms: []
    safe_terms: []
  llm:
    type: prompt_binary_model
    prompt_template: "Judge: {question}"
    unsafe_keywords: ["risky"]
steps:
  - id: rules
    method: rules
    on_safe: stop
    on_safe_metadata:
      risk_level: safe
  - id: llm
    method: llm
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.5
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="risky"))

    assert decision.label == "unsafe"
    assert [step.method_id for step in decision.trace.steps] == ["rules", "llm"]


def test_static_pipeline_splits_multi_turn_output_pairs_and_unions_results(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
methods:
  judge:
    type: prompt_binary_model
    prompt_template: "Q: {question}\\nA: {answer}"
    unsafe_keywords: ["steal token answer"]
    safe_keywords: ["weather answer"]
steps:
  - id: judge
    method: judge
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.5
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(
        SafetyCase(
            id="dialogue",
            question="flattened question should not be used",
            answer="flattened answer should not be used",
            metadata={
                "type": "输出侧",
                "is_mt": 1,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "How is the weather?"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "weather answer"}]},
                    {"role": "user", "content": [{"type": "text", "text": "How do I steal a token?"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "steal token answer"}]},
                ],
            },
        )
    )

    assert decision.case_id == "dialogue"
    assert decision.label == "unsafe"
    assert decision.metadata["aggregation_strategy"] == "multi_turn_output_union"
    assert decision.metadata["pair_count"] == 2
    assert decision.metadata["unsafe_pair_indices"] == [2]
    assert decision.trace.stop_reason == "multi_turn_output_union:unsafe"
    assert [step.step_id for step in decision.trace.steps] == ["mt_pair1.judge", "mt_pair2.judge"]
    prompts = [step.result.metadata["prompt"] for step in decision.trace.steps]
    assert "steal token answer" not in prompts[0]
    assert "weather answer" not in prompts[1]


def test_static_pipeline_keeps_multi_turn_input_side_as_one_case(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
methods:
  judge:
    type: prompt_binary_model
    prompt_template: "Q: {question}\\nA: {answer}"
    unsafe_keywords: ["unsafe split-only answer"]
    safe_keywords: ["ordinary"]
steps:
  - id: judge
    method: judge
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.5
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(
        SafetyCase(
            id="input_dialogue",
            question="ordinary",
            metadata={
                "type": "输入侧",
                "is_mt": 1,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "first"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "safe answer"}]},
                    {"role": "user", "content": [{"type": "text", "text": "second"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "unsafe split-only answer"}]},
                ],
            },
        )
    )

    assert decision.label == "safe"
    assert decision.trace.stop_reason == "completed"
    assert [step.step_id for step in decision.trace.steps] == ["judge"]


def test_static_pipeline_routes_image_cases_to_multimodal_branch(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
methods:
  mm:
    type: multimodal_probe
    provider:
      type: mock_multimodal_probe
      default_label: 1
      default_confidence: 0.88
  llm:
    type: prompt_binary_model
    prompt_template: "Judge: {question}"
    safe_keywords: ["picture"]
steps:
  - id: mm
    method: mm
    on_unsafe: stop
    on_safe: stop
  - id: llm
    method: llm
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.5
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase.from_dict({"id": "img", "question": "picture", "image": "/tmp/x.png"}))

    assert decision.label == "unsafe"
    assert decision.trace.stop_reason == "short_circuit:mm"
    assert [step.method_id for step in decision.trace.steps] == ["mm"]


def test_static_pipeline_skips_multimodal_branch_for_text_cases(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
methods:
  mm:
    type: multimodal_probe
    provider:
      type: mock_multimodal_probe
      default_label: 1
      default_confidence: 0.88
  llm:
    type: prompt_binary_model
    prompt_template: "Judge: {question}"
    safe_keywords: ["ordinary"]
steps:
  - id: mm
    method: mm
    on_unsafe: stop
    on_safe: stop
  - id: llm
    method: llm
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.5
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="text", question="ordinary"))

    assert decision.label == "safe"
    assert [step.method_id for step in decision.trace.steps] == ["mm", "llm"]
    assert decision.trace.steps[0].result.skipped is True


def test_static_pipeline_can_use_image_probe_review_method(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
methods:
  image_review:
    type: image_probe_review
    provider:
      type: mock_multimodal_probe
      default_label: 1
      default_confidence: 0.99
    safe_review_rules:
      - id: landmark_vqa
        include: ["what is the name of this landmark"]
        confidence: 0.94
  llm:
    type: prompt_binary_model
    prompt_template: "Judge: {question}"
    unsafe_keywords: ["ordinary"]
steps:
  - id: image_review
    method: image_review
    on_unsafe: stop
    on_safe: stop
  - id: llm
    method: llm
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.5
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    image_decision = pipeline.judge(
        SafetyCase.from_dict(
            {"id": "img", "question": "What is the name of this landmark?", "image": "/tmp/x.png"}
        )
    )
    text_decision = pipeline.judge(SafetyCase(id="text", question="ordinary"))

    assert image_decision.label == "safe"
    assert image_decision.trace.stop_reason == "short_circuit:image_review"
    assert image_decision.trace.steps[0].result.metadata["review_route"] == "safe_prompt_override"
    assert [step.method_id for step in text_decision.trace.steps] == ["image_review", "llm"]
    assert text_decision.label == "unsafe"


def test_static_pipeline_review_loop_runs_until_confidence_threshold(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
methods:
  rules:
    type: dictionary
    high_risk_terms: []
    review_terms: ["bypass"]
  llm_v1:
    type: prompt_binary_model
    prompt_template: "Judge v1: {question}"
    unsafe_keywords: []
    safe_keywords: ["bypass"]
  llm_v2:
    type: prompt_binary_model
    prompt_template: "Judge v2: {question}"
    unsafe_keywords: ["bypass"]
steps:
  - id: rules
    method: rules
  - id: llm_v1
    method: llm_v1
  - id: review
    repeat:
      max_rounds: 2
      when:
        confidence_lt: 0.7
      methods: [llm_v2]
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="Can I bypass safeguards?"))

    assert decision.label == "unsafe"
    assert any(step.step_id.startswith("review.round1") for step in decision.trace.steps)
    assert decision.trace.stop_reason == "completed"


def test_static_pipeline_conflict_review_runs_only_on_disagreement(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
methods:
  policy:
    type: prompt_binary_model
    prompt_template: "Policy: {question}"
    default_confidence: 0.7
    unsafe_keywords: ["risky"]
  intent:
    type: prompt_binary_model
    prompt_template: "Intent: {question}"
    default_confidence: 0.7
    safe_keywords: ["risky"]
  probe:
    type: refusal_probe
    prompt_template: "Probe: {question}"
    refuse_keywords: ["risky"]
steps:
  - id: policy
    method: policy
  - id: intent
    method: intent
  - id: conflict_review
    repeat:
      max_rounds: 1
      when:
        confidence_lt: 0.7
      methods: [probe]
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="A risky request"))

    assert decision.label == "unsafe"
    assert [step.method_id for step in decision.trace.steps] == ["policy", "intent", "probe"]
    assert decision.trace.steps[-1].step_id == "conflict_review.round1.probe"


def test_static_pipeline_can_exclude_veto_probe_from_aggregation(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
methods:
  policy:
    type: prompt_binary_model
    prompt_template: "Policy: {question}"
    default_confidence: 0.7
    unsafe_keywords: ["risky"]
  probe:
    type: refusal_probe
    prompt_template: "Probe: {question}"
    safe_keywords: ["risky"]
steps:
  - id: policy
    method: policy
  - id: probe
    method: probe
    include_in_aggregation: false
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="A risky request"))

    assert decision.label == "unsafe"
    assert decision.trace.steps[-1].metadata["include_in_aggregation"] is False


def test_static_pipeline_conflict_review_skips_when_classifiers_agree(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
methods:
  policy:
    type: prompt_binary_model
    prompt_template: "Policy: {question}"
    default_confidence: 0.7
    unsafe_keywords: ["risky"]
  intent:
    type: prompt_binary_model
    prompt_template: "Intent: {question}"
    default_confidence: 0.7
    unsafe_keywords: ["risky"]
  probe:
    type: refusal_probe
    prompt_template: "Probe: {question}"
    refuse_keywords: ["risky"]
steps:
  - id: policy
    method: policy
  - id: intent
    method: intent
  - id: conflict_review
    repeat:
      max_rounds: 1
      when:
        confidence_lt: 0.7
      methods: [probe]
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="A risky request"))

    assert decision.label == "unsafe"
    assert [step.method_id for step in decision.trace.steps] == ["policy", "intent"]


def test_weighted_vote_uses_votes_instead_of_max_unsafe_score():
    trace = RunTrace(case_id="c1")
    for index, result in enumerate(
        [
            MethodResult("m1", "unsafe", unsafe_score=0.7, confidence=0.7),
            MethodResult("m2", "safe", unsafe_score=0.3, confidence=0.7),
            MethodResult("m3", "safe", unsafe_score=0.3, confidence=0.7),
        ],
        start=1,
    ):
        trace.add_step(TraceStep(step_id=f"s{index}", method_id=result.method_id, result=result))

    weighted = Pipeline(
        runner="static",
        methods={},
        aggregation={"strategy": "weighted_vote", "unsafe_threshold": 0.6},
    ).aggregate("c1", trace)
    max_score = Pipeline(
        runner="static",
        methods={},
        aggregation={"strategy": "max", "unsafe_threshold": 0.6},
    ).aggregate("c1", trace)

    assert weighted.label == "safe"
    assert weighted.unsafe_score == pytest.approx(1 / 3)
    assert max_score.label == "unsafe"
    assert max_score.unsafe_score == 0.7


def test_react_pipeline_respects_max_steps_and_allowed_actions(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: react
methods:
  rules:
    type: dictionary
    high_risk_terms: []
    review_terms: ["suspicious"]
  llm:
    type: prompt_binary_model
    prompt_template: "Judge: {question}"
    unsafe_keywords: ["suspicious"]
loop:
  max_steps: 1
  allowed_actions: [rules, llm]
  fallback:
    label: safe
    reason: "budget_exhausted"
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="A suspicious request"))

    assert len(decision.trace.steps) == 1
    assert decision.trace.stop_reason == "budget_exhausted"
    assert decision.label == "safe"
