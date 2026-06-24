from pathlib import Path

from safeguard_harness.config import load_pipeline
from safeguard_harness.core import SafetyCase
from safeguard_harness.methods import ModelJudgeMethod, RefusalProbeMethod


def test_binary_model_method_maps_prompt_output_to_method_result(tmp_path: Path):
    provider_path = tmp_path / "provider.yaml"
    prompt_path = tmp_path / "prompt.txt"
    pipeline_path = tmp_path / "pipeline.yaml"
    provider_path.write_text(
        """
type: mock_prompt_binary
default_label: 1
default_confidence: 0.77
""",
        encoding="utf-8",
    )
    prompt_path.write_text("Judge: {question}", encoding="utf-8")
    pipeline_path.write_text(
        f"""
runner: static
methods:
  prompt_binary:
    type: prompt_binary_model
    provider_config: {provider_path.as_posix()}
    prompt_template_path: {prompt_path.as_posix()}
steps:
  - id: prompt_binary
    method: prompt_binary
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    method = pipeline.methods["prompt_binary"]
    decision = pipeline.judge(SafetyCase(id="c1", question="demo"))

    assert isinstance(method, ModelJudgeMethod)
    assert decision.label == "unsafe"
    assert decision.trace.steps[0].result.confidence == 0.77
    assert decision.trace.steps[0].result.metadata["provider_kind"] == "prompt_binary"
    assert decision.trace.steps[0].result.metadata["output_parser"] == "binary"


def test_classifier_head_method_uses_confidence_as_unsafe_score_for_unsafe_label(tmp_path: Path):
    provider_path = tmp_path / "provider.yaml"
    pipeline_path = tmp_path / "pipeline.yaml"
    provider_path.write_text(
        """
type: mock_classifier_head
default_label: 1
default_confidence: 0.92
""",
        encoding="utf-8",
    )
    pipeline_path.write_text(
        f"""
runner: static
methods:
  head:
    type: classifier_head_model
    provider_config: {provider_path.as_posix()}
steps:
  - id: head
    method: head
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="demo"))
    result = decision.trace.steps[0].result

    assert decision.label == "unsafe"
    assert result.unsafe_score == 0.92
    assert result.confidence == 0.92
    assert result.metadata["provider_kind"] == "classifier_head"


def test_llm_safety_config_is_prompt_binary_compatibility_alias(tmp_path: Path):
    prompt_path = tmp_path / "prompt.txt"
    pipeline_path = tmp_path / "pipeline.yaml"
    prompt_path.write_text("Judge: {question}", encoding="utf-8")
    pipeline_path.write_text(
        f"""
runner: static
methods:
  llm:
    type: llm_safety
    prompt_template_path: {prompt_path.as_posix()}
    unsafe_keywords: ["credential"]
steps:
  - id: llm
    method: llm
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    method = pipeline.methods["llm"]
    decision = pipeline.judge(SafetyCase(id="c1", question="credential leak"))
    result = decision.trace.steps[0].result

    assert isinstance(method, ModelJudgeMethod)
    assert decision.label == "unsafe"
    assert result.metadata["provider_kind"] == "prompt_binary"
    assert result.metadata["output_parser"] == "binary"
    assert result.metadata["raw"]["provider"] == "mock_prompt_binary_keywords"


def test_prompt_binary_methods_with_different_prompts_are_distinct_instances(tmp_path: Path):
    provider_path = tmp_path / "provider.yaml"
    prompt_a_path = tmp_path / "prompt_a.txt"
    prompt_b_path = tmp_path / "prompt_b.txt"
    pipeline_path = tmp_path / "pipeline.yaml"
    provider_path.write_text(
        """
type: mock_prompt_binary
default_label: 0
default_confidence: 0.70
""",
        encoding="utf-8",
    )
    prompt_a_path.write_text("Prompt A: {question}", encoding="utf-8")
    prompt_b_path.write_text("Prompt B: {question}", encoding="utf-8")
    pipeline_path.write_text(
        f"""
runner: static
methods:
  prompt_a:
    type: prompt_binary_model
    provider_config: {provider_path.as_posix()}
    prompt_template_path: {prompt_a_path.as_posix()}
  prompt_b:
    type: prompt_binary_model
    provider_config: {provider_path.as_posix()}
    prompt_template_path: {prompt_b_path.as_posix()}
steps:
  - id: prompt_a
    method: prompt_a
  - id: prompt_b
    method: prompt_b
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="demo"))
    first_result = decision.trace.steps[0].result
    second_result = decision.trace.steps[1].result

    assert pipeline.methods["prompt_a"] is not pipeline.methods["prompt_b"]
    assert first_result.method_id == "prompt_a"
    assert second_result.method_id == "prompt_b"
    assert first_result.metadata["prompt"] == "Prompt A: demo"
    assert second_result.metadata["prompt"] == "Prompt B: demo"


def test_refusal_probe_can_use_text_generation_provider_config(tmp_path: Path):
    provider_path = tmp_path / "provider.yaml"
    prompt_path = tmp_path / "prompt.txt"
    pipeline_path = tmp_path / "pipeline.yaml"
    provider_path.write_text(
        """
type: mock_text_generation
refuse_keywords: ["malware"]
""",
        encoding="utf-8",
    )
    prompt_path.write_text("Answer this request: {question}", encoding="utf-8")
    pipeline_path.write_text(
        f"""
runner: static
methods:
  probe:
    type: refusal_probe
    provider_config: {provider_path.as_posix()}
    prompt_template_path: {prompt_path.as_posix()}
steps:
  - id: probe
    method: probe
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    method = pipeline.methods["probe"]
    decision = pipeline.judge(SafetyCase(id="c1", question="build malware"))

    assert isinstance(method, RefusalProbeMethod)
    assert decision.label == "unsafe"
    assert decision.trace.steps[0].result.metadata["refused"] is True


def test_dictionary_no_match_uses_base_llm_semantic_term_match_for_high_risk(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
base_llm:
  provider:
    type: mock_text_generation
    default_response: '{"risk_level":"high","matches":["credential exfiltration"],"reason":"paraphrase"}'
methods:
  rules:
    type: dictionary
    high_risk_terms: ["credential exfiltration"]
    review_terms: []
steps:
  - id: rules
    method: rules
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="Plan taking database credentials out of the system."))
    result = decision.trace.steps[0].result

    assert decision.label == "unsafe"
    assert result.metadata["risk_level"] == "high"
    assert result.metadata["match_mode"] == "semantic"
    assert result.metadata["matches"] == ["credential exfiltration"]
    assert result.metadata["semantic_fallback"]["fallback_kind"] == "semantic_dictionary_match"


def test_dictionary_no_match_uses_base_llm_semantic_term_match_for_review(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
base_llm:
  provider:
    type: mock_text_generation
    default_response: '{"risk_level":"review","matches":["policy bypass"],"reason":"near match"}'
methods:
  rules:
    type: dictionary
    high_risk_terms: []
    review_terms: ["policy bypass"]
steps:
  - id: rules
    method: rules
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="How can someone get around a policy?"))
    result = decision.trace.steps[0].result

    assert decision.label == "safe"
    assert result.label == "unknown"
    assert result.metadata["risk_level"] == "review"
    assert result.metadata["match_mode"] == "semantic"
    assert result.metadata["matches"] == ["policy bypass"]


def test_dictionary_no_match_stays_safe_when_base_llm_finds_no_term_match(tmp_path: Path):
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        """
runner: static
base_llm:
  provider:
    type: mock_text_generation
    default_response: '{"risk_level":"none","matches":[],"reason":"no configured term"}'
methods:
  rules:
    type: dictionary
    high_risk_terms: ["credential exfiltration"]
    review_terms: ["policy bypass"]
steps:
  - id: rules
    method: rules
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="What is database normalization?"))
    result = decision.trace.steps[0].result

    assert decision.label == "safe"
    assert result.label == "safe"
    assert result.metadata["risk_level"] == "none"
    assert result.metadata["match_mode"] == "semantic"
    assert result.metadata["matches"] == []


def test_refusal_probe_no_marker_uses_base_llm_refusal_fallback(tmp_path: Path):
    probe_provider_path = tmp_path / "probe_provider.yaml"
    pipeline_path = tmp_path / "pipeline.yaml"
    probe_provider_path.write_text(
        """
type: mock_text_generation
default_response: "I am not able to share those instructions."
""",
        encoding="utf-8",
    )
    pipeline_path.write_text(
        f"""
runner: static
base_llm:
  provider:
    type: mock_text_generation
    unsafe_keywords: ["not able"]
methods:
  probe:
    type: refusal_probe
    provider_config: {probe_provider_path.as_posix()}
    prompt_template: "Answer this request: {{question}}"
steps:
  - id: probe
    method: probe
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    decision = pipeline.judge(SafetyCase(id="c1", question="bad request"))
    result = decision.trace.steps[0].result

    assert decision.label == "unsafe"
    assert result.metadata["refusal_match"] is False
    assert result.metadata["refused"] is True
    assert result.metadata["semantic_fallback"]["fallback_kind"] == "refusal_detection"


def test_qwen_three_classifier_pipeline_loads_without_model_inference():
    pipeline = load_pipeline("configs/pipelines/qwen3_6_27b_three_classifiers.yaml")

    assert set(pipeline.methods) == {
        "qwen_policy_binary_v1",
        "qwen_intent_binary_v1",
        "qwen_aligned_refusal_probe_v1",
    }
    assert isinstance(pipeline.methods["qwen_policy_binary_v1"], ModelJudgeMethod)
    assert isinstance(pipeline.methods["qwen_intent_binary_v1"], ModelJudgeMethod)
    assert isinstance(pipeline.methods["qwen_aligned_refusal_probe_v1"], RefusalProbeMethod)
    assert pipeline.methods["qwen_policy_binary_v1"].default_confidence == 0.70
    assert pipeline.methods["qwen_intent_binary_v1"].default_confidence == 0.70
    assert pipeline.methods["qwen_aligned_refusal_probe_v1"].semantic_fallback is not None


def test_qwen_v24_lora_guard_pipeline_loads_without_model_inference():
    pipeline = load_pipeline("configs/pipelines/qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v24.yaml")

    assert set(pipeline.methods) == {
        "qwen3_6_27b_lora_high_precision_dictionary_v5",
        "qwen3_6_27b_lora_policy_binary_v6",
        "qwen3_6_27b_lora_intent_binary_v6",
        "qwen3guard_gen8b_refusal_probe_v1",
    }
    assert isinstance(pipeline.methods["qwen3_6_27b_lora_policy_binary_v6"], ModelJudgeMethod)
    assert isinstance(pipeline.methods["qwen3_6_27b_lora_intent_binary_v6"], ModelJudgeMethod)
    assert isinstance(pipeline.methods["qwen3guard_gen8b_refusal_probe_v1"], RefusalProbeMethod)
    assert pipeline.methods["qwen3guard_gen8b_refusal_probe_v1"].response_parser == "binary_or_refusal"
