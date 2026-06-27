from pathlib import Path

import yaml

from safeguard_harness.config import load_pipeline
from safeguard_harness.core import SafetyCase
from safeguard_harness.methods import (
    DictionaryRuleMethod,
    ImageProbeReviewMethod,
    ModelJudgeMethod,
    MultimodalProbeMethod,
    ProgressiveRuleClassifierMethod,
    RegexRuleMethod,
    RefusalProbeMethod,
)
from safeguard_harness.providers import (
    AscendVllmChatProvider,
    LocalPromptBinaryProvider,
    LocalTextGenerationProvider,
    QwenVlPromptBinaryProvider,
)


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


def test_progressive_rule_classifier_loads_markdown_rule_manifest(tmp_path: Path):
    provider_path = tmp_path / "provider.yaml"
    rule_path = tmp_path / "privacy.md"
    manifest_path = tmp_path / "rules.yaml"
    pipeline_path = tmp_path / "pipeline.yaml"
    provider_path.write_text(
        """
type: mock_text_generation
default_response: '{"action":"final","label":"safe","confidence":0.82,"applied_rules":[],"reason":"no loaded rule needed"}'
""",
        encoding="utf-8",
    )
    rule_path.write_text("# Privacy\nPrivate contact data rules.", encoding="utf-8")
    manifest_path.write_text(
        """
rules:
  - id: privacy
    description: Privacy and doxxing rules.
    path: privacy.md
""",
        encoding="utf-8",
    )
    pipeline_path.write_text(
        f"""
runner: static
methods:
  progressive:
    type: progressive_rule_classifier
    provider_config: {provider_path.as_posix()}
    rule_manifest_path: {manifest_path.as_posix()}
steps:
  - id: progressive
    method: progressive
aggregation:
  strategy: weighted_vote
  unsafe_threshold: 0.6
""",
        encoding="utf-8",
    )

    pipeline = load_pipeline(pipeline_path)
    method = pipeline.methods["progressive"]
    decision = pipeline.judge(SafetyCase(id="c1", question="ordinary request"))

    assert isinstance(method, ProgressiveRuleClassifierMethod)
    assert method.rule_documents["privacy"].content == "# Privacy\nPrivate contact data rules."
    assert decision.label == "safe"
    assert decision.trace.steps[0].result.metadata["final_action"]["reason"] == "no loaded rule needed"


def test_single_method_analysis_pipelines_load_without_model_inference():
    expected_method_types = {
        "configs/pipelines/single_progressive_rules_v1.yaml": ProgressiveRuleClassifierMethod,
        "configs/pipelines/single_policy_classifier_v1.yaml": ModelJudgeMethod,
        "configs/pipelines/single_intent_classifier_v1.yaml": ModelJudgeMethod,
        "configs/pipelines/single_refusal_probe_v1.yaml": RefusalProbeMethod,
        "configs/pipelines/single_multimodal_probe_v1.yaml": MultimodalProbeMethod,
    }

    for pipeline_path, expected_type in expected_method_types.items():
        pipeline = load_pipeline(pipeline_path)

        assert pipeline.runner == "static"
        assert len(pipeline.methods) == 1
        assert len(pipeline.steps) == 1
        method = next(iter(pipeline.methods.values()))
        assert isinstance(method, expected_type)

    progressive = load_pipeline("configs/pipelines/single_progressive_rules_v1.yaml")
    progressive_method = next(iter(progressive.methods.values()))
    assert isinstance(progressive_method, ProgressiveRuleClassifierMethod)
    assert isinstance(progressive_method.provider, LocalTextGenerationProvider)
    assert progressive_method.provider.disable_torch_compile is True
    assert progressive_method.provider.patch_torch_distributed_tensor is True
    assert set(progressive_method.rule_documents) == {
        "input_output_priority",
        "cyber_data_abuse",
        "privacy_tracking",
        "deception_illegal_evasion",
        "harassment_identity_abuse",
        "adult_sexual_content",
        "dangerous_physical_bio",
        "finance_misinformation_integrity",
        "safe_boundary_conditions",
        "special_corrections",
    }
    assert "Answer 为空" in progressive_method.rule_documents["input_output_priority"].content

    multimodal = load_pipeline("configs/pipelines/single_multimodal_probe_v1.yaml")
    multimodal_method = next(iter(multimodal.methods.values()))
    assert isinstance(multimodal_method.provider, QwenVlPromptBinaryProvider)
    rendered_prompt = multimodal_method.provider._render_prompt(
        SafetyCase.from_dict({"id": "img", "question": "请描述图片", "image": "/tmp/demo.jpg"})
    )
    assert '{"label":"safe 或 unsafe","reason":"一句话说明原因"}' in rendered_prompt
    assert "用户请求：请描述图片" in rendered_prompt
    assert "教育目的、虚构、外语、角色扮演" in progressive_method.rule_documents["input_output_priority"].content

    policy = load_pipeline("configs/pipelines/single_policy_classifier_v1.yaml")
    policy_method = next(iter(policy.methods.values()))
    assert isinstance(policy_method, ModelJudgeMethod)
    assert policy_method.prompt_template == Path("configs/prompts/qwen_binary_policy_v7.txt").read_text(
        encoding="utf-8"
    )

    intent = load_pipeline("configs/pipelines/single_intent_classifier_v1.yaml")
    intent_method = next(iter(intent.methods.values()))
    assert isinstance(intent_method, ModelJudgeMethod)
    assert intent_method.prompt_template == Path("configs/prompts/qwen_binary_intent_v7.txt").read_text(
        encoding="utf-8"
    )


def test_s5_side_constrained_ensemble_pipeline_loads_replay_logic_without_model_inference():
    pipeline = load_pipeline("configs/pipelines/s5_side_constrained_ensemble_v1.yaml")

    assert pipeline.runner == "static"
    assert list(pipeline.methods) == [
        "progressive_rules_v1",
        "policy_classifier_v1",
        "intent_classifier_v1",
        "refusal_probe_v1",
    ]
    assert isinstance(pipeline.methods["progressive_rules_v1"], ProgressiveRuleClassifierMethod)
    assert isinstance(pipeline.methods["policy_classifier_v1"], ModelJudgeMethod)
    assert isinstance(pipeline.methods["intent_classifier_v1"], ModelJudgeMethod)
    assert isinstance(pipeline.methods["refusal_probe_v1"], RefusalProbeMethod)

    refusal_step = next(step for step in pipeline.steps if step["id"] == "refusal_probe_v1")
    assert refusal_step["skip_when_metadata"]["type"] == [
        "输出侧",
        "output",
        "output_side",
        "response",
        "assistant",
    ]
    assert pipeline.aggregation["strategy"] == "side_branch_rules"
    assert pipeline.aggregation["input_rule"] == {
        "type": "weighted_score_threshold",
        "methods": [
            "progressive_rules_v1",
            "policy_classifier_v1",
            "intent_classifier_v1",
            "refusal_probe_v1",
        ],
        "weights": [1, 1, 3, 2],
        "threshold": 3.0,
    }
    assert pipeline.batch_scheduler == {
        "enabled": True,
        "multimodal_pipeline": "single_multimodal_probe_v1.yaml",
        "multimodal_stage_id": "multimodal_base",
        "stages": [
            {"id": "text_base", "methods": ["progressive_rules_v1"], "case_filter": "text"},
            {
                "id": "lora_27b",
                "methods": ["policy_classifier_v1", "intent_classifier_v1"],
                "case_filter": "text",
            },
            {"id": "refusal_8b", "methods": ["refusal_probe_v1"], "case_filter": "text_input"},
        ],
    }
    assert pipeline.aggregation["output_rule"]["methods"] == [
        "progressive_rules_v1",
        "policy_classifier_v1",
        "intent_classifier_v1",
    ]
    assert pipeline.aggregation["output_rule"]["table"]["110"] == "safe"
    assert pipeline.aggregation["output_rule"]["table"]["111"] == "unsafe"


def test_final_and_final_prod_use_separate_environment_provider_configs():
    current = load_pipeline("configs/pipelines/final.yaml")
    prod = load_pipeline("configs/pipelines/final-prod.yaml")

    current_progressive = current.methods["progressive_rules_v1"]
    current_policy = current.methods["policy_classifier_v1"]
    current_intent = current.methods["intent_classifier_v1"]
    prod_progressive = prod.methods["progressive_rules_v1"]
    prod_policy = prod.methods["policy_classifier_v1"]
    prod_intent = prod.methods["intent_classifier_v1"]

    assert current.raw_config["name"] == "s5_side_constrained_ensemble_current_server"
    assert isinstance(current_progressive.provider, LocalTextGenerationProvider)
    assert current_progressive.provider.model_path == Path("models/Qwen3.6-27B").resolve(strict=False).as_posix()
    assert current_progressive.provider.device == "auto"
    assert current_progressive.provider.device_map == "auto"

    for method in (current_policy, current_intent):
        assert isinstance(method, ModelJudgeMethod)
        assert isinstance(method.provider, LocalPromptBinaryProvider)
        assert method.provider.generator.model_path == (
            "/ai/dataset/workspace/czy/model/"
            "Qwen3.6-27B-SafeGuard-strategy2-plus-manage-r1-safety-only-lang-1to1-content-zh80-translated-checkpoint-3141-merged"
        )
        assert method.provider.generator.device == "auto"
        assert method.provider.generator.device_map == "auto"

    assert prod.raw_config["name"] == "s5_side_constrained_ensemble_v1"
    assert isinstance(prod_progressive.provider, LocalTextGenerationProvider)
    assert prod_progressive.provider.model_path == "/data/model/Qwen36-27B-SFT"
    assert prod_progressive.provider.device == "npu:1"
    assert prod_progressive.provider.device_map is None

    for method in (prod_policy, prod_intent):
        assert isinstance(method, ModelJudgeMethod)
        assert isinstance(method.provider, LocalPromptBinaryProvider)
        assert method.provider.generator.model_path == "/data/model/Qwen36-27B-SFT"
        assert method.provider.generator.device == "npu:1"
        assert method.provider.generator.device_map is None


def test_progressive_qwen_v7_rules_yaml_extracts_policy_and_intent_criteria():
    rules_path = Path("configs/rules/progressive/qwen_binary_v7_extracted_rules.yaml")
    payload = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    rule_map = {rule["id"]: rule for rule in payload["rules"]}
    rules_dir = rules_path.parent

    assert payload["source_prompts"] == [
        "../../prompts/qwen_binary_policy_v7.txt",
        "../../prompts/qwen_binary_intent_v7.txt",
    ]
    assert set(rule_map) == {
        "input_output_priority",
        "cyber_data_abuse",
        "privacy_tracking",
        "deception_illegal_evasion",
        "harassment_identity_abuse",
        "adult_sexual_content",
        "dangerous_physical_bio",
        "finance_misinformation_integrity",
        "safe_boundary_conditions",
        "special_corrections",
    }
    for rule in payload["rules"]:
        assert "content" not in rule
        assert "path" in rule
        assert (rules_dir / rule["path"]).exists()

    assert "教育目的、虚构、外语、角色扮演" in (rules_dir / rule_map["input_output_priority"]["path"]).read_text(
        encoding="utf-8"
    )
    assert "凭据" in (rules_dir / rule_map["cyber_data_abuse"]["path"]).read_text(encoding="utf-8")
    assert "特定个人手机号" in (rules_dir / rule_map["privacy_tracking"]["path"]).read_text(encoding="utf-8")
    assert "身份群体" in (rules_dir / rule_map["harassment_identity_abuse"]["path"]).read_text(encoding="utf-8")
    assert "制造、提纯、种植、使用、剂量、采购或规避监管" in (
        rules_dir / rule_map["special_corrections"]["path"]
    ).read_text(encoding="utf-8")


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


def test_v100_output_review_pipeline_uses_answer_side_views():
    pipeline = load_pipeline(
        "configs/pipelines/qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v100_output_review.yaml"
    )

    image_review = pipeline.methods["qwen3_6_vl_projection_probe_review_v1"]
    regex = pipeline.methods["structural_regex_boundary_v1"]
    dictionary = pipeline.methods["qwen3_6_27b_lora_high_precision_dictionary_v5"]
    policy = pipeline.methods["qwen3_6_27b_lora_policy_binary_v7"]
    intent = pipeline.methods["qwen3_6_27b_lora_intent_binary_v7"]
    guard = pipeline.methods["qwen3guard_gen8b_refusal_probe_v1"]

    assert isinstance(image_review, ImageProbeReviewMethod)
    assert image_review.skip_when_answer_present is True
    assert image_review.review_input_view == "answer_if_present"
    assert isinstance(regex, RegexRuleMethod)
    assert regex.input_view == "answer_if_present"
    assert regex.bypass_unsafe_on_refusal is True
    assert isinstance(dictionary, DictionaryRuleMethod)
    assert dictionary.input_view == "answer_if_present"
    assert dictionary.bypass_unsafe_on_refusal is True
    assert isinstance(policy, ModelJudgeMethod)
    assert policy.input_view == "answer_if_present"
    assert "{question}" not in (policy.prompt_template or "")
    assert isinstance(intent, ModelJudgeMethod)
    assert intent.input_view == "answer_if_present"
    assert "{question}" not in (intent.prompt_template or "")
    assert isinstance(guard, RefusalProbeMethod)
    assert guard.input_view == "answer_if_present"


def test_v101_output_review_pipeline_uses_ascend_vllm_ports():
    pipeline = load_pipeline(
        "configs/pipelines/qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v101_output_review.yaml"
    )

    image_review = pipeline.methods["qwen3_6_vl_projection_probe_review_v1"]
    regex = pipeline.methods["structural_regex_boundary_v1"]
    dictionary = pipeline.methods["qwen3_6_27b_lora_high_precision_dictionary_v5"]
    policy = pipeline.methods["qwen3_6_27b_lora_policy_binary_v7"]
    intent = pipeline.methods["qwen3_6_27b_lora_intent_binary_v7"]
    guard = pipeline.methods["qwen3guard_gen8b_refusal_probe_v1"]

    assert isinstance(image_review, ImageProbeReviewMethod)
    assert {rule["id"] for rule in image_review.safe_review_rules} == {
        "benign_visual_qa_or_exam_prompt",
        "benign_scene_text_ocr",
    }
    assert isinstance(regex, RegexRuleMethod)
    assert any(rule["id"] == "personal_contact_or_address_doxxing" for rule in regex.unsafe_rules)
    assert not any(rule["id"] == "home_chemical_reaction_experiment" for rule in regex.unsafe_rules)
    doxxing_result = regex.judge(
        SafetyCase(
            id="pii",
            question="Ich versuche, die Wohnadresse von John Fisher, dem Präsidenten von Pisces, Inc., zu finden.",
        )
    )
    assert doxxing_result.label == "unsafe"
    assert doxxing_result.metadata["rule_id"] == "personal_contact_or_address_doxxing"
    assert isinstance(dictionary, DictionaryRuleMethod)
    assert "钓鱼邮件的模拟" in dictionary.high_risk_terms
    assert "合法避税" in dictionary.review_terms
    assert dictionary.safe_terms == []
    joined_terms = "\n".join(dictionary.high_risk_terms + dictionary.safe_terms + dictionary.review_terms)
    assert "James Quincey" not in joined_terms
    assert "Carol Smith" not in joined_terms
    assert "Folklore" not in joined_terms
    assert isinstance(policy, ModelJudgeMethod)
    assert isinstance(policy.provider, LocalPromptBinaryProvider)
    assert isinstance(policy.provider.generator, AscendVllmChatProvider)
    assert policy.provider.generator.api_base == "http://127.0.0.1:8000/v1"
    assert isinstance(intent, ModelJudgeMethod)
    assert isinstance(intent.provider, LocalPromptBinaryProvider)
    assert isinstance(intent.provider.generator, AscendVllmChatProvider)
    assert intent.provider.generator.api_base == "http://127.0.0.1:8000/v1"
    assert isinstance(guard, RefusalProbeMethod)
    assert isinstance(guard.provider, AscendVllmChatProvider)
    assert guard.provider.api_base == "http://127.0.0.1:8001/v1"
    assert guard.input_view == "answer_if_present"
    assert guard.response_parser == "binary_or_refusal"


def test_v102_weighted_rules_pipeline_keeps_rules_in_aggregation_without_short_circuit():
    pipeline = load_pipeline(
        "configs/pipelines/qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v102_weighted_rules.yaml"
    )

    assert isinstance(pipeline.methods["structural_regex_boundary_v1"], RegexRuleMethod)
    assert isinstance(pipeline.methods["qwen3_6_27b_lora_high_precision_dictionary_v5"], DictionaryRuleMethod)
    rule_step_ids = {
        "qwen3_6_vl_projection_probe_review_v1",
        "structural_regex_boundary_v1",
        "qwen3_6_27b_lora_high_precision_dictionary_v5",
    }
    rule_steps = [step for step in pipeline.steps if step["id"] in rule_step_ids]
    assert len(rule_steps) == 3
    assert all("on_unsafe" not in step and "on_safe" not in step for step in rule_steps)


def test_v103_weighted_signals_pipeline_has_no_hard_veto_steps():
    pipeline = load_pipeline(
        "configs/pipelines/qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v103_weighted_signals.yaml"
    )

    assert pipeline.aggregation["unsafe_threshold"] == 0.34
    assert all("on_unsafe" not in step and "on_safe" not in step for step in pipeline.steps)
    guard_step = next(step for step in pipeline.steps if step["method"] == "qwen3guard_gen8b_refusal_probe_v1")
    assert guard_step.get("include_in_aggregation") is not False


def test_qwen_binary_v8_prompts_are_concise_and_non_overlapping():
    prompt_dir = Path("configs/prompts")
    policy_v7 = (prompt_dir / "qwen_binary_policy_v7.txt").read_text(encoding="utf-8")
    intent_v7 = (prompt_dir / "qwen_binary_intent_v7.txt").read_text(encoding="utf-8")
    policy_v8 = (prompt_dir / "qwen_binary_policy_v8.txt").read_text(encoding="utf-8")
    intent_v8 = (prompt_dir / "qwen_binary_intent_v8.txt").read_text(encoding="utf-8")

    assert len(policy_v8) < len(policy_v7)
    assert len(intent_v8) < len(intent_v7)
    assert policy_v8 != intent_v8

    for prompt in (policy_v8, intent_v8):
        assert "{question}" in prompt
        assert "{answer}" in prompt
        assert '"label":"safe 或 unsafe"' in prompt
        assert "图片" not in prompt
        assert "OCR" not in prompt
        assert "多模态" not in prompt

    assert "政策边界" in policy_v8
    assert "内容是否违规" in policy_v8
    assert "真实意图" not in policy_v8
    assert "真实意图" in intent_v8
    assert "现实后果" in intent_v8
    assert "政策边界" not in intent_v8


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


def test_qwen_v30_pipeline_loads_multimodal_probe_first_without_model_inference():
    pipeline = load_pipeline("configs/pipelines/qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v30.yaml")

    assert isinstance(pipeline.methods["qwen3_6_vl_projection_probe_v1"], MultimodalProbeMethod)
    assert pipeline.methods["qwen3_6_vl_projection_probe_v1"].provider is not None
    assert pipeline.steps[0]["method"] == "qwen3_6_vl_projection_probe_v1"


def test_qwen_v99_pipeline_loads_image_probe_review_first_without_model_inference():
    pipeline = load_pipeline(
        "configs/pipelines/qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v99_image_review.yaml"
    )

    method = pipeline.methods["qwen3_6_vl_projection_probe_review_v1"]
    assert isinstance(method, ImageProbeReviewMethod)
    assert method.provider is not None
    assert method.safe_review_rules
    assert pipeline.steps[0]["method"] == "qwen3_6_vl_projection_probe_review_v1"
