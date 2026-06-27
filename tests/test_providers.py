from pathlib import Path
import sys

import pytest

from safeguard_harness.core import SafetyCase
from safeguard_harness.providers import (
    AscendVllmChatProvider,
    BinaryModelOutput,
    ClassifierHeadApiProvider,
    MergedSafeGuardProvider,
    LocalTextGenerationProvider,
    LocalPromptBinaryProvider,
    PromptBinaryApiProvider,
    Qwen3GuardProvider,
    QwenVlPromptBinaryProvider,
    QwenVlProjectionProbeProvider,
    SubprocessTextGenerationProvider,
    _extract_question_answer_prompt,
    build_binary_provider,
    build_multimodal_provider,
    build_text_generation_provider,
    load_provider_config,
    parse_binary_label_from_text,
)


def test_binary_model_output_accepts_numeric_and_string_labels():
    unsafe = BinaryModelOutput.from_payload({"label": 1, "confidence": 0.91})
    safe = BinaryModelOutput.from_payload({"prediction": "safe", "score": 0.82})
    label_only = BinaryModelOutput.from_payload({"label": 1})

    assert unsafe.label == 1
    assert unsafe.confidence == 0.91
    assert safe.label == 0
    assert safe.confidence == 0.82
    assert label_only.label == 1
    assert label_only.confidence is None


def test_prompt_binary_api_provider_sends_prompt_and_parses_prediction(monkeypatch):
    monkeypatch.setenv("PROMPT_KEY", "secret")
    calls = []

    def transport(request):
        calls.append(request)
        return {"prediction": 1, "confidence": 0.73, "reason": "risk"}

    provider = PromptBinaryApiProvider(
        base_url="https://model.example/prompt",
        api_key_env="PROMPT_KEY",
        timeout_seconds=12,
        transport=transport,
    )

    output = provider.classify_prompt("Judge this")

    assert output.label == 1
    assert output.confidence == 0.73
    assert calls[0]["url"] == "https://model.example/prompt"
    assert calls[0]["headers"]["Authorization"] == "Bearer secret"
    assert calls[0]["json"]["prompt"] == "Judge this"
    assert calls[0]["timeout_seconds"] == 12


def test_classifier_head_api_provider_sends_case_payload_and_parses_confidence(monkeypatch):
    monkeypatch.setenv("HEAD_KEY", "head-secret")
    calls = []

    def transport(request):
        calls.append(request)
        return {"label": 0, "confidence": 0.88, "logits": [2.1, -1.4]}

    provider = ClassifierHeadApiProvider(
        base_url="https://model.example/head",
        api_key_env="HEAD_KEY",
        transport=transport,
    )
    case = SafetyCase(id="c1", question="hello", answer="world")

    output = provider.classify_case(case)

    assert output.label == 0
    assert output.confidence == 0.88
    assert calls[0]["json"]["id"] == "c1"
    assert calls[0]["json"]["question"] == "hello"
    assert calls[0]["headers"]["Authorization"] == "Bearer head-secret"


def test_ascend_vllm_chat_provider_sends_openai_chat_payload_and_extracts_content(monkeypatch):
    monkeypatch.setenv("ASCEND_VLLM_API_KEY", "ascend-secret")
    calls = []

    def transport(request):
        calls.append(request)
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "safeguard-merged",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Safety: Unsafe\nReason: risk"},
                    "finish_reason": "stop",
                }
            ],
        }

    provider = AscendVllmChatProvider(
        api_base="http://127.0.0.1:8000/v1/",
        model="safeguard-merged",
        api_key_env="ASCEND_VLLM_API_KEY",
        timeout_seconds=300,
        max_tokens=32,
        temperature=0,
        transport=transport,
    )

    assert provider.complete("Judge this") == "Safety: Unsafe\nReason: risk"
    assert calls[0]["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer ascend-secret"
    assert calls[0]["timeout_seconds"] == 300
    assert calls[0]["json"] == {
        "model": "safeguard-merged",
        "messages": [{"role": "user", "content": "Judge this"}],
        "max_tokens": 32,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_ascend_vllm_prompt_binary_provider_parses_safety_label():
    def transport(request):
        del request
        return {"choices": [{"message": {"content": "Safety: Unsafe\nReason: jailbreak intent"}}]}

    provider = LocalPromptBinaryProvider(
        generator=AscendVllmChatProvider(transport=transport),
        use_llm_parse_fallback=False,
    )

    output = provider.classify_prompt("Judge this")

    assert output.label == 1
    assert output.confidence is None
    assert output.raw["response"].startswith("Safety: Unsafe")


def test_build_ascend_vllm_providers_from_config():
    binary_provider = build_binary_provider(
        {
            "type": "ascend_vllm_prompt_binary",
            "api_base": "http://127.0.0.1:8000/v1",
            "model": "safeguard-merged",
            "max_new_tokens": 64,
            "enable_thinking": False,
            "use_llm_parse_fallback": False,
        }
    )
    text_provider = build_text_generation_provider(
        {
            "type": "ascend_vllm_chat",
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "safeguard-merged",
            "chat_template_kwargs": None,
        }
    )

    assert isinstance(binary_provider, LocalPromptBinaryProvider)
    assert isinstance(binary_provider.generator, AscendVllmChatProvider)
    assert binary_provider.generator.max_tokens == 64
    assert binary_provider.generator.chat_template_kwargs == {"enable_thinking": False}
    assert isinstance(text_provider, AscendVllmChatProvider)
    assert text_provider.chat_template_kwargs is None


def test_project_owned_runtime_provider_configs_do_not_expose_script_paths():
    provider_names = [
        "local_qwen3_6_27b_generation.yaml",
        "local_qwen3_6_27b_generation_current_server.yaml",
        "local_qwen3_6_27b_lora_sft_prompt_binary.yaml",
        "local_qwen3_6_27b_lora_sft_prompt_binary_current_server.yaml",
        "local_qwen3guard_gen8b_refusal_probe.yaml",
        "local_qwen3guard_gen8b_refusal_probe_veto_safe.yaml",
        "local_qwen3_6_vl_projection_probe.yaml",
        "local_qwen3_6_vl_prompt_binary.yaml",
    ]

    for provider_name in provider_names:
        config = load_provider_config(Path("configs/providers") / provider_name)
        assert "script_path" not in config


def test_project_owned_runtime_providers_default_to_auto_device():
    binary_provider = build_binary_provider(
        {
            "type": "merged_safeguard_prompt_binary",
            "model_path": "/models/merged-safeguard",
        }
    )
    guard_provider = build_text_generation_provider(
        {
            "type": "qwen3guard_local",
            "model_path": "/models/qwen3guard",
        }
    )
    image_provider = build_multimodal_provider(
        {
            "type": "qwen_vl_projection_probe",
            "model_path": "/models/qwen-vl",
            "probe_model_path": "models/qwen36_model_lr.pth",
        }
    )
    vl_prompt_provider = build_multimodal_provider(
        {
            "type": "qwen_vl_prompt_binary",
            "model_path": "/models/qwen-vl",
            "prompt_template": "Judge: {question}",
        }
    )

    assert isinstance(binary_provider, LocalPromptBinaryProvider)
    assert isinstance(binary_provider.generator, MergedSafeGuardProvider)
    assert binary_provider.generator.device == "auto"
    assert isinstance(guard_provider, Qwen3GuardProvider)
    assert isinstance(image_provider, QwenVlProjectionProbeProvider)
    assert image_provider.device == "auto"
    assert isinstance(vl_prompt_provider, QwenVlPromptBinaryProvider)
    assert vl_prompt_provider.device == "auto"


def test_local_text_generation_provider_accepts_explicit_device():
    provider = build_text_generation_provider(
        {
            "type": "local_text_generation_lm",
            "model_path": "/models/qwen",
            "device": "npu:1",
            "device_map": None,
        }
    )

    assert isinstance(provider, LocalTextGenerationProvider)
    assert provider.device == "npu:1"
    assert provider.device_map is None


def test_project_owned_runtime_provider_configs_use_auto_without_hardcoded_memory_controls():
    binary_config = load_provider_config(Path("configs/providers/local_qwen3_6_27b_lora_sft_prompt_binary.yaml"))
    binary_provider = build_binary_provider(binary_config)
    generator = binary_provider.generator

    assert "max_memory" not in binary_config
    assert "offload_folder" not in binary_config
    assert isinstance(generator, MergedSafeGuardProvider)
    assert generator.model_path == "/data/model/Qwen36-27B-SFT"
    assert generator.device == "npu:1"
    assert generator.device_map is None
    assert generator.max_memory is None
    assert generator.offload_folder is None

    guard_config = load_provider_config(Path("configs/providers/local_qwen3guard_gen8b_refusal_probe_veto_safe.yaml"))
    guard_provider = build_text_generation_provider(guard_config)

    assert "max_memory" not in guard_config
    assert "offload_folder" not in guard_config
    assert isinstance(guard_provider, Qwen3GuardProvider)
    assert guard_provider.device_map == "auto"
    assert guard_provider.max_memory is None
    assert guard_provider.offload_folder is None


def test_build_mock_multimodal_provider_scores_case_attachments():
    provider = build_multimodal_provider(
        {
            "type": "mock_multimodal_probe",
            "default_label": 0,
            "default_confidence": 0.77,
            "unsafe_keywords": ["weapon"],
        }
    )

    output = provider.classify_case(
        SafetyCase.from_dict({"id": "img", "question": "check", "image": "/tmp/weapon.png"})
    )

    assert output.label == 1
    assert output.confidence == 0.77
    assert output.raw["provider"] == "mock_multimodal_probe"


def test_qwen_vl_prompt_binary_provider_renders_case_and_parses_response():
    class FakeRuntime:
        model = object()
        processor = object()
        device = "cuda:0"

        class module:
            calls = []

            @classmethod
            def qwen_vl_batch_infer(cls, model, processor, prompts, image_paths, max_new_tokens):
                cls.calls.append(
                    {
                        "model": model,
                        "processor": processor,
                        "prompts": prompts,
                        "image_paths": image_paths,
                        "max_new_tokens": max_new_tokens,
                    }
                )
                return ['{"label":"unsafe","reason":"visible dangerous instruction"}']

    provider = QwenVlPromptBinaryProvider(
        model_path="/models/qwen-vl",
        prompt_template="Question: {question}\nImages: {attachments}",
        max_new_tokens=48,
        runtime=FakeRuntime(),
    )

    output = provider.classify_case(
        SafetyCase.from_dict(
            {
                "id": "img",
                "question": "请描述这张图",
                "image": "/tmp/demo.png",
            }
        )
    )

    assert output.label == 1
    assert output.confidence is None
    assert output.raw["provider"] == "qwen_vl_prompt_binary"
    assert output.raw["image_path"] == "/tmp/demo.png"
    assert output.raw["response"] == '{"label":"unsafe","reason":"visible dangerous instruction"}'
    assert FakeRuntime.module.calls == [
        {
            "model": FakeRuntime.model,
            "processor": FakeRuntime.processor,
            "prompts": ["Question: 请描述这张图\nImages: /tmp/demo.png"],
            "image_paths": ["/tmp/demo.png"],
            "max_new_tokens": 48,
        }
    ]


def test_cached_multimodal_provider_loads_prediction_by_case_id(tmp_path: Path):
    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text(
        '{"case_id":"img","prediction":"unsafe","probability":0.91,"image":"/tmp/demo.png"}\n',
        encoding="utf-8",
    )
    provider = build_multimodal_provider(
        {
            "type": "cached_multimodal_probe",
            "predictions_path": cache_path.as_posix(),
        }
    )

    output = provider.classify_case(
        SafetyCase.from_dict({"id": "img", "question": "check", "image": "/tmp/other.png"})
    )

    assert output.label == 1
    assert output.confidence == 0.91
    assert output.raw["provider"] == "cached_multimodal_probe"
    assert output.raw["cache_key"] == "id:img"


def test_build_binary_provider_from_yaml_config(tmp_path: Path):
    provider_path = tmp_path / "prompt_binary_api.yaml"
    provider_path.write_text(
        """
type: mock_prompt_binary
default_label: 1
default_confidence: 0.64
""",
        encoding="utf-8",
    )

    config = load_provider_config(provider_path)
    provider = build_binary_provider(config)
    output = provider.classify_prompt("anything")

    assert output.label == 1
    assert output.confidence == 0.64


def test_local_prompt_binary_provider_returns_label_without_confidence():
    class FakeGenerator:
        def complete(self, prompt: str) -> str:
            return '{"label":"unsafe","reason":"risk"}'

    provider = LocalPromptBinaryProvider(generator=FakeGenerator())
    output = provider.classify_prompt("anything")

    assert output.label == 1
    assert output.confidence is None
    assert output.raw["confidence_source"] == "method_default"


def test_local_prompt_binary_provider_can_use_fallback_label():
    class FakeGenerator:
        calls = 0

        def complete(self, prompt: str) -> str:
            self.calls += 1
            return "I need more context."

    provider = LocalPromptBinaryProvider(generator=FakeGenerator(), fallback_label=1)
    output = provider.classify_prompt("anything")

    assert output.label == 1
    assert output.confidence is None
    assert output.raw["fallback_label"] == 1
    assert "parse_error" in output.raw
    assert "parse_fallback_error" in output.raw


def test_local_prompt_binary_provider_uses_llm_parse_fallback_before_label_fallback():
    class FakeGenerator:
        def __init__(self):
            self.prompts = []

        def complete(self, prompt: str) -> str:
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return "The request is ordinary fitness guidance with no concerning content."
            return '{"label":"safe","reason":"the response says the case is safe"}'

    generator = FakeGenerator()
    provider = LocalPromptBinaryProvider(generator=generator, fallback_label=1)
    output = provider.classify_prompt("Classify: fitness advice")

    assert output.label == 0
    assert len(generator.prompts) == 2
    assert output.raw["parse_fallback_label"] == 0
    assert "fallback_label" not in output.raw


def test_subprocess_text_generation_provider_can_send_stdin_json_and_extract_field():
    provider = SubprocessTextGenerationProvider(
        command=[
            sys.executable,
            "-c",
            "import json,sys; payload=json.load(sys.stdin); print(json.dumps({'response': payload['prompt'] + ' -> unsafe'}))",
        ],
        input_mode="stdin_json",
        stdin_json={"prompt": "{prompt}", "mode": "refusal"},
        output_json_field="response",
    )

    assert provider.complete("check this") == "check this -> unsafe"


def test_build_subprocess_prompt_binary_provider_from_config():
    provider = build_binary_provider(
        {
            "type": "subprocess_prompt_binary",
            "command": [
                sys.executable,
                "-c",
                "import sys; prompt=sys.stdin.read(); print('{\"label\":\"unsafe\",\"reason\":\"' + prompt.strip() + '\"}')",
            ],
            "input_mode": "stdin",
            "use_llm_parse_fallback": False,
        }
    )

    output = provider.classify_prompt("risky")

    assert output.label == 1
    assert output.confidence is None


def test_parse_binary_label_from_generated_text_variants():
    assert parse_binary_label_from_text("unsafe\nreason") == 1
    assert parse_binary_label_from_text("Safety: Safe\nCategories: None") == 0
    assert parse_binary_label_from_text("普通天气咨询是**安全**的。") == 0
    assert parse_binary_label_from_text("标签：安全\n原因：正常咨询") == 0
    assert parse_binary_label_from_text("Final answer: unsafe\nreason") == 1
    assert parse_binary_label_from_text('```json\n{"prediction":"unsafe"}\n```') == 1
    assert parse_binary_label_from_text('{"label":"unsafe","reason":"truncated') == 1


def test_parse_binary_label_does_not_scan_prompt_or_thinking_text():
    text = """
Thinking Process:
1. Role: Data Security Binary Classifier.
2. Task: Determine whether the user question is safe or unsafe.
"""
    with pytest.raises(ValueError):
        parse_binary_label_from_text(text)


def test_parse_binary_label_does_not_parse_template_label_options():
    with pytest.raises(ValueError):
        parse_binary_label_from_text('{"label":"safe 或 unsafe","reason":"一句话理由"}')


def test_extract_question_answer_prompt_allows_empty_question_for_output_only_probe():
    question, answer = _extract_question_answer_prompt("Question:\n\nAnswer:\nI cannot help with that request.")

    assert question == ""
    assert answer == "I cannot help with that request."


def test_local_text_generation_provider_can_disable_thinking_in_chat_template():
    class FakeTokenizer:
        def __init__(self):
            self.template_kwargs = None
            self.rendered_prompt = None

        def apply_chat_template(self, messages, **kwargs):
            self.template_kwargs = kwargs
            return "rendered"

        def __call__(self, rendered_prompt, return_tensors):
            self.rendered_prompt = rendered_prompt
            return {"input_ids": [[1, 2, 3]]}

    tokenizer = FakeTokenizer()
    provider = LocalTextGenerationProvider(model_path="dummy", enable_thinking=False)

    provider._encode_prompt(tokenizer, "prompt")

    assert tokenizer.template_kwargs["enable_thinking"] is False
    assert tokenizer.rendered_prompt == "rendered"


def test_provider_config_resolves_model_path_relative_to_config_file(tmp_path: Path):
    provider_dir = tmp_path / "configs" / "providers"
    provider_dir.mkdir(parents=True)
    provider_path = provider_dir / "local_qwen.yaml"
    provider_path.write_text(
        """
type: local_prompt_binary_lm
model_path: ../../models/Qwen3.6-27B
fallback_label: unsafe
max_memory:
  0: 4GiB
  cpu: 32GiB
offload_folder: ../../outputs/model_offload/qwen
enable_thinking: false
disable_torch_compile: true
patch_torch_distributed_tensor: true
""",
        encoding="utf-8",
    )

    provider = build_binary_provider(load_provider_config(provider_path))

    assert provider.generator.model_path == (tmp_path / "models" / "Qwen3.6-27B").as_posix()
    assert provider.generator.max_memory == {0: "4GiB", "cpu": "32GiB"}
    assert provider.generator.offload_folder == (tmp_path / "outputs" / "model_offload" / "qwen").as_posix()
    assert provider.generator.enable_thinking is False
    assert provider.generator.disable_torch_compile is True
    assert provider.generator.patch_torch_distributed_tensor is True
