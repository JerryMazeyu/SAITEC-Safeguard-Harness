from pathlib import Path
import sys

import pytest

from safeguard_harness.core import SafetyCase
from safeguard_harness.providers import (
    BinaryModelOutput,
    ClassifierHeadApiProvider,
    LocalTextGenerationProvider,
    LocalPromptBinaryProvider,
    PromptBinaryApiProvider,
    SubprocessTextGenerationProvider,
    build_binary_provider,
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
