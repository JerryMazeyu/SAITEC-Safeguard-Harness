from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml

from safeguard_harness.core import SafetyCase

JsonTransport = Callable[[dict[str, Any]], dict[str, Any]]

_BINARY_PARSE_FALLBACK_PROMPT = """You are an internal parser for a safety-classifier response.

The classifier was asked to label the case as safe or unsafe, but its response was not valid JSON.
Infer whether the classifier response says the case is safe or unsafe.

Return only JSON:
{{"label":"safe or unsafe","reason":"one short reason"}}

Original classifier prompt:
{prompt}

Classifier response:
{response}
"""


class PromptBinaryProvider(Protocol):
    def classify_prompt(self, prompt: str) -> "BinaryModelOutput":
        ...


class CaseBinaryProvider(Protocol):
    def classify_case(self, case: SafetyCase) -> "BinaryModelOutput":
        ...


class MultimodalCaseProvider(Protocol):
    def classify_case(self, case: SafetyCase) -> "BinaryModelOutput":
        ...


class TextGenerationProvider(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class BinaryModelOutput:
    label: int
    confidence: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in {0, 1}:
            raise ValueError(f"binary label must be 0 or 1, got {self.label!r}")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _clamp01(self.confidence))

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BinaryModelOutput":
        label_value = _first_present(payload, ["label", "prediction", "pred", "class", "output"])
        confidence_value = _first_optional(payload, ["confidence", "score", "probability", "prob"])
        return cls(
            label=parse_binary_label(label_value),
            confidence=None if confidence_value is None else float(confidence_value),
            raw=dict(payload),
        )


@dataclass
class PromptBinaryApiProvider:
    base_url: str
    api_key_env: str | None = None
    timeout_seconds: int = 30
    transport: JsonTransport = field(default_factory=lambda: _http_json_transport)
    prompt_field: str = "prompt"

    def classify_prompt(self, prompt: str) -> BinaryModelOutput:
        response = self.transport(
            {
                "url": self.base_url,
                "headers": self._headers(),
                "json": {self.prompt_field: prompt},
                "timeout_seconds": self.timeout_seconds,
            }
        )
        return BinaryModelOutput.from_payload(response)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env and os.environ.get(self.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        return headers


@dataclass
class ClassifierHeadApiProvider:
    base_url: str
    api_key_env: str | None = None
    timeout_seconds: int = 30
    transport: JsonTransport = field(default_factory=lambda: _http_json_transport)

    def classify_case(self, case: SafetyCase) -> BinaryModelOutput:
        response = self.transport(
            {
                "url": self.base_url,
                "headers": self._headers(),
                "json": case.to_dict(),
                "timeout_seconds": self.timeout_seconds,
            }
        )
        return BinaryModelOutput.from_payload(response)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env and os.environ.get(self.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        return headers


@dataclass
class AscendVllmChatProvider:
    api_base: str = "http://127.0.0.1:8000/v1"
    model: str = "safeguard-merged"
    api_key_env: str | None = None
    timeout_seconds: int = 300
    max_tokens: int = 32
    temperature: float | None = 0.0
    top_p: float | None = None
    system_prompt: str | None = None
    chat_template_kwargs: dict[str, Any] | None = field(default_factory=lambda: {"enable_thinking": False})
    extra_payload: dict[str, Any] = field(default_factory=dict)
    endpoint: str = "/chat/completions"
    transport: JsonTransport = field(default_factory=lambda: _http_json_transport)

    def complete(self, prompt: str) -> str:
        response = self.chat_completion(prompt)
        return _extract_chat_completion_content(response)

    def chat_completion(self, prompt: str) -> dict[str, Any]:
        response = self.transport(
            {
                "url": self._url(),
                "headers": self._headers(),
                "json": self._payload(prompt),
                "timeout_seconds": self.timeout_seconds,
            }
        )
        if not isinstance(response, dict):
            raise ValueError("Ascend vLLM response must be a JSON object")
        return response

    def _payload(self, prompt: str) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        payload.update(self.extra_payload)
        return payload

    def _url(self) -> str:
        endpoint = self.endpoint if self.endpoint.startswith("/") else f"/{self.endpoint}"
        return f"{self.api_base.rstrip('/')}{endpoint}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env and os.environ.get(self.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        return headers


@dataclass
class MockPromptBinaryProvider:
    default_label: int = 0
    default_confidence: float | None = 0.8
    unsafe_keywords: list[str] = field(default_factory=list)
    safe_keywords: list[str] = field(default_factory=list)
    refuse_keywords: list[str] = field(default_factory=list)

    def classify_prompt(self, prompt: str) -> BinaryModelOutput:
        lowered = prompt.casefold()
        unsafe_match = _first_keyword_match(lowered, self.unsafe_keywords + self.refuse_keywords)
        if unsafe_match is not None:
            return BinaryModelOutput(
                label=1,
                confidence=self.default_confidence,
                raw={
                    "provider": "mock_prompt_binary_keywords",
                    "prompt": prompt,
                    "matched_keyword": unsafe_match,
                },
            )

        safe_match = _first_keyword_match(lowered, self.safe_keywords)
        if safe_match is not None:
            return BinaryModelOutput(
                label=0,
                confidence=self.default_confidence,
                raw={
                    "provider": "mock_prompt_binary_keywords",
                    "prompt": prompt,
                    "matched_keyword": safe_match,
                },
            )

        return BinaryModelOutput(
            label=parse_binary_label(self.default_label),
            confidence=self.default_confidence,
            raw={"provider": "mock_prompt_binary", "prompt": prompt},
        )


@dataclass
class MockClassifierHeadProvider:
    default_label: int = 0
    default_confidence: float | None = 0.8

    def classify_case(self, case: SafetyCase) -> BinaryModelOutput:
        return BinaryModelOutput(
            label=parse_binary_label(self.default_label),
            confidence=self.default_confidence,
            raw={"provider": "mock_classifier_head", "case_id": case.id},
        )


@dataclass
class MockMultimodalProbeProvider:
    default_label: int = 0
    default_confidence: float | None = 0.8
    unsafe_keywords: list[str] = field(default_factory=list)

    def classify_case(self, case: SafetyCase) -> BinaryModelOutput:
        attachment_text = " ".join(case.attachments).casefold()
        matched_keyword = _first_keyword_match(attachment_text, self.unsafe_keywords)
        label = 1 if matched_keyword is not None else parse_binary_label(self.default_label)
        return BinaryModelOutput(
            label=label,
            confidence=self.default_confidence,
            raw={
                "provider": "mock_multimodal_probe",
                "case_id": case.id,
                "attachments": list(case.attachments),
                "matched_keyword": matched_keyword,
            },
        )


@dataclass
class CachedMultimodalProbeProvider:
    predictions_path: str
    default_confidence: float | None = None
    _outputs_by_key: dict[str, BinaryModelOutput] | None = field(default=None, init=False, repr=False)

    def classify_case(self, case: SafetyCase) -> BinaryModelOutput:
        outputs_by_key = self._load_outputs()
        for key in self._case_keys(case):
            output = outputs_by_key.get(key)
            if output is not None:
                return BinaryModelOutput(
                    label=output.label,
                    confidence=output.confidence,
                    raw={**output.raw, "provider": "cached_multimodal_probe", "cache_key": key},
                )
        keys = ", ".join(self._case_keys(case))
        raise KeyError(f"no cached multimodal prediction for case {case.id!r}; tried {keys}")

    def _load_outputs(self) -> dict[str, BinaryModelOutput]:
        if self._outputs_by_key is not None:
            return self._outputs_by_key
        path = Path(self.predictions_path)
        outputs_by_key: dict[str, BinaryModelOutput] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"cached multimodal record {line_number} must be an object")
                output = self._output_from_record(record)
                for key in self._record_keys(record):
                    outputs_by_key.setdefault(key, output)
        self._outputs_by_key = outputs_by_key
        return outputs_by_key

    def _output_from_record(self, record: dict[str, Any]) -> BinaryModelOutput:
        label_value = _first_optional(record, ["prediction", "pred", "label", "output"])
        if label_value is None:
            raise ValueError(f"cached multimodal record missing prediction/label: {record!r}")
        confidence_value = _first_optional(record, ["confidence", "score"])
        probability_value = _first_optional(record, ["probability", "prob"])
        confidence: float | None
        if confidence_value is not None:
            confidence = float(confidence_value)
        elif probability_value is not None:
            probability = float(probability_value)
            confidence = max(probability, 1.0 - probability)
        else:
            confidence = self.default_confidence
        return BinaryModelOutput(label=parse_binary_label(label_value), confidence=confidence, raw={"cached_record": record})

    def _record_keys(self, record: dict[str, Any]) -> list[str]:
        keys: list[str] = []
        for field_name in ("case_id", "id"):
            value = record.get(field_name)
            if value is not None and value != "":
                keys.append(f"id:{value}")
        for field_name in ("image_path", "image", "attachment"):
            value = record.get(field_name)
            if value:
                keys.append(f"image:{_normalize_cache_path(str(value))}")
        question_sha1 = record.get("question_sha1")
        if question_sha1:
            keys.append(f"question_sha1:{question_sha1}")
        question = record.get("question")
        if question:
            keys.append(f"question_sha1:{_sha1_text(str(question))}")
        return _dedupe_preserve_order(keys)

    def _case_keys(self, case: SafetyCase) -> list[str]:
        keys = [f"id:{case.id}"]
        image_path = _first_image_attachment(case)
        if image_path:
            keys.append(f"image:{_normalize_cache_path(image_path)}")
        keys.append(f"question_sha1:{_sha1_text(case.question)}")
        return _dedupe_preserve_order(keys)


@dataclass
class MockTextGenerationProvider:
    unsafe_keywords: list[str] = field(default_factory=list)
    safe_keywords: list[str] = field(default_factory=list)
    refuse_keywords: list[str] = field(default_factory=list)
    default_response: str = "safe: no configured risk detected."

    def complete(self, prompt: str) -> str:
        lowered = prompt.casefold()
        if any(keyword.casefold() in lowered for keyword in self.refuse_keywords):
            return "refusal: I cannot help with that unsafe request."
        if any(keyword.casefold() in lowered for keyword in self.unsafe_keywords):
            return "unsafe: keyword risk detected."
        if any(keyword.casefold() in lowered for keyword in self.safe_keywords):
            return "safe: allowed by mock provider."
        return self.default_response


@dataclass
class SubprocessTextGenerationProvider:
    command: list[str]
    timeout_seconds: int = 300
    input_mode: str = "stdin"
    prompt_arg: str | None = None
    stdin_json: dict[str, Any] = field(default_factory=dict)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    output_json_field: str | None = None

    def complete(self, prompt: str) -> str:
        command = [_render_template_value(item, prompt) for item in self.command]
        input_text = self._input_text(prompt)
        if self.input_mode in {"arg", "prompt_arg"}:
            command.extend([self.prompt_arg or "--prompt", prompt])

        completed = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            cwd=self.cwd,
            env={**os.environ, **{key: _render_template_value(value, prompt) for key, value in self.env.items()}},
            check=False,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if completed.returncode != 0:
            raise RuntimeError(
                "subprocess text generation failed "
                f"(exit {completed.returncode}): {_truncate_for_error(stderr or stdout)}"
            )
        if not stdout:
            raise RuntimeError(f"subprocess text generation produced empty stdout: {_truncate_for_error(stderr)}")
        if self.output_json_field:
            payload = _extract_json_payload(stdout)
            if payload is None:
                raise ValueError(f"subprocess output did not contain a JSON object: {stdout!r}")
            value = _lookup_json_field(payload, self.output_json_field)
            return str(value).strip()
        return stdout

    def _input_text(self, prompt: str) -> str | None:
        mode = self.input_mode
        if mode == "stdin":
            return prompt
        if mode == "stdin_line":
            return prompt + "\n"
        if mode == "stdin_json":
            payload = self.stdin_json or {"prompt": "{prompt}"}
            rendered = _render_template_value(payload, prompt)
            return json.dumps(rendered, ensure_ascii=False) + "\n"
        if mode in {"arg", "prompt_arg", "none"}:
            return None
        raise ValueError(f"unknown subprocess input_mode: {mode!r}")


@dataclass
class Qwen3GuardSubprocessProvider:
    python_executable: str
    script_path: str
    model_path: str
    timeout_seconds: int = 600
    max_new_tokens: int = 128
    controversial_label: str = "unsafe"
    use_refusal_as_unsafe: bool = False
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    def complete(self, prompt: str) -> str:
        user_prompt, assistant_response = _extract_question_answer_prompt(prompt)
        has_assistant = bool(assistant_response.strip())
        command = [
            self.python_executable,
            self.script_path,
            "--model-path",
            self.model_path,
            "--task",
            "response" if has_assistant else "prompt",
            "--user",
            user_prompt,
            "--max-new-tokens",
            str(self.max_new_tokens),
        ]
        if has_assistant:
            command.extend(["--assistant", assistant_response])

        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            cwd=self.cwd,
            env={**os.environ, **self.env},
            check=False,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if completed.returncode != 0:
            raise RuntimeError(
                "qwen3guard subprocess failed "
                f"(exit {completed.returncode}): {_truncate_for_error(stderr or stdout)}"
            )
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = _extract_json_payload(stdout)
        if payload is None:
            raise ValueError(f"qwen3guard output did not contain a JSON object: {stdout!r}")
        if not isinstance(payload, dict):
            raise ValueError(f"qwen3guard output JSON must be an object: {stdout!r}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"qwen3guard output missing result object: {stdout!r}")

        safety_label = str(result.get("safety_label") or "").strip()
        refusal = str(result.get("refusal") or "").strip()
        binary_label = self._binary_label(safety_label=safety_label, refusal=refusal)
        return json.dumps(
            {
                "label": binary_label,
                "reason": f"qwen3guard safety={safety_label or 'unknown'}, refusal={refusal or 'n/a'}",
                "qwen3guard": payload,
            },
            ensure_ascii=False,
        )

    def _binary_label(self, *, safety_label: str, refusal: str) -> str:
        normalized = safety_label.casefold()
        if normalized == "unsafe":
            return "unsafe"
        if normalized == "controversial":
            return self.controversial_label
        if self.use_refusal_as_unsafe and refusal.casefold() == "yes":
            return "unsafe"
        return "safe"


@dataclass
class MergedSafeGuardProvider:
    model_path: str
    device: str = "auto"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 32
    cache_model: bool = True

    def complete(self, prompt: str) -> str:
        from safeguard_harness.runtimes import merged_safeguard

        runtime = self._get_runtime()
        result = merged_safeguard.infer_safety(runtime, prompt=prompt, max_new_tokens=self.max_new_tokens)
        return str(result.get("prediction_text") or result.get("prediction_label") or "").strip()

    def _get_runtime(self) -> Any:
        from safeguard_harness.runtimes import merged_safeguard

        cache_key = (self.model_path, self.device, self.torch_dtype)
        if self.cache_model and cache_key in _MERGED_SAFEGUARD_RUNTIME_CACHE:
            return _MERGED_SAFEGUARD_RUNTIME_CACHE[cache_key]
        runtime = merged_safeguard.load_merged_safeguard(
            model_path=self.model_path,
            device=self.device,
            torch_dtype=self.torch_dtype,
        )
        if self.cache_model:
            _MERGED_SAFEGUARD_RUNTIME_CACHE[cache_key] = runtime
        return runtime


@dataclass
class Qwen3GuardProvider:
    model_path: str
    max_new_tokens: int = 128
    controversial_label: str = "unsafe"
    use_refusal_as_unsafe: bool = False
    cache_model: bool = True

    def complete(self, prompt: str) -> str:
        from safeguard_harness.runtimes import qwen3guard

        user_prompt, assistant_response = _extract_question_answer_prompt(prompt)
        tokenizer, model = self._get_runtime()
        if assistant_response.strip():
            result = qwen3guard.infer_response_safety_local(
                user_prompt=user_prompt,
                assistant_response=assistant_response,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=self.max_new_tokens,
            )
        else:
            result = qwen3guard.infer_prompt_safety_local(
                user_prompt=user_prompt,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=self.max_new_tokens,
            )

        payload = _dataclass_to_dict(result)
        safety_label = str(payload.get("safety_label") or "").strip()
        refusal = str(payload.get("refusal") or "").strip()
        binary_label = self._binary_label(safety_label=safety_label, refusal=refusal)
        return json.dumps(
            {
                "label": binary_label,
                "reason": f"qwen3guard safety={safety_label or 'unknown'}, refusal={refusal or 'n/a'}",
                "qwen3guard": {"result": payload},
            },
            ensure_ascii=False,
        )

    def _get_runtime(self) -> tuple[Any, Any]:
        from safeguard_harness.runtimes import qwen3guard

        cache_key = (self.model_path,)
        if self.cache_model and cache_key in _QWEN3GUARD_RUNTIME_CACHE:
            return _QWEN3GUARD_RUNTIME_CACHE[cache_key]
        tokenizer, model = qwen3guard.load_qwen3guard_gen8b_local(self.model_path)
        if self.cache_model:
            _QWEN3GUARD_RUNTIME_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model

    def _binary_label(self, *, safety_label: str, refusal: str) -> str:
        normalized = safety_label.casefold()
        if normalized == "unsafe":
            return "unsafe"
        if normalized == "controversial":
            return self.controversial_label
        if self.use_refusal_as_unsafe and refusal.casefold() == "yes":
            return "unsafe"
        return "safe"


@dataclass
class QwenVlProjectionProbeProvider:
    model_path: str
    probe_model_path: str
    device: str = "auto"
    save_root: str = "outputs/multimodal_probe"
    save_layers: Any = field(default_factory=lambda: [42])
    source_layer: int = 42
    target_layers: Any = 27
    project_save_layers: Any = "last"
    token_slice: Any = "all"
    probe_input_dim: int = 248320
    threshold: float = 0.5
    cache_model: bool = True
    disable_torch_compile: bool = False
    patch_torch_distributed_tensor: bool = False

    def classify_case(self, case: SafetyCase) -> BinaryModelOutput:
        image_path = _first_image_attachment(case)
        if image_path is None:
            raise ValueError(f"multimodal probe case {case.id!r} has no image attachment")

        runtime = self._get_runtime()
        data_path, data_name = self._write_case_data(case, image_path)

        self._clear_generated_case_dirs(data_name)
        runtime.module.save_merge_layer(
            data_path=data_path,
            save_root=self.save_root,
            model=runtime.model,
            processor=runtime.processor,
            save_layers=self.save_layers,
        )
        runtime.module.run_project_mode(
            data_path=data_path,
            save_root=self.save_root,
            model=runtime.model,
            processor=runtime.processor,
            source_layer=self.source_layer,
            target_layers=self.target_layers,
            project_save_layers=self.project_save_layers,
            token_slice=self.token_slice,
        )
        probability, confidence, feature_paths = self._score_projected_features(runtime, data_name)
        label = 1 if probability > self.threshold else 0
        return BinaryModelOutput(
            label=label,
            confidence=confidence,
            raw={
                "provider": "qwen_vl_projection_probe",
                "case_id": case.id,
                "image_path": image_path,
                "data_path": data_path,
                "probability": probability,
                "threshold": self.threshold,
                "feature_paths": feature_paths,
            },
        )

    def _get_runtime(self) -> "_OneCaseMultimodalRuntime":
        cache_key = (
            self.model_path,
            self.probe_model_path,
            self.device,
            self.probe_input_dim,
        )
        if self.cache_model and cache_key in _ONE_CASE_MULTIMODAL_RUNTIME_CACHE:
            return _ONE_CASE_MULTIMODAL_RUNTIME_CACHE[cache_key]

        from safeguard_harness.runtimes import qwen_vl_projection_probe

        torch = _import_torch()
        original_torch_compile = _apply_transformers_load_compat(
            torch,
            disable_torch_compile=self.disable_torch_compile,
            patch_torch_distributed_tensor=self.patch_torch_distributed_tensor,
        )
        try:
            model, processor, resolved_device = qwen_vl_projection_probe.load_model(
                model_path=self.model_path,
                device=self.device,
            )
            import torch.nn as nn

            probe_model = nn.Linear(self.probe_input_dim, 1)
            try:
                state_dict = torch.load(self.probe_model_path, map_location=resolved_device, weights_only=True)
            except TypeError:
                state_dict = torch.load(self.probe_model_path, map_location=resolved_device)
            probe_model.load_state_dict(state_dict)
            probe_model.to(resolved_device)
            probe_model.eval()
        finally:
            if original_torch_compile is not None:
                torch.compile = original_torch_compile

        runtime = _OneCaseMultimodalRuntime(
            module=qwen_vl_projection_probe,
            model=model,
            processor=processor,
            probe_model=probe_model,
            device=resolved_device,
        )
        if self.cache_model:
            _ONE_CASE_MULTIMODAL_RUNTIME_CACHE[cache_key] = runtime
        return runtime

    def _write_case_data(self, case: SafetyCase, image_path: str) -> tuple[str, str]:
        input_dir = Path(self.save_root) / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = hashlib.sha1(
            json.dumps(
                {"id": case.id, "question": case.question, "image": image_path},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        data_name = f"case_{_safe_filename(case.id)}_{fingerprint}"
        data_path = input_dir / f"{data_name}.json"
        data_path.write_text(
            json.dumps([{"id": case.id, "question": case.question, "image": image_path}], ensure_ascii=False),
            encoding="utf-8",
        )
        return data_path.as_posix(), data_name

    def _clear_generated_case_dirs(self, data_name: str) -> None:
        save_root = Path(self.save_root)
        for path in (save_root / "source" / data_name, save_root / "projected" / data_name):
            if path.exists():
                shutil.rmtree(path)

    def _score_projected_features(
        self,
        runtime: "_OneCaseMultimodalRuntime",
        data_name: str,
    ) -> tuple[float, float, list[str]]:
        project_dir = runtime.module.projected_data_dir(self.save_root, data_name)
        feature_paths = sorted(path for path in project_dir.rglob("*.pt") if path.is_file())
        if not feature_paths:
            raise RuntimeError(f"Qwen VL projection probe produced no projected features under {project_dir}")

        torch = _import_torch()
        probabilities: list[float] = []
        with torch.no_grad():
            for feature_path in feature_paths:
                hidden_states = runtime.module.load_hidden_states(feature_path)
                feature = runtime.module.get_logit(hidden_states, runtime.model).to(torch.float32)
                std = feature.std(unbiased=False)
                if torch.isnan(std) or torch.isinf(std) or std.item() < 1e-8:
                    std = torch.tensor(1.0, dtype=feature.dtype)
                feature = (feature - feature.mean()) / std
                output = runtime.probe_model(feature.clone().detach().unsqueeze(0).to(runtime.device).float())
                probabilities.append(float(torch.sigmoid(output).flatten()[0].item()))

        probability = sum(probabilities) / len(probabilities)
        confidence = max(probability, 1.0 - probability)
        return probability, confidence, [path.as_posix() for path in feature_paths]


@dataclass
class LocalClassifierHeadProvider:
    model_path: str

    def classify_case(self, case: SafetyCase) -> BinaryModelOutput:
        raise RuntimeError(
            f"local classifier head at {self.model_path!r} is configured but no local inference adapter is implemented"
        )


@dataclass
class LocalTextGenerationProvider:
    model_path: str
    max_new_tokens: int = 128
    trust_remote_code: bool = True
    device_map: Any | None = "auto"
    torch_dtype: str | None = "auto"
    max_memory: dict[Any, Any] | None = None
    offload_folder: str | None = None
    use_chat_template: bool = True
    enable_thinking: bool | None = None
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    cache_model: bool = True
    disable_torch_compile: bool = False
    patch_torch_distributed_tensor: bool = False
    _backend: Any = field(default=None, init=False, repr=False)

    def complete(self, prompt: str) -> str:
        backend = self._get_backend()
        inputs = self._encode_prompt(backend.tokenizer, prompt)
        if hasattr(backend.model, "device") and hasattr(inputs, "to"):
            inputs = inputs.to(backend.model.device)
        input_length = int(inputs["input_ids"].shape[-1])

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            if self.temperature is not None:
                generation_kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                generation_kwargs["top_p"] = self.top_p
        if getattr(backend.tokenizer, "eos_token_id", None) is not None:
            generation_kwargs["eos_token_id"] = backend.tokenizer.eos_token_id
        if getattr(backend.tokenizer, "pad_token_id", None) is None and getattr(backend.tokenizer, "eos_token_id", None) is not None:
            generation_kwargs["pad_token_id"] = backend.tokenizer.eos_token_id

        torch = _import_torch()
        backend.model.eval()
        with torch.inference_mode():
            output_ids = backend.model.generate(**inputs, **generation_kwargs)
        completion_ids = output_ids[0][input_length:]
        return backend.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    def _get_backend(self):
        if self._backend is not None:
            return self._backend
        cache_key = self._cache_key()
        if self.cache_model and cache_key in _LOCAL_GENERATION_BACKEND_CACHE:
            self._backend = _LOCAL_GENERATION_BACKEND_CACHE[cache_key]
            return self._backend

        backend = _load_transformers_backend(
            model_path=self.model_path,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            max_memory=self.max_memory,
            offload_folder=self.offload_folder,
            disable_torch_compile=self.disable_torch_compile,
            patch_torch_distributed_tensor=self.patch_torch_distributed_tensor,
        )
        if self.cache_model:
            _LOCAL_GENERATION_BACKEND_CACHE[cache_key] = backend
        self._backend = backend
        return backend

    def _cache_key(self) -> tuple[str, bool, str, str | None, str, str | None, bool | None, bool, bool]:
        path = Path(self.model_path).expanduser()
        resolved_path = path.resolve().as_posix() if path.exists() else os.path.abspath(os.path.expandvars(str(path)))
        return (
            resolved_path,
            self.trust_remote_code,
            _stable_config_key(self.device_map),
            self.torch_dtype,
            _stable_config_key(self.max_memory),
            self.offload_folder,
            self.enable_thinking,
            self.disable_torch_compile,
            self.patch_torch_distributed_tensor,
        )

    def _encode_prompt(self, tokenizer: Any, prompt: str) -> Any:
        rendered_prompt = prompt
        if self.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            try:
                template_kwargs: dict[str, Any] = {}
                if self.enable_thinking is not None:
                    template_kwargs["enable_thinking"] = self.enable_thinking
                rendered_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            except (KeyError, TypeError, ValueError):
                rendered_prompt = prompt
        return tokenizer(rendered_prompt, return_tensors="pt")


@dataclass
class LocalPromptBinaryProvider:
    generator: TextGenerationProvider
    fallback_label: int | None = None
    use_llm_parse_fallback: bool = True

    def classify_prompt(self, prompt: str) -> BinaryModelOutput:
        response = self.generator.complete(prompt)
        raw: dict[str, Any] = {
            "provider": "local_prompt_binary_lm",
            "prompt": prompt,
            "response": response,
            "confidence_source": "method_default",
        }
        try:
            label = parse_binary_label_from_text(response)
        except ValueError as exc:
            raw["parse_error"] = str(exc)
            label = self._fallback_parse_with_llm(prompt=prompt, response=response, raw=raw)
            if label is None:
                if self.fallback_label is None:
                    raise
                label = parse_binary_label(self.fallback_label)
                raw["fallback_label"] = label
        return BinaryModelOutput(label=label, confidence=None, raw=raw)

    def _fallback_parse_with_llm(self, *, prompt: str, response: str, raw: dict[str, Any]) -> int | None:
        if not self.use_llm_parse_fallback:
            return None
        fallback_prompt = _BINARY_PARSE_FALLBACK_PROMPT.format(prompt=prompt, response=response)
        fallback_response = self.generator.complete(fallback_prompt)
        raw["parse_fallback_prompt"] = fallback_prompt
        raw["parse_fallback_response"] = fallback_response
        try:
            label = parse_binary_label_from_text(fallback_response)
        except ValueError as exc:
            raw["parse_fallback_error"] = str(exc)
            return None
        raw["parse_fallback_label"] = label
        return label


@dataclass
class _TransformersBackend:
    tokenizer: Any
    model: Any


@dataclass
class _OneCaseMultimodalRuntime:
    module: Any
    model: Any
    processor: Any
    probe_model: Any
    device: str


_LOCAL_GENERATION_BACKEND_CACHE: dict[
    tuple[str, bool, str, str | None, str, str | None, bool | None, bool, bool],
    _TransformersBackend,
] = {}
_MERGED_SAFEGUARD_RUNTIME_CACHE: dict[tuple[str, str, str], Any] = {}
_QWEN3GUARD_RUNTIME_CACHE: dict[tuple[str], tuple[Any, Any]] = {}
_ONE_CASE_MULTIMODAL_RUNTIME_CACHE: dict[tuple[str, str, str, int], _OneCaseMultimodalRuntime] = {}


def load_provider_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"provider config root must be a mapping: {path}")
    payload.setdefault("__config_dir__", str(config_path.parent))
    return payload


def build_binary_provider(config: dict[str, Any]) -> Any:
    provider_type = config.get("type")
    if provider_type == "prompt_binary_api":
        return PromptBinaryApiProvider(
            base_url=str(config["base_url"]),
            api_key_env=config.get("api_key_env"),
            timeout_seconds=int(config.get("timeout_seconds", 30)),
            prompt_field=str(config.get("prompt_field", "prompt")),
        )
    if provider_type == "classifier_head_api":
        return ClassifierHeadApiProvider(
            base_url=str(config["base_url"]),
            api_key_env=config.get("api_key_env"),
            timeout_seconds=int(config.get("timeout_seconds", 30)),
        )
    if provider_type == "mock_prompt_binary":
        return MockPromptBinaryProvider(
            default_label=parse_binary_label(config.get("default_label", 0)),
            default_confidence=config.get("default_confidence", 0.8),
            unsafe_keywords=list(config.get("unsafe_keywords") or []),
            safe_keywords=list(config.get("safe_keywords") or []),
            refuse_keywords=list(config.get("refuse_keywords") or []),
        )
    if provider_type == "mock_classifier_head":
        return MockClassifierHeadProvider(
            default_label=parse_binary_label(config.get("default_label", 0)),
            default_confidence=config.get("default_confidence", 0.8),
        )
    if provider_type == "local_classifier_head":
        return LocalClassifierHeadProvider(model_path=resolve_provider_path(config["model_path"], config))
    if provider_type == "local_prompt_binary_lm":
        fallback_label = config.get("fallback_label")
        return LocalPromptBinaryProvider(
            generator=build_local_text_generation_provider(config),
            fallback_label=None if fallback_label is None else parse_binary_label(fallback_label),
            use_llm_parse_fallback=bool(config.get("use_llm_parse_fallback", True)),
        )
    if provider_type == "subprocess_prompt_binary":
        fallback_label = config.get("fallback_label")
        return LocalPromptBinaryProvider(
            generator=build_subprocess_text_generation_provider(config),
            fallback_label=None if fallback_label is None else parse_binary_label(fallback_label),
            use_llm_parse_fallback=bool(config.get("use_llm_parse_fallback", True)),
        )
    if provider_type == "merged_safeguard_prompt_binary":
        fallback_label = config.get("fallback_label")
        return LocalPromptBinaryProvider(
            generator=build_merged_safeguard_provider(config),
            fallback_label=None if fallback_label is None else parse_binary_label(fallback_label),
            use_llm_parse_fallback=bool(config.get("use_llm_parse_fallback", True)),
        )
    if provider_type == "ascend_vllm_prompt_binary":
        fallback_label = config.get("fallback_label")
        return LocalPromptBinaryProvider(
            generator=build_ascend_vllm_chat_provider(config),
            fallback_label=None if fallback_label is None else parse_binary_label(fallback_label),
            use_llm_parse_fallback=bool(config.get("use_llm_parse_fallback", True)),
        )
    raise ValueError(f"unknown binary provider type: {provider_type!r}")


def build_text_generation_provider(config: dict[str, Any]) -> TextGenerationProvider:
    provider_type = config.get("type")
    if provider_type == "local_text_generation_lm":
        return build_local_text_generation_provider(config)
    if provider_type == "mock_text_generation":
        return MockTextGenerationProvider(
            unsafe_keywords=list(config.get("unsafe_keywords") or []),
            safe_keywords=list(config.get("safe_keywords") or []),
            refuse_keywords=list(config.get("refuse_keywords") or []),
            default_response=str(config.get("default_response", "safe: no configured risk detected.")),
        )
    if provider_type == "subprocess_text_generation":
        return build_subprocess_text_generation_provider(config)
    if provider_type == "qwen3guard_subprocess":
        return build_qwen3guard_subprocess_provider(config)
    if provider_type in {"qwen3guard_local", "qwen3guard_script"}:
        return build_qwen3guard_provider(config)
    if provider_type in {"merged_safeguard_local", "merged_safeguard_script"}:
        return build_merged_safeguard_provider(config)
    if provider_type == "ascend_vllm_chat":
        return build_ascend_vllm_chat_provider(config)
    raise ValueError(f"unknown text generation provider type: {provider_type!r}")


def build_multimodal_provider(config: dict[str, Any]) -> MultimodalCaseProvider:
    provider_type = config.get("type")
    if provider_type == "mock_multimodal_probe":
        return MockMultimodalProbeProvider(
            default_label=parse_binary_label(config.get("default_label", 0)),
            default_confidence=config.get("default_confidence", 0.8),
            unsafe_keywords=list(config.get("unsafe_keywords") or []),
        )
    if provider_type == "cached_multimodal_probe":
        return CachedMultimodalProbeProvider(
            predictions_path=resolve_provider_path(config["predictions_path"], config),
            default_confidence=config.get("default_confidence"),
        )
    if provider_type in {"qwen_vl_projection_probe", "one_case_multimodal_probe"}:
        return build_qwen_vl_projection_probe_provider(config)
    raise ValueError(f"unknown multimodal provider type: {provider_type!r}")


def build_qwen_vl_projection_probe_provider(config: dict[str, Any]) -> QwenVlProjectionProbeProvider:
    return QwenVlProjectionProbeProvider(
        model_path=resolve_provider_path(config["model_path"], config),
        probe_model_path=resolve_provider_path(config["probe_model_path"], config),
        device=str(config.get("device", "auto")),
        save_root=resolve_provider_path(config.get("save_root", "outputs/multimodal_probe"), config),
        save_layers=config.get("save_layers", [42]),
        source_layer=int(config.get("source_layer", 42)),
        target_layers=config.get("target_layers", 27),
        project_save_layers=config.get("project_save_layers", "last"),
        token_slice=config.get("token_slice", "all"),
        probe_input_dim=int(config.get("probe_input_dim", 248320)),
        threshold=float(config.get("threshold", 0.5)),
        cache_model=bool(config.get("cache_model", True)),
        disable_torch_compile=bool(config.get("disable_torch_compile", False)),
        patch_torch_distributed_tensor=bool(config.get("patch_torch_distributed_tensor", False)),
    )


def build_merged_safeguard_provider(config: dict[str, Any]) -> MergedSafeGuardProvider:
    return MergedSafeGuardProvider(
        model_path=resolve_provider_path(config["model_path"], config),
        device=str(config.get("device", "auto")),
        torch_dtype=str(config.get("torch_dtype", "bfloat16")),
        max_new_tokens=int(config.get("max_new_tokens", 32)),
        cache_model=bool(config.get("cache_model", True)),
    )


def build_qwen3guard_provider(config: dict[str, Any]) -> Qwen3GuardProvider:
    return Qwen3GuardProvider(
        model_path=resolve_provider_path(config["model_path"], config),
        max_new_tokens=int(config.get("max_new_tokens", 128)),
        controversial_label=str(config.get("controversial_label", "unsafe")),
        use_refusal_as_unsafe=bool(config.get("use_refusal_as_unsafe", False)),
        cache_model=bool(config.get("cache_model", True)),
    )


def build_qwen3guard_subprocess_provider(config: dict[str, Any]) -> Qwen3GuardSubprocessProvider:
    return Qwen3GuardSubprocessProvider(
        python_executable=str(config.get("python_executable", "python3")),
        script_path=resolve_provider_path(config["script_path"], config),
        model_path=resolve_provider_path(config["model_path"], config),
        timeout_seconds=int(config.get("timeout_seconds", 600)),
        max_new_tokens=int(config.get("max_new_tokens", 128)),
        controversial_label=str(config.get("controversial_label", "unsafe")),
        use_refusal_as_unsafe=bool(config.get("use_refusal_as_unsafe", False)),
        cwd=resolve_provider_path(config["cwd"], config) if config.get("cwd") else None,
        env={str(key): str(value) for key, value in dict(config.get("env") or {}).items()},
    )


def build_subprocess_text_generation_provider(config: dict[str, Any]) -> SubprocessTextGenerationProvider:
    return SubprocessTextGenerationProvider(
        command=_build_subprocess_command(config),
        timeout_seconds=int(config.get("timeout_seconds", 300)),
        input_mode=str(config.get("input_mode", "stdin")),
        prompt_arg=config.get("prompt_arg"),
        stdin_json=dict(config.get("stdin_json") or {}),
        cwd=resolve_provider_path(config["cwd"], config) if config.get("cwd") else None,
        env={str(key): str(value) for key, value in dict(config.get("env") or {}).items()},
        output_json_field=config.get("output_json_field"),
    )


def build_ascend_vllm_chat_provider(config: dict[str, Any]) -> AscendVllmChatProvider:
    return AscendVllmChatProvider(
        api_base=str(config.get("api_base") or config.get("base_url") or "http://127.0.0.1:8000/v1"),
        model=str(config.get("model", "safeguard-merged")),
        api_key_env=config.get("api_key_env"),
        timeout_seconds=int(config.get("timeout_seconds", 300)),
        max_tokens=int(config.get("max_tokens", config.get("max_new_tokens", 32))),
        temperature=_optional_float(config.get("temperature", 0.0)),
        top_p=_optional_float(config.get("top_p")),
        system_prompt=_optional_string(config.get("system_prompt")),
        chat_template_kwargs=_coerce_chat_template_kwargs(config),
        extra_payload=dict(config.get("extra_payload") or {}),
        endpoint=str(config.get("endpoint", "/chat/completions")),
    )


def _build_subprocess_command(config: dict[str, Any]) -> list[str]:
    if "command" in config:
        return [str(item) for item in list(config["command"])]

    if "script_path" not in config:
        raise ValueError("subprocess providers require command or script_path")

    command = [
        str(config.get("python_executable", "python3")),
        resolve_provider_path(config["script_path"], config),
    ]
    if config.get("model_path"):
        command.extend([str(config.get("model_arg", "--model-path")), resolve_provider_path(config["model_path"], config)])
    command.extend(str(item) for item in list(config.get("extra_args") or []))
    return command


def build_local_text_generation_provider(config: dict[str, Any]) -> LocalTextGenerationProvider:
    return LocalTextGenerationProvider(
        model_path=resolve_provider_path(config["model_path"], config),
        max_new_tokens=int(config.get("max_new_tokens", 128)),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        device_map=config.get("device_map", "auto"),
        torch_dtype=_optional_string(config.get("torch_dtype", "auto")),
        max_memory=config.get("max_memory"),
        offload_folder=resolve_provider_path(config["offload_folder"], config) if config.get("offload_folder") else None,
        use_chat_template=bool(config.get("use_chat_template", True)),
        enable_thinking=_optional_bool(config.get("enable_thinking")),
        do_sample=bool(config.get("do_sample", False)),
        temperature=_optional_float(config.get("temperature")),
        top_p=_optional_float(config.get("top_p")),
        cache_model=bool(config.get("cache_model", True)),
        disable_torch_compile=bool(config.get("disable_torch_compile", False)),
        patch_torch_distributed_tensor=bool(config.get("patch_torch_distributed_tensor", False)),
    )


def resolve_provider_path(value: Any, config: dict[str, Any]) -> str:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False).as_posix()
    config_dir = config.get("__config_dir__")
    if config_dir:
        return (Path(str(config_dir)) / expanded).resolve(strict=False).as_posix()
    return expanded.resolve(strict=False).as_posix()


def parse_binary_label(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        if value in {0, 1}:
            return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"1", "unsafe", "risk", "risky", "harmful", "danger"}:
            return 1
        if lowered in {"0", "safe", "benign", "allowed", "normal"}:
            return 0
    raise ValueError(f"cannot parse binary label from {value!r}")


def parse_binary_label_from_text(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        raise ValueError("cannot parse binary label from empty text")

    json_payloads = _extract_json_payloads(stripped)
    for json_payload in reversed(json_payloads):
        try:
            return BinaryModelOutput.from_payload(json_payload).label
        except ValueError:
            continue

    lowered = stripped.casefold()
    json_label_match = re.search(
        r"[\"']?(?:label|prediction|prediction_label|verdict|safety)[\"']?\s*[:：]\s*[\"']?"
        r"(unsafe|safe|1|0|不安全|安全|有害|无害|违规|合规|危险)\s*"
        r"(?=[\"',}\]\n\r]|$)",
        stripped,
        flags=re.IGNORECASE,
    )
    if json_label_match:
        return parse_binary_label(_normalize_text_label(json_label_match.group(1)))

    explicit_label_pattern = (
        r"(?:^|[\n\r])\s*(?:label|prediction|prediction_label|verdict|safety|final\s+(?:answer|label|verdict)|"
        r"标签|结论|判定|最终(?:答案|标签|结论|判定))\s*[:：]\s*"
        r"(unsafe|safe|1|0|不安全|安全|有害|无害|违规|合规|危险)"
    )
    label_match = re.search(explicit_label_pattern, lowered)
    if label_match:
        return parse_binary_label(_normalize_text_label(label_match.group(1)))

    chinese_sentence_label_match = re.search(
        r"(?:^|[\n\r])\s*[^\n\r。！？!?]{0,80}?(?:是|为|属于|判为|判断为)\s*[*`\"']*"
        r"(不安全|安全|unsafe|safe)[*`\"']*",
        stripped,
        flags=re.IGNORECASE,
    )
    if chinese_sentence_label_match:
        return parse_binary_label(_normalize_text_label(chinese_sentence_label_match.group(1)))

    first_line = next((line.strip() for line in stripped.splitlines() if line.strip()), "")
    first_token = re.split(r"[\s,，。.:：;；]+", first_line.casefold(), maxsplit=1)[0].strip("\"'`*[](){}")
    if first_token:
        try:
            return parse_binary_label(_normalize_text_label(first_token))
        except ValueError:
            pass

    raise ValueError(f"cannot parse binary label from generated text: {text!r}")


def _first_present(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    raise ValueError(f"payload missing any of required keys: {keys}")


def _first_optional(payload: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _first_keyword_match(lowered_text: str, keywords: list[str]) -> str | None:
    for keyword in keywords:
        if keyword.casefold() in lowered_text:
            return keyword
    return None


def _first_image_attachment(case: SafetyCase) -> str | None:
    for attachment in case.attachments:
        text = str(attachment).strip()
        if text:
            return text
    return None


def _safe_filename(value: str, limit: int = 40) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return (safe or "case")[:limit]


def _normalize_cache_path(value: str) -> str:
    expanded = Path(os.path.expandvars(value)).expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False).as_posix()
    return str(value).strip()


def _sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().casefold()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"cannot parse bool from {value!r}")


def _coerce_chat_template_kwargs(config: dict[str, Any]) -> dict[str, Any] | None:
    value = config.get("chat_template_kwargs", {"enable_thinking": False})
    if value is None:
        kwargs = None
    elif isinstance(value, dict):
        kwargs = dict(value)
    else:
        raise TypeError("chat_template_kwargs must be a mapping or null")

    if "enable_thinking" in config:
        if kwargs is None:
            kwargs = {}
        kwargs["enable_thinking"] = _optional_bool(config["enable_thinking"])
    return kwargs


def _stable_config_key(value: Any) -> str:
    return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, default=str)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _render_template_value(value: Any, prompt: str) -> Any:
    if isinstance(value, str):
        if "{prompt}" not in value:
            return value
        return value.format(prompt=prompt)
    if isinstance(value, dict):
        return {str(key): _render_template_value(item, prompt) for key, item in value.items()}
    if isinstance(value, list):
        return [_render_template_value(item, prompt) for item in value]
    return value


def _lookup_json_field(payload: dict[str, Any], field_path: str) -> Any:
    current: Any = payload
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"subprocess output missing JSON field {field_path!r}")
        current = current[part]
    return current


def _dataclass_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        return {field_name: getattr(value, field_name) for field_name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"expected dataclass or dict result, got {type(value).__name__}")


def _extract_question_answer_prompt(prompt: str) -> tuple[str, str]:
    match = re.search(
        r"^\s*Question:[ \t]*(?P<question>.*?)(?:\r?\n[ \t]*Answer:[ \t]*(?P<answer>.*))?\s*$",
        prompt,
        flags=re.DOTALL,
    )
    if not match:
        return prompt, ""
    question = (match.group("question") or "").strip()
    answer = (match.group("answer") or "").strip()
    if answer:
        return question, answer
    return question or prompt, answer


def _truncate_for_error(value: str, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _normalize_text_label(value: str) -> str:
    normalized = value.strip().casefold()
    mapping = {
        "不安全": "unsafe",
        "有害": "unsafe",
        "违规": "unsafe",
        "危险": "unsafe",
        "安全": "safe",
        "无害": "safe",
        "合规": "safe",
    }
    return mapping.get(normalized, normalized)


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    payloads = _extract_json_payloads(text)
    return payloads[-1] if payloads else None


def _extract_json_payloads(text: str) -> list[dict[str, Any]]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate).strip()
    decoder = json.JSONDecoder()
    payloads: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", candidate):
        try:
            payload, _ = decoder.raw_decode(candidate[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _extract_chat_completion_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"chat completion response missing choices: {payload!r}")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError(f"chat completion choice must be an object: {choice!r}")

    message = choice.get("message")
    if isinstance(message, dict) and "content" in message:
        return _coerce_chat_content_to_text(message.get("content"))
    if "text" in choice:
        return _coerce_chat_content_to_text(choice.get("text"))
    raise ValueError(f"chat completion choice missing message.content: {choice!r}")


def _coerce_chat_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return str(content).strip()


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "local model inference requires PyTorch. Install the optional local-model dependencies before using a local LM provider."
        ) from exc
    return torch


def _module_torch(module: Any) -> Any:
    module_torch = getattr(module, "torch", None)
    return module_torch if module_torch is not None else _import_torch()


def _load_transformers_backend(
    *,
    model_path: str,
    trust_remote_code: bool,
    device_map: Any | None,
    torch_dtype: str | None,
    max_memory: dict[Any, Any] | None,
    offload_folder: str | None,
    disable_torch_compile: bool = False,
    patch_torch_distributed_tensor: bool = False,
) -> _TransformersBackend:
    torch = _import_torch()
    original_torch_compile = _apply_transformers_load_compat(
        torch,
        disable_torch_compile=disable_torch_compile,
        patch_torch_distributed_tensor=patch_torch_distributed_tensor,
    )
    try:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "local model inference requires transformers. Install the optional local-model dependencies before using a local LM provider."
            ) from exc

        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory
        if offload_folder is not None:
            model_kwargs["offload_folder"] = offload_folder
        dtype = _coerce_torch_dtype(torch, torch_dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        return _TransformersBackend(tokenizer=tokenizer, model=model)
    finally:
        if original_torch_compile is not None:
            torch.compile = original_torch_compile


def _apply_transformers_load_compat(
    torch: Any,
    *,
    disable_torch_compile: bool,
    patch_torch_distributed_tensor: bool,
) -> Any | None:
    original_torch_compile = None
    if patch_torch_distributed_tensor:
        _patch_torch_distributed_tensor_namespace()
    if disable_torch_compile and hasattr(torch, "compile"):
        original_torch_compile = torch.compile

        def identity_compile(model: Any = None, *args: Any, **kwargs: Any) -> Any:
            if model is None:
                return lambda wrapped: wrapped
            return model

        torch.compile = identity_compile
    return original_torch_compile


def _patch_torch_distributed_tensor_namespace() -> None:
    try:
        import torch.distributed._tensor as source_module
        import torch.distributed.tensor as target_module
    except (ImportError, AttributeError):
        return

    for name in ("DTensor", "Placement", "Replicate", "Shard", "distribute_module"):
        if not hasattr(target_module, name) and hasattr(source_module, name):
            setattr(target_module, name, getattr(source_module, name))


def _coerce_torch_dtype(torch: Any, value: str | None) -> Any:
    if value is None:
        return None
    normalized = value.strip().casefold()
    if normalized == "auto":
        return "auto"
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    dtype_name = aliases.get(normalized, normalized)
    if not hasattr(torch, dtype_name):
        raise ValueError(f"unknown torch dtype {value!r}")
    return getattr(torch, dtype_name)


def _http_json_transport(request: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(request["json"], ensure_ascii=False).encode("utf-8")
    http_request = urllib.request.Request(
        str(request["url"]),
        data=body,
        headers=dict(request.get("headers") or {}),
        method="POST",
    )
    with urllib.request.urlopen(http_request, timeout=int(request.get("timeout_seconds", 30))) as response:
        response_body = response.read().decode("utf-8")
    payload = json.loads(response_body)
    if not isinstance(payload, dict):
        raise ValueError("model provider response must be a JSON object")
    return payload
