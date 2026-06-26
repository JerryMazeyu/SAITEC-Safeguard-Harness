from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from safeguard_harness.runtimes.devices import (
    coerce_torch_dtype,
    import_torch,
    patch_broken_triton_namespace,
    resolve_torch_device,
)

SAFETY_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe)", re.IGNORECASE)


@dataclass
class MergedSafeGuardRuntime:
    model: Any
    processor: Any
    device: str


def parse_safety_label(text: str | None) -> str | None:
    if text is None:
        return None
    match = SAFETY_PATTERN.search(text)
    if not match:
        return None
    return "Unsafe" if match.group(1).casefold() == "unsafe" else "Safe"


def load_merged_safeguard(
    model_path: str,
    device: str = "auto",
    torch_dtype: str = "bfloat16",
) -> MergedSafeGuardRuntime:
    patch_broken_triton_namespace()
    torch = import_torch()
    resolved_device = resolve_torch_device(device)
    dtype = coerce_torch_dtype(torch, torch_dtype)

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "merged SafeGuard local inference requires transformers. Install the local-model dependencies."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    model.to(resolved_device)
    model.eval()

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.do_sample = False
        generation_config.top_k = None
        generation_config.top_p = None
        generation_config.temperature = None

    return MergedSafeGuardRuntime(model=model, processor=processor, device=resolved_device)


def build_chat_input(runtime: MergedSafeGuardRuntime, prompt: str) -> dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = runtime.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = runtime.processor(text=[text], images=None, videos=None, return_tensors="pt")
    return {key: value.to(runtime.device) for key, value in inputs.items()}


def infer_safety(
    runtime: MergedSafeGuardRuntime,
    prompt: str,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    torch = import_torch()
    inputs = build_chat_input(runtime, prompt)
    with torch.inference_mode():
        generated = runtime.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated[:, input_len:]
    output_text = runtime.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return {
        "prediction_text": output_text,
        "prediction_label": parse_safety_label(output_text),
    }

