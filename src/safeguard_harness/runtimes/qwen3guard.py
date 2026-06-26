from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safeguard_harness.runtimes.devices import import_torch, patch_broken_triton_namespace

LABEL_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)")
CATEGORIES_PATTERN = re.compile(r"Categories:\s*(.+)")
REFUSAL_PATTERN = re.compile(r"Refusal:\s*(Yes|No)")


@dataclass
class Qwen3GuardResult:
    task_type: str
    rendered_input_text: str
    model_output: str
    safety_label: str | None
    categories: list[str]
    refusal: str | None


def load_qwen3guard_gen8b_local(model_path: str | Path) -> tuple[Any, Any]:
    patch_broken_triton_namespace()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3Guard local inference requires transformers. Install the local-model dependencies."
        ) from exc

    model_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def parse_qwen3guard_output(text: str) -> tuple[str | None, list[str], str | None]:
    label_match = LABEL_PATTERN.search(text)
    categories_match = CATEGORIES_PATTERN.search(text)
    refusal_match = REFUSAL_PATTERN.search(text)

    label = label_match.group(1) if label_match else None
    refusal = refusal_match.group(1) if refusal_match else None
    if categories_match:
        raw_categories = categories_match.group(1).strip()
        if raw_categories.casefold() == "none":
            categories = []
        else:
            categories = [item.strip() for item in raw_categories.split(",") if item.strip()]
    else:
        categories = []
    return label, categories, refusal


def infer_qwen3guard_local(
    messages: list[dict[str, str]],
    tokenizer: Any,
    model: Any,
    max_new_tokens: int = 128,
) -> Qwen3GuardResult:
    torch = import_torch()
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    model_output = tokenizer.decode(output_ids, skip_special_tokens=True)
    safety_label, categories, refusal = parse_qwen3guard_output(model_output)
    task_type = "response" if messages and messages[-1]["role"] == "assistant" else "prompt"
    return Qwen3GuardResult(
        task_type=task_type,
        rendered_input_text=prompt_text,
        model_output=model_output,
        safety_label=safety_label,
        categories=categories,
        refusal=refusal,
    )


def infer_prompt_safety_local(
    user_prompt: str,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int = 128,
) -> Qwen3GuardResult:
    return infer_qwen3guard_local(
        messages=[{"role": "user", "content": user_prompt}],
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
    )


def infer_response_safety_local(
    user_prompt: str,
    assistant_response: str,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int = 128,
) -> Qwen3GuardResult:
    return infer_qwen3guard_local(
        messages=[
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ],
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
    )

