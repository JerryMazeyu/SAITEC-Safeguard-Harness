from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safeguard_harness.runtimes.devices import default_model_dtype, import_torch, resolve_torch_device

PROMPT_KEYS = (
    "Question",
    "question",
    "Rephrased Question",
    "rephrased_question",
    "Rephrased Question(SD)",
    "prompt",
    "Prompt",
)
IMAGE_KEYS = ("image", "image_path", "image_file", "img")


def load_model(model_path: str, device: str = "auto") -> tuple[Any, Any, str]:
    torch = import_torch()
    resolved_device = resolve_torch_device(device)
    try:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Qwen VL projection probe requires transformers with Qwen VL support."
        ) from exc

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=default_model_dtype(torch, resolved_device),
        trust_remote_code=True,
    ).to(resolved_device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    return model, processor, resolved_device


def get_item_value(item: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in item and item[key]:
            return item[key]
    return None


def load_image(image_input: Any) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Qwen VL projection probe requires Pillow to load images.") from exc

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    raise ValueError(f"unsupported image input type: {type(image_input)}")


def resolve_sample(data: Any, index: int) -> tuple[Any, str, Any, dict[str, Any]]:
    if isinstance(data, dict):
        keys = list(data.keys())
        sample_id = keys[index]
        item = data[sample_id]
    elif isinstance(data, list):
        sample_id = index
        item = data[index]
    else:
        raise TypeError(f"unsupported data format: {type(data)}")

    if not isinstance(item, dict):
        raise TypeError(f"sample {sample_id!r} must be a dict, got {type(item)}")

    prompt = get_item_value(item, PROMPT_KEYS)
    image_path = get_item_value(item, IMAGE_KEYS)
    if prompt is None or image_path is None:
        raise KeyError(f"sample {sample_id!r} is missing a prompt or image field")
    return sample_id, str(prompt), image_path, item


def extract_hidden_tensor(output: Any) -> Any:
    torch = import_torch()
    hidden_states = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(hidden_states):
        raise TypeError(f"hook output is not a tensor: {type(hidden_states)}")
    return hidden_states


def save_tensor(tensor: Any, save_path: str | Path) -> None:
    torch = import_torch()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), save_path)


def source_case_dir(save_root: str | Path, data_name: str, sample_index: int) -> Path:
    return Path(save_root) / "source" / data_name / f"case_{sample_index}"


def projected_case_dir(save_root: str | Path, data_name: str, sample_index: int) -> Path:
    return Path(save_root) / "projected" / data_name / f"case_{sample_index}"


def projected_data_dir(save_root: str | Path, data_name: str) -> Path:
    return Path(save_root) / "projected" / data_name


def make_save_hook(save_root: str | Path, data_name: str, sample_index: int, layer_idx: int):
    save_path = source_case_dir(save_root, data_name, sample_index) / f"layer_{layer_idx}.pt"

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        save_tensor(extract_hidden_tensor(output), save_path)

    return hook


def get_text_model(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    raise AttributeError("failed to locate the Qwen text model.")


def get_text_layers(model: Any) -> Any:
    text_model = get_text_model(model)
    if not hasattr(text_model, "layers"):
        raise AttributeError("failed to locate Qwen text layers.")
    return text_model.layers


def build_messages(prompt: str, image_path: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": load_image(image_path)},
                {"type": "text", "text": str(prompt)},
            ],
        }
    ]


def build_batch_inputs(processor: Any, prompts: list[str], image_paths: list[str], device: str) -> Any:
    messages_list = [build_messages(prompt, image_path) for prompt, image_path in zip(prompts, image_paths)]
    prompt_texts = []
    image_inputs = []
    for messages in messages_list:
        try:
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt_texts.append(prompt_text)
        image_inputs.append(messages[0]["content"][0]["image"])

    try:
        inputs = processor(
            text=prompt_texts,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
    except TypeError:
        inputs = processor(
            text=prompt_texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
    return inputs.to(device)


def qwen_vl_batch_infer(
    model: Any,
    processor: Any,
    prompts: list[str],
    image_paths: list[str],
    max_new_tokens: int = 1,
) -> list[str]:
    if len(prompts) != len(image_paths):
        raise ValueError("image and text counts must match")

    torch = import_torch()
    inputs = build_batch_inputs(processor, prompts, image_paths, model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def save_generate_all_layer_all_token(
    sample_index: int,
    data: Any,
    data_name: str,
    model: Any,
    processor: Any,
    layer_indices: list[int],
    save_root: str | Path,
    max_new_tokens: int = 1,
) -> list[str]:
    _sample_id, prompt, image_path, _item = resolve_sample(data, sample_index)
    layers = get_text_layers(model)
    handles = []
    for layer_idx in layer_indices:
        handles.append(layers[layer_idx].register_forward_hook(make_save_hook(save_root, data_name, sample_index, layer_idx)))

    try:
        return qwen_vl_batch_infer(model, processor, [prompt], [image_path], max_new_tokens=max_new_tokens)
    finally:
        for handle in handles:
            handle.remove()


def save_merge_layer(
    data_path: str | Path,
    save_root: str | Path,
    model: Any,
    processor: Any,
    save_layers: list[int],
) -> None:
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    data_name = Path(data_path).stem
    for sample_index in _progress(range(len(data)), desc=f"Saving: {data_name}"):
        save_generate_all_layer_all_token(
            sample_index,
            data,
            data_name,
            model,
            processor,
            layer_indices=save_layers,
            save_root=save_root,
        )


def parse_slice_spec(slice_spec: Any) -> slice:
    if isinstance(slice_spec, slice):
        return slice_spec
    if slice_spec is None:
        return slice(None, None, None)
    text = str(slice_spec).strip().lower()
    if text in {"all", ":", "slice(none, none, none)"}:
        return slice(None, None, None)
    if ":" not in text:
        index = int(text)
        return slice(index, index + 1, None)

    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"cannot parse slice: {slice_spec}")
    parts += [""] * (3 - len(parts))
    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if parts[1] else None
    step = int(parts[2]) if parts[2] else None
    return slice(start, end, step)


def parse_layer_spec(layer_spec: Any, num_layers: int, default_last: bool = False) -> list[int]:
    if layer_spec is None:
        layer_indices = [num_layers - 1] if default_last else []
    elif isinstance(layer_spec, int):
        layer_indices = [layer_spec]
    elif isinstance(layer_spec, (list, tuple)):
        layer_indices = [int(layer) for layer in layer_spec]
    else:
        text = str(layer_spec).strip().lower()
        if text == "last":
            layer_indices = [num_layers - 1]
        elif text in {"all", "full"}:
            layer_indices = list(range(num_layers))
        else:
            layer_indices = []
            for part in str(layer_spec).split(","):
                part = part.strip()
                if not part:
                    continue
                lower_part = part.lower()
                if lower_part == "last":
                    layer_indices.append(num_layers - 1)
                elif lower_part in {"all", "full"}:
                    layer_indices.extend(range(num_layers))
                elif "-" in part:
                    start, end = part.split("-", 1)
                    layer_indices.extend(range(int(start), int(end) + 1))
                else:
                    layer_indices.append(int(part))

    deduped = []
    seen = set()
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"layer index out of range: {layer_idx}; valid range [0, {num_layers - 1}]")
        if layer_idx not in seen:
            deduped.append(layer_idx)
            seen.add(layer_idx)
    return deduped


def normalize_hidden_state_shape(hidden_states: Any) -> Any:
    if hidden_states.ndim == 1:
        return hidden_states.unsqueeze(0).unsqueeze(0)
    if hidden_states.ndim == 2:
        return hidden_states.unsqueeze(0)
    if hidden_states.ndim == 3:
        return hidden_states
    raise ValueError(f"unsupported hidden-state dimensions: {hidden_states.shape}")


def align_injected_state(injected_state: Any, target_slice: Any) -> Any:
    if injected_state.shape[-1] != target_slice.shape[-1]:
        raise ValueError(
            f"hidden size mismatch: source={injected_state.shape[-1]}, target={target_slice.shape[-1]}"
        )

    if injected_state.shape[0] == 1 and target_slice.shape[0] != 1:
        injected_state = injected_state.expand(target_slice.shape[0], -1, -1)
    elif injected_state.shape[0] != target_slice.shape[0]:
        raise ValueError(
            f"batch size mismatch: source={injected_state.shape[0]}, target={target_slice.shape[0]}"
        )

    if injected_state.shape[1] == 1 and target_slice.shape[1] != 1:
        injected_state = injected_state.expand(-1, target_slice.shape[1], -1)
    elif injected_state.shape[1] != target_slice.shape[1]:
        raise ValueError(
            f"token size mismatch: source={injected_state.shape[1]}, target={target_slice.shape[1]}"
        )
    return injected_state


def project_hidden_slice(current_hidden_states: Any, injected_hidden_states: Any, token_slice: slice) -> Any:
    torch = import_torch()
    target_slice = current_hidden_states[:, token_slice, :].to(torch.float32)
    injected_hidden_states = align_injected_state(
        injected_hidden_states.to(current_hidden_states.device, dtype=torch.float32),
        target_slice,
    )
    dot = torch.sum(target_slice * injected_hidden_states, dim=-1, keepdim=True)
    norm2 = torch.sum(injected_hidden_states * injected_hidden_states, dim=-1, keepdim=True).clamp_min(1e-6)
    projected = (dot / norm2) * injected_hidden_states
    new_hidden_states = current_hidden_states.clone()
    new_hidden_states[:, token_slice, :] = projected.to(current_hidden_states.dtype)
    return new_hidden_states


def make_projection_hook(injected_hidden_states: Any, token_slice: slice):
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden_states = extract_hidden_tensor(output)
        projected_hidden_states = project_hidden_slice(hidden_states, injected_hidden_states, token_slice)
        if isinstance(output, tuple):
            output_list = list(output)
            output_list[0] = projected_hidden_states
            return tuple(output_list)
        if isinstance(output, list):
            output[0] = projected_hidden_states
            return output
        return projected_hidden_states

    return hook


def make_projection_save_hook(save_dir: str | Path, layer_idx: int):
    save_path = Path(save_dir) / f"layer_{layer_idx}.pt"

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        save_tensor(extract_hidden_tensor(output), save_path)

    return hook


def substitute_and_save_last(
    source_layer: int,
    target_layers: Any,
    sample_index: int,
    data: Any,
    data_name: str,
    model: Any,
    processor: Any,
    token_slice_spec: Any,
    save_root: str | Path,
    save_layers: Any,
) -> list[str]:
    _sample_id, prompt, image_path, _item = resolve_sample(data, sample_index)
    layers = get_text_layers(model)
    token_slice = parse_slice_spec(token_slice_spec)
    parsed_target_layers = parse_layer_spec(target_layers, len(layers), default_last=False)
    if not parsed_target_layers:
        raise ValueError("target layers cannot be empty")
    parsed_save_layers = parse_layer_spec(save_layers, len(layers), default_last=True)

    source_path = source_case_dir(save_root, data_name, sample_index) / f"layer_{source_layer}.pt"
    if not source_path.exists():
        raise FileNotFoundError(f"source hidden state does not exist: {source_path}")
    injected_hidden_states = normalize_hidden_state_shape(load_hidden_states(source_path))

    projection_dir = (
        projected_case_dir(save_root, data_name, sample_index)
        / f"slice_{_safe_slice_name(token_slice)}_layer_{source_layer}_to_{_safe_layers_name(parsed_target_layers)}"
    )

    handles = []
    for layer_idx in parsed_target_layers:
        handles.append(layers[layer_idx].register_forward_hook(make_projection_hook(injected_hidden_states, token_slice)))
    for layer_idx in parsed_save_layers:
        handles.append(layers[layer_idx].register_forward_hook(make_projection_save_hook(projection_dir, layer_idx)))

    try:
        return qwen_vl_batch_infer(model, processor, [prompt], [image_path])
    finally:
        for handle in handles:
            handle.remove()


def run_project_mode(
    data_path: str | Path,
    save_root: str | Path,
    model: Any,
    processor: Any,
    source_layer: int,
    target_layers: Any,
    project_save_layers: Any,
    token_slice: Any,
) -> None:
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    data_name = Path(data_path).stem
    for sample_index in _progress(range(len(data)), desc=f"Projecting: {data_name}"):
        substitute_and_save_last(
            source_layer,
            target_layers,
            sample_index,
            data,
            data_name,
            model,
            processor,
            token_slice_spec=token_slice,
            save_root=save_root,
            save_layers=project_save_layers,
        )


def load_hidden_states(layer_file: str | Path) -> Any:
    torch = import_torch()
    try:
        return torch.load(layer_file, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(layer_file, map_location="cpu")


def normalize_last_token_state(hidden_states: Any) -> Any:
    if hidden_states.ndim == 3:
        return hidden_states[0, -1, :]
    if hidden_states.ndim == 2:
        return hidden_states[-1]
    if hidden_states.ndim == 1:
        return hidden_states
    raise ValueError(f"unsupported hidden-state shape: {tuple(hidden_states.shape)}")


def get_logit(hidden_states: Any, model: Any) -> Any:
    torch = import_torch()
    last_token_state = normalize_last_token_state(hidden_states)
    device = next(model.parameters()).device
    text_model = get_text_model(model)
    lm_head_weight = getattr(model.lm_head, "weight", None)
    target_dtype = lm_head_weight.dtype if lm_head_weight is not None else torch.float32
    hidden = last_token_state.unsqueeze(0).to(device=device, dtype=target_dtype)

    with torch.no_grad():
        final_norm = getattr(text_model, "norm", None)
        if final_norm is not None:
            hidden = final_norm(hidden)
        logits = model.lm_head(hidden).squeeze(0).float()

    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
    return logits.cpu()


def _safe_slice_name(token_slice: slice) -> str:
    return f"{token_slice.start or ''}_{token_slice.stop or ''}_{token_slice.step or ''}".strip("_") or "all"


def _safe_layers_name(layers: list[int]) -> str:
    return "-".join(str(layer) for layer in layers)


def _progress(iterable: Any, *, desc: str) -> Any:
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc)
