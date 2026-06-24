# Local Model Files

Put local classifier-head weights, tokenizer files, and model configs here only for local development.

The repository ignores `models/*` by default so large or sensitive model artifacts are not committed. Prefer referencing real paths with environment variables, for example:

```powershell
$env:SAFEGUARD_HEAD_MODEL_PATH="G:\Models\safeguard\classifier_head_v1"
```

If a model is served by an HTTP API, keep the files outside this repository and only configure `configs/providers/*.yaml`.

For large local base models, prefer a symlink instead of copying weights into this directory:

```bash
ln -sfn /ai/dataset/workspace/czy/model/Qwen3.6-27B models/Qwen3.6-27B
```

The current Qwen provider configs expect `models/Qwen3.6-27B` to resolve to the real model directory.
