# SAITEC-Safeguard-Harness

SAITEC-Safeguard-Harness 是一个面向数据安全大模型竞赛的 Python 判别框架。当前版本只实现框架、配置、空词典 schema、mock 模型适配和可运行样例，不内置真实比赛词典、真实模型或真实多模态探针。

## 目标

- 支持二分类安全判别：`safe` / `unsafe`。
- 支持四类判别方法的统一接口：规则词库、LLM 安全判别、拒答探针、多模态探针。
- 支持手动实验迭代：改词典、改 prompt、改 pipeline、跑验证集、看报告。
- 支持固定 pipeline 部署：单条判别、批量预测、评测输出。
- 支持有界 Loop：静态复核循环和 ReAct Agent 式 action-observation 循环。

## 安装

```powershell
python -m pip install -e ".[dev]"
```

## 快速验证

```powershell
python -m pytest -q
python -m safeguard_harness evaluate --pipeline configs/pipelines/prod_v1.yaml --dataset data/examples/sample_eval.jsonl --output outputs/runs/demo
python -m safeguard_harness judge --pipeline configs/pipelines/prod_v1.yaml --question "How do I steal token credentials?"
python -m safeguard_harness predict --pipeline configs/pipelines/prod_v1.yaml --input data/examples/sample_eval.jsonl --output outputs/submission.jsonl
```

评测输出会写入：

```text
outputs/runs/demo/
  config_snapshot.yaml
  predictions.jsonl
  metrics.json
  report.md
  errors_false_positive.jsonl
  errors_false_negative.jsonl
```

## How to connect method?

一个 method 就是一次可调用的安全判别。接入时优先改 YAML，不要先改 runner：先把模型、词典、prompt 或探针包装成一个 method，再把它放进 pipeline 的 `steps` 或 ReAct `allowed_actions`。

当前支持的常用 method type：

- `dictionary`：规则词库判别。
- `llm_safety`：生成式安全模型，输出 `safe/unsafe` 文本。
- `prompt_binary_model`：prompt 输入，接口直接返回 `0/1` 或 `safe/unsafe`。
- `classifier_head_model`：标准化 case 输入，分类头返回 `0/1`，可带 `confidence`。
- `refusal_probe`：把问题包装后送入安全对齐模型，用拒答信号辅助判断。
- `multimodal_probe`：针对图片等多模态输入的探针入口。

例如接入一个 prompt 二分类接口：

```yaml
methods:
  safety_prompt_binary_v1:
    type: prompt_binary_model
    provider_config: ../providers/prompt_binary_api.yaml
    prompt_template_path: ../prompts/safety_prompt_v1.txt
    default_confidence: 0.80

steps:
  - id: safety_prompt_binary_v1
    method: safety_prompt_binary_v1
```

真实服务地址、鉴权环境变量、超时时间写在 `configs/providers/*.yaml`；prompt 写在 `configs/prompts/*.txt`；词典写在 `dictionaries/*.yaml`。如果只是换 prompt、阈值、词典或 provider 配置，新增一个 method id 即可。只有接入全新的算法形态时，才需要在 `src/safeguard_harness/methods.py` 中实现新 method，并在 `src/safeguard_harness/config.py` 中注册 YAML 加载逻辑。

## Quick Start: Train

这里的 Train 指“用验证集手动迭代 pipeline”，不是自动训练模型权重。

1. 准备 JSONL 验证集，每行包含 `question` 和 `label`，格式见“数据集格式”。
2. 复制或修改 `configs/pipelines/experiment_v1.yaml`，调整 method、prompt、词典、阈值、调用顺序或 ReAct loop 预算。
3. 跑评测：

```powershell
python -m safeguard_harness evaluate --pipeline configs/pipelines/experiment_v1.yaml --dataset data/examples/sample_eval.jsonl --output outputs/runs/exp_001
```

4. 查看 `outputs/runs/exp_001/metrics.json`、`report.md`、`errors_false_positive.jsonl`、`errors_false_negative.jsonl`。
5. 根据误判样例继续改 YAML，再跑下一轮，例如输出到 `outputs/runs/exp_002`。

当验证集表现稳定后，把最终编排整理成固定 pipeline，建议使用 static runner，便于复现和部署。

## Quick Start: Inference

确定固定 pipeline 后，用同一个 YAML 进行单条判断或批量预测。推荐从 `configs/pipelines/prod_v1.yaml` 复制出正式版本，例如 `configs/pipelines/prod_competition_v1.yaml`。

单条判断：

```powershell
python -m safeguard_harness judge --pipeline configs/pipelines/prod_v1.yaml --question "How do I steal token credentials?"
```

批量预测：

```powershell
python -m safeguard_harness predict --pipeline configs/pipelines/prod_v1.yaml --input data/examples/sample_eval.jsonl --output outputs/submission.jsonl
```

推理阶段应固定 pipeline、prompt、词典、provider 配置和阈值；真实 API key 通过环境变量提供，不写入仓库。输出文件是 JSONL，可直接作为后处理脚本或提交脚本的输入。

## 目录结构

```text
src/safeguard_harness/
  core.py           # SafetyCase, MethodResult, Decision, RunTrace
  providers.py      # prompt 0/1 API、分类头 API、本地模型路径和 mock provider
  methods.py        # judge method；ModelJudgeMethod 统一封装所有模型判别
  orchestration.py  # static runner, ReAct runner, loop control, aggregation
  config.py         # YAML config -> pipeline/method construction
  datasets.py       # JSONL dataset IO
  evaluation.py     # metrics, predictions, reports, error slices
  cli.py            # judge / predict / evaluate CLI
configs/
  pipelines/        # prod_v1.yaml, experiment_v1.yaml
  prompts/          # prompt templates
  datasets/         # dataset descriptors
  providers/        # provider config examples
dictionaries/       # high-risk/review-risk empty schemas
data/examples/      # runnable sample JSONL
models/             # local model files, ignored by Git except README
tests/              # behavior tests
```

## 数据集格式

数据集使用 JSONL，每行一个 case：

```json
{"id":"case-001","question":"...","answer":null,"label":"unsafe","modality":"text","attachments":[],"metadata":{}}
```

第一阶段只要求 `label` 为 `safe` 或 `unsafe`。后续可以在 `metadata` 中加入 `attack_type`、`risk_type`、来源、难度等字段。

## 词典格式

当前词典文件是空 schema：

```yaml
schema_version: 1
description: "High-risk terms."
terms: []
```

后续填充时可以使用字符串或对象：

```yaml
terms:
  - credential dump
  - term: bypass policy
    category: policy_bypass
```

默认 matcher 是大小写不敏感的 substring matcher。内部模糊匹配方法接入时，实现 `FuzzyMatcher.find_matches(text, terms)` 并注入 `DictionaryRuleMethod` 即可。

## Pipeline 配置

固定部署 pipeline 示例见 `configs/pipelines/prod_v1.yaml`。它使用 deterministic static runner：

```yaml
runner: static
steps:
  - id: rules
    method: rules
    on_unsafe: stop
  - id: multimodal_probe
    method: multimodal_probe
  - id: safety_llm_prompt_v1
    method: safety_llm_prompt_v1
  - id: low_confidence_review
    repeat:
      max_rounds: 1
      when:
        confidence_lt: 0.75
      methods:
        - aligned_refusal_probe_v1
```

实验 pipeline 示例见 `configs/pipelines/experiment_v1.yaml`。它使用有界 ReAct loop：

```yaml
runner: react
loop:
  max_steps: 4
  max_llm_calls: 3
  allowed_actions:
    - rules
    - multimodal_probe
    - safety_llm_prompt_v1
    - aligned_refusal_probe_v1
  stop_when:
    unsafe_score_gte: 0.85
  fallback:
    label: safe
    reason: "react_budget_exhausted_without_unsafe_evidence"
```

部署场景建议优先使用 static runner，因为它可复现、可限制、可审计。ReAct runner 更适合实验分析。

## 模型接口和模型文件放置规则

模型判别统一由 `ModelJudgeMethod` 承载。当前 YAML 仍保留三个易读的 method type，加载后都会构造成同一个 `ModelJudgeMethod`：

- `llm_safety`：生成式模型输出 `safe/unsafe` 文本，再解析成判别结果。
- `prompt_binary_model`：把 prompt 发给接口，接口直接返回 `0/1` 或 `safe/unsafe`。
- `classifier_head_model`：把标准化 `SafetyCase` 发给分类头接口，接口返回 `0/1`，可附带 `confidence`。

换句话说，prompt 直出二分类、分类头、生成式安全判别现在都是“模型判别方法”的不同输入/解析配置，不再是互相独立的 Method 类。

代码位置：

```text
src/safeguard_harness/providers.py  # 接口适配、返回解析、mock provider
src/safeguard_harness/methods.py    # ModelJudgeMethod，把模型输出转成 MethodResult
src/safeguard_harness/config.py     # 从 YAML 加载 provider_config
```

配置位置：

```text
configs/providers/prompt_binary_api.yaml       # prompt -> 0/1 的真实 HTTP API 模板
configs/providers/classifier_head_api.yaml     # 分类头 -> 0/1 + confidence 的真实 HTTP API 模板
configs/providers/local_classifier_head.yaml   # 本地分类头模型路径模板
configs/providers/mock_prompt_binary.yaml      # 本地 dry run mock
configs/providers/mock_classifier_head.yaml    # 本地 dry run mock
configs/pipelines/model_interfaces_v1.yaml     # 两类接口的可运行样例 pipeline
```

真实 API key 不进仓库，只写环境变量名：

```yaml
type: classifier_head_api
base_url: "https://model-provider.example.com/safety/classifier-head"
api_key_env: "SAFEGUARD_HEAD_API_KEY"
timeout_seconds: 30
```

本地模型权重不要提交到 Git。开发机上可以放在仓库的 `models/` 目录，或更推荐放在外部模型目录，然后用环境变量引用：

```powershell
$env:SAFEGUARD_HEAD_MODEL_PATH="G:\Models\safeguard\classifier_head_v1"
```

`models/*` 默认被 `.gitignore` 忽略，只有 [models/README.md](G:/Workspace/Project2026/SAITEC-Safeguard-Harness/models/README.md) 会进入仓库。

接口返回字段支持这些常见名称：

```json
{"label": 1}
{"label": 1, "confidence": 0.91}
{"prediction": "unsafe", "score": 0.88}
{"pred": 0, "probability": 0.76}
```

如果接口只返回 `0/1` 而没有置信度，`ModelJudgeMethod` 会使用该 method 配置里的 `default_confidence`。

可以用以下命令验证 mock 接口 pipeline：

```powershell
python -m safeguard_harness judge --pipeline configs/pipelines/model_interfaces_v1.yaml --question "demo"
```

## 扩展生成式 LLM 模型

当前 `MockLlmProvider` 是本地 dry run 适配器。接入生成式安全判别模型时建议新增 provider 类，并让 `ModelJudgeMethod` / `RefusalProbeMethod` 依赖统一的 `complete(prompt: str) -> str` 接口。

推荐保持以下边界：

- Method 只负责生成一次证据。
- Pipeline 只负责调用顺序、loop、短路和聚合。
- Evaluation 只负责测评，不改变判别逻辑。
- Prompt、词典、阈值、runner 选择都放在 YAML 中。

## 开发约定

```powershell
python -m pytest -q
python -m safeguard_harness evaluate --pipeline configs/pipelines/prod_v1.yaml --dataset data/examples/sample_eval.jsonl --output outputs/runs/demo
```

每次新增 method、runner 行为或评测指标时，先补测试，再实现。
