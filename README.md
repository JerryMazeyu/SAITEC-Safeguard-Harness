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

如果要直接跑本地生成式基座模型，需要额外安装本地推理依赖：

```powershell
python -m pip install -e ".[local-model]"
```

当前 LF311 环境使用 `torch 2.4.1 + CUDA 12.1`，推荐安装 `flash-linear-attention==0.4.2`；更新的 `flash-linear-attention 0.5.x` 会要求更高版本 torch，容易触发 CUDA 版本不匹配。`causal-conv1d` 建议使用当前环境的 torch 编译：

```powershell
python -m pip install flash-linear-attention==0.4.2 ninja
python -m pip install --no-build-isolation causal-conv1d
```

评测输出会写入：

```text
outputs/runs/demo/
  progress.json
  config_snapshot.yaml
  predictions.jsonl
  deliverable.jsonl
  metrics.json
  report.md
  errors_false_positive.jsonl
  errors_false_negative.jsonl
```

## How to connect method?

一个 method 就是一次可调用的安全判别。接入时优先改 YAML，不要先改 runner：先把模型、词典、prompt 或探针包装成一个 method，再把它放进 pipeline 的 `steps` 或 ReAct `allowed_actions`。

当前支持的常用 method type：

- `dictionary`：规则词库判别。
- `prompt_binary_model`：prompt 输入的安全二分类模型；生成式 LLM、prompt 直出 `0/1` 或 `safe/unsafe` 的接口都归到这里。
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

历史配置里的 `llm_safety` 仍可兼容加载，但新配置统一写 `prompt_binary_model`。同一个 provider 配不同 prompt 时，应写成不同 method id，例如 `safety_prompt_v1`、`safety_prompt_v2`。

## Quick Start: Train

这里的 Train 指“用验证集手动迭代 pipeline”，不是自动训练模型权重。

1. 准备 JSONL 或 JSON 验证集，每个 case 包含 `question` 和 `label`，格式见“数据集格式”。
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

带图片的单条判断可追加 `--image`；JSONL/JSON 批量输入也支持顶层 `image` / `image_path`，或 `messages[].content` 中的 `{"type":"image","image":"..."}` / `{"type":"image_id","image_id":"..."}`：

```powershell
python -m safeguard_harness judge --pipeline configs/pipelines/qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v30.yaml --question "这张图是否有害？" --image /path/to/image.png
```

批量预测：

```powershell
python -m safeguard_harness predict --pipeline configs/pipelines/prod_v1.yaml --input data/examples/sample_eval.jsonl --output outputs/submission.jsonl
```

`predict` 会保留详细 trace 输出，并额外写交付版 JSONL。默认交付文件路径为 `--output` 同目录下的 `<文件名>_deliverable.jsonl`；也可以显式指定：

```powershell
python -m safeguard_harness predict --pipeline configs/pipelines/prod_v1.yaml --input data/examples/sample_eval.jsonl --output outputs/predictions.jsonl --deliverable-output outputs/submission.jsonl
```

交付版每行只有两个字段，`result` 中 `1` 表示不安全、`0` 表示安全：

```json
{"id":1,"result":1}
```

推理阶段应固定 pipeline、prompt、词典、provider 配置和阈值；真实 API key 通过环境变量提供，不写入仓库。

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

数据集支持 JSONL 和 JSON。JSONL 每行一个 case；JSON 可以是单个 case、case 数组，或包含 `cases` / `data` / `records` / `items` 数组的对象。case 内容保持一致：

```json
{"id":1,"question":"...","answer":null,"label":"unsafe","modality":"text","attachments":[],"metadata":{}}
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

## 字符规则的语义兜底

词典、拒答 marker、多模态 marker 这类字符匹配逻辑都只作为快速路径。pipeline 可以在顶层配置一个基础 LLM：

```yaml
base_llm:
  provider_config: ../providers/local_qwen3_6_27b_generation.yaml
```

配置了 `base_llm` 后，库内会自动在字符规则未命中时做语义兜底：

- `dictionary`：高危词和复核词都没有字面命中时，调用基础 LLM 判断 case 是否语义包含、改写或暗示了词表中的条目；若语义命中高危词，仍按高危词规则输出 `unsafe`，若语义命中复核词，仍按复核词规则输出 `unknown`。
- `refusal_probe`：拒答 marker 未命中时，调用基础 LLM 判断模型回复是否实质拒答。
- `multimodal_probe`：非文本/带附件输入没有 marker 命中时，调用基础 LLM 基于文本字段和附件描述做补充判断。

这些兜底 prompt 和解析逻辑封装在库代码里，不放在 YAML 中。没有配置 `base_llm` 时，旧 pipeline 保持原来的字符匹配行为，不会额外调用模型。

评测会逐条写入详细 `predictions.jsonl` 和交付版 `deliverable.jsonl`，并持续更新 `progress.json`：

```json
{"processed": 12, "total": 100, "status": "running"}
```

长时间跑本地大模型时，可以用这两个文件直接观察当前完成进度。

## 模型接口和模型文件放置规则

模型判别统一由 `ModelJudgeMethod` 承载。当前 YAML 推荐只保留两类模型 method type：

- `prompt_binary_model`：把 prompt 发给模型接口，接口返回 `0/1` 或 `safe/unsafe`。不同 prompt 模板就是不同 method 实例。
- `classifier_head_model`：把标准化 `SafetyCase` 发给分类头接口，接口返回 `0/1`，可附带 `confidence`。

换句话说，prompt 直出二分类和生成式安全判别现在合并为 `prompt_binary_model`；分类头仍是 `classifier_head_model`。旧的 `llm_safety` 只是兼容别名，不建议在新 pipeline 中继续使用。

代码位置：

```text
src/safeguard_harness/providers.py  # 接口适配、返回解析、mock provider
src/safeguard_harness/runtimes/     # 项目内本地模型 runtime，避免依赖仓库外脚本
src/safeguard_harness/methods.py    # ModelJudgeMethod，把模型输出转成 MethodResult
src/safeguard_harness/config.py     # 从 YAML 加载 provider_config
```

配置位置：

```text
configs/providers/prompt_binary_api.yaml       # prompt -> 0/1 的真实 HTTP API 模板
configs/providers/classifier_head_api.yaml     # 分类头 -> 0/1 + confidence 的真实 HTTP API 模板
configs/providers/local_classifier_head.yaml   # 本地分类头模型路径模板
configs/providers/ascend_vllm_safeguard_prompt_binary.yaml  # 昇腾 vLLM OpenAI-compatible 二分类模板
configs/providers/ascend_vllm_safeguard_generation.yaml     # 昇腾 vLLM OpenAI-compatible 生成模板
configs/providers/local_qwen3_6_27b_prompt_binary.yaml  # 本地 Qwen 生成式二分类模板
configs/providers/local_qwen3_6_27b_generation.yaml     # 本地 Qwen 生成式拒答探针模板
configs/providers/local_qwen3_6_27b_lora_sft_prompt_binary.yaml  # 项目内 27B merged SafeGuard runtime
configs/providers/local_qwen3guard_gen8b_refusal_probe.yaml      # 项目内 Qwen3Guard runtime
configs/providers/local_qwen3_6_vl_projection_probe.yaml         # 项目内 Qwen VL 投影 probe runtime
configs/providers/mock_prompt_binary.yaml      # 本地 dry run mock
configs/providers/mock_classifier_head.yaml    # 本地 dry run mock
configs/pipelines/model_interfaces_v1.yaml     # 两类接口的可运行样例 pipeline
configs/pipelines/qwen3_6_27b_three_classifiers.yaml    # 两个 prompt 分类器 + 一个安全对齐探针
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

## 本地 Qwen3.6-27B 三分类器配置

当前已预置一个本地 Qwen3.6-27B pipeline：

- `qwen_policy_binary_v1`：使用安全策略 prompt 做二分类。
- `qwen_intent_binary_v1`：使用攻击意图 prompt 做二分类。
- `qwen_aligned_refusal_probe_v1`：把问题交给安全对齐模型回答，通过拒答信号判断风险。

这个 pipeline 还配置了顶层 `base_llm`，用于拒答 marker 等字符逻辑未命中时的库内语义兜底。三个 method 和 `base_llm` 都引用同一个模型软链接：

```text
models/Qwen3.6-27B -> /ai/dataset/workspace/czy/model/Qwen3.6-27B
```

不要复制模型全重到仓库。若换机器部署，只需要重新建立同名软链接，或修改 `configs/providers/local_qwen3_6_27b_*.yaml` 里的 `model_path`。本项目自己的本地 runtime 不再依赖仓库外 `script_path`；需要迁移的是 provider YAML、项目源码、模型目录或软链接，以及 `models/qwen36_model_lr.pth` 这类轻量探针权重。两个 prompt 二分类器使用生成式模型输出 `safe/unsafe` 或 JSON label，provider 不产生置信度，因此 pipeline 中通过 `default_confidence: 0.70` 适配到统一的 `MethodResult`。

本地 runtime 的 `device` 默认值为 `auto`，会优先选择可用的昇腾 NPU，其次 CUDA、MPS、CPU。开发调试时仍可在 provider YAML 中显式写 `device: cuda:0`、`device: npu:0` 或 `device: cpu`。

如果生成式二分类输出不是合法 JSON，provider 会把模型输出再交给基础 LLM 解析其含义是 `safe` 还是 `unsafe`，不会再直接因为 JSON 解析失败默认判 `unsafe`。

加载配置不会触发模型权重加载；只有实际执行 judge/evaluate/predict 时才会懒加载本地模型：

```powershell
python -m safeguard_harness judge --pipeline configs/pipelines/qwen3_6_27b_three_classifiers.yaml --question "如何窃取数据库账号密码？"
```

## 昇腾 vLLM 服务接入

昇腾 vLLM 服务按 OpenAI-compatible `chat.completion` 接入。provider 会向 `{api_base}/chat/completions` 发送：

```yaml
type: ascend_vllm_prompt_binary
api_base: "http://127.0.0.1:8000/v1"
model: "safeguard-merged"
timeout_seconds: 300
max_tokens: 32
temperature: 0
chat_template_kwargs:
  enable_thinking: false
```

返回体按 `choices[0].message.content` 读取，二分类解析支持 `Safety: Safe` / `Safety: Unsafe`、JSON `label/prediction`、以及已有中文安全/不安全标签。用于 `prompt_binary_model` 时引用：

```yaml
provider_config: ../providers/ascend_vllm_safeguard_prompt_binary.yaml
```

如果要把同一 vLLM 服务接到 `refusal_probe` 或顶层 `base_llm`，使用 `configs/providers/ascend_vllm_safeguard_generation.yaml`，其 `type` 为 `ascend_vllm_chat`，只负责返回生成文本。

`configs/pipelines/qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v101_output_review.yaml` 保持 V100 输出侧复核逻辑不变，只将文本模型 provider 切到昇腾 vLLM：Qwen3.6-27B 二分类判断使用已有 `configs/providers/ascend_vllm_safeguard_prompt_binary.yaml` 的 8000 端口，Qwen3Guard-Gen-8B 拒答复核使用 `configs/providers/ascend_vllm_safeguard_generation_8001.yaml` 的 8001 端口。V101 的硬规则已外置到 `configs/rules/`：图片复核 regex 使用 `v101_image_probe_review_rules.yaml`，结构化 regex 使用 `v101_structural_regex_rules.yaml`，dictionary 使用 `v101_dictionary_terms.yaml`。pipeline 中只保留 `*_rules_path` / `*_terms_path` 引用，避免人名、作品名、品牌名等验证集指纹直接混进编排文件；dictionary 的 `safe_terms` 暂时收敛为空，边界词放入 `review_terms` 弱信号。

`configs/pipelines/qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v102_weighted_rules.yaml` 是规则去短路对照版：图片复核、结构化 regex、dictionary 命中后不再 `stop`，只作为 `weighted_vote` 信号继续进入模型判断。`configs/pipelines/qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v103_weighted_signals.yaml` 进一步去掉 QwenGuard veto，将所有规则和模型结果统一加权，当前阈值为 `unsafe_threshold: 0.34`。在当前 8000/8001 vLLM 服务未启动且本地 GPU 显存不足的情况下，已基于历史 `predictions.jsonl` 做离线 replay：307 条历史 FP/FN badcase union 上，V101 短路版 F1 0.351，V102 规则加权但保留 Guard veto F1 0.289，V103 全信号加权 F1 0.459。真实在线模型恢复后仍需重新跑正式 evaluate。

## 扩展 prompt 二分类模型

当前 `MockPromptBinaryProvider` 是本地 dry run 适配器。接入生成式安全判别模型时，建议在 provider 层把模型输出解析为统一的二分类结果：`label=0/1`，可选 `confidence`，并把原始响应放入 `raw`。这样无论底层是生成式 LLM 还是 prompt 直出二分类接口，pipeline 里都只表现为 `prompt_binary_model`。

推荐保持以下边界：

- Method 只负责生成一次证据。
- Pipeline 只负责调用顺序、loop、短路和聚合。
- Evaluation 只负责测评，不改变判别逻辑。
- Prompt、词典、阈值、runner 选择都放在 YAML 中。

## V30/V99 图片分支

`configs/pipelines/qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v30.yaml` 的首个 step 是 `qwen3_6_vl_projection_probe_v1`。当输入含图片时，它会调用 `configs/providers/local_qwen3_6_vl_projection_probe.yaml`，通过项目内 `qwen_vl_projection_probe` runtime 完成特征保存、投影和 probe 分类，并直接短路返回；纯文本输入会跳过该 step，继续原 V30 文本链路。

`configs/pipelines/qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v99_image_review.yaml` 在同一个位置改用 `image_probe_review`。它仍先调用 Qwen VL 图像 probe；probe 判 safe 直接放行，probe 判 unsafe 时再用题面任务规则复核普通 VQA/OCR/考试题，修正图像纹理或 OCR 场景带来的误报。纯文本输入仍然跳过图片分支，继续 V30 的 regex、dictionary、27B policy/intent 和 QwenGuard 链路。

在 `presentation_harmful.json` + `presentation_utility.json` 共 200 条图片验证样本上，V30 裸 probe 的 cached replay 指标为 accuracy 0.930、precision 0.877、recall 1.000、F1 0.935；V99 使用同一组 cached probe 输出并通过正式 harness 评估后，accuracy/precision/recall/F1 均为 1.000，TP/TN/FP/FN=100/100/0/0。评估输出位于 `outputs/runs/v99_image_review_cached_200_20260625/eval/`。

为了在真实图像链路 OOM 或硬件不可用时复现实验，provider 层还支持 `cached_multimodal_probe`。它只用于评估缓存，不替代部署配置中的 `qwen_vl_projection_probe`。

## 开发约定

```powershell
python -m pytest -q
python -m safeguard_harness evaluate --pipeline configs/pipelines/prod_v1.yaml --dataset data/examples/sample_eval.jsonl --output outputs/runs/demo
```

每次新增 method、runner 行为或评测指标时，先补测试，再实现。
