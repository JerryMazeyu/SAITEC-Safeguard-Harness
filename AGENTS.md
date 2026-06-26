# 安全Harness项目

## 项目背景
我们正在参加一个数据安全大模型竞赛，任务是问答推或者是用户的提问是否安全。在第一阶段仅仅需要做二分类的判别，在后续的阶段，可能要涉及到针对具体的攻击方式以及更细力度的安全分析的判别。因此，我们希望做一套harness系统针对平台方给出的案例进行判断。
目前已经有了以下的判别方法，但是这些内容还没有进行组合，我们的Harneess框架希望能够将这些判别方法进行组合，从而实验到底哪个编排是更好的：
- Method 1：第一层是基于规则的判别，我们会从大量的数据集中构建两个危险词库，第一个级别的危险词库，只要一旦命中，那么则以非常高的概率判断为危险，第二个级别的危险词库，如果一旦命中，则需要大模型进行详细的审查。注意到这里应该是模糊词的匹配，我们自己有一个内部的模糊匹配方法，可以预留出一个方法类或者接口。
- Method 2：第二层是基于大模型的判别，我们有一个专门针对安全检测的基座模型，那么这个模型可以针对目前的问答对或者是问题进行一个详细的判断，但是提示词模板可以自己定义。可以将这个模型作为一个接口来使用。
- Method 3：第三层则是专门针对问题的。我们会有一个已经实现了安全对齐的大模型，那么一旦检测到可能或者是有风险的提问方式，我们会故意的对它进行一些提示词的封装，并把它送进安全对齐大模型中。如果一旦安全对齐大模型拒绝回答，那么我们就以很高的概率判断这个问题很有可能是不安全的。
- Method 4：第四层是针对多模态的安全性，我们自研了一种针对模型投影和探针的方法，可以针对图片等多模态的输入进行判断，如果一旦碰到了跟图片相关的安全判断问题，我们则需要走这样的链路。
我们目前这些方法都暂时通过一个基类封装成的可调用的方法，最终我们是要将这四种方法进行不断的组合以及配置。注意到，例如大模型Method 2，提示词模板不一样，那么每一次其实也可以看作是一个不同的判断方法。
在整套系统中，我还需要专门的构建测评的数据集来实现编排流的快速迭代。

## 任务需求
请你根据这样的项目背景，使用成熟的Harness框架或者原生手搓，来实现整套的harness系统，这个系统中应该具备常见的react Agent工作模式。
最后是有两个阶段，一个阶段以及训练我可以通过验证测评数据集的效果来进行编排，但是这种训练并不是自动化的，而是需要手动的一次一次去进行调整。
第二个则是构建完固定后pipeline之后，那么可以输出一个可部署的完整的判断，套一个脚本就可以最后输出结果。

## 技术栈
需要考虑到整体模型的适配，目前我们主要采用Python语言进行开发。可以将编排尽可能的写成通俗易懂的配置文件的形式。

## 规范
- 每当实现了一个完整的功能之后，都需要进行GitHub的PR。
- 每次更新维护AGENTS.md的“最新进度”。

## 最新进度
- 2026.06.12 12:01 完成了AGNETS.md的撰写。
- 2026.06.12 12:45 完成安全 Harness 第一版框架实现：Python 包、四类 Method stub、static/ReAct runner、有界 loop、JSONL 评测、CLI、空词典 schema、样例配置与 README。
- 2026.06.15 完成两类集成模型接口落点：prompt 二分类 API、分类头二分类 API、本地模型目录忽略规则、provider 配置模板、mock 接口样例 pipeline 与 README 说明。
- 2026.06.15 重构模型判别方法：`llm_safety`、`prompt_binary_model`、`classifier_head_model` 统一映射到 `ModelJudgeMethod`，并支持纯 0/1 接口使用 `default_confidence`。
- 2026.06.15 补充 README 使用入口说明：method 接入方式、验证集手动迭代流程、固定 pipeline 推理流程。
- 2026.06.15 合并 `llm_safety` 与 `prompt_binary_model` 语义：新 pipeline 统一使用 `prompt_binary_model`，不同 prompt 模板实例化为不同 method id，旧 `llm_safety` 仅保留兼容加载。
- 2026.06.17 将旧验证集 `safeguard_strategy2_plus_manage_r1_safety_only_shared_validation_zh80_llamafactory.json` 转换为 sample_eval 消息格式 JSONL，并保留 `safe` / `unsafe` 标签。
- 2026.06.17 适配评估数据加载逻辑，支持 `messages` 格式 JSONL 自动转换为 `SafetyCase`，并验证 `validate_v1.jsonl` 可完整跑通评估。
- 2026.06.17 11:17 接入本地 Qwen3.6-27B 软链接模型：新增生成式本地 provider、两个不同 prompt 的二分类器、一个安全对齐拒答探针 pipeline，并验证相关测试与轻量 CLI 入口通过。
- 2026.06.17 11:36 完成字符匹配逻辑的库内语义兜底：新增顶层 `base_llm` 配置入口，词典/拒答/多模态 marker 未命中时自动调用内部 LLM judge，Qwen pipeline 已接入该基础 LLM。
- 2026.06.17 16:40 完成 Qwen3.5-4B 软链接 smoke evaluate：新增 4B 本地 provider/pipeline、local provider `max_memory`/`offload_folder` 支持、评估异常进度失败态落盘；10 条验证集跑通，accuracy 0.40、recall 1.00，测试集 37 passed。
- 2026.06.17 17:02 修复二分类解析与聚合偏置：解析器不再全文扫描 safe/unsafe 关键词，Qwen provider 默认关闭 thinking，`weighted_vote` 改为真实加权投票；Qwen3.5-4B 10 条 smoke accuracy 提升至 0.80，测试集 40 passed。
- 2026.06.17 17:14 完成修复后 Qwen3.5-4B 20 条验证集评估：accuracy 0.90、precision 1.00、recall 0.50、F1 0.667，false positive 为 0，主要漏检为更广义欺骗/骚扰类不当请求。
- 2026.06.17 11:42 修正词典语义兜底边界：`dictionary` 未字面命中时基础 LLM 只判断是否语义命中高危/复核词表，不直接裁决整体 `safe` / `unsafe`。
- 2026.06.17 15:46 改进 Qwen 验证链路：评测逐条写 `predictions.jsonl` 与 `progress.json`，二分类 JSON 解析失败改由 LLM 判读输出含义，拒答 marker 增加否定语境过滤，并在 LF311 环境安装 `flash-linear-attention==0.4.2` 与 `causal-conv1d==1.6.2.post1`。
- 2026.06.17 21:43 优化 Qwen3.5-4B 二分类 prompt 覆盖多子任务、欺骗伪造、骚扰、网络/数据安全和虚构安全边界，并完成 200 条验证集评估：accuracy 0.88、precision 0.792、recall 0.95、F1 0.864；误差主要来自拒答探针对安全样本/安全拒答回答过敏。
- 2026.06.17 21:53 基于 200 条评估 bad cases 为三个 Qwen prompt 增加简短纠偏规则：二分类器补充输出侧 Answer 优先、敏感主题不过度拦截、版权/成人规避/身份群体贬损漏检边界；拒答探针补充 Answer-aware 审查和普通澄清不算拒答，相关配置测试 25 passed。
- 2026.06.17 23:01 新增全局 Codex skill `optim-safeguard-pipeline`：固化 200 条随机验证、bad cases 分析、参数优先优化、候选 pipeline 对比保留/回滚和持续迭代流程，并提供 JSONL 抽样与 metrics 对比脚本。
- 2026.06.17 23:09 将 `optim-safeguard-pipeline` skill 的验证样本量改为可配置 `sample_size` 参数，默认 200，并同步更新默认提示与抽样命令说明。
- 2026.06.17 23:12 按命名要求将 pipeline 优化 skill 目录与 frontmatter 统一重命名为 `optim-safeguard-pipeline`，并同步更新默认提示和内部脚本路径。
- 2026.06.17 23:22 新增 Qwen3.5-4B `policy+intent` balanced pipeline，并基于已完成的 161 条验证结果分析：accuracy 0.925、precision 0.953、recall 0.871、F1 0.910，主要取舍为显著降低 FP 但冲突样本 FN 增多；同时修复截断 JSON 中明确 `"label"` 无法解析的问题，相关测试 26 passed。
- 2026.06.17 23:28 新增 Qwen3.5-4B 冲突复核 pipeline：先跑 `policy+intent`，仅当两路加权投票 confidence < 0.70 时调用安全对齐拒答探针复核；新增冲突触发/一致跳过测试，相关测试 28 passed。
- 2026.06.18 01:16 完成 Qwen3.5-4B 冲突复核 pipeline 100 条固定随机样本优化：新增 transformers/FLA 兼容加载开关、v2 二分类 prompt、敲诈高危词典前置 stop，并将 `qwen3_5_4b_conflict_review.yaml` 提升到阈值 0.70 的 accepted v4；最终 accuracy 0.98、precision 0.963、recall 1.00、F1 0.981，测试集 28 passed。
- 2026.06.18 10:05 中止 Qwen3.5-4B 冲突复核 pipeline 500 条评估于 423/500：partial accuracy 0.887、precision 0.871、recall 0.888、F1 0.879；基于 FP/FN 聚类新增 v4 policy/intent prompt 与 `qwen3_5_4b_conflict_review_candidate_v5.yaml`，重点补跨语言隐私、骚扰/侮辱、规避执法、非同意恶作剧、授权网络边界和 Answer-aware 安全化误判，配置加载通过，相关测试 22 passed。
- 2026.06.18 10:49 完成 `qwen3_5_4b_conflict_review_candidate_v5.yaml` 新随机 200 条验证（seed 20260618）：accuracy 0.900、precision 0.930、recall 0.877、F1 0.903，TP/TN/FP/FN 为 93/87/7/13；v5 明显压低 FP，但 FN 仍集中在虚构包装危险请求、钓鱼模拟、公共财产恶作剧、争议/误导输出和部分跨语言敏感内容。
- 2026.06.19 00:29 基于 `optim-safeguard-pipeline` 从 `qwen3_5_4b_conflict_review_candidate_v5.yaml` 继续优化至 v23：新增 `safe_terms` 字典兜底并逐轮基于完整 200 条失败集修正，所有 200 条失败 run 均完整跑完后统一分析；v20/v21/v22/v23 固定 100 条准入均为 F1 1.000，v23 在随机 200 条 seed2026061811 上 accuracy 0.975、precision 0.970、recall 0.980、F1 0.975，仍未达到连续 3 次 200 条 F1>0.98，按收束要求停止继续迭代；当前 200 条随机验证综合最优三版为 v16、v23、v22，相关测试 24 passed。
- 2026.06.19 14:32 完成 v16/v23 三个随机 200 条样本对比（seeds 2026061901/2026061902/2026061903）：v23 三组 F1 分别为 0.9588/0.9784/0.9617，平均 F1 0.9663；v16 三组 F1 分别为 0.9137/0.9286/0.9231，平均 F1 0.9218；结论为 v23 明显优于 v16，评估输出位于 `outputs/runs/compare_v16_v23_random200_20260619/`。
- 2026.06.19 15:23 基于 v23 整理 `qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v24.yaml`：保持词典/两路 policy+intent/低置信冲突复核编排不变，将二分类基座切到 27B LoRA SFT merged 外部脚本 provider，将拒答探针切到 Qwen3Guard-Gen-8B 并支持 guard safe/unsafe verdict 优先解析；配置加载、subprocess provider smoke、qwenguard parser smoke、词典短路 smoke 与 `compileall` 通过，真实模型 smoke 因 `/ai/dataset/workspace/czy/...` 脚本和模型目录当前 `czy` 私有权限而阻塞。
- 2026.06.19 15:33 复试 v24 真实模型链路：主脚本与模型目录权限已开放，并按实际 CLI 修正 27B provider 为 `--prompt` + `prediction_label` 解析、QwenGuard provider 为 `--task prompt/response --user/--assistant` 自适应；配置加载与 `compileall` 通过。真实 27B smoke 仍因 `SafeGuard/triton_compat/sitecustomize.py` 无组读/执行失败，QwenGuard 在 `/opt/conda` 下因 transformers 包目录 `root:root 700` 不可读失败，原脚本 shebang 指向的 LF311 环境目录仍为 `czy:czy 2700` 不可进入。
- 2026.06.19 15:55 再次复试 v24：权限已满足并将两个外部脚本 provider 切到 LF311 Python；27B LoRA SFT 单脚本 smoke 跑通，QwenGuard-Gen-8B 单脚本 smoke 跑通。修正 27B provider 改读 `prediction_text` 并补充 `Safety:`/中文安全句解析，修正 QwenGuard subprocess 外层 JSON 解析；v24 harness safe smoke 跑通，两路 27B 均判 `safe` 且未触发冲突复核；单独调用 v24 QwenGuard probe 跑通，返回 `safe`、confidence 0.75；`compileall` 通过，GPU 无残留进程。
- 2026.06.19 16:19 在 LF311 用户 site 安装 `pytest` 并验证全量测试 `49 passed`；为 v24 增加外部脚本进程内加载与模型缓存 provider，避免 100 条评估时反复冷启动 27B/QwenGuard，cached v24 smoke 通过。固定随机 seed 20260619 抽样 100 条（safe 52 / unsafe 48）完成评估：accuracy 0.940、precision 0.957、recall 0.917、F1 0.936，TP/TN/FP/FN=44/50/2/4；QwenGuard 触发 2 次，词典短路 4 次。误差主要为：FP 对“激进左翼思想传播”和“非法物质成分科普边界”偏严；FN 漏检民族/狗肉侮辱、房屋陷阱、虚构逃避当局、制作/吸食大麻教学。
- 2026.06.19 15:28 完成 `/ai/dataset/workspace/czy/...` 外部脚本与模型目录权限打通：`SafeGuard` 脚本目录及 `.py` 文件设为 `750`，两个模型目录递归开放组读/执行，并统一属组为 `safeguard_harness`；已确认 `jerry` 属于该组。
- 2026.06.19 15:38 补齐 v24 真实模型链路依赖权限：`SafeGuard/triton_compat` 与 `/ai/dataset/workspace/czy/conda/LF311` 已统一属组为 `safeguard_harness` 并开放组读/执行，`/opt/conda` 下 transformers 包与 dist-info 已开放全局读/执行；已用 `jerry` 身份验证关键路径可读/可穿透。
- 2026.06.19 16:24 完成将 `optim-safeguard-pipeline` skill 安装至 Jerry 用户 Codex：复制到 `/home/jerry/.codex/skills/optim-safeguard-pipeline`，修正 helper 脚本路径为 Jerry Codex home，并用 `jerry` 身份验证说明可读、脚本可执行。
- 2026.06.22 14:37 完成 V30 pipeline `qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v30.yaml` 全量验证集 2048 条评估：初始评估于 429 条后中断，续跑追加至同一输出目录 `outputs/runs/v30_full_validate_2048_20260622/` 并完整收敛；最终 accuracy 0.9761、precision 0.9710、recall 0.9814、F1 0.9762，TP/TN/FP/FN=1005/994/30/19。
- 2026.06.25 接入 V30 图片输入独立分支：输入层支持 `image`/`image_path` 和 messages 图片内容归一化，新增 `one_case_multimodal_probe` provider 复用 `/ai/dataset/workspace/wwy/比赛/one_case.py` 完成特征保存、投影与 probe 分类；V30 首步在检测到图片时直接短路使用该分支，纯文本继续原 V30 链路。
- 2026.06.25 将 `/ai/dataset/workspace/wwy/比赛/Data/presentation_harmful.json` 与 `presentation_utility.json` 共 200 条图片样本追加到 `data/examples/validate_v1.jsonl`：新增 id 2048-2247，100 unsafe / 100 safe，来源覆盖 FigStep、MM-SafetyBench、VLGuard、MM-Vet、ScienceQA、COCO；messages 使用 `type: image_id` 格式并保留绝对图片路径，验证集总量增至 2248 条。
- 2026.06.25 验证 V30 图片链路：为 `one_case_multimodal_probe` 打开 `disable_torch_compile` 与 `patch_torch_distributed_tensor` 兼容项，真实 harness smoke 已越过导入并开始加载权重，但因当前 `cuda:1` 仅剩约 33MiB 被 OOM 阻塞；基于已有完整投影缓存复现 probe 分类，200 条图片样本 accuracy 0.930、precision 0.877、recall 1.000、F1 0.935，TP/TN/FP/FN=100/86/14/0，输出位于 `outputs/runs/v30_image_probe_cached_200_20260625/`。
- 2026.06.25 构建 V99 图片复核 pipeline `qwen3_6_27b_lora_qwen3guard_conflict_review_candidate_v99_image_review.yaml`：新增 `image_probe_review` method，在图片 probe 判 unsafe 后用题面任务规则复核普通 VQA/OCR/考试题，纯文本继续原 V30 链路；新增 `cached_multimodal_probe` 用于缓存评估。基于同一组 cached probe 输出跑 200 条图片验证，accuracy/precision/recall/F1 均为 1.000，TP/TN/FP/FN=100/100/0/0，输出位于 `outputs/runs/v99_image_review_cached_200_20260625/eval/`，全量测试 `67 passed` 且 `compileall` 通过。
- 2026.06.25 18:12 接入昇腾 vLLM OpenAI-compatible 推理 provider：新增 `ascend_vllm_prompt_binary` 与 `ascend_vllm_chat`，按 `/v1/chat/completions` 发送 `chat_template_kwargs.enable_thinking=false` 并从 `choices[0].message.content` 解析 `Safety: Safe/Unsafe`；补充 provider 配置模板、README 说明和单元测试。
- 2026.06.25 18:42 扩展批量输入/输出交付格式：`predict`/`evaluate` 支持 JSON 与 JSONL case 输入，保留详细 trace 输出的同时额外生成只含 `id` 与 `result` 的交付版 JSONL，其中 `1` 表示 unsafe、`0` 表示 safe，并补充 README 与测试。
- 2026.06.25 18:52 调整交付版输出 ID 语义：批量读取时保留输入原始 `id` 到 metadata，`deliverable.jsonl` 优先输出原始 ID 类型，因此数字 ID 会输出为 `{"id":1,"result":...}` 而非字符串。
- 2026.06.25 18:54 调整多轮输出侧判别逻辑：`is_mt`/`MT` 为真且 `type` 为输出侧时，将 messages 拆成多个 Q&A 子 case 逐个沿用原输出侧 pipeline 判别，并按并集聚合为原 id 的最终结果；输入侧多轮保持单 case 处理，相关测试与全量测试通过。
- 2026.06.26 新增 V101 输出侧复核 pipeline：整体逻辑与 V100 保持一致，Qwen3.6-27B 二分类 provider 切到昇腾 vLLM 8000 端口，Qwen3Guard-Gen-8B 拒答复核 provider 切到昇腾 vLLM 8001 端口，并补充配置加载测试与 README 说明。
- 2026.06.26 09:37 将图片 probe 线性权重 `qwen36_model_lr.pth` 从 `/ai/dataset/workspace/wwy/比赛/` 复制到仓库 `models/` 目录，SHA256 校验一致。
- 2026.06.26 10:24 剥离本地 provider 配置中的外部 `script_path` 依赖：新增项目内 `runtimes/` 承接 27B merged SafeGuard、Qwen3Guard-Gen-8B 与 Qwen VL 投影 probe 核心逻辑；本地 runtime provider 默认 `device: auto`，自动优先选择昇腾 NPU，其次 CUDA/MPS/CPU，图片 probe 权重改用仓库内 `models/qwen36_model_lr.pth`，并补充配置无脚本路径与自动设备测试。
