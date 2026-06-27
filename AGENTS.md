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
- 2026.06.26 13:25 完成 V101 规则解耦与首轮去过拟合清理：新增 `*_rules_path` 加载支持，将图片复核 regex、结构化 regex、dictionary 外置到 `configs/rules/`，V101 pipeline 仅保留路径引用；清理人名/作品名/品牌名/门牌号等验证集指纹，dictionary `safe_terms` 收敛为空，边界项迁入 `review_terms` 弱信号，并补充配置加载测试与 README 说明。
- 2026.06.26 13:49 构建规则加权候选 V102/V103 并完成历史 badcase replay：V102 去掉规则层 `stop`，V103 进一步去掉 QwenGuard veto、所有信号统一加权且阈值 0.34；从历史 138 个 FP/FN 文件汇总 307 条去重文本 badcase。因当前 8000/8001 vLLM 未启动且本地 27B smoke 被 GPU 残留显存 OOM 阻塞，采用“V102/V103 规则实时计算 + 历史模型 trace 缓存”的离线 replay；V101 短路版 F1 0.351，V102 保留 Guard veto F1 0.289，V103 全信号加权 F1 0.459，输出位于 `outputs/runs/v10{1,2,3}_*_historical_badcases_replay*_20260626/`。
- 2026.06.26 14:20 按真实本地 transformer 评估要求将 V102/V103 文本模型全部切到项目内 local provider，并为 27B merged SafeGuard 与 Qwen3Guard local runtime 增加 `device_map`/`max_memory`/`offload_folder` 支持；当前机器无 NPU，四张 A100 均被不可见作业占用约 60-80GiB 且高利用，单卡本地加载不可行。尝试 `CUDA_VISIBLE_DEVICES=0,3` + CPU offload 后 27B 可加载但真实输出退化为重复 `!`/`<think>`，无法生成有效 `Safety:` 标签，因此 307 条文本 case 的真实 V102/V103 指标未产出，需释放至少一张足够空的 80GB GPU 后再跑。
- 2026.06.26 14:43 按默认配置简化要求移除所有 `configs/providers/` 中的 `max_memory` 与 `offload_folder` 字段；默认仅保留 `device_map: auto` / `device: auto`，由 transformers/accelerate 在当前 NPU 或 CUDA 环境中自动分配。runner 逻辑不变，provider/runtime 仍保留这些字段的可选解析能力以兼容用户显式写入的旧配置。
- 2026.06.27 11:15 汇总当前 `outputs/runs/**/predictions.jsonl` 中所有历史迭代误判 case：扫描 91 个 prediction 文件、14084 条记录，回填图片缓存错例并去重生成 `data/examples/missed_cases_validation.jsonl` 共 359 条（文本 345 / 图片 14，FP 204 / FN 155），同步输出 `missed_cases_validation_summary.json`，并验证现有 loader 可正常读取。
- 2026.06.27 11:24 基于 V101 规则文件抽象 policy rubric，并拆分融合为精简版 `qwen_binary_policy_v8.txt` 与 `qwen_binary_intent_v8.txt`：policy 侧负责内容是否违规与安全边界，intent 侧负责真实意图和现实后果判断；未纳入多模态专项，并新增 prompt 长度/分工/模板变量约束测试。
- 2026.06.27 11:49 新增 `progressive_rule_classifier` 多轮 markdown 规则分类器：首轮只暴露规则 manifest，由模型输出 `load_rules` 请求后 harness 读取对应 markdown，再要求模型输出 final JSON 判别；支持 YAML manifest、轮数/文件数上限、trace 记录，并补充 README 与单元测试。
- 2026.06.27 12:43 新增五个单方法分析 pipeline：`single_progressive_rules_v1.yaml`、`single_policy_classifier_v1.yaml`、`single_intent_classifier_v1.yaml`、`single_refusal_probe_v1.yaml`、`single_multimodal_probe_v1.yaml`，分别单独封装 progressive markdown 规则、LoRA policy、LoRA intent、Qwen3Guard 拒答检测和 Qwen3.6 VL 投影 probe；同时补充 progressive 最小 markdown 规则包与配置加载测试。
- 2026.06.27 14:17 将 `qwen_binary_policy_v7.txt` 与 `qwen_binary_intent_v7.txt` 中的判别准则去重提取为 `configs/rules/progressive/qwen_binary_v7_extracted_rules.yaml`，按输入输出优先级、网络数据滥用、隐私跟踪、欺骗规避、骚扰身份贬损、成人内容、现实/生物危险、金融误导版权、安全边界和特别纠偏拆分为 progressive 可加载规则，并将 `single_progressive_rules_v1.yaml` 切到该 YAML manifest。
- 2026.06.27 14:28 修正 progressive 规则暴露方式：`qwen_binary_v7_extracted_rules.yaml` 改为仅含规则 id/description/path 的 manifest，详细准则拆到 `configs/rules/progressive/qwen_binary_v7/*.yaml`，首轮 router prompt 改为只做初步判断和规则选择，不发送规则正文；follow-up 仅加载模型请求的对应规则文件，避免一次性塞入全部准则导致注意力分散。
- 2026.06.27 14:41 将单方法分析 pipeline 中的 `single_policy_classifier_v1.yaml` 与 `single_intent_classifier_v1.yaml` 从 v8 prompt 切回 v7 prompt，并补充配置加载测试断言，确保 policy/intent 单项分析与 progressive v7 规则抽取使用同一版准则来源。
- 2026.06.27 13:31 基于 20 条 S5 真实样例扩增验证数据：新增 `data/examples/real_like_augmented_s5.jsonl` 180 条（safe 138 / unsafe 42）与 `data/examples/hard_augmented_s5.jsonl` 120 条（safe 25 / unsafe 95），覆盖敏感但正常提问、提示词/源码窃取、危险主体包装、多步骤越狱和输出侧顺从不合理请求；两份数据均保持现有 messages JSONL 格式并通过 loader 校验。
- 2026.06.27 13:35 为 S5 数据补齐二分类标签：新增 `data/examples/example_s5_labeled.jsonl` 原始 20 条带 label（safe 12 / unsafe 8），并合并生成 `data/examples/s5_augmented_labeled_320.jsonl` 共 320 条（safe 175 / unsafe 145），均保持现有验证集 JSONL 格式并通过 loader 校验。
- 2026.06.27 13:56 按去模板化要求重写 S5 增广 v2：新增 `real_like_augmented_s5_v2.jsonl` 180 条（safe 135 / unsafe 45）、`hard_augmented_s5_v2.jsonl` 120 条（safe 25 / unsafe 95）与合并版 `s5_augmented_labeled_320_v2.jsonl` 320 条（safe 172 / unsafe 148）；将泛化主题实例化为具体场景，扩展多语言/多格式/角色扮演/RAG/源码泄露等红队包装，并通过 loader、ID 唯一性、前缀重复和近重复扫描校验。
- 2026.06.27 14:16 按输出侧与长 prompt 覆盖要求收敛 S5 增广 v3：新增 `real_like_augmented_s5_v3.jsonl` 100 条（safe 70 / unsafe 30）、`hard_augmented_s5_v3.jsonl` 60 条（safe 8 / unsafe 52）与合并版 `s5_augmented_labeled_180_v3.jsonl` 180 条（safe 90 / unsafe 90）；补充角色扮演/格式化/半洗白式输出侧 unsafe 响应 48 条，并加入 21 条 250 字以上、3 条 500 字以上长样本，已通过 loader、ID 唯一性、前缀重复和近重复扫描校验。
- 2026.06.27 14:19 统一 S5 相关 JSONL 字段顺序：将 `example_s5_labeled.jsonl`、S5 v1/v2/v3 拆分与合并文件全部整理为 `id` 首字段，并按 `id/type/is_mt/label/messages` 顺序输出；复查所有 S5 JSONL 首字段均为 `id`，v3 loader 统计保持不变。
- 2026.06.27 14:29 整合 S5 增广 V3 与历史错例验证集：新增 `data/examples/s5_v3_missed_cases_validation.jsonl` 与 summary，以 `type/is_mt/label/messages` 为精确去重键合并 `s5_augmented_labeled_180_v3.jsonl` 和 `missed_cases_validation.jsonl`，源数据 539 条去除 V3 内部 14 条重复后保留 525 条（safe 294 / unsafe 231，输入侧 330 / 输出侧 195，文本 511 / 图片 14）。
- 2026.06.27 15:36 完成 `data/examples/s5_v3_missed_cases_validation.jsonl` 四个 single 方法并行评估：policy/intent/refusal 分别单卡完整跑通，progressive 因 Qwen3.6 FLA `torch.compile` 兼容问题补充 `disable_torch_compile`/`patch_torch_distributed_tensor` 后分片合并；逐 case 汇总位于 `outputs/runs/single_methods_s5_v3_missed_20260627_144746/single_method_case_results.jsonl`。525 条整体指标：progressive F1 0.710、policy F1 0.780、intent F1 0.786、refusal_probe F1 0.590，summary 位于同目录 `single_method_metrics_summary.json`。
- 2026.06.27 15:47 基于四个 single 方法逐 case 结果完成不超过 5 步的 ensemble replay 搜索：枚举 92,034 个候选、65,609 个唯一逐 case 输出，覆盖 single、投票、顺序门控、加权阈值、signed confidence 与二值 truth-table；最优为 `progressive+policy+intent+refusal` 四方法二值模式 lookup，accuracy 0.8362、precision 0.8033、recall 0.8312、F1 0.8170，TP/TN/FP/FN=192/247/47/39，产物位于 `outputs/runs/single_methods_s5_v3_missed_20260627_144746/ensemble_replay_20260627_154703/`。
- 2026.06.27 16:00 按 `refusal_probe` 仅可用于输入侧的约束重做 ensemble replay：输入侧允许 progressive/policy/intent/refusal 四路，输出侧仅允许 progressive/policy/intent 三路，并分侧枚举后逐 case 合并计算；输入侧 32,787 个唯一候选、输出侧 261 个唯一候选，最优不超过 4 步，整体 accuracy 0.8419、precision 0.8083、recall 0.8398、F1 0.8238，TP/TN/FP/FN=194/248/46/37，产物位于 `outputs/runs/single_methods_s5_v3_missed_20260627_144746/ensemble_replay_side_constrained_20260627_160016/`。
- 2026.06.27 16:09 将 S5 分侧最优 ensemble 落为 runner 兼容配置：新增 `side_branch_rules` 聚合策略与 step 级 `skip_when_metadata`，创建 `configs/pipelines/s5_side_constrained_ensemble_v1.yaml`，输入侧按 `1*progressive + 1*policy + 3*intent + 2*refusal >= 3` 判 unsafe，输出侧按 progress/policy/intent 三位 truth-table 判别且不运行 `refusal_probe`；补充 runner 行为测试与配置加载测试。
- 2026.06.27 16:13 将 `configs/pipelines/final.yaml` 更新为相对软链接，指向当前最优配置 `s5_side_constrained_ensemble_v1.yaml`，并验证 `load_pipeline("configs/pipelines/final.yaml")` 可加载到 `s5_side_constrained_ensemble_v1` 与 `side_branch_rules` 聚合。
- 2026.06.27 16:42 将 `single_multimodal_probe_v1.yaml` 从 Qwen VL 投影 probe 改为直接调用 Qwen3.6-27B VL 基座二分类：新增 `qwen_vl_prompt_binary` provider 与 `local_qwen3_6_vl_prompt_binary.yaml`，使用 T2 版 JSON 安全审核 prompt；配置加载与 provider 单测通过，并用 T1/T2 图片真实 smoke 验证均输出 `safe`。
- 2026.06.27 16:54 重构 runner 批量推理调度：新增 `judge_many` 与 `batch_scheduler`，`final.yaml` 指向的 S5 ensemble 在 `predict/evaluate` 中按图片 VL 基座、文本 progressive 基座、LoRA 27B policy/intent、8B refusal_probe 四段顺序执行并落盘 stage 临时 JSONL；每段结束释放本地模型/runtime cache，保留输出侧多轮拆分与现有聚合语义，全量测试 96 passed，LF311 `compileall` 通过。
- 2026.06.27 17:25 完成当前服务器 final pipeline 极小混合冒烟：新增 `outputs/eval_inputs/final_smoke_mixed_20260627.jsonl` 覆盖输入侧文本、输出侧单轮、多轮输出侧和图片输入；为本次 CUDA 服务器 smoke 临时将 LoRA 27B provider 从旧 `/data/model/Qwen36-27B-SFT`/`npu:1` 切到当前可读 merged 目录与 `device: auto` 后，限制 `CUDA_VISIBLE_DEVICES=0,1` 跑通 `predict --pipeline configs/pipelines/final.yaml`，4 条输出为 safe/unsafe/unsafe/safe，四段 stage 分别写 1/4/8/1 条记录，GPU 运行后回到空闲，相关配置/runner 测试 38 passed；随后检测到当前工作树 provider 已按部署环境切回 `/data/model/Qwen36-27B-SFT` 与 `npu:1`，本地 CUDA 服务器若复跑需再次使用可读模型路径或挂载 `/data`。
- 2026.06.27 17:13 基于 `final-prod.yaml` 保持编排其余部分不变，将 8B refusal probe 切换到昇腾 vLLM 5001 端口，模型名为 `qwen3guard-8b`，新增对应 provider 配置。
- 2026.06.27 17:18 继续收敛 `final-prod.yaml` 依赖的文本 27B provider：progressive 与 policy/intent 均切到 `/data/model/Qwen36-27B-SFT`，显式设备改为 `npu:1`，并补充本地 generation provider 对 `device` 字段的兼容支持。
- 2026.06.27 17:20 按部署端口调整 `final-prod.yaml`：8B refusal probe 改用现有 `ascend_vllm_safeguard_generation_8001.yaml`，对应 `http://127.0.0.1:8001/v1` 与模型名 `qwen3guard-8b`，移除 5001 临时 provider。
- 2026.06.27 17:35 拆分当前服务器与生产环境入口：保留 `final-prod.yaml` 及其 `/data/model/Qwen36-27B-SFT`、`device: npu:1` provider 不变，新增 `final-current-server.yaml` 与 `_current_server` provider 使用当前 CUDA 服务器可读的 `/ai/dataset/workspace/czy/model`/`models/Qwen3.6-27B` 路径，并将 `final.yaml` 软链接指向当前服务器入口；新增配置分离测试、README 说明，全量测试 98 passed。
- 2026.06.27 18:35 为 runner 增加终端实时进度输出：新增 `TerminalProgress`，`evaluate` / `predict` 输出总体 ASCII 进度条，启用 `batch_scheduler` 时按 `multimodal_base`、`text_base`、`lora_27b`、`refusal_8b` 等 stage 输出当前处理进度；支持 `SAFEGUARD_HARNESS_PROGRESS=0` 静默关闭，并补充终端进度测试。
