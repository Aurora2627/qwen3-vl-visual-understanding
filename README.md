# Qwen3-VL 视觉生成结果评测与诊断

## 当前版本 v0.7.0：扩充训练与封存测试

新增160图/80指令组：96图/430条件训练，32图/160条件验证，32图/192条件封存测试。比较相同数据/种子/lr5e-5/1epoch下的全词表与条件二分类损失；仅按开发验证NLL选定full-vocabulary，锁定权重哈希后一次评测封存测试。基座准确率0.7196，adapter0.7354；投票NLL1.2201→0.7906。准确率差的分组置信区间不支持明确提升，不宣称准确率改善。验证集拟合并在测试前锁定的温度基线T=7.0421，测试NLL为0.6171，准确率不变；评估微调收益时一并对照。

[本轮实验报告](reports/v0.7.0/RESULTS.md) · [训练与调试说明](EXPANDED_TRAINING.md)。Windows/PyTorch CUDA，43项测试通过。保留冻结基座作为默认入口；新adapter为可复现的训练结果。LLaVA与本项目在Windows总目录中各有独立Git仓库和工作区入口。

## 当前版本 v0.6.0：新指令评测与视觉依赖诊断

在排除全部历史指令的新32张图上比较原模型、LoRA、无图输入和训练标签先验。LoRA未提高准确率，但投票NLL有所改善；纯文本对照的视觉收益尚不确定。详见[本轮报告](reports/v0.6.0/RESULTS.md)与[运行说明](FRESH_EVALUATION.md)。当前使用Windows/PyTorch CUDA，36项测试通过；此前训练和评测结果保留。

## 当前进展：逐条件 LoRA（v0.5.0）

已在 Windows RTX 4060 上完成 Qwen3-VL-2B-Instruct 的人工投票监督 LoRA 训练，而前期温度校准只拟合一个标量。训练语言模型最后 4 层 q/v 的 114,688 个 LoRA 参数，视觉编码器与基座冻结；一轮 32 次更新、checkpoint 新进程重载和 31 项测试通过，峰值已分配显存约 4.01 GiB。

本轮验证准确率 **76.9%→73.1%**、NLL **1.0398→1.0522**，没有改善。数据为此前观察过的开发图，尚无新的独立测试结论，不能宣称微调提升效果。完整方法、常数先验对照和失败记录见 [v0.5 报告](reports/v0.5.0/RESULTS.md)，运行方式见 [训练指南](ELEMENT_LORA.md)。下文保留前期评测实验的真实历史。


当前维护与求职展示以 main 分支的 Windows / NVIDIA CUDA 版本为准。后续编写、调试、测试与实验均通过 VS Code Remote SSH 在 Windows 完成；Mac 仅作为远程连接终端。旧提交、旧标签和已完成实验保留真实历史，不代表当前默认运行方式。

当前运行入口与验证范围见 [CURRENT_STATUS.md](CURRENT_STATUS.md)。

当前开发环境：Windows + CUDA，Mac 通过 VS Code Remote SSH 编辑。模型入口默认使用 NVIDIA GPU。启动步骤见 [VSCODE_START.md](VSCODE_START.md)，兼容性记录见 [本版检查](reports/v0.4.1/compatibility.json)。下文历史 MPS 指标仍描述原实验。

目标：判断生成图片是否满足原始指令，输出逐条件判断与可见证据，并研究评估器的误判、拒答和输入依赖。

## 当前实现

- 官方 Qwen3-VL-2B-Instruct，Transformers/PyTorch；默认 CUDA，缺少 CUDA 明确报错；其他设备仅保留兼容接口。
- 两种协议：整体判断 baseline、逐条件判断并以确定性规则聚合。
- 无图与错配图片控制，用于诊断是否依赖图像。错配图片不能沿用原图标签计算准确率。
- 严格 JSON 解析，PASS/FAIL/UNCERTAIN/INVALID 分开统计；所有样本进入总准确率分母。
- 记录逐题原始输出、输入 token、耗时、配置、权重 revision、源码快照和图片哈希。

## 第一阶段数据边界

先生成四张简单几何图，每张配一条满足和一条不满足的指令，共八例，覆盖颜色、数量、空间关系和属性绑定。它们是程序生成的开发诊断样例，不是扩散模型生成图、人工标注或独立 benchmark。确定答案来自绘图场景规格，不传给模型。逐条件拆分目前由清单提供，尚未验证自动拆分能力。

本版不训练模型，因此没有微调权重；`.pt` 保存评测计数张量。模型原始 safetensors 在本地校验后使用，不上传。

已在 MacBook Air M5 16GB 的 VS Code 中以系统 Python / PyTorch / MPS 完成 32 次真实推理；六项单元测试通过。整体判断 8/8，逐条件判断 6/8（一个格式错误、一个空间关系误接受），全 PASS/FAIL 基线均为 4/8。无图控制全部输出 PASS，说明该协议不能可靠识别视觉证据缺失。这些是四张图的开发诊断结果，不代表真实生成图效果。

运行见 [VS Code 说明](VSCODE_START.md)，完整结果、失败分析和原始输出见 [诊断报告](reports/v0.2.0/RESULTS.md)。

## 真实生成图主实验（v0.3.0）

接入 EvalMuse 公开训练数据中的真实文生图及人类评分。固定抽取 32 条指令、每条两张图，共 64 张；开发/保留各 16 条指令，并检查规范化文本、指令 ID 和图片哈希无交叉。抽样不参考评分高低或模型效果。

- overall baseline：模型直接给 1–5 分。
- elements：输入数据集提供的要素，逐项判断并聚合；不传入人工分数或正确性标签。
- PyTorch 计算 SRCC/PLCC、按指令分组的 bootstrap 区间、同指令非平局排序及细粒度错误；同时给出覆盖率、全 PASS/FAIL 基线和平衡准确率。
- 按需读取固定 revision 的远程 ZIP 字节范围，提取图片时校验 CRC32 并记录 SHA256，避免下载完整约 55 GB 压缩包；未校验完整压缩包哈希。

来源、许可证和边界见 [数据说明](DATA_SOURCES.md)。这是小规模人工一致性试验，未复现完整官方榜单，也没有微调。后续校准或后训练需要新的未观察指令保留集。

本项目衡量内容指令遵循，不把它等同于审美质量、安全性、视频时序或通用视觉质量。

## 官方来源

- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- https://github.com/QwenLM/Qwen3-VL
- https://github.com/djghosh13/geneval （任务划分参考，当前未复现官方评估器）

### v0.3.0 本机结果

| 数据/协议 | 有分数/总数 | SRCC | 排序正确/非平局 |
|---|---:|---:|---:|
| development/overall | 19/32 | 未定义 | 0/12 |
| development/elements | 25/32 | 0.545 | 2/12 |
| heldout/overall | 16/32 | 未定义 | 0/14 |
| heldout/elements | 31/32 | 0.352 | 3/14 |

已在 VS Code 使用系统 Python / PyTorch / MPS 完成 128 次推理；13 项测试通过。相关性仅对有分数输出计算，排序中的模型平局、格式错误和不计分均算错。详细区间、全 PASS/FAIL 基线、类别误差及完整性校验见 [真实图报告](reports/v0.3.0/RESULTS.md)。

每部分只有 16 个指令组，不能据此宣称显著提升。本版没有微调；原始输入、完整预测及源码快照保留在 `runs/real-20261012-101552`。导出与收尾校验程序在模型运行期间完善，不改动运行前保存的源码快照。

## 固定选项与开发集校准（v0.4.0）

官方 Qwen3-VL-2B-Instruct 固定权重，在 VS Code Remote SSH 的 Windows RTX 4060 上以 PyTorch CUDA/FP16 运行。自由评分与固定选项使用相同短提示词。开发集 32 张沿用既有数据；保留集 32 张来自此前未观察指令。

| 数据/协议 | 有分数/总数 | SRCC | 排序正确/非平局 |
|---|---:|---:|---:|
| development/overall_free | 32/32 | 0.477 | 3/12 |
| development/overall_fixed | 32/32 | 0.477 | 3/12 |
| development/elements_fixed | 24/32 | 0.478 | 3/12 |
| development/overall_expected | 32/32 | 0.469 | 7/12 |
| development/overall_calibrated | 32/32 | 0.445 | 8/12 |
| heldout/overall_free | 32/32 | 0.446 | 7/15 |
| heldout/overall_fixed | 32/32 | 0.446 | 7/15 |
| heldout/elements_fixed | 26/32 | 0.248 | 4/15 |
| heldout/overall_expected | 32/32 | 0.442 | 12/15 |
| heldout/overall_calibrated | 32/32 | 0.439 | 13/15 |

共 428 次模型生成、192 条图片/协议记录，23 项测试通过。完整指标、按指令重采样区间、概率校准与限制见 [v0.4 报告](reports/v0.4.0/RESULTS.md)。

温度由 PyTorch 在开发集人工评分票分布上拟合，仅改变一个标量，不更新 Qwen 参数；候选集合内相对概率不能当作真实正确置信度。公开训练数据的小规模子集未复现官方榜单，新保留结果公开后不继续用它调参。

Mac 中断实验保持原样；本次 CUDA 运行单独保存在 runs/choice-cuda-20261012-175148。源码、输入、日志、预测和 .pt 参数保留本机，数据和权重不上传 GitHub。

本次自由与固定选项在 64/64 张图上离散分数相同，不能宣称约束打分提高准确率。校准改善保留集 NLL/Brier，未改善 SRCC；细粒度判断仍有明显误差。详见 v0.4 报告的结果解读。
