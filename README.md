# Qwen3-VL 生成图指令符合度评测

输入生成图片及指令条件，使用 Qwen3-VL-2B-Instruct 判断各条件是否满足，研究评估器的概率质量、视觉依赖与微调效果。项目评估内容指令遵循，不生成图片，也不将该分数等同于审美或通用视觉质量。

## 实现

- 官方 Transformers/PyTorch 模型与固定 revision；整体评分、逐条件判断及固定选项概率协议。
- 原生 PyTorch LoRA：语言模型末 4 层注意力 q/v，rank=4、alpha=8，114,688 个可训练参数；冻结视觉与其余基座，BF16 CUDA，峰值已分配显存约 4.01 GiB。
- 人工投票均值作为 PASS 软标签，比较全词表与 PASS/FAIL 条件二分类损失；仅监督答案首 token，不训练解释文本。
- 按指令分组隔离、输入/权重哈希锁定、验证选模后封存测试、图片/纯文本控制、训练先验及温度校准基线。
- NLL、Brier、准确率、平衡准确率、按指令组的配对 bootstrap 与误判诊断；正式权重新进程重载核验。

## 最新实验

EvalMuse 公开训练数据中新增 80 个指令组/160 张图，排除既有指令和图片：训练 96 图/430 条件、验证 32 图/160 条件、封存测试 32 图/192 条件。两种损失在相同种子、学习率 5e-5、1 epoch 下对照；只按验证 NLL 选定全词表目标，在首次测试前锁定 checkpoint SHA256。

| 封存测试方法 | 投票 NLL | 准确率 |
|---|---:|---:|
| 原模型 + 图片 | 1.2201 | 0.7196 |
| LoRA + 图片 | 0.7906 | 0.7354 |
| 原模型 + 验证温度 | 0.6171 | 0.7196 |

温度 T=7.0421 仅由验证集拟合并在测试前锁定。LoRA 减少了相对未经校准基座的 NLL，但未优于校准基座；准确率差 95% CI [-0.0327, 0.0625]，不支持明确提升。测试有 3 条人工平票条件不进入硬准确率，192 条均进入软概率评分。

[完整结果与限制](reports/v0.7.0/RESULTS.md) · [训练和封存测试流程](EXPANDED_TRAINING.md)

这些数据来自公开 train 的小规模子集，不是官方 benchmark，不能保证排除基础模型预训练污染；只有一个训练种子和 16 个测试指令组。测试结果现已观察，后续调参需另留最终测试。基座保持默认入口，adapter 是可复现实验结果。

## 运行与代码

已验证 Windows RTX 4060 Laptop 8GB、Python 3.9.18、PyTorch 2.8.0+cu126、Transformers 4.57.6。模型默认 CUDA；43 项测试通过，两个正式 checkpoint 的新进程重载通过。

[VS Code 启动](VSCODE_START.md) · [当前状态](CURRENT_STATUS.md) · [数据来源](DATA_SOURCES.md)

- src/modeling.py：官方模型加载与生成。
- src/element_training.py、scripts/train_element_lora.py：软投票监督与原生 LoRA。
- scripts/prepare_element_corpus.py：指令隔离、固定抽样与图片哈希。
- scripts/run_expanded_study.py：训练、验证选模与封存测试。
- src/element_vote_temperature.py：仅验证集温度校准。
- src/element_diagnostics.py、scripts/report_expanded_study.py：误判和统计报告。

原生 adapter.pt 不兼容 PEFT 格式；不提供优化器状态恢复。每次运行保存源码、配置、输入哈希、日志、预测及权重，使用新的输出目录。GitHub 仅保存源码、汇总报告和哈希；图片、逐条标签/预测、模型权重、虚拟环境与本机配置不上传。

## 相关实验记录

[固定选项与评分校准](reports/v0.4.0/RESULTS.md) · [初始 LoRA](reports/v0.5.0/RESULTS.md) · [新指令视觉依赖对照](reports/v0.6.0/RESULTS.md) · [版本历史](VERSION_HISTORY.md)

早期合成诊断和自由评分记录保留在 reports/v0.2.0、reports/v0.3.0；不作为最新训练结果。
