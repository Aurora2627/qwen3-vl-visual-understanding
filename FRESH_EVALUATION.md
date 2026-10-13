# 新指令评测与视觉依赖诊断

在 Windows 的 VS Code 中运行，选择项目隔离环境 Python 3.9.18（PyTorch 2.8.0+cu126）。

本流程冻结 v0.5.0 的 LoRA checkpoint，不继续训练。抽取 16 个历史指令之外的新指令组、32 张图片、128 条条件；按固定种子哈希排序，不按模型输出或标签分数挑选。下载后检查图片哈希。来自 EvalMuse 公开训练划分，不是官方测试集。

在运行和调试中选择 CUDA fresh element evaluation (locked adapter)，配置模板见 configs/vscode-launch.fresh-evaluation.json。使用已有数据和 checkpoint，输出必须为新目录。图片缩放和答案协议一致，纯文本组完全不输入图片。比较原模型/LoRA 的图片/纯文本条件，以及只用训练标签计算的常数先验。

入口：scripts/prepare_fresh_elements.py、scripts/evaluate_fresh_elements.py、scripts/report_fresh_elements.py；抽样与配对统计在 src/fresh_evaluation.py。准备数据时提供 checkpoint，下载前锁定其 SHA256；评测拒绝修改后的 checkpoint。已有实验禁止覆盖。

报告：reports/v0.6.0/RESULTS.md。图片、逐条标签和 .pt 权重只留在 Windows。新评测用于调参后，下一轮需另留最终测试集。
