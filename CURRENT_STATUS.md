# 当前项目入口

当前维护和求职展示以 main 分支的 Windows / NVIDIA CUDA 代码为准。编写、调试、测试与实验均在 Windows 的 VS Code Remote SSH 工作区进行；Mac 仅作为连接终端。历史版本与旧设备记录保留真实来源。

当前版本 **v0.5.0**：Qwen3-VL 生成图指令符合度评测，以及逐条件人工投票监督 LoRA 开发实验。

- 前期 v0.4：64 张真实图、428 次模型调用，固定选项与概率校准研究，见 [报告](reports/v0.4.0/RESULTS.md)。温度校准不更新 Qwen 参数。
- 当前 v0.5：实际更新 Qwen 的 LoRA 参数；24 张训练图、8 张验证图，均来自既有开发数据。见 [训练报告](reports/v0.5.0/RESULTS.md) 与 [训练运行指南](ELEMENT_LORA.md)。
- 训练、评测和新进程 checkpoint 重载通过，31 项测试通过。
- 本轮 LoRA 未改善验证结果。当前仍推荐冻结基座；已训练 adapter 作为可复现实验保留。
- 新未观察评测图因 Windows huggingface.co DNS 故障尚未获取；没有独立测试效果结论。

技术栈：Python 3.9.18、PyTorch 2.8.0+cu126、Transformers 4.57.6、RTX 4060 Laptop 8GB。训练使用 BF16 基座和 FP32 LoRA。旧推理入口默认 CUDA/FP16，运行参数与环境均随实验保存。

使用 [VS Code 启动说明](VSCODE_START.md)；基础调试模板为 configs/vscode-launch.cuda.json，训练增量模板为 configs/vscode-launch.element-lora.json。原始图片、逐条人工标签、模型权重、环境与本机配置不上传 GitHub。
