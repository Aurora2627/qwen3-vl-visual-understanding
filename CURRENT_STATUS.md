# 当前项目状态

当前版本 **v0.6.0**：Qwen3-VL 生成图逐条件符合度评测，已完成 LoRA 训练与新指令对照评测。

- Windows RTX4060 8GB、PyTorch CUDA；VS Code Remote SSH 开发/调试。Mac仅作客户端。
- v0.5训练：24张训练图、8张开发验证图，114,688 LoRA参数。验证未改善，adapter保留用于复现。
- v0.6评测：16个新指令组、32张图、128元素；固定checkpoint，512次模型前向。
- 原模型准确率65.62%，LoRA64.06%；差值CI跨零。LoRA NLL改善但仍差于常数先验；图片相对纯文本的准确率收益CI跨零。
- 36项测试通过，运行190.1秒，峰值已分配显存3.99GiB。
- 原模型继续作为默认方案。新子集来自公开train划分，不是官方benchmark，不保证排除基础模型预训练污染；现已观察，后续调参须另留最终测试。

[本轮报告](reports/v0.6.0/RESULTS.md) · [评测与调试说明](FRESH_EVALUATION.md) · [LoRA训练说明](ELEMENT_LORA.md)
