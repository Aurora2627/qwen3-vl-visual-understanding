# 当前项目入口

当前维护与求职展示以 main 分支的 Windows / NVIDIA CUDA 版本为准。后续编写、调试、测试与实验均通过 VS Code Remote SSH 在 Windows 完成；Mac 仅作为远程连接终端。旧提交、旧标签和已完成实验保留真实历史，不代表当前默认运行方式。

- 推荐入口：main；当前版本 v0.4.2。
- 技术栈：Python、PyTorch 2.8.0+cu126、Transformers 4.57.6；Windows RTX 4060 Laptop 8GB。
- 启动：按 VSCODE_START.md 配置隔离解释器，使用 configs/vscode-launch.cuda.json 调试模板。
- 历史版本仍可从 Git 查看，用于展示迭代过程；请勿照搬旧 MPS 命令作为当前启动方式。

当前主要实验为 [v0.4 CUDA 固定选项与校准研究](reports/v0.4.0/RESULTS.md)：64 张真实图、428 次模型调用；23 项测试通过。后续 v0.4.1 已验证实际单图 CUDA 推理。当前未进行 Qwen 参数微调；温度校准只拟合一个标量。

兼容性证据见 [CUDA 检查](reports/v0.4.1/compatibility.json)。早期 v0.2/v0.3 MPS 报告仅为历史对照。
