# 开发约定

模型训练与评测使用 Python 源码和 PyTorch。所有运行从 VS Code 集成终端或调试器启动。保存每次实验的源码快照、配置、输入哈希、日志、.pt 权重和结果，不覆盖已有实验。数据、依赖、权重、环境文件和本机配置不上传 GitHub。main 保留真实可追溯版本；缺失历史只能从快照恢复并注明，禁止虚构训练结果。

每次完成更新后维护 VERSION、CHANGELOG.md、VERSION_HISTORY.md；检查通过后从 VS Code 提交 main、创建带说明的版本标签并推送 GitHub。模型实现坚持 Python/PyTorch；运行环境与硬件兼容性需实测，不能把替代框架的运行当作 PyTorch 实验。

用户于 2026-10-11 授权实施本项目：Qwen3-VL 视觉生成结果评测与诊断。早期已验证 PyTorch/MPS 流程，当前运行使用 Windows PyTorch/CUDA；程序生成的测试图片必须标记为合成诊断，不作为真实生成模型 benchmark 或人工评测。

## 当前执行约定（2026-10-12）

当前维护与求职展示以 main 分支的 Windows / NVIDIA CUDA 版本为准。后续编写、调试、测试与实验均通过 VS Code Remote SSH 在 Windows 完成；Mac 仅作为远程连接终端。旧提交、旧标签和已完成实验保留真实历史，不代表当前默认运行方式。
