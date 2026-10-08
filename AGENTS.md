# 开发约定

模型训练与评测使用 Python 源码和 PyTorch。所有运行从 VS Code 集成终端或调试器启动。保存每次实验的源码快照、配置、输入哈希、日志、.pt 权重和结果，不覆盖已有实验。数据、依赖、权重、环境文件和本机配置不上传 GitHub。main 保留真实可追溯版本；缺失历史只能从快照恢复并注明，禁止虚构训练结果。

每次完成更新后维护 VERSION、CHANGELOG.md、VERSION_HISTORY.md；检查通过后从 VS Code 提交 main、创建带说明的版本标签并推送 GitHub。模型实现坚持 Python/PyTorch；运行环境与硬件兼容性需实测，不能把替代框架的运行当作 PyTorch 实验。

用户指定先推进 LLaVA；本项目仅初始化，未获后续具体实验任务前不启动训练。
