# VS Code 运行与调试

打开本目录。使用 `/usr/bin/python3`；本机已有依赖通过 PYTHONPATH 加载，未创建新的虚拟环境。`.vscode` 中提供本机设置和 F5 配置，绝对路径配置不上传。

在本目录的 VS Code 集成终端运行：

```sh
export PYTHONPATH=/Users/meiying/Documents/Codex/2026-10-07/zhe/.system-python-packages
/usr/bin/python3 -m unittest discover -s tests -v
/usr/bin/python3 scripts/prepare_diagnostics.py
/usr/bin/python3 scripts/download_model.py
/usr/bin/python3 scripts/run_study.py --device mps
```

图片生成入口拒绝覆盖已有目录；同一数据可直接重复评测，评测自动建立带时间戳的新 runs 目录。权重下载可断点恢复，检查官方 revision 与 LFS SHA256。网络需要代理时，在本机终端配置 HTTP_PROXY/HTTPS_PROXY，代码不保存凭据。

F5 选择“02 Qwen3-VL MPS 评测”，可在 src/judging.py 的解析函数和 src/modeling.py 的生成调用处设置断点。MPS 不可用会明确报错。CPU 必须显式传 --device cpu，CUDA 需要对应硬件；本版尚未验证云端。

结果包含原始 JSONL、汇总、源码快照和状态。格式错误、达到生成长度限制和拒答需复核；不可只挑正确输出展示。数据、权重与本机环境不推送 GitHub。

本机已完成 `runs/study-20261011-220741`，32 次 MPS 推理。导出已完成实验可用 `scripts/export_report.py <运行目录> --output <新的报告目录>`；入口拒绝覆盖已有报告。导出源码在首次模型实验后添加，原始运行快照保持原样。
