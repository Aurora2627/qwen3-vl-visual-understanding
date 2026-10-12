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

## EvalMuse 真实生成图

仍在本项目的 VS Code 集成终端运行（沿用上面的 PYTHONPATH）：

```sh
/usr/bin/python3 scripts/inspect_dataset.py
/usr/bin/python3 scripts/prepare_evalmuse.py
/usr/bin/python3 -m unittest discover -s tests -v
/usr/bin/python3 scripts/run_real_study.py --device mps
/usr/bin/python3 scripts/report_real_study.py runs/real-实际时间戳 --output reports/新的报告目录
```

`inspect_dataset.py` 获取官方元数据和公开标注；`prepare_evalmuse.py` 验证标注哈希，固定抽样并按需提取图片。完整数据清单存在时拒绝覆盖；准备中断可使用已有字节缓存恢复。模型实验不覆盖历史运行，也不自动恢复中断推理。

F5 选择“03 EvalMuse 真实图人工一致性评测”。断点可放在 `src/real_judging.py` 的提示词与解析、`src/modeling.py` 的生成调用、`src/correlation.py` 的指标统计。一次完整运行含 128 次推理（64 张图 × 2 种协议）；完整输入、输出、源码快照及分数 `.pt` 仅保留本机，报告导出排除原始数据。

长实验可在另一 VS Code 终端运行收尾程序：

```sh
/usr/bin/python3 scripts/finalize_real_study.py runs/real-实际时间戳 --publish
```

它等待本次实验成功，依次运行测试、报告导出、快照/输入/张量/发布字段校验，再更新结果文档并提交、标记和推送 v0.3.0。去掉 `--publish` 则只生成报告和文档。等待期间若源码、Git HEAD 或其他文件发生更改，会停止收尾，留待检查；模型失败或校验失败也不发布。报告与收尾计划拒绝覆盖，不能盲目重复执行。收尾程序本身不启动模型实验。
