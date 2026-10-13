# VS Code / CUDA 运行说明

在 Windows 上打开本仓库文件夹，使用隔离的 Python 环境；也可通过 VS Code Remote SSH 打开服务器上的仓库。选择 Python 解释器后，复制对应 configs/vscode-launch*.json 模板到本机 .vscode/launch.json。模板使用所选解释器，不要求固定盘符、账户名或 SSH 别名。

已验证环境：Python 3.9.18、torch 2.8.0+cu126、torchvision 0.23.0+cu126、transformers 4.57.6、RTX 4060 Laptop 8GB。重建环境参考 requirements-cuda.txt，使用新的虚拟环境。模型入口默认 CUDA；CUDA 不可用会报错。数据处理、指标和标量校准使用 CPU。

先运行 configs/vscode-launch.cuda.json 中的 CUDA preflight，再运行模型任务。下载的模型、数据和 adapter 必须与实验要求匹配；每次使用新的输出目录，保留源码、配置、输入哈希、日志、预测和权重。不要覆盖已有 runs；同一张 GPU 同时运行一个模型任务。

## 逐条件训练与评测

- configs/vscode-launch.expanded-elements.json：Qwen3-VL CUDA expanded element training，选择损失目标、已准备数据和新的输出目录。流程见 [EXPANDED_TRAINING.md](EXPANDED_TRAINING.md)。
- configs/vscode-launch.fresh-evaluation.json：CUDA fresh element evaluation (locked adapter)，见 [FRESH_EVALUATION.md](FRESH_EVALUATION.md)。
- configs/vscode-launch.element-lora.json：初始 LoRA 训练与新进程重载，见 [ELEMENT_LORA.md](ELEMENT_LORA.md)。

最新逐条件 LoRA 采用 BF16 基座、FP32 adapter 和最长边 224 预缩放；较早的生成评分入口采用 FP16，两者精度由各自实验配置记录。不要把不同协议的输入尺寸和精度直接混用。

完整流程 scripts/run_expanded_study.py 依次检查数据、训练、重载、验证选模和封存测试。运行前阅读对应脚本帮助与数据说明；已有数据与权重不随 GitHub 下载。后台运行时保存 PID 与日志，重连后先检查进程和状态文件，避免重复启动。
