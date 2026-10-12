# Windows CUDA / VS Code 使用说明

当前维护与求职展示以 main 分支的 Windows / NVIDIA CUDA 版本为准。后续编写、调试、测试与实验均通过 VS Code Remote SSH 在 Windows 完成；Mac 仅作为远程连接终端。旧提交、旧标签和已完成实验保留真实历史，不代表当前默认运行方式。

当前运行入口与验证范围见 [CURRENT_STATUS.md](CURRENT_STATUS.md)。

Mac 连接 SSH 别名 hry26278，打开 Windows 的 D:/career-projects/multimodal.code-workspace。Mac 负责编辑，模型计算在 Windows 执行。

1. 在 VS Code 选择 Python 解释器 D:/career-projects/envs/multimodal-gpu/Scripts/python.exe。
2. 运行和调试面板选择 CUDA preflight，F5，填写一个从未使用过的运行目录名。
3. LLaVA 可选 CUDA pretrained LoRA compatibility、CUDA pretrained adapter reload、CUDA fixture LoRA training、CUDA fixture inference 或 CUDA document QA demo。软件测试图片只验证兼容性。
4. Qwen 可选 CUDA pretrained inference smoke（一个已观察开发样本），或原有三类 CUDA study。完整研究需要对应处理后数据及本地权重。
5. 每次实验使用新目录，源码、配置、输入哈希、环境、日志、预测与 .pt 权重均保留。不要覆盖历史 runs。

本机 .vscode/launch.json 已配置。仓库提供 configs/vscode-launch.cuda.json 模板，重新克隆后复制到 .vscode/launch.json，并设置自己的解释器路径。调试模板通过 VS Code 的解释器设置选取环境，没有写入个人绝对路径。不要在外层 career-projects 普通文件夹模式下寻找子项目调试配置；应打开上述多根工作区或单独项目文件夹。

CUDA 是默认设备，缺少 CUDA 会报错，避免意外占用 Mac。显式 --device mps/cpu 的兼容入口仍可使用；训练仅支持 CUDA/MPS。分词、图片读取、指标统计和温度标量校准不需要强制放入显卡。

已实测 Python 3.9.18、torch 2.8.0+cu126、torchvision 0.23.0+cu126、transformers 4.57.6。已有隔离环境不需要重装。requirements-cuda.txt 是重建依赖配方；先在新环境中安装，不能替换原 Conda 环境。

LLaVA 默认小模型为 OneVision 0.5B，训练入口在支持时使用 BF16，FP16 路径使用梯度缩放；兼容性 smoke 使用 FP16 基座。Qwen 2B 使用 FP16。8GB 显存下建议训练长边 384/512、梯度检查点与梯度累积；更大模型/分辨率需要另行实测。

历史数据集、旧 adapter 与本轮环境不随 GitHub 下载。旧 MPS 指标不改写成 CUDA 结果；迁移可运行代码与完整复跑历史实验是不同工作。历史协议仍保存原设备，新入口在运行配置中记录实际 CUDA 覆盖。

## 逐条件 LoRA（v0.5.0）

当前工作区已增加 BF16 CUDA 训练与新进程重载入口；完整顺序见 [ELEMENT_LORA.md](ELEMENT_LORA.md)。增量模板为 configs/vscode-launch.element-lora.json。已有数据准备目录不重建，实验输出必须用新名字。
