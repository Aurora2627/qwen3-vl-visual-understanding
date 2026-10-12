# 逐条件 LoRA 训练与调试

本功能在 Windows NVIDIA CUDA 上运行。数据准备、训练、评测和测试均从 VS Code 集成终端或调试器启动；使用隔离环境中的 GPU PyTorch，依赖沿用 requirements-cuda.txt。

## 运行顺序

1. 首次运行 scripts/prepare_element_sft.py，从既有 data/processed/evalmuse-pilot-v2/manifest.jsonl 按指令组建立训练/验证。默认输出 data/processed/element-sft-v1，已有目录会报错以保护历史；本工作区已经准备完毕，无需重建。
2. 在运行和调试面板选择 CUDA element LoRA development training (BF16)，输入新的实验目录名。默认一轮，学习率 1e-4、梯度累积 4、rank=4、alpha=8、最后 4 层语言注意力 q/v。模型已缓存时全程无需联网。
3. 选择 CUDA element LoRA fresh-process reload，指定上一步实验目录和新的输出名，检查重新加载原模型及 adapter 后的验证 logits 是否一致。

当前工作区 .vscode/launch.json 已合并三个入口。重新克隆时，将 configs/vscode-launch.element-lora.json 的 configurations 和 inputs 合并到 .vscode/launch.json，选定自己的 GPU Python 解释器。不要覆盖原有调试项或复用已有输出目录名。

在集成终端等效执行（将 NEW_RUN 改为新的名字）：

    python scripts/train_element_lora.py --model-dir data/cache/local-qwen3-vl/89644892e4d85e24eaac8bacfd4f463576704203 --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 --precision bf16 --output runs/NEW_RUN

重新加载时额外传入 --reload-checkpoint runs/TRAIN_RUN/adapter.pt，并使用新输出目录。若训练时修改了 rank、alpha、last-layers、max-edge、precision、seed 或 max-validation，重载时须保持一致，否则拒绝继续。

## 源码入口与实验边界

- src/element_data.py：指令组隔离、标签校验、逐条件提示词。
- src/element_lora.py：原生 PyTorch LoRA，零初始化保持基座行为，参数冻结与严格重载。
- src/element_training.py：官方图像处理、答案首 token 软投票监督、二类概率指标。
- scripts/train_element_lora.py：训练、验证选 epoch、保存 .pt、显存和梯度检查、新进程重载。
- scripts/report_element_study.py：核对训练源码与快照一致，导出不含图片/逐条标签/权重的汇总报告。

训练目标是首个 PASS/FAIL token 的全词表交叉熵，人工投票比例为软目标；问题和图像 token 不参与损失。评测概率在 PASS/FAIL 两个 token 内归一化，不是已校准的真实正确置信度。当前二类任务不输出 UNCERTAIN；不能与旧版三类判定或整体五分评分直接比较。

已完成 24 图/128 元素训练与 8 图/28 元素验证。验证只有 4 个指令组，且属于已观察开发数据，不能作为独立效果证明。原 FP16 全数据训练失败记录保留，BF16 运行另建目录。当前 adapter 验证结果劣于基座，不推荐替换。

每轮保存源码快照、输入哈希、模型文件指纹、环境、配置、日志、预测与 adapter.pt。adapter 为本项目原生格式，不能直接当作 PEFT 权重使用。没有保存完整基础权重或优化器状态，本入口暂不支持中断续训。

下一阶段在网络恢复后获取未参与开发的新指令与图片，锁定协议和 checkpoint，进行基座/LoRA/标签先验/无图控制的独立比较。已观察的历史保留集继续隔离。
