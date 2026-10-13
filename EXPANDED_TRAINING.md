# 扩充训练与封存测试运行说明

使用 Windows RTX 4060 Laptop 8GB、PyTorch CUDA 和项目专用虚拟环境。在 VS Code 打开本仓库文件夹，选择该环境的 Python 解释器；无需固定盘符或工作区名称。

## 本轮问题

旧模型对不少误判仍给出很高置信度，而且旧训练监督完整词表、评测却只比较 PASS/FAIL 两个词。新增 binary_vote_loss 与旧 full-vocabulary 损失做控制实验。它仍然使用原生 PyTorch LoRA，更新语言末4层注意力q/v；视觉编码器和其余基座冻结。

每条条件的人工多票取均值作为 PASS 软目标。全词表损失包含其他词的概率质量；条件二分类损失先在 PASS/FAIL 上归一化。两种目标都只监督答案首个 token，不监督图片/问题 token，也不训练解释或生成图片。不能仅凭目标更贴合评测就宣称性能更好。

## 数据与选模

80个新指令组、160张图，按固定哈希顺序分48/16/16组，对应96/32/32图；过滤所有已观察指令、prompt_id、源图片路径，下载后检验跨划分图片哈希。先保存 selection-lock.json 和计划，再下载。train.jsonl、validation.jsonl 用于开发；sealed_test.jsonl不用于训练、校准或选模。

学习率5e-5、1epoch、种子20261013、rank4/alpha8/末4层、最长边224、BF16均固定；只比较两种损失，用开发验证NLL选一个adapter，锁定checkpoint SHA256，然后首次运行封存测试。结果现被观察后，下一轮调参必须另留最终测试集。数据来自EvalMuse公开train划分，不是官方测试，也不保证基础模型未在预训练中见过。

## VS Code 调试

选择项目专用 multimodal-gpu 解释器。在运行和调试中选择 Qwen3-VL CUDA expanded element training，选择loss mode、已完成的数据目录和新的输出目录。模板 configs/vscode-launch.expanded-elements.json。输出不允许覆盖旧实验。

完整自动流程源码 scripts/run_expanded_study.py：测试 → 数据准备 → 8元素训练兼容性检查 → 两种正式训练及新进程重载 → 验证选模与锁定 → 封存测试。当前实际由 VS Code 集成终端通过 Windows Start-Process 启动独立后台进程，用于减少远程窗口关闭导致中断的风险；本轮未主动断开连接验证后台任务存活，重连后仍应检查PID；stdout/stderr 和每个stage的日志分别落盘。调试器直接运行适合排错，远程窗口关闭可能中止这类前台任务。

本机活动任务记录包含 run/evidence/pid；其位置由实际启动方式决定。先检查进程、active-step.json、steps.json、FAILED.json和completed.json再决定后续操作；不要重复启动覆盖目录。后台任务不会自行重复或定时运行。

## 证据

每个模型run保存源码快照、配置、输入哈希、环境、训练日志、预测、原生adapter.pt及重载结果。总study保存每stage退出码、验证选模记录、封存测试哈希。最终报告由scripts/report_expanded_study.py重新核对源码和预测计算。

GitHub仅保存源码、汇总报告、哈希和版本历史；不上传图片、逐条标签/预测、模型权重、虚拟环境或本机launch。旧失败下载目录保留。

## 补充的校准基线

在封存测试开始前，另行注册并锁定仅用开发验证集拟合的正温度。src/element_vote_temperature.py使用PyTorch CPU在0.25至32的固定logspace网格（含1）上最小化验证软投票NLL。scripts/fit_element_temperature.py拒绝sealed_test划分，并核验验证ID及标签一致。该基线不改变adapter选模；测试只应用固定温度，不再次拟合。正温度不改变PASS/FAIL的argmax，只用于检查概率评分改善是否能由简单置信度调整获得。
