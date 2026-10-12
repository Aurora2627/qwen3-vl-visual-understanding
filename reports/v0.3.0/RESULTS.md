# EvalMuse 真实生成图试验

来源：[EvalMuse 官方数据](https://huggingface.co/datasets/DY-Evalab/EvalMuse)。使用公开训练部分中的固定子集，非官方测试集、非完整榜单复现。64 张真实文生图、32 条指令，每条两张图；开发与保留各 16 条指令，按规范化文本隔离，并检查指令 ID 和图片哈希无交叉。

使用公开人类评分，无新增人工标注。整体参考是 1–5 分的人工均值；细粒度参考是每项二元标注的严格多数票。标注置信度信息保留，没有按评分高低或模型效果挑样本。

## 方法

- overall：Qwen3-VL 直接给 1–5 分与证据。
- elements：输入数据集提供的内容要素（不是人工正确性标签），输出逐项 PASS/FAIL/UNCERTAIN。所有项可判断时，以 PASS 比例作为图片分数；任一项不确定则整体不计分。没有自动生成内容拆分，也没有训练评估器。
- 512 个最大输出 token、448 长边、FP16 MPS、greedy、eager attention。先固定协议，随后同一次运行执行开发与保留数据，不根据保留结果改提示词。

## 整体分数与人工判断

| 数据/协议 | 有分数/总数 | SRCC | 按指令重采样的 95% 区间 | PLCC | 非平局排序正确/总数 |
|---|---:|---:|---|---:|---:|
| development/overall | 19/32 | 未定义 | 未定义 | 未定义 | 0/12 |
| development/elements | 25/32 | 0.545 | [0.143, 0.804] | 0.608 | 2/12 |
| heldout/overall | 16/32 | 未定义 | 未定义 | 未定义 | 0/14 |
| heldout/elements | 31/32 | 0.352 | [0.017, 0.627] | 0.380 | 3/14 |

SRCC/PLCC 只在有分数样本上计算，必须和覆盖率一起读。相关性衡量与人类评分的一致性，不是准确率。排序排除人工均分平局；模型平局、不计分、格式错误算错。随机非平局排序基线为 50%；常数分数的相关性因零方差而未定义。每个数据部分仅 16 个独立指令组，区间不稳定，不能据此宣称方法显著优于 baseline。

## 细粒度与输出可靠性

| 数据/协议 | 格式错误 | 输出截断 | 耗时中位数/秒 | 细粒度正确/总数 | 人工有分歧的要素 |
|---|---:|---:|---:|---:|---:|
| development/overall | 13 | 13 | 28.23 | — | — |
| development/elements | 4 | 3 | 37.89 | 87/150 | 80 |
| heldout/overall | 16 | 16 | 46.96 | — | — |
| heldout/elements | 0 | 0 | 42.41 | 75/112 | 59 |

| 数据/逐项协议 | 细粒度准确率 | 全 PASS 基线 | 全 FAIL 基线 | 平衡准确率 |
|---|---:|---:|---:|---:|
| development/elements | 0.580 | 0.607 | 0.393 | 0.508 |
| heldout/elements | 0.670 | 0.598 | 0.402 | 0.603 |

细粒度格式错误/不确定算错，人工票数平局不参与准确率。全 PASS/FAIL 基线揭示标签偏斜；平衡准确率取正负类召回均值，不确定与格式错误均为对应类未答对。速度包含输入处理和同步，不含模型加载；两个协议输出长度不同，不能当作严格速度基准。

### 按要素类别诊断

| 数据 | 类别 | 要素数 | 正确 | 误接受 | 误拒绝 | 不确定/格式错误 |
|---|---|---:|---:|---:|---:|---:|
| development | Counting | 4 | 3 | 1 | 0 | 0 |
| development | Other-time | 2 | 1 | 1 | 0 | 0 |
| development | Other-weather | 4 | 4 | 0 | 0 | 0 |
| development | Spatial | 4 | 2 | 2 | 0 | 0 |
| development | activity | 15 | 4 | 4 | 1 | 6 |
| development | animal | 4 | 3 | 1 | 0 | 0 |
| development | attribute | 16 | 11 | 0 | 1 | 4 |
| development | attribute-style-artist/works | 4 | 4 | 0 | 0 | 0 |
| development | color | 4 | 0 | 2 | 0 | 2 |
| development | human | 22 | 10 | 5 | 1 | 6 |
| development | location | 14 | 13 | 0 | 0 | 1 |
| development | material | 2 | 2 | 0 | 0 | 0 |
| development | object | 15 | 8 | 5 | 1 | 1 |
| development | object-cloth&accessories | 2 | 2 | 0 | 0 | 0 |
| development | object-other | 4 | 4 | 0 | 0 | 0 |
| development | other | 20 | 6 | 1 | 0 | 13 |
| development | other-perspective | 6 | 5 | 0 | 1 | 0 |
| development | spatial | 4 | 2 | 1 | 0 | 1 |
| development | time | 4 | 3 | 1 | 0 | 0 |
| heldout | Color)  | 4 | 2 | 2 | 0 | 0 |
| heldout | Counting | 3 | 1 | 2 | 0 | 0 |
| heldout | Material)  | 4 | 4 | 0 | 0 | 0 |
| heldout | Other-time | 2 | 2 | 0 | 0 | 0 |
| heldout | Other-weather | 2 | 1 | 1 | 0 | 0 |
| heldout | Other-writing&symbol | 2 | 0 | 2 | 0 | 0 |
| heldout | activity | 11 | 7 | 4 | 0 | 0 |
| heldout | animal | 10 | 9 | 1 | 0 | 0 |
| heldout | animal/human | 4 | 4 | 0 | 0 | 0 |
| heldout | attribute | 12 | 7 | 4 | 1 | 0 |
| heldout | attribute-style-artist/works | 2 | 0 | 0 | 2 | 0 |
| heldout | color | 6 | 6 | 0 | 0 | 0 |
| heldout | counting | 2 | 1 | 1 | 0 | 0 |
| heldout | food | 2 | 2 | 0 | 0 | 0 |
| heldout | human | 10 | 7 | 2 | 0 | 1 |
| heldout | location | 2 | 1 | 1 | 0 | 0 |
| heldout | material | 2 | 1 | 1 | 0 | 0 |
| heldout | object | 20 | 14 | 6 | 0 | 0 |
| heldout | other | 8 | 2 | 6 | 0 | 0 |
| heldout | other-perspective | 4 | 4 | 0 | 0 | 0 |

误接受指人工多数票为 FAIL、模型为 PASS；误拒绝相反。不确定与格式错误单列，同时在准确率中按未答对处理。类别样本数较小，表格用于定位问题，不代表稳定的类别能力排序。

## 能证明什么及下一步

本版把真实生成图、人工参考、指令分组隔离、两种评估协议、排序/相关性/细粒度指标和不确定性记录串起来。尚未形成微调后的模型改进闭环，也不涉及视频时序、跨帧主体一致性或生成模型训练。

本次保留结果已经公开；后续调参应把它视为已观察的开发材料，再从未使用的指令组建立新保留集。公开数据是否进入基础模型预训练未知，本实验不能消除预训练污染。下一步先依据开发错例提出校准或轻量后训练方案，再用新保留数据检验。

仓库只发布聚合指标、匿名样本 ID、模型回答和哈希，不上传原始图片、指令或完整人工标注。完整输入、预测、源码快照和 PyTorch 分数张量保存在本机 runs 目录。
