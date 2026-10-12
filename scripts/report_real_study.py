"""Publish aggregate pilot results and model outputs, excluding third-party data."""
import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def number(value):
    return '未定义' if value is None else '{:.3f}'.format(value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/v0.3.0')
    args = parser.parse_args()
    run = args.run.resolve()
    if json.loads((run/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Run is incomplete')
    summary = json.loads((run/'summary.json').read_text())
    config = json.loads((run/'config.json').read_text())
    rows = [json.loads(line) for line in (run/'predictions.jsonl').read_text().splitlines()]
    if set(summary['groups']) != {'development/overall','development/elements','heldout/overall','heldout/elements'}:
        raise ValueError('This release report requires both splits and protocols')
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['summary.json','runtime.json']:
        shutil.copy2(run/name,args.output/name)
    allowed = ['id','split','protocol','image_sha256','raw','parsed','error','valid','score',
               'seconds','input_tokens','output_tokens','reached_token_limit']
    (args.output/'model-outputs.jsonl').write_text(''.join(json.dumps({k:r[k] for k in allowed},ensure_ascii=False)+'\n' for r in rows))
    source = json.loads((run/'source-manifest.json').read_text())
    environment = json.loads((run/'environment.json').read_text())
    provenance = {'run_id':run.name,'model_id':config['model']['model_id'],'model_revision':config['model']['revision'],
        'source_sha256':source['source_sha256'],
        'input_sha256':{Path(k).name:v for k,v in source['input_sha256'].items()},
        'packages':environment['packages'],'python':environment['python'],'executable':environment['executable'],
        'max_new_tokens':config['max_new_tokens'],'device':config['device'],
        'protocol_policy':config['protocol_policy'],
        'outputs_sha256':hashlib.sha256((args.output/'model-outputs.jsonl').read_bytes()).hexdigest()}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    lines = ['# EvalMuse 真实生成图试验', '',
        '来源：[EvalMuse 官方数据](https://huggingface.co/datasets/DY-Evalab/EvalMuse)。使用公开训练部分中的固定子集，非官方测试集、非完整榜单复现。64 张真实文生图、32 条指令，每条两张图；开发与保留各 16 条指令，按规范化文本隔离，并检查指令 ID 和图片哈希无交叉。', '',
        '使用公开人类评分，无新增人工标注。整体参考是 1–5 分的人工均值；细粒度参考是每项二元标注的严格多数票。标注置信度信息保留，没有按评分高低或模型效果挑样本。', '',
        '## 方法', '',
        '- overall：Qwen3-VL 直接给 1–5 分与证据。',
        '- elements：输入数据集提供的内容要素（不是人工正确性标签），输出逐项 PASS/FAIL/UNCERTAIN。所有项可判断时，以 PASS 比例作为图片分数；任一项不确定则整体不计分。没有自动生成内容拆分，也没有训练评估器。',
        '- 512 个最大输出 token、448 长边、FP16 MPS、greedy、eager attention。先固定协议，随后同一次运行执行开发与保留数据，不根据保留结果改提示词。', '',
        '## 整体分数与人工判断', '',
        '| 数据/协议 | 有分数/总数 | SRCC | 按指令重采样的 95% 区间 | PLCC | 非平局排序正确/总数 |',
        '|---|---:|---:|---|---:|---:|']
    for key,metrics in summary['groups'].items():
        ci = metrics['srcc_95_interval']
        lines.append('| {} | {}/{} | {} | {} | {} | {}/{} |'.format(key,metrics['scored_images'],metrics['n_images'],
            number(metrics['srcc_scored_only']), '未定义' if ci is None else '[{}, {}]'.format(*[number(v) for v in ci]),
            number(metrics['plcc_scored_only']),metrics['pair_correct'],metrics['strict_human_pairs']))
    lines += ['', 'SRCC/PLCC 只在有分数样本上计算，必须和覆盖率一起读。相关性衡量与人类评分的一致性，不是准确率。排序排除人工均分平局；模型平局、不计分、格式错误算错。随机非平局排序基线为 50%；常数分数的相关性因零方差而未定义。每个数据部分仅 16 个独立指令组，区间不稳定，不能据此宣称方法显著优于 baseline。', '',
        '## 细粒度与输出可靠性', '',
        '| 数据/协议 | 格式错误 | 输出截断 | 耗时中位数/秒 | 细粒度正确/总数 | 人工有分歧的要素 |',
        '|---|---:|---:|---:|---:|---:|']
    for key,metrics in summary['groups'].items():
        split,protocol = key.split('/')
        subset = [r for r in rows if r['split']==split and r['protocol']==protocol]
        fine = metrics.get('elements')
        lines.append('| {} | {} | {} | {:.2f} | {} | {} |'.format(key,metrics['invalid_outputs'],
            sum(r['reached_token_limit'] for r in subset),statistics.median(r['seconds'] for r in subset),
            '—' if not fine else '{}/{}'.format(fine['correct'],fine['n']),
            '—' if not fine else str(fine['human_disagreement_elements'])))
    lines += ['', '| 数据/逐项协议 | 细粒度准确率 | 全 PASS 基线 | 全 FAIL 基线 | 平衡准确率 |',
              '|---|---:|---:|---:|---:|']
    for key,metrics in summary['groups'].items():
        fine = metrics.get('elements')
        if fine:
            lines.append('| {} | {} | {} | {} | {} |'.format(key,number(fine['accuracy_all']),
                number(fine['always_pass_accuracy']),number(fine['always_fail_accuracy']),number(fine['balanced_accuracy_all'])))
    lines += ['', '细粒度格式错误/不确定算错，人工票数平局不参与准确率。全 PASS/FAIL 基线揭示标签偏斜；平衡准确率取正负类召回均值，不确定与格式错误均为对应类未答对。速度包含输入处理和同步，不含模型加载；两个协议输出长度不同，不能当作严格速度基准。', '',
        '### 按要素类别诊断', '',
        '| 数据 | 类别 | 要素数 | 正确 | 误接受 | 误拒绝 | 不确定/格式错误 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for key,metrics in summary['groups'].items():
        fine = metrics.get('elements')
        if fine:
            for category,counts in sorted(fine['categories'].items()):
                lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(key.split('/')[0],category,
                    counts['n'],counts['correct'],counts['false_accept'],counts['false_reject'],counts['unanswered']))
    lines += ['', '误接受指人工多数票为 FAIL、模型为 PASS；误拒绝相反。不确定与格式错误单列，同时在准确率中按未答对处理。类别样本数较小，表格用于定位问题，不代表稳定的类别能力排序。', '',
        '## 能证明什么及下一步', '',
        '本版把真实生成图、人工参考、指令分组隔离、两种评估协议、排序/相关性/细粒度指标和不确定性记录串起来。尚未形成微调后的模型改进闭环，也不涉及视频时序、跨帧主体一致性或生成模型训练。', '',
        '本次保留结果已经公开；后续调参应把它视为已观察的开发材料，再从未使用的指令组建立新保留集。公开数据是否进入基础模型预训练未知，本实验不能消除预训练污染。下一步先依据开发错例提出校准或轻量后训练方案，再用新保留数据检验。', '',
        '仓库只发布聚合指标、匿名样本 ID、模型回答和哈希，不上传原始图片、指令或完整人工标注。完整输入、预测、源码快照和 PyTorch 分数张量保存在本机 runs 目录。']
    (args.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('REAL_REPORT_READY',args.output,flush=True)

if __name__ == '__main__':
    main()
