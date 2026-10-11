"""Export a completed local study into a portable, inspectable result bundle."""
import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/v0.2.0')
    args = parser.parse_args()
    run = args.run.resolve()
    if json.loads((run/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Only completed runs can be published')
    summary = json.loads((run/'summary.json').read_text())
    config = json.loads((run/'config.json').read_text())
    rows = [json.loads(line) for line in (run/'predictions.jsonl').read_text().splitlines()]
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for name in ['summary.json', 'predictions.jsonl', 'runtime.json']:
        shutil.copy2(run/name, out/name)
    model = config['model']
    provenance = {
        'run_id': run.name,
        'model_id': model['model_id'], 'revision': model['revision'],
        'generation': config['generation'], 'attention': config['attention'],
        'max_new_tokens': config['max_new_tokens'],
        'image_long_edge_limit': config['image_long_edge_limit'],
        'environment': json.loads((run/'environment.json').read_text()),
        'source_sha256': json.loads((run/'source-manifest.json').read_text())['source_sha256'],
        'predictions_sha256': hashlib.sha256((out/'predictions.jsonl').read_bytes()).hexdigest(),
        'scope': summary['scope'],
    }
    # System interpreter location is public; omit machine/platform string from published metadata.
    provenance['environment'].pop('platform', None)
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    runtime = json.loads((run/'runtime.json').read_text())
    lines = ['# 本机诊断结果', '',
             '系统 Python / PyTorch / {}，Qwen3-VL-2B-Instruct。{} 张程序绘制图、{} 条指令、{} 组协议，共 {} 次推理。仅开发诊断，无人工标签、无训练、无独立测试集。'.format(
                 runtime['device'], len({r['image'] for r in rows}), len({r['id'] for r in rows}),
                 len({r['group'] for r in rows}), len(rows)), '',
             '| 协议 | 正确/总数 | 格式错误 | 误接受 | 成对全对 | 耗时中位数/秒 |',
             '|---|---:|---:|---:|---:|---:|']
    for group, metrics in summary['groups'].items():
        subset = [r for r in rows if r['group'] == group]
        correct = sum(r['verdict'] == ('PASS' if r['expected'] else 'FAIL') for r in subset)
        lines.append('| {} | {}/{} | {} | {} | {}/{} | {:.2f} |'.format(
            group, correct, len(subset), sum(r['verdict']=='INVALID' for r in subset),
            metrics['false_accept_count'], metrics['both_pair_members_correct'], metrics['paired_images'],
            statistics.median(r['seconds'] for r in subset)))
    lines += ['', '简单全 PASS / 全 FAIL 基线均为 4/8。耗时含输入处理和生成，首题包括首次设备执行开销，不含模型加载；不同协议的输出长度不同。', '',
              '## 失败案例', '']
    for row in rows:
        if row['group'] in ['direct', 'decomposed'] and row['verdict'] != ('PASS' if row['expected'] else 'FAIL'):
            lines += ['- {} / {}：参考 {}，输出 {}；解析错误 {}。'.format(
                row['group'], row['id'], 'PASS' if row['expected'] else 'FAIL', row['verdict'], row['parse_error'])]
    lines += ['', '## 输入控制', '']
    for group, control in summary['controls'].items():
        lines += ['- {}：{}；与原图整体判断相同 {}/{}。不计算正确率，因为原图标签不适用于更换或移除图像后的输入。'.format(
            group, json.dumps(control['verdict_counts']), control['same_verdict_as_direct'], control['n'])]
    lines += ['', '无图时仍全部回答 PASS，说明本协议未能可靠让模型识别缺少视觉证据。真实使用必须在入口校验图片；不能把该模型直接当作可靠的自动裁判。错配图全部 FAIL 仅是输入敏感性观察，不能证明可靠 grounding。', '',
              '## 结论与下一步', '',
              '本次逐条件协议未优于整体判断：少返回一个数量条件，且把左右关系判断反。八例结果不能推广至真实生成图。保留失败输出，不调整标签；下一步扩充真实文生图与独立人工标注，按指令分组隔离开发/测试数据，再研究协议和校准。', '',
              '本目录 predictions.jsonl 保留全部原始回答。原始运行目录另存源码快照、输入哈希、设备信息和 PyTorch 混淆矩阵。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('REPORT_READY', out)


if __name__ == '__main__':
    main()
