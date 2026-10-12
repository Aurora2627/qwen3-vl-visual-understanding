"""Verify and publish fixed-choice/calibration results without source data."""
import argparse
import hashlib
import json
import shutil
import statistics
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_row(row):
    allowed = ['id','split','protocol','image_sha256','raw','score','valid','error',
        'seconds','input_tokens','output_tokens','model_calls','intentional_fixed_length',
        'decision','choices','token_ids','candidate_logits','candidate_probabilities',
        'candidate_entropy','candidate_margin','candidate_mass_full_vocabulary',
        'unconstrained_top_token_id','generation_length_policy','reached_token_limit']
    result = {k:row[k] for k in allowed if k in row}
    if 'raw_choices' in row:
        result['raw_choices'] = [{k:answer[k] for k in allowed+['index'] if k in answer}
                                 for answer in row['raw_choices']]
    if 'parsed' in row:
        result['parsed'] = {'checks':[{k:check[k] for k in ['index','verdict']}
                                     for check in row['parsed']['checks']]}
    return result


def verify_run(run):
    from temperature_calibration import vote_distribution, calibration_metrics
    if json.loads((run/'status.json').read_text())['status']!='complete':
        raise ValueError('Incomplete experiment')
    provenance = json.loads((run/'source-manifest.json').read_text())
    if not provenance['source_sha256'] or not provenance['input_sha256']:
        raise ValueError('Missing provenance')
    for name,value in provenance['source_sha256'].items():
        if sha(run/'source_snapshot'/name)!=value:
            raise ValueError('Source snapshot changed')
    for index,(name,value) in enumerate(provenance['input_sha256'].items()):
        if sha(run/'input_manifests'/(str(index)+'-'+Path(name).name))!=value:
            raise ValueError('Input snapshot changed')
    manifest = run/'input_manifests/0-manifest.jsonl'
    inputs = [json.loads(line) for line in manifest.read_text().splitlines()]
    by_id = {r['id']:r for r in inputs}
    rows = [json.loads(line) for line in (run/'predictions.jsonl').read_text().splitlines()]
    protocols = ['overall_free','overall_fixed','elements_fixed']
    keys = [(r['id'],r['protocol']) for r in rows]
    expected = {(r['id'],p) for r in inputs for p in protocols}
    if len(keys)!=len(expected) or set(keys)!=expected or len(inputs)!=64:
        raise ValueError('Incomplete or duplicate image/protocol records')
    for row in rows:
        if any(row[k]!=v for k,v in by_id[row['id']].items()):
            raise ValueError('Prediction input identity changed')
        if row['protocol']=='overall_free':
            valid = row['raw'].strip() in ['1','2','3','4','5']
            if row['valid']!=valid or row['score']!=(int(row['raw'].strip()) if valid else None):
                raise ValueError('Free score parsing mismatch')
        else:
            answers = [row] if row['protocol']=='overall_fixed' else row['raw_choices']
            choices = ['1','2','3','4','5'] if row['protocol']=='overall_fixed' else ['0','1','2']
            for answer in answers:
                if answer['choices']!=choices or answer['decision'] not in choices or answer['raw']!=answer['decision']:
                    raise ValueError('Choice identity mismatch')
                if answer['output_tokens']!=1:
                    raise ValueError('Expected exactly one complete decision token')
                probability = torch.tensor(answer['candidate_logits'],dtype=torch.float64).softmax(-1)
                saved = torch.tensor(answer['candidate_probabilities'],dtype=torch.float64)
                if not torch.allclose(probability,saved,atol=5e-6,rtol=5e-6):
                    raise ValueError('Candidate probability mismatch')
            if row['protocol']=='overall_fixed' and row['score']!=int(row['decision']):
                raise ValueError('Fixed score mismatch')
            if row['protocol']=='elements_fixed':
                verdicts = [{'0':'FAIL','1':'PASS','2':'UNCERTAIN'}[a['decision']] for a in answers]
                expected_score = None if 'UNCERTAIN' in verdicts else verdicts.count('PASS')/len(verdicts)
                if [c['verdict'] for c in row['parsed']['checks']]!=verdicts or row['score']!=expected_score:
                    raise ValueError('Element aggregation mismatch')
    summary = json.loads((run/'summary.json').read_text())
    if summary['image_protocol_records']!=len(rows) or summary['actual_generation_calls']!=sum(r['model_calls'] for r in rows):
        raise ValueError('Generation count mismatch')
    fitted = torch.load(run/'calibration.pt',weights_only=True)
    development = [r for r in rows if r['split']=='development' and r['protocol']=='overall_fixed']
    heldout = [r for r in rows if r['split']=='heldout' and r['protocol']=='overall_fixed']
    if fitted['training_ids']!=[r['id'] for r in development]:
        raise ValueError('Calibration training identities mismatch')
    if set(fitted['training_prompt_groups'])!={r['prompt_group'] for r in development} or set(fitted['training_prompt_groups']) & {r['prompt_group'] for r in heldout}:
        raise ValueError('Calibration group leakage')
    temperature = float(fitted['log_temperature'].exp())
    if abs(temperature-summary['temperature'])>1e-9:
        raise ValueError('Temperature checkpoint mismatch')
    for split,subset in [('development',development),('heldout',heldout)]:
        saved = torch.load(run/(split+'-ordinal-scores.pt'),weights_only=True)
        logits = torch.tensor([r['candidate_logits'] for r in subset],dtype=torch.float64)
        targets = torch.stack([vote_distribution(r['human_overall']) for r in subset])
        if not torch.equal(saved['candidate_logits'],logits) or not torch.equal(saved['human_vote_distributions'],targets):
            raise ValueError('Ordinal score tensors mismatch')
        for name,value in [('expected',1.),('calibrated',temperature)]:
            metrics = calibration_metrics(logits,targets,value)
            if metrics!=summary['calibration'][split][name]:
                raise ValueError('Calibration metrics mismatch')
    return rows,summary,provenance


def number(value):
    return '未定义' if value is None else '{:.3f}'.format(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/v0.4.0')
    args = parser.parse_args()
    run = args.run.resolve()
    rows,summary,source = verify_run(run)
    config = json.loads((run/'config.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['summary.json','runtime.json']:
        shutil.copy2(run/name,args.output/name)
    (args.output/'model-outputs.jsonl').write_text(''.join(json.dumps(public_row(r),ensure_ascii=False)+'\n' for r in rows))
    record = {'status':'passed','run_id':run.name,'image_protocol_records':len(rows),
        'actual_generation_calls':summary['actual_generation_calls'],'model_id':config['model']['model_id'],
        'model_revision':config['model']['revision'],'source_sha256':source['source_sha256'],
        'input_sha256':{str(i)+'-'+Path(k).name:v for i,(k,v) in enumerate(source['input_sha256'].items())},
        'public_outputs_sha256':sha(args.output/'model-outputs.jsonl'),
        'export_source_sha256':sha(Path(__file__).resolve()),
        'environment':json.loads((run/'environment.json').read_text()),
        'checks':['source/input hashes','all image/protocol identities','choice probabilities',
                  'score aggregation','calibration training-group isolation','PyTorch checkpoints and metrics'],
        'method_scope':config['method_scope']}
    (args.output/'validation.json').write_text(json.dumps(record,indent=2))
    lines = ['# 固定选项评分与温度校准', '',
        '64 张真实生成图；32 张沿用 v1 开发图片，32 张来自新的 16 条保留指令，排除 v1 全部 32 条指令。使用 [EvalMuse 公开训练标注](https://huggingface.co/datasets/DY-Evalab/EvalMuse)，非官方测试榜单。基础模型是否预训练见过公开数据未知。', '',
        '## 方法与对照', '',
        '- overall_free / overall_fixed 使用完全相同的短提示词，前者 greedy 最多 16 token，后者通过 PyTorch logits masking 只允许一个 1–5 的数字 token。格式正确由约束保证，不等于视觉判断正确。',
        '- elements_fixed 对每个要素分别输出 0/1/2（FAIL/PASS/UNCERTAIN），保留完整指令上下文。任一项不确定则整图不计分，不将不确定当 FAIL。输入要素来自数据集，未训练自动拆分模型。',
        '- overall_expected 使用候选分数分布的期望；overall_calibrated 在此基础上，用开发图片的人工评分分布优化一个温度参数（200 步 Adam，学习率 0.05，T 范围 0.05–20）。训练不使用保留标签，也不更新 Qwen 权重。',
        '- v0.3 的长 JSON 提示词是历史诊断；由于提示词不同，不能把 v0.3 到 v0.4 的差异全部归因于解码约束。', '',
        '## 人工一致性', '',
        '| 数据/协议 | 有分数/总数 | SRCC | 分组 bootstrap 95% 区间 | PLCC | 排序正确/非平局 |',
        '|---|---:|---:|---|---:|---:|']
    for key,metrics in summary['groups'].items():
        ci = metrics['srcc_95_interval']
        lines.append('| {} | {}/{} | {} | {} | {} | {}/{} |'.format(key,metrics['scored_images'],metrics['n_images'],
            number(metrics['srcc_scored_only']),'未定义' if ci is None else '[{}, {}]'.format(*[number(v) for v in ci]),
            number(metrics['plcc_scored_only']),metrics['pair_correct'],metrics['strict_human_pairs']))
    lines += ['', '相关性只计算有分数输出，必须结合覆盖率。人工排序平局排除；模型平局、不计分与格式错误算错。随机非平局排序基线 50%。每部分只有 16 个独立指令组，区间较宽，不能据此宣称显著提升。', '',
        '## 校准（越低越好）', '',
        '温度 T = '+number(summary['temperature'])+'。NLL 是相对于人工评分票分布的交叉熵；Brier 是相对于该分布的平方误差。正温度不改变最大候选类别，但可能改变期望分数和概率分布。', '',
        '| 数据 | 原始 NLL | 校准 NLL | 原始 Brier | 校准 Brier |',
        '|---|---:|---:|---:|---:|']
    for split,metrics in summary['calibration'].items():
        before,after = metrics['expected'],metrics['calibrated']
        lines.append('| {} | {} | {} | {} | {} |'.format(split,number(before['nll']),number(after['nll']),
            number(before['soft_target_brier']),number(after['soft_target_brier'])))
    held = summary['calibration']['heldout']
    lines += ['', '保留集 NLL '+('下降' if held['calibrated']['nll']<held['expected']['nll'] else '未下降')+
              '；这是本次固定子集的结果，不代表总体提升。开发集下降本身不能证明泛化。', '',
        '## 输出可靠性与耗时', '',
        '| 数据/协议 | 格式错误 | 耗时中位数/秒 | 细粒度准确率 | 全 PASS 基线 | 平衡准确率 |',
        '|---|---:|---:|---:|---:|---:|']
    for split in ['development','heldout']:
        for protocol in ['overall_free','overall_fixed','elements_fixed']:
            subset = [r for r in rows if r['split']==split and r['protocol']==protocol]
            metrics = summary['groups'][split+'/'+protocol]
            fine = metrics.get('elements')
            lines.append('| {} | {} | {:.2f} | {} | {} | {} |'.format(split+'/'+protocol,metrics['invalid_outputs'],
                statistics.median(r['seconds'] for r in subset),'—' if fine is None else number(fine['accuracy_all']),
                '—' if fine is None else number(fine['always_pass_accuracy']),
                '—' if fine is None else number(fine['balanced_accuracy_all'])))
    fixed = [r for r in rows if r['protocol']=='overall_fixed']
    lines += ['', 'overall_fixed 候选数字在完整词表中的概率质量中位数为 '+number(statistics.median(r['candidate_mass_full_vocabulary'] for r in fixed))+
        '。候选集合内 softmax 会重新归一化，即使候选的原始总概率很低也能给出高相对概率，因此不能解释为真实正确置信度。', '',
        '耗时包括预处理及同步，不含加载；固定选项只生成一个完整决策 token，不能把达到该固定长度当作解释被截断。逐要素协议会重复处理图片，模型调用数与整体协议不同。运行顺序与输出长度也不同，速度仅作本机工程观察。', '',
        '本次共 '+str(summary['actual_generation_calls'])+' 次模型生成，形成 '+str(len(rows))+' 条图片/协议记录；期望评分与校准评分从同一批 logits 导出，不增加模型生成。', '',
        '## 边界', '',
        '本版展示受约束评分、对照实验、人工分歧处理与开发集校准；没有进行 Qwen LoRA/SFT、偏好优化、生成模型训练或视频时序评估。新保留结果公开后，后续调参必须换新的未观察指令。源码、输入快照、人工标签、预测和 PyTorch 校准参数保留本机，仓库仅导出匿名模型输出与汇总，不上传原始数据。']
    (args.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('CHOICE_REPORT_VERIFIED_AND_EXPORTED',args.output,flush=True)


if __name__ == '__main__':
    main()
