"""Export aggregate pilot evidence without publishing images, labels or weights."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from element_data import read_jsonl
from element_training import binary_metrics
from run_record import sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reload-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run,reload_run=args.run,args.reload_run
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    reload=json.loads((reload_run/'summary.json').read_text(encoding='utf-8'))
    if not reload.get('reload_passed') or summary['optimizer_updates']<=0:
        raise ValueError('Training/reload not complete')
    source=json.loads((run/'source-manifest.json').read_text(encoding='utf-8'))
    snapshots={k.replace('\\','/'):v for k,v in source['source_sha256'].items()}
    runtime=['src/modeling.py','src/element_data.py','src/element_lora.py','src/element_training.py',
             'src/constrained_choice.py','scripts/train_element_lora.py']
    hashes={}
    for file in runtime:
        if sha(ROOT/file)!=snapshots[file]:
            raise ValueError('Runtime source changed since training: '+file)
        hashes[file]=hashlib.sha256((ROOT/file).read_text(encoding='utf-8-sig').replace('\r\n','\n').encode()).hexdigest()
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    architecture=json.loads((run/'architecture.json').read_text(encoding='utf-8'))
    data=json.loads((Path(config['dataset'])/'dataset-record.json').read_text(encoding='utf-8'))
    train=read_jsonl(Path(config['dataset'])/'train.jsonl')
    predictions=json.loads((run/'selected-predictions.json').read_text(encoding='utf-8'))
    prior=sum(r['p_pass'] for r in train)/len(train)
    prior_metrics=binary_metrics([dict(r,p_pass_model=prior) for r in predictions])
    evidence={'version':'0.5.0','scope':summary['scope'],'independent_test_available':False,
        'algorithm_improvement_claim':False,'run':run.name,'reload_run':reload_run.name,
        'dataset':data['splits'],'training':summary,'architecture':architecture,
        'training_label_prior':prior,'constant_train_prior_validation':prior_metrics,
        'fresh_process_reload':reload,'checkpoint_sha256':sha(run/'adapter.pt'),
        'runtime_source_sha256_utf8_lf':hashes,
        'notes':['31 tests passed for the training source',
          'Historical holdout results were previously observed; they are quarantined',
          'New unseen-data download blocked by Windows huggingface.co DNS resolution',
          'Original FP16 full-data run failed nonfinite gradient; preserved, not counted as successful',
          'BF16 run is separate; no rerun overwrites the original evidence',
          'Binary PASS/FAIL differs from the older PASS/FAIL/UNCERTAIN protocol',
          'Public training-data pilot, not the official EvalMuse benchmark']}
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    (out/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    before,after=summary['baseline'],summary['selected_validation']
    lines=['# Qwen3-VL 逐条件 LoRA 开发实验（v0.5.0）','',
      '本轮已实际更新 Qwen3-VL 的 LoRA 参数，验证训练、评测与新进程重载。以下结果属于已经观察过的开发数据，不能证明独立测试效果提升。','',
      '## 方法与数据','',
      '官方 Qwen3-VL-2B-Instruct 固定权重；Windows RTX 4060 Laptop 8GB，PyTorch CUDA。冻结视觉编码器、语言模型基座和输出头，仅训练语言模型最后 4 层注意力 q/v 的 rank=4、alpha=8 LoRA，共 114,688 个可训练参数。',
      '按文字指令分组，将既有 32 张开发图固定为 24 张训练图（12 指令组、128 条元素）和 8 张验证图（4 指令组、28 条元素）。图片哈希、指令组和指令 ID 无跨组重叠。旧保留集不参与训练或验证。',
      '每条元素单独判断 PASS/FAIL，使用人工二值投票比例作为软目标，在完整词表上监督答案的首个 token；图片及问题 token 不计损失。评测使用两个答案 token 的条件概率。训练不生成或监督理由，也不训练图片生成器。',
      'BF16 冻结基座、FP32 LoRA，梯度检查点、梯度累积 4、AdamW 学习率 1e-4、1 epoch、32 次优化更新。输入图片先缩略至最长边 224，再由官方 processor 处理；因此不能与旧版 448 输入/不同协议的指标直接比较。','',
      '## 验证结果','',
      '| 模型 | 人工投票 NLL ↓ | 人工投票 Brier ↓ | 多数票准确率 | 平衡准确率 |',
      '|---|---:|---:|---:|---:|']
    for name,metrics in [('冻结基座',before),('LoRA（验证 NLL 选取）',after),('仅用训练标签先验的常数基线',prior_metrics)]:
        values=[metrics[k] for k in ['vote_nll','vote_brier','majority_accuracy','balanced_accuracy']]
        lines.append('| '+name+' | '+' | '.join('—' if v is None else format(v,'.4f') for v in values)+' |')
    lines+=['','NLL/Brier 使用所有 28 条元素的软投票目标；多数票标签恰好平局的 2 条只从准确率和平衡准确率中排除。Brier 是对人工二值投票的期望两类 Brier，不能与旧版五档整体评分 Brier 横向比较。','',
      '本轮 LoRA 的验证 NLL、Brier 和准确率均未改善；冻结基座优于这个训练 checkpoint，因此当前不推荐用该 adapter 替换基座。验证 NLL 只在本轮训练 epoch 之间选取，本轮仅 1 epoch。','',
      '## 工程验证与限制','',
      '31 项测试通过；8 个 LoRA B 矩阵均非零，冻结基座无梯度。适配器保存为原生 PyTorch .pt；同进程和新进程重载的答案 logits 最大差均为 '+str(reload['max_logit_difference'])+'。',
      '完整训练峰值 PyTorch 已分配显存 '+format(summary['peak_allocated_bytes']/1024**3,'.2f')+' GiB、保留显存 '+format(summary['peak_reserved_bytes']/1024**3,'.2f')+' GiB。这是当前小模型、短输入配置的实测值。',
      '最初 FP16 全量开发训练因非有限梯度停止，失败日志与源码保留；之后单独使用 BF16 重跑。',
      '验证集只有 4 个指令组，且属于此前观察过的开发数据。Windows 无法解析 huggingface.co，新的未观察图片尚未下载；本版没有独立测试指标，不宣称算法泛化改进。',
      '下一步：网络恢复后锁定排除全部历史指令的新数据；冻结本版 checkpoint 和评测协议，比较基座、LoRA、训练先验基线及无图控制，再按指令组计算配对置信区间。','',
      '源码、输入哈希、配置、日志、逐条预测及 .pt checkpoint 保存在 Windows 每次实验目录；GitHub 仅发布源码和汇总证据。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('PILOT_REPORT_EXPORTED',json.dumps({'baseline':before,'adapter':after,'train_prior':prior_metrics,
        'peak_allocated_GiB':summary['peak_allocated_bytes']/1024**3}),flush=True)

if __name__=='__main__':main()
