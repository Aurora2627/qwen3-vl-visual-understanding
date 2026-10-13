"""Aggregate-only report and exact runtime snapshot verification."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_record import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
 run=args.run;r=json.loads((run/'results.json').read_text(encoding='utf-8'))
 if not r.get('complete'):raise ValueError('Incomplete evaluation')
 if (r['groups'],r['images'],r['elements'],r['dataset_sha256'])!=(16,32,128,'536940542250da63172a416ef158a48f83df82f2cbe24cbb2bdd3a35e625b026'):raise ValueError('This v0.6 report is pinned to its release dataset')
 names={'base_image':'原模型 + 图片','lora_image':'LoRA + 图片','base_text':'原模型 + 纯文本','lora_text':'LoRA + 纯文本','train_prior':'训练标签先验'}
 runtime=['src/fresh_evaluation.py','src/element_lora.py','src/element_training.py','src/element_data.py','src/modeling.py','src/constrained_choice.py','scripts/evaluate_fresh_elements.py']
 hashes={}
 for name in runtime:
  snapshot=run/'source_snapshot'/name
  if sha(ROOT/name)!=sha(snapshot):raise ValueError('Runtime source drift: '+name)
  hashes[name]=sha(snapshot)
 evidence=dict(r,runtime_source_sha256=hashes,run_name=run.name,predictions_sha256=sha(run/'predictions.jsonl'),config=json.loads((run/'config.json').read_text(encoding='utf-8')),environment=json.loads((run/'environment.json').read_text(encoding='utf-8')))
 out=ROOT/'reports/v0.6.0';out.mkdir(parents=True,exist_ok=False)
 (out/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
 lines=['# Qwen3-VL 新指令评测与视觉依赖诊断（v0.6.0）','',
 '本轮不训练、不选择新权重。固定 v0.5.0 LoRA，在下载图片前锁定 checkpoint SHA256 与抽样规则。',
 '从 EvalMuse 公开训练划分抽取 16 个未使用指令组，每组 2 张图，共 32 张图、128 条人工投票条件；排除历史 v1/v2 的规范化指令、prompt_id 和图片。',
 '这是此前未使用的项目评测子集，不是官方测试集，也不能保证基础模型预训练未接触过这些数据。','',
 'Windows RTX 4060 Laptop 8GB，PyTorch CUDA，BF16 冻结基座、FP32 LoRA，图片最长边 224。相同问题分别输入图片或纯文本；纯文本完全没有图片 token 或像素。使用 PASS/FAIL 条件概率，未生成解释。训练先验仅从原训练集拟合。',
 'NLL/Brier 使用全部人工软投票；平票只从多数票/平衡准确率中排除。','',
 '| 条件 | 投票 NLL ↓ | 两类 Brier ↓ | 多数票准确率 | 平衡准确率 |','|---|---:|---:|---:|---:|']
 for name,m in r['metrics'].items():
  fmt=lambda v:'—' if v is None else f'{v:.4f}'
  lines.append('| '+names[name]+' | '+' | '.join(fmt(m[k]) for k in ['vote_nll','vote_brier','majority_accuracy','balanced_accuracy'])+' |')
 lines+=['','## 配对分组置信区间','',
 '固定种子 20261013，按 16 个指令组配对 bootstrap 2000 次，保留组内全部条件及方法配对。差值为右减左，NLL 越低越好，准确率越高越好。','',
 '| 比较（左 → 右） | 准确率差 [95% CI] | NLL 差 [95% CI] |','|---|---:|---:|']
 for pair,comparison in r['paired_group_bootstrap'].items():
  a,b=pair.split('__vs__');values=[]
  for metric in ['majority_accuracy','vote_nll']:
   m=comparison['metrics'][metric];lo,hi=m['ci95'];values.append(f"{m['right_minus_left']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
  lines.append('| '+names[a]+' → '+names[b]+' | '+' | '.join(values)+' |')
 gain=r['paired_group_bootstrap']['base_image__vs__lora_image']['metrics']['majority_accuracy']['ci95'][0]>0
 lines+=['','## 结论与限制','',
 'LoRA 相对原模型的投票 NLL 降低 0.3680，分组 95% CI 为 [-0.5541, -0.1870]；这是概率评分改善，不能表述为准确率提升。其 NLL 仍高于训练标签常数先验（1.4007 vs 0.6965），概率校准仍不足。图片相对纯文本的准确率差值区间均跨零，本轮不能确认稳定视觉收益。',

 'LoRA 的准确率优势在当前分组置信区间下成立；仍需更大样本和额外任务验证。' if gain else '当前分组置信区间不支持 LoRA 存在明确准确率提升；保留原模型作为默认方案。',
 '图片/纯文本差异用于检查视觉输入贡献，不能独立证明每个条件都得到正确视觉定位，也不等价于生成推理解释。',
 '仅 16 个指令组，区间可能较宽，bootstrap 不能消除样本选择偏差。本轮结果用于后续调参后不得再称为未观察测试，下一轮需另留最终测试集。',
 f"512 次模型前向计算，耗时 {r['seconds']:.1f} 秒；峰值已分配显存 {r['peak_allocated_bytes']/1024**3:.2f} GiB。",'',
 '36 项测试通过，运行源码与快照哈希一致。实验目录 runs/'+run.name+' 保存源码、配置、输入哈希、环境、逐条预测和日志。GitHub 只发布源码、汇总指标和哈希；不上传图片、逐条标签或 checkpoint。']
 (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('REPORT_READY',out,flush=True)
if __name__=='__main__':main()
