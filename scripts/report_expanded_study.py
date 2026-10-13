"""Aggregate report for the preregistered expanded-data loss comparison."""
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_record import sha
from element_training import binary_metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);args=p.parse_args()
 study=args.study;result=json.loads((study/'completed.json').read_text(encoding='utf-8'))
 if not result['complete']:raise ValueError('Incomplete study')
 models=result['models'];final=Path(result['final_run']);test=result['final_results'];selected=result['selection']
 runtime=['scripts/train_element_lora.py','src/element_training.py','src/element_lora.py','src/modeling.py','src/element_data.py','src/constrained_choice.py']
 hashes={}
 for name in runtime:
  expected=sha(ROOT/name)
  for model in models.values():
   if sha(Path(model['run'])/'source_snapshot'/name)!=expected:raise ValueError('Training source drift: '+name)
  if sha(final/'source_snapshot'/name)!=expected:raise ValueError('Evaluation source drift: '+name)
  hashes[name]=expected
 for name in ['scripts/run_expanded_study.py','scripts/prepare_element_corpus.py','src/element_diagnostics.py']:
  if sha(ROOT/name)!=sha(study/'source_snapshot'/name):raise ValueError('Study source drift: '+name)
  hashes[name]=sha(ROOT/name)
 reload_summaries={}
 stamp=study.name.replace('expanded-study-','')
 for mode,model in models.items():
  reload_run=ROOT/'runs'/('expanded-reload-'+mode+'-'+stamp)
  check=json.loads((reload_run/'summary.json').read_text(encoding='utf-8'))
  if not check.get('reload_passed') or check['max_logit_difference']>.002:raise ValueError('Fresh reload failed')
  reload_summaries[mode]=check
 a=models['full-vocabulary']['summary']['baseline'];b=models['binary-conditional']['summary']['baseline']
 for k in ['vote_nll','vote_brier','majority_accuracy','balanced_accuracy']:
  if a[k]!=b[k] and (a[k] is None or b[k] is None or not math.isclose(a[k],b[k],abs_tol=1e-6)):raise ValueError('Mismatched frozen development baselines')
 predictions=[json.loads(line) for line in (final/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
 if len(predictions)!=test['elements']*5:raise ValueError('Prediction count mismatch')
 for condition,metrics in test['metrics'].items():
  rows=[r for r in predictions if r['condition']==condition]
  recalculated=binary_metrics(rows)
  if len(rows)!=test['elements'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Incomplete paired condition')
  for key,value in metrics.items():
   if isinstance(value,float):
    if not math.isclose(value,recalculated[key],abs_tol=1e-12):raise ValueError('Metric mismatch')
   elif value!=recalculated[key]:raise ValueError('Count mismatch')
 from datetime import datetime,timezone
 active=json.loads((study/'active-step.json').read_text(encoding='utf-8'))
 if active['stage']!='sealed-final-evaluation' or selected['locked_before_test']>active['started']:raise ValueError('Selection was not locked before test start')
 if sha(Path(result['dataset'])/'sealed_test.jsonl')!=selected['sealed_test_sha256']:raise ValueError('Sealed test changed')
 if sha(Path(result['dataset'])/'final-evaluation/elements.jsonl')!=selected['sealed_test_sha256']:raise ValueError('Evaluated test differs from sealed partition')
 chosen=Path(selected['checkpoint'])
 if sha(chosen)!=selected['checkpoint_sha256']:raise ValueError('Selected checkpoint drift')
 out=ROOT/'reports/v0.7.0';out.mkdir(parents=True,exist_ok=False)
 old_diag=json.loads((ROOT/'runs/element-diagnostics-20261013-173422/diagnostics.json').read_text(encoding='utf-8'))
 calibration_dir=ROOT/'runs/expanded-temperature-20261013-185835'
 calibration_lock=json.loads((calibration_dir/'calibration-lock.json').read_text(encoding='utf-8'))
 if calibration_lock['locked_at']>active['started']:raise ValueError('Temperature not locked before test')
 for filename in ['src/element_vote_temperature.py','scripts/fit_element_temperature.py']:
  if sha(ROOT/filename)!=sha(calibration_dir/'source_snapshot'/filename):raise ValueError('Calibration source drift')
  hashes[filename]=sha(ROOT/filename)
 from element_vote_temperature import apply_vote_temperature
 from fresh_evaluation import paired_bootstrap
 calibrated=apply_vote_temperature([r for r in predictions if r['condition']=='base_image'],calibration_lock['fit']['temperature'])
 calibrated_metrics=binary_metrics(calibrated)
 if calibrated_metrics['majority_accuracy']!=test['metrics']['base_image']['majority_accuracy']:raise ValueError('Positive temperature changed binary decisions')
 calibration_comparison=paired_bootstrap(calibrated,[r for r in predictions if r['condition']=='lora_image'])
 calibration_baseline={'lock':calibration_lock,'test_metrics':calibrated_metrics,'calibrated_base_vs_adapter':calibration_comparison,
  'scope':'Supplemental comparator registered and fitted on validation before sealed-test execution; does not affect adapter selection'}
 observed={'source_prediction_sha256':old_diag['source_prediction_sha256'],'conditions':{k:{'confusion':v['confusion'],'wrong_at_confidence_ge_0_9':v['wrong_at_confidence_ge_0_9'],'bins':v['bins']} for k,v in old_diag['conditions'].items()}}
 evidence={'fresh_process_reloads':reload_summaries,'calibration_baseline':calibration_baseline,'observed_v0_6_diagnostics':observed,'study':result,'runtime_source_sha256':hashes,'predictions_sha256':sha(final/'predictions.jsonl'),
  'matched_development_baselines':True,'test_prediction_records':len(predictions),'tests':43,
  'environment':json.loads((final/'environment.json').read_text(encoding='utf-8'))}
 (out/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
 counts=result['dataset_record']['split_counts'];names={'base_image':'原模型 + 图片','lora_image':'选定 LoRA + 图片','base_text':'原模型 + 纯文本','lora_text':'选定 LoRA + 纯文本','train_prior':'训练标签先验'}
 lines=['# 扩充数据、训练损失对照与封存测试（v0.7.0）','',
 '## 问题与实验约束','',
 '旧 v0.6 数据中，原模型图片条件有 35 条置信度至少 90% 的错误预测；该诊断数据已被观察，本轮不再用作测试。',
 '新增 80 个未使用文字指令组、160 张图，排除全部历史文字指令、prompt_id、source path 和跨划分图片哈希。按预先固定哈希顺序分组划分：','',
 '| 划分 | 指令组 | 图片 | 条件 |','|---|---:|---:|---:|']
 for name in ['train','validation','sealed_test']:
  c=counts[name];lines.append(f"| {name} | {c['groups']} | {c['images']} | {c['elements']} |")
 lines+=['','数据来自 EvalMuse 公开 train 划分；这是项目内未使用指令的评测，不是官方 benchmark，不保证排除基础模型预训练污染。',
 'Windows RTX4060 Laptop 8GB，Qwen3-VL-2B-Instruct 固定基座、BF16，FP32 LoRA；冻结视觉和基座，仅训练语言末4层q/v，rank4、alpha8、114688参数；最长边224、seed20261013、lr5e-5、累积4、各1epoch。',
 '只比较预先固定的两种损失：全词表首个答案token软投票交叉熵，以及仅PASS/FAIL上的条件二分类软投票交叉熵。后者直接对应评测时的两选项概率。数据、初始化种子和超参数相同。',
 '## 开发验证集：用于选模','',
 '| 方法 | NLL ↓ | Brier ↓ | 准确率 | 平衡准确率 |','|---|---:|---:|---:|---:|']
 def metric_row(name,m):
  f=lambda v:'—' if v is None else f'{v:.4f}'
  return '| '+name+' | '+' | '.join(f(m[k]) for k in ['vote_nll','vote_brier','majority_accuracy','balanced_accuracy'])+' |'
 lines.append(metric_row('冻结基座',a))
 for mode,m in models.items():lines.append(metric_row(mode,m['summary']['selected_validation']))
 lines+=['',f"按开发验证 NLL 选定 {selected['chosen_loss_mode']}，checkpoint SHA256 在首次封存测试前锁定。两种损失的基座验证结果核对一致。没有用测试结果选方法或重新训练。",'',
 '## 封存测试：选定权重后运行一次','',
 '| 条件 | NLL ↓ | Brier ↓ | 准确率 | 平衡准确率 |','|---|---:|---:|---:|---:|']
 for name,m in test['metrics'].items():lines.append(metric_row(names[name],m))
 lines.append(metric_row('原模型 + 验证集温度校准',calibrated_metrics))
 lines+=['',f"补充校准基线在最终测试前注册，温度只用160条开发验证条件拟合并锁定为 {calibration_lock['fit']['temperature']:.4f}。使用固定值处理已保存的测试logits，无新增模型前向，不参与adapter选择。正温度不改变二分类判定，因此不能提升准确率。"]
 lines+=['','## 配对分组不确定性','',
 '2000次按测试指令组配对bootstrap，保留组内元素及两种方法的配对关系。差值是右减左；准确率越高越好，NLL越低越好。区间为逐项95%，未做多重比较校正；每种方法仅一个训练种子，区间不包含训练种子的波动，用于描述本子集的不确定性。','',
 '| 比较（左 → 右） | 准确率差 [95% CI] | NLL差 [95% CI] |','|---|---:|---:|']
 for pair,comp in test['paired_group_bootstrap'].items():
  left,right=pair.split('__vs__');values=[]
  for key in ['majority_accuracy','vote_nll']:
   m=comp['metrics'][key];lo,hi=m['ci95'];values.append(f"{m['right_minus_left']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
  lines.append('| '+names[left]+' → '+names[right]+' | '+' | '.join(values)+' |')
 values=[]
 for key in ['majority_accuracy','vote_nll']:
  m=calibration_comparison['metrics'][key];lo,hi=m['ci95'];values.append(f"{m['right_minus_left']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
 lines.append('| 温度校准基座 → 选定LoRA | '+' | '.join(values)+' |')
 lines+=['',f"温度校准基座的测试NLL为{calibrated_metrics['vote_nll']:.4f}，选定adapter为{test['metrics']['lora_image']['vote_nll']:.4f}；判断参数微调收益时，同时参考这一简单校准基线。"]
 change=test['paired_group_bootstrap']['base_image__vs__lora_image']['metrics']['majority_accuracy']
 conclusion='当前指令组bootstrap支持准确率提升，但样本组数仍有限，需在新数据上复核。' if change['ci95'][0]>0 else '当前指令组bootstrap不支持明确的准确率提升；不把训练成功或点估计变化写成算法提升。'
 lines+=['','## 结论与复现','',conclusion,
 '训练先验用于揭示标签比例与过度自信问题；纯文本对照衡量模型只看问题时的行为。不能把这些指标等同于像素级定位或完整视觉推理能力。',
 '最终43项测试通过（原流程41项+2项温度基线测试），两个正式checkpoint的新进程重载通过；源码与每轮快照、全部预测条数及重新计算的指标已核验。',
 '最初终端关闭导致下载在43/160中止，失败日志保留；新目录复用已校验HTTP范围缓存后重新完成。新实验由VS Code启动Windows后台进程，完整保存日志。']
 for mode,m in models.items():
  summary=m['summary'];lines.append(f"{mode}：{summary['optimizer_updates']}次更新，训练阶段{summary['training_seconds']:.1f}秒，峰值已分配显存{summary['peak_allocated_bytes']/1024**3:.2f}GiB。")
 lines+=[f"封存测试：{test['elements']*4}次模型前向，{test['seconds']:.1f}秒，峰值{test['peak_allocated_bytes']/1024**3:.2f}GiB。",'',
 'Windows总实验目录：'+str(study)+'. 原始数据、逐条标签、预测、权重、环境与源码快照留在Windows；GitHub仅发布源码、汇总结果、版本记录和哈希。',
 '本轮测试结果现已观察，后续调参不得再将其作为未观察测试集；必须再保留新的最终评测指令。']
 (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('REPORT_READY',out,flush=True)
if __name__=='__main__':main()
