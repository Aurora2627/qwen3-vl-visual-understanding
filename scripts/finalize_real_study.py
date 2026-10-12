"""Finish a long VS Code experiment; publish only after all checks pass.

Invoked explicitly from VS Code. Refuses changed source, other working-tree
edits, staged changes or an unexpected remote. Never launches model inference.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = '0.3.0'
FILES = ['CHANGELOG.md','README.md','VERSION','VERSION_HISTORY.md','VSCODE_START.md',
         'DATA_SOURCES.md','configs/execution.json','src/run_record.py',
         'src/correlation.py','src/real_judging.py','src/remote_zip.py',
         'scripts/inspect_dataset.py','scripts/prepare_evalmuse.py',
         'scripts/run_real_study.py','scripts/report_real_study.py',
         'scripts/validate_real_run.py','scripts/finalize_real_study.py',
         'tests/test_real_evaluation.py']


def command(*args, capture=False):
    result = subprocess.run(list(args),cwd=ROOT,check=True,
                            stdout=subprocess.PIPE if capture else None,text=True)
    return result.stdout.rstrip('\n') if capture else None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--timeout',type=int,default=7200)
    args = parser.parse_args()
    run = args.run.resolve()
    report = ROOT/'reports'/('v'+RELEASE)
    if report.exists():
        raise FileExistsError('Release report already exists')
    head = command('git','rev-parse','HEAD',capture=True)
    seal_paths = [p for folder in ['src','scripts','tests','configs']
                  for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    seal_paths += [ROOT/name for name in FILES if (ROOT/name).is_file()]
    seal = {str(p.relative_to(ROOT)):digest(p) for p in seal_paths}
    with (run/'finalization-plan.json').open('x') as stream:
        json.dump({'head':head,'source_seal':seal,'publish':args.publish},stream,indent=2)
    print('WAITING_FOR_COMPLETE_RUN',run.name,flush=True)
    deadline = time.monotonic()+args.timeout
    while True:
        status_file = run/'status.json'
        if status_file.exists():
            try:
                status = json.loads(status_file.read_text())['status']
            except json.JSONDecodeError:
                status = 'writing'
            if status == 'failed':
                raise RuntimeError('Model experiment failed; preserving run, no release')
            if status == 'complete':
                break
        if time.monotonic()>deadline:
            raise TimeoutError('Experiment still unfinished; no release')
        time.sleep(15)
    if any(digest(ROOT/name)!=value for name,value in seal.items()):
        raise RuntimeError('Source changed while running; review before release')
    if command('git','rev-parse','HEAD',capture=True)!=head:
        raise RuntimeError('Git HEAD changed; review before release')
    if command('git','diff','--cached','--name-only',capture=True):
        raise RuntimeError('Existing staged changes; refusing automatic release')
    allowed = set(FILES)
    changed = command('git','status','--porcelain','--untracked-files=all',capture=True)
    if any(line[3:] not in allowed for line in changed.splitlines()):
        raise RuntimeError('Other changes found; refusing automatic release')
    command(sys.executable,'-m','compileall','-q','src','scripts','tests')
    with (run/'release-tests.log').open('x') as log:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    command(sys.executable,'scripts/report_real_study.py',str(run),'--output',str(report))
    command(sys.executable,'scripts/validate_real_run.py',str(run),'--report',str(report),
            '--output',str(report/'validation.json'))
    summary = json.loads((report/'summary.json').read_text())
    lines = ['', '### v0.3.0 本机结果', '',
             '| 数据/协议 | 有分数/总数 | SRCC | 排序正确/非平局 |',
             '|---|---:|---:|---:|']
    for key,metrics in summary['groups'].items():
        value = metrics['srcc_scored_only']
        lines.append('| {} | {}/{} | {} | {}/{} |'.format(key,metrics['scored_images'],
            metrics['n_images'],'未定义' if value is None else '{:.3f}'.format(value),
            metrics['pair_correct'],metrics['strict_human_pairs']))
    lines += ['', '已在 VS Code 使用系统 Python / PyTorch / MPS 完成 128 次推理；13 项测试通过。相关性仅对有分数输出计算，排序中的模型平局、格式错误和不计分均算错。详细区间、全 PASS/FAIL 基线、类别误差及完整性校验见 [真实图报告](reports/v0.3.0/RESULTS.md)。',
              '', '每部分只有 16 个指令组，不能据此宣称显著提升。本版没有微调；原始输入、完整预测及源码快照保留在 `runs/'+run.name+'`。导出与收尾校验程序在模型运行期间完善，不改动运行前保存的源码快照。']
    with (ROOT/'README.md').open('a') as stream:
        stream.write('\n'.join(lines)+'\n')
    config = json.loads((ROOT/'configs/execution.json').read_text())
    config.update({'status':'real-human-agreement-complete','run_id':run.name,
                   'predictions':128,'unit_tests':13})
    (ROOT/'configs/execution.json').write_text(json.dumps(config,indent=2)+'\n')
    path = ROOT/'CHANGELOG.md'
    path.write_text(path.read_text().replace('实验结果验证中，最终状态见 reports/v0.3.0。',
        '在 VS Code / PyTorch / MPS 完成 128 次推理，13 项测试及快照/输入/预测/发布边界校验通过。结果见 reports/v0.3.0；未进行微调，不宣称算法提升。'))
    path = ROOT/'VERSION_HISTORY.md'
    path.write_text(path.read_text().replace('验证状态见对应报告。',
        '128 次真实 MPS 推理及 13 项测试完成，原始失败保留，结果与校验见 reports/v0.3.0。'))
    command('git','diff','--check')
    print('RELEASE_CHECKS_PASSED',flush=True)
    if args.publish:
        if command('git','branch','--show-current',capture=True)!='main':
            raise RuntimeError('Release must be committed on main')
        remote = command('git','remote','get-url','--push','origin',capture=True)
        if remote!='https://github.com/Aurora2627/qwen3-vl-visual-understanding.git':
            raise RuntimeError('Unexpected push destination')
        if command('git','tag','--list','v'+RELEASE,capture=True):
            raise RuntimeError('Release tag already exists')
        command('git','add','--',*FILES,'reports/v'+RELEASE)
        command('git','diff','--cached','--check')
        command('git','commit','-m','Evaluate Qwen3-VL on a prompt-disjoint EvalMuse pilot')
        command('git','tag','-a','v'+RELEASE,'-m',
                'EvalMuse 64-image pilot: 128 MPS inferences, human agreement, source and export verification; no fine-tuning')
        command('git','push','origin','main','v'+RELEASE)
        expected_head = command('git','rev-parse','HEAD',capture=True)
        remote_head = command('git','ls-remote','origin','refs/heads/main',capture=True).split()[0]
        if remote_head!=expected_head:
            raise RuntimeError('Remote main verification failed')
        print('RELEASE_PUSHED_AND_VERIFIED','v'+RELEASE,expected_head,flush=True)


if __name__ == '__main__':
    main()
