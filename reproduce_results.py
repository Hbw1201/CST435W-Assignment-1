"""Recompute archived statistics, CUDA intervals and figures locally; never rerun GPU timings."""
import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from verify_archive import ROOT, verify


def compare_csv(first, second, allowed_extra=()):
    def read(path):
        with path.open(newline='',encoding='utf-8-sig') as stream:
            return list(csv.DictReader(stream))
    before,after=read(first),read(second)
    if len(before)!=len(after):
        raise ValueError('Changed row count: '+str(first))
    for old,new in zip(before,after):
        if not old.keys() <= new.keys() or new.keys()-old.keys() != set(allowed_extra):
            raise ValueError('Changed columns: '+str(first))
        for key in old:
            if old[key]==new[key]: continue
            try:
                equal=math.isclose(float(old[key]),float(new[key]),rel_tol=1e-10,abs_tol=1e-9)
            except ValueError:
                equal=False
            if not equal: raise ValueError(f'Recomputed value differs: {first}/{key}')


def reproduce(output):
    output=output.resolve()
    if output.exists() or output.is_relative_to(ROOT/'results'):
        raise FileExistsError('Choose a fresh output directory outside the archived results')
    verification=verify()
    output.mkdir(parents=True)
    copy=output/'results'
    shutil.copytree(ROOT/'results',copy)
    logs=output/'logs'
    logs.mkdir()
    commands=[]
    def run(script,*arguments):
        command=[sys.executable,'-B',str(ROOT/script),*map(str,arguments)]
        proc=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (logs/f'{len(commands)+1:02d}-{Path(script).stem}.txt').write_text(proc.stdout,encoding='utf-8')
        commands.append(dict(script=script,arguments=list(map(str,arguments)),returncode=proc.returncode))
        if proc.returncode:
            raise RuntimeError(f'{script} failed; see {logs}\n'+proc.stdout[-2000:])
    run('summarize_sessions.py',copy/'main_repeat')
    run('verify_results.py',copy/'full_run')
    for name in ['runs.csv','summary.csv','session_summary.csv','scaling.csv','session_scaling.csv',
                 'device_profile.csv','monitoring_summary.csv']:
        compare_csv(ROOT/'results/main_repeat'/name,copy/'main_repeat'/name)
    traces=[('gpu_diagnosis','knn_sync.sqlite','trace_sync',False),
            ('batch_sweep','sweep.sqlite','traced',True),
            ('batch_repeat5','sweep.sqlite','traced',True),
            ('trace_repeat','trace.sqlite','traced',False)]
    for name,database,requests,batch in traces:
        folder=copy/name
        options=['--by-batch'] if batch else []
        if name=='trace_repeat': options.append('--segments')
        run('analyze_gpu_trace.py',folder/database,folder/requests/'runs.csv',folder,*options)
        for filename in ['kernel_summary.csv','kernel_timeline.csv']:
            compare_csv(ROOT/'results'/name/filename,folder/filename)
        if name=='trace_repeat': compare_csv(ROOT/'results'/name/'segments.csv',folder/'segments.csv')
        if batch:
            run('summarize_batch.py',folder)
            extra=('mean_wall_ms','sample_sd_wall_ms','cv_wall_pct','mean_request_overlap_pct',
                   'sample_sd_request_overlap_pp') if name=='batch_sweep' else ()
            compare_csv(ROOT/'results'/name/'summary.csv',folder/'summary.csv',extra)
            compare_csv(ROOT/'results'/name/'overlap_runs.csv',folder/'overlap_runs.csv')
    figures=output/'figures'
    run('plot_results.py',copy/'full_run','--out',figures)
    run('plot_sessions.py',copy/'main_repeat','--trace',copy/'trace_repeat','--out',figures)
    report_figures=['figure-1-neighbour-vote','figure-2-query-partition','performance-sessions','confusion','batch-distributions']
    for name in report_figures:
        for suffix in ['.pdf','.svg','.png']:
            if not (figures/(name+suffix)).is_file(): raise FileNotFoundError(name+suffix)
    result=dict(status='passed',archive=verification,commands=commands,
                checked_trace_databases=len(traces),report_figures=report_figures,
                numeric_comparison=dict(relative_tolerance=1e-10,absolute_tolerance=1e-9),
                network_used=False,gpu_used=False)
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',output=str(output),trace_databases=4,report_figures=5)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    reproduce(args.out)
