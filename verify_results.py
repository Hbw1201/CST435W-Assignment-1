"""Audit run tables and spot-check labels directly from integer IDX pixels."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
import numpy as np


def rows(directory, name):
    with (directory/name).open(newline="") as f:
        return list(csv.DictReader(f))


def verify(folder, raw, figure_dir):
    env=json.loads((folder/"environment.json").read_text())
    for name,h in env["csv_sha256"].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h, name
    base=Path(__file__).resolve().parent
    for name,h in env["sources"].items():
        assert hashlib.sha256((base/name).read_bytes()).hexdigest()==h, name
    table, runs, samples = rows(folder,"summary.csv"),rows(folder,"runs.csv"),rows(folder,"predictions.csv")
    scaling, confusion, traces = rows(folder,"scaling.csv"),rows(folder,"confusion.csv"),rows(folder,"device_profile.csv")
    sizes=env["sizes"]; repeats=env["measured_per_cell"]
    assert len(runs)==3*len(sizes)*repeats and len(table)==3*len(sizes)
    assert len(confusion)==100*len(sizes) and len(scaling)==len(sizes)
    longest=[r for r in samples if int(r["n_test"])==max(sizes)]
    for n in sizes:
        part=[r for r in samples if int(r["n_test"])==n]
        assert [int(r["test_index"]) for r in part]==list(range(n))
        for i,r in enumerate(part):
            assert r["cpu"]==r["gpu1"]==r["gpu2"]==longest[i]["cpu"]
            assert r["true_label"]==longest[i]["true_label"]
        corr=sum(r["cpu"]==r["true_label"] for r in part)
        counts=Counter((int(r["true_label"]), int(r["cpu"])) for r in part)
        assert all(int(c["count"])==counts[int(c["true_label"]),int(c["predicted_label"])]
                   for c in confusion if int(c["n_test"])==n)
        med={}
        for mode in ("cpu","gpu1","gpu2"):
            rset=[r for r in runs if int(r["n_test"])==n and r["mode"]==mode]
            assert [int(r["run"]) for r in rset]==list(range(1,repeats+1))
            secs=[float(r["wall_s"]) for r in rset]
            assert min(secs)>0
            med[mode]=statistics.median(secs)
            ss=next(r for r in table if int(r["n_test"])==n and r["mode"]==mode)
            assert int(ss["correct"])==corr and int(ss["mismatches_vs_cpu"])==0
            for key,value in [("median_wall_s",med[mode]),("mean_wall_s",statistics.mean(secs)),
                              ("sample_std_wall_s",statistics.stdev(secs)),("throughput_images_s",n/med[mode])]:
                assert math.isclose(float(ss[key]),value,rel_tol=1e-10),key
            if mode!="cpu":
                pp=[r for r in traces if int(r["n_test"])==n and r["mode"]==mode]
                assert sum(int(r["shard_n"]) for r in pp)==n
                assert {int(r["device"]) for r in pp}==set(range(min(n,1 if mode=="gpu1" else 2)))
                assert all(float(r["compute_ms"])>0 for r in pp)
        s=next(r for r in scaling if int(r["n_test"])==n)
        for key,value in [("cpu_to_gpu1",med["cpu"]/med["gpu1"]),("cpu_to_gpu2",med["cpu"]/med["gpu2"]),
                          ("gpu2_speedup",med["gpu1"]/med["gpu2"]),("gpu2_efficiency_pct",50*med["gpu1"]/med["gpu2"])]:
            assert math.isclose(float(s[key]),value,rel_tol=1e-10),key
    audit=dict(official_runs=len(runs),compared_prefix_rows=len(samples),
               full_test_count=len(longest),mismatches_across_modes=0,
               recomputed_medians_sd_throughput_scaling=True)
    if raw is not None:
        im=np.memmap(raw/"train-images-idx3-ubyte",dtype=np.uint8,mode="r",offset=16,shape=(60000,784))
        y=np.memmap(raw/"train-labels-idx1-ubyte",dtype=np.uint8,mode="r",offset=8)
        q=np.memmap(raw/"t10k-images-idx3-ubyte",dtype=np.uint8,mode="r",offset=16,shape=(10000,784))
        target=np.memmap(raw/"t10k-labels-idx1-ubyte",dtype=np.uint8,mode="r",offset=8)
        example_indices=[0]
        err=next((i for i,r in enumerate(longest) if r["cpu"]!=r["true_label"]),None)
        if err is not None and err!=0: example_indices.append(err)
        ids=sorted(set(np.linspace(0,len(longest)-1,25,dtype=int).tolist()+example_indices+[min(99,len(longest)-1)]))
        examples=[]
        for i in ids:
            d2=np.empty(len(im),dtype=np.int32)
            for start in range(0,len(im),1024):
                stop=min(start+1024,len(im))
                delta=np.subtract(im[start:stop],q[i],dtype=np.int32)
                d2[start:stop]=np.einsum("ij,ij->i",delta,delta,dtype=np.int32)
            closest=np.argsort(d2,kind="stable")[:5]
            answer=int(np.bincount(y[closest],minlength=10).argmax())
            assert answer==int(longest[i]["cpu"]) and int(target[i])==int(longest[i]["true_label"])
            if i in example_indices:
                examples.append(dict(index=i,truth=int(target[i]),prediction=answer,
                                train_indices=closest.tolist(),labels=y[closest].tolist(),squared_distances=d2[closest].tolist()))
        audit.update(independent_integer_check_indices=ids,independent_disagreements=0,examples=examples)
        if figure_dir:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            figure_dir.mkdir(exist_ok=True,parents=True)
            fig,axs=plt.subplots(len(examples),6,squeeze=False,figsize=(8.5,1.95*len(examples)))
            for row,e in enumerate(examples):
                for col,ax in enumerate(axs[row]):
                    if col==0:
                        pixels=q[e["index"]]; title="Test image"; foot=f"Test {e['index']}  |  {e['truth']} to {e['prediction']}"
                    else:
                        tid=e["train_indices"][col-1]; pixels=im[tid]; title=f"Neighbour {col}"; foot=f"ID {tid}  /  {int(y[tid])}"
                    ax.imshow(pixels.reshape(28,28),cmap="gray_r",interpolation="nearest",vmin=0,vmax=255)
                    if row == 0:
                        ax.set_title(title,fontsize=9,fontweight="bold" if col==0 else "normal")
                    ax.set_xlabel(foot,fontsize=8); ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values(): spine.set_color("#22817e" if col==0 else "#d8dedf")
            fig.subplots_adjust(left=.025,right=.985,top=.89,bottom=.14,wspace=.23,hspace=.57)
            fig.savefig(figure_dir/"neighbors.png",dpi=210,bbox_inches="tight")
            plt.close(fig)
    if raw is not None:
        (folder/"validation.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path); p.add_argument("--raw",type=Path);p.add_argument("--figures",type=Path)
    a=p.parse_args(); verify(a.results,a.raw,a.figures)
