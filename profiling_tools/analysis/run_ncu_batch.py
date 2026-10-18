"""Run one resumable NCU Roofline collection per selected kernel."""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path

TASK_INPUTS = {"t2i":"inputs/omni/t2i.json", "t2v":"inputs/omni/t2v.json",
 "action":"inputs/omni/action_policy_robot.json", "reasoner_video":"inputs/reasoner/reasoner_video.json"}
OVERLAP = {"cosmos.component.reasoner_e2e"}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--selection",type=Path,default=Path("profiling_results/kernel_selection/all_selected_kernels.csv"))
 p.add_argument("--reports-dir",type=Path,default=Path("profiling_results/roofline/reports"))
 p.add_argument("--raw-dir",type=Path,default=Path("profiling_results/roofline/raw"))
 p.add_argument("--outputs-dir",type=Path,default=Path("outputs/roofline/batch"))
 p.add_argument("--index",type=Path,default=Path("profiling_results/roofline/index.csv"))
 p.add_argument("--tasks",nargs="+",choices=tuple(TASK_INPUTS)); p.add_argument("--launch-count",type=int,default=1)
 p.add_argument("--include-overlapping-ranges",action="store_true"); p.add_argument("--dry-run",action="store_true")
 p.add_argument("--limit",type=int); p.add_argument("--checkpoint-path",default="Cosmos3-Edge"); a=p.parse_args()
 rows=list(csv.DictReader(a.selection.open(newline="")))
 if a.tasks: rows=[r for r in rows if r["task"] in a.tasks]
 if not a.include_overlapping_ranges: rows=[r for r in rows if r["nvtx_range"] not in OVERLAP]
 if not a.dry_run:
  for d in (a.reports_dir,a.raw_dir,a.outputs_dir,a.index.parent): d.mkdir(parents=True,exist_ok=True)
 index=[]; pending=0
 for r in rows:
  task=r["task"]; comp=r["nvtx_range"].removeprefix("cosmos.component.")
  digest=hashlib.sha1(r["kernel_name"].encode()).hexdigest()[:10]
  stem=f"{task}_{comp}_k{int(r['rank']):03d}_{digest}"
  base=a.reports_dir/stem; report=base.with_suffix(".ncu-rep"); raw=a.raw_dir/f"{stem}.csv"; output=a.outputs_dir/stem
  if not report.is_file():
   if a.limit is not None and pending>=a.limit: continue
   pending+=1
   if output.exists() and not a.dry_run: raise SystemExit(f"Existing incomplete output: {output}\nMove/remove it before retrying.")
   cmd=["ncu","--set","roofline","--replay-mode","kernel","--nvtx","--nvtx-include",r["nvtx_range"]+"/",
    "--kernel-name-base","demangled","--rename-kernels","false","--kernel-name",r["kernel_name"],
    "--launch-count",str(a.launch_count),"--target-processes","all","--force-overwrite","--export",str(base),
    sys.executable,"-m","profiling_tools.runners.run_profiled","--parallelism-preset=latency","--no-guardrails",
    "-i",TASK_INPUTS[task],"-o",str(output),"--checkpoint-path",a.checkpoint_path,"--seed=0"]
   print(" ".join(json.dumps(x) if " " in x else x for x in cmd),flush=True)
   if not a.dry_run: subprocess.run(cmd,check=True)
  if report.is_file() and not raw.is_file():
   cmd=["ncu","--import",str(report),"--page","raw","--csv"]
   if not a.dry_run:
    with raw.open("w") as f: subprocess.run(cmd,check=True,stdout=f)
  status="complete" if report.is_file() and raw.is_file() else "pending"
  index.append({**r,"stem":stem,"report":str(report),"raw_csv":str(raw),"inference_output":str(output),"status":status})
 if index and not a.dry_run:
  with a.index.open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=list(index[0])); w.writeheader(); w.writerows(index)
  print(f"Wrote {a.index}")
if __name__=="__main__": main()
