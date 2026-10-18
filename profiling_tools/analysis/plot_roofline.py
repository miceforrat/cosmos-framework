"""Plot task/component-weighted L1, L2 and DRAM hierarchical Rooflines."""
from __future__ import annotations
import argparse, csv, json, math, statistics
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

TASKS=("t2i","t2v","action","reasoner_video")
from matplotlib.lines import Line2D
CMAPS={"t2i":"Blues","t2v":"Oranges","action":"Greens","reasoner_video":"Reds"}
OVERLAP={"reasoner_e2e"}

def num(s):
 try:return float(s.replace(",","").replace(" ",""))
 except:return None

def raw_metrics(path):
 lines=path.read_text(errors="replace").splitlines(); start=next((i for i,x in enumerate(lines) if x.startswith('"ID"') or x.startswith("ID,")),None)
 if start is None: raise ValueError(f"no NCU CSV header: {path}")
 table=list(csv.reader(lines[start:]))
 if "Metric Name" not in table[0]:
  header,units,values=table[0],table[1],table[2]
  out={}
  for name,unit,value in zip(header,units,values):
   v=num(value)
   if v is None:continue
   unit=unit.strip().lower(); scale=1.0
   for prefix,factor in (("t",1e12),("g",1e9),("m",1e6),("k",1e3)):
    if unit.startswith(prefix+"byte") or unit.startswith(prefix+"hz"):
     scale=factor; break
   out[name]=v*scale
  return out
 rows=list(csv.DictReader(lines[start:])); out={}
 for r in rows:
  name=r.get("Metric Name")
  value=r.get("Metric Value")
  if name and value is not None:
   v=num(value)
   if v is not None: out[name]=v
 return out

def first(m,*names):
 for n in names:
  if n in m:return m[n]
 return 0.0

def work(m):
 hz=first(m,"sm__cycles_elapsed.avg.per_second")
 tensor=[]
 for name,v in m.items():
  if name.startswith("sm__ops_path_tensor") and name.endswith(".sum.per_cycle_elapsed"):
   peak=m.get(name.replace(".sum.per_cycle_elapsed",".sum.peak_sustained"),0.0)
   tensor.append((v*hz,peak*hz,name))
 if tensor and max(tensor)[0]>0:return max(tensor)
 add=first(m,"smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed")
 mul=first(m,"smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed")
 fma=first(m,"smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed")
 fp32=((add+mul+2*fma)*hz,2*first(m,"sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained")*hz,"scalar_fp32")
 hadd=first(m,"smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_cycle_elapsed")
 hmul=first(m,"smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_cycle_elapsed")
 hfma=first(m,"smsp__sass_thread_inst_executed_op_hfma_pred_on.sum.per_cycle_elapsed")
 fp16=((2*hadd+2*hmul+4*hfma)*hz,4*first(m,"sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained")*hz,"scalar_fp16")
 return max(fp32,fp16,key=lambda item:item[0])

def traffic(m,level):
 if level=="dram": return (first(m,"dram__bytes.sum.per_second"), first(m,"dram__bytes.sum.peak_sustained")*first(m,"dram__cycles_elapsed.avg.per_second"))
 if level=="l2": return (first(m,"derived__lts__lts2xbar_bytes.sum.per_second"), first(m,"derived__lts__lts2xbar_bytes.sum.peak_sustained")*first(m,"lts__cycles_elapsed.avg.per_second"))
 achieved=first(m,"derived__l1tex__lsu_writeback_bytes_mem_lgds.sum.per_second","derived__l1tex__lsu_writeback_bytes_mem_lg.sum.per_second","derived__l1tex__lsu_writeback_bytes.sum.per_second")
 peak=first(m,"derived__l1tex__lsu_writeback_bytes_mem_lgds.sum.peak_sustained","derived__l1tex__lsu_writeback_bytes_mem_lg.sum.peak_sustained","derived__l1tex__lsu_writeback_bytes.sum.peak_sustained")*first(m,"l1tex__cycles_elapsed.avg.per_second")
 return achieved,peak

def component_shares(root):
 files={t:root/f"{t}_full.json" for t in TASKS}; result={}
 for task,path in files.items():
  sums=defaultdict(float)
  for r in json.loads(path.read_text())["records"]:
   if r.get("gpu_ms") is not None and r["component"] not in OVERLAP:sums[r["component"]]+=float(r["gpu_ms"])
  total=sum(sums.values())
  for c,v in sums.items():result[(task,c)]=v/total if total else 0
 return result

def colors(points):
 result={}
 for task in TASKS:
  comps=sorted({p["component"] for p in points if p["task"]==task}); cmap=plt.get_cmap(CMAPS[task])
  for i,c in enumerate(comps): result[(task,c)]=cmap(0.45+0.45*(i/max(1,len(comps)-1)))
 return result

def marker_style(weight,max_weight):
 ratio=weight/max_weight if max_weight else 0
 if ratio < 0.05:return "^",38
 if ratio < 0.15:return "D",50
 if ratio < 0.35:return "p",64
 return "o",75+145*(ratio-0.35)/0.65

def short_kernel(name,limit=58):
 name=name.removeprefix("void ")
 if "<" in name:
  name=name[:name.index("<")]+"<...>"
 return name if len(name)<=limit else name[:limit-3]+"..."

def kernel_handles(points,color_map,max_weight):
 result=[]
 for p in sorted(points,key=lambda x:(TASKS.index(x["task"]),x["component"],-x["weight"])):
  marker,size=marker_style(p["weight"],max_weight)
  label=f"{p['task']}/{p['component']} k{p['rank']:02d}  {short_kernel(p['kernel_name'])}"
  result.append(Line2D([0],[0],marker=marker,linestyle="",markersize=math.sqrt(size),
   markerfacecolor=color_map[(p["task"],p["component"])],markeredgecolor="black",
   markeredgewidth=.35,label=label))
 return result

def panel(ax,points,level,color_map,max_weight):
 valid=[p for p in points if p[level+"_ai"]>0 and p["performance"]>0]
 peaks=[p[level+"_peak"] for p in valid if p[level+"_peak"]>0]
 # One common hardware roof must use one operation type. Cosmos runs primarily on
 # BF16 Tensor Cores; mixing scalar FP16/FP32 peaks would draw a false low ceiling.
 computes=[p["compute_peak"] for p in valid
           if "tensor_src_bf16" in p["work_metric"] and p["compute_peak"]>0]
 if not computes: computes=[p["compute_peak"] for p in valid if p["compute_peak"]>0]
 if not valid:
  ax.text(.5,.5,"No kernels with non-zero work and traffic",ha="center",va="center",transform=ax.transAxes)
  return
 ai_values=[p[level+"_ai"] for p in valid]
 perf_tops=[p["performance"]/1e12 for p in valid]
 sorted_perf=sorted(perf_tops)
 low_cut=None
 if len(sorted_perf)>=4:
  gaps=[math.log10(sorted_perf[i+1]/sorted_perf[i]) for i in range(len(sorted_perf)-1)]
  gap_index=max(range(len(gaps)),key=gaps.__getitem__)
  if gaps[gap_index]>1.0 and gap_index < max(1,len(sorted_perf)//3):
   low_cut=sorted_perf[gap_index+1]/1.8
 xmin=min(ai_values)/1.6; xmax=max(ai_values)*1.6
 if peaks and computes:
  bw=statistics.median(peaks); cp=statistics.median(computes)
  knee=cp/bw
  if min(ai_values)<=knee<=max(ai_values)*4:xmax=max(xmax,knee*1.25)
  xs=np.logspace(math.log10(xmin),math.log10(xmax),500)
  roof=np.minimum(cp,bw*xs)/1e12
  ax.plot(xs,roof,"k-",lw=2)
 for p in sorted(valid,key=lambda x:x["weight"]):
  marker,size=marker_style(p["weight"],max_weight)
  y=p["performance"]/1e12
  clipped=low_cut is not None and y<low_cut
  if clipped:marker,y="v",low_cut*1.08
  ax.scatter(p[level+"_ai"],y,s=size,marker=marker,
   c="none" if clipped else [color_map[(p["task"],p["component"])]],
   edgecolors=[color_map[(p["task"],p["component"])] if clipped else "black"],linewidths=1.0 if clipped else .35,alpha=.9 if clipped else .82)
 ax.set_xscale("log"); ax.set_yscale("log")
 ax.set_xlim(xmin,xmax)
 ymin=low_cut if low_cut is not None else min(perf_tops)/1.7
 ymax=max(perf_tops)*1.7
 if peaks and computes:ymax=max(ymax,cp/1e12*1.18)
 ax.set_ylim(ymin,ymax)
 if low_cut is not None:
  lower=[y for y in perf_tops if y<low_cut]
  ax.text(.015,.018,f"▽ {len(lower)} kernel(s) below view; min={min(lower):.2g} TOP/s",
          transform=ax.transAxes,fontsize=8,color="dimgray",ha="left",va="bottom")
 ax.grid(True,which="both",alpha=.2); ax.set_title(level.upper())
 ax.set_xlabel("Arithmetic intensity [OP/byte]",fontsize=11); ax.set_ylabel("Performance [TOP/s]",fontsize=11)
 ax.tick_params(axis="both",which="major",labelsize=9.5)

def comparison_figure(points,color_map,output_dir,stem,title):
 """Write one three-panel BF16 figure for a deliberately selected component group."""
 group=[p for p in points
        if "tensor_src_bf16" in p["work_metric"] and p["performance"]>0]
 if not group:return
 max_weight=max(p["weight"] for p in group) or 1
 handles=kernel_handles(group,color_map,max_weight)
 fig,axes=plt.subplots(1,3,figsize=(22,9))
 for ax,level in zip(axes,("l1","l2","dram")):
  panel(ax,group,level,color_map,max_weight)
 fig.suptitle(title,fontsize=15)
 fig.subplots_adjust(left=.055,right=.73,bottom=.09,top=.91,wspace=.27)
 fig.legend(handles=handles,loc="center right",bbox_to_anchor=(.995,.5),
            fontsize=9.0 if len(handles)<=20 else 8.0,ncol=1,
            title="Kernels grouped by task/component; descending within component",
            title_fontsize=9.5,framealpha=.9,handletextpad=.5)
 fig.savefig(output_dir/f"{stem}.png",dpi=220)
 fig.savefig(output_dir/f"{stem}.pdf")
 plt.close(fig)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--index",type=Path,default=Path("profiling_results/roofline/index.csv")); p.add_argument("--components-dir",type=Path,default=Path("profiling_results/components")); p.add_argument("--output-dir",type=Path,default=Path("profiling_results/roofline/plots")); a=p.parse_args()
 shares=component_shares(a.components_dir); points=[]
 for r in csv.DictReader(a.index.open(newline="")):
  if r["status"]!="complete":continue
  comp=r["nvtx_range"].removeprefix("cosmos.component.")
  if comp in OVERLAP:continue
  m=raw_metrics(Path(r["raw_csv"])); perf,compute_peak,work_metric=work(m)
  point={"task":r["task"],"component":comp,"rank":int(r["rank"]),"kernel_name":r["kernel_name"],"kernel_share":float(r["share"]),"component_share":shares.get((r["task"],comp),0),"performance":perf,"compute_peak":compute_peak,"work_metric":work_metric}
  point["weight"]=point["kernel_share"]*point["component_share"]
  for level in ("l1","l2","dram"):
   rate,peak=traffic(m,level); point[level+"_ai"]=perf/rate if rate else 0; point[level+"_peak"]=peak
  points.append(point)
 if not points:raise SystemExit("No complete, parseable NCU rows in index")
 a.output_dir.mkdir(parents=True,exist_ok=True); cmap=colors(points); mw=max(x["weight"] for x in points) or 1
 overall_points=[x for x in points if "tensor_src_bf16" in x["work_metric"] and x["performance"]>0]
 handles=kernel_handles(overall_points,cmap,mw)
 fig,axes=plt.subplots(1,3,figsize=(22,9));
 for ax,l in zip(axes,("l1","l2","dram")):panel(ax,overall_points,l,cmap,mw)
 fig.suptitle("Cosmos3 BF16 Tensor Core hierarchical hardware Roofline",fontsize=15)
 fig.subplots_adjust(left=.055,right=.73,bottom=.09,top=.91,wspace=.27)
 fig.legend(handles=handles,loc="center right",bbox_to_anchor=(.995,.5),fontsize=9.0 if len(handles)<=20 else 8.0,
            ncol=1,title="Kernels grouped by task/component; descending within component",title_fontsize=9.5,
            framealpha=.9,handletextpad=.5,columnspacing=.8)
 fig.savefig(a.output_dir/"roofline_overall.png",dpi=220); fig.savefig(a.output_dir/"roofline_overall.pdf")
 for level in ("l1","l2","dram"):
  fig,ax=plt.subplots(figsize=(15,8)); panel(ax,overall_points,level,cmap,mw)
  fig.subplots_adjust(left=.08,right=.66,bottom=.11,top=.93)
  fig.legend(handles=handles,loc="center right",bbox_to_anchor=(.995,.5),fontsize=9.0 if len(handles)<=20 else 8.0,
             ncol=1,title="Kernels grouped by task/component; descending within component",title_fontsize=9.5,
             framealpha=.9,handletextpad=.5,columnspacing=.8)
  fig.savefig(a.output_dir/f"roofline_{level}.png",dpi=220); plt.close(fig)
 precision_groups={
  "bf16_tensor":("BF16 Tensor Core",lambda x:"tensor_src_bf16" in x["work_metric"]),
  "scalar_fp16":("Scalar FP16",lambda x:x["work_metric"]=="scalar_fp16"),
  "scalar_fp32":("Scalar FP32",lambda x:x["work_metric"]=="scalar_fp32"),
 }
 for tag,(title,accept) in precision_groups.items():
  group=[x for x in points if accept(x) and x["performance"]>0]
  if not group:continue
  group_handles=kernel_handles(group,cmap,mw)
  fig,axes=plt.subplots(1,3,figsize=(22,9))
  for ax,level in zip(axes,("l1","l2","dram")):panel(ax,group,level,cmap,mw)
  fig.suptitle(f"Cosmos3 {title} hierarchical hardware Roofline")
  fig.subplots_adjust(left=.055,right=.73,bottom=.09,top=.91,wspace=.27)
  fig.legend(handles=group_handles,loc="center right",bbox_to_anchor=(.995,.5),fontsize=9.0 if len(group_handles)<=20 else 8.0,
             ncol=1,title="Kernels grouped by task/component; descending within component",
             title_fontsize=9.5,framealpha=.9,handletextpad=.5)
  fig.savefig(a.output_dir/f"roofline_{tag}_overall.png",dpi=220)
  fig.savefig(a.output_dir/f"roofline_{tag}_overall.pdf"); plt.close(fig)
  for level in ("l1","l2","dram"):
   fig,ax=plt.subplots(figsize=(15,8)); panel(ax,group,level,cmap,mw)
   fig.suptitle(f"Cosmos3 {title} {level.upper()} Roofline")
   fig.subplots_adjust(left=.08,right=.66,bottom=.11,top=.90)
   fig.legend(handles=group_handles,loc="center right",bbox_to_anchor=(.995,.5),fontsize=9.0 if len(group_handles)<=20 else 8.0,
              ncol=1,title="Kernels grouped by task/component; descending within component",
              title_fontsize=9.5,framealpha=.9,handletextpad=.5)
   fig.savefig(a.output_dir/f"roofline_{tag}_{level}.png",dpi=220); plt.close(fig)
 comparison_figure(
  [x for x in points if x["component"]=="diffusion_sampling"],cmap,a.output_dir,
  "roofline_compare_diffusion_sampling",
  "Diffusion sampling across Action, Text-to-Image and Text-to-Video (BF16 Tensor Core)")
 comparison_figure(
  [x for x in points if x["task"]=="reasoner_video" and x["component"] in
   {"vision_encoder","reasoner_prefill","reasoner_decode"}],cmap,a.output_dir,
  "roofline_compare_reasoner_components",
  "Video Reasoner components (BF16 Tensor Core)")
 with (a.output_dir/"roofline_points.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(points[0])); w.writeheader(); w.writerows(points)
 print(f"Wrote {len(points)} points to {a.output_dir}")
if __name__=="__main__":main()
