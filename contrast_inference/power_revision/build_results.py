from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).parent;RESULTS=BASE/'results';PAPER=BASE/'manuscript'
def wilson(k,n):
 z=norm.ppf(.975);q=k/n;den=1+z*z/n;c=(q+z*z/(2*n))/den;r=z*np.sqrt(q*(1-q)/n+z*z/(4*n*n))/den
 return [float(max(0,c-r)),float(min(1,c+r))]
def build():
 df=pd.concat([pd.read_csv(RESULTS/'validation/replications.csv.gz'),pd.read_csv(RESULTS/'multiplier_validation/replications.csv.gz')],ignore_index=True)
 assert len(df)==132000 and df.method.nunique()==20 and not df.duplicated(['cell','rep','method']).any()
 config=json.loads((RESULTS/'validation/design.json').read_text())
 for ci,c in enumerate(config['cells']):
  for name,g in df[df.cell==ci].groupby('method'):assert len(g)==c['repetitions'] and (g.seed==config['seed']+ci*100000+g.rep).all()
 summary=[]
 for key,g in df.groupby(['model','n','method']):
  model,n,method=key;N=len(g);F=int(g.false_order.sum())
  s=dict(model=model,n=int(n),method=method,repetitions=N,false_orders=F,fwer=F/N,fwer_ci=wilson(F,N),
         excluded=float(g.excluded.mean()),exact_degrees=float(g.exact_degrees.mean()),
         degree_width=float(g.degree_width.mean()),rank_width=float(g.rank_width.mean()))
  if g.hub.notna().all():
   k=int(g.hub.sum());s.update(hubs=k,power=k/N,power_ci=wilson(k,N))
  summary.append(s)
 (RESULTS/'report_numbers.json').write_text(json.dumps(dict(datasets=6600,method_dataset_records=len(df),methods=20,summary=summary),indent=2))
 lookup={(s['model'],s['n'],s['method']):s for s in summary}
 methods=['P','E64','FS-T64','M-C'];labels={'gaussian_star':'Gaussian star','t5_star':r'$t_5$ star','near_tie':'Near-tie star'}
 table=[r'\begin{table}[htbp]\centering',r'\caption{Unique-hub certification on fresh, matched validation samples with $p=12$. Percentages are shown. There are 300 datasets per separated-star cell and 200 per near-tie cell. M-C is asymptotic; the other displayed methods have finite-sample guarantees under IID sampling.}\label{tab:new-power}',r'\begin{tabular}{llrrrr}\toprule Model & $n$ & P & E64 & FS-T64 & M-C\\\midrule']
 for model in labels:
  for n in ([252,1008,4096] if model!='near_tie' else [1008,4096]):
   table.append(f"{labels[model]} & {n} & "+' & '.join(f"{100*lookup[model,n,m]['power']:.1f}" for m in methods)+r'\\')
 table+= [r'\bottomrule\end{tabular}\end{table}']
 (PAPER/'power_table.tex').write_text('\n'.join(table)+'\n')
 nulls=['independent','discrete_independent','equicorrelated','tied_hubs'];nl=['Gaussian independence','Ordinal independence','Equal correlation 0.5','Two tied leading loadings']
 table=[r'\begin{table}[htbp]\centering',r'\caption{False strict-order familywise error rates at $n=1008$, $p=12$, with 1000 fresh datasets per cell and nominal level 0.05. The last column gives a Wilson 95\% interval for FS-T64.}\label{tab:new-null}',r'\begin{tabular}{lrrrrl}\toprule Model & P & E64 & FS-T64 & M-C & FS-T64 interval\\\midrule']
 for mod,label in zip(nulls,nl):
  s=lookup[mod,1008,'FS-T64'];ci=s['fwer_ci']
  table.append(label+' & '+' & '.join(f"{lookup[mod,1008,m]['fwer']:.3f}" for m in methods)+f" & [{ci[0]:.3f}, {ci[1]:.3f}]"+r'\\')
 table+=[r'\bottomrule\end{tabular}\end{table}'];(PAPER/'null_table.tex').write_text('\n'.join(table)+'\n')
 nodes=pd.read_csv(RESULTS/'application/nodes.csv');main=nodes[(nodes.dataset=='adults')&nodes.method.eq('FS-T64')].set_index('item');old=nodes[(nodes.dataset=='adults')&nodes.method.eq('E64')].set_index('item')
 table=[r'\begin{table}[htbp]\centering',r'\caption{Adult questionnaire analysis. Degree and rank bounds are sharp relative to each method\textquotesingle s order relation. FS-T64 uses 7015 training and 28064 verification observations. Its three guaranteed hub members are Q1, Q3 and Q10; Q6 may also attain degree three. No compatible tree has a unique hub.}\label{tab:rse}',r'\begin{tabular}{lrrrrl}\toprule Item & Sample degree & E64 degree & FS-T64 degree & FS-T64 rank & Guaranteed hub\\\midrule']
 for i in range(1,11):
  name='Q'+str(i);r=main.loc[name];o=old.loc[name]
  table.append(f"{name} & {int(r.point_degree)} & [{int(o.degree_lower)}, {int(o.degree_upper)}] & [{int(r.degree_lower)}, {int(r.degree_upper)}] & [{int(r.rank_lower)}, {int(r.rank_upper)}] & "+('Yes' if i in [1,3,10] else '')+r'\\')
 table+=[r'\bottomrule\end{tabular}\end{table}'];(PAPER/'rse_table.tex').write_text('\n'.join(table)+'\n')
 # Main figure: fixed locations, exact edge sets, uncertainty without invented geometry.
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
 fig,(ax,bx)=plt.subplots(1,2,figsize=(10.4,4.0),gridspec_kw={'width_ratios':[1.15,1]})
 coords={1:(0,1.0),2:(-.75,1.5),4:(-.75,.5),6:(1.0,1),7:(1.05,.15),3:(2,1),5:(2,1.8),10:(3,1),8:(3.6,1.6),9:(3.6,.4)}
 common=[(1,2),(1,4),(1,6),(3,5),(3,10),(6,7),(8,10),(9,10)]
 for u,v in common:ax.plot([coords[u][0],coords[v][0]],[coords[u][1],coords[v][1]],color='#566573',lw=1.7,zorder=1)
 for u,v in [(3,6),(3,7)]:ax.plot([coords[u][0],coords[v][0]],[coords[u][1],coords[v][1]],color='#bc6c25',lw=1.8,ls='--',zorder=1)
 for i in range(1,11):
  color='#1f6e58' if i in [1,3,10] else '#f1d7b8' if i==6 else '#e9eef1'
  ax.scatter(*coords[i],s=480,color=color,edgecolor='white',lw=1,zorder=2)
  ax.text(*coords[i],'Q'+str(i),ha='center',va='center',color='white' if i in [1,3,10] else '#16212b',fontsize=9,zorder=3)
 ax.set_xlim(-1.15,4.0);ax.set_ylim(-.35,2.0);ax.axis('off');ax.set_title('(a) Three guaranteed hub members',loc='left',fontweight='bold')
 ax.text(.02,-.03,'Solid links occur in both compatible trees.\nExactly one dashed link is included.',transform=ax.transAxes,va='top',fontsize=9)
 for i in range(1,11):
  name='Q'+str(i);y=10-i
  for source,offset,color,label in [(old,.12,'#607d8b','E64'),(main,-.12,'#1f6e58','FS-T64')]:
   r=source.loc[name];lo,hi=r.degree_lower,r.degree_upper
   bx.plot([lo,hi],[y+offset,y+offset],color=color,lw=2.7);bx.scatter([lo,hi],[y+offset]*2,s=16,color=color)
 bx.set_yticks(range(10),['Q'+str(i) for i in range(10,0,-1)]);bx.set_xticks([1,2,3,4]);bx.set_xlim(.7,4.3);bx.set_ylim(-.6,9.6);bx.set_xlabel('Certified degree');bx.set_title('(b) Two additional exact degrees',loc='left',fontweight='bold');bx.grid(axis='x',alpha=.16)
 from matplotlib.lines import Line2D
 bx.legend(handles=[Line2D([0],[0],color='#607d8b',lw=3,label='E64'),Line2D([0],[0],color='#1f6e58',lw=3,label='FS-T64')],loc='upper right',frameon=False)
 for sp in ['top','right']:bx.spines[sp].set_visible(False)
 fig.tight_layout(w_pad=1.3);fig.subplots_adjust(bottom=.18);fig.savefig(PAPER/'Fig_RSE_certificates.pdf',bbox_inches='tight');fig.savefig(PAPER/'Fig_RSE_certificates.png',dpi=180,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(9.4,3.6),sharey=True)
 colors={'E64':'#718391','FS-T64':'#1f6e58','FS-S64':'#bc6c25','M-C':'#6a50a2'}
 for ax,model in zip(axes,['gaussian_star','t5_star']):
  for method in colors:
   a=[lookup[model,n,method] for n in [252,1008,4096]];v=np.array([100*r['power'] for r in a]);ci=np.array([r['power_ci'] for r in a])*100
   # Roundoff can place a Wilson endpoint a few ulps past the observed
   # proportion at k=0 or k=n. Clip only plotted error lengths, not intervals.
   err=np.maximum(0,np.array([v-ci[:,0],ci[:,1]-v]))
   ax.errorbar([0,1,2],v,yerr=err,fmt='o-',lw=1.5,capsize=2,ms=4,label=method,color=colors[method])
  ax.set_xticks([0,1,2],['252','1008','4096']);ax.set_xlabel('Observations');ax.set_title(labels[model]);ax.grid(alpha=.15);ax.set_ylim(-3,103)
  for sp in ['top','right']:ax.spines[sp].set_visible(False)
 axes[0].set_ylabel('Unique-hub certification (%)');axes[1].legend(frameon=False,loc='center right');fig.tight_layout();fig.savefig(PAPER/'Fig_power.pdf');fig.savefig(PAPER/'Fig_power.png',dpi=180);plt.close(fig)
 verification=dict(passed=True,distinct_datasets=int(df[['cell','rep']].drop_duplicates().shape[0]),method_dataset_records=len(df),checks=['unique dataset-method keys','cell sizes and deterministic seeds','all 20 procedures on all 6600 datasets','Wilson intervals from integer counts'])
 (RESULTS/'aggregate_verification.json').write_text(json.dumps(verification,indent=2));print(json.dumps(verification))
if __name__=='__main__':build()
