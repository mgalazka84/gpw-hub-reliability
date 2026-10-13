"""Build manuscript tables from checked replication-level outputs."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent;R=ROOT/'results';M=ROOT/'manuscript'
names={'gaussian_star':'Gaussian star','t5_star':r'$t_5$ star','near_tie':'Near tie','independent':'Independence','discrete_independent':'Discrete null','ar_star':'AR star','ar_independent':'AR null'}
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;mid=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return mid-half,mid+half
def write(name,s):(M/name).write_text(s)
def table(rows,cols,caption,label):
 return '\\begin{table}[htbp]\n\\caption{'+caption+'}\\label{'+label+'}\n\\centering\\small\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
def run():
 e=pd.read_csv(R/'exact_replications.csv');b=pd.read_csv(R/'bootstrap_replications.csv');g=pd.read_csv(R/'gpw_refined_windows.csv');nodes=pd.read_csv(R/'gpw_refined_nodes.csv.gz')
 assert len(e)==8800 and len(b)==4800 and len(g)==693
 assert not e.duplicated(['p','model','n','rep','method']).any() and not b.duplicated(['p','model','n','rep','method']).any()
 assert (e.groupby(['p','model','n','method']).size()==200).all();assert (b.groupby(['p','model','n','method']).size()==100).all()
 es=e.groupby(['p','model','n','method']).mean(numeric_only=True);bs=b.groupby(['p','model','n','method']).mean(numeric_only=True)
 es.to_csv(R/'exact_summary_verified.csv');bs.to_csv(R/'bootstrap_summary_verified.csv')
 exact=e[e.method=='exact_order'];boot=b[b.method=='bootstrap_order'];gs=g.groupby('network').mean(numeric_only=True)
 rows=[r'$p$ & Model & $n$ & Order FWER & Hub: order & Hub: box \\',r'\midrule']
 for (p,model,n),grp in exact.groupby(['p','model','n'],sort=True):
  a=es.loc[(p,model,n,'exact_order')];old=es.loc[(p,model,n,'hoeffding_box')]
  hub='---' if pd.isna(a.true_hub_certified) else f'{a.true_hub_certified:.3f}'
  hb='---' if pd.isna(old.true_hub_certified) else f'{old.true_hub_certified:.3f}'
  rows.append(f'{p} & {names[model]} & {n} & {a.order_fwer:.3f} & {hub} & {hb} '+r'\\')
 txt=table(rows,'rlrrrr','Finite-sample experiment: 200 replications per cell. FWER is the frequency of any false strict order. Hub columns report certification of vertex zero as uniquely first; null cells have no unique population hub.','tab:exact')
 maximum=exact.groupby(['p','model','n']).order_fwer.mean().idxmax();mm=exact[(exact.p==maximum[0])&(exact.model==maximum[1])&(exact.n==maximum[2])];lo,hi=wilson(int(mm.order_fwer.sum()),len(mm))
 txt+=f'Across the finite-sample cells, the smallest observed simultaneous degree coverage was {exact.groupby(["p","model","n"]).degree_coverage.mean().min():.3f}, and the smallest rank coverage was {exact.groupby(["p","model","n"]).rank_coverage.mean().min():.3f}. The largest observed false-order frequency was {mm.order_fwer.mean():.3f} ({int(mm.order_fwer.sum())}/200), in the $p={maximum[0]}$, $n={maximum[2]}$ {names[maximum[1]]} cell; its Wilson 95\\% interval is [{lo:.3f}, {hi:.3f}]. A Monte Carlo proportion above 0.05 is not itself a contradiction of the finite-sample theorem. We report it rather than truncating observed errors at the nominal level.\n\n'
 txt+=r'Figure~\ref{fig:power} distinguishes hub certification from complete-tree recovery in the separated models. The full-sample rectangular baseline certified no hub at the displayed sample size. Successful hub inference need not imply that every edge is forced. Conversely, the short-sample and near-tie cells largely remain unresolved. A condition based on four uniform-band half-widths exceeding a tree separation margin is a sufficient recovery condition; failure of that condition is not an impossibility result for all Kendall methods.'+'\n'
 txt+=r'\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{Fig_power.pdf}\caption{Certification at $n=4096$, from 200 replications per cell. Hub certification uses the upper rank endpoint; full recovery requires all $p-1$ edges to be forced by screening. Whiskers are Wilson 95\% Monte Carlo intervals. The Hoeffding-box hub rate is zero in all four cells.}\label{fig:power}\end{figure}'+'\n'
 write('simulation_results.tex',txt)
 rows=[r'Model & $n$ & Box cover & Contrast cover & Order FWER \\',r'\midrule']
 for (p,model,n),grp in boot[boot.p==52].groupby(['p','model','n']):
  a=bs.loc[(p,model,n,'bootstrap_order')];old=bs.loc[(p,model,n,'bootstrap_box')]
  rows.append(f'{names[model]} & {n} & {old.input_coverage:.2f} & {a.input_coverage:.2f} & {a.order_fwer:.2f} '+r'\\')
 txt=table(rows,'lrrrr','Pearson-bootstrap calibration at $p=52$: 100 data sets and 499 resamples per cell. Coverage concerns the whole input region, not a selected edge. Full results for $p=12$ are supplied with the replication files.','tab:bootstrap')
 mincell=boot.groupby(['p','model','n']).input_coverage.mean().idxmin();bc=boot[(boot.p==mincell[0])&(boot.model==mincell[1])&(boot.n==mincell[2])];lo,hi=wilson(int(bc.input_coverage.sum()),100)
 txt+=f'The smallest contrast-region coverage is {bc.input_coverage.mean():.2f}, in the $p={mincell[0]}$, $n={mincell[2]}$ {names[mincell[1]]} cell (Wilson interval [{lo:.3f}, {hi:.3f}]). For Gaussian independence at $p=52,n=252$, contrast coverage is {bs.loc[(52,"independent",252,"bootstrap_order"),"input_coverage"]:.2f} and false-order frequency is {bs.loc[(52,"independent",252,"bootstrap_order"),"order_fwer"]:.2f}. Under the exact null, every false strict order is a calibration error even if the downstream tree family remains broad enough to cover every degree. Projected degree coverage therefore cannot be used as a substitute for input calibration.\n\n'
 write('bootstrap_results.tex',txt)
 rows=[r'Representation & Minor width & Path-LP width & Full ranks & Hubs \\',r'\midrule']
 gn={'raw':'Raw','demeaned':'Demeaned','factor_residual':'WIG residual'}
 for kind in ['raw','demeaned','factor_residual']:
  a=gs.loc[kind];rows.append(f'{gn[kind]} & {a.minor_mean_degree_width:.2f} & {a.mean_degree_width:.2f} & {a.full_rank_fraction:.2f} & {int(g[g.network==kind].top1_certified.sum())} '+r'\\')
 txt=table(rows,'lrrrr','All 231 windows per representation. Widths are averages of the within-window mean degree width. Full ranks is the mean fraction of vertices with interval $[1,p]$. Hubs counts certified unique hubs over all windows. These are nominal bootstrap diagnostics.','tab:gpw')
 reduction=(gs.loc['raw','minor_mean_degree_width']-gs.loc['raw','mean_degree_width'])/gs.loc['raw','minor_mean_degree_width']*100
 txt+=f'The refined projection reduces the raw-network mean degree width from {gs.loc["raw","minor_mean_degree_width"]:.2f} to {gs.loc["raw","mean_degree_width"]:.2f}, a {reduction:.1f}\\% reduction relative to the minor. It does not resolve hub ranks: all {len(nodes):,} reported vertex--window--representation rank intervals remain $[1,p]$. No unique hub and no top-decile membership is certified. There are {int(g.nontrivial_dual_certificates.sum()):,} saved rational certificates for improved degree or anchor-contrast endpoints. These improvements demonstrate a stricter feasible-set projection, but not an informative financial hub ranking.\n\n'
 txt+=r'The all-ranks result is specific to these windows, this calibration and these outer projections. It does not prove that no population hub exists or that no other statistically justified procedure could identify one. It also does not validate the nominal error rate: the calibration failures in Table~\ref{tab:bootstrap} remain relevant.'+'\n'
 write('gpw_results.tex',txt)
 v=json.loads((R/'gpw_certificate_verification.json').read_text());assert v['status']=='PASS' and v['windows']==693 and v['certificates']==int(g.nontrivial_dual_certificates.sum())
 write('verification_results.tex',f'For the empirical run, independent Python-integer verification passed for all {v["certificates"]:,} saved improving certificates in all {v["windows"]} origin--representation combinations. Source-input hashes, fixed seeds, full replication records, derived order relations and the verifier are retained. No raw third-party price series is required to check the saved optimization certificates.\n')
 # Scientific figure: inference of a functional need not recover the object.
 plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
 fig,ax=plt.subplots(figsize=(6.4,3.2));xx=np.arange(4);labels=[]
 for j,(p,model) in enumerate([(12,'gaussian_star'),(12,'t5_star'),(52,'gaussian_star'),(52,'t5_star')]):
  d=exact[(exact.p==p)&(exact.model==model)&(exact.n==4096)];labels.append(f'p={p}\n'+('Gaussian' if model=='gaussian_star' else '$t_5$'))
  for off,key,color,label in [(-.18,'hub','#185b88','Hub certified'),(.18,'tree','#c47a28','All edges forced')]:
   k=int(d.true_hub_certified.sum()) if key=='hub' else int((d.forced_edges==p-1).sum());rate=k/len(d);ll,hh=wilson(k,len(d))
   ax.bar(j+off,rate,.32,color=color,label=label if j==0 else None);ax.errorbar(j+off,rate,yerr=[[max(0,rate-ll)],[max(0,hh-rate)]],fmt='none',ecolor='#222222',capsize=3,lw=.8)
 ax.set_xticks(xx,labels);ax.set_ylabel('Proportion of 200 replications');ax.set_ylim(-.015,1.18);ax.legend(frameon=False,ncols=2,loc='upper center');ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(M/'Fig_power.pdf');fig.savefig(M/'Fig_power.png',dpi=160);plt.close(fig)
 audit={'status':v['status'],'exact_data_sets':len(e)//2,'bootstrap_data_sets':len(b)//2,'gpw_windows':len(g),'gpw_vertex_records':len(nodes),'gpw_all_ranks_full':bool((g.full_rank_fraction==1).all()),'gpw_certificates':v['certificates'],'exact_worst_fwer':float(mm.order_fwer.mean()),'bootstrap_worst_contrast_coverage':float(bc.input_coverage.mean())}
 (R/'aggregate_verification.json').write_text(json.dumps(audit,indent=2));print(audit)
if __name__=='__main__':run()
