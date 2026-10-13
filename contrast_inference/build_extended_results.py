"""Report the entire prespecified empirical grid and paired comparisons."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_results import table
ROOT=Path(__file__).parent;R=ROOT/'results';M=ROOT/'manuscript'

def run():
 w=pd.read_csv(R/'gpw_extended_windows.csv');nodes=pd.read_csv(R/'gpw_extended_nodes.csv.gz');v=json.loads((R/'gpw_extended_verification.json').read_text());assert v['status']=='PASS' and len(w)==466
 main=w[w.analysis=='primary'];raw=main[main.network=='raw'];paired=[];rows=[r'$p$ & Origins & $n$ & Degree width & Forced edges & Full ranks \\',r'\midrule']
 for cap in [10,20,40]:
  common=set(raw[(raw.cap==cap)&(raw.lookback==1008)].origin);d=raw[(raw.cap==cap)&raw.origin.isin(common)]
  for n,z in d.groupby('lookback'):
   x=dict(cap=cap,lookback=n,origins=len(common),degree_width=float(z.mean_degree_width.mean()),forced=float(z.forced_edges.mean()),rank_width_fraction=float(z.rank_width_fraction.mean()));paired.append(x)
   rows.append(f'{cap} & {len(common)} & {n} & {x["degree_width"]:.2f} & {x["forced"]:.2f} & {1-z.nonfull_rank_fraction.mean():.2f} '+r'\\')
 paired=pd.DataFrame(paired);paired.to_csv(R/'gpw_extended_paired_summary.csv',index=False)
 text=r'''\subsection{Longer windows, nested liquid universes and forward diagnostics}\label{sec:extended}
The full-panel result motivates a sensitivity analysis that separates sample length from changing stock composition. Before computing the extended outcomes, we fixed the last recorded origin in every represented calendar year, giving 19 annual origins. This is a retrospective sensitivity grid, not a preregistration or an independent replication of method development. At each origin, we use the longest calendar-available lookback from 504 and 1008 sessions to screen the original eligible list. Eligibility requires complete past returns, supplied prices and volumes, positive volume on at least 90\% of sessions, at most 15\% zero returns, and positive variance. Within this eligible set we rank stocks by median supplied closing price times volume, breaking ties by ISIN. This is an available-data liquidity proxy, not a certified corporate-action-adjusted traded-value measure.

We take nested groups of 10, 20 and 40 stocks. Their membership is kept fixed when comparing 252, 504 and 1008 sessions at the same origin. Changing group size changes the population tree target; a hub of a small liquid subnetwork need not be a hub of the full market. Demeaning is recomputed within each selected group, while factor residualization retains the same WIG series. All dated observations used for selection precede or coincide with the origin; future observations never enter selection. Vendor histories can incorporate later revisions, so this rule does not establish a fully point-in-time data source.

Of 513 planned origin--lookback--group--representation combinations, 81 cannot be evaluated because of insufficient calendar history or too few eligible stocks. We retain the full exclusion list. The remaining 432 primary analyses use block length ten and 999 common-row resamples per origin and lookback, across all three representations. A further 34 raw-network analyses examine block lengths five and twenty at $p=20,n=1008$, for all 17 available origins. These are nominal, per-configuration bootstrap projections. We do not claim simultaneous coverage over the grid or valid 95\% coverage for dependent returns.

'''
 text+=table(rows,'rrrrrr','Raw-network comparison on identical stocks and matched origins. The 1008-session availability criterion leaves 17 origins for groups of 10 and 20, and ten for groups of 40. Widths and forced-edge counts are within-group origin averages. Full ranks is the fraction of vertex ranges remaining $[1,p]$.','tab:extended')
 improvements={}
 for cap in [10,20,40]:
  d=paired[paired.cap==cap].set_index('lookback');improvements[cap]=100*(1-d.loc[1008,'degree_width']/d.loc[252,'degree_width'])
 text+=f'The matched-origin mean degree width decreases by {improvements[10]:.1f}\\%, {improvements[20]:.1f}\\% and {improvements[40]:.1f}\\% when moving from 252 to 1008 observations for groups of 10, 20 and 40 respectively. More edges are forced under the nominal order region, but none of the 432 primary or 34 block-sensitivity configurations yields a non-full vertex rank interval or a certified unique hub. The negative ranking result therefore survives a fourfold lookback extension, liquidity-based dimension reduction, and the stated block checks. It remains a result about this procedure and these targets, not an impossibility theorem.\n\n'
 block=w[w.analysis=='block_sensitivity'];base=raw[(raw.lookback==1008)&(raw.cap==20)]
 records=[];rrows=[r'Block length & Origins & Degree width & Forced edges & Full ranks \\',r'\midrule']
 for length,d in [(5,block[block.block==5]),(10,base),(20,block[block.block==20])]:
  records.append(dict(block=length,origins=len(d),degree_width=float(d.mean_degree_width.mean()),forced_edges=float(d.forced_edges.mean()),nonfull_rank_fraction=float(d.nonfull_rank_fraction.mean())))
  rrows.append(f'{length} & {len(d)} & {d.mean_degree_width.mean():.2f} & {d.forced_edges.mean():.2f} & {1-d.nonfull_rank_fraction.mean():.2f} '+r'\\')
 pd.DataFrame(records).to_csv(R/'gpw_extended_block_summary.csv',index=False)
 text+=table(rrows,'rrrrr','Block-length sensitivity for the same 17 raw-network origins, $p=20,n=1008$. Resampling seeds differ between block lengths.','tab:blocks')
 text+=r'''For a forward diagnostic, we hold the training stock group fixed and estimate a new tree using the next 252 sessions. The diagnostic is available only when every selected stock and the factor have complete future returns. We neither impute unavailable histories nor drop a future-missing stock from the selected group. There are 93 complete primary configurations per representation, out of 144; the remaining 51 have incomplete future histories or insufficient follow-up. Overlapping lookbacks and nested groups mean these are not 93 independent samples. Forward summaries are conditional on future-history availability and may be affected by attrition; no reweighting corrects that selection. We compute the fraction of transitive training order statements whose direction agrees with the future correlation estimates, and whether the training anchor remains tied for largest future degree. These are descriptive temporal-stability measures, not population coverage tests or portfolio-performance results.

'''
 forward=[];rrows=[r'Representation & Configurations & Order agreement & Anchor still top \\',r'\midrule'];gn={'raw':'Raw','demeaned':'Demeaned','factor_residual':'WIG residual'}
 for kind in ['raw','demeaned','factor_residual']:
  d=main[(main.network==kind)&main.future_available];num=int((d.future_anchor_best_rank==1).sum());rate=float(d.future_order_agreement.mean());rec=dict(network=kind,configurations=len(d),mean_order_agreement=rate,anchor_still_top=num,anchor_still_top_fraction=num/len(d));forward.append(rec)
  rrows.append(f'{gn[kind]} & {len(d)} & {100*rate:.1f}\\% & {num}/{len(d)} '+r'\\')
 pd.DataFrame(forward).to_csv(R/'gpw_extended_forward_summary.csv',index=False)
 text+=table(rrows,'lrrr','Forward diagnostics over the next 252 sessions. Order agreement averages the within-configuration fractions, giving each configuration equal weight. Anchor still top includes ties and does not establish a unique hub.','tab:forward')
 text+=r'''Estimated edge-order directions persist much more frequently than the identity of the estimated highest-degree vertex. This comparison involves different functionals: preservation of many inequalities is not equivalent to preservation of a hub, and a high fraction of correct directions need not mean that every direction is preserved jointly. Because the population can change over time, the forward sample is not a known-truth coverage benchmark. The calibration failures in Section~\ref{sec:simulation} remain unresolved.

'''
 # Transparent illustration rule, distinct from grid-level evidence.
 example=raw[raw.future_available&(raw.anchor_bootstrap_top_tied_frequency>=.9)&(raw.future_anchor_best_rank>1)].sort_values(['origin','lookback','cap']).iloc[0]
 matching=nodes[(nodes.origin==example.origin)&(nodes.lookback==example.lookback)&(nodes.cap==example.cap)&(nodes.network=='raw')&(nodes.block==10)&nodes.is_anchor].iloc[0]
 text+=f'As an illustration, the earliest complete raw-network configuration with anchor bootstrap frequency at least 0.9 and loss of top position in the future sample is the {example.origin[:4]} annual origin, $p={example.cap},n={example.lookback}$. The anchor is {example.anchor_name}; it is tied for highest degree in {100*example.anchor_bootstrap_top_tied_frequency:.1f}\\% of bootstrap trees, while its nominal degree range is [{matching.degree_lower}, {matching.degree_upper}] and rank range is [1, {example.cap}]. Its future estimated rank range is [{int(example.future_anchor_best_rank)}, {int(example.future_anchor_worst_rank)}], despite {100*example.future_order_agreement:.1f}\\% directional agreement for the certified training orders. The illustration is selected retrospectively by the stated rule; all configurations are reported, and it is not additional confirmatory evidence.\n\n'
 text+=r'\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{Fig_empirical.pdf}\caption{Extended empirical diagnostics. Left: mean raw-network degree width divided by $p-2$, on the matched origins and fixed stock groups in Table~\ref{tab:extended}. Right: forward edge-order agreement and retention of the training anchor among the highest-degree vertices, from Table~\ref{tab:forward}. Both panels are descriptive; neither supplies a validated confidence level for GPW returns.}\label{fig:empirical}\end{figure}'+'\n'
 text+=f'Independent verification reconstructs the selection rule for all 50 origin--group universes, checks five source-input hashes and all 9,710 additional vertex records, recomputes the 303 available forward analyses including block sensitivities, and validates all {v["rational_certificates"]:,} improving rational LP certificates. Every check passes. Longer financial windows increase sample size while raising the possibility of regime changes, so their improved degree resolution must be interpreted alongside that stationarity limitation.\n'
 (M/'extended_gpw_results.tex').write_text(text)
 plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
 fig,axs=plt.subplots(1,2,figsize=(7.0,3.15));ax=axs[0]
 for cap,col in [(10,'#1e6189'),(20,'#c07829'),(40,'#527e58')]:
  d=paired[paired.cap==cap];ax.plot(d.lookback,d.degree_width/(cap-2),'-o',color=col,label=f'p={cap}',lw=1.5,ms=4)
 ax.set_xticks([252,504,1008]);ax.set_xlabel('Training sessions');ax.set_ylabel('Degree width / (p - 2)');ax.set_ylim(.6,1);ax.legend(frameon=False,fontsize=8);ax.set_title('Fixed groups and matched origins',fontsize=9)
 ax=axs[1];xx=np.arange(3)
 for off,key,label,col in [(-.18,'mean_order_agreement','Order agreement','#1e6189'),(.18,'anchor_still_top_fraction','Anchor still top','#c07829')]:
  ax.bar(xx+off,[r[key] for r in forward],.34,label=label,color=col)
 ax.set_xticks(xx,['Raw','Demeaned','WIG residual']);ax.set_ylim(0,1.12);ax.set_ylabel('Forward proportion');ax.legend(frameon=False,fontsize=8,loc='upper center');ax.set_title('Next 252 sessions',fontsize=9)
 for ax in axs:ax.spines[['top','right']].set_visible(False)
 fig.tight_layout();fig.savefig(M/'Fig_empirical.pdf');fig.savefig(M/'Fig_empirical.png',dpi=170);plt.close(fig)
 stats=dict(status='PASS',analyses=len(w),primary_analyses=len(main),block_analyses=len(block),vertex_records=len(nodes),all_ranks_full=bool((w.nonfull_rank_fraction==0).all()),rational_certificates=v['rational_certificates'],paired_degree_width_reductions_percent=improvements,forward=forward,illustration=example.to_dict())
 (R/'gpw_extended_report_numbers.json').write_text(json.dumps(stats,indent=2));print(stats)
if __name__=='__main__':run()
