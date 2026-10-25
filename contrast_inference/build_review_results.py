"""Build publication tables/figures only from complete, independently checked runs."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_comparative_simulations import design

ROOT=Path(__file__).parent;R=ROOT/'results';M=ROOT/'manuscript'
METHODS=['paired_exact','merged_8','evalue_8','evalue_64','u_hoeffding','hoeffding_box','multiplier_contrast','multiplier_box']
DISPLAY=['paired_exact','merged_8','evalue_8','evalue_64','multiplier_contrast','multiplier_box']
SHORT=dict(zip(METHODS,['P','M8','E8','E64','U-H','B-H','M-C','M-B']))
LABEL=dict(zip(METHODS,['Paired exact','Merged p, 8','Evidence, 8','Evidence, 64','U Hoeffding','Box Hoeffding','Multiplier contrasts','Multiplier box']))
MODELS={'gaussian_star':'Gaussian star','t5_star':r'$t_5$ star','near_tie':'Near-tie star','independent':'Gaussian null','discrete_independent':'Discrete null','chain':'Chain','two_hubs':'Two hubs'}

def wilson(k,n):
    z=norm.ppf(.975);p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (0.0 if k==0 else max(0.0,c-h)),(1.0 if k==n else min(1.0,c+h))

def table(caption,label,headers,rows,align=None,placement='htbp'):
    align=align or 'l'+'r'*(len(headers)-1)
    return '\n'.join([r'\begin{table}['+placement+r']\centering\small',r'\setlength{\tabcolsep}{4pt}',
        r'\caption{'+caption+r'}\label{'+label+r'}',r'\begin{lrbox}{\resultsbox}\begin{tabular}{'+align+'}',r'\toprule',
        ' & '.join(headers)+r'\\',r'\midrule',*[' & '.join(map(str,row))+r'\\' for row in rows],
        r'\bottomrule\end{tabular}\end{lrbox}\ifdim\wd\resultsbox>\linewidth\resizebox{\linewidth}{!}{\usebox{\resultsbox}}\else\usebox{\resultsbox}\fi\end{table}',''])

def load_complete(from_records=False):
    config=design();base=R/'comparative'/'production';extension=R/'comparative'/'family_null_extension'
    expected=sum(c['repetitions'] for c in config['cells'])
    if from_records:
        df=pd.read_csv(R/'comparative'/'replications_all.csv.gz')
    else:
        assert len(list(base.glob('cell*.json')))==expected,'Incomplete primary Monte Carlo'
        assert len(list(extension.glob('cell*.json')))==3600,'Incomplete complete-family null extension'
        rows=[x for p in sorted(base.glob('cell*.json'))+sorted(extension.glob('cell*.json')) for x in json.loads(p.read_text())]
        evalue=R/'evalue'/'production'
        assert len(list(evalue.glob('cell*.json')))==expected,'Incomplete evidence-aggregation validation'
        rows += [x for p in sorted(evalue.glob('cell*.json')) for x in json.loads(p.read_text())]
        df=pd.DataFrame(rows)
    # Chain/two-hub populations have tied highest degrees, so no true unique hub.
    df.loc[df.model.isin(['chain','two_hubs']),'true_hub_certified']=np.nan
    key=['p','n','model','rep','method','family','stage'];assert not df.duplicated(key).any()
    assert len(df[(df.method=='paired_exact')&(df.family=='local')&(df.stage=='joint')])==expected
    for c in config['cells']:
        sub=df[(df.p==c['p'])&(df.n==c['n'])&(df.model==c['model'])&(df.family=='local')&(df.stage=='joint')]
        counts=sub.groupby('method').size()
        assert set(counts.index)==set(METHODS)
        assert all(counts==c['repetitions'])
    group=['p','model','n','method','family','stage'];summary=df.groupby(group).mean(numeric_only=True).reset_index()
    summary=summary.merge(df.groupby(group).size().rename('repetitions').reset_index(),on=group)
    for target in ['order_fwer','degree_coverage','rank_coverage','true_hub_certified']:
        bounds=[wilson(round(v*n),n) if pd.notna(v) else (np.nan,np.nan) for v,n in zip(summary[target],summary.repetitions)]
        summary[target+'_wilson_lower']=[v[0] for v in bounds];summary[target+'_wilson_upper']=[v[1] for v in bounds]
    if not from_records:
        df.to_csv(R/'comparative'/'replications_all.csv.gz',index=False)
    summary.to_csv(R/'comparative'/'summary_all.csv',index=False)
    return df,summary,config

def cell(summary,p,model,n,method,family='local',stage='joint'):
    out=summary[(summary.p==p)&(summary.model==model)&(summary.n==n)&(summary.method==method)&(summary.family==family)&(summary.stage==stage)]
    assert len(out)==1,(p,model,n,method,family,stage)
    return out.iloc[0]

def build_comparative(df,s,config):
    text=r'''\subsection{Calibration and power results}
Table~\ref{tab:nullnew} reports null FWER for the common minor-plus-joint-rank projection. Exact methods have their finite-sample guarantee; the multiplier methods are evaluated empirically. All cellwise Wilson intervals, simultaneous coverage outcomes and replication records accompany the article.
'''
    rows=[]
    for p,model,n in [(12,'independent',252),(12,'independent',1008),(12,'independent',4096),(12,'discrete_independent',1008),(52,'independent',1008)]:
        rows.append([MODELS[model]+f', {p}, {n}']+[f"{cell(s,p,model,n,m).order_fwer:.3f}" for m in DISPLAY])
    text+=table('False-order FWER at nominal 0.05; 1,000 datasets in each null cell. P: one-matching exact; M8: merged p-values; E8/E64: evidence aggregation over 8/64 matchings; M-C: multiplier contrasts; M-B: multiplier rectangle. Both Hoeffding controls have zero false orders in these cells. All methods use the local family.','tab:nullnew',['Model, $p$, $n$']+[SHORT[m] for m in DISPLAY],rows)
    exact=s[(s.family=='local')&(s.stage=='joint')&(s.method=='paired_exact')&(s.model.isin(['independent','discrete_independent']))]
    worst=exact.sort_values('order_fwer').iloc[-1];lo,hi=wilson(round(worst.order_fwer*worst.repetitions),worst.repetitions)
    text+=f"The largest one-matching null estimate is {worst.order_fwer:.3f}, with an individual Wilson interval [{lo:.3f},{hi:.3f}] ({MODELS[worst.model]}, $p={int(worst.p)}$, $n={int(worst.n)}$). This is a cellwise Monte Carlo summary; simultaneous calibration is established by the exact-test argument.\n"
    rows=[]
    for p,model,n in [(12,'gaussian_star',1008),(12,'gaussian_star',4096),(12,'t5_star',1008),(12,'t5_star',4096),(52,'gaussian_star',4096),(52,'t5_star',4096),(12,'near_tie',4096),(52,'near_tie',4096)]:
        rows.append([MODELS[model]+f', {p}, {n}']+[f"{cell(s,p,model,n,m).true_hub_certified:.3f}" for m in DISPLAY])
    text+=table('Probability of declaring the true unique hub using the same minor and joint-rank projection. Local comparisons; 200 repetitions for the displayed $p=12$ star cells and 100 for $p=52$. Both Hoeffding controls declare no unique hub in these star cells; full results are archived. Path projection is evaluated separately on paired subsets.','tab:powernew',['Model, $p$, $n$']+[SHORT[m] for m in DISPLAY],rows)
    text+=r'''Power is interpreted jointly with FWER. The multiplier contrast procedure exceeds nominal error in some IID null cells; higher declaration rates in those regimes do not establish valid superiority. The exact evidence methods provide a comparison whose error guarantee does not rely on the Gaussian approximation.
'''
    text+=r'''Figure~\ref{fig:powernew} compares the full sample-size grids. The merged rule is assessed as an exact alternative, rather than assumed to recover U-statistic efficiency. The bounded-range U-statistic comparator shows whether using all pairs alone is sufficient; the matched multiplier rectangle and contrast procedures isolate the effect of shared-edge covariance and direct calibration.
\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{Fig_comparative.pdf}
\caption{Unique-hub declarations under the common joint projection. Curves use matched data; intervals are individual 95\% Wilson Monte Carlo intervals. All specified sample sizes and both separated distributions are shown.}\label{fig:powernew}\end{figure}
\subsection{Projection ablation}
'''
    rows=[];gains=[]
    for model in ['gaussian_star','near_tie','chain','two_hubs']:
        for method in ['paired_exact','evalue_64','multiplier_contrast']:
            sub=df[(df.p==12)&(df.n==1008)&(df.model==model)&(df.method==method)&(df.family=='local')&(df.rep<100)]
            means=sub.groupby('stage').mean(numeric_only=True)
            rows.append([MODELS[model],SHORT[method],f"{means.loc['joint','mean_degree_width']:.2f}",f"{means.loc['path','mean_degree_width']:.2f}",f"{means.loc['marginal','mean_rank_width']:.2f}",f"{means.loc['joint','mean_rank_width']:.2f}",f"{means.loc['path','mean_rank_width']:.2f}"])
            gains.append(means.loc['joint','mean_degree_width']-means.loc['path','mean_degree_width'])
    text+=table('Paired projection ablation at $p=12,n=1008$, using the same first 100 datasets per cell. Degree widths compare the minor with path LP. Rank widths compare marginal, joint pair/conservation and path projection. Complete results for all eight tests and sizes are archived. Width is upper minus lower endpoint.','tab:ablation',['Model','Test',r'$W_d$ minor',r'$W_d$ path',r'$W_r$ marg.',r'$W_r$ joint',r'$W_r$ path'],rows,align='llrrrrr')
    def projection_means(model,method):
        subset=df[(df.p==12)&(df.n==1008)&(df.model==model)&(df.method==method)&(df.family=='local')&(df.rep<100)]
        return subset.groupby('stage').mean(numeric_only=True)
    gaussian=projection_means('gaussian_star','paired_exact')
    chain=projection_means('chain','paired_exact')
    chain_e=projection_means('chain','evalue_64')
    near=projection_means('near_tie','multiplier_contrast')
    text+=f"Table~\\ref{{tab:ablation}} locates the path-projection gain mainly in degree precision. For one-matching inference, Gaussian-star mean degree width decreases from {gaussian.loc['joint','mean_degree_width']:.2f} to {gaussian.loc['path','mean_degree_width']:.2f}; in the chain it decreases from {chain.loc['joint','mean_degree_width']:.2f} to {chain.loc['path','mean_degree_width']:.2f}. The corresponding E64 chain widths decrease from {chain_e.loc['joint','mean_degree_width']:.2f} to {chain_e.loc['path','mean_degree_width']:.2f}. Most exact rank widths remain approximately 11, the full-range width at $p=12$. In the near-tie multiplier example, the path stage decreases mean rank width only from {near.loc['joint','mean_rank_width']:.2f} to {near.loc['path','mean_rank_width']:.2f}. Thus these $n=1008$ simulations demonstrate useful degree restrictions but little additional rank resolution: tighter degree bounds need not establish a positive degree difference between two vertices.\n\n"
    text+=r"Population degree ties also impose an irreducible rank width. Even with the exact Gaussian-star tree known, its eleven leaves occupy ranks 2--12 under the tie-inclusive target, giving mean rank width $110/12=9.17$. For the chain and two-hub trees, the corresponding mean is $92/12=7.67$. The displayed multiplier results at these widths therefore retain the required tied ranks. Degree-conservation and direct-pair rank outputs are archived separately from the path stage."+'\n\n'
    paths=df[(df.stage=='path')&(df.family=='local')];solves=int(paths.lp_solves.sum());certs=int(paths.verified_certificates.sum())
    complete_paths=df[(df.stage=='path')&(df.family=='all')]
    complete_solves=int(complete_paths.lp_solves.sum());complete_certs=int(complete_paths.verified_certificates.sum())
    text+=f"Across all sample sizes and methods in the $p=12$ simulation path stage, the local family accounts for {solves:,} LP objectives and {certs:,} independently verified improving rational certificates. The complete family adds {complete_solves:,} objectives and {complete_certs:,} certificates. The totals across both families are therefore {solves+complete_solves:,} LP objectives and {certs+complete_certs:,} certificates. These totals include all evaluated path-stage cells, rather than only the subset displayed in Table~\\ref{{tab:ablation}}.\n".replace(',','{,}')
    text+=r'\subsection{Comparison-family ablation}'+'\n'
    rows=[]
    for model in ['gaussian_star','t5_star','near_tie','chain','two_hubs']:
        for method in ['paired_exact','evalue_64','multiplier_contrast']:
            means=[]
            for family in ['local','all']:
                sub=df[(df.p==12)&(df.n==4096)&(df.model==model)&(df.method==method)&(df.family==family)&(df.stage=='joint')&(df.rep<100)]
                means.append(sub.mean(numeric_only=True))
            a,b=means;rows.append([MODELS[model],SHORT[method],f'{a.closed_orders:.1f}',f'{b.closed_orders:.1f}',f'{a.mean_degree_width:.2f}',f'{b.mean_degree_width:.2f}',f'{a.mean_rank_width:.2f}',f'{b.mean_rank_width:.2f}'])
    text+=table('Matched local/complete family ablation at $p=12,n=4096$, 100 datasets per cell. Closed-order counts use the same $m(m-1)$ potential ordered pairs in either family. More tested comparisons also incur a larger multiplicity penalty.','tab:family',['Model','Test','Orders L','Orders A',r'$W_d$ L',r'$W_d$ A',r'$W_r$ L',r'$W_r$ A'],rows,align='llrrrrrr')
    text+=r"Table~\ref{tab:family} shows a model-dependent trade-off. For one-matching inference in the Gaussian star, complete comparisons increase mean degree width from 0.14 to 0.25 despite increasing the closed-order count from 113.0 to 530.7. In the near-tie E64 example, width increases from 5.53 to 6.54. Conversely, complete comparisons improve the one-matching chain width from 0.14 to 0.00 and the two-hub width from 0.99 to 0.38; for E64 in the two-hub model they reduce it from 0.97 to 0.02."+'\n\n'
    text+=r"The procedures recalibrate their tested families, so their rejected-order relations need not be nested. At $p=12$, the local and complete families contain 1,320 and 4,290 ordered hypotheses. For the Holm methods, the first threshold decreases from $0.05/1320$ to $0.05/4290$; multiplier contrasts also recalibrate a larger maximum. This can remove weak local comparisons important to screening incident hub edges, while adding many nonlocal orders. The number of certified orders alone therefore does not determine degree resolution. Holding one valid relation fixed and adding further valid comparisons can only tighten exact bounds over the resulting outer family; changing the calibrated relation is a different comparison. Complete testing is useful when its additional tree restrictions outweigh its multiplicity cost, as in the chain and two-hub examples."+'\n'
    text+=r'\subsection{Frequency-reference results}'+'\n'
    f=pd.read_csv(R/'frequency_reference'/'summary.csv');assert len(f)==6 and int(f.repetitions.sum())==2800
    rows=[]
    for _,row in f.sort_values(['model','n']).iterrows():
        power='--' if pd.isna(row.true_hub_certified) else f'{row.true_hub_certified:.3f}'
        rows.append([MODELS[row.model],int(row.n),int(row.repetitions),power,f'{row.false_hub_declaration:.3f}',f'{row.false_forced_edge:.3f}'])
    text+=table('Practical row-bootstrap frequency rule on matched Gaussian samples, $p=12$, 499 draws. Hub declarations require unique-hub frequency at least 0.95; edge declarations require selection frequency at least 0.95. False edge declarations are familywise events. Frequencies have no general confidence guarantee.','tab:frequency',['Model','$n$','$R$','True hub','False hub','False edge'],rows,align='lrrrrr',placement='H')
    text+=r'''The frequency reference targets estimated-tree stability. Under the Gaussian null every edge can occur in an optimal population tree, but none is necessary and no vertex is a population unique hub. A high-frequency edge declaration is therefore counted as false necessity in that setting. This definition avoids interpreting frequent selection as a simultaneous population certificate.
'''
    (M/'comparative_results.tex').write_text(text)
    colors=['#30343b','#bd6a00','#64a28d','#136a48','#1766ad','#7d4fa6'];markers=['o','s','v','P','D','^']
    fig,axes=plt.subplots(2,2,figsize=(6.4,5.3),sharex='col',sharey=True)
    for ax,(p,model) in zip(axes.ravel(),[(12,'gaussian_star'),(52,'gaussian_star'),(12,'t5_star'),(52,'t5_star')]):
        for method,color,marker in zip(DISPLAY,colors,markers):
            d=s[(s.p==p)&(s.model==model)&(s.method==method)&(s.family=='local')&(s.stage=='joint')].sort_values('n')
            y=d.true_hub_certified.to_numpy();lower=d.true_hub_certified_wilson_lower.to_numpy();upper=d.true_hub_certified_wilson_upper.to_numpy()
            ax.errorbar(d.n,y,yerr=np.array([y-lower,upper-y]),color=color,marker=marker,label=LABEL[method],linewidth=1.3,markersize=4,capsize=2)
        ax.set_xscale('log',base=2);ax.set_xticks([252,1008,4096] if p==12 else [252,4096]);ax.set_xticklabels([252,1008,4096] if p==12 else [252,4096]);ax.set_ylim(-.035,1.035);ax.grid(alpha=.2)
        ax.set_title(('Gaussian' if model=='gaussian_star' else r'$t_5$')+f' star, p={p}')
    for ax in axes[:,0]:ax.set_ylabel('True unique-hub declaration rate')
    for ax in axes[1]:ax.set_xlabel('Sample size n')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='upper center',frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(M/'Fig_comparative.pdf',bbox_inches='tight');fig.savefig(M/'Fig_comparative.png',dpi=180,bbox_inches='tight');plt.close(fig)
    return dict(datasets=7800,cells=24,local_lp_solves=solves,local_verified_certificates=certs,
        all_lp_solves=int(df.loc[df.stage=='path','lp_solves'].sum()),
        all_verified_certificates=int(df.loc[df.stage=='path','verified_certificates'].sum()),
        worst_exact_null_fwer=float(worst.order_fwer),worst_exact_null_interval=[lo,hi])

def build_audit():
    df=pd.read_csv(R/'bootstrap_audit'/'replications.csv');assert len(df)==12000
    group=['p','n','method'];s=df.groupby(group).mean(numeric_only=True).reset_index()
    s=s.merge(df.groupby(group).size().rename('repetitions').reset_index(),on=group);assert (s.repetitions==1000).all()
    bounds=[wilson(round(v*n),n) for v,n in zip(s.order_fwer,s.repetitions)]
    s['wilson_lower']=[v[0] for v in bounds];s['wilson_upper']=[v[1] for v in bounds];s.to_csv(R/'bootstrap_audit'/'summary_intervals.csv',index=False)
    names={'legacy_row_499':'Row, bootstrap SD, 499','legacy_row_1999':'Row, bootstrap SD, 1999',
        'influence_row_1999':'Row, influence SD, 1999','influence_multiplier_1999':'Gaussian, influence SD, 1999'}
    rows=[];order=list(names)
    for p,n in [(12,252),(52,252),(52,1008)]:
        for method in order:
            a=s[(s.p==p)&(s.n==n)&(s.method==method)].iloc[0]
            rows.append([f'{p}, {n}',names[method],f'{a.order_fwer:.3f}',f'[{a.wilson_lower:.3f}, {a.wilson_upper:.3f}]'])
    text=table('Paired Gaussian IID null audit. Each procedure uses the same 1,000 datasets in a cell; 499 row draws are nested in 1,999. FWER equals input undercoverage in this all-ties population. Intervals are individual Wilson Monte Carlo intervals.','tab:audit',['$p$, $n$','Calibration, studentizer, $B$','FWER','95\% MC interval'],rows,align='llrr')
    paired=[]
    for p,n in [(12,252),(52,252),(52,1008)]:
        wide=df[(df.p==p)&(df.n==n)].pivot(index='rep',columns='method',values='order_fwer')
        for a,b in [('legacy_row_499','legacy_row_1999'),('legacy_row_1999','influence_row_1999'),('influence_row_1999','influence_multiplier_1999')]:
            difference=wide[a]-wide[b];mean=float(difference.mean());se=float(difference.std(ddof=1)/np.sqrt(len(difference)))
            paired.append(dict(p=p,n=n,first=a,second=b,difference=mean,lower=mean-1.96*se,upper=mean+1.96*se))
    pd.DataFrame(paired).to_csv(R/'bootstrap_audit'/'paired_differences.csv',index=False)
    d=[x for x in paired if x['p']==52 and x['n']==1008 and x['first']=='legacy_row_499'][0]
    text+=f"At $p=52,n=1008$, the paired FWER difference (499 minus 1,999 row draws with the original studentizer) is {d['difference']:.3f}, with Monte Carlo interval [{d['lower']:.3f},{d['upper']:.3f}] based on paired replication differences. The studentizer and mechanism comparisons in Table~\\ref{{tab:audit}} provide separate diagnostics. The verified resampling identities support implementation correctness; the calibration results concern the particular finite-sample procedures.\n"
    text+=r'''\begin{figure}[htbp]\centering\includegraphics[width=\textwidth]{Fig_audit.pdf}
\caption{False-order FWER in the paired IID audit. Horizontal intervals are individual Wilson Monte Carlo intervals; the vertical line marks 0.05.}\label{fig:audit}\end{figure}
'''
    (M/'audit_results.tex').write_text(text)
    fig,axes=plt.subplots(1,3,figsize=(6.4,2.8),sharey=True)
    for ax,(p,n) in zip(axes,[(12,252),(52,252),(52,1008)]):
        sub=s[(s.p==p)&(s.n==n)].set_index('method').loc[order];y=np.arange(4)
        ax.errorbar(sub.order_fwer,y,xerr=np.array([sub.order_fwer-sub.wilson_lower,sub.wilson_upper-sub.order_fwer]),fmt='o',capsize=3,color='#1766ad')
        ax.axvline(.05,color='#b04735',linestyle='--',linewidth=1);ax.set_title(f'p={p}, n={n}');ax.set_yticks(y);ax.set_yticklabels(['Row SD, B499','Row SD, B1999','Row IF, B1999','Gaussian IF, B1999']);ax.invert_yaxis();ax.set_xlabel('False-order FWER');ax.grid(axis='x',alpha=.2);ax.set_xlim(0,max(.13,float(s.wilson_upper.max())*1.08))
    fig.tight_layout();fig.savefig(M/'Fig_audit.pdf',bbox_inches='tight');fig.savefig(M/'Fig_audit.png',dpi=180,bbox_inches='tight');plt.close(fig)
    return dict(datasets=3000,cells=3,rows=12000,paired_B_difference=d)

def build_cross():
    s=pd.read_csv(R/'cross_section'/'summary.csv');nodes=pd.read_csv(R/'cross_section'/'nodes.csv')
    extra=[json.loads(p.read_text()) for p in sorted((R/'evalue'/'cross_section').glob('dataset*.json'))]
    assert len(extra)==12,'Incomplete evidence measurement applications'
    s=pd.concat([s,pd.DataFrame([r for d in extra for r in d['rows']])],ignore_index=True)
    nodes=pd.concat([nodes,pd.DataFrame([r for d in extra for r in d['nodes']])],ignore_index=True)
    assert len(s)==192
    s.to_csv(R/'cross_section'/'summary_all.csv',index=False);nodes.to_csv(R/'cross_section'/'nodes_all.csv',index=False)
    primary=s[(s.dataset=='Abalone')&(s.stratum=='pooled')&(s.family=='local')].set_index('method').loc[METHODS]
    rows=[]
    for method,a in primary.iterrows():
        rows.append([SHORT[method],int(a.raw_orders),int(a.forced_edges),int(a.exact_degrees),int(a.certified_hubs),f'{a.minor_degree_width:.3f}',f'{a.lp_degree_width:.3f}',f'{a.mean_rank_width:.3f}'])
    text=table('Primary Abalone application, $n=4177,p=8$, local family. P, M8, E8, E64, U-H and B-H have finite-sample guarantees under the stated IID sampling model; M-C and M-B use asymptotic multiplier calibration. All methods receive the same full degree-pair path projection.','tab:abalone',['Test','Orders','Forced','Fixed degrees','Hubs',r'$W_d$ minor',r'$W_d$ path',r'$W_r$'],rows)
    x=nodes[(nodes.dataset=='Abalone')&(nodes.stratum=='pooled')&(nodes.family=='local')]
    rings=x[(x.method=='paired_exact')&(x.name=='Rings')].iloc[0];whole=x[(x.method=='multiplier_contrast')&(x.name=='Whole weight')].iloc[0]
    a=primary.loc['paired_exact'];improvement=100*(a.minor_degree_width-a.lp_degree_width)/a.minor_degree_width
    freq=float(whole.bootstrap_unique_hub_frequency)
    text+=f"For the single-matching exact reference, path projection reduces mean degree width from {a.minor_degree_width:.3f} to {a.lp_degree_width:.3f} ({improvement:.1f}\\%). Ring count has degree [{int(rings.degree_lower)},{int(rings.degree_upper)}] and rank [{int(rings.rank_lower)},{int(rings.rank_upper)}], excluding hub status under the stated model. Whole weight appears as the unique sample hub in all 499 ordinary Kendall bootstrap trees, but its one-matching exact rank remains [1,8]. The multiplier-contrast projection instead gives whole-weight degree [{int(whole.degree_lower)},{int(whole.degree_upper)}] and rank [{int(whole.rank_lower)},{int(whole.rank_upper)}]. This nominal bootstrap declaration and the exact confidence output are reported separately.\n"
    edges=pd.read_csv(R/'cross_section'/'edges.csv');shared=int(s[(s.dataset=='Abalone')&(s.stratum=='pooled')].pearson_kendall_shared_edges.iloc[0])
    text+=f"The estimated Pearson and Kendall Abalone trees share {shared} of seven edges. Both weight definitions and all edge frequencies are archived. Their agreement in this dataset does not make the population targets interchangeable outside models with a common monotone correlation ordering. Pooled and sex-stratified results, every complete-family comparison, and all Dry Bean stress outputs are supplied without selecting favorable strata.\n"
    ewhole=x[(x.method=='evalue_64')&(x.name=='Whole weight')].iloc[0]
    eprimary=primary.loc['evalue_64'];gain=100*(a.lp_degree_width-eprimary.lp_degree_width)/a.lp_degree_width
    text+=f"The exact E64 construction gives whole-weight degree [{int(ewhole.degree_lower)},{int(ewhole.degree_upper)}] and rank [{int(ewhole.rank_lower)},{int(ewhole.rank_upper)}], excluding {int(eprimary.excluded_hubs)} of the eight vertices from hub status. Its mean degree width is {eprimary.lp_degree_width:.3f}, {gain:.1f}\\% below the single-matching exact result after the same projection. It retains a finite-sample guarantee under the same sampling model; unique-hub identification and a top-three rank restriction are distinct conclusions.\n"
    text+=r'''\begin{figure}[!ht]\centering\includegraphics[width=\textwidth]{Fig_cross_section.pdf}
\caption{All eight Abalone degree and rank ranges after identical path projection. P, E8 and E64 have finite-sample guarantees under the stated IID model; M-C has asymptotic multiplier calibration. Population degree ties retain their full rank ranges.}\label{fig:cross}\end{figure}
'''
    (M/'cross_section_results.tex').write_text(text)
    fig,axes=plt.subplots(1,2,figsize=(6.4,4.3),sharey=True)
    names=x[x.method=='paired_exact'].sort_values('vertex').name.to_list();ys=np.arange(8)
    for method,offset,color,marker in [('paired_exact',-.27,'#30343b','o'),('evalue_8',-.09,'#64a28d','v'),('evalue_64',.09,'#136a48','P'),('multiplier_contrast',.27,'#1766ad','D')]:
        sub=x[x.method==method].sort_values('vertex')
        for ax,low,high in [(axes[0],'degree_lower','degree_upper'),(axes[1],'rank_lower','rank_upper')]:
            ax.hlines(ys+offset,sub[low],sub[high],color=color,linewidth=1.6)
            ax.scatter(sub[low],ys+offset,c=color,marker=marker,s=22,label=LABEL[method]);ax.scatter(sub[high],ys+offset,c=color,marker=marker,s=22)
    for ax in axes:ax.set_yticks(ys);ax.set_yticklabels(names);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    axes[0].set_xlim(.7,7.3);axes[0].set_xticks(range(1,8));axes[0].set_xlabel('Degree');axes[0].set_title('Degree ranges')
    axes[1].set_xlim(.7,8.3);axes[1].set_xticks(range(1,9));axes[1].set_xlabel('Descending rank');axes[1].set_title('Rank ranges')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(M/'Fig_cross_section.pdf',bbox_inches='tight');fig.savefig(M/'Fig_cross_section.png',dpi=180,bbox_inches='tight');plt.close(fig)
    ewhole=x[(x.method=='evalue_64')&(x.name=='Whole weight')].iloc[0]
    return dict(datasets=12,analyses=192,primary_exact_degree_width_reduction=improvement,
        rings_degree=[int(rings.degree_lower),int(rings.degree_upper)],rings_rank=[int(rings.rank_lower),int(rings.rank_upper)],
        multiplier_whole_weight_degree=[int(whole.degree_lower),int(whole.degree_upper)],
        multiplier_whole_weight_rank=[int(whole.rank_lower),int(whole.rank_upper)],
        primary_bootstrap_unique_hub_frequency=freq,pearson_kendall_shared_edges=shared,
        evidence_whole_weight_degree=[int(ewhole.degree_lower),int(ewhole.degree_upper)],
        evidence_whole_weight_rank=[int(ewhole.rank_lower),int(ewhole.rank_upper)],
        verified_certificates=int(s.verified_certificates.sum()))

def build_gpw_verification():
    base=json.loads((R/'gpw_certificate_verification.json').read_text())
    extended=json.loads((R/'gpw_extended_verification.json').read_text())
    assert base['status']==extended['status']=='PASS'
    windows=base['windows']+extended['analyses']
    certificates=base['certificates']+extended['rational_certificates']
    text=f"For the Polish-market sensitivity study, independent Python-integer verification checks {base['certificates']:,} improving rational certificates in the {base['windows']:,} base origin--representation analyses, plus {extended['rational_certificates']:,} certificates in the {extended['analyses']:,} extended analyses. The combined GPW total is {certificates:,} certificates across {windows:,} analyses. The base and extended scopes are reported separately in the appendix. Source-input hashes, fixed seeds, full replication records, derived order relations and the verifiers are retained. Raw third-party price series are not required to check the saved optimization certificates.\n"
    (M/'verification_results.tex').write_text(text.replace(',','{,}'))
    return dict(base_certificates=base['certificates'],extended_certificates=extended['rational_certificates'],total_certificates=certificates,total_analyses=windows)

def run(from_records=False):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    df,s,config=load_complete(from_records);numbers={'comparative':build_comparative(df,s,config),'audit':build_audit(),'cross_section':build_cross(),'gpw_verification':build_gpw_verification()}
    (R/'review_report_numbers.json').write_text(json.dumps(numbers,indent=2));print(json.dumps(numbers,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-records',action='store_true',help='Rebuild tables and figures from the published replication archive; no simulation caches needed.')
    run(parser.parse_args().from_records)
