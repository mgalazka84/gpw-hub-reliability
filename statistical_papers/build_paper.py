"""Build every table and vector figure from saved numerical outputs."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from scipy.stats import norm

ROOT=Path(__file__).parent;RESULTS=ROOT/'results';OUT=ROOT/'manuscript'
MODEL_NAMES={'independent':'Independent','gaussian_star':'Gaussian star','near_tie':'Near tie','ar_star':'AR star','t5_star':r'$t_5$ star'}
MODELS=list(MODEL_NAMES)
MODEL_NAMES.update({'two_hubs':'Two hubs','chain':'Chain'})
COLORS=['#777777','#0072B2','#E69F00','#009E73','#CC79A7']
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Serif','pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})

def wilson(p,n):
    z=norm.ppf(.975);den=1+z*z/n
    c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return c-h,c+h

def enrich_summary():
    s=pd.read_csv(RESULTS/'simulation_summary.csv')
    for col in ['band_coverage','degree_coverage','rank_coverage','marginal_degree_coverage','percentile_degree_coverage','true_hub_certified']:
        lo,hi=wilson(s[col],s.repetitions)
        s[col+'_wilson_lower']=lo;s[col+'_wilson_upper']=hi
    s.to_csv(RESULTS/'simulation_summary.csv',index=False)
    k=pd.read_csv(RESULTS/'kendall_summary.csv')
    for col in ['band_coverage','degree_coverage','top1_certified']:
        lo,hi=wilson(k[col],500);k[col+'_wilson_lower']=lo;k[col+'_wilson_upper']=hi
    k.to_csv(RESULTS/'kendall_summary.csv',index=False)
    return s,k

def figures(s):
    fig,axs=plt.subplots(1,2,figsize=(6.5,2.6))
    xy=np.array([[0,0],[-1,1],[-1,-1],[1,1],[1,0],[1,-1]])
    for ax,tri,title in zip(axs,[[(0,1),(0,2)],[(0,1),(1,2)]],['Tree A: hub degree 5','Tree B: hub degree 4']):
        for a,b in [(0,3),(0,4),(0,5)]:ax.plot(xy[[a,b],0],xy[[a,b],1],color='#333333',lw=1.6,zorder=1)
        for a,b in tri:ax.plot(xy[[a,b],0],xy[[a,b],1],color='#555555',lw=1.6,ls='--',zorder=1)
        ax.scatter(xy[:,0],xy[:,1],s=430,c=['#0072B2']+['#eeeeee']*5,edgecolors='#333333',zorder=2)
        for i,(x,y) in enumerate(xy):ax.text(x,y,str(i+1),ha='center',va='center',color='white' if i==0 else 'black',zorder=3)
        ax.set_xlim(-1.4,1.4);ax.set_ylim(-1.4,1.4);ax.axis('off');ax.set_title(title,fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'Fig1.pdf',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7.0,3.5))
    for model,color in zip(MODELS,COLORS):
        d=s[s.model==model].sort_values('n')
        axs[0].plot(d.n,d.band_coverage,color=color,marker='o',ls=':',lw=1.4)
        axs[0].plot(d.n,d.degree_coverage,color=color,marker='s',ls='-',lw=1.3)
        axs[1].plot(d.n,d.top1_candidate_size,color=color,marker='o',label=MODEL_NAMES[model],lw=1.3)
    axs[0].axhline(.95,color='black',lw=.8,ls='--');axs[0].set_ylim(min(.8,float(s.band_coverage.min())-.03),1.01)
    axs[0].set_ylabel('Simultaneous coverage');axs[0].set_title('Dotted: band; solid: degree',fontsize=9)
    axs[1].set_ylabel('Mean top-rank candidate count');axs[1].set_ylim(.5,12.5)
    for ax in axs:
        ax.set_xscale('log');ax.set_xticks([126,252,1008],['126','252','1008'])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('Observations');ax.grid(alpha=.18)
    handles,labels=axs[1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8,frameon=False)
    fig.tight_layout(rect=[0,.18,1,1]);fig.savefig(OUT/'Fig2.pdf',bbox_inches='tight');plt.close(fig)
    g=pd.read_csv(RESULTS/'gpw_windows.csv',parse_dates=['origin'])
    fig,ax=plt.subplots(figsize=(7,2.9))
    for kind,label,color in zip(['raw','demeaned','factor_residual'],['Raw','Demeaned','WIG residual'],COLORS[1:4]):
        d=g[g.network==kind].sort_values('origin')
        ax.plot(d.origin,d.mean_degree_width/(d.p-2),lw=1,label=label,color=color)
    ax.set_ylim(0,1.025);ax.set_ylabel('Mean degree width / (p - 2)');ax.set_xlabel('Estimation origin')
    ax.legend(loc='lower left',ncol=3,frameon=False);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'Fig3.pdf',bbox_inches='tight');plt.close(fig)

def simulation_table(s):
    lines=[r'\begin{table}[t]',r'\caption{Primary Monte Carlo results. Coverage columns are percentages; width and candidate count are averages. Each row uses 500 data sets and 999 resamples. Percentile degree coverage is not assigned a target at complete population edge ties.}\label{tab:sim}',
           r'\centering\small',r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',
           r'Model & $n$ & Band & Degree & Percentile & Width & $|C_1|$\\\midrule']
    for model in MODELS:
        d=s[s.model==model].sort_values('n')
        for j,r in enumerate(d.itertuples()):
            pct='--' if np.isnan(r.percentile_degree_coverage) else f'{r.percentile_degree_coverage*100:.1f}'
            name=MODEL_NAMES[model] if j==0 else ''
            lines.append(f'{name} & {r.n} & {r.band_coverage*100:.1f} & {r.degree_coverage*100:.1f} & {pct} & {r.mean_degree_width:.2f} & {r.top1_candidate_size:.2f}'+r'\\')
        if model!=MODELS[-1]:lines.append(r'\addlinespace')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    return '\n'.join(lines)

def kendall_table(k):
    lines=[r'\begin{table}[t]',r'\caption{Finite-sample Kendall variant under independent sampling. Coverage and certification columns are percentages. Each row uses 500 data sets. The concentration radius $\epsilon$ is specified in Corollary~\ref{cor:kendall}.}\label{tab:kendall}',
       r'\centering\small',r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',r'Model & $n$ & $\epsilon$ & Band & Degree & $|C_1|$ & Certified\\\midrule']
    for r in k.itertuples():
        lines.append(f'{MODEL_NAMES[r.model]} & {r.n} & {r.epsilon:.3f} & {100*r.band_coverage:.1f} & {100*r.degree_coverage:.1f} & {r.top1_candidate_size:.2f} & {100*r.top1_certified:.1f}'+r'\\')
    return '\n'.join(lines+[r'\bottomrule\end{tabular}',r'\end{table}'])

def gpw_table(g):
    names={'raw':'Raw','demeaned':'Demeaned','factor_residual':'WIG residual'}
    lines=[r'\begin{table}[t]',r'\caption{GPW results over 231 windows per representation. Stable windows contain at least one top-decile selection frequency of 0.80 or more in the earlier $B=500$ calculation. Width is the mean degree-envelope width from the new $B=999$ procedure. Certification refers to the nominal simultaneous top-decile criterion.}\label{tab:gpw}',
           r'\centering\small',r'\begin{tabular}{@{}lrrr@{}}\toprule',r'Representation & Stable windows & Degree width & Certified windows\\\midrule']
    for kind in ['raw','demeaned','factor_residual']:
        r=g.set_index('network').loc[kind]
        lines.append(f'{names[kind]} & {int(r.origins_stable_bootstrap)} & {r.mean_degree_width:.2f} & {int(r.origins_certified_topk)}'+r'\\')
    return '\n'.join(lines+[r'\bottomrule\end{tabular}',r'\end{table}'])

def main():
    s,k=enrich_summary();g=pd.read_csv(RESULTS/'gpw_summary.csv')
    extra=pd.read_csv(RESULTS/'simulation_shapes_summary.csv')
    for col in ['band_coverage','degree_coverage','rank_coverage','marginal_degree_coverage','percentile_degree_coverage']:
        lo,hi=wilson(extra[col],extra.repetitions);extra[col+'_wilson_lower']=lo;extra[col+'_wilson_upper']=hi
    extra.to_csv(RESULTS/'simulation_shapes_summary.csv',index=False)
    assert len(s)==15 and len(k)==6 and len(g)==3
    figures(s)
    at=lambda model,n,col:float(s[(s.model==model)&(s.n==n)].iloc[0][col])
    values={'BAND_MIN':f'{100*s.band_coverage.min():.1f}','BAND_MAX':f'{100*s.band_coverage.max():.1f}',
     'DEG_MIN':f'{100*s.degree_coverage.min():.1f}','DEG_MAX':f'{100*s.degree_coverage.max():.1f}',
     'STAR_CERT':f'{100*at("gaussian_star",1008,"true_hub_certified"):.1f}',
     'STAR_SMALL_CERT':f'{100*at("gaussian_star",252,"true_hub_certified"):.1f}',
     'NEAR_PERCENTILE':f'{100*at("near_tie",252,"percentile_degree_coverage"):.1f}',
     'NEAR_DEG':f'{100*at("near_tie",252,"degree_coverage"):.1f}',
     'SIM_TABLE':simulation_table(s),'KENDALL_TABLE':kendall_table(k),'GPW_TABLE':gpw_table(g),
     'GPW_SENSITIVITY':'Every rank interval remains $[1,p]$ under both choices, and no top-decile membership is certified.'}
    shape_lines=[r'\begin{table}[t]',r'\caption{Additional non-star Gaussian models. Coverage is simultaneous and expressed as a percentage; width and candidate count are averages. Each row uses 500 data sets and 999 resamples.}\label{tab:shapes}',r'\centering\small',r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',r'Model & $n$ & Band & Degree & Percentile & Width & $|C_1|$\\\midrule']
    for row in extra.itertuples():
        shape_lines.append(f'{MODEL_NAMES[row.model]} & {row.n} & {100*row.band_coverage:.1f} & {100*row.degree_coverage:.1f} & {100*row.percentile_degree_coverage:.1f} & {row.mean_degree_width:.2f} & {row.top1_candidate_size:.2f}'+r'\\')
    values['SHAPE_TABLE']='\n'.join(shape_lines+[r'\bottomrule\end{tabular}',r'\end{table}'])
    for name in ['gpw_block5','gpw_block20']:
        d=pd.read_csv(RESULTS/(name+'_windows.csv'))
        assert len(d)==141 and (d.mean_rank_width==d.p-1).all() and (d.topk_certified==0).all()
    text=(OUT/'manuscript.template.tex').read_text()
    for key,value in values.items():text=text.replace('@@'+key+'@@',value)
    assert '@@' not in text
    (OUT/'SP_manuscript.tex').write_text(text)
    nums={key:value for key,value in values.items() if not key.endswith('TABLE')}
    (RESULTS/'manuscript_numbers.json').write_text(json.dumps(nums,indent=2))
    abstract=re.search(r'\\abstract\{(.*?)\}\s*\\keywords',text,re.S).group(1)
    count=len(abstract.split());assert 150<=count<=250,count
    print(json.dumps(dict(abstract_words=count,simulation_rows=len(s),kendall_rows=len(k),gpw_rows=len(g),numbers=nums),indent=2))

if __name__=='__main__':main()
