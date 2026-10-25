"""Build decision-yield tables and measured costs from the complete audit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from build_review_results import table as _table

ROOT=Path(__file__).resolve().parent;R=ROOT/'results'/'practical_value';M=ROOT/'manuscript'
STAGES={'minor':'Minor','degree_lp':'Degree LP','full_lp':'Full LP'}

def table(*args,**kwargs):
    # Keep the decision table at its paragraph boundary, beside its conclusion.
    return '\n\n'+_table(*args,**kwargs,placement='H')+'\n\n'

def interval(node,prefix):return f"[{int(node[prefix+'_lower'])},{int(node[prefix+'_upper'])}]"

def run():
    s=pd.read_csv(R/'cross_section_stages.csv');nodes=pd.read_csv(R/'cross_section_nodes.csv')
    trans=pd.read_csv(R/'cross_section_transitions.csv');sim=pd.read_csv(R/'simulation_decision_summary.csv')
    timing=pd.read_csv(R/'benchmark'/'timings.csv')
    med=timing.groupby(['case_id','source','n','p','method','mode'],dropna=False).median(numeric_only=True).reset_index()
    med.to_csv(R/'benchmark'/'case_medians.csv',index=False)
    info=json.loads((R/'benchmark'/'design.json').read_text())
    summary=dict(measurement_analyses=len(trans),identical_rank_analyses=int(trans.identical_rank_endpoints.sum()),
        degree_lp_decision_gain_analyses=int(trans.degree_vs_minor_any_decision_gain.sum()),
        full_lp_extra_decision_analyses=int(trans.full_vs_degree_any_decision_gain.sum()),
        extra_hub_exclusions_from_full=int(trans.full_vs_degree_excluded_hubs.sum()),
        benchmark_cases=info['cases'],benchmark_measured_runs=len(timing))
    # Hold calibration and sample fixed in every simulation comparison.
    z=sim[(sim.method=='evalue_64')&(sim.family=='local')]
    models=['gaussian_star','t5_star','near_tie','chain','two_hubs']
    names={'gaussian_star':'Gaussian star','t5_star':r'$t_5$ star','near_tie':'Near tie','chain':'Chain','two_hubs':'Two hubs'}
    rows=[]
    for model in models:
        for n in [1008,4096]:
            v=z[(z.model==model)&(z.n==n)].iloc[0]
            rows.append([names[model],str(n),f'{v.exact_degrees_minor:.2f}',f'{v.exact_degrees_full:.2f}',
                f'{v.excluded_hubs_minor:.2f}',f'{v.excluded_hubs_full:.2f}',f'{int(round(v.any_new_decision*100))}/100'])
    text=r'''\subsection{Decisions gained on the same simulated samples}\label{sec:decision-simulation}
An interval-width reduction is useful for a given question only if it changes the answer or the degree range relevant to that question. We therefore separately count fixed degrees, exclusions from hub status and unique-hub declarations. Table~\ref{tab:decision-simulation} uses the same first 100 samples for both projections and retains every non-null $p=12$ model at $n=1008$ and $4096$. The archived audit includes all 17 $p=12$ cells, both families and all eight calibrations, giving 27,200 paired method--family--dataset comparisons. These are repeated analyses of the original datasets, not new independent samples.
'''
    text+=table(r'Paired E64 decision yield, local family, $p=12$. Fixed degrees and hub exclusions are mean counts per dataset before (M: minor with pair/conservation ranks) and after full path projection (F). The last column counts datasets gaining a fixed degree, a hub exclusion or a unique-hub declaration. Each row contains the same 100 samples in both stages.',
        'tab:decision-simulation',['Model','$n$','Fixed M','Fixed F','Excluded M','Excluded F','Any gain'],rows,'llrrrrr')
    text+=r'''At $n=1008$, neither exact star experiment yields a fixed degree or a hub exclusion, even after full projection. For the chain, degree width falls from $6.477$ to $3.968$ without changing these decisions. For near ties at $n=4096$, projection produces a new decision in 13 of 100 samples: the mean number of fixed degrees and excluded hubs both increase from $0.06$ to $0.21$, but no unique hub is identified. The two-hub model at $n=1008$ gains exclusions in one of 100 samples. At $n=4096$, separated-star hubs are already identified by the minor stage; the additional LP work does not create that success. Thus the exact method's larger-sample power comes primarily from calibrated edge evidence, while projection has selective value for partial decisions. Full cellwise counts and original coverage results accompany the article.
'''
    (M/'practical_simulation_results.tex').write_text(text)
    # Main application: explicit answer changes, then all populations.
    pooled=s[(s.dataset=='Abalone')&(s.stratum=='pooled')&(s.family=='local')]
    wn=nodes[(nodes.dataset=='Abalone')&(nodes.stratum=='pooled')&(nodes.family=='local')&(nodes.name=='Whole weight')]
    rows=[]
    for method,stage in [('paired_exact','full_lp'),('evalue_64','minor'),('evalue_64','degree_lp'),('evalue_64','full_lp')]:
        v=pooled[(pooled.method==method)&(pooled.stage==stage)].iloc[0]
        w=wn[(wn.method==method)&(wn.stage==stage)].iloc[0]
        rows.append(['P' if method=='paired_exact' else 'E64',STAGES[stage],str(int(v.exact_degrees)),str(int(v.excluded_hubs)),interval(w,'degree'),interval(w,'rank')])
    text=r'''\subsection{Which conclusions change after projection?}\label{sec:decision-application}
The primary question is which measurements can be certified as leaves, excluded from hub status, or placed within the top three. These are statements about the marginal association tree; leaf status alone does not justify removing a measurement from a predictive model. Table~\ref{tab:decision-abalone} separates the gain from replacing one matching by E64 from the gain due to subsequent optimization.
'''
    text+=table(r'Abalone pooled: concrete conclusions under the IID working model and a separately chosen calibration. All rows use local edge comparisons. Fixed and excluded denote counts among eight measurements. The last two columns concern whole weight. P is one-matching exact McNemar--Holm.',
        'tab:decision-abalone',['Test','Projection','Fixed','Excluded','Degree','Rank'],rows,'llrrrr')
    text+=r'''With E64 fixed, the minor stage excludes length and shucked weight from hub status. Degree-only path projection additionally excludes viscera weight and ring count, certifies ring count as a leaf, and raises the lower degree of whole weight from two to three. Together with degree conservation, these statements place whole weight within the top three. It is certified to have strictly greater degree than length, shucked weight, viscera weight and ring count in every population-optimal tree. The extra 56 ordered-pair LP objectives change none of the reported degree or rank endpoints. The top-three statement has a simple structural explanation. The eight degrees sum to 14. If whole weight and three competitors each had degree at least three, these four vertices plus the four remaining positive degrees would sum to at least 16. At most two competitors can therefore match or exceed whole weight. The inferential conclusion is four excluded hubs and a top-three restriction, with the identity of a unique population hub still unresolved.

Table~\ref{tab:decision-populations} reports the entire E64 local-family measurement grid, including cases with no new decision. At $n=1307$ and $1528$, degree LP certifies ring count as a leaf in the female and male Abalone strata, respectively; neither minor projection resolves those degrees. In the infant stratum, shucked weight is already a certified leaf before LP. The corresponding guarantees require an IID model separately within each analysed population. The comparisons across populations, methods and families are descriptive and have no joint 95\% coverage statement.
'''
    rows=[]
    for (dataset,stratum),d in s[(s.method=='evalue_64')&(s.family=='local')].groupby(['dataset','stratum'],sort=False):
        a=d[d.stage=='minor'].iloc[0];b=d[d.stage=='degree_lp'].iloc[0]
        label=('Abalone '+stratum) if dataset=='Abalone' else ('Bean '+stratum.title())
        rows.append([label,str(int(a.n)),str(int(a.p)),f'{int(a.exact_degrees)} $\\to$ {int(b.exact_degrees)}',
            f'{int(a.excluded_hubs)} $\\to$ {int(b.excluded_hubs)}',f'{int(a.top_three)} $\\to$ {int(b.top_three)}'])
    text+=table(r'All 12 measurement populations: E64 local-family decision counts, minor $\to$ degree-only LP. Full ordered-pair projection gives the same rank endpoints as degree-only LP in every row. Dry Bean remains a computational stress study because the sampling design and repeated rows limit a population interpretation.',
        'tab:decision-populations',['Population','$n$','$p$','Fixed','Excluded','Top three'],rows,'lrrrrr')
    text+=f'''Across all 192 original method--family--population analyses, degree-only LP adds at least one of the stated decisions in {summary['degree_lp_decision_gain_analyses']} analyses. Full pair projection gives exactly the same rank endpoints in {summary['identical_rank_analyses']} of 192 analyses. Its remaining benefit is six additional hub exclusions across four analyses: female Abalone with multiplier rectangles and male Abalone with P, each under both comparison families. These four analyses concern two populations and are not independent confirmations. No stage produces a new unique-hub declaration on the measurement grid. Dry Bean's E64 degree widths narrow, but no leaf, hub exclusion or top-three restriction is certified, even in the pooled sample of 13,611 records. This distinguishes a narrower numerical bound from a new substantive conclusion.\n'''
    (M/'practical_cross_results.tex').write_text(text)
    # Controlled cost audit, not inherited timings from parallel production.
    e=med[(med.source=='measurement')&(med.case_id==12)]
    t={v.mode:float(v.total_seconds) for v in e.itertuples()}
    summary['abalone_e64_seconds']=t
    summary['abalone_e64_projection_speedup']=t['full_lp']/t['degree_lp']
    summary['abalone_e64_projection_reduction_percent']=100*(1-t['degree_lp']/t['full_lp'])
    text=r'''\section{Computation matched to the inferential question}\label{sec:cost}
The decision-first workflow uses a fixed calibration and comparison family, then adds optimization only for the desired output. We benchmark one-matching P and E64 on all 12 measurement populations and the first replication of each of the 13 $p=12$ simulation cells with $n\ge1008$: 50 cases in total. This sample rule is independent of benchmark outcomes. Each case and stage runs in a fresh process, with one untimed warm-up and three timed repetitions, serial execution, one pinned CPU, and one BLAS/OpenMP/HiGHS thread. Interpreter startup and data loading are excluded. Projection timings include the minor construction, LP work and independent rational-certificate checks. Calibration is timed separately, including full Kendall computation for E64. These are computational repetitions of fixed instances, not Monte Carlo estimates of inferential power.
'''
    text+=f"The recorded environment is Python {info['hardware']['python']}, NumPy {info['hardware']['numpy']} and SciPy {info['hardware']['scipy']} on an {info['hardware']['cpu']} processor. Raw per-run timings, process peak resident memory, CPU affinity and all reconstructed endpoints are archived. Peak resident memory includes imports, input and warm-up; it is not interpreted as incremental solver storage.\n"
    rows=[]
    for label,mode in [('Edge calibration','calibration'),('Minor','minor'),('Degree LP','degree_lp'),('Full LP','full_lp')]:
        rows.append([label,'--' if mode=='calibration' else str({'minor':0,'degree_lp':16,'full_lp':72}[mode]),f'{t[mode]:.4f}'])
    text+=table(r'Controlled cost for E64 on pooled Abalone ($p=8,n=4177$). Seconds are medians of three timed repetitions. Each projection time includes the minor stage; calibration is separate. Degree-only and full LP have identical degree and rank endpoints.',
        'tab:cost-abalone',['Stage','LP objectives','Seconds'],rows,'lrr')
    text+=f"For this example, degree-only projection requires 16 instead of 72 LP solves and takes {t['degree_lp']:.4f} rather than {t['full_lp']:.4f} seconds, a {summary['abalone_e64_projection_reduction_percent']:.1f}\\% reduction in projection time. E64 calibration itself takes {t['calibration']:.4f} seconds, so the {summary['abalone_e64_projection_speedup']:.2f}-fold projection speedup is not an end-to-end speedup of the entire statistical method.\n"
    rows=[];cost_by_group=[]
    for source,p,label in [('measurement',8,'Abalone'),('simulation',12,'Simulated'),('measurement',16,'Dry Bean')]:
        q=med[(med.source==source)&(med.p==p)]
        vals={mode:float(q[q['mode']==mode].total_seconds.median()) for mode in ['minor','degree_lp','full_lp']}
        count=q.case_id.nunique();rows.append([label,str(p),str(count)]+[f'{vals[k]:.4f}' for k in ['minor','degree_lp','full_lp']])
        cost_by_group.append(dict(source=source,p=p,cases=count,**vals))
    summary['group_cost_seconds']=cost_by_group
    text+=table(r'All 50 benchmark cases, by data source and dimension. Each entry is the median across case-specific medians of three runs. Empty order relations and forced unique trees skip redundant LP solves, including in the simulated group. These medians therefore reflect the stated case mix as well as dimension.',
        'tab:cost-grid',['Source','$p$','Cases','Minor (s)','Degree LP (s)','Full LP (s)'],rows,'lrrrrr')
    text+=r'''The degree-only stage is the economical default for degree bounds and the tested partial decisions. Ordered-pair objectives remain an optional refinement when an unresolved comparison or rank question warrants the extra work. Their benefit should be reported as a changed conclusion, along with cost, rather than inferred from the number of solved programs. The audit is retrospective and the measured costs are instance- and hardware-specific; it establishes the gains on this complete evaluation grid rather than universal dominance of a projection stage.
'''
    (M/'practical_cost_results.tex').write_text(text)
    (R/'report_numbers.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
