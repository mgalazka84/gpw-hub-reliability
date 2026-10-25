"""Retrospective decision-yield audit; every original case is retained.

Reconstruct the degree-only path stage from archived rational certificates.
The full stage is checked against the published per-node endpoints. Simulation
comparisons use the same first 100 datasets in both stages, never a 200-versus-
100 aggregate. No new samples, calibrations or familywise guarantees are claimed.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from order_inference import edge_index, project_order, conservation_ranks
from path_projection import path_constraints, _problem
from run_comparative_simulations import verify_dual_python

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'results'/'practical_value'
KEY=['dataset','stratum','method','family']

def ranks(lo,hi,diff):
    rl,rh=conservation_ranks(lo,hi)
    return np.maximum(rl,1+(diff.T>0).sum(1)),np.minimum(rh,len(lo)-(diff>0).sum(1))

def metrics(ans):
    lo,hi=ans['degree_lower'],ans['degree_upper']
    rl,rh=ans['rank_lower'],ans['rank_upper']
    out=dict(exact_degrees=int((lo==hi).sum()),leaves=int((hi==1).sum()),
        excluded_hubs=int((rl>1).sum()),hub_candidates=int((rl==1).sum()),
        certified_unique_hubs=int((rh==1).sum()),top_three=int((rh<=min(3,len(lo))).sum()),
        mean_degree_width=float(np.mean(hi-lo)),mean_rank_width=float(np.mean(rh-rl)))
    if 'degree_difference_lower' in ans:
        # Degree endpoints imply additional valid contrasts after degree LP.
        d=np.maximum(ans['degree_difference_lower'],lo[:,None]-hi[None,:])
        out['strict_degree_pairs']=int((d>0).sum())
    else:out['strict_degree_pairs']=None
    return out

def cross_section():
    summaries=[];nodes=[];transitions=[];verified=0;groups=0;missing=0
    for folder in [ROOT/'results'/'cross_section',ROOT/'results'/'evalue'/'cross_section']:
        for source in sorted(folder.glob('dataset*.json')):
            original=json.loads(source.read_text())
            certs={(c['method'],c['family']):c for c in original['certificates']}
            for row in original['rows']:
                meta={k:row[k] for k in KEY+['n','p']};p=row['p']
                nd=sorted([v for v in original['nodes'] if v['method']==row['method'] and v['family']==row['family']],key=lambda v:v['vertex'])
                full_saved={k:np.array([v[k] for v in nd],int) for k in ['degree_lower','degree_upper','rank_lower','rank_upper']}
                c=certs.get((row['method'],row['family']))
                if c is None:
                    # No improved certificate means every endpoint equals the
                    # starting minor endpoint; this is recorded, not imputed.
                    assert row['verified_certificates']==0
                    stages={s:full_saved.copy() for s in ['minor','degree_lp','full_lp']};missing+=1
                else:
                    edges=edge_index(p);rel=np.zeros((len(edges),len(edges)),bool)
                    for i,rs in enumerate(c['relation_rows']):rel[i,rs]=True
                    minor=project_order(dict(edges=edges,relation=rel),p)
                    paths=path_constraints(rel,edges,p,minor['allowed'],minor['forced'])
                    A,b,E,eb,L,U=_problem(edges,p,minor['allowed'],minor['forced'],paths)
                    lo=minor['degree_lower'].copy();hi=minor['degree_upper'].copy();diff=minor['degree_difference_lower'].copy()
                    for d in c['diagnostics']:
                        cert=d['certificate'];assert cert is not None
                        obj=np.zeros(3*len(edges),int);i=d['vertex'];obj[:len(edges)]=np.any(edges==i,axis=1)
                        if 'other' in d:obj[:len(edges)]-=np.any(edges==d['other'],axis=1).astype(int)
                        else:obj*=d['direction']
                        verify_dual_python(cert,obj,A,b,E,eb,L,U);verified+=1
                        num,den=cert['numerator'],cert['denominator']
                        v=-((-num)//den) if d['direction']==1 else (-num)//den
                        assert v==d['integer_bound']
                        if 'other' not in d:
                            if d['direction']==1:lo[i]=max(lo[i],v)
                            else:hi[i]=min(hi[i],v)
                    rl,rh=ranks(lo,hi,diff)
                    degree={**minor,'degree_lower':lo,'degree_upper':hi,'rank_lower':rl,'rank_upper':rh}
                    for d in c['diagnostics']:
                        if 'other' in d:diff[d['vertex'],d['other']]=max(diff[d['vertex'],d['other']],d['integer_bound'])
                    rl,rh=ranks(lo,hi,diff)
                    full={**degree,'degree_difference_lower':diff,'rank_lower':rl,'rank_upper':rh}
                    stages=dict(minor=minor,degree_lp=degree,full_lp=full);groups+=1
                for k,v in full_saved.items():assert np.array_equal(stages['full_lp'][k],v),(meta,k)
                for k in ['degree_lower','degree_upper']:
                    assert np.array_equal(stages['degree_lp'][k],stages['full_lp'][k])
                assert np.all(stages['degree_lp']['rank_lower']<=stages['full_lp']['rank_lower'])
                assert np.all(stages['degree_lp']['rank_upper']>=stages['full_lp']['rank_upper'])
                for stage,ans in stages.items():
                    summaries.append({**meta,'stage':stage,**metrics(ans)})
                    for i,v in enumerate(nd):
                        nodes.append({**meta,'stage':stage,'vertex':i,'name':v['name'],
                            **{k:int(ans[k][i]) for k in full_saved}})
                a,d,f=(metrics(stages[s]) for s in ['minor','degree_lp','full_lp'])
                t={**meta}
                for label,first,last in [('degree_vs_minor',a,d),('full_vs_degree',d,f),('full_vs_minor',a,f)]:
                    for metric in ['exact_degrees','leaves','excluded_hubs','certified_unique_hubs','top_three']:
                        t[label+'_'+metric]=last[metric]-first[metric]
                    t[label+'_any_decision_gain']=int(any(last[m]>first[m] for m in ['exact_degrees','excluded_hubs','certified_unique_hubs','top_three']))
                t['identical_rank_endpoints']=int(all(np.array_equal(stages['degree_lp'][k],stages['full_lp'][k]) for k in ['rank_lower','rank_upper']))
                t['identical_hub_exclusions']=int(np.array_equal(stages['degree_lp']['rank_lower']>1,stages['full_lp']['rank_lower']>1))
                transitions.append(t)
    pd.DataFrame(summaries).to_csv(DEST/'cross_section_stages.csv',index=False)
    pd.DataFrame(nodes).to_csv(DEST/'cross_section_nodes.csv',index=False)
    pd.DataFrame(transitions).to_csv(DEST/'cross_section_transitions.csv',index=False)
    assert len(transitions)==192 and groups==188 and missing==4 and verified==16530
    return dict(cases=len(transitions),certificate_groups=groups,verified_certificates=verified,
                cases_without_improved_certificates=missing,full_node_reconstruction='PASS')

def simulation():
    df=pd.read_csv(ROOT/'results'/'comparative'/'replications_all.csv.gz')
    keys=['p','model','n','method','family','rep','seed']
    lp=df[(df.stage=='path') & (df.rep<100)].copy()
    base=df[(df.stage=='joint') & (df.rep<100)].copy()
    paired=lp.merge(base,on=keys,suffixes=('_full','_minor'),validate='one_to_one')
    assert len(paired)==len(lp)==27200
    paired['new_exact_degrees']=paired.exact_degrees_full-paired.exact_degrees_minor
    paired['new_excluded_hubs']=paired.top1_candidates_minor-paired.top1_candidates_full
    paired['new_unique_hubs']=paired.top1_certified_full-paired.top1_certified_minor
    for c in ['new_exact_degrees','new_excluded_hubs','new_unique_hubs']:assert (paired[c]>=0).all()
    paired['any_new_decision']=(paired[['new_exact_degrees','new_excluded_hubs','new_unique_hubs']]>0).any(axis=1).astype(int)
    paired['new_fixed_degree_event']=(paired.new_exact_degrees>0).astype(int)
    paired['new_exclusion_event']=(paired.new_excluded_hubs>0).astype(int)
    paired['excluded_hubs_minor']=paired.p-paired.top1_candidates_minor
    paired['excluded_hubs_full']=paired.p-paired.top1_candidates_full
    keep=keys+['new_exact_degrees','new_excluded_hubs','new_unique_hubs','any_new_decision','new_fixed_degree_event','new_exclusion_event',
        'exact_degrees_minor','exact_degrees_full','excluded_hubs_minor','excluded_hubs_full','top1_certified_minor','top1_certified_full',
        'true_hub_certified_full','degree_coverage_full','rank_coverage_full','order_fwer_full','mean_degree_width_minor','mean_degree_width_full']
    paired[keep].to_csv(DEST/'simulation_matched_decisions.csv.gz',index=False)
    g=['p','model','n','method','family'];summ=paired[keep].groupby(g).mean(numeric_only=True).reset_index()
    summ=summ.merge(paired.groupby(g).size().rename('repetitions').reset_index(),on=g)
    summ.to_csv(DEST/'simulation_decision_summary.csv',index=False)
    return dict(paired_method_family_datasets=len(paired),cells=len(summ),same_samples='PASS',
                scope='First 100 replications per p=12 design cell, both families, all eight methods. Nested projection only; calibration held fixed.')

def run():
    DEST.mkdir(exist_ok=True)
    design=dict(version='0.6',analysis='Retrospective practical-decision audit of all published cases',
        main_endpoints=['fixed degree','hub exclusion','unique hub','top-three restriction'],
        projection_stages=['minor plus direct-pair and conservation ranks','degree-only path LP','full ordered-pair path LP'],
        scope='Each dataset/method/family is a separate 95% analysis; no joint coverage across 192 analyses',
        interpretation='Tree association summaries; leaf status does not justify deleting a predictor or imply conditional independence',
        sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'results'/'comparative'/'replications_all.csv.gz',ROOT/'results'/'cross_section'/'nodes_all.csv']})
    (DEST/'design.json').write_text(json.dumps(design,indent=2))
    result=dict(status='PASS',cross_section=cross_section(),simulation=simulation())
    (DEST/'audit_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':run()
