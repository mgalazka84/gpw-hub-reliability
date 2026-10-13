"""Independent final invariants, file provenance, paths and saved exact bounds."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from order_inference import edge_index,project_order,conservation_ranks
from path_projection import path_constraints,_problem
from verify_revision import verify_certificate

ROOT=Path(__file__).parent;R=ROOT/'results'

def check_path(path,edges,p,relation):
    degree=np.bincount(edges[list(path)].ravel(),minlength=p)
    endpoints=np.flatnonzero(degree==1);assert len(endpoints)==2 and np.all(degree[degree>0]<=2)
    assert len(np.flatnonzero(degree))==len(path)+1
    parent=list(range(p))
    def root(v):
        while parent[v]!=v:v=parent[v]
        return v
    for e in path:
        a,b=edges[e];a,b=root(int(a)),root(int(b));assert a!=b;parent[b]=a
    assert root(int(endpoints[0]))==root(int(endpoints[1]))
    closing=np.flatnonzero(np.all(edges==np.sort(endpoints),axis=1));assert len(closing)==1
    assert relation[closing[0],list(path)].any()

def run():
    counts={};df=pd.read_csv(R/'comparative'/'replications_all.csv.gz')
    key=['p','model','n','rep','method','family','stage'];assert not df.duplicated(key).any()
    assert not ((df.order_fwer==0)&((df.degree_coverage<1)|(df.rank_coverage<1))).any(), 'Projection fails on the valid-order event'
    pair=df.pivot(index=['p','model','n','rep','method','family'],columns='stage',values='mean_rank_width')
    assert ((pair['joint']<=pair['marginal']+1e-12)).all()
    with_path=pair.dropna(subset=['path']);assert (with_path['path']<=with_path['joint']+1e-12).all()
    degree=df.pivot(index=['p','model','n','rep','method','family'],columns='stage',values='mean_degree_width').dropna(subset=['path'])
    assert (degree['path']<=degree['joint']+1e-12).all()
    counts['monte_carlo_records']=len(df);counts['path_paired_records']=len(with_path)
    counts['runtime_verified_simulation_certificates']=int(df.loc[df.stage=='path','verified_certificates'].sum())
    counts['runtime_lp_solves']=int(df.loc[df.stage=='path','lp_solves'].sum())
    for method in df.method.unique():
        null=df[(df.method==method)&(df.stage=='joint')&df.model.isin(['independent','discrete_independent'])]
        assert (null.groupby(['p','model','n','family']).size()==1000).all()
    provenance=json.loads((ROOT/'data'/'cross_section'/'PROVENANCE.json').read_text());files=0
    for record in provenance:
        archive=ROOT/'data'/'cross_section'/(record['dataset']+'.zip')
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==record['archive_sha256']
        for item in record['files']:
            assert hashlib.sha256((ROOT/'data'/'cross_section'/item['file']).read_bytes()).hexdigest()==item['sha256'];files+=1
    counts['original_measurement_files']=files
    nodes=pd.read_csv(R/'cross_section'/'nodes_all.csv');assert not nodes.duplicated(['dataset','stratum','method','family','vertex']).any()
    source_dirs=[R/'cross_section',R/'evalue'/'cross_section'];paths_checked=certificates=records=0
    for source in source_dirs:
        for file in sorted(source.glob('dataset*.json')):
            obj=json.loads(file.read_text())
            for record in obj['certificates']:
                p=record['p'];edges=edge_index(p);relation=np.zeros((len(edges),len(edges)),bool)
                for e,row in enumerate(record['relation_rows']):relation[e,row]=True
                ans=project_order(dict(relation=relation,edges=edges),p)
                paths=path_constraints(relation,edges,p,ans['allowed'],ans['forced'])
                for path in paths:check_path(path,edges,p,relation);paths_checked+=1
                problem=_problem(edges,p,ans['allowed'],ans['forced'],paths)
                dlo=ans['degree_lower'].copy();dhi=ans['degree_upper'].copy();difference=ans['degree_difference_lower'].copy()
                for diagnostic in record['diagnostics']:
                    verify_certificate(diagnostic,edges,p,ans['allowed'],ans['forced'],paths,problem);certificates+=1
                    sub=nodes[(nodes.dataset==record['dataset'])&(nodes.stratum==record['stratum'])&(nodes.method==record['method'])&(nodes.family==record['family'])]
                    i=diagnostic['vertex'];bound=diagnostic['integer_bound']
                    if 'other' not in diagnostic:
                        node=sub[sub.vertex==i].iloc[0]
                        assert (node.degree_lower>=bound) if diagnostic['direction']==1 else (node.degree_upper<=bound)
                        if diagnostic['direction']==1:dlo[i]=max(dlo[i],bound)
                        else:dhi[i]=min(dhi[i],bound)
                    else:difference[i,diagnostic['other']]=max(difference[i,diagnostic['other']],bound)
                rl,rh=conservation_ranks(dlo,dhi)
                rl=np.maximum(rl,1+(difference.T>0).sum(axis=1));rh=np.minimum(rh,p-(difference>0).sum(axis=1))
                ordered=sub.sort_values('vertex')
                for actual,expected in [(ordered.degree_lower,dlo),(ordered.degree_upper,dhi),(ordered.rank_lower,rl),(ordered.rank_upper,rh)]:
                    np.testing.assert_array_equal(actual,expected)
                records+=1
    counts.update(saved_measurement_certificate_groups=records,saved_measurement_certificates=certificates,checked_simple_paths=paths_checked)
    original_summary=pd.read_csv(R/'cross_section'/'summary_all.csv');assert certificates==int(original_summary.verified_certificates.sum())
    a=pd.read_csv(R/'bootstrap_audit'/'replications.csv');assert len(a)==12000
    assert ((a.input_coverage+a.order_fwer)==1).all()
    assert (a.groupby(['p','n','method']).size()==1000).all()
    f=pd.read_csv(R/'frequency_reference'/'replications.csv');assert len(f)==2800
    # Every practical frequency reference is evaluated on an existing main sample.
    seeds=df[(df.method=='paired_exact')&(df.family=='local')&(df.stage=='joint')][['p','model','n','rep','seed']]
    matched=f.merge(seeds,on=['p','model','n','rep','seed'],how='inner');assert len(matched)==2800
    counts['matched_frequency_reference_datasets']=len(matched)
    for name in ['verification.json','comparator_verification.json','evalue_verification.json','estimand_verification.json','gpw_certificate_verification.json','gpw_extended_verification.json']:
        assert json.loads((R/name).read_text())['status']=='PASS'
    counts['status']='PASS';(R/'review_verification.json').write_text(json.dumps(counts,indent=2));print(json.dumps(counts,indent=2),flush=True)

if __name__=='__main__':run()
