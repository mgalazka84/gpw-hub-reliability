"""Independent aggregation and selected-seed replay for split sensitivity."""
from pathlib import Path
import csv, gzip, itertools, json, sys
import numpy as np
from methods import split_orders, project_order, edge_index, ROOT
from application import load_data
sys.path.insert(0,str(ROOT/'review_audit'))
from audit_abalone import python_compatible

BASE=Path(__file__).parent;D=BASE/'results/split_stability'

def main():
    records=[json.loads(p.read_text()) for p in sorted((D/'splits').glob('*.json'))]
    design=json.loads((D/'design.json').read_text());assert len(records)==100
    assert [r['seed'] for r in records]==design['seeds']
    assert len(set(design['seeds']))==100 and design['original_split_seed'] not in design['seeds']
    E=edge_index(10);featured=[];matches=dict(degrees=0,ranks=0,hubs=0,unique_exclusion=0)
    for record in records:
        assert len(record['methods'])==15
        s=record['methods']['FS-T64'];a=s['sharp_audit'];assert a['status']=='exact'
        ds=[]
        for tree in a['trees']:
            assert len(tree)==len(set(tree))==9
            parents=list(range(10))
            def root(i):
                while i!=parents[i]:i=parents[i]
                return i
            for edge in tree:
                u,v=map(int,E[edge]);u,v=root(u),root(v);assert u!=v;parents[u]=v
            ds.append(np.bincount(E[tree].ravel(),minlength=10))
        ds=np.array(ds);mx=ds.max(1)
        members=list(np.flatnonzero(np.all(ds==mx[:,None],axis=0)))
        unique_excluded=bool(np.all((ds==mx[:,None]).sum(1)>1))
        matches['degrees']+=all(s[k]==a[k] for k in ['degree_lower','degree_upper'])
        matches['ranks']+=all(s[k]==a[k] for k in ['rank_lower','rank_upper'])
        matches['hubs']+=members==s['guaranteed_members']
        matches['unique_exclusion']+=unique_excluded==s['unique_hub_excluded']
        featured.append(s)
    x=load_data()[0]['adults'];replayed=[]
    for i in [0,49,99]:
        raw,_=split_orders(x,seed=records[i]['seed']);ans=project_order(raw['FS-T64'],10)
        saved=records[i]['methods']['FS-T64']
        for k in ['degree_lower','degree_upper','rank_lower','rank_upper']:assert ans[k].tolist()==saved[k]
        forced=set(np.flatnonzero(ans['forced']));optional=np.flatnonzero(ans['allowed']&~ans['forced'])
        actual={tuple(sorted(forced|set(c))) for c in itertools.combinations(optional,9-len(forced))
                if python_compatible(10,E,forced|set(c),ans['relation'])}
        expected={tuple(sorted(t)) for t in saved['sharp_audit']['trees']}
        assert actual==expected;replayed.append(i)
    out=dict(passed=True,splits=100,methods_per_split=15,independently_replayed_indices=replayed,
       exact_tree_audits=100,minor_matches_sharp_counts=matches,
       FS_T64_all_three_count=sum(set([0,2,9])<=set(s['guaranteed_members']) for s in featured),
       FS_T64_unique_exclusion_count=sum(s['unique_hub_excluded'] for s in featured),
       FS_T64_median_fixed_degrees=float(np.median([s['fixed_degrees'] for s in featured])))
    assert out['FS_T64_all_three_count']==16 and out['FS_T64_unique_exclusion_count']==16
    with gzip.GzipFile(filename=str(D/'records.json.gz'),mode='wb',mtime=0) as f:
        f.write(json.dumps(records,separators=(',',':')).encode())
    (D/'independent_verification.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
