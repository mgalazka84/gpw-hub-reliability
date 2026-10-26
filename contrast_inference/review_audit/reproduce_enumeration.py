"""Standalone reproduction: Python 3 standard library and a C++17 compiler.

Run: python reproduce_enumeration.py
No statistical recalibration, external data download or repository imports.
"""
from pathlib import Path
from itertools import combinations
import csv, io, json, subprocess

ROOT=Path(__file__).resolve().parent

def connected(p,edges,selected,a=0,b=None):
    adjacency=[[] for _ in range(p)]
    for e in selected:
        u,v=edges[e];adjacency[u].append(v);adjacency[v].append(u)
    visited={a};todo=[a]
    for u in todo:
        for v in adjacency[u]:
            if v not in visited:visited.add(v);todo.append(v)
    return len(visited)==p if b is None else b in visited

def compatible(p,edges,selected,relation):
    if not connected(p,edges,selected):return False
    adjacency=[[] for _ in range(p)]
    for e in selected:
        u,v=edges[e];adjacency[u].append((v,e));adjacency[v].append((u,e))
    reach=[set(r) for r in relation]
    for e,(a,b) in enumerate(edges):
        if e in selected:continue
        parents={a:(None,None)};todo=[a]
        for u in todo:
            for v,f in adjacency[u]:
                if v not in parents:parents[v]=(u,f);todo.append(v)
        v=b
        while v!=a:
            v,f=parents[v];reach[f].add(e)
    for k in range(len(edges)):
        for i in range(len(edges)):
            if k in reach[i]:reach[i].update(reach[k])
    return all(i not in row for i,row in enumerate(reach))

def main():
    exe=ROOT/'exhaustive_trees'
    subprocess.run(['g++','-O3','-std=c++17',str(ROOT/'exhaustive_trees.cpp'),'-o',str(exe)],check=True)
    reports=[]
    for manifest in sorted((ROOT/'results').glob('*_relation.json')):
        d=json.loads(manifest.read_text());p=d['p'];edges=d['edges'];relation=d['relation_rows'];m=len(edges)
        pairs=[(i,j) for i,row in enumerate(relation) for j in row]
        source=f'{p} {len(pairs)}\n'+''.join(f'{i} {j}\n' for i,j in pairs)
        run=subprocess.run([str(exe)],input=source,capture_output=True,text=True,check=True)
        records=[list(map(int,line)) for line in csv.reader(io.StringIO(run.stdout))]
        actual={r[1] for r in records}
        assert len(actual)==len(records)
        allowed=set();forced=set()
        for e,(u,v) in enumerate(edges):
            stronger=[f for f in range(m) if e in relation[f]]
            if not connected(p,edges,stronger,u,v):allowed.add(e)
            alternatives=[f for f in range(m) if f!=e and f not in relation[e]]
            if not connected(p,edges,alternatives,u,v):forced.add(e)
        independent=set()
        for rest in combinations(sorted(allowed-forced),p-1-len(forced)):
            selected=forced|set(rest)
            if compatible(p,edges,selected,relation):independent.add(sum(1<<e for e in selected))
        assert actual==independent,'Pruefer and edge-set enumerations differ'
        name=manifest.stem.replace('_relation','')
        with (ROOT/'results'/f'{name}_compatible_trees.csv').open() as f:
            archived={int(r['edge_mask']) for r in csv.DictReader(f)}
        assert actual==archived,'Archived and recomputed compatible trees differ'
        answer=dict(analysis=name,all_trees=p**(p-2),compatible_trees=len(actual),independent_check=True,
            degree_lower=[min(r[2+i] for r in records) for i in range(p)],
            degree_upper=[max(r[2+i] for r in records) for i in range(p)],
            rank_lower=[min(r[2+p+2*i] for r in records) for i in range(p)],
            rank_upper=[max(r[3+p+2*i] for r in records) for i in range(p)])
        reports.append(answer);print(json.dumps(answer),flush=True)
    (ROOT/'results'/'standalone_verification.json').write_text(json.dumps(reports,indent=2)+'\n')

if __name__=='__main__':main()
