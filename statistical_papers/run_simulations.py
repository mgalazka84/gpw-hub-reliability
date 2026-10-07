"""Monte Carlo evaluation with known population correlation trees."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from scipy.stats import norm
from hub_inference import *

def sample(model,n,p,rng):
    if model in ['two_hubs','chain']:
        coefficients=np.zeros((p,p));coefficients[0,0]=1
        for i in range(1,p):
            parent=(i-1) if model=='chain' else (0 if i<=6 else 1)
            rho=.75 if model=='chain' else (.5 if i==1 else .7)
            coefficients[i]=rho*coefficients[parent];coefficients[i,i]=np.sqrt(1-rho*rho)
        return rng.normal(size=(n,p))@coefficients.T,coefficients@coefficients.T
    a=np.r_[.99,np.linspace(.45,.55,p-1)]
    if model=='near_tie':a=np.r_[.8+.5/np.sqrt(n),.8,np.linspace(.45,.55,p-2)]
    if model=='independent':a=np.zeros(p)
    corr=np.outer(a,a)+np.diag(1-a*a)
    burn=100 if model=='ar_star' else 0
    z=rng.normal(size=(n+burn,p+1))
    x=z[:,0,None]*a+z[:,1:]*np.sqrt(1-a*a)
    if model=='t5_star':x=x/np.sqrt(rng.chisquare(5,size=(n,1))/5)
    if model=='ar_star':
        for t in range(1,len(x)):x[t]=.4*x[t-1]+np.sqrt(1-.4**2)*x[t]
        x=x[burn:]
    return x,corr

def run(reps=500,B=999,output='simulation',sizes=(126,252,1008),models=None):
    if models is None:models=['independent','gaussian_star','near_tie','ar_star','t5_star']
    p=12;edges=edge_index(p);rows=[];start=time.time()
    root=Path(__file__).parent/'results';root.mkdir(exist_ok=True)
    config=dict(repetitions=reps,bootstrap_repetitions=B,p=p,sample_sizes=list(sizes),models=models,
                alpha=.05,seed=713019,band='Centered Fisher-z bootstrap max standardized deviation',
                status='retrospective method development; no preregistration')
    (root/(output+'_design.json')).write_text(json.dumps(config,indent=2))
    for mi,model in enumerate(models):
      for n in sizes:
        block=max(2,round(n**(1/3))) if model=='ar_star' else 1
        for rep in range(reps):
            model_id=['independent','gaussian_star','near_tie','ar_star','t5_star','two_hubs','chain'].index(model)
            seed=713019+model_id*10000000+n*1000+rep
            rng=np.random.default_rng(seed);x,corr=sample(model,n,p,rng)
            obs,boots=bootstrap_covariances(x,B,block,rng)
            ans=analyse_covariances(obs,boots,p)
            truth=corr[edges[:,0],edges[:,1]]
            tlo,thi=degree_envelopes(truth,truth,edges,p)
            trlo,trhi=rank_envelopes(tlo,thi)
            # For full independence every labelled tree and every rank is possible.
            # For the remaining designs the population tree is the unique star.
            if model!='independent':
                _,true_degree=max_tree(truth,edges,p)
                assert np.array_equal(tlo,true_degree) and np.array_equal(thi,true_degree)
                trlo=1+(true_degree[None,:]>true_degree[:,None]).sum(axis=1)
                trhi=(true_degree[None,:]>=true_degree[:,None]).sum(axis=1)
            degree_cover=np.all(ans['degree_lower']<=tlo) and np.all(ans['degree_upper']>=thi)
            rank_cover=np.all(ans['rank_lower']<=trlo) and np.all(ans['rank_upper']>=trhi)
            # Marginal 95% Fisher-z intervals ignore multiplicity but use the same resamples.
            br=correlation_edges(boots,edges);bz=np.arctanh(br);bz=bz-bz.mean(axis=1,keepdims=True)
            se=bz.std(axis=0,ddof=1)
            z=np.arctanh(ans['r']);z=z-z.mean();q=norm.ppf(.975)
            ml,mh=degree_envelopes(z-q*se,z+q*se,edges,p)
            point_cover=np.all(ml<=tlo) and np.all(mh>=thi)
            percentile_cover=np.nan if model=='independent' else float(np.all(ans['percentile_lower']<=tlo) and np.all(ans['percentile_upper']>=thi))
            # Top-1 candidates avoid arbitrary membership among tied leaves.
            truth_z=np.arctanh(truth);truth_z=truth_z-truth_z.mean()
            rows.append(dict(model=model,n=n,p=p,rep=rep,seed=seed,block=block,
                band_coverage=float(np.all(ans['lower']<=truth_z)&np.all(ans['upper']>=truth_z)),
                degree_coverage=float(degree_cover),rank_coverage=float(rank_cover),
                marginal_degree_coverage=float(point_cover),percentile_degree_coverage=percentile_cover,
                mean_degree_width=float(np.mean(ans['degree_upper']-ans['degree_lower'])),
                top1_candidate_size=int(np.sum(ans['rank_lower']<=1)),
                top1_certified=int(np.sum(ans['rank_upper']<=1)),
                true_hub_certified=np.nan if model in ['independent','two_hubs','chain'] else float(ans['rank_upper'][0]<=1),
                exact_degree_fraction=float(np.mean(ans['degree_lower']==ans['degree_upper'])),
                bootstrap_max=float(ans['selection'].max())))
        pd.DataFrame(rows).to_csv(root/(output+'_replications.csv'),index=False)
        print(f'{output}: {model}, n={n}, R={reps}; {time.time()-start:.1f}s',flush=True)
    df=pd.DataFrame(rows)
    group=['model','n','p','block']
    metrics=[c for c in df if c not in group+['rep','seed']]
    summary=df.groupby(group)[metrics].mean().reset_index()
    summary['repetitions']=reps
    for col in ['band_coverage','degree_coverage','rank_coverage','marginal_degree_coverage','percentile_degree_coverage','true_hub_certified']:
        summary[col+'_mcse']=np.sqrt(summary[col]*(1-summary[col])/reps)
    summary.to_csv(root/(output+'_summary.csv'),index=False)
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--repetitions',type=int,default=500)
    a.add_argument('--bootstrap',type=int,default=999);a.add_argument('--output',default='simulation')
    a.add_argument('--sizes',type=int,nargs='+',default=[126,252,1008]);a.add_argument('--models',nargs='+')
    args=a.parse_args();run(args.repetitions,args.bootstrap,args.output,args.sizes,args.models)
