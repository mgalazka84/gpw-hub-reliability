"""Decision and cost illustration, drawn only from verified recorded results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;R=ROOT/'results';M=ROOT/'manuscript'

def main():
    check=json.loads((R/'intro_figure_verification.json').read_text());assert check['status']=='PASS'
    edges=pd.read_csv(R/'cross_section'/'edges.csv')
    edges=edges[(edges.dataset=='Abalone')&(edges.stratum=='pooled')&(edges.kendall_tree==1)]
    s=pd.read_csv(R/'practical_value'/'cross_section_stages.csv')
    s=s[(s.dataset=='Abalone')&(s.stratum=='pooled')&(s.method=='evalue_64')&(s.family=='local')].set_index('stage')
    n=pd.read_csv(R/'practical_value'/'cross_section_nodes.csv')
    n=n[(n.dataset=='Abalone')&(n.stratum=='pooled')&(n.method=='evalue_64')&(n.family=='local')&(n.name=='Whole weight')].set_index('stage')
    report=json.loads((R/'practical_value'/'report_numbers.json').read_text());t=report['abalone_e64_seconds']
    stages=['minor','degree_lp','full_lp'];labels=['Minor','Degree LP','Full LP']
    assert list(s.loc[stages,'exact_degrees'])==[1,2,2]
    assert list(s.loc[stages,'excluded_hubs'])==[2,4,4]
    assert list(n.loc[stages,'rank_upper'])==[6,3,3]
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'pdf.fonttype':42})
    ink='#233542';blue='#4f7c95';orange='#b95a18';light='#c2cbd1'
    fig=plt.figure(figsize=(8.5,4.2))
    ax=fig.add_axes([.015,.11,.36,.76])
    positions={'Length':(-1.25,.47),'Diameter':(-1.85,1.08),'Height':(1.45,1),
        'Whole weight':(-.35,.00),'Shucked weight':(-1.35,-.75),'Viscera weight':(-.35,1.12),
        'Shell weight':(.72,.02),'Rings':(1.47,-.89)}
    offsets={'Length':(-.07,.22),'Diameter':(0,.22),'Height':(0,.23),'Whole weight':(0,-.5),
        'Shucked weight':(0,-.32),'Viscera weight':(0,.32),'Shell weight':(0,-.5),'Rings':(0,-.25)}
    for e in edges.itertuples():
        a,b=positions[e.a],positions[e.b];ax.plot([a[0],b[0]],[a[1],b[1]],c=light,lw=2,zorder=1)
    for name,xy in positions.items():
        special=name=='Whole weight';ax.scatter(*xy,s=260 if special else 95,c=orange if special else blue,zorder=2)
        dx,dy=offsets[name];ax.text(xy[0]+dx,xy[1]+dy,name.replace(' weight','\nweight'),ha='center',va='center',
            fontsize=10,color=orange if special else ink,fontweight='bold' if special else 'normal')
    ax.set_xlim(-2.4,1.95);ax.set_ylim(-1.7,1.8);ax.axis('off')
    ax.set_title('(a) Sample correlation tree',fontsize=11.8,color=ink,pad=9)
    ax.text(.5,-.03,'Whole weight: degree 4',transform=ax.transAxes,ha='center',color=orange,fontsize=11)
    ax=fig.add_axes([.41,.55,.57,.32]);ax.axis('off')
    ax.set_title('(b) Certified conclusions: E64 fixed',loc='left',fontsize=11.8,color=ink,pad=9)
    rows=[[labels[i],str(int(s.loc[k,'exact_degrees'])),str(int(s.loc[k,'excluded_hubs'])),
        f"[{int(n.loc[k,'rank_lower'])}, {int(n.loc[k,'rank_upper'])}]"] for i,k in enumerate(stages)]
    tb=ax.table(cellText=rows,colLabels=['Projection','Fixed\ndegrees','Excluded\nhubs','Whole-weight\nrank'],
        colWidths=[.30,.20,.20,.30],cellLoc='center',bbox=[0,-.08,1,1.08])
    tb.auto_set_font_size(False);tb.set_fontsize(10.3)
    for (r,c),cell in tb.get_celld().items():
        cell.set_edgecolor('#e4e9ec');cell.set_linewidth(.7)
        if r==0:cell.set_facecolor('#eef2f4');cell.get_text().set_fontweight('bold')
        elif r==2:cell.set_facecolor('#f7eadf')
        cell.get_text().set_color(ink)
    ax=fig.add_axes([.52,.13,.43,.24])
    vals=[t[k] for k in stages];ax.barh([2,1,0],vals,color=[light,orange,blue],height=.56)
    ax.set_yticks([2,1,0],labels,fontsize=10.5);ax.tick_params(axis='y',length=0)
    ax.set_xlim(0,max(vals)*1.28);ax.set_xticks([0,.05,.10,.15],['0','.05','.10','.15']);ax.tick_params(axis='x',labelsize=9)
    for y,v,k in zip([2,1,0],vals,stages):ax.text(v+.004,y,f'{v:.3f} s',va='center',fontsize=10,color=ink)
    ax.set_xlabel('Projection time (seconds)',fontsize=10,color=ink,labelpad=2)
    ax.set_title('(c) Same decisions, less computation',fontsize=11.6,color=ink,pad=9,x=.39)
    for spine in ['left','right','top']:ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(light)
    fig.savefig(M/'Fig_practical_abalone.pdf',bbox_inches='tight',pad_inches=.1)
    fig.savefig(M/'Fig_practical_abalone.png',dpi=200,bbox_inches='tight',pad_inches=.1);plt.close(fig)
    (R/'practical_value'/'figure_verification.json').write_text(json.dumps(dict(status='PASS',
        method='evalue_64',family='local',dataset='Abalone pooled',fixed_degrees=[1,2,2],
        excluded_hubs=[2,4,4],whole_rank_upper=[6,3,3],seconds=vals,
        source_sample_tree_verification='results/intro_figure_verification.json'),indent=2))
    print('Practical figure PASS')

if __name__=='__main__':main()
