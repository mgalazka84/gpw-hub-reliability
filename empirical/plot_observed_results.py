from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE=Path(__file__).resolve().parent
RESULTS=BASE/'results'
FIGURES=BASE/'figures'
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.labelcolor':'#263447','text.color':'#263447',
    'axes.edgecolor':'#abb5bf','grid.color':'#dce2e8','grid.alpha':.7,
    'savefig.facecolor':'white'})
LABELS={'raw':'Raw','demeaned':'Cross-sectionally demeaned','factor_residual':'WIG-factor residual'}
COLORS={'raw':'#1765a8','demeaned':'#b86524','factor_residual':'#147f74'}

def finish(fig,path):
    fig.savefig(path,dpi=210,bbox_inches='tight')
    plt.close(fig)

def main():
    close=pd.read_csv(RESULTS/'features_close.csv',parse_dates=['origin_date','label_end_date'])
    fig,axes=plt.subplots(3,1,figsize=(7.5,5.65),sharex=True)
    for ax,kind in zip(axes,LABELS):
        ax.plot(close.origin_date,close['degree_'+kind],color='#717c86',lw=1.2,label='Observed concentration C')
        ax.plot(close.origin_date,close['reliability_'+kind],color=COLORS[kind],lw=1.4,label='Reliability share H')
        ax.set_ylabel('Degree share');ax.set_ylim(.08,max(.42,float(close['degree_'+kind].max())+.015))
        ax.text(.012,.87,LABELS[kind],transform=ax.transAxes,fontsize=10,fontweight='bold')
        ax.grid(axis='y')
    axes[0].legend(loc='upper right',fontsize=8,frameon=False,ncol=1)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(4));axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout(h_pad=.5)
    finish(fig,FIGURES/'network_concentration.png')

    fig,axes=plt.subplots(2,1,figsize=(7.5,4.75),sharex=True)
    for ax,convention in zip(axes,['close','intraday']):
        p=pd.read_csv(RESULTS/f'predictions_{convention}_factor_residual.csv',parse_dates=['origin_date'])
        p=p.loc[p.origin_date.dt.year.between(2019,2025)].dropna(subset=['future_y','base','degree','reliability','persistence','ewma'])
        diff=((p.degree-p.future_y)**2-(p.reliability-p.future_y)**2)
        ax.axhline(0,color='#808c99',lw=.8)
        ax.plot(p.origin_date,diff.cumsum(),color='#147f74',lw=1.8)
        ax.set_ylabel('Cumulative loss difference')
        ax.set_title('Close-to-close' if convention=='close' else 'Open-to-close',loc='left',fontsize=10,fontweight='bold',pad=5)
        ax.grid(axis='y')
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2));axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout(h_pad=.65)
    finish(fig,FIGURES/'cumulative_loss_comparison.png')

    coverage=pd.read_csv(RESULTS/'origin_coverage.csv',parse_dates=['origin_date'])
    fig,axes=plt.subplots(2,1,figsize=(7.5,4.7),sharex=True,gridspec_kw={'height_ratios':[1.4,1]})
    axes[0].plot(coverage.origin_date,coverage.eligible_assets,color='#1765a8',lw=1.4)
    axes[0].axhline(60,color='#8d99a6',ls='--',lw=1,label='60 portfolio constituents')
    axes[0].set_ylabel('Eligible stocks');axes[0].set_ylim(20,62);axes[0].legend(frameon=False,fontsize=8,loc='lower right');axes[0].grid(axis='y')
    counts=coverage.groupby(coverage.origin_date.dt.year).agg(total=('future_complete','size'),complete=('future_complete','sum'))
    axes[1].bar(pd.to_datetime(counts.index.astype(str)+'-07-01'),counts.complete,width=230,color='#1765a8',label='Complete targets')
    axes[1].bar(pd.to_datetime(counts.index.astype(str)+'-07-01'),counts.total-counts.complete,bottom=counts.complete,width=230,color='#c36949',label='Missing targets')
    axes[1].set_ylabel('Origins per year');axes[1].legend(frameon=False,fontsize=8,loc='lower right',bbox_to_anchor=(1,1.02),ncol=2)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(4));axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout(h_pad=.65)
    finish(fig,FIGURES/'coverage.png')

    fig,axes=plt.subplots(2,1,figsize=(7.5,4.85),sharex=True)
    for ax,convention in zip(axes,['close','intraday']):
        p=pd.read_csv(RESULTS/f'predictions_{convention}_factor_residual.csv',parse_dates=['origin_date'])
        p=p.loc[p.origin_date.dt.year.between(2019,2025)]
        ax.plot(p.origin_date,p.future_y,color='#263447',lw=1.25,label='Observed future Y')
        ax.plot(p.origin_date,p.reliability,color='#147f74',lw=1.3,label='Base + C + H')
        ax.plot(p.origin_date,p.ewma,color='#b86524',lw=1,alpha=.8,label='EWMA benchmark')
        ax.set_ylabel('Diversification loss Y');ax.grid(axis='y')
        ax.text(.012,.9,'Close-to-close' if convention=='close' else 'Open-to-close',transform=ax.transAxes,fontweight='bold')
    axes[0].legend(frameon=False,fontsize=7.5,ncol=3,loc='lower left')
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2));axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout(h_pad=.65)
    finish(fig,FIGURES/'observed_forecasts.png')

if __name__=='__main__':main()
