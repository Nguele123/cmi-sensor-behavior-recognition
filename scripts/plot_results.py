"""Render reported CMI experiment results; does not run models or estimate uncertainty."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORANGE, GREEN = '#2563eb', '#ea580c', '#059669'
records = []
def record(experiment, condition, method, score, order='not_applicable', fold='all'):
    records.append(dict(experiment=experiment, condition=condition, method=method, score=score, order=order, fold=fold))
def save(fig, name):
    fig.savefig(ROOT / 'assets' / (name + '.svg'), bbox_inches='tight')
    fig.savefig(ROOT / 'assets' / (name + '.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)

seeds = [11,22,33,44,55]
series = {
    'Original hybrid': [.877440,.880908,.877054,.872937,.877277],
    'Arithmetic 75/25': [.881294,.882732,.879898,.876557,.879819],
    'Arithmetic 50/50': [.878519,.880453,.879173,.875525,.877357],
    'Geometric 75/25': [.881000,.882020,.880643,.876433,.878813],
    'IMU25 / Full50': [.879775,.881836,.879492,.876471,.878112],
    'IMU50 / Full25': [.879768,.881816,.879983,.876452,.878310],
}
fig, (ax, ax2) = plt.subplots(1,2,figsize=(13,4.8), gridspec_kw={'width_ratios':[1,1.2]})
for i,(method,values) in enumerate(series.items()):
    for seed,value in zip(seeds,values): record('blend_grid','mixed',method,value,str(seed))
    mean = sum(values)/len(values)
    ax.scatter(values,[i]*5,s=20,color='#94a3b8',alpha=.7)
    ax.scatter([mean],[i],s=65,color=GREEN if method=='Arithmetic 75/25' else BLUE,zorder=3)
    ax.text(mean+.0002,i-.18,f'{mean:.6f}',fontsize=8)
ax.set_yticks(range(6),list(series)); ax.invert_yaxis(); ax.set_xlim(.872,.885)
ax.set_xlabel('OOF mixed score'); ax.set_title('Blending: five arrival orders\nGray dots: orders; colored dot: mean')
for method,color in [('Original hybrid',BLUE),('Arithmetic 75/25',GREEN)]:
    ax2.plot(range(5),series[method],marker='o',label=method,color=color)
ax2.set_xticks(range(5),seeds); ax2.set_xlabel('Arrival-order seed'); ax2.set_ylabel('OOF mixed score')
ax2.set_title('Paired comparison on the same sequences'); ax2.legend(); ax2.grid(alpha=.2)
fig.suptitle('Final blend improves the original hybrid in all five tested orders',fontsize=13)
fig.tight_layout(); save(fig,'blending')

fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,condition,values in zip(axes,['Public','Private'],[[.883769,.881144],[.862221,.864288]]):
    ax.bar(['V2: original hybrid','V3: arithmetic 75/25'],values,color=[BLUE,GREEN],width=.55)
    ax.set_ylim(min(values)-.003,max(values)+.002)
    for i,v in enumerate(values):
        ax.text(i,v+.0002,f'{v:.6f}',ha='center')
        record('leaderboard',condition.lower(),['V2','V3'][i],v)
    ax.set_title(condition+' leaderboard'); ax.set_ylabel('Score (zoomed axis)')
fig.suptitle('Evaluated late submissions: private score increases by 0.002067')
fig.tight_layout(); save(fig,'leaderboard')

fig,axes=plt.subplots(1,3,figsize=(15,4.7))
experiments=[
 ('BiGRU pilots: fold 1, IMU only', ['3-CNN reference','BiGRU mean','BiGRU attention'],[.8252022099,.75283,.76272], 'bigru_pilots','imu','1'),
 ('ToF contrast: fold 2, all sensors', ['Original','Contrast x1.25','Contrast x0.8'],[.87402,.87352,.87112],'tof_contrast','full','2'),
 ('CNN family weights: mixed, five-order mean', ['Equal weights','Heading 50%','Spatial 50%','Phases 50%'],[.878998,.878298,.877345,.876086],'family_weights','mixed','all'),
]
for ax,(title,methods,values,experiment,condition,fold) in zip(axes,experiments):
    ax.bar(range(len(values)),values,color=[BLUE]+[ORANGE]*(len(values)-1))
    ax.set_xticks(range(len(values)),methods,rotation=25,ha='right',fontsize=8)
    ax.set_ylim(min(values)-.005,max(values)+.008)
    for i,(method,v) in enumerate(zip(methods,values)):
        ax.text(i,v+.0006,f'{v:.5f}',ha='center',fontsize=8)
        record(experiment,condition,method,v,'five_order_mean' if experiment=='family_weights' else 'reported_validation',fold)
    ax.set_title(title,fontsize=10); ax.set_ylabel('Score (zoomed axis)')
fig.suptitle('Evidence for choices not retained — different experiments, not a common benchmark')
fig.tight_layout(); save(fig,'experiments')

with (ROOT/'results'/'reported_scores.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
print(f'Rendered 3 figures and {len(records)} reported score rows.')
