"""Plot saved field examples with shared scales."""
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'tempo-npod-figures'))
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.tri import Triangulation
from matplotlib.ticker import MaxNLocator, FuncFormatter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style
style.apply()
proof = {}


def load(task):
    path = args.data/(task+'.npz')
    proof[task] = dict(data_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def bar(fig, image, bounds):
    cb = fig.colorbar(image, cax=fig.add_axes(bounds))
    ticks = MaxNLocator(nbins=3).tick_values(image.norm.vmin, image.norm.vmax)
    cb.set_ticks(ticks[(ticks >= image.norm.vmin)&(ticks <= image.norm.vmax)])
    cb.formatter = FuncFormatter(lambda value, _: f'{value:.2g}')
    cb.update_ticks(); cb.ax.tick_params(pad=2)


def finish(fig, name, task):
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            assert box.x0 >= -.5 and box.y0 >= -.5 and box.x1 <= fig.bbox.width+.5 and box.y1 <= fig.bbox.height+.5, text.get_text()
    outputs = {}
    for ext in ['pdf', 'png']:
        path = args.out/(name+'.'+ext)
        fig.savefig(path, dpi=300, facecolor='white',
                    metadata={'CreationDate':None, 'ModDate':None} if ext=='pdf' else None)
        outputs[ext] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    plt.close(fig)
    proof[task].update(outputs=outputs, text_inside_canvas=True,
                       source_values_modified=False, common_reference_for_methods=True)


def grid(rows, name, task, height):
    """Rows contain reference, prediction(s), coordinates, label, and field palette."""
    methods = len(rows[0][1]); cols = 1+2*methods
    # One field scale and one error scale per row, shared by both methods.
    lefts = [.09, .36, .69] if methods==1 else [.08, .24, .44, .60, .78]
    width = .205 if methods==1 else .12
    fieldbar = .585 if methods==1 else .375
    errorbar = .915 if methods==1 else .915
    titles = ['Reference', 'TEMPO', 'Error'] if methods==1 else ['Reference', 'POD', 'Error', 'NeuralPOD', 'Error']
    fig = plt.figure(figsize=style.size(height))
    for left, title in zip(lefts, titles):
        fig.text(left+width/2, .985, title, ha='center', va='top', fontsize=8)
    step = .85/len(rows); rowh = step-.045
    for row, (truth, predictions, x, y, label, cmap) in enumerate(rows):
        bottom = .925-(row+1)*step+.02
        fields = [truth]+list(predictions)
        lo = float(min(v.min() for v in fields)); hi = float(max(v.max() for v in fields))
        if cmap==style.SIGNED: hi=max(abs(lo),abs(hi)); lo=-hi
        errors = [np.abs(pred-truth) for pred in predictions]
        emax = max(float(e.max()) for e in errors)
        values = [truth]
        for pred, error in zip(predictions, errors): values.extend([pred, error])
        fig.text(lefts[0], bottom+rowh+.008, label, va='bottom', fontsize=8)
        for col, (left, value) in enumerate(zip(lefts, values)):
            ax = fig.add_axes([left, bottom, width, rowh])
            error = col in [2,4]
            image = ax.pcolormesh(x, y, value, shading='nearest', rasterized=True,
                cmap=style.ERROR if error else cmap, vmin=0 if error else lo, vmax=emax if error else hi)
            ax.set(xlim=(float(x[0]),float(x[-1])), ylim=(float(y[0]),float(y[-1])))
            ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(labelleft=col==0, labelbottom=row==len(rows)-1, pad=2)
            if col==0: ax.set_ylabel('$t$' if task!='helmholtz' else '$y$', labelpad=2)
            if row==len(rows)-1: ax.set_xlabel('$x$', labelpad=2)
            if col==1: bar(fig,image,[fieldbar,bottom,.009,rowh])
            if col==cols-1: bar(fig,image,[errorbar,bottom,.009,rowh])
    finish(fig,name,task)


b = load('burgers'); rows = []
for i, nu in enumerate(b['parameter_pod']):
    rows.append((b['reference_pod'][i], [b['pod'][i]], b['x'], b['t'], rf'POD, $\nu={nu:g}$', style.SIGNED))
rows.append((b['reference_npod'], [b['npod']], b['x'], b['t'], r'NeuralPOD, $\nu=0.1$', style.SIGNED))
grid(rows,'burgers_fields_combined','burgers',164)

s = load('sod'); rows = []
for key in ['reference','pod','npod']:
    s[key] = s[key].reshape(4,3,16,128);s[key][:,1] *= np.sqrt(1.4)
for channel,label,ratio in [(0,r'$\rho$',.1),(1,'$u$',.1),(2,'$p$',.1),(0,r'$\rho$',.05),(0,r'$\rho$',.4)]:
    i = int(np.flatnonzero(np.isclose(s['parameter'],ratio))[0])
    rows.append((s['reference'][i,channel],[s['pod'][i,channel],s['npod'][i,channel]],
        np.unique(s['coords'][:,1]),np.unique(s['coords'][:,2]),label+rf', $r={ratio:g}$',
        style.SIGNED if channel==1 else style.POSITIVE))
grid(rows,'sod_fields_combined','sod',186)

c = load('coanda'); mesh = Triangulation(c['xy'][:,0],c['xy'][:,1],c['triangles'])
truth,pod,npod = [c[k].reshape(2,-1)[1] for k in ['reference','pod','npod']]
hi = max(float(abs(v).max()) for v in [truth,pod,npod]); emax=max(float(abs(pod-truth).max()),float(abs(npod-truth).max()))
fig = plt.figure(figsize=style.size(142))
for row,(values,title,error) in enumerate([(truth,'Reference',False),(pod,'POD',False),(npod,'NeuralPOD',False),
                                         (abs(pod-truth),'POD error',True),(abs(npod-truth),'NeuralPOD error',True)]):
    bottom=.82-row*.17;ax=fig.add_axes([.09,bottom,.79,.125])
    image=ax.tripcolor(mesh,values,shading='flat',rasterized=True,cmap=style.ERROR if error else style.SIGNED,
        vmin=0 if error else -hi,vmax=emax if error else hi)
    ax.set_aspect('equal');ax.set(xlim=(c['xy'][:,0].min(),c['xy'][:,0].max()),ylim=(c['xy'][:,1].min(),c['xy'][:,1].max()))
    ax.set_xticks([0,25,50]);ax.set_yticks([0,7.5]);ax.set_ylabel('$y$',labelpad=2);ax.set_title(title,pad=2)
    ax.tick_params(labelbottom=row==4,pad=2)
    if row==4:ax.set_xlabel('$x$',labelpad=2)
    if row==2:bar(fig,image,[.915,.48,.012,.39])
    if row==4:bar(fig,image,[.915,.14,.012,.22])
finish(fig,'coanda_fields_combined','coanda')

n = load('ns');fig=plt.figure(figsize=style.size(88))
lefts=[.075,.215,.355,.535,.675,.815]
for group,frame in enumerate([10,20]):
    fig.text(.275+group*.46,.985,rf'$t={n["time"][frame]:g}$',ha='center',va='top')
    for j,title in enumerate(['Reference','POD','NeuralPOD']):
        fig.text(lefts[group*3+j]+.055,.885,title,ha='center',va='top',fontsize=7)
for channel in [0,1]:
    bottom=.51-channel*.395
    hi=max(float(abs(n[k].reshape(21,64,64,2)[:,:,:,channel]).max()) for k in ['reference','pod','npod'])
    fig.text(.075,bottom+.29,'$u$' if channel==0 else '$v$',va='bottom')
    for group,frame in enumerate([10,20]):
        for j,key in enumerate(['reference','pod','npod']):
            col=group*3+j;field=n[key].reshape(21,64,64,2)[frame,:,:,channel]
            ax=fig.add_axes([lefts[col],bottom,.11,.25])
            image=ax.imshow(field.T,origin='lower',extent=(0,1,0,1),cmap=style.SIGNED,
                vmin=-hi,vmax=hi,rasterized=True,aspect='equal')
            ax.set(xticks=[0,1],yticks=[0,1]);ax.tick_params(labelleft=col==0,labelbottom=channel==1,pad=2)
            if col==0:ax.set_ylabel(r'$\xi_2$',labelpad=2)
            if channel==1:ax.set_xlabel(r'$\xi_1$',labelpad=2)
    bar(fig,image,[.936,bottom,.011,.25])
finish(fig,'ns_fields_improved','ns')

h = load('helmholtz');rows=[]
for i,omega in enumerate(h['omega']):
    rows.append((h['reference'][i].reshape(101,101).T,[h['pod'][i].reshape(101,101).T,h['npod'][i].reshape(101,101).T],
        np.linspace(0,1,101),np.linspace(0,1,101),rf'$\omega={omega:g}$',style.SIGNED))
grid(rows,'helmholtz_four_omega_fields','helmholtz',128)
(args.out/'neuralpod_fields_manifest.json').write_text(json.dumps(proof,indent=2)+'\n')
