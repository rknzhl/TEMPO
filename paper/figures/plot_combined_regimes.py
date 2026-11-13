"""Plot regime assignments, physical parameters and POD coordinates."""
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'tempo-fig-mpl'))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, SymmetricalLogLocator

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--burgers', type=Path, required=True)
p.add_argument('--other-tasks', type=Path, required=True)
p.add_argument('--helmholtz', type=Path, required=True)
p.add_argument('--darcy', type=Path, required=True)
p.add_argument('--style-dir', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
sys.path.insert(0, str(args.style_dir))
import style
style.apply()
plt.rcParams.update({'font.size': 7, 'axes.labelsize': 6.7,
    'axes.titlesize': 7, 'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3,
    'legend.fontsize': 6.3})
args.out.mkdir(parents=True, exist_ok=True)


def read(path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


b = read(args.burgers, 'f46f579a59a0657409fed8e179590f63e0de096c75650de1997e92230b7a2253')
o = read(args.other_tasks, '9bd1512c7a5019826a0618c413da971159d27d321f7ac6a01c5d5f5cd9b6b512')
h = read(args.helmholtz, 'b7eecb1bfa63194a7384531b3a97efc1a028ea16cea705652e8e5086be80b4bf')
d = read(args.darcy, 'bac0900cc578d308b44418e4c5cb41b102dcf95115131da62c939b57f28fb8b1')
rows = [('Burgers', b['embedding'], b['alpha'], b['gamma'], b['parameter'],
         b['centroid_embedding'], b['mu'], b['Sigma'])]
for task, name in [('sod', 'Sod'), ('coanda', 'Coanda')]:
    rows.append((name, *(o[task+'__'+key] for key in ('embedding', 'alpha', 'gamma',
        'parameter', 'center_embedding', 'responsibility_centers', 'responsibility_covariance'))))
rows.append(('Helmholtz', h['embedding'], h['alpha'], h['gamma'], h['omega'],
             h['centres_embedding'], h['centres_pod'], None))
rows.append(('Darcy', d['embedding'], d['alpha'], d['gamma'], d['mean_permeability'],
             d['center_embedding'], d['centers'], d['covariance']))

fig = plt.figure(figsize=(345/72.27, 6.2))
lefts = [.14, .435, .75]
width, height = .22, .13
for left, title in zip(lefts, ['Regime', 'Parameter', 'POD coordinates']):
    fig.text(left+width/2, .978, title, ha='center', va='top')
proof = []
continuous = LinearSegmentedColormap.from_list('viscosity_blue_red',
        [style.REGIMES[0], '#F4F4F4', style.REGIMES[1]])
for index, (name, xy, alpha, gamma, parameter, cxy, centers, cov) in enumerate(rows):
    bottom = [.815, .6325, .45, .2675, .085][index]
    assert len(xy) == len(alpha) == len(gamma)
    assert parameter is None or len(xy) == len(parameter)
    assert np.isfinite(xy).all() and np.allclose(gamma.sum(1), 1, atol=1e-6)
    axes = [fig.add_axes([left, bottom, width, height]) for left in lefts]
    fig.text(.018, bottom+height/2, name, rotation=90, ha='center', va='center', fontsize=7)
    hard = gamma.argmax(1)
    colors = np.asarray(style.REGIMES)
    point_size = 2.2 if len(xy) > 2000 else 5
    axes[0].scatter(*xy.T, c=colors[hard], s=point_size, alpha=.8,
                    linewidths=0, rasterized=True)
    axes[2].scatter(*alpha[:, :2].T, c=colors[hard], s=point_size, alpha=.6,
                    linewidths=0, rasterized=True)
    unique = np.unique(parameter)
    if len(unique) <= 4:
        parameter_color = colors[np.searchsorted(unique, parameter)]
        axes[1].scatter(*xy.T, c=parameter_color, s=point_size, alpha=.8,
                        linewidths=0, rasterized=True)
        handles = [Line2D([], [], marker='o', ls='', color=colors[k], ms=3,
                          label=f'{v:g}') for k,v in enumerate(unique)]
        parameter_symbol = '$r$' if name=='Sod' else r'$\omega$' if name=='Helmholtz' else r'$\nu$'
        axes[1].legend(handles=handles, ncol=len(unique), title=parameter_symbol,
            loc='upper center', bbox_to_anchor=(.5,-.04), handletextpad=.15,
            handlelength=.55, columnspacing=.3, labelspacing=.15,
            borderaxespad=0, fontsize=6.3, title_fontsize=6.3)
    else:
        points = axes[1].scatter(*xy.T, c=parameter, cmap=continuous,
            s=point_size, alpha=.9, linewidths=0, rasterized=True)
        cb = fig.colorbar(points, cax=fig.add_axes([lefts[1]+.02,bottom-.017,width-.04,.006]),
                          orientation='horizontal')
        cb.set_ticks([.5,1.25,2] if name == 'Coanda' else [parameter.min(), (parameter.min()+parameter.max())/2, parameter.max()])
        cb.ax.tick_params(labelsize=6.3,length=2,pad=1)
        if name == 'Darcy':
            cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))
        cb.set_label(r'$\nu$' if name == 'Coanda' else r'$\bar a$',labelpad=0, fontsize=6.3)
        cb.outline.set_linewidth(.35)
    for m in range(gamma.shape[1]):
        for ax in axes[:2]:
            ax.scatter(*cxy[m], marker='D', s=25,
                c=colors[m] if ax is axes[0] else '#777777',
                edgecolors='white', linewidths=.55, zorder=5)
        axes[2].scatter(*centers[m,:2], marker='D', s=25, color=colors[m],
                         edgecolors='white', linewidths=.55, zorder=5)
        if cov is not None:
            eig, vec = np.linalg.eigh(cov[m,:2,:2])
            assert eig.min() >= -1e-5
            # Two-standard-deviation covariance contours.
            angle = np.degrees(np.arctan2(vec[1,0],vec[0,0]))
            axes[2].add_patch(Ellipse(centers[m,:2], *list(4*np.sqrt(np.maximum(eig,0))),
                             angle=angle, fill=False, edgecolor=colors[m], lw=.7))
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    for ax in axes[:2]:
        ax.set_xticks([]);ax.set_yticks([])
    axes[1].set_xlim(axes[0].get_xlim());axes[1].set_ylim(axes[0].get_ylim())
    axes[0].set_ylabel('UMAP 2')
    axes[2].set_xlabel(r'$\alpha_1$',labelpad=2)
    axes[2].set_ylabel(r'$\alpha_2$',labelpad=2)
    lo,hi=axes[2].get_xlim()
    ticks=MaxNLocator(3).tick_values(lo,hi)
    axes[2].set_xticks(ticks[(ticks>=lo)&(ticks<=hi)])
    if name == 'Helmholtz':
        threshold = float(np.percentile(abs(alpha[:,1]),95)/10)
        axes[2].set_yscale('symlog', linthresh=threshold, linscale=1)
        axes[2].yaxis.set_major_locator(SymmetricalLogLocator(base=10,linthresh=threshold))
        axes[2].set_ylabel(r'$\alpha_2$ (symlog)',labelpad=1)
        lo,hi=axes[2].get_ylim()
        ticks=axes[2].yaxis.get_major_locator().tick_values(lo,hi)
        ticks=ticks[(ticks==0)|(abs(ticks)>=10**np.ceil(np.log10(threshold)))]
        axes[2].set_yticks(ticks[(ticks>=lo)&(ticks<=hi)])
    else:
        lo,hi=axes[2].get_ylim()
        ticks=MaxNLocator(3).tick_values(lo,hi)
        axes[2].set_yticks(ticks[(ticks>=lo)&(ticks<=hi)])
    axes[2].tick_params(labelsize=6.3,pad=2)
    handles = [Line2D([], [], marker='o', ls='', color=colors[m], ms=3,
                      label=str(m+1)) for m in range(gamma.shape[1])]
    axes[0].legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.04),
        ncol=len(handles),handletextpad=.15,handlelength=.55,columnspacing=.5,
        borderaxespad=0,fontsize=6.3)
    if name == 'Darcy':
        axes[0].set_xlabel('UMAP 1',labelpad=18)
    proof.append(dict(task=name, points=len(xy), regimes=gamma.shape[1],
        embedding_sha256=hashlib.sha256(xy.tobytes()).hexdigest(),
        gamma_sha256=hashlib.sha256(gamma.tobytes()).hexdigest(),
        array_sha256={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in
            [('alpha',alpha),('centers_embedding',cxy),('centers_pod',centers)]+([] if parameter is None else [('parameter',parameter)])+([] if cov is None else [('covariance',cov)])},
        covariance_ellipses=cov is not None,
        pod_vertical_scale='symlog' if name=='Helmholtz' else 'linear',
        all_points_inside_limits=bool(
            axes[0].get_xlim()[0] <= xy[:,0].min() <= xy[:,0].max() <= axes[0].get_xlim()[1]
            and axes[0].get_ylim()[0] <= xy[:,1].min() <= xy[:,1].max() <= axes[0].get_ylim()[1]
            and axes[2].get_xlim()[0] <= alpha[:,0].min() <= alpha[:,0].max() <= axes[2].get_xlim()[1]
            and axes[2].get_ylim()[0] <= alpha[:,1].min() <= alpha[:,1].max() <= axes[2].get_ylim()[1])))
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
font_sizes=[]
for text in fig.findobj(Text):
    if text.get_visible() and text.get_text():
        box=text.get_window_extent(renderer)
        assert box.x0 >= -.5 and box.y0 >= -.5 and box.x1 <= fig.bbox.width+.5 and box.y1 <= fig.bbox.height+.5, text.get_text()
        font_sizes.append(text.get_fontsize())
assert min(font_sizes)>=6.3
assert all(row['all_points_inside_limits'] for row in proof)
outputs={}
for extension in ['pdf','png']:
    path=args.out/('regimes_combined.'+extension)
    fig.savefig(path,dpi=300,facecolor='white',metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else None)
    outputs[extension]=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
plt.close(fig)
(args.out/'regimes_combined_manifest.json').write_text(json.dumps(dict(
    status='complete', source_arrays=[str(args.burgers),str(args.other_tasks),str(args.helmholtz),str(args.darcy)],
    source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.burgers,args.other_tasks,args.helmholtz,args.darcy)},
    rows=proof, darcy_parameter_panel=True,darcy_parameter="spatial mean permeability",coanda_colorbar_unchanged=True,plot_only=True, model_training=False, embedding_refit=False,
    original_png_used=False, covariance_contour='Original two-standard-deviation ellipses for Burgers, Sod, Coanda and Darcy; no ellipses for Helmholtz.',
    centers='Original saved/transformed centers for Burgers M3; responsibility-weighted training centers for Sod, Coanda, Helmholtz and Darcy.',
    layout=dict(width_tex_pt=345,height_tex_pt=6.2*72.27,minimum_text_size_pt=min(font_sizes),placed_scale_at_345pt=1,text_inside_canvas=True),
    palette=style.REGIMES[:4],outputs=outputs),indent=2)+'\n')
