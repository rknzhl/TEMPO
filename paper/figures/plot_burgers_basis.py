"""Plot Burgers basis spectra and modes from saved arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--style-dir', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.style_dir))
import style
style.apply()
args.out.mkdir(parents=True, exist_ok=True)
expected = 'dfedbb04e97843d124483052a62cf48fb57e2e80f2a1015e2cb8f5dea93d399f'
assert hashlib.sha256(args.data.read_bytes()).hexdigest() == expected
with np.load(args.data, allow_pickle=False) as z:
    arrays = {k: z[k].copy() for k in z.files}
metadata = json.loads(str(arrays['metadata_json']))
manifest = dict(source_sha256=expected, models=metadata['models'], grid=metadata['grid'],
                training=False, figures={})


def save(fig, name):
    fig.canvas.draw()
    for extension in ('pdf', 'png'):
        path = args.out / (name + '.' + extension)
        fig.savefig(path, dpi=300, facecolor='white')
    manifest['figures'][name] = dict(pdf=name+'.pdf', png=name+'.png')
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=style.size(55))
fig.subplots_adjust(left=.095, right=.985, bottom=.23, top=.88, wspace=.44)
series = [arrays['singular_value'] / arrays['singular_value'][0],
          arrays['neural_residual'] / arrays['neural_residual'][0]]
for ax, values, color, title, label in zip(axes, series,
        [style.REGIMES[0], style.REGIMES[1]], ['POD', 'NeuralPOD'],
        [r'$\sigma_k/\sigma_1$', r'$R_k/R_1$']):
    k = np.arange(1, len(values) + 1)
    ax.semilogy(k, values, color=color, lw=1.2,
                marker='o', ms=2.2, markevery=2 if len(values) > 32 else 1)
    ax.set_title(title)
    ax.set_xlabel('Modes')
    ax.set_ylabel(label)
    ax.set_xlim(1, len(values))
    ax.set_xticks([1, 16, 32, 48, 64] if len(values) == 64 else [1, 8, 16, 24, 32])
    ax.grid(axis='y', which='major', alpha=.17, lw=.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_minor_formatter(NullFormatter())
    if title == 'NeuralPOD':
        ax.set_yticks([.4, .6, 1.], labels=['0.4', '0.6', '1'])
save(fig, 'burgers_basis_spectrum')

# Normalize display amplitude and fix the arbitrary mode sign.
fields, signs, amplitudes = [], [], []
nt, nx = metadata['grid']
for key in ('pod_modes', 'neural_modes'):
    phi = arrays[key].astype('f8')
    peak = abs(phi).max(0)
    sign = np.sign(phi[np.argmax(abs(phi), axis=0), np.arange(phi.shape[1])])
    sign[sign == 0] = 1
    fields.append((phi / peak * sign).reshape(nt, nx, 8))
    signs.append(sign.tolist())
    amplitudes.append(peak.tolist())
cmap = LinearSegmentedColormap.from_list('basis_blue_red',
        [style.REGIMES[0], '#F8F8F8', style.REGIMES[1]], N=256)
fig, axes = plt.subplots(2, 8, figsize=style.size(68))
fig.subplots_adjust(left=.105, right=.945, bottom=.14, top=.88, wspace=.12, hspace=.18)
for row in range(2):
    for col in range(8):
        ax = axes[row, col]
        im = ax.imshow(fields[row][:, :, col], origin='lower', interpolation='none',
            extent=[float(arrays['x'][0]), float(arrays['x'][-1]),
                    float(arrays['t'][0]), float(arrays['t'][-1])],
            aspect='auto', cmap=cmap, vmin=-1, vmax=1, rasterized=True)
        ax.set_xticks([0, 1], labels=['0', '1'])
        ax.set_yticks([0, 2], labels=['0', '2'])
        ax.tick_params(pad=1.5, labelleft=col == 0, labelbottom=row == 1 and col == 0,
                       length=2, labelsize=6)
        if row == 0:
            ax.set_title(r'$\phi_{%d}$' % (col + 1), pad=4)
        if col == 0:
            ax.set_ylabel('$t$', labelpad=1)
        for spine in ax.spines.values():
            spine.set_linewidth(.35)
    box = axes[row, 0].get_position()
    fig.text(.016, box.y0 + box.height/2, ['POD', 'NeuralPOD'][row],
             rotation=90, ha='center', va='center', fontsize=7)
fig.text(.525, .026, '$x$', ha='center')
cb = fig.colorbar(im, cax=fig.add_axes([.965, .22, .008, .58]))
cb.set_ticks([-1, 0, 1]); cb.ax.tick_params(labelsize=6, pad=2, length=2)
cb.outline.set_linewidth(.35)
save(fig, 'burgers_basis_modes')
manifest['display'] = dict(mode_indices=list(range(1,9)), normalization='per-mode maximum absolute value',
                           sign_rule='largest-magnitude entry positive', signs=signs, amplitudes=amplitudes,
                           scientific_array_resampling=False)
(args.out / 'basis_figures_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
