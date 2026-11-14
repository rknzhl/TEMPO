"""Plot the two TEMPO predictions on one saved Darcy test field."""
from pathlib import Path
import argparse
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'tempo-darcy-plot-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
from style import apply, size, POSITIVE, ERROR

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, default=HERE)
parser.add_argument('--out', type=Path, default=HERE)
args = parser.parse_args()
with np.load(args.data / 'darcy_fields_inputs.npz', allow_pickle=False) as data:
    reference, pod, npod = [data[key] * float(data['beta']) for key in ['reference', 'pod', 'npod']]
fields = [reference, pod, npod, np.abs(pod - reference), np.abs(npod - reference)]
pressure = Normalize(min(0., min(float(f.min()) for f in fields[:3])),
                     max(float(f.max()) for f in fields[:3]))
error = Normalize(0., max(float(f.max()) for f in fields[3:]))
apply()
fig = plt.figure(figsize=size(46))
images = []
for i, (field, x, title) in enumerate(zip(fields, [.06, .215, .37, .60, .765],
        ['Reference', 'POD', 'NeuralPOD', 'POD error', 'NeuralPOD error'])):
    ax = fig.add_axes([x, .23, .135, .56])
    images.append(ax.imshow(field.T, origin='lower', extent=(0, 1, 0, 1),
                            cmap=POSITIVE if i < 3 else ERROR,
                            norm=pressure if i < 3 else error, interpolation='nearest'))
    ax.set_title(title, fontsize=7.5)
    ax.set_xticks([0, .5, 1], ['0', '0.5', '1'])
    ax.set_yticks([0, .5, 1], ['0', '0.5', '1'] if i == 0 else [])
    ax.tick_params(labelsize=6.5, pad=2)
    if i == 0:
        ax.set_ylabel('$y$', fontsize=8)
for image, x in [(images[2], .52), (images[-1], .92)]:
    ax = fig.add_axes([x, .261, .012, .498])
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    bar = fig.colorbar(image, cax=ax, format=formatter)
    bar.ax.tick_params(labelsize=6.5, pad=2)
    fig.canvas.draw()
    exponent = bar.ax.yaxis.get_offset_text().get_text()
    bar.ax.yaxis.get_offset_text().set_visible(False)
    fig.text(x, .82, exponent, ha='left', fontsize=6.5)
fig.text(.48, .075, '$x$', ha='center', fontsize=8)
args.out.mkdir(parents=True, exist_ok=True)
for extension in ['pdf', 'png']:
    metadata = {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None
    fig.savefig(args.out / f'darcy_fields.{extension}', dpi=300, metadata=metadata)
plt.close(fig)
print('Rendered reference, both TEMPO variants and errors on shared scales.')
