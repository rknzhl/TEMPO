"""Plot Coanda EM diagnostics and training losses."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
from style import apply, size, save, REGIMES

def clean(ax):
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4) if ax.get_yscale() == 'log' else MaxNLocator(3))
    ax.spines[['top', 'right']].set_visible(False)


def training(z, output):
    d = lambda key: z['coanda__'+key]
    fig = plt.figure(figsize=size(93))
    gs = fig.add_gridspec(2, 6, left=.09, right=.985, bottom=.13, top=.91, hspace=.88, wspace=1.2)
    axes = [fig.add_subplot(gs[0, i:i+2]) for i in (0, 2, 4)]
    iteration = d('em_iter')
    axes[0].plot(iteration, d('em_ll'), color=REGIMES[0], marker='o', markersize=2)
    axes[0].set_ylabel('Log-likelihood')
    axes[1].plot(iteration, d('em_entropy'), color='#777777', marker='o', markersize=2)
    axes[1].set_ylabel('Entropy')
    for m in range(d('em_delta').shape[1]):
        axes[2].semilogy(iteration, d('em_delta')[:, m], color=REGIMES[m], marker='o', markersize=2, label=f'Regime {m+1}')
    axes[2].set_ylabel(r'$\Delta_m$')
    axes[2].legend(fontsize=6, loc='upper right', handlelength=1)
    for ax in axes:
        ax.set_xlabel('EM iteration')
        clean(ax)
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    epochs = np.arange(1, len(d('online_data'))+1)
    ax = fig.add_subplot(gs[1, :3])
    ax.plot(epochs, d('online_total'), label='Total', color=REGIMES[0])
    ax.plot(epochs, d('online_data'), label='Data', color='#777777')
    ax.set(xlabel='Epoch', ylabel='Training objective')
    ax.legend(ncol=2, loc='upper right', handlelength=1.2)
    clean(ax)
    ax = fig.add_subplot(gs[1, 3:])
    ax.plot(epochs, d('online_kl'), label='KL', color=REGIMES[1])
    ax.plot(epochs, d('online_ent'), label='Entropy', color=REGIMES[2])
    ax.set(xlabel='Epoch', ylabel='Regularization')
    ax.legend(ncol=2, loc='upper right', handlelength=1.2)
    clean(ax)
    fig.text(.5, .985, 'TEMPO · Coanda', ha='center', va='top')
    save(fig, output/'coanda_training')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    apply()
    with np.load(args.data, allow_pickle=False) as arrays:
        training(arrays, args.out)


if __name__ == "__main__":
    main()
