"""Shared plotting utilities."""


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    PALETTE = sns.color_palette("Set2")
except ImportError:
    PALETTE = [plt.cm.tab10(i) for i in range(10)]

COLOR_MEAN = PALETTE[2]
COLOR_RESIDUAL = PALETTE[1]
C0, C1, C2     = plt.cm.tab10(0), plt.cm.tab10(1), plt.cm.tab10(2)


def plot_pod_phase1(sigmas: np.ndarray, save_path: str) -> None:
    energy = sigmas ** 2
    cumvar = np.cumsum(energy) / energy.sum() * 100
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.semilogy(range(1, len(sigmas) + 1), sigmas, "o-", color=C0, markersize=4, lw=1.5)
    ax.set_xlabel("Mode index"); ax.set_ylabel("Singular value")
    ax.set_title("Phase 1: singular value decay", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.plot(range(1, len(cumvar) + 1), cumvar, "o-", color=C0, markersize=4, lw=1.5)
    ax.axhline(99.99, color=C1, ls="--", lw=1.2, label="99.99%")
    ax.set_xlabel("Mode index"); ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("Phase 1: cumulative variance explained", fontweight="bold")
    ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_error_dist(errors: np.ndarray, title: str, save_path: str) -> None:
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.hist(errors, bins=30, color=C0, alpha=0.8, linewidth=0)
    ax.axvline(errors.mean(),     color=C1, ls="--", lw=1.5, label=f"Mean {errors.mean():.4f}")
    ax.axvline(np.median(errors), color=C2, ls="--", lw=1.5, label=f"Median {np.median(errors):.4f}")
    ax.set_xlabel("Relative L2 error"); ax.set_ylabel("Count")
    ax.set_title("Test error distribution", fontweight="bold")
    ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.plot(np.sort(errors), color=C0, lw=1.5)
    ax.axhline(errors.mean(), color=C1, ls="--", lw=1.2, alpha=0.8)
    ax.set_xlabel("Sample rank"); ax.set_ylabel("Relative L2 error")
    ax.set_title("Sorted test errors", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_cross_param_bar(
    cross_metrics: dict,
    trained_val,
    param_label: str,
    title: str,
    save_path: str,
    ylim: float = 1.0,
) -> None:
    keys = list(cross_metrics.keys())
    labels = [f"{param_label}={k:.4g}" if isinstance(k, float) else f"{param_label}={k}" for k in keys]
    means = [cross_metrics[k]["mean"]   for k in keys]
    medians = [cross_metrics[k]["median"] for k in keys]
    x_pos = np.arange(len(labels))
    _, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x_pos - 0.2,  means,   0.35, label="Mean",   color=C0, alpha=0.85, linewidth=0)
    ax.bar(x_pos + 0.15, medians, 0.35, label="Median", color=C1, alpha=0.85, linewidth=0)
    if trained_val in keys:
        ax.axvline(keys.index(trained_val), color="gray", ls="--", lw=1.2, alpha=0.7, label="Trained on")
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Relative L2 error")
    ax.set_title(title, fontweight="bold")
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.legend(framealpha=0.7); ax.grid(True, ls="--", alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_umap_regimes(
    alpha: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    hard_labels: np.ndarray,
    param_vals: np.ndarray,
    regime_colors: list,
    param_label: str,
    title: str,
    save_path: str,
    seed: int = 0,
) -> None:
    """3-panel UMAP: regime assignment, param coloring, and POD coefficient space."""
    from umap import UMAP
    from matplotlib.patches import Ellipse

    def _cov_ellipse(ax, mean2d, cov2d, color, n_std=2.0, **kw):
        vals, vecs = np.linalg.eigh(cov2d)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        w, h = 2 * n_std * np.sqrt(np.abs(vals))
        ax.add_patch(Ellipse(mean2d, w, h, angle=angle,
                             edgecolor=color, facecolor="none", lw=2, **kw))

    reducer = UMAP(n_neighbors=30, min_dist=0.0, random_state=seed)
    embedding = reducer.fit_transform(alpha)
    mu_umap = reducer.transform(mu)

    M = mu.shape[0]
    param_unique = np.unique(param_vals)
    PCMAP = plt.cm.Dark2
    param_to_col = {v: PCMAP(i / max(len(param_unique) - 1, 1))
                    for i, v in enumerate(param_unique)}
    point_colors_reg = [regime_colors[m] for m in hard_labels]
    point_colors_param = [param_to_col[float(v)] for v in param_vals]
    centroid_kw = dict(s=120, marker="D", zorder=10, edgecolors="white", linewidths=0.8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(embedding[:, 0], embedding[:, 1],
                    c=point_colors_reg, s=3, alpha=0.3, rasterized=True, linewidths=0)
    for m in range(M):
        axes[0].scatter(mu_umap[m, 0], mu_umap[m, 1], color=regime_colors[m], **centroid_kw)
    axes[0].set_title("UMAP — regime assignment", fontweight="bold")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=regime_colors[m], markersize=7,
                          label=f"Regime {m+1}") for m in range(M)]
    axes[0].legend(handles=handles, fontsize=9, framealpha=0.7)

    axes[1].scatter(embedding[:, 0], embedding[:, 1],
                    c=point_colors_param, s=3, alpha=0.3, rasterized=True, linewidths=0)
    for m in range(M):
        axes[1].scatter(mu_umap[m, 0], mu_umap[m, 1], color=regime_colors[m], **centroid_kw)
    axes[1].set_title(f"UMAP — colored by {param_label}", fontweight="bold")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=param_to_col[v], markersize=7,
                          label=f"{param_label}={v:.4g}") for v in param_unique]
    axes[1].legend(handles=handles, fontsize=9, framealpha=0.7)

    axes[2].scatter(alpha[:, 0], alpha[:, 1],
                    c=point_colors_param, s=3, alpha=0.3, rasterized=True, linewidths=0)
    for m in range(M):
        _cov_ellipse(axes[2], mu[m, :2], Sigma[m, :2, :2], color=regime_colors[m], linestyle="--")
        axes[2].scatter(*mu[m, :2], color=regime_colors[m], **centroid_kw, label=f"Regime {m+1}")
    axes[2].set_title(r"POD space: $\alpha_1$ vs $\alpha_2$", fontweight="bold")
    axes[2].set_xlabel(r"$\alpha_1$"); axes[2].set_ylabel(r"$\alpha_2$")
    axes[2].legend(fontsize=9, framealpha=0.7)

    for ax in axes[:2]:
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    for ax in axes:
        ax.grid(True, ls="--", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle(title, fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_reconstruction_xy(
    true_list: list,
    pred_list: list,
    rl2_list: list,
    x_np: np.ndarray,
    y_np: np.ndarray,
    model_name: str,
    save_path: str,
    row_labels: list = None,
) -> None:
    """Spatial reconstruction panels for 2D steady-state PDEs."""
    n = len(true_list)
    _, axes = plt.subplots(n, 3, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[None, :]
    for row, (true, pred, rl2) in enumerate(zip(true_list, pred_list, rl2_list)):
        err = np.abs(true - pred)
        vmax = np.abs(true).max()
        label = f"{row_labels[row]},  rel L2={rl2:.3f}" if row_labels else f"rel L2={rl2:.3f}"
        for col, (arr, ttl, cmap, vmin, vm) in enumerate([
            (true, "Ground Truth",   "viridis", 0,    vmax),
            (pred, "Prediction",     "viridis", 0,    vmax),
            (err,  "Absolute Error", "Oranges", 0,    err.max()),
        ]):
            ax = axes[row, col]
            im = ax.imshow(arr, aspect="equal", origin="lower", cmap=cmap,
                           extent=[x_np.min(), x_np.max(), y_np.min(), y_np.max()],
                           vmin=vmin, vmax=vm)
            if row == 0: ax.set_title(ttl, fontweight="bold")
            if col == 0:
                ax.set_ylabel("y", fontsize=9)
                ax.text(0.02, 0.98, label, transform=ax.transAxes,
                        ha="left", va="top", fontsize=8, fontweight="bold",
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))
            if row == n - 1: ax.set_xlabel("x")
            ax.spines[["top", "right"]].set_visible(False)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(f"{model_name}: reconstruction examples", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
