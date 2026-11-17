import numpy as np
import matplotlib.pyplot as plt


def draw_graph(coords, edge_index, skel=None, background=None, node_labels=None,
               title="graph", save_path=None, node_size=40, class_names=None,
               root_idx=None):                                    # <-- new arg
    if skel is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(8, 8))

    if background is not None:
        ax.imshow(background, cmap="gray", interpolation="nearest")

    ys, xs = coords[:, 0], coords[:, 1]

    for u, v in edge_index.T:
        ax.plot([xs[u], xs[v]], [ys[u], ys[v]],
                "-", color="deepskyblue", lw=1.5, alpha=0.9, zorder=2)

    if node_labels is None:
        ax.scatter(xs, ys, s=node_size, c="red",
                   edgecolors="white", linewidths=0.6, zorder=3)
    else:
        node_labels = np.asarray(node_labels)
        cmap = plt.get_cmap("tab10")
        for i, lab in enumerate(sorted(set(node_labels.tolist()))):
            m = node_labels == lab
            name = (class_names.get(lab, str(lab)) if class_names else str(lab))
            ax.scatter(xs[m], ys[m], s=node_size, color=cmap(i % 10),
                       edgecolors="white", linewidths=0.6, zorder=3, label=name)
        ax.legend(loc="upper right", fontsize=9, title="node class")

    deg = np.zeros(len(coords), int)
    for u, v in edge_index.T:
        deg[u] += 1; deg[v] += 1
    for mask, mk in [(deg == 1, "s"), (deg >= 3, "D")]:
        if mask.any():
            ax.scatter(xs[mask], ys[mask], s=node_size * 2.2, facecolors="none",
                       edgecolors="yellow", linewidths=1.6, marker=mk, zorder=4)

    # ---- ROOT marker ----
    if root_idx is not None:
        ax.scatter(xs[root_idx], ys[root_idx], s=node_size * 6, marker="*",
                   color="magenta", edgecolors="black", linewidths=1.2, zorder=5,
                   label="root")

    ax.set_title(f"{title}  |  {len(coords)} nodes, {edge_index.shape[1]} edges")
    ax.set_aspect("equal"); ax.axis("off")
    if skel is not None:
        axes[1].imshow(skel, cmap="gray")
        if root_idx is not None:
            axes[1].scatter(xs[root_idx], ys[root_idx], s=node_size * 6,
                            marker="*", color="magenta", edgecolors="black",
                            linewidths=1.2, zorder=5)
        axes[1].set_title("skeleton")
        axes[1].axis("off")
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()