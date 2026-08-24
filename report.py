"""
Turn results/metrics.json into charts and a self-contained HTML report.

Every chart is rendered twice — once for a light surface, once for a dark one —
and the HTML swaps them with prefers-color-scheme / [data-theme], so the report
is readable in either theme. Palette is the validated categorical default
(blue / orange / aqua / yellow, adjacent-pair CVD-safe in both modes).
"""

import logging as logger
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from models import ExperimentResults
from template import build_html

Theme = dict[str, object]

logger.basicConfig(level=logger.INFO, format="%(asctime)s: %(levelname)s: %(message)s")
logging = logger.getLogger(__name__)

RESULTS: Path = Path("results")
CHARTS: Path = RESULTS / "charts"

THEMES: dict[str, Theme] = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e3e2de",
        "faint": "#c9c8c3",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "seq": ["#eef4fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#383835",
        "faint": "#52514e",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "seq": ["#14202e", "#104281", "#184f95", "#256abf", "#3987e5", "#86b6ef"],
    },
}

# Display order: the two LLM arms, the headline embedding arm, then the control.
ORDER: list[str] = [
    "gemini-2.5-flash",
    "gpt-4o-mini",
    "embed-CLASSIFICATION",
    "embed-SEMANTIC_SIMILARITY",
]
TERSE: dict[str, str] = {
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gpt-4o-mini": "GPT-4o mini",
    "embed-CLASSIFICATION": "embed · CLASSIFICATION",
    "embed-SEMANTIC_SIMILARITY": "embed · SEMANTIC_SIMILARITY",
}
SHORT: dict[str, str] = {
    "gemini-2.5-flash": "Gemini 2.5 Flash\n(zero-shot)",
    "gpt-4o-mini": "GPT-4o mini\n(zero-shot)",
    "embed-CLASSIFICATION": "gemini-embedding-001\nCLASSIFICATION + LR",
    "embed-SEMANTIC_SIMILARITY": "gemini-embedding-001\nSEMANTIC_SIMILARITY + LR",
}


def style(theme: Theme) -> None:
    """Apply theme colors to matplotlib rcParams."""
    plt.rcParams.update({
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "text.color": theme["text"],
        "axes.labelcolor": theme["muted"],
        "axes.edgecolor": theme["grid"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["muted"],
        "grid.color": theme["grid"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
    })


def bare(ax: Axes, yaxis: bool = True) -> None:
    """Recessive grid, no vertical clutter."""
    if yaxis:
        ax.yaxis.grid(True, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save(fig: Figure, name: str, mode: str) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / f"{name}_{mode}.png", bbox_inches="tight")
    plt.close(fig)


def chart_headline(m: ExperimentResults, theme: Theme, mode: str) -> None:
    """Grouped bars: accuracy and macro-F1 per system, with 95% CI."""
    sys = m.systems
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(ORDER))
    width = 0.34
    acc = [sys[n].accuracy for n in ORDER]
    f1 = [sys[n].macro_f1 for n in ORDER]
    err = np.array([[sys[n].accuracy - sys[n].accuracy_ci95[0] for n in ORDER],
                    [sys[n].accuracy_ci95[1] - sys[n].accuracy for n in ORDER]])

    ax.bar(x - width / 2, acc, width * 0.9, label="Accuracy",
           color=theme["series"][0], yerr=err, capsize=3,
           error_kw={"ecolor": theme["muted"], "elinewidth": 1.2},
           edgecolor=theme["surface"], linewidth=2)
    ax.bar(x + width / 2, f1, width * 0.9, label="Macro F1",
           color=theme["series"][1], edgecolor=theme["surface"], linewidth=2)
    for xi, (a, f) in enumerate(zip(acc, f1)):
        top = sys[ORDER[xi]].accuracy_ci95[1]
        ax.text(xi - width / 2, top + 0.022, f"{a:.3f}", ha="center",
                fontsize=9, color=theme["text"])
        ax.text(xi + width / 2, f + 0.015, f"{f:.3f}", ha="center",
                fontsize=9, color=theme["text"])
    ax.set_xticks(x, [SHORT[n] for n in ORDER], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("Classification quality on 20 Newsgroups (4 classes)")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=9)
    bare(ax)
    fig.text(0.5, -0.06, "Error bars: 95% bootstrap CI on accuracy (2,000 resamples)",
             ha="center", fontsize=8, color=theme["muted"])
    save(fig, "accuracy_f1", mode)


def chart_per_class(m: ExperimentResults, theme: Theme, mode: str) -> None:
    """Per-class F1 grouped bars."""
    sys, labels = m.systems, m.config.labels
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(labels))
    width = 0.2
    for i, name in enumerate(ORDER):
        vals = [sys[name].per_class_f1[c] for c in labels]
        ax.bar(x + (i - 1.5) * width, vals, width * 0.9,
               label=SHORT[name].replace("\n", " "), color=theme["series"][i],
               edgecolor=theme["surface"], linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1.5) * width, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=7, color=theme["muted"], rotation=90)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 — where each system loses points")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.32))
    bare(ax)
    save(fig, "per_class_f1", mode)


def chart_confusion(m: ExperimentResults, theme: Theme, mode: str) -> None:
    """One sequential heatmap per system."""
    sys, labels = m.systems, m.config.labels
    cmap = LinearSegmentedColormap.from_list("seq", theme["seq"])
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, name in zip(axes, ORDER):
        cm = np.array(sys[name].confusion, dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        fontsize=9,
                        color="#ffffff" if norm[i, j] > 0.5 else theme["text"])
        ax.set_xticks(range(len(labels)), labels, fontsize=8, rotation=45,
                      ha="right")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(SHORT[name], fontsize=9)
        ax.set_xlabel("predicted", fontsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("actual", fontsize=8)
    fig.suptitle("Confusion matrices (counts; shaded by row-normalised rate)",
                 fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "confusion", mode)


def chart_latency(m: ExperimentResults, theme: Theme, mode: str) -> None:
    """Per-document wall-clock latency distribution."""
    sys = m.systems
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [sys[n].latencies for n in ORDER]
    bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.55,
                    showfliers=True,
                    flierprops={"marker": ".", "markersize": 3,
                                "markerfacecolor": theme["faint"],
                                "markeredgecolor": "none"},
                    medianprops={"color": theme["surface"], "linewidth": 2},
                    whiskerprops={"color": theme["muted"], "linewidth": 1},
                    capprops={"color": theme["muted"], "linewidth": 1})
    for patch, color in zip(bp["boxes"], theme["series"]):
        patch.set(facecolor=color, edgecolor=theme["surface"], linewidth=2)
    for i, name in enumerate(ORDER):
        ax.text(sys[name].latency_mean_s, i + 1.42,
                f"mean {sys[name].latency_mean_s*1000:.0f} ms",
                fontsize=8, color=theme["muted"], ha="center")
    ax.set_yticks(range(1, len(ORDER) + 1),
                  [SHORT[n].replace("\n", " ") for n in ORDER], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("seconds per document (log scale)")
    ax.set_title("Latency per document — amortised for the embedding arms")
    ax.xaxis.grid(True, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    save(fig, "latency", mode)


def chart_cost(m: ExperimentResults, theme: Theme, mode: str) -> None:
    """Cost per 1,000 documents against accuracy."""
    sys = m.systems
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, name in enumerate(ORDER):
        s = sys[name]
        ax.scatter(s.cost_usd_per_1k_docs, s.accuracy, s=170,
                   color=theme["series"][i], edgecolor=theme["surface"],
                   linewidth=2, zorder=3)
        ax.annotate(TERSE[name], (s.cost_usd_per_1k_docs, s.accuracy),
                    textcoords="offset points", xytext=(0, 15), ha="center",
                    fontsize=8.5, color=theme["text"])
    ax.set_xscale("log")
    ax.set_xlabel("inference cost, USD per 1,000 documents (log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title("Cost vs accuracy — up and to the left is better")
    bare(ax)
    ax.xaxis.grid(True, linewidth=0.7, alpha=0.8)
    ax.margins(x=0.9, y=0.28)
    save(fig, "cost_accuracy", mode)


def chart_pca(m: ExperimentResults, theme: Theme, mode: str) -> None:
    """Faceted 2-D PCA of the CLASSIFICATION-task embeddings, one class lit per panel."""
    path = RESULTS / "test_embeddings.npy"
    if not path.exists():
        return
    from sklearn.decomposition import PCA

    vec = np.load(path)
    labels = np.array(m.y_test)
    coords = PCA(n_components=2, random_state=42).fit_transform(vec)
    classes = m.config.labels
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True, sharey=True)
    for ax, cls, color in zip(axes, classes, theme["series"]):
        other = labels != cls
        ax.scatter(coords[other, 0], coords[other, 1], s=9, color=theme["faint"],
                   alpha=0.55, linewidth=0)
        ax.scatter(coords[~other, 0], coords[~other, 1], s=14, color=color,
                   edgecolor=theme["surface"], linewidth=0.4)
        ax.set_title(cls, fontsize=10, color=theme["text"])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(theme["grid"])
    fig.suptitle("Why the embedding arm works: PCA of CLASSIFICATION-task "
                 "embeddings, one class highlighted per panel",
                 fontsize=12, fontweight="bold", y=1.06)
    fig.tight_layout()
    save(fig, "embedding_pca", mode)



def main() -> None:
    m = ExperimentResults.model_validate_json(
        (RESULTS / "metrics.json").read_bytes())
    for mode, theme in THEMES.items():
        style(theme)
        chart_headline(m, theme, mode)
        chart_per_class(m, theme, mode)
        chart_confusion(m, theme, mode)
        chart_latency(m, theme, mode)
        chart_cost(m, theme, mode)
        chart_pca(m, theme, mode)
        logging.info("charts rendered: %s", mode)
    out = RESULTS / "report.html"
    out.write_text(build_html(m))
    logging.info("wrote %s (%.1f KB)", out, out.stat().st_size / 1024)


if __name__ == "__main__":
    main()
