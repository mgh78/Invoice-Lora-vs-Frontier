"""Build the README charts from the prediction files.

Usage (from the repo root):
    python scripts/make_charts.py --test data/test.jsonl --preds results --out assets
"""
import argparse, importlib.util, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("score", os.path.join(HERE, "score_invoices.py"))
score = importlib.util.module_from_spec(spec); spec.loader.exec_module(score)

# ---- style: validated categorical slots 1-3, recessive grid, text in ink colours ----
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
for f in font_manager.findSystemFonts():
    if "Inter" in os.path.basename(f):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "semibold", "axes.titlesize": 13, "axes.titlecolor": INK, "axes.titlelocation": "left",
})

CONFIGS = {  # file stem -> (label, colour, schema-constrained?)
    "preds_base": ("Llama 3.2 1B (prompt only)", AQUA, False),
    "preds_base_guided": ("Llama 3.2 1B + schema", AQUA, True),
    "preds_adapter": ("1B + LoRA adapter", BLUE, False),
    "preds_adapter_guided": ("1B + LoRA adapter + schema", BLUE, True),
    "preds_luna": ("GPT-5.6 Luna (keys only)", ORANGE, False),
    "preds_luna_guided": ("GPT-5.6 Luna + schema", ORANGE, True),
}
FAIR = ["preds_base_guided", "preds_adapter_guided", "preds_luna_guided"]
SHORT = {"preds_base_guided": "Base 1B + schema", "preds_adapter_guided": "1B + LoRA + schema",
         "preds_luna_guided": "GPT-5.6 Luna + schema"}

# measured during the runs (not stored per row)
THROUGHPUT = {"preds_base": 21503, "preds_base_guided": 24341, "preds_adapter": 36871,
              "preds_adapter_guided": 36616, "preds_luna": 3887, "preds_luna_guided": 3008}
GPU_PER_HOUR = 0.73          # RunPod RTX PRO 4500
LUNA_IN, LUNA_OUT = 0.20, 1.20  # $ per 1M tokens
STARTUP_MIN_PER_BATCH = 5    # daily batch: boot + model load


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def headline(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, (metric, title) in zip(axes, [("schema_valid", "Schema-valid outputs"),
                                          ("MEAN_FIELD_ACC", "Mean field accuracy")]):
        y = np.arange(len(FAIR))[::-1]
        vals = [results[k][metric] * 100 for k in FAIR]
        ax.barh(y, vals, height=0.56, color=[CONFIGS[k][1] for k in FAIR], edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v + 1.5, yi, f"{v:.1f}%", va="center", color=INK, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels([f"{t}%" for t in [0, 25, 50, 75, 100]])
        ax.set_yticks(y); ax.set_yticklabels([SHORT[k] for k in FAIR], color=INK)
        ax.grid(axis="y", visible=False); ax.tick_params(length=0)
        ax.set_title(title)
    fig.text(0.0, -0.04, "381 held-out invoices. 'Schema' = JSON-schema-constrained decoding "
             "(vLLM guided decoding / OpenAI response_format, non-strict).", color=MUTED, fontsize=9)
    save(fig, out, "headline.png")


def fields(results, out):
    flds = list(score.FIELDS)
    order = sorted(flds, key=lambda f: results["preds_luna_guided"][f] - results["preds_adapter_guided"][f])
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(order))
    for k in FAIR:
        xs = [results[k][f] * 100 for f in order]
        ax.plot(xs, y, "o", ms=9, color=CONFIGS[k][1], label=SHORT[k],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    for yi, f in enumerate(order):  # thin connector: adapter -> Luna gap
        a, b = results["preds_adapter_guided"][f] * 100, results["preds_luna_guided"][f] * 100
        ax.plot([a, b], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.set_yticks(y); ax.set_yticklabels(order, color=INK, family="DejaVu Sans Mono", fontsize=10)
    ax.set_xlim(0, 100); ax.set_xticks(range(0, 101, 20)); ax.set_xticklabels([f"{t}%" for t in range(0, 101, 20)])
    ax.grid(axis="y", visible=False); ax.tick_params(length=0)
    ax.set_title("Accuracy by field  (sorted by the frontier model's lead over the adapter)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, labelcolor=INK)
    save(fig, out, "field_accuracy.png")


def heatmap(results, out):
    metrics = ["schema_valid", "n_items_match"] + list(score.FIELDS) + ["MEAN_FIELD_ACC"]
    labels = {"schema_valid": "schema valid", "n_items_match": "line-item count", "MEAN_FIELD_ACC": "MEAN"}
    keys = list(CONFIGS)
    M = np.array([[results[k][m] * 100 for k in keys] for m in metrics])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#9cc1ef", BLUE, "#123f78"])
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10,
                    color="white" if v > 55 else INK, fontweight="bold" if metrics[i] == "MEAN_FIELD_ACC" else "normal")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([CONFIGS[k][0].replace(" + ", "\n+ ").replace(" (", "\n(") for k in keys],
                                                        fontsize=9.5, color=INK)
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels([labels.get(m, m) for m in metrics],
                                                           family="DejaVu Sans Mono", fontsize=10, color=INK)
    ax.xaxis.tick_top(); ax.tick_params(length=0); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(keys)), minor=True); ax.set_yticks(np.arange(-.5, len(metrics)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2); ax.tick_params(which="minor", length=0)
    ax.axhline(len(metrics) - 1.5, color=SURFACE, lw=5)
    fig.text(0.0, 0.02, "All six configurations, % of test invoices (fields: % of invoices where the gold label has a value).",
             color=MUTED, fontsize=9)
    save(fig, out, "all_metrics_heatmap.png")


def cost(tokens, out):
    luna_in, luna_out = tokens["preds_luna_guided"]
    luna_per_inv = (luna_in * LUNA_IN + luna_out * LUNA_OUT) / 1e6
    thr = THROUGHPUT["preds_adapter_guided"]
    V = np.logspace(3, 7, 300)
    luna = V * luna_per_inv
    batch = (V / thr + 30 * STARTUP_MIN_PER_BATCH / 60) * GPU_PER_HOUR
    always = np.maximum(720, V / thr) * GPU_PER_HOUR
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(V, luna, color=ORANGE, lw=2, label="GPT-5.6 Luna API")
    ax.plot(V, batch, color=BLUE, lw=2, label="LoRA adapter, daily batch on rented GPU")
    ax.plot(V, always, color=BLUE, lw=2, ls=(0, (5, 3)), label="LoRA adapter, always-on GPU")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvline(40000, color=INK2, lw=1, ls=":")
    ax.text(40000 * 1.08, 1.6, "brokerage today:\n40k invoices / month", color=INK2, fontsize=9.5, va="center")
    be = 720 * GPU_PER_HOUR / luna_per_inv
    ax.plot([be], [720 * GPU_PER_HOUR], "o", ms=8, color=INK, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    ax.annotate(f"always-on breaks even\nat ~{be/1000:,.0f}k / month", (be, 720 * GPU_PER_HOUR), xytext=(be * 1.6, 120),
                color=INK, fontsize=9.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    y_l = 40000 * luna_per_inv; y_b = (40000 / thr + 30 * STARTUP_MIN_PER_BATCH / 60) * GPU_PER_HOUR
    for yv, col, off, ha in [(y_l, ORANGE, (10, -16), "left"), (y_b, BLUE, (-10, 10), "right"),
                             (720 * GPU_PER_HOUR, BLUE, (-10, 8), "right")]:
        ax.plot([40000], [yv], "o", ms=8, color=col, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.annotate(f"${yv:,.1f}" if yv < 10 else f"${yv:,.0f}", (40000, yv), xytext=off, textcoords="offset points",
                    ha=ha, va="center", color=INK, fontsize=10.5, fontweight="bold")
    ax.set_xlim(1e3, 1e7); ax.set_ylim(0.5, 2e4)
    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7]); ax.set_xticklabels(["1k", "10k", "100k", "1M", "10M"])
    ax.set_yticks([1, 10, 100, 1000, 10000]); ax.set_yticklabels(["$1", "$10", "$100", "$1k", "$10k"])
    ax.set_xlabel("invoices per month"); ax.set_ylabel("cost per month")
    ax.set_title("Monthly inference cost vs volume")
    ax.legend(loc="upper left", frameon=False, labelcolor=INK)
    fig.text(0.0, -0.04, f"GPU ${GPU_PER_HOUR}/h (RTX PRO 4500), {thr:,} invoices/h measured; daily batch adds "
             f"{STARTUP_MIN_PER_BATCH} min start-up. Luna ${LUNA_IN}/${LUNA_OUT} per 1M tokens, {luna_in:.0f} in / "
             f"{luna_out:.0f} out tokens per invoice measured. Excludes engineering time.", color=MUTED, fontsize=8.5)
    save(fig, out, "cost_vs_volume.png")
    return luna_per_inv, be


def latency(lat, out):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    y = np.arange(len(FAIR))[::-1]
    for yi, k in zip(y, FAIR):
        p50, p95 = np.percentile(lat[k], 50), np.percentile(lat[k], 95)
        col = CONFIGS[k][1]
        ax.plot([p50, p95], [yi, yi], color=col, lw=3, alpha=0.35, solid_capstyle="round")
        ax.plot([p50], [yi], "o", ms=9, color=col, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.plot([p95], [yi], "D", ms=8, color=col, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(p50, yi + 0.27, f"p50 {p50:.2f}s", ha="center", color=INK2, fontsize=9)
        ax.text(p95, yi + 0.27, f"p95 {p95:.2f}s", ha="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([SHORT[k] for k in FAIR], color=INK)
    ax.set_xlim(0, 9); ax.set_ylim(-0.6, len(FAIR) - 0.2)
    ax.set_xlabel("seconds per invoice"); ax.grid(axis="y", visible=False); ax.tick_params(length=0)
    ax.set_title("Latency per invoice  (● p50   ◆ p95)")
    fig.text(0.0, -0.08, "vLLM runs at 16 concurrent requests on one RTX PRO 4500; the OpenAI API at 4 (rate limits).",
             color=MUTED, fontsize=9)
    save(fig, out, "latency.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True); ap.add_argument("--preds", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    gold = score.load(a.test)
    preds = {k: score.load(os.path.join(a.preds, k + ".jsonl")) for k in CONFIGS}
    results = {k: score.score(gold, p) for k, p in preds.items()}
    lat = {k: np.array([r["latency_s"] for r in p]) for k, p in preds.items()}
    tokens = {k: (np.mean([(r.get("usage") or {}).get("prompt_tokens", 0) for r in p]),
                  np.mean([(r.get("usage") or {}).get("completion_tokens", 0) for r in p])) for k, p in preds.items()}
    headline(results, a.out); fields(results, a.out); heatmap(results, a.out); latency(lat, a.out)
    per_inv, be = cost(tokens, a.out)
    json.dump({"metrics": results, "tokens": tokens, "luna_cost_per_invoice": per_inv, "break_even_always_on": be},
              open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print("charts written to", a.out, "| Luna $/1k =", round(per_inv * 1000, 3), "| break-even =", round(be))


if __name__ == "__main__":
    main()
