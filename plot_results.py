"""
plot_results.py — 生成论文所需全部图表（修正版）

用法:
    export CACHE_DIR=/root/cache
    python plot_results.py
"""
import json
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CACHE   = Path(os.environ.get("CACHE_DIR", "/root/cache"))
RESULTS = CACHE / "results"
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":     150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
          "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]

CAN_LABEL_MAP = {
    "0": "DoS", "1": "Fuzzy", "2": "Gear_Spoofing",
    "3": "Normal", "4": "RPM_Spoofing",
}
CIC_LABEL_MAP = {
    "0": "BENIGN", "1": "Botnet", "2": "BruteForce",
    "3": "DoS", "4": "Infiltration", "5": "PortScan", "6": "WebAttack",
}

def load_json(path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)

def save_fig(name):
    out = FIGURES / name
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓ {out}")

# ══════════════════════════════════════════════════════════════════════════
# EXP1 — 基线对比
# ══════════════════════════════════════════════════════════════════════════
def plot_baseline(tag, fig_acc, fig_perclass):
    data = load_json(RESULTS / f"exp1_baseline_{tag}.json")
    if not data:
        print(f"  [skip] exp1_baseline_{tag}.json not found")
        return

    models = list(data.keys())
    accs   = [data[m]["accuracy"]  * 100 for m in models]
    f1s    = [data[m]["f1_macro"]  * 100 for m in models]
    times  = [data[m]["train_time_s"]    for m in models]

    x = np.arange(len(models))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy (%)", color=COLORS[0], alpha=0.85)
    b2 = ax.bar(x + w/2, f1s,  w, label="F1-macro (%)", color=COLORS[1], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylim(0, 115); ax.set_ylabel("Score (%)")
    title = "CAN-Intrusion Dataset" if tag == "can" else "CICIDS2017 Dataset"
    ax.set_title(f"{title} — Accuracy & F1-macro")
    ax.legend()
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    ax2 = axes[1]
    bars = ax2.bar(models, times, color=COLORS[2], alpha=0.85)
    ax2.set_ylabel("Training Time (s)")
    ax2.set_title(f"{title} — Training Time")
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1,
                 f"{h:.0f}s", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_fig(fig_acc)

    # per-class recall & precision（修正：用真实类名）
    best = max(models, key=lambda m: data[m]["f1_macro"])
    pc   = data[best].get("per_class", {})
    label_map = CAN_LABEL_MAP if tag == "can" else CIC_LABEL_MAP
    classes = [k for k in pc if k not in ("accuracy","macro avg","weighted avg")]
    if not classes:
        return

    # 用真实类名替换数字
    class_names = [label_map.get(str(c), str(c)) for c in classes]
    recalls    = [pc[c].get("recall",    0) for c in classes]
    precisions = [pc[c].get("precision", 0) for c in classes]

    x2 = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x2 - 0.2, recalls,    0.38, label="Recall",    color=COLORS[0], alpha=0.85)
    ax.bar(x2 + 0.2, precisions, 0.38, label="Precision", color=COLORS[1], alpha=0.85)
    ax.set_xticks(x2)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title(f"{title} — Per-class Recall & Precision ({best})")
    ax.legend()
    for i, (r, p) in enumerate(zip(recalls, precisions)):
        ax.text(i - 0.2, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    save_fig(fig_perclass)

# ══════════════════════════════════════════════════════════════════════════
# EXP2 — 数据增强
# ══════════════════════════════════════════════════════════════════════════
def plot_augmentation():
    aug_stats = load_json(RESULTS / "exp2_cic_aug_stats.json")
    rare_data = load_json(RESULTS / "exp2_rare_recall.json")

    before_dist = {
        "BENIGN": 1817055, "DoS": 303798, "PortScan": 127043,
        "BruteForce": 11066, "WebAttack": 1744,
        "Botnet": 1565, "Infiltration": 29,
    }

    if aug_stats and "class_counts" in aug_stats:
        after_dist = aug_stats["class_counts"]
    else:
        after_dist = before_dist.copy()
        print("  [warn] aug_stats not found, using before dist")

    labels = list(before_dist.keys())
    x      = np.arange(len(labels))
    bvals  = [before_dist.get(l, 0) for l in labels]
    avals  = [after_dist.get(l, 0)  for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, scale in zip(axes, ["linear", "log"]):
        ax.bar(x - 0.2, [max(v,1) for v in bvals], 0.38,
               label="Before SMOTE-Tomek", color=COLORS[0], alpha=0.85)
        ax.bar(x + 0.2, [max(v,1) for v in avals], 0.38,
               label="After SMOTE-Tomek",  color=COLORS[1], alpha=0.85)
        if scale == "log":
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Sample Count" + (" (log scale)" if scale == "log" else ""))
        ax.set_title(f"CICIDS2017 — Class Distribution ({scale} scale)")
        ax.legend()
        if scale == "linear":
            ax.annotate("BENIGN略减: Tomek Links\n删除边界噪声样本",
                        xy=(0, avals[0]), xytext=(1.5, avals[0]*0.9),
                        fontsize=8, color="gray",
                        arrowprops=dict(arrowstyle="->", color="gray"))

    plt.tight_layout()
    save_fig("fig5_aug_distribution.png")

    # 稀有类召回率对比（修正：正确读取before/after键）
    rare_classes = ["WebAttack", "Botnet", "Infiltration"]

    if rare_data:
        # 支持两种格式
        if "before" in rare_data:
            before_recalls = rare_data["before"]
            after_recalls  = rare_data["after"]
        else:
            before_recalls = {k: rare_data.get(k, 0) for k in rare_classes}
            after_recalls  = {k: rare_data.get(k + "_after", 0) for k in rare_classes}
    else:
        before_recalls = {c: 0 for c in rare_classes}
        after_recalls  = {c: 0 for c in rare_classes}
        print("  [warn] exp2_rare_recall.json not found")

    x3 = np.arange(len(rare_classes))
    fig, ax = plt.subplots(figsize=(8, 5))
    bv = [before_recalls.get(c, 0) for c in rare_classes]
    av = [after_recalls.get(c, 0)  for c in rare_classes]
    b1 = ax.bar(x3 - 0.2, bv, 0.38,
                label="Before SMOTE-Tomek", color=COLORS[3], alpha=0.85)
    b2 = ax.bar(x3 + 0.2, av, 0.38,
                label="After SMOTE-Tomek",  color=COLORS[2], alpha=0.85)
    ax.set_xticks(x3); ax.set_xticklabels(rare_classes)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Recall")
    ax.set_title("Rare Attack Class Recall — Before vs After SMOTE-Tomek")
    ax.legend()
    for i, (b, a) in enumerate(zip(bv, av)):
        ax.text(i - 0.2, b + 0.02, f"{b:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.2, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    save_fig("fig6_rare_recall.png")

# ══════════════════════════════════════════════════════════════════════════
# EXP3 — Stacking 对比
# ══════════════════════════════════════════════════════════════════════════
def plot_stacking():
    for tag in ["can", "cic"]:
        single = load_json(RESULTS / f"exp3_{tag}_single.json")
        hetero = load_json(RESULTS / f"exp3_{tag}_stacking_hetero.json")
        homo   = load_json(RESULTS / f"exp3_{tag}_stacking_homo.json")

        if not single:
            print(f"  [skip] EXP3 {tag} not found")
            continue

        models   = list(single.keys())
        accs     = [single[m]["accuracy"] * 100 for m in models]
        f1s      = [single[m]["f1_macro"]  * 100 for m in models]
        labels   = models.copy()
        acc_vals = accs.copy()
        f1_vals  = f1s.copy()

        if homo:
            labels.append("Stacking\n(Homo)")
            acc_vals.append(homo["accuracy"] * 100)
            f1_vals.append(homo["f1_macro"]  * 100)
        if hetero:
            labels.append("Stacking\n(Hetero)")
            acc_vals.append(hetero["accuracy"] * 100)
            f1_vals.append(hetero["f1_macro"]  * 100)

        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(12, 5))
        b1 = ax.bar(x - w/2, acc_vals, w,
                    label="Accuracy (%)", color=COLORS[0], alpha=0.85)
        b2 = ax.bar(x + w/2, f1_vals,  w,
                    label="F1-macro (%)", color=COLORS[1], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(0, 115); ax.set_ylabel("Score (%)")
        title = "CAN-Intrusion" if tag == "can" else "CICIDS2017"
        ax.set_title(f"{title} — Single Models vs Stacking")
        ax.legend()

        # 虚线分隔单模型和stacking
        n_single = len(models)
        ax.axvline(n_single - 0.5, color="gray",
                   linestyle="--", linewidth=1)

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        save_fig(f"fig7_stacking_{tag}.png")

# ══════════════════════════════════════════════════════════════════════════
# EXP4 — PSO vs Grid Search（修正：改标题说明）
# ══════════════════════════════════════════════════════════════════════════
def plot_pso():
    for tag in ["cic", "can"]:
        cmp = load_json(RESULTS / f"exp4_{tag}_comparison.json")
        if not cmp:
            print(f"  [skip] EXP4 {tag} not found")
            continue

        gs_t  = cmp["grid_search"]["time_s"]
        pso_t = cmp["pso"]["time_s"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        title = "CAN-Intrusion" if tag == "can" else "CICIDS2017"

        # 时间对比
        ax = axes[0]
        bars = ax.bar(["Grid Search\n(3 params)", "PSO (ours)\n(18 params)"],
                      [gs_t, pso_t],
                      color=[COLORS[3], COLORS[2]], alpha=0.85, width=0.4)
        ax.set_ylabel("Tuning Time (s)")
        ax.set_title(f"{title} — PSO vs Grid Search Tuning Time\n"
                     f"(PSO searches 18-dim space vs Grid Search 3-dim)")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 50,
                    f"{h:.0f}s\n({h/3600:.1f}h)",
                    ha="center", va="bottom", fontsize=9)

        # 收敛曲线（若有）
        conv_png = RESULTS / f"exp4_{tag}_convergence.png"
        ax2 = axes[1]
        if conv_png.exists():
            img = plt.imread(str(conv_png))
            ax2.imshow(img)
            ax2.axis("off")
            ax2.set_title(f"{title} — PSO Convergence Curve")
        else:
            ax2.text(0.5, 0.5, "Convergence plot\nnot available",
                     ha="center", va="center", transform=ax2.transAxes,
                     fontsize=12, color="gray")
            ax2.set_title(f"{title} — PSO Convergence")

        plt.tight_layout()
        save_fig(f"fig8_pso_time_{tag}.png")

# ══════════════════════════════════════════════════════════════════════════
# EXP5 — 特征曲线 & 消融实验（修正：标签截断）
# ══════════════════════════════════════════════════════════════════════════
def plot_deploy():
    for tag in ["cic", "can"]:
        title = "CAN-Intrusion" if tag == "can" else "CICIDS2017"

        # 特征曲线
        curve = load_json(RESULTS / f"exp5_{tag}_feature_curve.json")
        if curve:
            xs = [r["n_features"] for r in curve]
            ys = [r["accuracy"] * 100 for r in curve]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(xs, ys, "o-", linewidth=2, markersize=6,
                    color=COLORS[0])
            ax.set_xlabel("Number of Features")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"{title} — Feature Count vs Accuracy")
            ax.grid(True, alpha=0.3)

            # 推荐点（L2，90%重要性）
            mid = len(xs) // 2
            rec_x = xs[mid]
            rec_y = ys[mid]
            ax.axvline(rec_x, color="red", linestyle="--",
                       linewidth=1.5, label=f"Recommended L2 (n={rec_x})")
            ax.annotate(f"{rec_y:.2f}%",
                        xy=(rec_x, rec_y),
                        xytext=(rec_x + 1, rec_y - 0.3),
                        fontsize=9, color="red")
            ax.legend()
            plt.tight_layout()
            save_fig(f"fig9_feature_curve_{tag}.png")
        else:
            print(f"  [skip] EXP5 feature curve {tag} not found")

        # 消融实验（修正：加大左边距，避免标签截断）
        ablation = load_json(RESULTS / f"exp5_{tag}_ablation.json")
        if ablation:
            # 简化标签
            label_map = {
                "1. Baseline (default params)":           "1. Baseline",
                "2. + PSO tuning (full features)":        "2. +PSO tuning",
                "3. + L3 feature selection (95%)":        "3. +L3 (95%, features)",
                "4. + L2 feature selection (90%) *":      "4. +L2 (90%, features) ★",
                "5. + L1 feature selection (70%, fast)":  "5. +L1 (70%, features)",
            }
            configs = []
            for r in ablation:
                raw = r["config"]
                configs.append(label_map.get(raw, raw))
            accs = [r["accuracy"] * 100 for r in ablation]
            f1s  = [r["f1_macro"]  * 100 for r in ablation]
            n_feats = [r["n_features"]   for r in ablation]

            fig, ax = plt.subplots(figsize=(12, 5))
            y = np.arange(len(configs))
            b1 = ax.barh(y - 0.2, accs, 0.35,
                         label="Accuracy (%)", color=COLORS[0], alpha=0.85)
            b2 = ax.barh(y + 0.2, f1s,  0.35,
                         label="F1-macro (%)", color=COLORS[1], alpha=0.85)
            ax.set_yticks(y)
            ax.set_yticklabels(configs)
            ax.set_xlabel("Score (%)")
            ax.set_title(f"{title} — Ablation Study")
            ax.set_xlim(min(min(accs), min(f1s)) - 3, 102)
            ax.legend(loc="lower right")

            for bar, val in zip(b1, accs):
                ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                        f"{val:.2f}%", va="center", fontsize=8)
            for bar, val in zip(b2, f1s):
                ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                        f"{val:.2f}%", va="center", fontsize=8)

            plt.tight_layout()
            save_fig(f"fig10_ablation_{tag}.png")
        else:
            print(f"  [skip] EXP5 ablation {tag} not found")

# ══════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 52)
    print(f"生成论文图表 → {FIGURES}/")
    print("=" * 52)

    print("\n[EXP1] 基线对比图")
    plot_baseline("can", "fig1_can_baseline.png", "fig3_can_perclass.png")
    plot_baseline("cic", "fig2_cic_baseline.png", "fig4_cic_perclass.png")

    print("\n[EXP2] 数据增强分布图")
    plot_augmentation()

    print("\n[EXP3] Stacking 对比图")
    plot_stacking()

    print("\n[EXP4] PSO 调参图")
    plot_pso()

    print("\n[EXP5] 轻量化部署图")
    plot_deploy()

    print(f"\n✓ 全部图表已生成到 {FIGURES}/ 目录")
    for f in sorted(FIGURES.iterdir()):
        size = f.stat().st_size // 1024
        print(f"  {f.name:45s}  {size:4d} KB")
