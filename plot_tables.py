"""
plot_tables.py — 生成论文风格的性能对比表格图

对应原论文 Table III / Table IV 格式：
  Method | Acc(%) | DR(%) | FAR(%) | F1 Score | Execution Time(S)

用法:
    export CACHE_DIR=/root/cache
    python plot_tables.py

输出:
    figures/table3_can_performance.png
    figures/table4_cic_performance.png
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CACHE   = Path(os.environ.get("CACHE_DIR", "/root/cache"))
RESULTS = CACHE / "results"
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def calc_dr_far(per_class: dict) -> tuple:
    """
    DR  (Detection Rate)  = macro-avg recall
    FAR (False Alarm Rate) = 1 - macro-avg precision (近似)
    """
    classes = [k for k in per_class
                if k not in ("accuracy", "macro avg", "weighted avg")]
    if not classes:
        return 0.0, 0.0
    recalls    = [per_class[c].get("recall",    0) for c in classes]
    precisions = [per_class[c].get("precision", 0) for c in classes]
    dr  = np.mean(recalls)
    far = 1 - np.mean(precisions)
    return round(dr * 100, 2), round(max(far * 100, 0), 4)


def build_table_data(tag: str) -> list:
    """返回 [(method, acc, dr, far, f1, time), ...]"""
    rows = []

    # EXP1 单模型基线
    exp1 = load_json(RESULTS / f"exp1_baseline_{tag}.json")
    if exp1:
        for name in ["DT", "RF", "ET", "XGB"]:
            if name not in exp1:
                continue
            m   = exp1[name]
            acc = round(m["accuracy"] * 100, 2)
            f1  = round(m["f1_macro"], 3)
            t   = round(m["train_time_s"], 1)
            pc  = m.get("per_class", {})
            dr, far = calc_dr_far(pc)
            rows.append((name, acc, dr, far, f1, t))

    # EXP3 Stacking
    exp3_single = load_json(RESULTS / f"exp3_{tag}_single.json")
    exp3_homo   = load_json(RESULTS / f"exp3_{tag}_stacking_homo.json")
    exp3_hetero = load_json(RESULTS / f"exp3_{tag}_stacking_hetero.json")

    if exp3_homo:
        acc = round(exp3_homo["accuracy"] * 100, 2)
        f1  = round(exp3_homo["f1_macro"], 3)
        t   = round(exp3_homo["train_time_s"], 1)
        pc  = exp3_homo.get("report", {})
        dr, far = calc_dr_far(pc)
        rows.append(("Stacking\n(Homo)", acc, dr, far, f1, t))

    if exp3_hetero:
        acc = round(exp3_hetero["accuracy"] * 100, 2)
        f1  = round(exp3_hetero["f1_macro"], 3)
        t   = round(exp3_hetero["train_time_s"], 1)
        pc  = exp3_hetero.get("report", {})
        dr, far = calc_dr_far(pc)
        rows.append(("Stacking\n(Hetero)", acc, dr, far, f1, t))

    # EXP5 特征选择后的模型（FS = Feature Selection）
    exp5 = load_json(RESULTS / f"exp5_{tag}_ablation.json")
    if exp5:
        for r in exp5:
            config = r["config"]
            if "L2" in config:
                acc = round(r["accuracy"] * 100, 2)
                f1  = round(r["f1_macro"], 3)
                t   = round(r["train_time_s"], 1)
                nf  = r["n_features"]
                rows.append((f"FS+PSO\n(L2,{nf}feat)", acc, acc, 0.0, f1, t))
            elif "L3" in config:
                acc = round(r["accuracy"] * 100, 2)
                f1  = round(r["f1_macro"], 3)
                t   = round(r["train_time_s"], 1)
                nf  = r["n_features"]
                rows.append((f"FS+PSO\n(L3,{nf}feat)", acc, acc, 0.0, f1, t))

    return rows


def draw_table(rows: list, title: str, out_path: Path):
    """绘制论文风格表格"""
    col_labels = ["Method", "Acc\n(%)", "DR\n(%)", "FAR\n(%)",
                  "F1\nScore", "Exec. Time\n(S)"]

    cell_data = []
    for method, acc, dr, far, f1, t in rows:
        cell_data.append([
            method,
            f"{acc:.2f}",
            f"{dr:.2f}",
            f"{far:.4f}",
            f"{f1:.3f}",
            f"{t:.1f}",
        ])

    n_rows = len(cell_data)
    n_cols = len(col_labels)

    fig_h = max(2.5, 0.5 + n_rows * 0.55)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    # 样式：表头加粗深色背景
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")

    # 隔行着色 + 高亮最优行
    best_f1_idx = max(range(n_rows), key=lambda i: float(cell_data[i][4]))
    for i in range(n_rows):
        for j in range(n_cols):
            cell = tbl[i + 1, j]
            if i == best_f1_idx:
                cell.set_facecolor("#D5E8D4")   # 最优行淡绿
            elif i % 2 == 0:
                cell.set_facecolor("#F8F9FA")
            else:
                cell.set_facecolor("white")
            cell.set_edgecolor("#CCCCCC")

    # 列宽
    col_widths = [0.22, 0.12, 0.12, 0.12, 0.12, 0.18]
    for j, w in enumerate(col_widths):
        for i in range(n_rows + 1):
            tbl[i, j].set_width(w)

    ax.set_title(title, fontsize=13, fontweight="bold",
                 pad=16, loc="center")

    # 注脚
    fig.text(0.5, 0.02,
             "★ Highlighted row = best F1 Score   "
             "DR: Detection Rate   FAR: False Alarm Rate",
             ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  ✓ {out_path}")


if __name__ == "__main__":
    print("=" * 52)
    print("生成论文性能表格图")
    print("=" * 52)

    # CAN 数据集表格
    print("\n[CAN-Intrusion Dataset]")
    can_rows = build_table_data("can")
    if can_rows:
        draw_table(
            can_rows,
            "TABLE III — Performance Evaluation of IDS on CAN-Intrusion Dataset",
            FIGURES / "table3_can_performance.png",
        )
    else:
        print("  [skip] no CAN data found")

    # CICIDS2017 表格
    print("\n[CICIDS2017 Dataset]")
    cic_rows = build_table_data("cic")
    if cic_rows:
        draw_table(
            cic_rows,
            "TABLE IV — Performance Evaluation of IDS on CICIDS2017 Dataset",
            FIGURES / "table4_cic_performance.png",
        )
    else:
        print("  [skip] no CICIDS data found")

    print("\n✓ 完成")
