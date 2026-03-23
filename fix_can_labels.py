"""
fix_can_labels.py — 修正 CAN 数据集的 R/T 标签

问题:
    原始 CSV 最后一列是 R（正常帧）和 T（注入的攻击帧）
    之前的 prepare_data.py 把整列都覆盖成攻击标签
    导致攻击 CSV 里大量正常帧（R）被错误标记为攻击，模型无法区分

修正:
    R → Normal
    T → 对应攻击类型（DoS / Fuzzy / RPM_Spoofing / Gear_Spoofing）

用法:
    python fix_can_labels.py

输出:
    data/raw/car_hacking_rt.csv   （修正后的完整数据集）
"""
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

COL = [
    "Timestamp", "CAN_ID", "DLC",
    "DATA0", "DATA1", "DATA2", "DATA3",
    "DATA4", "DATA5", "DATA6", "DATA7",
    "RT",   # 最后一列: R=正常帧  T=注入攻击帧
]

ATTACK_FILES = {
    RAW / "DoS_dataset.csv":   "DoS",
    RAW / "Fuzzy_dataset.csv": "Fuzzy",
    RAW / "RPM_dataset.csv":   "RPM_Spoofing",
    RAW / "gear_dataset.csv":  "Gear_Spoofing",
}

print("=" * 56)
print("修正 CAN 数据集 R/T 标签")
print("=" * 56)

dfs = []
for path, attack_label in ATTACK_FILES.items():
    if not path.exists():
        print(f"  [SKIP] {path.name} 不存在")
        continue

    df = pd.read_csv(path, header=None, names=COL)

    # R = 正常帧，T = 注入的攻击帧
    df["Flag"] = df["RT"].apply(
        lambda x: attack_label if str(x).strip() == "T" else "Normal"
    )
    df = df.drop(columns=["RT"])

    normal_cnt = (df["Flag"] == "Normal").sum()
    attack_cnt = (df["Flag"] == attack_label).sum()
    print(f"\n  {path.name}")
    print(f"    Normal (R帧): {normal_cnt:>10,}")
    print(f"    {attack_label:12s} (T帧): {attack_cnt:>10,}")
    dfs.append(df)

print("\n" + "=" * 56)
combined = pd.concat(dfs, ignore_index=True)
print(f"\n合并后标签分布:")
for label, cnt in combined["Flag"].value_counts().items():
    print(f"  {label:15s}  {cnt:>10,}")

out = RAW / "car_hacking_rt.csv"
combined.to_csv(out, index=False)
print(f"\n✓ 保存 → {out}  ({len(combined):,} 行)")
print()
print("下一步，清缓存并重新跑 EXP1:")
print()
print("  rm /workspaces/cache/data/can_*.pkl")
print("  rm /workspaces/cache/results/exp1_baseline_can.json")
print("  rm /workspaces/cache/pipeline_state.json")
print()
print("  export RAW_CAN_PATH=data/raw/car_hacking_rt.csv")
print("  export RAW_CIC_PATH=data/raw/cicids2017.csv")
print("  export CACHE_DIR=/workspaces/cache")
print("  python pipeline.py --stage 1 --force")
