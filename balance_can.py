"""
balance_can.py — 对 CAN 数据集做欠采样平衡

问题: Normal 只有 98 万行，攻击类各有 300~460 万行
      严重不平衡导致模型 F1 偏低
解决: 每类最多取 TARGET 条，生成平衡版本

用法:
    python balance_can.py

输出:
    data/raw/car_hacking_balanced.csv
"""
from pathlib import Path
import pandas as pd

RAW     = Path("data/raw/car_hacking.csv")
OUT     = Path("data/raw/car_hacking_balanced.csv")
TARGET  = 800_000   # 每类最多取 80 万条

print("=" * 52)
print("CAN 数据集欠采样平衡")
print("=" * 52)

print(f"\n读取 {RAW} ...")
df = pd.read_csv(RAW)
print(f"原始行数: {len(df):,}")
print(f"\n原始标签分布:")
for label, cnt in df["Flag"].value_counts().items():
    print(f"  {label:15s}  {cnt:>10,}")

# 每类采样
dfs = []
for label, group in df.groupby("Flag"):
    n = min(len(group), TARGET)
    sampled = group.sample(n=n, random_state=42)
    dfs.append(sampled)
    print(f"  {label:15s}  {len(group):>10,}  →  采样 {n:>8,}")

balanced = (pd.concat(dfs, ignore_index=True)
              .sample(frac=1, random_state=42)   # shuffle
              .reset_index(drop=True))

balanced.to_csv(OUT, index=False)

print(f"\n平衡后标签分布:")
for label, cnt in balanced["Flag"].value_counts().items():
    print(f"  {label:15s}  {cnt:>10,}")

print(f"\n总行数: {len(balanced):,}")
print(f"✓ 保存 → {OUT}")
print()
print("下一步，清缓存并重新跑 EXP1:")
print()
print("  rm -rf cache/")
print("  export RAW_CAN_PATH=data/raw/car_hacking_balanced.csv")
print("  export RAW_CIC_PATH=data/raw/cicids2017.csv")
print("  export CACHE_DIR=cache")
print("  python pipeline.py --stage 1")
