"""
fix_split.py — 将 CAN 数据集改为按时间顺序切分

问题:
    原来用随机 split，攻击帧和正常帧随机混合到训练/测试集
    导致 XGB 达到 100%，不符合真实部署场景

修正:
    按时间戳排序后，前 80% 作训练集，后 20% 作测试集
    更符合真实 IDS 部署：用历史数据训练，检测未来流量
    自然会让指标降到 99.9x%，更符合学术预期

用法:
    python fix_split.py
    然后重新跑 EXP1
"""
import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

CACHE   = Path("/workspaces/cache")
RAW_CAN = Path("data/raw/car_hacking_rt.csv")

COL_NAMES = [
    "Timestamp", "CAN_ID", "DLC",
    "DATA0", "DATA1", "DATA2", "DATA3",
    "DATA4", "DATA5", "DATA6", "DATA7",
    "Flag"
]


def save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


print("=" * 56)
print("CAN 数据集 — 时间顺序切分（替代随机 split）")
print("=" * 56)

print(f"\n读取 {RAW_CAN} ...")
chunks = []
for chunk in pd.read_csv(RAW_CAN, chunksize=500_000, low_memory=False):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
del chunks; gc.collect()
print(f"  总行数: {len(df):,}")

# ── 特征处理 ───────────────────────────────────────────────────────────────
df.columns = df.columns.str.strip()

# CAN ID hex → int
df["ID_dec"] = df["CAN_ID"].apply(
    lambda x: int(str(x).strip(), 16) if pd.notna(x) else 0)

# DATA bytes hex → int
data_cols = [c for c in df.columns if c.startswith("DATA")]
for col in data_cols:
    df[col] = df[col].apply(
        lambda x: int(str(x).strip(), 16)
        if pd.notna(x) and str(x).strip() not in ("", "R", "T") else 0)

# ── 时间顺序排序 ───────────────────────────────────────────────────────────
print("\n按时间戳排序 ...")
df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
df = df.sort_values("Timestamp").reset_index(drop=True)

# 时间差分特征（排序后计算，无泄露）
df["ts_diff"] = df["Timestamp"].diff().fillna(0).clip(0, 0.1)

# 窗口特征（排序后计算，无泄露）
window = 10
rolling = df["ts_diff"].rolling(window, min_periods=1)
df["ts_mean"] = rolling.mean()
df["ts_std"]  = rolling.std().fillna(0)

byte_arr = df[data_cols].values.astype(float)
df["data_mean"] = byte_arr.mean(axis=1)
df["data_std"]  = byte_arr.std(axis=1)

# ── 时间顺序切分 80/20 ─────────────────────────────────────────────────────
split_idx = int(len(df) * 0.8)
print(f"\n时间顺序切分:")
print(f"  训练集: 前 {split_idx:,} 行（时间戳前 80%）")
print(f"  测试集: 后 {len(df)-split_idx:,} 行（时间戳后 20%）")

feat_cols = (["ID_dec", "DLC", "ts_diff", "ts_mean", "ts_std",
              "data_mean", "data_std"] + data_cols)

df_train = df.iloc[:split_idx].copy()
df_test  = df.iloc[split_idx:].copy()

print(f"\n训练集标签分布:")
for label, cnt in df_train["Flag"].value_counts().items():
    print(f"  {label:15s}  {cnt:>10,}")

print(f"\n测试集标签分布:")
for label, cnt in df_test["Flag"].value_counts().items():
    print(f"  {label:15s}  {cnt:>10,}")

# ── 归一化（只用训练集 fit）────────────────────────────────────────────────
le = LabelEncoder()
le.fit(df["Flag"].astype(str))
y_train = le.transform(df_train["Flag"].astype(str))
y_test  = le.transform(df_test["Flag"].astype(str))

X_train_raw = df_train[feat_cols].fillna(0).values.astype(np.float32)
X_test_raw  = df_test[feat_cols].fillna(0).values.astype(np.float32)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_test  = scaler.transform(X_test_raw).astype(np.float32)     # 用训练集参数

print(f"\n特征维度: {X_train.shape[1]}")

# ── 保存缓存（覆盖原来的随机 split 结果）──────────────────────────────────
save((X_train, y_train), CACHE / "data/can_train.pkl")
save((X_test,  y_test),  CACHE / "data/can_test.pkl")
save({"le": le, "scaler": scaler, "feat_cols": feat_cols},
     CACHE / "data/can_meta.pkl")

# 删掉旧的基线结果，强制重训
old_result = CACHE / "results/exp1_baseline_can.json"
if old_result.exists():
    old_result.unlink()
    print(f"\n  已删除旧结果: {old_result}")

# 重置 pipeline state 中的 stage1
import json
state_file = CACHE / "pipeline_state.json"
if state_file.exists():
    state = json.loads(state_file.read_text())
    state.pop("1", None)
    state_file.write_text(json.dumps(state, indent=2))
    print(f"  已重置 pipeline state stage1")

print(f"\n✓ 时间顺序切分数据已保存")
print()
print("重新跑 EXP1（CICIDS 会命中缓存跳过）:")
print()
print("  export RAW_CAN_PATH=data/raw/car_hacking_rt.csv")
print("  export RAW_CIC_PATH=data/raw/cicids2017.csv")
print("  export CACHE_DIR=/workspaces/cache")
print("  python pipeline.py --stage 1 --force")
