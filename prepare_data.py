"""
prepare_data.py — 合并原始数据集

用法:
    python prepare_data.py

输出:
    data/raw/car_hacking.csv
    data/raw/cicids2017.csv
"""
import glob
import re
import sys
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")

# ─────────────────────────────────────────────────────────────────────────────
# CAN-Intrusion
# ─────────────────────────────────────────────────────────────────────────────
def merge_can():
    print("=" * 56)
    print("合并 CAN-Intrusion 数据集")
    print("=" * 56)

    COL = [
        "Timestamp", "CAN_ID", "DLC",
        "DATA0", "DATA1", "DATA2", "DATA3",
        "DATA4", "DATA5", "DATA6", "DATA7",
        "Flag",
    ]

    # ── 四个攻击 CSV（无表头） ────────────────────────────────────────────
    attack_files = {
        RAW / "DoS_dataset.csv":   "DoS",
        RAW / "Fuzzy_dataset.csv": "Fuzzy",
        RAW / "RPM_dataset.csv":   "RPM_Spoofing",
        RAW / "gear_dataset.csv":  "Gear_Spoofing",
    }

    dfs = []
    for path, label in attack_files.items():
        if not path.exists():
            print(f"  [SKIP] {path} not found")
            continue
        df = pd.read_csv(path, header=None, names=COL)
        df["Flag"] = label          # 覆盖最后一列（原始是 R/T）
        print(f"  {label:15s}  {len(df):>10,} rows  ← {path.name}")
        dfs.append(df)

    # ── normal_run_data.txt（固定宽度文本）───────────────────────────────
    normal_path = RAW / "normal_run_data.txt"
    if normal_path.exists():
        rows = []
        pat  = re.compile(
            r"Timestamp:\s+([\d.]+)\s+ID:\s+([0-9a-fA-F]+)"
            r"\s+\S+\s+DLC:\s+(\d+)\s+(.*)"
        )
        with open(normal_path) as f:
            for line in f:
                m = pat.search(line)
                if not m:
                    continue
                ts, cid, dlc, data_str = m.groups()
                data_bytes = (data_str.strip().split() + ["00"] * 8)[:8]
                rows.append([float(ts), cid, int(dlc)] + data_bytes + ["Normal"])
        normal_df = pd.DataFrame(rows, columns=COL)
        print(f"  {'Normal':15s}  {len(normal_df):>10,} rows  ← normal_run_data.txt")
        dfs.append(normal_df)
    else:
        print(f"  [SKIP] {normal_path} not found")

    if not dfs:
        print("  ERROR: 没有找到任何 CAN 文件，请检查 data/raw/ 目录")
        return False

    out = pd.concat(dfs, ignore_index=True)
    out_path = RAW / "car_hacking.csv"
    out.to_csv(out_path, index=False)

    print(f"\n✓ 输出 → {out_path}  ({len(out):,} 行)")
    print("  标签分布:")
    for label, cnt in out["Flag"].value_counts().items():
        print(f"    {label:15s}  {cnt:>10,}")
    print(f"  列名: {out.columns.tolist()}\n")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CICIDS2017
# ─────────────────────────────────────────────────────────────────────────────
def merge_cicids():
    print("=" * 56)
    print("合并 CICIDS2017 数据集")
    print("=" * 56)

    files = sorted(glob.glob(str(RAW / "*.pcap_ISCX.csv")))
    if not files:
        print("  ERROR: 没有找到 *.pcap_ISCX.csv 文件，请检查 data/raw/ 目录")
        return False

    print(f"  找到 {len(files)} 个文件:")
    dfs = []
    for fpath in files:
        fname = Path(fpath).name
        try:
            df = pd.read_csv(fpath, encoding="cp1252", low_memory=False)
            df.columns = df.columns.str.strip()
            print(f"    {fname:60s}  {len(df):>8,} rows")
            dfs.append(df)
        except Exception as e:
            print(f"    [WARN] {fname} 读取失败: {e}")

    if not dfs:
        print("  ERROR: 所有文件读取失败")
        return False

    combined = pd.concat(dfs, ignore_index=True)
    out_path  = RAW / "cicids2017.csv"
    combined.to_csv(out_path, index=False)

    # 找标签列（可能带空格前缀）
    label_col = " Label" if " Label" in combined.columns else "Label"

    print(f"\n✓ 输出 → {out_path}  ({len(combined):,} 行)")
    print("  标签分布:")
    for label, cnt in combined[label_col].value_counts().items():
        print(f"    {str(label):40s}  {cnt:>8,}")
    print(f"  列数: {combined.shape[1]}\n")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)

    ok_can = merge_can()
    ok_cic = merge_cicids()

    print("=" * 56)
    if ok_can and ok_cic:
        print("✓ 两个数据集合并完成，可以开始实验：")
        print()
        print("  export RAW_CAN_PATH=data/raw/car_hacking.csv")
        print("  export RAW_CIC_PATH=data/raw/cicids2017.csv")
        print("  export CACHE_DIR=cache")
        print("  python src/pipeline.py --stage 1")
    else:
        print("✗ 部分数据集合并失败，请检查上方报错信息")
        sys.exit(1)