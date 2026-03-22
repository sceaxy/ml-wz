"""
EXP1: 数据准备、预处理、基线训练

输入 (环境变量):
    RAW_CAN_PATH  — car_hacking.csv 路径
    RAW_CIC_PATH  — cicids2017.csv  路径

输出 (cache/data/):
    can_train.pkl, can_test.pkl, can_meta.pkl
    cic_train.pkl, cic_test.pkl, cic_meta.pkl

输出 (cache/results/):
    exp1_baseline_can.json
    exp1_baseline_cic.json
"""
import gc
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RAW_CAN = Path(os.environ.get("RAW_CAN_PATH", "/data/raw/car_hacking.csv"))
RAW_CIC = Path(os.environ.get("RAW_CIC_PATH", "/data/raw/cicids2017.csv"))

# CICIDS 标签合并（用函数处理，兼容乱码/编码变体）
def _map_cic_label(raw: str) -> str:
    s = str(raw).strip()
    if s == "BENIGN":                        return "BENIGN"
    if s.startswith("DoS") or s == "DDoS" \
            or s == "Heartbleed":            return "DoS"
    if s == "PortScan":                      return "PortScan"
    if "Patator" in s:                       return "BruteForce"
    if s == "Bot":                           return "Botnet"
    if "Web Attack" in s:                    return "WebAttack"
    if s == "Infiltration":                  return "Infiltration"
    return "Other"

# ── pickle helpers ─────────────────────────────────────────────────────────
def _save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)

def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ── metric helper ──────────────────────────────────────────────────────────
def _metrics(y_true, y_pred, t_train: float) -> dict:
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy":     round(accuracy_score(y_true, y_pred), 6),
        "f1_macro":     round(f1_score(y_true, y_pred,
                                       average="macro", zero_division=0), 6),
        "train_time_s": round(t_train, 1),
        "per_class":    {k: {m: round(v, 4) for m, v in vs.items()}
                         for k, vs in report.items()
                         if isinstance(vs, dict)},
    }

# ── CAN preprocessing ──────────────────────────────────────────────────────
def _add_window_features(df: pd.DataFrame, id_col: str, ts_col: str,
                         data_cols: list, window: int = 10) -> pd.DataFrame:
    """
    添加滑动窗口统计特征（对标原论文关键特征工程）:
      - ID 频率：过去 window 条消息中相同 CAN ID 出现次数
      - 时间间隔统计：过去 window 条消息的平均/标准差间隔
      - 数据字节均值：过去 window 条消息 DATA 字节的均值
    """
    # 时间戳差分
    df["ts_diff"] = pd.to_numeric(df[ts_col], errors="coerce").diff().fillna(0).clip(0, 0.1)

    # 滑动窗口：相同 ID 在过去 window 条中的频率
    df["id_freq"] = (df[id_col]
                     .rolling(window, min_periods=1)
                     .apply(lambda x: (x == x.iloc[-1]).sum(), raw=False))

    # 滑动窗口：时间间隔均值和标准差
    rolling_ts = df["ts_diff"].rolling(window, min_periods=1)
    df["ts_mean"] = rolling_ts.mean()
    df["ts_std"]  = rolling_ts.std().fillna(0)

    # 滑动窗口：DATA 字节均值（所有字节的平均值）
    df["data_mean"] = df[data_cols].mean(axis=1)
    df["data_std"]  = df[data_cols].std(axis=1).fillna(0)

    return df

def preprocess_can(logger, data_dir: Path):
    cache_tr = data_dir / "can_train.pkl"
    if cache_tr.exists():
        logger.info("CAN  cache hit — loading")
        return _load(cache_tr), _load(data_dir / "can_test.pkl"), \
               _load(data_dir / "can_meta.pkl")

    logger.info(f"CAN  reading {RAW_CAN} (chunked) …")
    chunks = []
    for chunk in pd.read_csv(RAW_CAN, chunksize=500_000, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()
    logger.info(f"  raw shape: {df.shape}")

    # normalise column names
    df.columns = df.columns.str.strip()
    label_col = "Flag" if "Flag" in df.columns else df.columns[-1]
    logger.info(f"  label col: {label_col}\n{df[label_col].value_counts()}")

    # CAN ID hex → int
    id_col = "CAN_ID" if "CAN_ID" in df.columns else (
             "ID"     if "ID"     in df.columns else df.columns[1])
    if df[id_col].dtype == object:
        df["ID_dec"] = df[id_col].apply(
            lambda x: int(str(x).strip(), 16) if pd.notna(x) else 0)
    else:
        df["ID_dec"] = df[id_col].astype(float)

    # DATA bytes — 只取 DATA0~DATA7
    data_cols = [c for c in df.columns
                 if c.startswith("DATA") and c != label_col]
    for col in data_cols:
        df[col] = df[col].apply(
            lambda x: int(str(x).strip(), 16)
            if pd.notna(x) and str(x).strip() not in ("", "R", "T")
            else 0)

    # DLC
    dlc_col = "DLC" if "DLC" in df.columns else None

    # 时间戳列
    ts_col = "Timestamp" if "Timestamp" in df.columns else df.columns[0]

    # 窗口统计特征
    logger.info("  adding window features (id_freq, ts_mean, ts_std, data_mean, data_std) …")
    df = _add_window_features(df, "ID_dec", ts_col, data_cols, window=10)

    feat_cols = (["ID_dec"] +
                 ([dlc_col] if dlc_col else []) +
                 ["ts_diff", "id_freq", "ts_mean", "ts_std",
                  "data_mean", "data_std"] +
                 data_cols)
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(df[label_col].astype(str).str.strip())

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    logger.info(f"  train {X_tr.shape}  test {X_te.shape}")

    _save((X_tr, y_tr), cache_tr)
    _save((X_te, y_te), data_dir / "can_test.pkl")
    _save({"le": le, "scaler": scaler, "feat_cols": feat_cols},
          data_dir / "can_meta.pkl")

    del df, X; gc.collect()
    return (X_tr, y_tr), (X_te, y_te), \
           {"le": le, "scaler": scaler, "feat_cols": feat_cols}

# ── CICIDS preprocessing ───────────────────────────────────────────────────
def preprocess_cicids(logger, data_dir: Path):
    cache_tr = data_dir / "cic_train.pkl"
    if cache_tr.exists():
        logger.info("CICIDS cache hit — loading")
        return _load(cache_tr), _load(data_dir / "cic_test.pkl"), \
               _load(data_dir / "cic_meta.pkl")

    logger.info(f"CICIDS reading {RAW_CIC} …")
    df = pd.read_csv(RAW_CIC, low_memory=False)
    df.columns = df.columns.str.strip()
    logger.info(f"  raw shape: {df.shape}")

    # clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    label_col = " Label" if " Label" in df.columns else "Label"
    df[label_col] = df[label_col].apply(_map_cic_label)
    logger.info(f"  labels:\n{df[label_col].value_counts()}")

    feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[feat_cols].values.astype(np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(df[label_col].astype(str))

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    logger.info(f"  train {X_tr.shape}  test {X_te.shape}")

    _save((X_tr, y_tr), cache_tr)
    _save((X_te, y_te), data_dir / "cic_test.pkl")
    _save({"le": le, "scaler": scaler, "feat_cols": feat_cols},
          data_dir / "cic_meta.pkl")

    del df, X; gc.collect()
    return (X_tr, y_tr), (X_te, y_te), \
           {"le": le, "scaler": scaler, "feat_cols": feat_cols}

# ── baseline training ──────────────────────────────────────────────────────
def train_baseline(X_tr, y_tr, X_te, y_te,
                   tag: str, logger, result_dir: Path) -> dict:
    logger.info(f"[{tag}] baseline training …")

    models = {
        "DT": DecisionTreeClassifier(
            max_depth=8, min_samples_split=8,
            min_samples_leaf=3, random_state=42),
        "RF": RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_split=8, min_samples_leaf=3,
            n_jobs=-1, random_state=42),
    }
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        models["ET"] = ExtraTreesClassifier(
            n_estimators=200, max_depth=8,
            n_jobs=-1, random_state=42)
    except Exception:
        pass
    try:
        from xgboost import XGBClassifier
        models["XGB"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, use_label_encoder=False,
            eval_metric="mlogloss", n_jobs=-1, random_state=42)
    except Exception:
        pass

    results = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        t_tr = time.time() - t0
        y_pred = model.predict(X_te)
        m = _metrics(y_te, y_pred, t_tr)
        results[name] = m
        logger.info(f"  {name:6s}  Acc={m['accuracy']:.4f}  "
                    f"F1={m['f1_macro']:.4f}  t={m['train_time_s']:.0f}s")

    out = result_dir / f"exp1_baseline_{tag}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info(f"  saved → {out}")
    return results

# ── entry point ────────────────────────────────────────────────────────────
def run(data_dir: Path, model_dir: Path, result_dir: Path, logger) -> dict:
    meta = {}

    can_tr, can_te, _ = preprocess_can(logger, data_dir)
    r_can = train_baseline(*can_tr, *can_te, "can", logger, result_dir)
    meta["can_rf_acc"] = r_can.get("RF", {}).get("accuracy")
    del can_tr, can_te; gc.collect()

    cic_tr, cic_te, _ = preprocess_cicids(logger, data_dir)
    r_cic = train_baseline(*cic_tr, *cic_te, "cic", logger, result_dir)
    meta["cic_rf_acc"] = r_cic.get("RF", {}).get("accuracy")
    del cic_tr, cic_te; gc.collect()

    logger.info("EXP1 complete ✓")
    return meta
