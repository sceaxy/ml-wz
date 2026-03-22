"""
EXP1: 数据准备、预处理、基线训练
输出:
  cache/data/can_train.pkl, can_test.pkl
  cache/data/cic_train.pkl, cic_test.pkl
  cache/results/exp1_baseline.json
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
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ── 数据集下载路径（挂载到容器里的原始数据目录）──────────────────────────
RAW_CAN_PATH  = Path(os.environ.get("RAW_CAN_PATH",  "/data/raw/car_hacking.csv"))
RAW_CIC_PATH  = Path(os.environ.get("RAW_CIC_PATH",  "/data/raw/cicids2017.csv"))

# CAN 标签列名
CAN_LABEL_COL = "Flag"
CIC_LABEL_COL = "Label"

# CICIDS 标签合并映射（细粒度 → 8 大类）
CIC_LABEL_MAP = {
    "BENIGN":                  "BENIGN",
    "DoS Hulk":                "DoS",
    "DoS GoldenEye":           "DoS",
    "DoS slowloris":           "DoS",
    "DoS Slowhttptest":        "DoS",
    "Heartbleed":              "DoS",
    "DDoS":                    "DoS",
    "PortScan":                "PortScan",
    "FTP-Patator":             "BruteForce",
    "SSH-Patator":             "BruteForce",
    "Bot":                     "Botnet",
    "Web Attack – Brute Force":"WebAttack",
    "Web Attack – XSS":        "WebAttack",
    "Web Attack – Sql Injection":"WebAttack",
    "Infiltration":            "Infiltration",
}

# ── helpers ────────────────────────────────────────────────────────────────
def _save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)

def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _metrics(y_true, y_pred, t_train: float) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy":  round(acc, 6),
        "f1_macro":  round(f1,  6),
        "train_time_s": round(t_train, 1),
        "per_class": {
            k: {m: round(v, 4) for m, v in vs.items()}
            for k, vs in report.items()
            if isinstance(vs, dict)
        },
    }

# ── CAN 数据预处理 ─────────────────────────────────────────────────────────
def preprocess_can(logger, data_dir: Path):
    cache = data_dir / "can_train.pkl"
    if cache.exists():
        logger.info("CAN cache hit, loading...")
        return _load(data_dir / "can_train.pkl"), _load(data_dir / "can_test.pkl"), _load(data_dir / "can_le.pkl")

    logger.info(f"Reading CAN data from {RAW_CAN_PATH} ...")
    # 分块读入以节省内存
    chunks = []
    for chunk in pd.read_csv(RAW_CAN_PATH, chunksize=500_000, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    logger.info(f"  Raw shape: {df.shape}")
    logger.info(f"  Label dist:\n{df[CAN_LABEL_COL].value_counts()}")

    # CAN ID 十六进制 → 十进制
    if df["ID"].dtype == object:
        df["ID_dec"] = df["ID"].apply(lambda x: int(str(x), 16))
    else:
        df["ID_dec"] = df["ID"]

    # DATA[0]~DATA[7] 十六进制 → 整数
    data_cols = [c for c in df.columns if c.startswith("DATA[")]
    for col in data_cols:
        df[col] = df[col].apply(lambda x: int(str(x), 16) if pd.notna(x) else 0)

    # 时间戳差分特征
    df["ts_diff"] = df["Timestamp"].diff().fillna(0).clip(0, 1)

    # 特征 & 标签
    feature_cols = ["ID_dec", "DLC", "ts_diff"] + data_cols
    X = df[feature_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df[CAN_LABEL_COL].astype(str))

    # 归一化
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"  CAN train: {X_train.shape}, test: {X_test.shape}")

    _save((X_train, y_train), data_dir / "can_train.pkl")
    _save((X_test,  y_test),  data_dir / "can_test.pkl")
    _save((le, scaler, feature_cols), data_dir / "can_le.pkl")

    del df, X
    gc.collect()
    return (X_train, y_train), (X_test, y_test), (le, scaler, feature_cols)

# ── CICIDS 数据预处理 ──────────────────────────────────────────────────────
def preprocess_cicids(logger, data_dir: Path):
    cache = data_dir / "cic_train.pkl"
    if cache.exists():
        logger.info("CICIDS cache hit, loading...")
        return _load(data_dir / "cic_train.pkl"), _load(data_dir / "cic_test.pkl"), _load(data_dir / "cic_le.pkl")

    logger.info(f"Reading CICIDS2017 from {RAW_CIC_PATH} ...")
    df = pd.read_csv(RAW_CIC_PATH, low_memory=False)
    logger.info(f"  Raw shape: {df.shape}")

    # 清洗
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # 标签合并
    df[CIC_LABEL_COL] = df[CIC_LABEL_COL].str.strip().map(CIC_LABEL_MAP).fillna("Other")
    logger.info(f"  Label dist:\n{df[CIC_LABEL_COL].value_counts()}")

    # 特征
    feature_cols = [c for c in df.columns if c != CIC_LABEL_COL]
    # 只保留数值列
    feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    X = df[feature_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df[CIC_LABEL_COL].astype(str))

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"  CICIDS train: {X_train.shape}, test: {X_test.shape}")

    _save((X_train, y_train), data_dir / "cic_train.pkl")
    _save((X_test,  y_test),  data_dir / "cic_test.pkl")
    _save((le, scaler, feature_cols), data_dir / "cic_le.pkl")

    del df, X
    gc.collect()
    return (X_train, y_train), (X_test, y_test), (le, scaler, feature_cols)

# ── 基线训练（对照原论文） ─────────────────────────────────────────────────
def train_baseline(X_train, y_train, X_test, y_test, dataset: str, logger, result_dir: Path):
    logger.info(f"[{dataset}] Training baseline models ...")
    results = {}

    configs = {
        "DT":  DecisionTreeClassifier(
                   criterion="gini", max_depth=8,
                   min_samples_split=8, min_samples_leaf=3, random_state=42),
        "RF":  RandomForestClassifier(
                   n_estimators=200, max_depth=8,
                   min_samples_split=8, min_samples_leaf=3,
                   n_jobs=-1, random_state=42),
    }

    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from xgboost import XGBClassifier
        configs["ET"]  = ExtraTreesClassifier(
                             n_estimators=200, max_depth=8,
                             n_jobs=-1, random_state=42)
        configs["XGB"] = XGBClassifier(
                             n_estimators=200, max_depth=6,
                             learning_rate=0.1, subsample=0.8,
                             use_label_encoder=False,
                             eval_metric="mlogloss",
                             n_jobs=-1, random_state=42)
    except ImportError as e:
        logger.warning(f"Optional model unavailable: {e}")

    for name, model in configs.items():
        logger.info(f"  Training {name} ...")
        t0 = time.time()
        model.fit(X_train, y_train)
        t_train = time.time() - t0
        y_pred = model.predict(X_test)
        m = _metrics(y_test, y_pred, t_train)
        results[name] = m
        logger.info(f"  {name}: Acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}  "
                    f"time={m['train_time_s']:.0f}s")

    out_path = result_dir / f"exp1_baseline_{dataset}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info(f"  Saved → {out_path}")
    return results

# ── entry point ────────────────────────────────────────────────────────────
def run(data_dir: Path, model_dir: Path, result_dir: Path, logger) -> dict:
    meta = {}

    # CAN
    can_train, can_test, _ = preprocess_can(logger, data_dir)
    baseline_can = train_baseline(*can_train, *can_test, "CAN", logger, result_dir)
    meta["can_rf_acc"] = baseline_can.get("RF", {}).get("accuracy")
    del can_train, can_test
    gc.collect()

    # CICIDS
    cic_train, cic_test, _ = preprocess_cicids(logger, data_dir)
    baseline_cic = train_baseline(*cic_train, *cic_test, "CICIDS", logger, result_dir)
    meta["cic_rf_acc"] = baseline_cic.get("RF", {}).get("accuracy")
    del cic_train, cic_test
    gc.collect()

    logger.info("EXP1 complete ✓")
    return meta
