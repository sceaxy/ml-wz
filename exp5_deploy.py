"""
EXP5: 轻量化部署验证

  1. 特征重要性三级选择 L1/L2/L3
  2. ONNX 导出 + INT8 量化
  3. 推理延迟基准
  4. 消融实验 (5 配置)
  5. 特征数-精度曲线

输入:  cache/data/{tag}_aug_{X,y}.npy
              cache/data/{tag}_test.pkl
              cache/models/exp4_{tag}_pso_best_params.json
输出:  cache/models/exp5_{tag}_l2_{fp32,int8}.onnx
       cache/results/exp5_{tag}_ablation.json
       cache/results/exp5_{tag}_feature_curve.json / .png
"""
import gc
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

THRESHOLDS = {"L1": 0.70, "L2": 0.90, "L3": 0.95}

# ── helpers ────────────────────────────────────────────────────────────────
def _load(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)

def _make_model(params: dict):
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=params.get("xgb_n_estimators", 200),
            max_depth=params.get("xgb_max_depth", 6),
            learning_rate=params.get("xgb_learning_rate", 0.1),
            subsample=params.get("xgb_subsample", 0.8),
            colsample_bytree=params.get("xgb_colsample_bytree", 0.8),
            reg_lambda=params.get("xgb_reg_lambda", 1.0),
            use_label_encoder=False, eval_metric="mlogloss",
            n_jobs=-1, random_state=42)
    return RandomForestClassifier(
        n_estimators=params.get("rf_n_estimators", 200),
        max_depth=params.get("rf_max_depth", 8),
        n_jobs=-1, random_state=42)

# ── feature selection ──────────────────────────────────────────────────────
def select_features(model, level: str) -> np.ndarray:
    imp   = model.feature_importances_
    order = np.argsort(imp)[::-1]
    cum   = np.cumsum(imp[order])
    n     = int(np.searchsorted(cum, THRESHOLDS[level])) + 1
    return order[:n]

# ── ONNX export ────────────────────────────────────────────────────────────
def export_onnx(model, fdim: int, path: Path, logger) -> bool:
    try:
        if HAS_XGB:
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
            init = [("float_input", FloatTensorType([None, fdim]))]
            om   = convert_xgboost(model, initial_types=init)
            import onnxmltools
            onnxmltools.utils.save_model(om, str(path))
        else:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            init = [("float_input", FloatTensorType([None, fdim]))]
            om   = convert_sklearn(model, initial_types=init)
            with open(path, "wb") as f:
                f.write(om.SerializeToString())
        sz = path.stat().st_size / 1e6
        logger.info(f"  ONNX FP32 -> {path.name}  {sz:.2f} MB")
        return True
    except Exception as e:
        logger.warning(f"  ONNX export failed: {e}")
        return False

def quantise(fp32: Path, int8: Path, logger) -> bool:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
        sz = int8.stat().st_size / 1e6
        logger.info(f"  INT8      -> {int8.name}  {sz:.2f} MB")
        return True
    except Exception as e:
        logger.warning(f"  INT8 quantisation failed: {e}")
        return False

def latency(onnx_path: Path, X: np.ndarray, logger,
            n: int = 200, batch: int = 32) -> float:
    try:
        import onnxruntime as ort
        sess  = ort.InferenceSession(str(onnx_path),
                providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        Xb    = X[:batch].astype(np.float32)
        for _ in range(20): sess.run(None, {iname: Xb})
        t0 = time.perf_counter()
        for _ in range(n):  sess.run(None, {iname: Xb})
        ms = (time.perf_counter() - t0) / n * 1000
        logger.info(f"  latency (batch={batch}): {ms:.2f} ms")
        return round(ms, 3)
    except Exception as e:
        logger.warning(f"  latency benchmark failed: {e}")
        return -1.0

# ── single ablation config ─────────────────────────────────────────────────
def eval_ablation(name: str, model,
                  X_tr, y_tr, X_te, y_te,
                  feat_idx: np.ndarray, logger) -> dict:
    Xtr = X_tr[:, feat_idx]
    Xte = X_te[:, feat_idx]
    t0  = time.time()
    model.fit(Xtr, y_tr)
    t_tr = time.time() - t0
    t1 = time.time()
    y_pred = model.predict(Xte)
    t_inf  = (time.time() - t1) / len(Xte) * 1000
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
    r   = {
        "config":          name,
        "n_features":      int(len(feat_idx)),
        "accuracy":        round(acc, 6),
        "f1_macro":        round(f1, 6),
        "train_time_s":    round(t_tr, 1),
        "infer_ms_sample": round(t_inf, 4),
    }
    logger.info(f"  {name:40s}  Acc={acc:.4f}  F1={f1:.4f}  "
                f"feats={len(feat_idx)}  {t_inf:.3f} ms/s")
    return r

# ── feature-count vs accuracy curve ───────────────────────────────────────
def feature_curve(model, X_tr, y_tr, X_te, y_te,
                  result_dir: Path, tag: str, logger):
    imp    = model.feature_importances_
    order  = np.argsort(imp)[::-1]
    n_feat = X_tr.shape[1]
    steps  = sorted(set(
        max(5, int(n_feat * r))
        for r in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
                  0.55, 0.70, 0.85, 1.00]))

    # FIX: 特征曲线用子集，只需要相对趋势，避免全量1300万行重复训练
    n_sub = min(len(X_tr), 500_000)
    logger.info(f"  feature curve using {n_sub:,} subset ...")

    records = []
    for n in steps:
        feats = order[:n]
        m = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        m.fit(X_tr[:n_sub, feats], y_tr[:n_sub])
        acc = accuracy_score(y_te, m.predict(X_te[:, feats]))
        records.append({"n_features": n, "accuracy": round(acc, 5)})
        logger.info(f"  curve n={n:4d}  acc={acc:.4f}")

    (result_dir / f"exp5_{tag}_feature_curve.json").write_text(
        json.dumps(records, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["n_features"] for r in records]
        ys = [r["accuracy"]   for r in records]
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, "o-", linewidth=1.5, markersize=5)
        plt.xlabel("Number of features")
        plt.ylabel("Accuracy")
        plt.title(f"Feature count vs Accuracy ({tag})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(result_dir / f"exp5_{tag}_feature_curve.png", dpi=100)
        plt.close()
    except Exception:
        pass

# ── entry point ────────────────────────────────────────────────────────────
def run(data_dir: Path, model_dir: Path, result_dir: Path, logger) -> dict:
    meta = {}

    for tag in ["cic", "can"]:
        logger.info(f"\n{'='*48}")
        logger.info(f"[{tag}] lightweight deployment")

        X_tr = np.load(data_dir / f"{tag}_aug_X.npy")
        y_tr = np.load(data_dir / f"{tag}_aug_y.npy")
        X_te, y_te = _load(data_dir / f"{tag}_test.pkl")

        # load PSO best params (若没有则用默认)
        p_path = model_dir / f"exp4_{tag}_pso_best_params.json"
        params = json.loads(p_path.read_text()) if p_path.exists() else {}

        # train full model for feature importances
        full_model = _make_model(params)
        logger.info("  Training full model for feature importances ...")
        full_model.fit(X_tr, y_tr)

        # feature subsets
        all_idx = np.arange(X_tr.shape[1])
        L1 = select_features(full_model, "L1")
        L2 = select_features(full_model, "L2")
        L3 = select_features(full_model, "L3")
        for lvl, idx in [("L1", L1), ("L2", L2), ("L3", L3), ("ALL", all_idx)]:
            logger.info(f"  {lvl}: {len(idx)} features")

        # feature curve (用子集)
        feature_curve(full_model, X_tr, y_tr, X_te, y_te,
                      result_dir, tag, logger)

        # ablation
        logger.info("\n  Ablation experiment:")
        ablation = []
        ablation.append(eval_ablation(
            "1. Baseline (default params)",
            _make_model({}), X_tr, y_tr, X_te, y_te, all_idx, logger))
        ablation.append(eval_ablation(
            "2. + PSO tuning (full features)",
            _make_model(params), X_tr, y_tr, X_te, y_te, all_idx, logger))
        ablation.append(eval_ablation(
            "3. + L3 feature selection (95%)",
            _make_model(params), X_tr, y_tr, X_te, y_te, L3, logger))
        ablation.append(eval_ablation(
            "4. + L2 feature selection (90%) *",
            _make_model(params), X_tr, y_tr, X_te, y_te, L2, logger))
        ablation.append(eval_ablation(
            "5. + L1 feature selection (70%, fast)",
            _make_model(params), X_tr, y_tr, X_te, y_te, L1, logger))

        out = result_dir / f"exp5_{tag}_ablation.json"
        out.write_text(json.dumps(ablation, indent=2, ensure_ascii=False))
        logger.info(f"  ablation -> {out}")

        # ONNX / INT8 on L2 subset
        m_l2 = _make_model(params)
        m_l2.fit(X_tr[:, L2], y_tr)
        fp32 = model_dir / f"exp5_{tag}_l2_fp32.onnx"
        int8 = model_dir / f"exp5_{tag}_l2_int8.onnx"
        ok = export_onnx(m_l2, len(L2), fp32, logger)
        if ok:
            quantise(fp32, int8, logger)
            lat_fp32 = latency(fp32, X_te[:, L2], logger)
            lat_int8 = latency(int8, X_te[:, L2], logger) \
                       if int8.exists() else -1.0
            meta[f"{tag}_lat_fp32_ms"] = lat_fp32
            meta[f"{tag}_lat_int8_ms"] = lat_int8

        del X_tr, y_tr, X_te, y_te
        gc.collect()

    logger.info("EXP5 complete *")
    return meta
