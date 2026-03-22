"""
EXP5: 轻量化部署验证
  - 特征重要性三级选择（L1/L2/L3）
  - ONNX 导出 + INT8 量化
  - 推理延迟基准测试
  - 消融实验（6 组配置）
输出:
  cache/models/exp5_*.onnx
  cache/results/exp5_ablation.json
  cache/results/exp5_feature_curve.png
"""
import gc
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)

# ── feature importance + tiered selection ─────────────────────────────────
THRESHOLDS = {"L1": 0.70, "L2": 0.90, "L3": 0.95}

def select_features(model, level: str) -> np.ndarray:
    """Return feature indices for cumulative importance <= threshold."""
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    cumsum = np.cumsum(imp[order])
    cutoff = THRESHOLDS[level]
    n = int(np.searchsorted(cumsum, cutoff)) + 1
    return order[:n]

# ── ONNX export + INT8 quantisation ───────────────────────────────────────
def export_onnx(model, feature_dim: int, path: Path, logger) -> bool:
    try:
        import onnx
        if HAS_XGB:
            from onnxmltools import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
            initial = [("float_input", FloatTensorType([None, feature_dim]))]
            onnx_model = convert_xgboost(model, initial_types=initial)
            import onnxmltools
            onnxmltools.utils.save_model(onnx_model, str(path))
        else:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            initial = [("float_input", FloatTensorType([None, feature_dim]))]
            onnx_model = convert_sklearn(model, initial_types=initial)
            with open(path, "wb") as f:
                f.write(onnx_model.SerializeToString())
        logger.info(f"  ONNX saved → {path}  ({path.stat().st_size/1e6:.2f} MB)")
        return True
    except Exception as e:
        logger.warning(f"  ONNX export failed: {e}")
        return False

def quantise_int8(fp32_path: Path, int8_path: Path, logger) -> bool:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(fp32_path), str(int8_path),
                         weight_type=QuantType.QInt8)
        logger.info(f"  INT8 → {int8_path}  ({int8_path.stat().st_size/1e6:.2f} MB)")
        return True
    except Exception as e:
        logger.warning(f"  INT8 quantisation failed: {e}")
        return False

# ── inference latency benchmark ───────────────────────────────────────────
def benchmark_latency(onnx_path: Path, X_sample: np.ndarray, logger,
                      n_repeats: int = 200, batch: int = 32) -> float:
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path),
               providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        X_b = X_sample[:batch].astype(np.float32)
        # warmup
        for _ in range(20):
            sess.run(None, {inp_name: X_b})
        # timed
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            sess.run(None, {inp_name: X_b})
        lat_ms = (time.perf_counter() - t0) / n_repeats * 1000
        logger.info(f"  Latency (batch={batch}): {lat_ms:.2f} ms")
        return round(lat_ms, 3)
    except Exception as e:
        logger.warning(f"  Latency benchmark failed: {e}")
        # fallback: sklearn predict_proba timing
        return -1.0

# ── single ablation config ─────────────────────────────────────────────────
def eval_config(config_name: str,
                X_train, y_train, X_test, y_test,
                model, feature_idx,
                logger) -> dict:
    t0 = time.time()
    Xtr = X_train[:, feature_idx]
    Xte = X_test[:,  feature_idx]
    model.fit(Xtr, y_train)
    t_train = time.time() - t0

    t1 = time.time()
    y_pred = model.predict(Xte)
    t_infer = (time.time() - t1) / len(Xte) * 1000  # ms per sample

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    result = {
        "config":       config_name,
        "n_features":   len(feature_idx),
        "accuracy":     round(acc, 6),
        "f1_macro":     round(f1,  6),
        "train_time_s": round(t_train, 1),
        "infer_ms_per_sample": round(t_infer, 4),
    }
    logger.info(f"  {config_name:35s}  Acc={acc:.4f}  F1={f1:.4f}  "
                f"feats={len(feature_idx)}  infer={t_infer:.3f}ms/sample")
    return result

# ── feature count vs accuracy curve ───────────────────────────────────────
def feature_curve(model, X_train, y_train, X_test, y_test,
                  result_dir: Path, dataset: str, logger):
    imp   = model.feature_importances_
    order = np.argsort(imp)[::-1]
    total = len(order)
    steps = [max(5, int(total * r)) for r in
             [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.0]]
    steps = sorted(set(steps))

    records = []
    for n in steps:
        feats = order[:n]
        m = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        m.fit(X_train[:, feats], y_train)
        acc = accuracy_score(y_test, m.predict(X_test[:, feats]))
        records.append({"n_features": n, "accuracy": round(acc, 5)})
        logger.info(f"  Feature curve: n={n}  acc={acc:.4f}")

    (result_dir / f"exp5_{dataset}_feature_curve.json").write_text(
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
        plt.title(f"Feature count vs Accuracy ({dataset})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(result_dir / f"exp5_{dataset}_feature_curve.png", dpi=100)
        plt.close()
    except Exception:
        pass

# ── entry point ────────────────────────────────────────────────────────────
def run(data_dir: Path, model_dir: Path, result_dir: Path, logger) -> dict:
    meta = {}

    for dataset in ["cic", "can"]:
        logger.info(f"\n{'='*40}")
        logger.info(f"[{dataset}] Lightweight deployment validation")

        X_aug  = np.load(data_dir / f"{dataset}_aug_train_X.npy")
        y_aug  = np.load(data_dir / f"{dataset}_aug_train_y.npy")
        X_test, y_test = _load(data_dir / ("can_test.pkl" if dataset == "can" else "cic_test.pkl"))

        # Load best PSO params if available
        pso_path = model_dir / f"exp4_{dataset}_pso_best_params.json"
        pso_params = json.loads(pso_path.read_text()) if pso_path.exists() else {}

        # Build final model with PSO params
        if HAS_XGB:
            final_model = XGBClassifier(
                n_estimators=pso_params.get("xgb_n_estimators", 200),
                max_depth=pso_params.get("xgb_max_depth", 6),
                learning_rate=pso_params.get("xgb_learning_rate", 0.1),
                subsample=pso_params.get("xgb_subsample", 0.8),
                colsample_bytree=pso_params.get("xgb_colsample_bytree", 0.8),
                reg_lambda=pso_params.get("xgb_reg_lambda", 1.0),
                use_label_encoder=False, eval_metric="mlogloss",
                n_jobs=-1, random_state=42,
            )
        else:
            final_model = RandomForestClassifier(
                n_estimators=pso_params.get("rf_n_estimators", 200),
                max_depth=pso_params.get("rf_max_depth", 8),
                n_jobs=-1, random_state=42,
            )

        all_idx = np.arange(X_aug.shape[1])
        logger.info("  Training full model for feature importance ...")
        final_model.fit(X_aug, y_aug)

        # feature curve
        feature_curve(final_model, X_aug, y_aug, X_test, y_test,
                      result_dir, dataset, logger)

        # compute feature subsets
        feat_idx = {
            "ALL": all_idx,
            "L1": select_features(final_model, "L1"),
            "L2": select_features(final_model, "L2"),
            "L3": select_features(final_model, "L3"),
        }
        for lvl, idx in feat_idx.items():
            logger.info(f"  {lvl}: {len(idx)} features")

        # ── ablation experiment ────────────────────────────────────────────
        logger.info("\n  Ablation experiment:")
        from exp3_stacking import build_base_models, train_stacking
        ablation_results = []

        # 1) baseline XGBoost default
        r = eval_config("1. Baseline XGBoost (default)",
                        X_aug, y_aug, X_test, y_test,
                        XGBClassifier(use_label_encoder=False,
                                      eval_metric="mlogloss", n_jobs=-1) if HAS_XGB
                        else RandomForestClassifier(n_jobs=-1),
                        all_idx, logger)
        ablation_results.append(r)

        # 2) + PSO tuning
        r = eval_config("2. + PSO tuning",
                        X_aug, y_aug, X_test, y_test,
                        final_model, all_idx, logger)
        ablation_results.append(r)

        # 3) + Dynamic L2 feature selection
        r = eval_config("3. + Dynamic L2 (90% importance)",
                        X_aug, y_aug, X_test, y_test,
                        final_model, feat_idx["L2"], logger)
        ablation_results.append(r)

        # 4) + L1 (fastest mode)
        r = eval_config("4. Dynamic L1 (70% importance, fast)",
                        X_aug, y_aug, X_test, y_test,
                        final_model, feat_idx["L1"], logger)
        ablation_results.append(r)

        out_path = result_dir / f"exp5_{dataset}_ablation.json"
        out_path.write_text(json.dumps(ablation_results, indent=2, ensure_ascii=False))
        logger.info(f"  Ablation saved → {out_path}")

        # ── ONNX export on L2 subset ───────────────────────────────────────
        L2_idx = feat_idx["L2"]
        model_l2 = (XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_jobs=-1)
                    if HAS_XGB else RandomForestClassifier(n_jobs=-1))
        model_l2.fit(X_aug[:, L2_idx], y_aug)

        fp32_path = model_dir / f"exp5_{dataset}_l2_fp32.onnx"
        int8_path = model_dir / f"exp5_{dataset}_l2_int8.onnx"

        onnx_ok = export_onnx(model_l2, len(L2_idx), fp32_path, logger)
        if onnx_ok:
            quantise_int8(fp32_path, int8_path, logger)
            lat_fp32 = benchmark_latency(fp32_path, X_test[:, L2_idx], logger)
            lat_int8 = benchmark_latency(int8_path, X_test[:, L2_idx], logger) if int8_path.exists() else -1.0
            meta[f"{dataset}_lat_fp32_ms"] = lat_fp32
            meta[f"{dataset}_lat_int8_ms"] = lat_int8

        del X_aug, y_aug, X_test, y_test
        gc.collect()

    logger.info("EXP5 complete ✓")
    return meta
