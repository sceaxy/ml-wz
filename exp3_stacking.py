"""
EXP3: 异构 Stacking 集成训练
修复:
  1. CV 后不再重复训练基模型（直接复用 train_single_models 的结果）
  2. 超过50万行用20%子集做CV，避免卡死
  3. SHAP 读取正确的 meta 文件，不存在则跳过
"""
import gc
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

CACHE = Path(os.environ.get("CACHE_DIR", "/root/cache"))

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


def _metrics(y_true, y_pred, t_train: float) -> dict:
    return {
        "accuracy":     round(accuracy_score(y_true, y_pred), 6),
        "f1_macro":     round(f1_score(y_true, y_pred, average="macro", zero_division=0), 6),
        "train_time_s": round(t_train, 1),
        "report":       classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def build_base_models(params: dict = None) -> dict:
    p = params or {}
    models = {
        "DT": DecisionTreeClassifier(
            max_depth=p.get("dt_max_depth", 8),
            min_samples_split=p.get("dt_min_samples_split", 8),
            min_samples_leaf=p.get("dt_min_samples_leaf", 3),
            random_state=42,
        ),
        "RF": RandomForestClassifier(
            n_estimators=p.get("rf_n_estimators", 200),
            max_depth=p.get("rf_max_depth", 8),
            min_samples_split=p.get("rf_min_samples_split", 8),
            min_samples_leaf=p.get("rf_min_samples_leaf", 3),
            n_jobs=-1, random_state=42,
        ),
        "ET": ExtraTreesClassifier(
            n_estimators=p.get("et_n_estimators", 200),
            max_depth=p.get("et_max_depth", 8),
            n_jobs=-1, random_state=42,
        ),
    }
    if HAS_XGB:
        models["XGB"] = XGBClassifier(
            n_estimators=p.get("xgb_n_estimators", 200),
            max_depth=p.get("xgb_max_depth", 6),
            learning_rate=p.get("xgb_learning_rate", 0.1),
            subsample=p.get("xgb_subsample", 0.8),
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1, random_state=42,
        )
    return models


def train_single_models(X_train, y_train, X_test, y_test,
                        dataset, logger, model_dir, result_dir, params=None):
    logger.info(f"[{dataset}] Training single models ...")
    models = build_base_models(params)
    results = {}
    for name, model in models.items():
        logger.info(f"  {name} ...")
        t0 = time.time()
        model.fit(X_train, y_train)
        t_train = time.time() - t0
        y_pred = model.predict(X_test)
        m = _metrics(y_test, y_pred, t_train)
        results[name] = m
        logger.info(f"  {name}: Acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}  time={m['train_time_s']:.0f}s")
        _save(model, model_dir / f"exp3_{dataset}_{name.lower()}.pkl")
    (result_dir / f"exp3_{dataset}_single.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    return models, results


def train_stacking(X_train, y_train, X_test, y_test,
                   base_models, dataset, label,
                   logger, model_dir, result_dir):
    logger.info(f"[{dataset}] Building Stacking ({label}) ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # FIX 1: 超过50万行用20%子集做CV
    if len(X_train) > 500_000:
        X_cv, _, y_cv, _ = train_test_split(
            X_train, y_train, train_size=0.2,
            stratify=y_train, random_state=42)
        logger.info(f"  CV subset: {X_cv.shape} (20% of {X_train.shape})")
    else:
        X_cv, y_cv = X_train, y_train

    logger.info("  Generating meta-features via 5-fold CV ...")
    meta_train_parts = []
    t0 = time.time()
    for name, model in base_models.items():
        logger.info(f"    cross_val_predict: {name}")
        proba = cross_val_predict(
            model, X_cv, y_cv,
            cv=skf, method="predict_proba", n_jobs=-1)
        meta_train_parts.append(proba)
    meta_train = np.hstack(meta_train_parts)
    logger.info(f"  Meta-features shape: {meta_train.shape}  ({(time.time()-t0)/60:.1f}min)")

    # FIX 2: 不重复训练基模型，直接复用已训练好的模型
    meta_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    meta_clf.fit(meta_train, y_cv)

    meta_test = np.hstack([m.predict_proba(X_test) for m in base_models.values()])
    y_pred = meta_clf.predict(meta_test)
    t_total = time.time() - t0
    m = _metrics(y_test, y_pred, t_total)

    logger.info(f"  Stacking({label}): Acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}  time={m['train_time_s']:.0f}s")

    safe_label = label.replace(" ", "_")
    _save((base_models, meta_clf), model_dir / f"exp3_{dataset}_stacking_{safe_label}.pkl")
    (result_dir / f"exp3_{dataset}_stacking_{safe_label}.json").write_text(
        json.dumps(m, indent=2, ensure_ascii=False))
    return m


def run_shap(model, X_sample, feature_names, name, dataset, result_dir, logger):
    try:
        import shap, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        logger.info(f"  Running SHAP for {name} on {dataset} ...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        plt.figure(figsize=(8, 6))
        shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False, max_display=15)
        plt.tight_layout()
        out = result_dir / f"exp3_{dataset}_{name}_shap.png"
        plt.savefig(out, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"  SHAP plot -> {out}")
    except Exception as e:
        logger.warning(f"  SHAP skipped: {e}")


def train_homo_stacking(X_train, y_train, X_test, y_test,
                        dataset, logger, model_dir, result_dir):
    homo_models = {
        "RF_s1": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=1),
        "RF_s2": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=2),
        "RF_s3": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=3),
    }
    logger.info(f"[{dataset}] Training homo base models ...")
    for name, m in homo_models.items():
        logger.info(f"  {name} ...")
        m.fit(X_train, y_train)
    return train_stacking(X_train, y_train, X_test, y_test,
                          homo_models, dataset, "homo",
                          logger, model_dir, result_dir)


def run(data_dir: Path, model_dir: Path, result_dir: Path, logger) -> dict:
    meta = {}

    for dataset in ["can", "cic"]:
        X_train = np.load(data_dir / f"{dataset}_aug_X.npy")
        y_train = np.load(data_dir / f"{dataset}_aug_y.npy")
        test_key = "can_test.pkl" if dataset == "can" else "cic_test.pkl"
        X_test, y_test = _load(data_dir / test_key)

        logger.info(f"[{dataset}] train={X_train.shape}, test={X_test.shape}")

        base_models, _ = train_single_models(
            X_train, y_train, X_test, y_test,
            dataset, logger, model_dir, result_dir)

        hetero_result = train_stacking(
            X_train, y_train, X_test, y_test,
            base_models, dataset, "hetero",
            logger, model_dir, result_dir)
        meta[f"{dataset}_stacking_f1"] = hetero_result["f1_macro"]

        homo_result = train_homo_stacking(
            X_train, y_train, X_test, y_test,
            dataset, logger, model_dir, result_dir)
        meta[f"{dataset}_homo_f1"] = homo_result["f1_macro"]

        # FIX 3: SHAP 读取正确文件，不存在则跳过
        try:
            meta_pkl = data_dir / f"{dataset}_meta.pkl"
            feature_names = _load(meta_pkl).get("feat_cols") if meta_pkl.exists() else None
            sample_idx = np.random.choice(len(X_test), min(500, len(X_test)), replace=False)
            shap_model = base_models.get("XGB") or base_models.get("RF")
            run_shap(shap_model, X_test[sample_idx], feature_names,
                     "xgb" if "XGB" in base_models else "rf",
                     dataset, result_dir, logger)
        except Exception as e:
            logger.warning(f"  SHAP skipped: {e}")

        del X_train, y_train, X_test, y_test
        gc.collect()

    logger.info("EXP3 complete ✓")
    return meta
