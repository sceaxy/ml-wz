"""
EXP2: 混合数据增强 — SMOTE-Tomek Links

策略:
  - CICIDS2017: 对稀有类（Infiltration/Botnet/WebAttack）做 SMOTE-Tomek
  - CAN 数据集: 类别较均衡，不做增强

输入:  /workspaces/cache/data/cic_train.pkl
输出:  /workspaces/cache/data/cic_aug_X.npy
              /workspaces/cache/data/cic_aug_y.npy
       /workspaces/cache/results/exp2_cic_aug_stats.json
       /workspaces/cache/results/exp2_rare_recall.json
"""
import gc
import json
import pickle
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

warnings.filterwarnings("ignore")

CACHE   = Path("/workspaces/cache")
DATA    = CACHE / "data"


def _load(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def run(data_dir: Path, model_dir: Path, result_dir: Path, logger) -> dict:
    aug_X_path = DATA / "cic_aug_X.npy"
    aug_y_path = DATA / "cic_aug_y.npy"

    if aug_X_path.exists() and aug_y_path.exists():
        logger.info("EXP2 cache hit — loading augmented data")
        X_aug = np.load(aug_X_path)
        logger.info(f"  augmented shape: {X_aug.shape}")
        return {"cic_aug_shape": list(X_aug.shape)}

    # 载入训练集
    logger.info("Loading CICIDS training set …")
    X_train, y_train = _load(DATA / "cic_train.pkl")
    X_test,  y_test  = _load(DATA / "cic_test.pkl")
    meta             = _load(DATA / "cic_meta.pkl")
    le               = meta["le"]

    logger.info(f"  train shape: {X_train.shape}")
    counts_before = Counter(y_train.tolist())
    logger.info("  class distribution before augmentation:")
    for cls_id, cnt in sorted(counts_before.items()):
        label = le.inverse_transform([cls_id])[0]
        logger.info(f"    {label:15s}: {cnt:>8,}")

    # 增强前基线召回率
    logger.info("\nBaseline RF (before augmentation) …")
    rf_before = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf_before.fit(X_train, y_train)
    y_pred_before  = rf_before.predict(X_test)
    report_before  = classification_report(y_test, y_pred_before, output_dict=True, zero_division=0)
    f1_before      = f1_score(y_test, y_pred_before, average="macro", zero_division=0)

    rare_names = ("Infiltration", "Botnet", "WebAttack")
    rare_map   = {le.transform([n])[0]: n for n in rare_names if n in le.classes_}
    before_recalls = {}
    for cls_id, label_name in rare_map.items():
        recall = report_before.get(str(cls_id), {}).get("recall", 0.0)
        before_recalls[label_name] = round(recall, 4)
        logger.info(f"  Before {label_name}: recall={recall:.4f}")

    # SMOTE-Tomek
    logger.info("\nApplying SMOTE-Tomek Links …")
    try:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.error("Run: pip install imbalanced-learn")
        raise

    min_samples = min(counts_before.values())
    k = min(5, min_samples - 1)
    logger.info(f"  k_neighbors={k}  (min class size={min_samples})")

    t0 = time.time()
    st = SMOTETomek(
        smote=SMOTE(k_neighbors=k, random_state=42),
        random_state=42, n_jobs=-1)
    X_aug, y_aug = st.fit_resample(X_train, y_train)
    logger.info(f"  SMOTE-Tomek done in {time.time()-t0:.1f}s")

    counts_after = Counter(y_aug.tolist())
    logger.info("  class distribution after augmentation:")
    for cls_id, cnt in sorted(counts_after.items()):
        label  = le.inverse_transform([cls_id])[0]
        before = counts_before.get(cls_id, 0)
        logger.info(f"    {label:15s}: {before:>8,} → {cnt:>8,}")

    # 增强后评估
    logger.info("\nEvaluating RF after augmentation …")
    rf_after = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf_after.fit(X_aug, y_aug)
    y_pred_after = rf_after.predict(X_test)
    report_after = classification_report(y_test, y_pred_after, output_dict=True, zero_division=0)
    f1_after     = f1_score(y_test, y_pred_after, average="macro", zero_division=0)

    after_recalls = {}
    for cls_id, label_name in rare_map.items():
        recall = report_after.get(str(cls_id), {}).get("recall", 0.0)
        after_recalls[label_name] = round(recall, 4)
        logger.info(f"  After  {label_name}: recall={recall:.4f}  "
                    f"(Δ={recall - before_recalls.get(label_name,0):+.4f})")

    logger.info(f"\n  F1-macro: {f1_before:.4f} → {f1_after:.4f}  "
                f"(Δ={f1_after-f1_before:+.4f})")

    # 保存缓存
    np.save(aug_X_path, X_aug)
    np.save(aug_y_path, y_aug)
    logger.info(f"  Saved → {aug_X_path}")

    # 保存统计（供 plot_results.py 读取）
    aug_stats = {
        "shape": list(X_aug.shape),
        "class_counts": {
            le.inverse_transform([k])[0]: int(v)
            for k, v in counts_after.items()
        },
    }
    (result_dir / "exp2_cic_aug_stats.json").write_text(
        json.dumps(aug_stats, indent=2, ensure_ascii=False))

    (result_dir / "exp2_rare_recall.json").write_text(
        json.dumps({
            "before":    before_recalls,
            "after":     after_recalls,
            "f1_before": round(f1_before, 4),
            "f1_after":  round(f1_after,  4),
        }, indent=2, ensure_ascii=False))

    logger.info("EXP2 complete ✓")
    del X_train, y_train, X_aug, y_aug; gc.collect()
    return {"f1_before": round(f1_before,4), "f1_after": round(f1_after,4)}
