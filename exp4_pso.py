"""
EXP4: PSO 超参数优化 vs Grid Search

策略: 用 20% 子集做搜索 → 最优参数在全量上重训

输入:  cache/data/{tag}_aug_{X,y}.npy
输出:  cache/models/exp4_{tag}_pso_best_params.json
       cache/results/exp4_{tag}_comparison.json
       cache/results/exp4_{tag}_convergence.png
"""
import gc
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── 18-dim parameter space (论文表 4-1) ────────────────────────────────────
# (name, type, low, high)
SPACE = [
    ("dt_max_depth",         "int",   3,    15),
    ("dt_min_samples_split", "int",   2,    20),
    ("dt_min_samples_leaf",  "int",   1,    10),
    ("rf_n_estimators",      "int",   50,  300),
    ("rf_max_depth",         "int",   5,    20),
    ("rf_min_samples_split", "int",   2,    20),
    ("rf_min_samples_leaf",  "int",   1,    10),
    ("et_n_estimators",      "int",   50,  300),
    ("et_max_depth",         "int",   5,    20),
    ("et_min_samples_split", "int",   2,    20),
    ("et_min_samples_leaf",  "int",   1,    10),
    ("xgb_n_estimators",     "int",   50,  300),
    ("xgb_max_depth",        "int",   3,    10),
    ("xgb_learning_rate",    "float", 0.01, 0.3),
    ("xgb_subsample",        "float", 0.6,  1.0),
    ("xgb_colsample_bytree", "float", 0.6,  1.0),
    ("xgb_reg_lambda",       "float", 0.1, 10.0),
    ("xgb_min_child_weight", "int",   1,    10),
]
DIMS = len(SPACE)
LOWS  = np.array([s[2] for s in SPACE], dtype=float)
HIGHS = np.array([s[3] for s in SPACE], dtype=float)

def decode(pos: np.ndarray) -> dict:
    pos = np.clip(pos, LOWS, HIGHS)
    return {
        name: int(round(pos[i])) if typ == "int" else float(pos[i])
        for i, (name, typ, *_) in enumerate(SPACE)
    }

# ── fitness (lower = better) ───────────────────────────────────────────────
def fitness(pos: np.ndarray, X_sub, y_sub, cv: int) -> float:
    from exp3_stacking import build_base_models
    params = decode(pos)
    models = build_base_models(params)
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = []
    for m in models.values():
        try:
            s = cross_val_score(m, X_sub, y_sub,
                                cv=skf, scoring="f1_macro", n_jobs=-1)
            scores.append(s.mean())
        except Exception:
            scores.append(0.0)

    f1_mean = float(np.mean(scores))
    # light complexity penalty
    complexity = sum(
        params.get(f"{k}_n_estimators", 100) / 1000 *
        params.get(f"{k}_max_depth", 8)      / 20
        for k in ["rf", "et", "xgb"]
    ) / 3
    return -(0.85 * f1_mean - 0.15 * complexity)   # minimise

# ── vanilla PSO (no extra lib) ─────────────────────────────────────────────
def pso(X_sub, y_sub, logger,
        n_particles: int = 15,
        n_iters:     int = 20,
        cv:          int = 3) -> tuple:

    rng = np.random.default_rng(42)
    pos = rng.uniform(LOWS, HIGHS, (n_particles, DIMS))
    vel = rng.uniform(-(HIGHS - LOWS) * 0.1,
                       (HIGHS - LOWS) * 0.1, (n_particles, DIMS))

    pbest_pos  = pos.copy()
    pbest_cost = np.array([fitness(p, X_sub, y_sub, cv) for p in pos])
    g_idx      = int(np.argmin(pbest_cost))
    gbest_pos  = pbest_pos[g_idx].copy()
    gbest_cost = pbest_cost[g_idx]
    history    = [gbest_cost]
    no_improve = 0

    logger.info(f"  PSO  {n_particles} particles × {n_iters} iters  "
                f"init_F1={-gbest_cost:.4f}")
    t0 = time.time()

    for it in range(n_iters):
        w  = 0.9 - 0.5 * it / n_iters   # linearly decay 0.9 → 0.4
        r1 = rng.random((n_particles, DIMS))
        r2 = rng.random((n_particles, DIMS))

        vel = (w * vel
               + 2.0 * r1 * (pbest_pos - pos)
               + 2.0 * r2 * (gbest_pos  - pos))
        pos = np.clip(pos + vel, LOWS, HIGHS)

        improved = False
        for i in range(n_particles):
            c = fitness(pos[i], X_sub, y_sub, cv)
            if c < pbest_cost[i]:
                pbest_cost[i] = c
                pbest_pos[i]  = pos[i].copy()
                if c < gbest_cost:
                    gbest_cost = c
                    gbest_pos  = pos[i].copy()
                    improved   = True
                    no_improve = 0

        history.append(gbest_cost)
        if not improved:
            no_improve += 1

        logger.info(f"  iter {it+1:3d}/{n_iters}  "
                    f"gbest_F1={-gbest_cost:.4f}  "
                    f"{(time.time()-t0)/60:.1f} min")

        if no_improve >= 8:
            logger.info(f"  early-stop at iter {it+1}")
            break

    return decode(gbest_pos), history

# ── Grid Search (control group) ────────────────────────────────────────────
def grid_search(X_sub, y_sub, logger) -> tuple:
    from sklearn.model_selection import GridSearchCV
    if HAS_XGB:
        model = XGBClassifier(use_label_encoder=False,
                              eval_metric="mlogloss", n_jobs=-1, random_state=42)
        grid  = {"n_estimators": [100, 200],
                 "max_depth":    [4, 6, 8],
                 "learning_rate":[0.05, 0.1, 0.2]}
    else:
        model = RandomForestClassifier(n_jobs=-1, random_state=42)
        grid  = {"n_estimators": [100, 200],
                 "max_depth":    [6, 8, 10]}

    logger.info("  Grid Search …")
    t0 = time.time()
    gs = GridSearchCV(model, grid, cv=3,
                      scoring="f1_macro", n_jobs=-1, refit=True)
    gs.fit(X_sub, y_sub)
    t_gs = time.time() - t0
    logger.info(f"  Grid Search done  F1={gs.best_score_:.4f}  t={t_gs:.0f}s")
    return gs.best_params_, t_gs

# ── convergence plot ───────────────────────────────────────────────────────
def plot_convergence(history: list, result_dir: Path, tag: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f1s = [-c for c in history]
        plt.figure(figsize=(7, 4))
        plt.plot(f1s, "o-", linewidth=1.5, markersize=4)
        plt.xlabel("Iteration")
        plt.ylabel("Best F1-macro")
        plt.title(f"PSO Convergence ({tag})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(result_dir / f"exp4_{tag}_convergence.png", dpi=100)
        plt.close()
    except Exception:
        pass

# ── entry point ────────────────────────────────────────────────────────────
def run(data_dir: Path, model_dir: Path, result_dir: Path, logger) -> dict:
    meta = {}

    for tag in ["cic", "can"]:
        logger.info(f"\n{'='*48}")
        logger.info(f"[{tag}] PSO hyperparameter search")

        X_aug = np.load(data_dir / f"{tag}_aug_X.npy")
        y_aug = np.load(data_dir / f"{tag}_aug_y.npy")

        # 20% subset for search
        X_sub, _, y_sub, _ = train_test_split(
            X_aug, y_aug, train_size=0.20, stratify=y_aug, random_state=42)
        logger.info(f"  search subset: {X_sub.shape}")

        # Grid Search (control)
        gs_params, gs_time = grid_search(X_sub, y_sub, logger)

        # PSO (experiment)
        t0 = time.time()
        best_params, history = pso(X_sub, y_sub, logger,
                                   n_particles=8, n_iters=10, cv=3)
        pso_time = time.time() - t0
        plot_convergence(history, result_dir, tag)

        # save params
        p_path = model_dir / f"exp4_{tag}_pso_best_params.json"
        p_path.write_text(json.dumps(best_params, indent=2))

        cmp = {
            "grid_search": {"time_s": round(gs_time, 1), "params": gs_params},
            "pso":         {"time_s": round(pso_time, 1), "params": best_params,
                            "speedup": round(gs_time / max(pso_time, 1), 2)},
        }
        (result_dir / f"exp4_{tag}_comparison.json").write_text(
            json.dumps(cmp, indent=2, ensure_ascii=False))
        logger.info(f"  PSO speedup vs Grid: {cmp['pso']['speedup']}×")
        meta[f"{tag}_pso_speedup"] = cmp["pso"]["speedup"]

        del X_aug, y_aug, X_sub, y_sub; gc.collect()

    logger.info("EXP4 complete ✓")
    return meta
