"""
EXP4: PSO 超参数优化 vs Grid Search 对比
  - 用 20% 子集做搜索，降低计算量
  - 记录收敛曲线
  - 最优参数在全量数据上重训
输出:
  cache/models/exp4_pso_best_params.json
  cache/results/exp4_pso_vs_grid.json
  cache/results/exp4_convergence.png
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

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

# ── PSO 参数空间（对应论文表 4-1，18 维） ─────────────────────────────────
# 格式: (name, type, low, high)
PARAM_SPACE = [
    # DT (3 dims)
    ("dt_max_depth",           "int",   3,    15),
    ("dt_min_samples_split",   "int",   2,    20),
    ("dt_min_samples_leaf",    "int",   1,    10),
    # RF (4 dims)
    ("rf_n_estimators",        "int",   50,  300),
    ("rf_max_depth",           "int",   5,    20),
    ("rf_min_samples_split",   "int",   2,    20),
    ("rf_min_samples_leaf",    "int",   1,    10),
    # ET (4 dims)
    ("et_n_estimators",        "int",   50,  300),
    ("et_max_depth",           "int",   5,    20),
    ("et_min_samples_split",   "int",   2,    20),
    ("et_min_samples_leaf",    "int",   1,    10),
    # XGB (7 dims)
    ("xgb_n_estimators",       "int",   50,  300),
    ("xgb_max_depth",          "int",   3,    10),
    ("xgb_learning_rate",      "float", 0.01, 0.3),
    ("xgb_subsample",          "float", 0.6,  1.0),
    ("xgb_colsample_bytree",   "float", 0.6,  1.0),
    ("xgb_reg_lambda",         "float", 0.1, 10.0),
    ("xgb_min_child_weight",   "int",   1,    10),
]

DIMS  = len(PARAM_SPACE)
LOWS  = np.array([p[2] for p in PARAM_SPACE], dtype=float)
HIGHS = np.array([p[3] for p in PARAM_SPACE], dtype=float)

def decode_particle(pos: np.ndarray) -> dict:
    """Map raw PSO position → hyperparameter dict."""
    p = np.clip(pos, LOWS, HIGHS)
    params = {}
    for i, (name, typ, lo, hi) in enumerate(PARAM_SPACE):
        val = p[i]
        params[name] = int(round(val)) if typ == "int" else float(val)
    return params

# ── fitness function ───────────────────────────────────────────────────────
def fitness(pos: np.ndarray, X_sub, y_sub, cv_folds: int, logger) -> float:
    """Lower is better (negative F1)."""
    from exp3_stacking import build_base_models
    params = decode_particle(pos)
    models = build_base_models(params)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    scores = []
    t0 = time.time()
    for name, model in models.items():
        try:
            s = cross_val_score(model, X_sub, y_sub,
                                cv=skf, scoring="f1_macro", n_jobs=-1)
            scores.append(s.mean())
        except Exception:
            scores.append(0.0)

    f1_mean  = np.mean(scores)
    # complexity penalty (lighter = better)
    complexity = sum(
        params.get(f"{m}_n_estimators", 100) / 1000 *
        params.get(f"{m}_max_depth",    8)   / 20
        for m in ["rf", "et", "xgb"]
    ) / 3

    alpha, beta = 0.85, 0.15
    fit = alpha * f1_mean - beta * complexity
    return -fit   # minimise

# ── simple PSO (no extra library needed) ───────────────────────────────────
def run_pso(X_sub, y_sub, logger,
            n_particles: int = 15,
            n_iters:     int = 20,
            cv_folds:    int = 3) -> tuple:
    """Returns (best_params_dict, convergence_history)."""
    rng = np.random.default_rng(42)

    # initialise
    pos = rng.uniform(LOWS, HIGHS, (n_particles, DIMS))
    vel = rng.uniform(-(HIGHS - LOWS) * 0.1,
                      (HIGHS - LOWS) * 0.1, (n_particles, DIMS))
    pbest_pos  = pos.copy()
    pbest_cost = np.array([fitness(p, X_sub, y_sub, cv_folds, logger) for p in pos])
    gbest_idx  = np.argmin(pbest_cost)
    gbest_pos  = pbest_pos[gbest_idx].copy()
    gbest_cost = pbest_cost[gbest_idx]

    history      = [gbest_cost]
    no_improve   = 0
    w_start, w_end = 0.9, 0.4
    c1 = c2 = 2.0

    logger.info(f"  PSO start: {n_particles} particles × {n_iters} iters")
    t0 = time.time()

    for it in range(n_iters):
        w = w_start - (w_start - w_end) * it / n_iters
        r1 = rng.random((n_particles, DIMS))
        r2 = rng.random((n_particles, DIMS))

        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos  - pos))
        pos = np.clip(pos + vel, LOWS, HIGHS)

        for i in range(n_particles):
            cost = fitness(pos[i], X_sub, y_sub, cv_folds, logger)
            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest_pos[i]  = pos[i].copy()
                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest_pos  = pos[i].copy()
                    no_improve = 0

        history.append(gbest_cost)
        elapsed = (time.time() - t0) / 60
        logger.info(f"  Iter {it+1:3d}/{n_iters}  gbest_F1={-gbest_cost:.4f}  "
                    f"({elapsed:.1f}min)")

        # early stop
        no_improve += 1
        if no_improve >= 8:
            logger.info(f"  Early stop at iter {it+1}")
            break

    logger.info(f"  PSO done: best F1={-gbest_cost:.4f}")
    return decode_particle(gbest_pos), history

# ── Grid Search (control group) ────────────────────────────────────────────
def run_grid_search(X_sub, y_sub, logger) -> tuple:
    from sklearn.model_selection import GridSearchCV
    if not HAS_XGB:
        logger.warning("XGB not available, using RF for grid search")
        model = RandomForestClassifier(n_jobs=-1, random_state=42)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth":    [6, 8, 10],
        }
    else:
        model = XGBClassifier(use_label_encoder=False,
                              eval_metric="mlogloss", n_jobs=-1, random_state=42)
        param_grid = {
            "n_estimators":  [100, 200],
            "max_depth":     [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
        }
    logger.info("  Grid Search running ...")
    t0 = time.time()
    gs = GridSearchCV(model, param_grid, cv=3,
                      scoring="f1_macro", n_jobs=-1, refit=True)
    gs.fit(X_sub, y_sub)
    t_gs = time.time() - t0
    logger.info(f"  Grid Search done: best_score={gs.best_score_:.4f}  time={t_gs:.0f}s")
    return gs.best_params_, t_gs, -gs.best_score_

# ── save convergence plot ──────────────────────────────────────────────────
def save_convergence_plot(history: list, result_dir: Path, dataset: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        iters = list(range(len(history)))
        f1_hist = [-c for c in history]
        plt.figure(figsize=(7, 4))
        plt.plot(iters, f1_hist, "o-", linewidth=1.5, markersize=4)
        plt.xlabel("Iteration")
        plt.ylabel("Best F1-macro")
        plt.title(f"PSO Convergence ({dataset})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = result_dir / f"exp4_{dataset}_convergence.png"
        plt.savefig(out, dpi=100)
        plt.close()
    except Exception:
        pass

# ── entry point ────────────────────────────────────────────────────────────
def run(data_dir: Path, model_dir: Path, result_dir: Path, logger) -> dict:
    meta = {}

    # Use CICIDS as the main PSO target (more complex, 78 features)
    for dataset in ["cic", "can"]:
        logger.info(f"\n{'='*40}")
        logger.info(f"[{dataset}] PSO hyperparameter search")

        X_aug = np.load(data_dir / f"{dataset}_aug_train_X.npy")
        y_aug = np.load(data_dir / f"{dataset}_aug_train_y.npy")

        # 20% subset for search
        X_sub, _, y_sub, _ = train_test_split(
            X_aug, y_aug, train_size=0.20, stratify=y_aug, random_state=42
        )
        logger.info(f"  PSO subset: {X_sub.shape}")

        # Grid Search (control)
        t0_gs = time.time()
        gs_params, gs_time, gs_cost = run_grid_search(X_sub, y_sub, logger)
        logger.info(f"  Grid Search time: {gs_time:.0f}s")

        # PSO
        t0_pso = time.time()
        pso_params, history = run_pso(X_sub, y_sub, logger,
                                      n_particles=15, n_iters=20, cv_folds=3)
        pso_time = time.time() - t0_pso

        save_convergence_plot(history, result_dir, dataset)

        # save best params
        params_path = model_dir / f"exp4_{dataset}_pso_best_params.json"
        params_path.write_text(json.dumps(pso_params, indent=2))
        logger.info(f"  Best PSO params → {params_path}")

        comparison = {
            "grid_search": {
                "time_s": round(gs_time, 1),
                "best_params": gs_params,
            },
            "pso": {
                "time_s": round(pso_time, 1),
                "best_params": pso_params,
                "speedup_vs_grid": round(gs_time / max(pso_time, 1), 2),
            },
        }
        (result_dir / f"exp4_{dataset}_comparison.json").write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False))

        meta[f"{dataset}_pso_speedup"] = comparison["pso"]["speedup_vs_grid"]

        del X_aug, y_aug, X_sub, y_sub
        gc.collect()

    logger.info("EXP4 complete ✓")
    return meta
