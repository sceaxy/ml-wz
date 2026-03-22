"""
EXP2: 混合数据增强
  - SMOTE-Tomek Links（中等稀有类别）
  - WGAN-GP（极稀有类别，IR > 500）
输出:
  cache/data/can_aug_train.pkl
  cache/data/cic_aug_train.pkl
  cache/results/exp2_augment.json
"""
import gc
import json
import pickle
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)

# ── IR 分析 ────────────────────────────────────────────────────────────────
def compute_ir(y) -> dict:
    counts = Counter(y.tolist())
    majority = max(counts.values())
    return {cls: round(majority / cnt, 1) for cls, cnt in counts.items()}

# ── SMOTE-Tomek ────────────────────────────────────────────────────────────
def apply_smote_tomek(X, y, logger):
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE

    ir = compute_ir(y)
    logger.info(f"  IR per class: {ir}")

    # 只对 IR > 5 的类别做 SMOTE（排除主类）
    minority_classes = [cls for cls, r in ir.items() if r > 5]
    if not minority_classes:
        logger.info("  No significant imbalance, skipping SMOTE-Tomek")
        return X, y

    logger.info(f"  Applying SMOTE-Tomek (minority classes: {minority_classes}) ...")
    st = SMOTETomek(
        smote=SMOTE(k_neighbors=5, random_state=42),
        random_state=42,
        n_jobs=-1,
    )
    X_res, y_res = st.fit_resample(X, y)
    logger.info(f"  After SMOTE-Tomek: {Counter(y_res.tolist())}")
    return X_res, y_res

# ── WGAN-GP（CPU极简版） ───────────────────────────────────────────────────
class Generator(object):
    """CPU-friendly minimal MLP generator"""
    def __init__(self, noise_dim, feature_dim, hidden=64):
        self.noise_dim = noise_dim
        self.layers = []
        dims = [noise_dim, hidden, hidden * 2, feature_dim]
        import torch.nn as nn
        import torch
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class Discriminator(object):
    def __init__(self, feature_dim, hidden=64):
        import torch.nn as nn
        dims = [feature_dim, hidden * 2, hidden, 1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_wgan_gp(X_rare, n_generate: int, n_epochs: int, logger, feature_dim: int):
    """Train WGAN-GP on CPU for rare class samples."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        logger.warning("PyTorch not available, skipping WGAN-GP")
        return X_rare

    NOISE_DIM  = min(32, feature_dim)
    HIDDEN     = 64
    LR         = 1e-4
    N_CRITIC   = 5
    LAMBDA_GP  = 10
    BATCH_SIZE = min(16, len(X_rare))

    X_t = torch.FloatTensor(X_rare)

    # networks
    G = nn.Sequential(
        nn.Linear(NOISE_DIM, HIDDEN), nn.LeakyReLU(0.2),
        nn.Linear(HIDDEN, HIDDEN * 2), nn.LeakyReLU(0.2),
        nn.Linear(HIDDEN * 2, feature_dim),
    )
    D = nn.Sequential(
        nn.Linear(feature_dim, HIDDEN * 2), nn.LeakyReLU(0.2),
        nn.Linear(HIDDEN * 2, HIDDEN), nn.LeakyReLU(0.2),
        nn.Linear(HIDDEN, 1),
    )
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.9))

    logger.info(f"  WGAN-GP: {len(X_rare)} samples → train {n_epochs} epochs on CPU ...")
    t0 = time.time()

    for epoch in range(n_epochs):
        # ── critic steps ──
        for _ in range(N_CRITIC):
            idx = torch.randint(0, len(X_t), (BATCH_SIZE,))
            real = X_t[idx]
            z = torch.randn(BATCH_SIZE, NOISE_DIM)
            fake = G(z).detach()

            # gradient penalty
            alpha = torch.rand(BATCH_SIZE, 1)
            interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
            d_interp = D(interp)
            grads = torch.autograd.grad(
                d_interp, interp,
                grad_outputs=torch.ones_like(d_interp),
                create_graph=True, retain_graph=True
            )[0]
            gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
            d_loss = D(fake).mean() - D(real).mean() + LAMBDA_GP * gp
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

        # ── generator step ──
        z = torch.randn(BATCH_SIZE, NOISE_DIM)
        g_loss = -D(G(z)).mean()
        opt_G.zero_grad(); g_loss.backward(); opt_G.step()

        if (epoch + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info(f"  Epoch {epoch+1}/{n_epochs}  D={d_loss.item():.3f}  "
                        f"G={g_loss.item():.3f}  {elapsed/60:.1f}min")

    # generate
    G.eval()
    with torch.no_grad():
        z = torch.randn(n_generate, NOISE_DIM)
        synthetic = G(z).numpy()
    logger.info(f"  Generated {n_generate} synthetic samples in {(time.time()-t0)/60:.1f}min")
    return synthetic

# ── per-class IR strategy ──────────────────────────────────────────────────
SMOTE_IR_THRESH = 10      # IR > 10 → use SMOTE
WGAN_IR_THRESH  = 500     # IR > 500 → use WGAN-GP
WGAN_EPOCHS     = 500     # reduce to 200 if too slow
WGAN_TARGET     = 500     # how many samples to generate per rare class

def augment_dataset(X, y, dataset: str, logger, data_dir: Path, result_dir: Path):
    cache_x = data_dir / f"{dataset}_aug_train_X.npy"
    cache_y = data_dir / f"{dataset}_aug_train_y.npy"
    if cache_x.exists():
        logger.info(f"  {dataset} augmented cache hit")
        return np.load(cache_x), np.load(cache_y)

    ir = compute_ir(y)
    logger.info(f"[{dataset}] IR: {ir}")

    # Step 1: WGAN-GP for ultra-rare classes
    X_extra_list = [X]
    y_extra_list = [y]
    feature_dim  = X.shape[1]

    for cls_id, ratio in ir.items():
        if ratio >= WGAN_IR_THRESH:
            mask = y == cls_id
            X_rare = X[mask]
            n_gen  = min(WGAN_TARGET, int(X_rare.shape[0] * 10))
            logger.info(f"  Class {cls_id}: IR={ratio:.0f}, applying WGAN-GP "
                        f"({len(X_rare)} real → {n_gen} synthetic)")
            synthetic = train_wgan_gp(X_rare, n_gen, WGAN_EPOCHS, logger, feature_dim)
            X_extra_list.append(synthetic.astype(np.float32))
            y_extra_list.append(np.full(len(synthetic), cls_id, dtype=y.dtype))

    if len(X_extra_list) > 1:
        X = np.vstack(X_extra_list)
        y = np.concatenate(y_extra_list)
        logger.info(f"  After WGAN-GP injection: {Counter(y.tolist())}")
    del X_extra_list, y_extra_list
    gc.collect()

    # Step 2: SMOTE-Tomek for moderately rare classes
    X, y = apply_smote_tomek(X, y, logger)

    np.save(cache_x, X)
    np.save(cache_y, y)
    logger.info(f"  Saved augmented data → {cache_x}")

    stats = {
        "shape": list(X.shape),
        "class_counts": {str(k): int(v) for k, v in Counter(y.tolist()).items()},
    }
    (result_dir / f"exp2_{dataset}_aug_stats.json").write_text(
        json.dumps(stats, indent=2))
    return X, y

# ── entry point ────────────────────────────────────────────────────────────
def run(data_dir: Path, model_dir: Path, result_dir: Path, logger) -> dict:
    meta = {}

    # CAN
    can_train = _load(data_dir / "can_train.pkl")
    X_can, y_can = can_train
    X_can_aug, y_can_aug = augment_dataset(X_can, y_can, "can", logger, data_dir, result_dir)
    meta["can_aug_shape"] = list(X_can_aug.shape)
    del X_can, y_can, X_can_aug, y_can_aug
    gc.collect()

    # CICIDS
    cic_train = _load(data_dir / "cic_train.pkl")
    X_cic, y_cic = cic_train
    X_cic_aug, y_cic_aug = augment_dataset(X_cic, y_cic, "cic", logger, data_dir, result_dir)
    meta["cic_aug_shape"] = list(X_cic_aug.shape)
    del X_cic, y_cic, X_cic_aug, y_cic_aug
    gc.collect()

    logger.info("EXP2 complete ✓")
    return meta
