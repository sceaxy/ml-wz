"""
Main pipeline orchestrator.
Usage:
    python pipeline.py                  # run all stages
    python pipeline.py --stage 1        # run only stage 1
    python pipeline.py --from-stage 3   # run from stage 3 onwards
    python pipeline.py --list           # show stage status
"""
import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
CACHE_DIR  = ROOT / "cache"
DATA_DIR   = CACHE_DIR / "data"
MODEL_DIR  = CACHE_DIR / "models"
RESULT_DIR = CACHE_DIR / "results"
LOG_DIR    = CACHE_DIR / "logs"

for d in [DATA_DIR, MODEL_DIR, RESULT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

STATE_FILE = CACHE_DIR / "pipeline_state.json"

# ── logging ────────────────────────────────────────────────────────────────
def setup_logger(stage_name: str) -> logging.Logger:
    log_path = LOG_DIR / f"{stage_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(stage_name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")
    # file handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.handlers = [fh, ch]
    logger.info(f"Log → {log_path}")
    return logger

# ── state management ───────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))

def mark_stage(state: dict, stage_id: int, status: str, meta: dict = None):
    state[str(stage_id)] = {
        "status": status,
        "time": datetime.now().isoformat(),
        **(meta or {}),
    }
    save_state(state)

# ── stage runner ───────────────────────────────────────────────────────────
def run_stage(stage_id: int, state: dict, force: bool = False) -> bool:
    stage_key = str(stage_id)
    if not force and state.get(stage_key, {}).get("status") == "done":
        print(f"  ✓ Stage {stage_id} already done, skipping (use --force to re-run)")
        return True

    stage_map = {
        1: ("exp1_data.run",      "EXP1 数据准备与基线"),
        2: ("exp2_augment.run",   "EXP2 混合数据增强"),
        3: ("exp3_stacking.run",  "EXP3 Stacking集成训练"),
        4: ("exp4_pso.run",       "EXP4 PSO超参数优化"),
        5: ("exp5_deploy.run",    "EXP5 轻量化部署验证"),
    }
    if stage_id not in stage_map:
        print(f"Unknown stage {stage_id}")
        return False

    module_path, stage_name = stage_map[stage_id]
    logger = setup_logger(f"stage{stage_id}")
    logger.info(f"{'='*50}")
    logger.info(f"Starting {stage_name}")
    logger.info(f"{'='*50}")

    t0 = time.time()
    mark_stage(state, stage_id, "running")

    try:
        mod_name, func_name = module_path.rsplit(".", 1)
        mod = __import__(mod_name, fromlist=[func_name])
        func = getattr(mod, func_name)
        meta = func(
            data_dir=DATA_DIR,
            model_dir=MODEL_DIR,
            result_dir=RESULT_DIR,
            logger=logger,
        )
        elapsed = time.time() - t0
        logger.info(f"✓ {stage_name} finished in {elapsed/3600:.2f}h")
        mark_stage(state, stage_id, "done", {
            "elapsed_h": round(elapsed / 3600, 3),
            **(meta or {}),
        })
        gc.collect()
        return True

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"✗ Stage {stage_id} FAILED after {elapsed/60:.1f}min")
        logger.error(traceback.format_exc())
        mark_stage(state, stage_id, "failed", {"error": str(e)})
        return False

# ── list status ────────────────────────────────────────────────────────────
def list_status(state: dict):
    names = {
        1: "EXP1 数据准备与基线",
        2: "EXP2 混合数据增强",
        3: "EXP3 Stacking集成训练",
        4: "EXP4 PSO超参数优化",
        5: "EXP5 轻量化部署验证",
    }
    print("\n Pipeline Status")
    print("─" * 52)
    for sid, name in names.items():
        info = state.get(str(sid), {})
        status = info.get("status", "pending")
        icon = {"done": "✓", "running": "▶", "failed": "✗", "pending": "○"}.get(status, "?")
        elapsed = f"  ({info['elapsed_h']:.2f}h)" if "elapsed_h" in info else ""
        print(f"  {icon} Stage {sid}: {name}{elapsed}")
    print()

# ── main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",      type=int, help="Run only this stage")
    parser.add_argument("--from-stage", type=int, help="Run from this stage onwards", dest="from_stage")
    parser.add_argument("--force",      action="store_true", help="Re-run even if already done")
    parser.add_argument("--list",       action="store_true", help="Show stage status")
    args = parser.parse_args()

    # add src to path so stage modules are importable
    sys.path.insert(0, str(Path(__file__).parent))

    state = load_state()

    if args.list:
        list_status(state)
        return

    stages_to_run = []
    if args.stage:
        stages_to_run = [args.stage]
    elif args.from_stage:
        stages_to_run = list(range(args.from_stage, 6))
    else:
        stages_to_run = [1, 2, 3, 4, 5]

    print(f"\n▶ Running stages: {stages_to_run}\n")
    overall_t0 = time.time()

    for sid in stages_to_run:
        ok = run_stage(sid, state, force=args.force)
        if not ok:
            print(f"\n✗ Pipeline stopped at stage {sid}. Fix the error and re-run with --from-stage {sid}\n")
            sys.exit(1)

    total = (time.time() - overall_t0) / 3600
    print(f"\n✓ All done in {total:.2f}h\n")
    list_status(load_state())

if __name__ == "__main__":
    main()
