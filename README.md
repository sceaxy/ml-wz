# IoV IDS 实验流水线

## 目录结构

```
ids_project/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── pipeline.py        # 主编排器
│   ├── exp1_data.py       # 数据准备 + 基线
│   ├── exp2_augment.py    # SMOTE-Tomek + WGAN-GP
│   ├── exp3_stacking.py   # 异构 Stacking 集成
│   ├── exp4_pso.py        # PSO 超参数优化
│   └── exp5_deploy.py     # 轻量化部署验证
└── cache/                 # 挂载持久盘到这里
    ├── data/              # 预处理后的 .pkl / .npy
    ├── models/            # 训练好的模型
    ├── results/           # 指标 JSON + 图片
    ├── logs/              # 各阶段日志
    └── pipeline_state.json
```

## 本地构建与推送

```bash
# 1. 构建镜像
docker build -t your-dockerhub/iov-ids:latest .

# 2. 推送到 Docker Hub（ClawCloud 从这里拉）
docker push your-dockerhub/iov-ids:latest
```

## ClawCloud Run 部署

在 ClawCloud Run 控制台：

**镜像**: `your-dockerhub/iov-ids:latest`

**环境变量**:
```
RAW_CAN_PATH=/data/raw/car_hacking.csv
RAW_CIC_PATH=/data/raw/cicids2017.csv
```

**挂载持久盘**: 把持久盘挂到 `/cache`（保存所有中间结果）

**数据盘**: 原始数据集挂到 `/data/raw/`（只读）

**CMD 覆盖**（按需）:
```bash
# 跑全部阶段（默认）
python src/pipeline.py

# 只跑某个阶段
python src/pipeline.py --stage 2

# 从某阶段断点续跑
python src/pipeline.py --from-stage 3

# 查看各阶段状态
python src/pipeline.py --list

# 强制重跑某阶段（即使已完成）
python src/pipeline.py --stage 3 --force
```

## 数据集下载（容器内执行）

```bash
# CAN-Intrusion Dataset
# 从 https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset 下载
# 或用 wget（需要注册后获取直链）

# CICIDS2017
wget -P /data/raw/ https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/cicids2017_combined.csv
```

## 中间结果说明

| 文件 | 阶段 | 说明 |
|------|------|------|
| `cache/data/can_train.pkl` | EXP1 | CAN 训练集（归一化后） |
| `cache/data/cic_train.pkl` | EXP1 | CICIDS 训练集 |
| `cache/data/can_aug_train_X.npy` | EXP2 | 增强后 CAN 训练特征 |
| `cache/data/cic_aug_train_X.npy` | EXP2 | 增强后 CICIDS 训练特征 |
| `cache/models/exp3_*_stacking_hetero.pkl` | EXP3 | 异构 Stacking 模型 |
| `cache/models/exp4_*_pso_best_params.json` | EXP4 | PSO 最优超参数 |
| `cache/models/exp5_*_l2_int8.onnx` | EXP5 | INT8 量化模型 |
| `cache/results/exp1_baseline_*.json` | EXP1 | 基线指标（对照原论文） |
| `cache/results/exp3_*_stacking_hetero.json` | EXP3 | Stacking 性能 |
| `cache/results/exp4_*_comparison.json` | EXP4 | PSO vs Grid 对比 |
| `cache/results/exp5_*_ablation.json` | EXP5 | 消融实验 |
| `cache/results/exp3_*_shap.png` | EXP3 | SHAP 特征重要性图 |
| `cache/results/exp4_*_convergence.png` | EXP4 | PSO 收敛曲线 |
| `cache/results/exp5_*_feature_curve.png` | EXP5 | 特征数-精度曲线 |
| `cache/logs/stage*.log` | 所有 | 每阶段详细日志 |
| `cache/pipeline_state.json` | 所有 | 流水线状态（断点续跑依赖） |

## 断点续跑

`pipeline_state.json` 记录每个阶段的完成状态，容器重启后直接跑：

```bash
python src/pipeline.py  # 自动跳过已完成阶段
```

## 8GB 内存注意事项

- CAN 数据集 ~3GB 内存，处理完立即释放
- 不要同时持有两个数据集的 DataFrame
- WGAN-GP 和 Stacking 的 5折CV 是内存高峰，约 5~6GB
- 如果 OOM，在环境变量里设置 `PYTHONMALLOC=malloc`
