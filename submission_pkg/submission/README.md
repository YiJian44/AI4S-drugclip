# DrugCLIP Virtual Screening Autonomous Agent
=============================================================
## 队伍名称: 弈剑
## 赛道: AI4S智能体CNS挑战赛·DrugCLIP虚拟筛选优化智能体

---

## 方法简介

**DrugCLIP + Fingerprint Centroid RRF融合** — CPU执行

本 Agent 结合 DrugCLIP 相似度与分子指纹 centroid 聚类，通过 Reciprocal Rank Fusion 融合多 K 值结果进行虚拟筛选。

| 数据集 | 评分方式 |
|--------|----------|
| DUD-E (102 tasks) | DrugCLIP raw score + RRF(k=50) |
| LIT-PCBA (15 tasks) | Fingerprint centroid (K=50/100/150/200) + RRF(k=60) |

**当前最佳平台分数: 51.039628**

---

## 是否使用外部模型/数据

- **外部数据**: DUD-E actives 参考配体 (data/dude_actives/)
- **外部模型**: 无 (纯本地计算)
- **GPU需求**: 无 (CPU only)

---

## 依赖说明

```
rdkit>=2022.03
scikit-learn>=1.0
pandas>=1.3
numpy>=1.21
scipy>=1.7
```

环境: `~/python_pkgs/conda_envs/ml/bin/python3`

---

## 预估运行时间

**30-90分钟** (取决于配体数量和迭代次数，默认 20 iterations 约 60-90 分钟)

---

## 运行方式

```bash
cd submission_pkg/submission
bash run.sh <input_dir> <output_dir> [max_iterations]

# 示例
bash run.sh /path/to/benchmark /path/to/output 20
```

---

## 输出文件说明

程序运行完成后，在 `<output_dir>` 下生成:

```
result.zip
├── result.csv    # task_id, ligand_id, score
└── result.log    # agent 运行过程记录
```

**result.csv 格式要求**:
- 每行: `task_id,ligand_id,score`
- 每个 (task_id, ligand_id) 必须唯一
- score 越高表示排序越靠前

**result.log 记录内容**:
- 初始化配置
- 数据加载过程
- 模型/策略选择过程
- 多轮决策/迭代过程
- 关键中间结果
- 最终结果生成过程

---

## 搜索空间

| 维度 | 探索选项 |
|------|---------|
| 指纹维度 | [2048, 167, 4096], [1024, 167, 2048] |
| 指纹权重 | [0.33,0.33,0.34], [0.4,0.3,0.3], [0.5,0.25,0.25] |
| 指纹缩放 | 1.5, 2.0, 2.5 |
| 聚类K值 | [30,50,80,120], [20,40,80,120], [25,50,100] |
| 质心模式 | sum_norm, mean, median |
| RRF_k | 30, 50, 60, 80 |
| 融合策略 | rrf, weighted_sum |

---

## 项目结构

```
submission_pkg/submission/
├── README.md
├── run.sh              # 统一启动脚本
├── requirements.txt
├── src/
│   ├── main.py
│   ├── agent/          # Agent 决策逻辑
│   ├── models/         # 评分模型
│   ├── utils/          # 工具函数
│   └── data/           # 数据加载
├── configs/
│   └── default.yaml
├── data/
│   └── dude_actives/    # DUD-E 参考配体
└── logs/               # 运行日志
```

---

## 可复现性

- 固定随机种子 (scikit-learn random_state=42)
- 使用相对路径，不依赖绝对路径
- 不依赖交互式人工干预
- 单机自动运行

---

## Agent 行为

本 Agent 体现以下自主能力:
- 基于反馈的多轮优化 (OptimizationHistory 记录每轮配置+分数)
- 短板优先策略 (LIT-PCBA 优先优化)
- 自动参数空间探索