# DrugCLIP Agent

DrugCLIP 虚拟筛选智能体 — 结合分子指纹聚类质心与 DrugCLIP 模型进行虚拟筛选。

## 最高分: 51.039628

## 目录结构

```
drugclip/
├── submission_pkg/
│   └── submission/          ← 提交版本 (最高分)
│       ├── src/
│       │   ├── agent/        # DrugClipAgent: Coach+Evaluator+Strategist+Submitter
│       │   ├── models/       # Scorer: DrugCLIP + Fingerprint RRF 融合
│       │   └── utils/        # IO、日志工具
│       └── data/dude_actives/ # DUD-E 活性配体参考数据
│
├── submission_pkg/improved_agent_v4/  ← 备选提交版本 (50.3763)
│
├── benchmark/                # 117 任务 benchmark (manifest.jsonl + tasks/)
│
├── DrugCLIP-BaseLine-master/ # 原始 baseline 代码
│
├── data/
│   └── dude_actives/         # DUD-E 活性配体数据
│
├── docs/
│   ├── 比赛说明.md
│   ├── 提交要求.md
│   └── multi_agent_loop_design.md  # 多智能体闭环架构设计
│
└── submission_with_model/    # 包含训练模型的完整版本
    └── models/               # 微调模型权重
```

## 方法

| 数据集 | 评分方式 |
|--------|----------|
| DUD-E (102 tasks) | DrugCLIP raw score + RRF(k=50) |
| LIT-PCBA (15 tasks) | Fingerprint centroid (K=50/100/150/200) + RRF(k=60) |

平台评分 = (DUD-E Mean EF1% + LIT-PCBA Mean EF1%) / 2

## 运行

```bash
cd submission_pkg/submission
bash run.sh <input_dir> <output_dir>
```

依赖: rdkit, scikit-learn, pandas, numpy, scipy

## 架构

- **Coach**: `sample_config()` — 随机采样超参配置，历史建议
- **Evaluator**: `Evaluator` class — 117 任务 EF1% 计算
- **Strategist**: `opt_history.suggest()` — 基于历史的改进方向建议
- **Submitter**: 内嵌在 `DrugClipAgent.run()` — result.zip 打包