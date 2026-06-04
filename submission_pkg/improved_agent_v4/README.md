# DrugCLIP Virtual Screening Agent - improved_agent_v4

## 队伍信息
- 队伍名称: DrugCLIP_Agent
- 赛道名称: DrugCLIP虚拟筛选优化智能体
- 平台得分: **50.3763**

## 方法简介
本agent采用多尺度分子指纹聚类质心方法进行虚拟筛选。

### DUD-E任务 (102个靶点)
- **指纹**: MACCS(167位) + Morgan2(2048位,r=2) + Morgan3(4096位,r=3)
- **聚类**: MiniBatchKMeans(n_clusters=10)
- **质心**: 求和归一化 (sum → normalize)
- **评分**: max(cosine_sim(配体指纹, 质心_i)) × 权重融合

### LIT-PCBA任务 (15个靶点)
- 单参考配体SMILES + 指纹相似度

## 运行方式
```bash
bash run.sh <input_dir> <output_dir>
```

## 依赖说明
- Python 3.8+
- RDKit >= 2022.03
- scikit-learn
- pandas
- numpy

## 输出文件
- `result.csv`: task_id, ligand_id, score
- `result.log`: agent运行日志

## GPU需求
无需GPU，纯CPU运行

## 预估运行时间
约20-30分钟（117任务）