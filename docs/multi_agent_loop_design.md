# DrugCLIP 多智能体闭环系统架构设计

> 版本: 1.0  
> 日期: 2026-05-25  
> 目标: 自动化执行"数据准备→训练/微调→评测→超参/策略迭代→结果汇报"闭环

---

## 1. 背景与约束

### 1.1 评测指标

```
平台评分 = (DUD-E Mean EF1% + LIT-PCBA Mean EF1%) / 2
```

当前状况:
- 最高提交: **50.3763**
- 本地 DUD-E EF1%: 可达 **3000%+**
- LIT-PCBA EF1%: **严重短板**（需重点优化）

### 1.2 数据集概览

| 数据集 | 任务数 | 配体总数 | 特点 |
|--------|--------|----------|------|
| DUD-E  | 102    | ~2M      | 单受体，活性数据充足 |
| LIT-PCBA | 15   | ~0.2M    | 多受体，数据稀疏 |
| **合计** | **117** | **~2.2M** | |

### 1.3 现有资源

- **Agent 代码**: `submission_pkg/` (历史版本 v2-v8, improved_agent_v4 等)
- **Benchmark**: `benchmark/` (117个任务)
- **DrugCLIP 模型**: `DrugCLIP-BaseLine-master/` + `auto_improve_agent.py`
- **指纹方法**: Morgan2(2048) + MACCS(167) + Morgan3(4096) 聚类质心
- **Coding Agent**: Claude Code CLI (`npm install -g @anthropic-ai/claude-code`)
- **活性数据**: `data/dude_actives/` (DUD-E actives)

---

## 2. 系统架构总览

### 2.1 多智能体角色定义

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Orchestration Layer                           │
│                    (主循环调度器 / 结果汇总 / 终止判断)                    │
└─────────────────────────────────────────────────────────────────────────┘
          │               │               │               │
          ▼               ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │  Coach   │   │ Trainer  │   │Evaluator │   │Strategist│
   └──────────┘   └──────────┘   └──────────┘   └──────────┘
                                                   │
                                                   ▼
                                            ┌──────────┐
                                            │Submitter │
                                            └──────────┘
```

### 2.2 角色职责矩阵

| 角色 | 核心职责 | 输入 | 输出 | 决策自主度 |
|------|----------|------|------|------------|
| **Coach** | 训练计划制定、进度跟踪、异常诊断 | 评测报告、上轮迭代结果 | 训练计划、数据需求清单 | 高 |
| **Trainer** | 模型训练/微调、checkpoint管理 | 训练数据、配置参数 | 模型权重、训练日志 | 中 |
| **Evaluator** | 完整评测(117任务)、分数计算 | 模型权重 | 评测报告(各任务EF1%) | 低 |
| **Strategist** | 策略分析、改进建议生成 | 评测结果、历史对比 | 策略调整提案 | 高 |
| **Submitter** | 结果打包、格式校验、提交 | result.csv | result.zip | 低 |

---

## 3. 各 Agent 详细设计

### 3.1 Coach Agent

**角色定位**: 训练过程的导演，负责制定训练计划、监控进度、诊断问题。

```python
class CoachAgent:
    """
    输入: 上轮评测结果, 训练日志, 当前迭代轮次
    输出: TrainingPlan {
        epochs: int,
        batch_size: int,
        learning_rate: float,
        data_sources: List[str],  # ["pdbbind", "dude_actives"]
        sampling_strategy: str,   # "balanced" | "focus_litpcba"
        early_stop_patience: int
    }
    """

    def create_plan(self, context: dict) -> TrainingPlan:
        """
        决策逻辑:
        1. 如果 LIT-PCBA EF1% < 5: 聚焦 LIT-PCBA 数据增强
        2. 如果 DUD-E > 2000%: 减少 DUD-E 权重，平衡 LIT-PCBA
        3. 如果连续3轮无提升: 调整学习率或数据采样策略
        4. 如果过拟合迹象: 增加正则化或早停
        """
        pass

    def diagnose(self, eval_report: dict) -> DiagnosisReport:
        """诊断当前瓶颈"""
        pass
```

**工具集**:
- `benchmark/manifest.jsonl` - 任务列表读取
- `data/dude_actives/` - DUD-E 活性配体读取
- `DrugCLIP-BaseLine-master/data/` - 字典文件读取
- 统计分析工具 (numpy/pandas)

**决策规则**:
```
IF litpcba_mean_ef1 < 3:
    plan.data_sources = ["pdbbind", "dude_actives", "litpcba_actives"]
    plan.sampling_strategy = "focus_litpcba"
    plan.epochs = 30
ELIF dude_mean_ef1 > 3000% AND litpcba_mean_ef1 < 10:
    plan.epochs = 20
    plan.focus_on_litpcba = True
ELSE:
    plan.epochs = 15
    plan.data_sources = ["pdbbind", "dude_actives"]
```

---

### 3.2 Trainer Agent

**角色定位**: 执行模型训练，负责数据准备、训练循环、checkpoint管理。

```python
class TrainerAgent:
    """
    输入: TrainingPlan, 初始模型路径(可选)
    输出: best_model.pt, train_log.json
    """

    def prepare_data(self, plan: TrainingPlan) -> DataLoader:
        """
        数据准备流程:
        1. 从 PDBbind 采样 (n_positives=800~2000)
        2. 采样 DUD-E actives 作为正样本
        3. 构建 (pocket, ligand) token 对
        4. 返回 DataLoader
        """
        pass

    def train(self, plan: TrainingPlan) -> Checkpoint:
        """
        训练循环伪代码:

        model = DrugCLIPModel().to(DEVICE)
        optimizer = AdamW(lr=plan.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=plan.epochs)

        FOR epoch IN range(plan.epochs):
            model.train()
            FOR batch IN dataloader:
                loss = train_step(model, batch)
                optimizer.step()

            scheduler.step()

            # 快速评测 (每3轮或最后一轮)
            IF (epoch+1) % 3 == 0 OR epoch == plan.epochs-1:
                quick_ef1 = quick_eval(model, n_ligands=500)
                IF quick_ef1 > best_ef1:
                    save_checkpoint(model, "best_model.pt")
                    best_ef1 = quick_ef1

            # 早停检查
            IF epochs_without_improvement > plan.early_stop_patience:
                BREAK

        RETURN best_checkpoint
        """
        pass
```

**与现有代码集成**:
- 继承 `auto_improve_agent.py` 的 `DrugCLIPModel`, `quick_eval`, `train_step`
- 数据准备复用 `prepare_training_data()` 函数
- 新增 `prepare_litpcba_data()` 专门处理 LIT-PCBA

**Checkpoint 管理**:
```
output/
├── best_model.pt          # 最佳模型
├── last_model.pt          # 最后模型
└── train_log.json         # 训练日志 {losses[], eval_ef1s[], epoch_times[]}
```

---

### 3.3 Evaluator Agent

**角色定位**: 对117个任务完整评测，计算 EF1% 指标。

```python
class EvaluatorAgent:
    """
    输入: model.pt
    输出: EvalReport {
        dude_mean_ef1: float,
        dude_median_ef1: float,
        litpcba_mean_ef1: float,
        litpcba_median_ef1: float,
        platform_score: float,  # (dude_mean + litpcba_mean) / 2
        per_task_results: Dict[task_id, {ef1, n_hits, n_actives}]
    }
    """

    def load_model(self, checkpoint_path: str) -> DrugCLIPModel:
        """加载模型权重"""
        pass

    def eval_task(self, model, task_id: str) -> TaskResult:
        """
        评测单任务流程:

        1. 读取 task.json 获取 receptor 路径
        2. 解析 pocket tokens
        3. 读取 ligands.csv
        4. 对每个 ligand 计算 score (与 pocket 的相似度)
        5. 按 score 降序排序
        6. 计算 top 1% 中的 hits 数
        7. EF1% = (hits / expected_hits) * 100

        其中 expected_hits = n_actives * 0.01
        """
        pass

    def eval_all(self, model) -> EvalReport:
        """评测全部117任务"""
        pass
```

**评测配置**:
- DUD-E: 102任务，使用 `dude_actives/` 标注 active
- LIT-PCBA: 15任务，使用 benchmark 内置活性标注
- 每任务评分: `score = ligand_embedding @ pocket_embedding`

**关键风险**: 评测耗时（2M+ 配体）
- 缓解: 分批处理，每批500配体，GPU加速

---

### 3.4 Strategist Agent

**角色定位**: 分析评测结果，生成策略改进建议。

```python
class StrategistAgent:
    """
    输入: EvalReport, HistoricalReports[]
    输出: StrategyProposal {
        diagnosis: str,           # "LIT-PCBA数据稀疏"
        hypothesis: str,          # "多受体融合可能提升"
        proposed_changes: List[Change],
        expected_improvement: float
    }
    """

    def analyze(self, current: EvalReport, history: List[EvalReport]) -> StrategyProposal:
        """
        分析决策逻辑:

        1. 对比 DUD-E vs LIT-PCBA 差距
        2. 识别最差任务 (bottom performers)
        3. 分析历史趋势 (上升/下降/ plateau)
        4. 生成改进假设

        策略库:
        - "litpcba_multi_receptor": LIT-PCBA 多受体 RRF 融合
        - "fingerprint_hybrid": 指纹+模型混合
        - "data_augmentation": LIT-PCBA 数据增强
        - "ensemble": 多模型投票
        """
        pass

    def generate_code_change(self, proposal: StrategyProposal) -> CodePatch:
        """
        使用 Claude Code CLI 生成代码修改:
        prompt = f"根据以下策略改进建议，修改 agent 代码:\n{proposal}"
        """
        pass
```

**策略候选库**:

| 策略ID | 描述 | 适用场景 | 预期提升 |
|--------|------|----------|----------|
| `fp_model_hybrid` | 指纹评分 + 模型评分 RRF 融合 | LIT-PCBA < 5% | +5-10 |
| `litpcba_multireceptor` | 多受体结构 RRF 融合 | LIT-PCBA 多结构任务 | +3-5 |
| `litpcba_data_augment` | 从相似 DUD-E 任务迁移 | LIT-PCBA 数据不足 | +2-3 |
| `ensemble_vote` | 多 checkpoint 投票 | 训练波动大 | +2-5 |
| `k_optimal` | 调整聚类 K 值 | 指纹方法调优 | +1-3 |

---

### 3.5 Submitter Agent

**角色定位**: 最终结果打包、格式校验、提交准备。

```python
class SubmitterAgent:
    """
    输入: result.csv, result.log
    输出: result.zip
    """

    def validate(self, result_csv: str) -> ValidationResult:
        """
        校验规则:
        1. 每个 (task_id, ligand_id) 出现且仅出现一次
        2. 所有117任务都有结果
        3. score 为数值
        4. ligand_id 格式正确
        """
        pass

    def package(self, output_dir: str) -> str:
        """
        打包流程:
        1. 校验 result.csv
        2. 写入 result.log (记录优化过程)
        3. zip result.csv result.log
        4. 返回 result.zip 路径
        """
        pass
```

**输出格式**:
```
result.zip
├── result.csv    # task_id, ligand_id, score
└── result.log    # agent 自主优化日志
```

---

## 4. 迭代循环流程

### 4.1 主循环文字版流程图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           MAIN LOOP                                         │
│                                                                              │
│  ┌─────────────┐                                                            │
│  │ 初始化      │                                                            │
│  │ n_iter=0    │                                                            │
│  │ best_score=0│                                                            │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │ ITERATION n_iter                                             │           │
│  │                                                              │           │
│  │  ┌───────────────┐                                           │           │
│  │  │ Step 1: Coach │                                           │           │
│  │  │ create_plan() │                                           │           │
│  │  │               │                                           │           │
│  │  │ 输入: 上轮结果 │                                           │           │
│  │  │ 输出: 训练计划 │                                           │           │
│  │  └───────┬───────┘                                           │           │
│  │          │                                                   │           │
│  │          ▼                                                   │           │
│  │  ┌───────────────┐                                           │           │
│  │  │ Step 2:Trainer│                                           │           │
│  │  │ train()       │                                           │           │
│  │  │               │                                           │           │
│  │  │ 输入: 训练计划 │                                           │           │
│  │  │ 输出: model.pt│                                           │           │
│  │  └───────┬───────┘                                           │           │
│  │          │                                                   │           │
│  │          ▼                                                   │           │
│  │  ┌───────────────┐                                           │           │
│  │  │ Step 3:       │                                           │           │
│  │  │ Evaluator     │                                           │           │
│  │  │ eval_all()    │                                           │           │
│  │  │               │                                           │           │
│  │  │ 输入: model.pt│                                           │           │
│  │  │ 输出: 评测报告│                                           │           │
│  │  └───────┬───────┘                                           │           │
│  │          │                                                   │           │
│  │          ▼                                                   │           │
│  │  ┌───────────────┐                                           │           │
│  │  │ Step 4:       │                                           │           │
│  │  │ Submitter     │                                           │           │
│  │  │ package()     │                                           │           │
│  │  │               │                                           │           │
│  │  │ 输出: result.zip                                           │           │
│  │  └───────┬───────┘                                           │           │
│  │          │                                                   │           │
│  │          ▼                                                   │           │
│  │  ┌──────────────────────────────────────────────────────────┐│           │
│  │  │ Step 5: 评分检查 & 策略决策                               ││           │
│  │  │                                                          ││           │
│  │  │ IF platform_score > best_score:                          ││           │
│  │  │     best_score = platform_score                          ││           │
│  │  │     保存 result.zip 为 best_submission.zip                ││           │
│  │  │                                                          ││           │
│  │  │ IF n_iter >= max_iterations:                              ││           │
│  │  │     退出循环，返回 best_submission                         ││           │
│  │  │                                                          ││           │
│  │  │ ELSE:                                                    ││           │
│  │  │    Strategist 分析结果                                    ││           │
│  │  │     生成策略改进提案                                       ││           │
│  │  │     n_iter += 1                                          ││           │
│  │  │     回到 Step 1                                          ││           │
│  │  └──────────────────────────────────────────────────────────┘│           │
│  │                                                              │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 伪代码实现

```python
def main_loop():
    # 初始化
    n_iter = 0
    max_iterations = 10  # 可配置
    best_score = 0.0
    best_submission = None

    # 初始教练计划 (第一轮使用默认配置)
    plan = TrainingPlan(
        epochs=20,
        batch_size=32,
        learning_rate=1e-3,
        data_sources=["pdbbind", "dude_actives"],
        sampling_strategy="balanced"
    )

    while n_iter < max_iterations:
        print(f"\n{'='*60}")
        print(f"迭代轮次: {n_iter + 1}/{max_iterations}")
        print(f"当前最佳分数: {best_score:.4f}")
        print(f"{'='*60}\n")

        # Step 1: Coach - 制定/调整训练计划
        if n_iter > 0 and eval_report:
            plan = coach.create_plan({
                "current_result": eval_report,
                "iterations": n_iter,
                "history": historical_results
            })
            print(f"[Coach] 生成训练计划: epochs={plan.epochs}, lr={plan.learning_rate}")

        # Step 2: Trainer - 执行训练
        print(f"[Trainer] 开始训练...")
        checkpoint_path = trainer.train(plan)
        print(f"[Trainer] 训练完成: {checkpoint_path}")

        # Step 3: Evaluator - 完整评测
        print(f"[Evaluator] 开始完整评测 (117任务)...")
        eval_report = evaluator.eval_all(checkpoint_path)
        print(f"[Evaluator] 评测完成:")
        print(f"  - DUD-E Mean EF1%: {eval_report.dude_mean_ef1:.2f}%")
        print(f"  - LIT-PCBA Mean EF1%: {eval_report.litpcba_mean_ef1:.2f}%")
        print(f"  - 平台评分: {eval_report.platform_score:.4f}")

        # Step 4: Submitter - 打包提交
        result_zip = submitter.package(output_dir=f"outputs/iter_{n_iter}")
        print(f"[Submitter] 打包完成: {result_zip}")

        # Step 5: 评分检查 & 策略决策
        if eval_report.platform_score > best_score:
            best_score = eval_report.platform_score
            best_submission = result_zip
            print(f"*** 新最佳分数: {best_score:.4f} ***")

        # 保存历史
        historical_results.append(eval_report)

        # 判断是否继续
        if n_iter >= max_iterations - 1:
            break

        # Strategist 分析 & 生成改进建议
        proposal = strategist.analyze(eval_report, historical_results)
        print(f"[Strategist] 策略建议: {proposal.hypothesis}")

        if proposal.expected_improvement < 0.5:
            print(f"[Strategist] 预期提升不足，提前终止")
            break

        # 应用策略调整 (通过 Claude Code 或手动)
        if use_claude_code:
            strategist.apply_via_claude(proposal)
        else:
            apply_manual_changes(proposal)

        n_iter += 1

    print(f"\n{'='*60}")
    print(f"闭环完成")
    print(f"最佳平台评分: {best_score:.4f}")
    print(f"最佳提交: {best_submission}")
    print(f"{'='*60}")

    return best_submission
```

---

## 5. 与现有代码集成

### 5.1 继承关系图

```
现有代码                           新架构
─────────────────────────────────────────────────────────────
auto_improve_agent.py              Trainer Agent
  ├── DrugCLIPModel                └── 继承使用
  ├── quick_eval()                └── 继承使用
  └── prepare_training_data()     └── 继承 + 扩展

submission_pkg/improved_agent_v4/  Fingerprint Method (基线)
  ├── mol_to_fp()                 Strategist 策略候选
  ├── build_centroids()           
  └── score_ligands_vec()         

DrugCLIP-BaseLine-master/          模型架构
  ├── train/data/utils.py          Trainer 数据准备
  └── 模型定义                    Trainer 模型加载
```

### 5.2 关键接口适配

```python
# 现有 auto_improve_agent.py 的 quick_eval() 只测2个任务
def quick_eval(model, n_ligands=500):
    """需扩展为 EvaluatorAgent.eval_task()"""

# 现有指纹方法返回排序，需要适配为输出原始分数
def score_ligands_vec(lig_fps, centroids):
    """返回每个 ligand 的分数，供模型评分融合使用"""
```

### 5.3 文件结构

```
drugclip/
├── multi_agent/
│   ├── __init__.py
│   ├── coach.py          # Coach Agent
│   ├── trainer.py        # Trainer Agent
│   ├── evaluator.py      # Evaluator Agent
│   ├── strategist.py     # Strategist Agent
│   ├── submitter.py      # Submitter Agent
│   ├── orchestrator.py  # 主循环调度器
│   └── config.py         # 配置管理
├── submission_pkg/       # 历史 agent 代码
├── benchmark/            # 评测benchmark
├── DrugCLIP-BaseLine-master/  # 模型代码
├── auto_improve_agent.py # 训练脚本
└── docs/
    └── multi_agent_loop_design.md  # 本文档
```

---

## 6. 关键技术风险与缓解

### 6.1 风险矩阵

| 风险ID | 风险描述 | 影响 | 概率 | 缓解方案 |
|--------|----------|------|------|----------|
| R1 | **LIT-PCBA 数据稀疏**: 15个任务中部分任务活性数据极少 | 平台评分 < 30 | 高 | 多受体 RRF 融合；从 DUD-E 相近靶点迁移 |
| R2 | **评测超时**: 2M+ 配体逐个评测耗时极长 | 无法完成评测 | 中 | 分批GPU加速；采样评测快速验证 |
| R3 | **策略梯度消失**: 多次迭代后改进越来越小 | plateau | 中 | 引入指纹方法作为补充策略 |
| R4 | **模型灾难性遗忘**: 训练新任务遗忘旧任务 | DUD-E 下降 | 低 | 保存最佳checkpoint；渐进式微调 |
| R5 | **Claude Code 调用失败**: 外部CLI不可用 | 策略无法自动应用 | 中 | 准备手动修改的代码diff |
| R6 | **提交格式错误**: 缺少字段或任务 | 提交无效 | 低 | Submitter 严格校验 |

### 6.2 LIT-PCBA 专项优化方案

```
问题根因: LIT-PCBA 活性配体数据不足，导致模型学不到有效表示

缓解方案:
1. 多受体结构融合 (RRF)
   - 每个 LIT-PCBA 任务可能有多个 receptor
   - 分别计算配体与各 receptor 的相似度
   - RRF 融合多个排序结果

2. 数据增强
   - 从 DUD-E 找结构相似的靶点
   - 迁移其活性配体作为辅助训练数据

3. 指纹-模型混合
   - 指纹方法在数据稀疏时更鲁棒
   - RRF 融合指纹分数和模型分数
   - 配重根据各方法在验证集上的表现自适应
```

---

## 7. 分阶段实现计划

### Phase 1: 现有代码编排 (1-2周)

**目标**: 实现自动化流程串联，不需要自主策略调整

**里程碑**:
- [ ] M1.1: 编写 `orchestrator.py` 主循环，可配置迭代次数
- [ ] M1.2: 迁移 `auto_improve_agent.py` 到 `TrainerAgent`
- [ ] M1.3: 实现 `EvaluatorAgent.eval_all()` 评测117任务
- [ ] M1.4: 实现 `SubmitterAgent` 打包校验
- [ ] M1.5: 端到端测试: 1次训练+评测+打包

**验收标准**: 一键运行，自动完成训练-评测-打包，不需人工干预

**代码示例** (Phase 1 orchestrator):

```python
def run_phase1():
    # 固定配置，不做策略调整
    plan = TrainingPlan(
        epochs=20,
        batch_size=32,
        lr=1e-3,
        data_sources=["pdbbind", "dude_actives"]
    )

    # 训练
    checkpoint = trainer.train(plan)

    # 评测
    report = evaluator.eval_all(checkpoint)

    # 打包
    submitter.package()

    print(f"平台评分: {report.platform_score:.4f}")
```

---

### Phase 2: 自主策略调整 (2-4周)

**目标**: Strategist 可生成策略建议，系统自动选择应用

**里程碑**:
- [ ] M2.1: 实现 `StrategistAgent.analyze()` 评测结果分析
- [ ] M2.2: 建立策略候选库 (≥5种策略)
- [ ] M2.3: 实现基于 Claude Code 的代码修改自动化
- [ ] M2.4: Coach 根据策略建议调整训练计划
- [ ] M2.5: 多轮迭代测试: 验证策略迭代有效性

**验收标准**: 
- 系统可自动完成 ≥3轮迭代
- 每轮迭代分数应有提升或持平（不下降）
- Claude Code 成功生成可用代码修改

**代码示例** (Phase 2 策略选择):

```python
def decide_strategy(eval_report, history):
    strategies = load_strategy_candidates()

    # 简单启发式选择
    if eval_report.litpcba_mean_ef1 < 5:
        return strategies['fp_model_hybrid']
    elif eval_report.litpcba_mean_ef1 < 10:
        return strategies['litpcba_multireceptor']
    else:
        return strategies['ensemble_vote']
```

---

### Phase 3: 完全自主 (4-8周)

**目标**: 系统自主发现新策略、自主实现、自主决策终止

**里程碑**:
- [ ] M3.1: 实现策略效果预测模型 (基于历史数据)
- [ ] M3.2: 实现新策略自动生成 (基于失败案例分析)
- [ ] M3.3: 实现自主终止判断 (基于提升边际)
- [ ] M3.4: 完整自动化: 无人值守运行至收敛
- [ ] M3.5: 达到平台评分 ≥55 (当前最高50.3763)

**高级特性**:
```python
class AdvancedStrategist:
    def generate_novel_strategy(self, failed_proposals: List[Proposal]) -> Strategy:
        """
        基于失败案例分析生成新策略
        1. 分析失败原因 (代码错误/策略无效/数据不足)
        2. 从策略库组合新策略
        3. 使用 Claude Code 生成实现
        """
        pass

    def should_terminate(self, history: List[EvalReport]) -> TerminationReason:
        """
        自主终止判断:
        - plateau: 连续5轮提升 < 0.5
        - resource: 达到最大迭代/时间
        - policy: 策略候选库枯竭
        """
        pass
```

---

## 8. 总结

本架构设计了一套5角色多智能体闭环系统:

| 角色 | 功能 | 自主度 |
|------|------|--------|
| Coach | 训练计划制定 | 高 |
| Trainer | 模型训练执行 | 中 |
| Evaluator | 完整评测计算 | 低 |
| Strategist | 策略分析生成 | 高 |
| Submitter | 结果打包校验 | 低 |

**核心创新点**:
1. **LIT-PCBA 专项优化**: 针对短板的策略候选库
2. **指纹-模型混合**: 融合指纹鲁棒性和模型表示能力
3. **渐进式自主**: Phase 1-3 逐步释放自主决策能力
4. **Claude Code 集成**: 自动化代码修改

**预期效果**:
- Phase 1: 稳定复现现有最佳分数 (~50)
- Phase 2: 通过策略迭代提升至 ~55
- Phase 3: 完全自主优化，冲击更高分数

---

## 附录: 快速参考

### A. 评分公式

```
EF1% = (top_1%中的hits数) / (总actives数 * 0.01) * 100
Platform Score = (DUD-E Mean EF1% + LIT-PCBA Mean EF1%) / 2
```

### B. 关键路径

```
训练数据: THU-ATOM_PDBbind/ + data/dude_actives/
模型代码: DrugCLIP-BaseLine-master/
Agent代码: submission_pkg/ (历史) + multi_agent/ (新)
评测基准: benchmark/ (117任务)
```

### C. 结果格式

```
result.csv: task_id, ligand_id, score
result.zip: result.csv + result.log
```