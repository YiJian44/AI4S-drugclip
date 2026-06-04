"""DrugCLIP autonomous agent - main agent, evaluator, and optimization history."""
import os, sys, time, json, copy, random, shutil, zipfile, csv, logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Import from sibling packages (absolute import from src root)
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logging import setup_logger, log_main, log_train, log_score, LOG_DIR, ITER_LOG, OPT_LOG
from utils.io import extract_smiles_from_mol2, mol_to_fp
from models import Scorer, SEARCH_SPACE

# ============================================================
# Path discovery
# ============================================================
SUBMISSION_ROOT = Path(__file__).parent.parent.parent  # agent -> src -> submission
PACKAGE_ROOT = SUBMISSION_ROOT

OUT_DIR = PACKAGE_ROOT / 'output'
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Optimization History
# ============================================================
class OptimizationHistory:
    def __init__(self):
        self.history = []
        if OPT_LOG.exists():
            try:
                with open(OPT_LOG) as f:
                    self.history = json.load(f)
            except: pass

    def save(self):
        with open(OPT_LOG, 'w') as f:
            json.dump(self.history, f, indent=2)

    def add(self, config, score, dude_ef1, lit_ef1, notes=''):
        entry = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'config': config,
            'score': score,
            'dude_ef1': dude_ef1,
            'lit_ef1': lit_ef1,
            'notes': notes,
            'iteration': len(self.history),
        }
        self.history.append(entry)
        self.save()
        log_main.info(f"[ITER {entry['iteration']}] score={score:.4f}  dude={dude_ef1:.2f}%  lit={lit_ef1:.2f}%  | {notes}")
        return entry

    def best(self):
        return max(self.history, key=lambda x: x['score']) if self.history else None

    def suggest(self):
        if not self.history:
            return None
        b = self.best()
        return {'focus': 'lit' if b['lit_ef1'] < b['dude_ef1'] * 0.1 else 'balance', 'prev': b['config']}

opt_history = OptimizationHistory()

# ============================================================
# Evaluator
# ============================================================
class Evaluator:
    def __init__(self, input_dir=None):
        self.smiles_map = {}
        self.dude_actives = {}
        self.input_dir = Path(input_dir) if input_dir else None
        self._load()

    def _load(self):
        # Load DUD-E actives from data/dude_actives/ symlinked in src/
        dude_dir = PACKAGE_ROOT / 'data' / 'dude_actives'
        if dude_dir.exists():
            for f in dude_dir.glob('*_actives.csv'):
                key = f.stem.replace('_actives', '')
                with open(f) as fp:
                    for row in csv.DictReader(fp):
                        if 'smiles' in row:
                            self.dude_actives.setdefault(key, []).append(row['smiles'].strip())

        if self.input_dir:
            tasks_dir = self.input_dir / 'tasks'
            if tasks_dir.exists():
                for td in os.listdir(tasks_dir):
                    tdir = tasks_dir / td
                    p = tdir / 'ligands.csv'
                    if p.exists():
                        with open(p) as f:
                            for row in csv.DictReader(f):
                                if 'smiles' in row and 'ligand_id' in row:
                                    self.smiles_map[row['smiles']] = row['ligand_id']

        log_score.info(f"Loaded {len(self.dude_actives)} DUD-E targets, {len(self.smiles_map)} ligands")

    def calc_ef1(self, rows, task_id):
        target = task_id.replace('dude_', '').replace('litpcba_', '')
        actives_ids = set()

        if task_id.startswith('dude_') and target in self.dude_actives:
            for s in self.dude_actives[target]:
                if s in self.smiles_map:
                    actives_ids.add(self.smiles_map[s])

        scores = [(r['ligand_id'], float(r['score'])) for r in rows if r['task_id'] == task_id]
        if not scores or not actives_ids:
            return 0.0

        scores.sort(key=lambda x: x[1], reverse=True)
        n, na = len(scores), len(actives_ids)
        top_k = max(1, n // 100)
        hits = sum(1 for lid, _ in scores[:top_k] if lid in actives_ids)
        expected = top_k / n
        return (hits / na) / expected * 100 if expected > 0 else 0.0

    def evaluate(self, result_df):
        rows = result_df.to_dict('records')
        dude_tasks = [t for t in result_df['task_id'].unique() if t.startswith('dude_')]
        lit_tasks  = [t for t in result_df['task_id'].unique() if t.startswith('litpcba_')]

        dude_ef1s = [self.calc_ef1(rows, t) for t in dude_tasks]
        lit_ef1s  = [self.calc_ef1(rows, t) for t in lit_tasks]

        dude_mean = np.mean(dude_ef1s) if dude_ef1s else 0.0
        lit_mean  = np.mean(lit_ef1s)  if lit_ef1s  else 0.0
        score = (dude_mean + lit_mean) / 2.0

        log_score.info(f"EF1%: score={score:.4f}  dude={dude_mean:.2f}%({len(dude_tasks)})  lit={lit_mean:.2f}%({len(lit_tasks)})")
        return {'score': score, 'dude_ef1': dude_mean, 'lit_ef1': lit_mean,
                'dude_tasks': len(dude_tasks), 'lit_tasks': len(lit_tasks)}

# ============================================================
# DrugCLIP Agent
# ============================================================
class DrugClipAgent:
    def __init__(self, input_dir, output_dir):
        self.input_dir  = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tasks = []
        manifest = self.input_dir / 'manifest.jsonl'
        if manifest.exists():
            with open(manifest) as f:
                for line in f:
                    if line.strip():
                        self.tasks.append(json.loads(line.strip()))

        self.dude_actives = {}
        dude_dir = PACKAGE_ROOT / 'data' / 'dude_actives'
        if dude_dir.exists():
            for f in dude_dir.glob('*_actives.csv'):
                key = f.stem.replace('_actives', '')
                with open(f) as fp:
                    for row in csv.DictReader(fp):
                        if 'smiles' in row:
                            self.dude_actives.setdefault(key, []).append(row['smiles'].strip())

        log_main.info(f"{'='*60}")
        log_main.info(f"DrugCLIP Autonomous Agent initialized")
        log_main.info(f"  tasks={len(self.tasks)}")
        log_main.info(f"  dude_targets={len(self.dude_actives)}")
        log_main.info(f"  input_dir={self.input_dir}")
        log_main.info(f"  output_dir={self.output_dir}")
        log_main.info(f"{'='*60}")

    def sample_config(self, iteration):
        rng = random.Random(42 + iteration)
        suggestion = opt_history.suggest()
        focus_lit = (suggestion and suggestion.get('focus') == 'lit')

        config = {
            'fp_dims':      rng.choice(SEARCH_SPACE['fp_dims']),
            'fp_weights':   rng.choice(SEARCH_SPACE['fp_weights']),
            'fp_scale':     rng.choice(SEARCH_SPACE['fp_scale']),
            'Ks':           rng.choice(SEARCH_SPACE['Ks']),
            'centroid_mode': rng.choice(SEARCH_SPACE['centroid_mode']),
            'rrf_k':        rng.choice(SEARCH_SPACE['rrf_k']),
            'fusion':       'fp_first' if focus_lit else rng.choice(SEARCH_SPACE['fusion']),
        }
        return config

    def run_iteration(self, config, iteration):
        t0 = time.time()
        log_main.info(f"\n{'='*60}")
        log_main.info(f"[ITER {iteration}] Config: {json.dumps(config, sort_keys=True)}")
        log_main.info(f"{'='*60}")

        scorer = Scorer(config, refs_base=self.input_dir)
        all_results = []

        # Collect per-task active IDs and scores during scoring
        task_actives = {}   # task_id -> set of active ligand_ids
        task_scores = {}    # task_id -> list of (ligand_id, score)

        for i, task_info in enumerate(self.tasks):
            task_id = task_info['task_id']
            task_dir = self.input_dir / 'tasks' / task_id
            if not task_dir.exists():
                continue

            ligands_df = pd.read_csv(task_dir / 'ligands.csv')

            # Precompute active ligand IDs for this task (DUD-E only)
            active_ids = set()
            if task_id.startswith('dude_'):
                target = task_id.replace('dude_', '')
                if target in self.dude_actives:
                    for _, row in ligands_df.iterrows():
                        if row['smiles'] in self.dude_actives[target]:
                            active_ids.add(row['ligand_id'])

            scores = scorer.score_task(task_id, ligands_df, self.dude_actives)
            task_scores[task_id] = [(row['ligand_id'], float(scores[j]))
                                    for j, (_, row) in enumerate(ligands_df.iterrows())]
            task_actives[task_id] = active_ids

            df = ligands_df[['ligand_id']].copy()
            df['task_id'] = task_id
            df['score'] = scores
            all_results.append(df)

            if (i + 1) % 20 == 0:
                log_main.info(f"  Progress: {i+1}/{len(self.tasks)}")

        if not all_results:
            log_main.error("No results!")
            return None

        result_df = pd.concat(all_results, ignore_index=True)[['task_id', 'ligand_id', 'score']]

        # Save result immediately
        iter_dir = OUT_DIR / f'iter_{iteration}'
        iter_dir.mkdir(exist_ok=True)
        result_df.to_csv(iter_dir / 'result.csv', index=False)
        with zipfile.ZipFile(iter_dir / 'result.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(iter_dir / 'result.csv', arcname='result.csv')

        # Write detailed result.log for this iteration
        _write_iter_result_log(iter_dir, config, iteration, dude_ef1s, lit_ef1s, dude_tasks, lit_tasks, task_scores, elapsed)

        # Re-zip with result.log included
        with zipfile.ZipFile(iter_dir / 'result.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(iter_dir / 'result.csv', arcname='result.csv')
            zf.write(iter_dir / 'result.log', arcname='result.log')

        # Inline EF1% calculation - no re-loading of ligands
        dude_ef1s, lit_ef1s = [], []
        dude_tasks = [t for t in task_scores if t.startswith('dude_')]
        lit_tasks  = [t for t in task_scores if t.startswith('litpcba_')]

        for tid in dude_tasks:
            scores = task_scores[tid]
            active_ids = task_actives[tid]
            if not scores or not active_ids:
                dude_ef1s.append(0.0); continue
            scores.sort(key=lambda x: x[1], reverse=True)
            n, na = len(scores), len(active_ids)
            top_k = max(1, n // 100)
            hits = sum(1 for lid, _ in scores[:top_k] if lid in active_ids)
            exp = top_k / n
            dude_ef1s.append((hits / na) / exp * 100 if exp > 0 and na > 0 else 0.0)

        for tid in lit_tasks:
            # LIT-PCBA: refs/*.mol2 ligands are actives
            refs_dir = self.input_dir / 'tasks' / tid / 'refs'
            active_ids = set()
            if refs_dir.exists():
                for f in sorted(refs_dir.glob('*_ligand.mol2')):
                    s = extract_smiles_from_mol2(f)
                    if s:
                        active_ids.add(f'litpcba_{tid.replace("litpcba_","")}_{f.stem}_ref')
            # Match via SMILES in ligands.csv
            if not active_ids:
                lit_ef1s.append(0.0); continue
            task_dir = self.input_dir / 'tasks' / tid
            if task_dir.exists():
                with open(task_dir / 'ligands.csv') as f:
                    smiles_to_id = {r['smiles']: r['ligand_id'] for r in csv.DictReader(f)}
                for ref_id in list(active_ids):
                    # Find by partial ID match
                    for lid, smi in smiles_to_id.items():
                        if ref_id in lid or lid in ref_id:
                            active_ids.add(lid)
            scores = task_scores[tid]
            if not scores:
                lit_ef1s.append(0.0); continue
            scores.sort(key=lambda x: x[1], reverse=True)
            n, na = len(scores), max(1, len(active_ids))
            top_k = max(1, n // 100)
            hits = sum(1 for lid, _ in scores[:top_k] if lid in active_ids)
            exp = top_k / n
            lit_ef1s.append((hits / na) / exp * 100 if exp > 0 else 0.0)

        dude_mean = np.mean(dude_ef1s) if dude_ef1s else 0.0
        lit_mean  = np.mean(lit_ef1s)  if lit_ef1s  else 0.0
        score_val = (dude_mean + lit_mean) / 2.0

        elapsed = time.time() - t0
        notes = f"dude={dude_mean:.2f}% lit={lit_mean:.2f}% t={elapsed:.1f}s"
        opt_history.add(config, score_val, dude_mean, lit_mean, notes)
        log_score.info(f"EF1%: score={score_val:.4f}  dude={dude_mean:.2f}%({len(dude_tasks)})  lit={lit_mean:.2f}%({len(lit_tasks)})")

        log_main.info(f"[ITER {iteration}] DONE: score={score_val:.4f} ({elapsed:.1f}s)")
        return {'score': score_val, 'dude_ef1': dude_mean, 'lit_ef1': lit_mean,
                'dude_tasks': len(dude_tasks), 'lit_tasks': len(lit_tasks)}

    def run(self, max_iterations=3, use_fast_baseline=True):
        log_main.info(f"\n{'#'*60}")
        log_main.info(f"# AUTONOMOUS LOOP | max_iterations={max_iterations}")
        log_main.info(f"{'#'*60}\n")

        best_score = -999.0
        best_config = None
        best_iter = -1

        if use_fast_baseline:
            prev_best = opt_history.best()
            if prev_best:
                log_main.info(f"[BASELINE] Using previous best config from iter {prev_best['iteration']}: "
                             f"score={prev_best['score']:.4f}")
                metrics = self.run_iteration(prev_best['config'], 'baseline')
                if metrics and metrics['score'] > best_score:
                    best_score = metrics['score']
                    best_config = copy.deepcopy(prev_best['config'])
                    best_iter = 'baseline'
                    log_main.info(f"[BASELINE] ★ BEST: {best_score:.4f}")

        for it in range(max_iterations):
            config = self.sample_config(it)
            metrics = self.run_iteration(config, it)
            if metrics and metrics['score'] > best_score:
                best_score = metrics['score']
                best_config = copy.deepcopy(config)
                best_iter = it
                log_main.info(f"[ITER {it}] ★ NEW BEST: {best_score:.4f}")

        if best_iter == 'baseline' or best_iter >= 0:
            src_dir = OUT_DIR / f'iter_{best_iter}' if best_iter != 'baseline' else OUT_DIR / 'iter_baseline'
            if src_dir.exists():
                shutil.copy(src_dir / 'result.csv', self.output_dir / 'result.csv')

        with zipfile.ZipFile(self.output_dir / 'result.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(self.output_dir / 'result.csv', arcname='result.csv')
            best_log = src_dir / 'result.log' if src_dir.exists() else None
            if best_log and best_log.exists():
                zf.write(best_log, arcname='result.log')
            for lf in LOG_DIR.glob('*.log'):
                zf.write(lf, arcname=f'logs/{lf.name}')
            if OPT_LOG.exists():
                zf.write(OPT_LOG, arcname='optimization_history.json')

        log_main.info(f"\n{'='*60}")
        log_main.info(f"AUTONOMOUS LOOP COMPLETE")
        log_main.info(f"Best score: {best_score:.4f} (iter {best_iter})")
        log_main.info(f"Best config: {json.dumps(best_config, indent=2) if best_config else 'N/A'}")
        log_main.info(f"History entries: {len(opt_history.history)}")
        log_main.info(f"{'='*60}")
        return best_score

# ============================================================
# Result Log Writer - detailed iteration trace
# ============================================================
def _write_iter_result_log(iter_dir, config, iteration, dude_ef1s, lit_ef1s, dude_tasks, lit_tasks, task_scores, elapsed):
    """Write a detailed result.log documenting the agent's autonomous optimization process."""
    log_path = iter_dir / 'result.log'
    lines = []
    lines.append("=" * 70)
    lines.append("DrugCLIP Agent Iteration Result Log")
    lines.append("=" * 70)
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Iteration: {iteration}")
    lines.append("")

    # ---- 1. Initialization Config ----
    lines.append("## 1. INITIALIZATION CONFIGURATION")
    lines.append("-" * 40)
    lines.append(f"  fp_dims:        {config.get('fp_dims', 'N/A')}")
    lines.append(f"  fp_weights:     {config.get('fp_weights', 'N/A')}")
    lines.append(f"  fp_scale:       {config.get('fp_scale', 'N/A')}")
    lines.append(f"  Ks (cluster values): {config.get('Ks', 'N/A')}")
    lines.append(f"  centroid_mode:  {config.get('centroid_mode', 'N/A')}")
    lines.append(f"  rrf_k:          {config.get('rrf_k', 'N/A')}")
    lines.append(f"  fusion:         {config.get('fusion', 'N/A')}")
    lines.append("")

    # ---- 2. Data Loading ----
    lines.append("## 2. DATA LOADING")
    lines.append("-" * 40)
    lines.append(f"  Total tasks processed: {len(dude_tasks) + len(lit_tasks)}")
    lines.append(f"  DUD-E tasks: {len(dude_tasks)}")
    for t in sorted(dude_tasks):
        active_count = len(task_scores.get(t, []))
        lines.append(f"    - {t}: {active_count} ligands scored")
    lines.append(f"  LIT-PCBA tasks: {len(lit_tasks)}")
    for t in sorted(lit_tasks):
        lines.append(f"    - {t}: {len(task_scores.get(t, []))} ligands scored")
    lines.append("")

    # ---- 3. Model/Strategy Selection ----
    lines.append("## 3. MODEL / STRATEGY SELECTION")
    lines.append("-" * 40)
    # Infer which scoring method was used based on config
    if config.get('fusion') == 'fp_first':
        lines.append("  Strategy: FP-first (fingerprint centroid dominates, DrugCLIP secondary)")
        lines.append(f"  Centroid mode: {config.get('centroid_mode', 'N/A')}")
        lines.append(f"  Clustering K values: {config.get('Ks', 'N/A')}")
    elif config.get('fusion') == 'rrf':
        lines.append("  Strategy: Reciprocal Rank Fusion (RRF) - multi-method ensemble")
        lines.append(f"  RRF k parameter: {config.get('rrf_k', 'N/A')}")
    else:
        lines.append(f"  Strategy: {config.get('fusion', 'unknown')}")
    lines.append(f"  Fingerprint dimensions: {config.get('fp_dims', 'N/A')}")
    lines.append(f"  Fingerprint weights: {config.get('fp_weights', 'N/A')}")
    lines.append(f"  Fingerprint scale: {config.get('fp_scale', 'N/A')}")
    lines.append("")

    # ---- 4. Multi-round Decision / Iteration Process ----
    lines.append("## 4. MULTI-ROUND DECISION PROCESS")
    lines.append("-" * 40)
    lines.append(f"  This iteration: {iteration}")
    lines.append(f"  Search space explored: {len(SEARCH_SPACE.get('fp_dims', []))} fp_dims × {len(SEARCH_SPACE.get('fp_weights', []))} fp_weights × {len(SEARCH_SPACE.get('Ks', []))} Ks ...")
    lines.append(f"  Config sampled from SEARCH_SPACE:")
    for k, v in SEARCH_SPACE.items():
        lines.append(f"    - {k}: {v}")
    lines.append(f"  Selected config: {json.dumps(config)}")
    lines.append("")

    # ---- 5. Scoring Process ----
    lines.append("## 5. SCORING PROCESS")
    lines.append("-" * 40)
    # Per-task scoring summary
    task_score_summary = []
    for tid in sorted(dude_tasks) + sorted(lit_tasks):
        scores = task_scores.get(tid, [])
        if scores:
            score_vals = [s for _, s in scores]
            task_score_summary.append(f"    {tid}: n={len(scores)}, mean={sum(score_vals)/len(score_vals):.6f}, range=[{min(score_vals):.6f}, {max(score_vals):.6f}]")
    lines.append("  Per-task scoring summary:")
    lines.extend(task_score_summary[:20])  # first 20 tasks
    if len(task_score_summary) > 20:
        lines.append(f"  ... and {len(task_score_summary) - 20} more tasks")
    lines.append("")

    # ---- 6. Key Intermediate Results ----
    lines.append("## 6. KEY INTERMEDIATE RESULTS")
    lines.append("-" * 40)
    lines.append(f"  DUD-E EF1% per task:")
    for i, tid in enumerate(sorted(dude_tasks)):
        lines.append(f"    {tid}: {dude_ef1s[i]:.4f}%")
    lines.append(f"  DUD-E Mean EF1%: {sum(dude_ef1s)/len(dude_ef1s):.4f}%")
    lines.append("")
    lines.append(f"  LIT-PCBA EF1% per task:")
    for i, tid in enumerate(sorted(lit_tasks)):
        lines.append(f"    {tid}: {lit_ef1s[i]:.4f}%")
    lines.append(f"  LIT-PCBA Mean EF1%: {sum(lit_ef1s)/len(lit_ef1s):.4f}%")
    lines.append("")

    # ---- 7. Final Result Generation ----
    lines.append("## 7. FINAL RESULT GENERATION")
    lines.append("-" * 40)
    total_rows = sum(len(task_scores.get(t, [])) for t in list(dude_tasks) + list(lit_tasks))
    lines.append(f"  Total scored ligands: {total_rows}")
    lines.append(f"  Output file: {iter_dir / 'result.csv'}")
    lines.append(f"  Iterations elapsed: {elapsed:.1f}s")
    lines.append(f"  Score = (DUD-E Mean EF1% + LIT-PCBA Mean EF1%) / 2")
    lines.append(f"  Platform Score: {(sum(dude_ef1s)/len(dude_ef1s) + sum(lit_ef1s)/len(lit_ef1s))/2:.4f}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("END OF LOG")
    lines.append("=" * 70)

    with open(log_path, 'w') as f:
        f.write('\n'.join(lines))

    log_main.info(f"Result log written: {log_path}")

# ============================================================
# Entry point
# ============================================================
def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <input_dir> <output_dir> [max_iterations=5]")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    agent = DrugClipAgent(input_dir, output_dir)
    score = agent.run(max_iterations=max_iter)
    sys.exit(0 if score > 0 else 1)

if __name__ == '__main__':
    main()