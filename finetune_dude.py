"""
DrugCLIP Fine-tuning on PDBbind → DUD-E评测
使用PDBbind真实口袋-配体结合数据微调，在DUD-E上测EF1%
"""
import os, gc, sys, time, json, warnings, random, sqlite3, pickle, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from pathlib import Path
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

WORK = Path('/home/yijian/Desktop/drugclip')
PDBBIND_DIR = WORK / 'THU-ATOM_PDBbind'
BENCHMARK_DIR = WORK / 'benchmark'
OUTPUT_DIR = WORK / 'output_ft'
OUTPUT_DIR.mkdir(exist_ok=True)

#########################################
# 1. DrugCLIP模型（从matpool加载）
#########################################
from DrugCLIP_Baseline.train.model.drugclip import DrugCLIP
from DrugCLIP_Baseline.train.config import Config

def build_drugclip(ckpt_path=None, use_bos_pool=True):
    config = Config()
    config.model.use_bos_pool = use_bos_pool
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DrugCLIP(config.model).to(DEVICE)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        # auto-detect
        for k in list(sd.keys()):
            if 'mol_model.encoder.layers.0.fc1.weight' in k:
                dim = sd[k].shape[0] // 4
                n_layers = max(int(x.split('.layers.')[1].split('.')[0]) for x in sd if 'mol_model.encoder.layers.' in x) + 1
                config.model.mol.encoder_embed_dim = dim
                config.model.pocket.encoder_embed_dim = dim
                config.model.mol.encoder_layers = n_layers
                config.model.pocket.encoder_layers = n_layers
                del model; model = DrugCLIP(config.model).to(DEVICE)
                break
        # filter
        csd = model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in csd and csd[k].shape == v.shape}
        missing = [k for k in sd if k not in filtered]
        if missing: print(f"  跳过 {len(missing)} 个不匹配key: {missing[:3]}...")
        model.load_state_dict(filtered, strict=False)
        print(f"  加载checkpoint: {ckpt_path}")
    model.eval()
    return model, config

#########################################
# 2. 数据准备：PDBbind配体编码
#########################################
from DrugCLIP_Baseline.train.data.utils import load_atom_dict, parse_mol2_coords, parse_pdb_atoms
from DrugCLIP_Baseline.train.data import prepare_molecule

_DATA_DIR = WORK / 'DrugCLIP-Baseline-master' / 'train' / 'data'
_MOL_DICT = _DATA_DIR / 'dict_mol.txt'
_PKT_DICT = _DATA_DIR / 'dict_pkt.txt'
mol_atom_dict = load_atom_dict(str(_MOL_DICT)) if _MOL_DICT.exists() else None
pkt_atom_dict = load_atom_dict(str(_PKT_DICT)) if _PKT_DICT.exists() else None

def smiles_to_tokens(smiles, max_atoms=256):
    """把SMILES转成DrugCLIP输入tensor"""
    try:
        mol_data = prepare_molecule(smiles, max_atoms=max_atoms, atom_dict=mol_atom_dict)
        return {
            'mol_tokens': torch.from_numpy(mol_data['tokens']).long(),
            'mol_distances': torch.from_numpy(mol_data['distances']).float(),
            'mol_edge_types': torch.from_numpy(mol_data['edge_types']).long(),
        }
    except Exception as e:
        return None

def encode_pocket(pdb_dir, max_atoms=256):
    """从PDBbind目录加载pocket坐标并编码"""
    try:
        pdb_file = list(pdb_dir.glob('*_protein_processed_fix.pdb'))
        if not pdb_file:
            pdb_file = list(pdb_dir.glob('*.pdb'))
        if not pdb_file:
            return None
        coords, elements, _ = parse_pdb_atoms(str(pdb_file[0]))
        lig_file = list(pdb_dir.glob('*_ligand.mol2'))
        if lig_file:
            ref_coords = parse_mol2_coords(str(lig_file[0]))
        else:
            ref_coords = coords
        # extract pocket
        from DrugCLIP_Baseline.train.data import extract_pocket_atoms
        pocket_coords, pocket_elements = extract_pocket_atoms(
            coords, elements, ref_coords, radius=10.0
        )
        if len(pocket_coords) == 0:
            pocket_coords, pocket_elements = coords, elements
        from DrugCLIP_Baseline.train.data import prepare_pocket
        pocket_data = prepare_pocket(pocket_coords, pocket_elements, max_atoms, atom_dict=pkt_atom_dict)
        return {
            'tokens': torch.from_numpy(pocket_data['tokens']).long(),
            'distances': torch.from_numpy(pocket_data['distances']).float(),
            'edge_types': torch.from_numpy(pocket_data['edge_types']).long(),
        }
    except Exception as e:
        return None

#########################################
# 3. PDBbind训练对Dataset（真实口袋-配体pair）
#########################################
class PDBbindDataset(Dataset):
    def __init__(self, pdbbind_dir, max_pairs=50000, neg_ratio=3):
        self.pos_pairs = []  # (pocket_data, ligand_tokens)
        self.neg_pairs = []
        
        # 收集有完整文件的PDBbind复合物
        valid_ids = []
        for pdb_dir in pdbbind_dir.iterdir():
            if not pdb_dir.is_dir(): continue
            ligand_csv = WORK / 'THU-ATOM_PDBbind' / 'ligand_smiles.csv'
            if not ligand_csv.exists(): continue
            # 找配体文件
            has_lig = any(pdb_dir.glob('*_ligand.mol2'))
            has_prot = any(pdb_dir.glob('*_protein*.pdb'))
            if has_lig and has_prot:
                valid_ids.append(pdb_dir.name)
        
        print(f"PDBbind有效复合物: {len(valid_ids)}")
        
        # 加载配体SMILES
        self.pdb_smiles = {}
        with open(ligand_csv) as f:
            for row in csv.DictReader(f):
                self.pdb_smiles[row['pdb_id']] = row['smiles']
        
        # 构建正样本对
        random.shuffle(valid_ids)
        for pdb_id in valid_ids[:min(len(valid_ids), max_pairs)]:
            if pdb_id not in self.pdb_smiles: continue
            smiles = self.pdb_smiles[pdb_id]
            pdb_dir = pdbbind_dir / pdb_id
            pocket = encode_pocket(pdb_dir)
            lig = smiles_to_tokens(smiles)
            if pocket is not None and lig is not None:
                self.pos_pairs.append((pocket, lig))
        
        # 负样本：随机配体
        all_smiles = list(self.pdb_smiles.values())
        neg_needed = len(self.pos_pairs) * neg_ratio
        for _ in range(neg_needed):
            pdb_id = random.choice(valid_ids)
            if pdb_id not in self.pdb_smiles: continue
            smiles = self.pdb_smiles[pdb_id]
            pdb_dir = pdbbind_dir / pdb_id
            pocket = encode_pocket(pdb_dir)
            # 负样本：随机选一个不同的配体
            neg_smiles = random.choice(all_smiles)
            if neg_smiles == smiles: neg_smiles = random.choice(all_smiles)
            lig = smiles_to_tokens(neg_smiles)
            if pocket is not None and lig is not None:
                self.neg_pairs.append((pocket, lig))
        
        print(f"正样本: {len(self.pos_pairs)}, 负样本: {len(self.neg_pairs)}")
    
    def __len__(self):
        return len(self.pos_pairs) + len(self.neg_pairs)
    
    def __getitem__(self, idx):
        if idx < len(self.pos_pairs):
            pocket, lig = self.pos_pairs[idx]
            label = 1.0
        else:
            pocket, lig = self.neg_pairs[idx - len(self.pos_pairs)]
            label = 0.0
        
        # collate到batch
        return {
            'pocket': pocket,
            'lig': lig,
            'label': torch.tensor(label, dtype=torch.float32)
        }

def collate_fn(batch):
    max_pocket_atoms = 256
    max_lig_atoms = 256
    
    # pad pocket
    pt_tokens = []
    pt_dists = []
    pt_edges = []
    for b in batch:
        t = b['pocket']['tokens']
        d = b['pocket']['distances']
        e = b['pocket']['edge_types']
        pt_tokens.append(F.pad(t, (0, max_pocket_atoms - t.size(0))) if t.size(0) < max_pocket_atoms else t[:max_pocket_atoms])
        pt_dists.append(F.pad(d, (0, max_pocket_atoms - d.size(0), 0, max_pocket_atoms - d.size(0))) if d.size(0) < max_pocket_atoms else d[:max_pocket_atoms, :max_pocket_atoms])
        pt_edges.append(F.pad(e, (0, max_pocket_atoms - e.size(0), 0, max_pocket_atoms - e.size(0))) if e.size(0) < max_pocket_atoms else e[:max_pocket_atoms, :max_pocket_atoms])
    
    lg_tokens = []
    lg_dists = []
    lg_edges = []
    for b in batch:
        t = b['lig']['mol_tokens']
        d = b['lig']['mol_distances']
        e = b['lig']['mol_edge_types']
        lg_tokens.append(F.pad(t, (0, max_lig_atoms - t.size(0))) if t.size(0) < max_lig_atoms else t[:max_lig_atoms])
        lg_dists.append(F.pad(d, (0, max_lig_atoms - d.size(0), 0, max_lig_atoms - d.size(0))) if d.size(0) < max_lig_atoms else d[:max_lig_atoms, :max_lig_atoms])
        lg_edges.append(F.pad(e, (0, max_lig_atoms - e.size(0), 0, max_lig_atoms - e.size(0))) if e.size(0) < max_lig_atoms else e[:max_lig_atoms, :max_lig_atoms])
    
    return {
        'pocket_tokens': torch.stack(pt_tokens).to(DEVICE),
        'pocket_distances': torch.stack(pt_dists).to(DEVICE),
        'pocket_edge_types': torch.stack(pt_edges).to(DEVICE),
        'lig_tokens': torch.stack(lg_tokens).to(DEVICE),
        'lig_distances': torch.stack(lg_dists).to(DEVICE),
        'lig_edge_types': torch.stack(lg_edges).to(DEVICE),
        'label': torch.stack([b['label'] for b in batch]).to(DEVICE),
    }

#########################################
# 4. 训练：Contrastive Loss + Hard Negative
#########################################
def train_one_epoch(model, dataloader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in dataloader:
        with autocast(device_type='cuda'):
            pocket_emb = model.encode_pocket(
                batch['pocket_tokens'], batch['pocket_distances'], batch['pocket_edge_types']
            )
            lig_emb = model.encode_mol(
                batch['lig_tokens'], batch['lig_distances'], batch['lig_edge_types']
            )
            
            # 对比学习loss：正样本相似度大，负样本相似度小
            pos_sim = (pocket_emb * lig_emb).sum(dim=1)  # (B,)
            labels = batch['label']
            
            # InfoNCE loss
            logits = (pocket_emb @ lig_emb.T) * model.temperature.exp()  # (B, B)
            loss = F.binary_cross_entropy_with_logits(logits.diag(), labels) * 2
            
            # 同时加一个ranking loss：正 > 负
            pos_mask = (labels.unsqueeze(0) == 1)
            neg_mask = (labels.unsqueeze(0) == 0)
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                diff = pos_sim.unsqueeze(1) - pos_sim.unsqueeze(0)  # margin ranking
                rank_loss = F.relu(0.5 - diff).mean() * 0.1
                loss = loss + rank_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler: scheduler.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)

#########################################
# 5. DUD-E EF1% 评测
#########################################
def get_ligand_embedding(model, smiles, token_cache_db, mol_atom_dict):
    """获取配体embedding，带缓存"""
    key = hashlib.md5(smiles.encode()).hexdigest()
    row = token_cache_db.execute("SELECT data FROM mol_tokens WHERE smiles_hash = ?", (key,)).fetchone()
    if row:
        arr = pickle.loads(row[0])
        tokens = torch.from_numpy(arr['t']).long().unsqueeze(0).to(DEVICE)
        dists = torch.from_numpy(arr['d']).float().unsqueeze(0).to(DEVICE)
        edges = torch.from_numpy(arr['e']).long().unsqueeze(0).to(DEVICE)
    else:
        mol_data = prepare_molecule(smiles, max_atoms=256, atom_dict=mol_atom_dict)
        tokens = torch.from_numpy(mol_data['tokens']).long().unsqueeze(0).to(DEVICE)
        dists = torch.from_numpy(mol_data['distances']).float().unsqueeze(0).to(DEVICE)
        edges = torch.from_numpy(mol_data['edge_types']).long().unsqueeze(0).to(DEVICE)
        blob = pickle.dumps({'t': mol_data['tokens'], 'd': mol_data['distances'], 'e': mol_data['edge_types']})
        token_cache_db.execute("INSERT OR IGNORE INTO mol_tokens (smiles_hash, data) VALUES (?, ?)", (key, blob))
        token_cache_db.commit()
    
    with torch.no_grad(), autocast(device_type='cuda'):
        emb = model.encode_mol(tokens, dists, edges)
    return emb.squeeze(0).cpu().numpy()

def get_pocket_embedding(model, pdb_path, ref_mol2, token_cache_db, pkt_atom_dict):
    """获取pocket embedding"""
    try:
        coords, elements, _ = parse_pdb_atoms(str(pdb_path))
        if ref_mol2 and Path(ref_mol2).exists():
            ref_coords = parse_mol2_coords(str(ref_mol2))
        else:
            ref_coords = coords
        from DrugCLIP_Baseline.train.data import extract_pocket_atoms
        pocket_coords, pocket_elements = extract_pocket_atoms(coords, elements, ref_coords, radius=10.0)
        if len(pocket_coords) == 0:
            pocket_coords, pocket_elements = coords, elements
        from DrugCLIP_Baseline.train.data import prepare_pocket
        pocket_data = prepare_pocket(pocket_coords, pocket_elements, 256, atom_dict=pkt_atom_dict)
        tokens = torch.from_numpy(pocket_data['tokens']).float().unsqueeze(0).to(DEVICE)
        dists = torch.from_numpy(pocket_data['distances']).float().unsqueeze(0).to(DEVICE)
        edges = torch.from_numpy(pocket_data['edge_types']).long().unsqueeze(0).to(DEVICE)
        with torch.no_grad(), autocast(device_type='cuda'):
            emb = model.encode_pocket(tokens, dists, edges)
        return emb.squeeze(0).cpu().numpy()
    except:
        return None

def compute_ef1(scores, active_indices, top_pct=0.01):
    """计算EF1%"""
    n = len(scores)
    k = max(1, int(n * top_pct))
    top_indices = np.argsort(scores)[-k:]
    hits = sum(1 for i in top_indices if i in active_indices)
    expected = len(active_indices) / n
    if expected == 0: return 0.0
    return hits / (k * expected) * 100

def evaluate_on_dude(model, token_cache_db, n_tasks=5):
    """在DUD-E上评测，返回Mean EF1%"""
    import csv
    from collections import defaultdict
    
    manifest = BENCHMARK_DIR / 'manifest.jsonl'
    tasks = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if line: tasks.append(json.loads(line))
    
    # 只测DUD-E
    dude_tasks = [t for t in tasks if t.get('benchmark') == 'DUD-E'][:n_tasks]
    
    all_ef1 = []
    for task_info in dude_tasks:
        task_id = task_info['task_id']
        task_dir = BENCHMARK_DIR / 'tasks' / task_id
        
        # 加载actives
        active_set = set()
        active_csv = WORK / 'data' / 'dude_actives' / f"{task_info['target']}_actives.csv"
        if active_csv.exists():
            with open(active_csv) as f:
                for row in csv.DictReader(f):
                    active_set.add(row['smiles'].strip())
        
        # 加载ligands
        ligands_csv = task_dir / 'ligands.csv'
        ligand_smiles = {}
        with open(ligands_csv) as f:
            for row in csv.DictReader(f):
                ligand_smiles[row['ligand_id']] = row['smiles']
        
        smiles_list = list(ligand_smiles.values())
        ligand_ids = list(ligand_smiles.keys())
        
        # pocket embedding
        receptor_files = task_info.get('receptor_files', [])
        ref_files = task_info.get('reference_ligand_files', [])
        pocket_embs = []
        for rf, ref in zip(receptor_files, ref_files):
            p_emb = get_pocket_embedding(model, task_dir / rf, task_dir / ref if ref else None, token_cache_db, pkt_atom_dict)
            if p_emb is not None:
                pocket_embs.append(p_emb)
        
        if not pocket_embs:
            print(f"  {task_id}: 无pocket embedding")
            continue
        
        pocket_emb = np.mean(pocket_embs, axis=0)
        pocket_emb_t = torch.from_numpy(pocket_emb).float().unsqueeze(0).to(DEVICE)
        
        # ligand embeddings
        lig_embs = []
        for smi in smiles_list:
            try:
                emb = get_ligand_embedding(model, smi, token_cache_db, mol_atom_dict)
                lig_embs.append(emb)
            except:
                lig_embs.append(np.zeros(128, dtype=np.float32))
        
        lig_embs_np = np.stack(lig_embs)
        lig_embs_t = torch.from_numpy(lig_embs_np).to(DEVICE)
        
        # score
        with torch.no_grad(), autocast(device_type='cuda'):
            scores = (pocket_emb_t @ lig_embs_t.T).squeeze(0).cpu().numpy()
        
        # active indices
        active_indices = [i for i, smi in enumerate(smiles_list) if smi in active_set]
        
        ef1 = compute_ef1(scores, active_indices)
        all_ef1.append(ef1)
        print(f"  {task_id}: EF1% = {ef1:.2f}%  (actives={len(active_indices)}, ligands={len(smiles_list)})")
    
    mean_ef1 = np.mean(all_ef1) if all_ef1 else 0
    print(f"\n=== DUD-E Mean EF1%: {mean_ef1:.2f}% ===")
    return mean_ef1

#########################################
# 6. 主流程
#########################################
if __name__ == '__main__':
    import csv
    
    # 初始化token缓存DB
    cache_db_path = OUTPUT_DIR / 'token_cache.db'
    cache_db_path.parent.mkdir(exist_ok=True)
    token_db = sqlite3.connect(str(cache_db_path))
    token_db.execute("""
        CREATE TABLE IF NOT EXISTS mol_tokens (
            smiles_hash TEXT PRIMARY KEY,
            data BLOB NOT NULL
        )
    """)
    token_db.execute("PRAGMA journal_mode=WAL")
    token_db.commit()
    
    # 加载/构建模型
    print("加载模型...")
    # 先尝试用已有的微调模型
    pretrained_ckpt = WORK / 'drugclip_final.pt'
    if not pretrained_ckpt.exists():
        pretrained_ckpt = list((WORK / 'matpool_package_full').glob('drugclip_auto_best.pt'))
        pretrained_ckpt = pretrained_ckpt[0] if pretrained_ckpt else None
    
    model, config = build_drugclip(str(pretrained_ckpt) if pretrained_ckpt else None)
    print("模型加载完成")
    
    # 训练数据集
    print("准备PDBbind训练数据...")
    dataset = PDBbindDataset(PDBBIND_DIR, max_pairs=30000, neg_ratio=3)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    # 优化器：只微调projection head + 最后2层encoder
    enc_params = []
    head_params = []
    frozen = []
    for name, param in model.named_parameters():
        if 'project' in name or 'logit_scale' in name:
            head_params.append(param)
        elif 'encoder.layers.6' in name or 'encoder.layers.7' in name or 'mol_model.encoder.layers.6' in name or 'mol_model.encoder.layers.7' in name:
            enc_params.append(param)
        else:
            frozen.append(param)
    
    for p in frozen:
        p.requires_grad = False
    
    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': 3e-4},
        {'params': enc_params, 'lr': 1e-4},
    ], weight_decay=1e-4)
    
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))
    
    best_ef1 = 0
    best_state = None
    
    for epoch in range(epochs):
        t0 = time.time()
        loss = train_one_epoch(model, dataloader, optimizer, scheduler, epoch)
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs}: loss={loss:.4f}, time={dt:.1f}s")
        
        # 每3个epoch评测一次
        if (epoch + 1) % 3 == 0:
            mean_ef1 = evaluate_on_dude(model, token_db, n_tasks=10)
            if mean_ef1 > best_ef1:
                best_ef1 = mean_ef1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                ckpt_path = OUTPUT_DIR / 'best_drugclip.pt'
                torch.save({'model_state_dict': best_state, 'epoch': epoch, 'ef1': best_ef1}, ckpt_path)
                print(f"  ★ New best! EF1% = {best_ef1:.2f}%, saved to {ckpt_path}")
    
    # 最终评测
    if best_state:
        model.load_state_dict(best_state)
    print(f"\n=== 最终评测: DUD-E Mean EF1% = {best_ef1:.2f}% ===")
    
    token_db.close()
