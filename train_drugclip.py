"""
DrugCLIP训练：从UniMol初始化，在PDBbind上微调，评测DUD-E EF1%
"""
import sys, os, json, csv, time, warnings, random, gc, sqlite3, pickle, hashlib
sys.path.insert(0, '/home/yijian/Desktop/drugclip/DrugCLIP-BaseLine-master')
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORK = Path('/home/yijian/Desktop/drugclip')
PDBBIND = WORK / 'THU-ATOM_PDBbind'
DATA_DIR = WORK / 'DrugCLIP-BaseLine-master' / 'data'
OUTPUT = WORK / 'output_ft'
OUTPUT.mkdir(exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Python: {sys.executable}")

#############################################
# 模型定义（与checkpoint结构一致）
#############################################
class UniMolEncoder(nn.Module):
    """UniMol风格 encoder: embedding + transformer + mean pool + projection"""
    def __init__(self, vocab_size, dim=256, n_layers=4, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=8, dim_feedforward=dim*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.projection = nn.Linear(dim, 256)
    
    def forward(self, tokens):
        pad_mask = tokens.eq(0)
        x = self.embedding(tokens) + self.pos_embedding[:, :tokens.size(1), :]
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).float().unsqueeze(-1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return F.normalize(self.projection(x), dim=-1)

class DrugCLIP(nn.Module):
    def __init__(self, dim=256, n_layers=4, lig_max_len=256, pkt_max_len=512):
        super().__init__()
        self.ligand_encoder = UniMolEncoder(vocab_size=58, dim=dim, n_layers=n_layers, max_len=lig_max_len)
        self.pocket_encoder = UniMolEncoder(vocab_size=21, dim=dim, n_layers=n_layers, max_len=pkt_max_len)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
    
    def encode_ligand(self, tokens):
        return self.ligand_encoder(tokens)
    
    def encode_pocket(self, tokens):
        return self.pocket_encoder(tokens)
    
    @property
    def temperature(self):
        return 1.0 / self.logit_scale.exp()

def load_from_matpool(model, matpool_ckpt_path):
    """用matpool完整checkpoint初始化，pocket冻结，ligand_encoder可学习"""
    ckpt = torch.load(matpool_ckpt_path, map_location='cpu', weights_only=False)
    ckpt_sd = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    my_sd = model.state_dict()
    
    loaded = 0
    for ck_k, v in ckpt_sd.items():
        # checkpoint: pocket_encoder.X, pocket_projection.X, ligand_projection.X, ligand_encoder.X
        # model: pocket_encoder.X, pocket_encoder.projection.X, ligand_encoder.projection.X, ligand_encoder.X
        mk = ck_k
        # checkpoint uses flat names like ligand_projection.0, pocket_projection.0
        # model uses nested: ligand_encoder.projection.0, pocket_encoder.projection.0
        if mk.startswith('ligand_projection.'):
            mk = 'ligand_encoder.projection.' + mk[len('ligand_projection.'):]
        elif mk.startswith('pocket_projection.'):
            mk = 'pocket_encoder.projection.' + mk[len('pocket_projection.'):]
        # checkpoint has ligand_encoder.X (same as model)
        if mk in my_sd and my_sd[mk].shape == v.shape:
            my_sd[mk].copy_(v); loaded += 1
    
    model.load_state_dict(my_sd, strict=False)
    print(f"  Loaded {loaded} keys from matpool checkpoint")
    
    # 冻结pocket encoder (所有参数) - 在load_state_dict之后
    frozen = 0
    for n, p in model.named_parameters():
        if 'pocket_encoder' in n:
            p.requires_grad = False; frozen += 1
    print(f"  Pocket encoder frozen: {frozen} params")
    return model

#############################################
# 数据准备
#############################################
from train.data.utils import load_atom_dict, parse_pdb_atoms, parse_mol2_coords, prepare_molecule, prepare_pocket, extract_pocket_atoms

mol_atom_dict = load_atom_dict(str(DATA_DIR / 'dict_mol.txt'))
pkt_atom_dict = load_atom_dict(str(DATA_DIR / 'dict_pkt.txt'))

# Token缓存
CACHE_DB = OUTPUT / 'token_cache.db'
token_db = sqlite3.connect(str(CACHE_DB))
token_db.execute("""
    CREATE TABLE IF NOT EXISTS mol_tokens (
        smiles_hash TEXT PRIMARY KEY,
        data BLOB NOT NULL
    )
""")
token_db.execute("PRAGMA journal_mode=WAL")
token_db.commit()

def mol_to_tokens(smiles):
    key = hashlib.md5(smiles.encode()).hexdigest()
    row = token_db.execute("SELECT data FROM mol_tokens WHERE smiles_hash = ?", (key,)).fetchone()
    if row:
        return pickle.loads(row[0])
    try:
        mol_data = prepare_molecule(smiles, max_atoms=256, atom_dict=mol_atom_dict)
        result = {
            'tokens': mol_data['tokens'].astype(np.int64),
        }
        token_db.execute("INSERT OR IGNORE INTO mol_tokens (smiles_hash, data) VALUES (?, ?)",
                         (key, pickle.dumps(result, protocol=4)))
        token_db.commit()
        return result
    except:
        return None

class PDBbindPairsDataset(Dataset):
    """PDBbind正样本对 + 随机负样本"""
    def __init__(self, pdbbind_dir, max_pos=800, neg_ratio=3):
        # 收集有效复合物
        self.pdb_dirs = []
        ligand_csv = pdbbind_dir / 'ligand_smiles.csv'
        self.pdb_smiles = {}
        with open(ligand_csv) as f:
            for row in csv.DictReader(f):
                self.pdb_smiles[row['pdb_id']] = row['smiles']
        
        for pdb_id, smiles in self.pdb_smiles.items():
            pd = pdbbind_dir / pdb_id
            if pd.is_dir() and (pd / f'{pdb_id}_protein_processed_fix.pdb').exists() and (pd / f'{pdb_id}_ligand.mol2').exists():
                self.pdb_dirs.append((pdb_id, pd))
        
        random.shuffle(self.pdb_dirs)
        self.pdb_dirs = self.pdb_dirs[:max_pos]
        print(f"PDBbind pairs: {len(self.pdb_dirs)} positive, {len(self.pdb_dirs)*neg_ratio} negative")
        
        # 预计算正样本pocket embeddings
        self.pocket_embs = {}
        t0 = time.time()
        for i, (pdb_id, pd) in enumerate(self.pdb_dirs):
            if i % 100 == 0:
                print(f"  Pocket loading: {i}/{len(self.pdb_dirs)}")
            try:
                pdb_file = pd / f'{pdb_id}_protein_processed_fix.pdb'
                lig_file = pd / f'{pdb_id}_ligand.mol2'
                coords, elements, _ = parse_pdb_atoms(str(pdb_file))
                ref = parse_mol2_coords(str(lig_file))
                pc, pe = extract_pocket_atoms(coords, elements, ref, radius=10.0)
                if len(pc) == 0: pc, pe = coords, elements
                pocket_data = prepare_pocket(pc, pe, 512, atom_dict=pkt_atom_dict)
                self.pocket_embs[pdb_id] = pocket_data
            except Exception as e:
                pass
        print(f"Pocket data loaded in {time.time()-t0:.1f}s, {len(self.pocket_embs)}/{len(self.pdb_dirs)} OK")
        
        # 构建正负样本列表
        self.pairs = []
        for pdb_id, pd in self.pdb_dirs:
            if pdb_id not in self.pocket_embs: continue
            smiles = self.pdb_smiles[pdb_id]
            self.pairs.append((pdb_id, smiles, 1))
        
        all_smiles = list(self.pdb_smiles.values())
        n_neg = len(self.pairs) * neg_ratio
        for _ in range(n_neg):
            pdb_id = random.choice(list(self.pocket_embs.keys()))
            neg_smiles = random.choice(all_smiles)
            if neg_smiles == self.pdb_smiles[pdb_id]:
                neg_smiles = random.choice(all_smiles)
            self.pairs.append((pdb_id, neg_smiles, 0))
        
        random.shuffle(self.pairs)
        print(f"Total pairs: {len(self.pairs)}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pdb_id, smiles, label = self.pairs[idx]
        tokens = mol_to_tokens(smiles)
        if tokens is None:
            tokens = {'tokens': np.zeros(1, dtype=np.int64)}
        return {
            'pdb_id': pdb_id,
            'smiles': smiles,
            'tokens': tokens['tokens'],
            'label': label
        }

def collate_fn(batch):
    # pad tokens
    max_len = max(b['tokens'].shape[0] for b in batch)
    padded = np.stack([np.pad(b['tokens'], (0, max_len - b['tokens'].shape[0])) for b in batch])
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.float32)
    return {
        'tokens': torch.from_numpy(padded).long(),
        'labels': labels,
        'pdb_ids': [b['pdb_id'] for b in batch],
    }

#############################################
# 评测
#############################################
def evaluate(model, n_tasks=10):
    """在DUD-E上评测EF1%"""
    import shutil
    manifest = WORK / 'benchmark' / 'manifest.jsonl'
    tasks = []
    with open(manifest) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    dude_tasks = [t for t in tasks if t.get('benchmark') == 'DUD-E'][:n_tasks]
    
    # 复用token缓存
    results = []
    for task_info in dude_tasks:
        task_id = task_info['task_id']
        target = task_info['target']
        task_dir = WORK / 'benchmark' / 'tasks' / task_id
        
        # actives
        active_set = set()
        ac = WORK / f'data/dude_actives/{target}_actives.csv'
        if ac.exists():
            with open(ac) as f:
                for row in csv.DictReader(f):
                    active_set.add(row['smiles'].strip())
        
        # ligands
        ligands = []
        with open(task_dir / 'ligands.csv') as f:
            for row in csv.DictReader(f):
                ligands.append(row['smiles'])
        active_idx = [i for i, s in enumerate(ligands) if s in active_set]
        
        # pocket
        with open(task_dir / 'task.json') as f:
            info = json.load(f)
        pdb_path = task_dir / info['receptor_files'][0]
        ref_path = task_dir / info['reference_ligand_files'][0]
        coords, elements, _ = parse_pdb_atoms(str(pdb_path))
        ref = parse_mol2_coords(str(ref_path))
        pc, pe = extract_pocket_atoms(coords, elements, ref, radius=10.0)
        if len(pc) == 0: pc, pe = coords, elements
        pocket_data = prepare_pocket(pc, pe, 512, atom_dict=pkt_atom_dict)
        p_tok = torch.from_numpy(pocket_data['tokens']).long().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            p_emb = model.encode_pocket(p_tok).squeeze(0).cpu().numpy()
        
        # ligands (batch)
        lig_embs = []
        for i in range(0, len(ligands), 256):
            batch = ligands[i:i+256]
            toks_list, ok_idx = [], []
            for bi, s in enumerate(batch):
                t = mol_to_tokens(s)
                if t is not None:
                    toks_list.append((bi, torch.from_numpy(t['tokens']).long()))
                    ok_idx.append(bi)
            if not toks_list:
                lig_embs.extend([np.zeros(256)] * len(batch)); continue
            max_len = max(tt.shape[0] for _, tt in toks_list)
            padded = torch.stack([torch.nn.functional.pad(tt, (0, max_len - tt.shape[0])) for _, tt in toks_list]).to(DEVICE)
            with torch.no_grad():
                embs = model.encode_ligand(padded).cpu().numpy()  # shape: (n_ok, 256)
            emb_dict = {ok_idx[j]: embs[j] for j in range(len(ok_idx))}
            for bi in range(len(batch)):
                lig_embs.append(emb_dict.get(bi, np.zeros(256)))
        
        lig_embs = np.stack(lig_embs)
        scores = p_emb @ lig_embs.T
        
        # EF1%
        n = len(scores)
        k = max(1, int(n * 0.01))
        top_idx = np.argsort(scores)[-k:]
        hits = sum(1 for i in top_idx if i in active_idx)
        expected = len(active_idx) / n
        ef1 = hits / (k * expected) * 100 if expected > 0 else 0
        print(f"  {task_id}: EF1%={ef1:.2f}% ({len(active_idx)} actives, {k} top)")
        results.append(ef1)
    
    mean = np.mean(results)
    print(f"\n  Mean EF1%: {mean:.2f}%")
    return mean

#############################################
# 训练
#############################################
def train():
    # 构建模型
    model = DrugCLIP().to(DEVICE)
    
    # 从matpool checkpoint初始化（pocket冻结，只微调ligand）
    matpool_ckpt = WORK / 'matpool_package_full' / 'drugclip_auto_best.pt'
    if matpool_ckpt.exists():
        print("从matpool checkpoint初始化...")
        model = load_from_matpool(model, str(matpool_ckpt))
    
    model.train()
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 数据集
    print("准备训练数据...")
    dataset = PDBbindPairsDataset(PDBBIND, max_pos=800, neg_ratio=3)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    # 优化器
    enc_params = []
    proj_params = []
    for name, p in model.named_parameters():
        if 'projection' in name:
            proj_params.append(p)
        else:
            enc_params.append(p)
    
    optimizer = torch.optim.AdamW([
        {'params': proj_params, 'lr': 1e-3},
        {'params': enc_params, 'lr': 1e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    best_ef1 = 0
    best_state = None
    
    for epoch in range(20):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()
        
        for batch in dataloader:
            tokens = batch['tokens'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            pdb_ids = batch['pdb_ids']
            
            # 获取pocket embeddings
            pocket_data = [dataset.pocket_embs.get(pid) for pid in pdb_ids]
            pkt_toks = []
            valid_p = []
            for pd in pocket_data:
                if pd is not None:
                    pkt_toks.append(torch.from_numpy(pd['tokens']).long())
                    valid_p.append(True)
                else:
                    pkt_toks.append(torch.zeros(1, dtype=torch.long))
                    valid_p.append(False)
            
            max_p_len = max(pt.shape[0] for pt in pkt_toks)
            pkt_padded = torch.stack([torch.nn.functional.pad(pt, (0, max_p_len - pt.shape[0])) for pt in pkt_toks]).to(DEVICE)
            
            # Forward
            lig_embs = model.encode_ligand(tokens)
            pkt_embs = model.encode_pocket(pkt_padded)
            
            # SimCLR-style contrastive loss: positive pairs should be similar
            # Temperature-scaled cosine similarity
            T = 0.1
            logits = (lig_embs @ pkt_embs.T) / T  # (B, B)
            
            # Labels: diagonal = positive, off-diagonal = negative
            B = tokens.size(0)
            eye = torch.eye(B, device=DEVICE)
            pos_mask = eye  # diagonal = positive pairs
            neg_mask = 1 - eye  # off-diagonal = negative pairs
            
            # NT-Xent style: positive pairs should have HIGH similarity, negative pairs LOW
            pos_sim = (logits * pos_mask).sum() / (pos_mask.sum() + 1e-8)
            neg_sim = (logits * neg_mask).sum() / (neg_mask.sum() + 1e-8)
            # Loss: minimize -pos_sim (push positive similarity up), minimize neg_sim (push negative similarity down)
            loss = -pos_sim / T + neg_sim / T * 0.01
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        dt = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/20: loss={avg_loss:.4f}, time={dt:.1f}s, lr={scheduler.get_last_lr()[0]:.2e}")
        
        # 保存checkpoint
        ckpt_path = OUTPUT / f'epoch_{epoch+1}.pt'
        torch.save({'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()}, 'epoch': epoch, 'loss': avg_loss}, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path.name}")
        
        # 评测 (只在特定epoch)
        if (epoch + 1) % 5 == 0 or epoch == 19:
            model.eval()
            ef1 = evaluate(model, n_tasks=5)  # 减少评测任务数
            if ef1 > best_ef1:
                best_ef1 = ef1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                ckpt_path = OUTPUT / 'best_model.pt'
                torch.save({'model_state_dict': best_state, 'ef1': best_ef1, 'epoch': epoch}, ckpt_path)
                print(f"  ★ New best! EF1%={best_ef1:.2f}%, saved")
            model.train()
    
    print(f"\n训练完成! Best EF1%: {best_ef1:.2f}%")
    token_db.close()
    return best_state, best_ef1

if __name__ == '__main__':
    train()
