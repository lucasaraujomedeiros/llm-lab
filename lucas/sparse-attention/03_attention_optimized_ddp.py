#!/usr/bin/env python3
"""
Treinamento DDP de GPTModel com variantes de atenção otimizadas.
Versão ajustada para estabilidade no ambiente Kaggle.
Uso: !torchrun --nproc_per_node=2 03_attention_optimized_ddp.py [--attention all|PyTorchOptimized|OriginalManual]
"""
import argparse
import math
import os
import pickle
import random
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Crucial para Kaggle
import matplotlib.pyplot as plt
import tiktoken
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.distributed import destroy_process_group, init_process_group, all_reduce, ReduceOp, barrier
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

# ── Kaggle Safety ─────────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Paths ─────────────────────────────────────────────────────────────────
DATASET_TRAIN_PATH = "/kaggle/input/datasets/lucasqueiros/llm-lab-corpus/train.txt"
DATASET_VAL_PATH   = "/kaggle/input/datasets/lucasqueiros/llm-lab-corpus/val.txt"
START_CONTEXT      = "The moment she entered in the house, she found out"

# ── Model config ──────────────────────────────────────────────────────────
GPT_CONFIG_BASE = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 256,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "emb_weight_tying": True,
}

# ── Training config ───────────────────────────────────────────────────────
TRAINING_CONFIG = {
    "num_epochs": 10,
    "learning_rate": 5e-4,       # Mantém igual
    "weight_decay": 0.1,
    "batch_size_per_gpu": 64,    # Aumentado
    "eval_freq": 50,
    "eval_iter": 20,
    "warmup_steps": 100,
    "max_grad_norm": 1.0,
    "accumulation_steps": 1,     # REDUZIDO: Agora atualiza a cada passo
    "use_amp": True,
    "num_workers": 4,            
}

# ── DDP Setup ─────────────────────────────────────────────────────────────
def ddp_setup(rank, world_size):
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355" # Porta padrão segura
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# ── Seeds ─────────────────────────────────────────────────────────────────
def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Tokenizer utilities ───────────────────────────────────────────────────
def load_tokenizer():
    return tiktoken.get_encoding("gpt2")

def _encode_text(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    if hasattr(encoded, "ids"):
        return encoded.ids
    return list(encoded)

def text_to_token_ids(text, tokenizer):
    encoded = _encode_text(text, tokenizer)
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# ── Dataset ───────────────────────────────────────────────────────────────
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = _encode_text(txt, tokenizer)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk  = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# ── Model Components ──────────────────────────────────────────────────────
class BaseAttention(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_coeff", torch.tensor(math.sqrt(2.0 / math.pi)))

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(self._coeff * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg, attention_module_class: type[BaseAttention], **attention_kwargs):
        super().__init__()
        self.att = attention_module_class(**attention_kwargs)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg, attention_module_class: type[BaseAttention]):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(
                cfg, attention_module_class,
                d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                dropout=cfg["drop_rate"],
                num_heads=cfg["n_heads"],
                qkv_bias=cfg["qkv_bias"],
            ) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        if cfg.get("emb_weight_tying", False):
            self.out_head.weight = self.tok_emb.weight

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        if seq_len > self.cfg["context_length"]:
            raise ValueError(f"Sequence ({seq_len}) exceeds context_length ({self.cfg['context_length']})")

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)

# ── Attention Modules ─────────────────────────────────────────────────────
class OriginalSelfAttention(BaseAttention):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout_layer = nn.Dropout(dropout)

        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        keys    = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

class PyTorchMultiHeadAttention(BaseAttention):
    def __init__(self, d_in, d_out, num_heads, context_length=None, dropout=0.0, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv

        use_dropout_prob = self.dropout_layer.p if self.training else 0.0
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout_prob, is_causal=True,
        )
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)
        return self.proj(context_vec)

class LightningIndexer(nn.Module):
    def __init__(self, emb_dim, index_n_heads, index_head_dim):
        super().__init__()

        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim

        self.W_q_index = nn.Linear(emb_dim, index_n_heads * index_head_dim, bias=False)
        self.W_k_index = nn.Linear(emb_dim, index_head_dim, bias=False)
        self.W_weights_index = nn.Linear(emb_dim, index_n_heads, bias=False)

        self.scale = index_head_dim ** -0.5

    def forward(self, x, x_ctx, k, causal_mask_bool=None):
        b = x.shape[0]
        T = x.shape[1]
        S = x_ctx.shape[1]

        #queries.shape = (b, T, index_n_heads, index_head_dim)
        queries = self.W_q_index(x).view(b, T, self.index_n_heads, self.index_head_dim)
        #keys.shape = (b, S, index_head_dim)
        keys = self.W_k_index(x_ctx)
        head_weights = self.W_weights_index(x)

        raw_scores = torch.relu((torch.einsum("bthd,bsd->bths", queries, keys) * self.scale))

        #index_scores.shape = (b, T, S)
        index_scores = torch.einsum("bths,bth->bts", raw_scores, head_weights) * self.index_n_heads ** -0.5 

        if causal_mask is not None:
            index_scores = index_scores.masked_fill_(mask_bool, -torch.inf)

        k = min(S, k)

        #top_k.shape = (b, T, k) -> each token T has a tensor of K positions,
        #each one containing the index of the position i of token Si 
        #on tensor index_scores, sorted by its value on that tensor
        top_k = torch.topk(index_scores, k, dim=-1)
        return top_k.indices





class SparseAttention(BaseAttention):
    def __init__(self, d_in, d_out, num_heads, dropout,
            qkv_bias=False, index_n_heads, index_head_dim,
            topk, use_cache=False):
        super().__init__()

        self.use_cache = use_cache

        assert d_out % num_heads == 0
        self.head_dim = d_out // num_heads
        self.num_heads = num_heads

        self.topk = topk
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out, bias=False)

        self.indexer = LightningIndexer(d_in, index_n_heads, index_head_dim)

        self.dropout = nn.Dropout(dropout)

        self.cache_k = None
        self.cache_v = None
        self.cache_x = None #stores past tokens, to pass as x_ctg to indexer

        self.ptr_current_pos = 0

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None
        self.cache_x = None
        self.ptr_current_pos = 0


    def forward(self, x): # x shape = (b, T, emb_dim)

        b, T, _ = x.shape

        #those matrices will have a shape of (b, num_heads, T, head_dim)
        #because it is (b, T, d_in) * (d_in, d_out) = (b, T, d_out)
        #break d_out in num_heads * head_dim, then transpose -> (b, num_heads, T, head_dim)
        queries = self.W_query(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(b, T, self.num_heads, self.head_dim).transpose(1,2)
        values = self.W_value(x).view(b, T, self.num_heads, self.head_dim).transpose(1,2)


        #inference mode
        if self.use_cache: 
            if self_cache_k:
                new_keys = torch.cat([self.cache_k, keys], dim=-2)
                new_values = torch.cat([self.cache_v, values], dim=-2)
                x_ctx = torch.cat([self.cache_x, x],dim=-2)

            else:
                new_keys = keys #shape = (b, num_heads, T, head_dim)
                new_values = values #shape = (b, num_heads, T, head_dim)
                x_ctx = x #shape = (b, T, d_in)
                #where probably T = 1

            self.cache_k = new_keys
            self.cache_v = new_values
            self.cache_x = x_ctx

        self.ptr_current_pos += T

        #training
        else:
            new_keys = keys #shape = (b, num_heads, T, head_dim)
            new_values = values #shape = (b, num_heads, T, head_dim)
            x_ctx = x #shape = (b, T, d_in)
            #probably T = context_length

            self.ptr_current_pos = 0

        pointer = self.ptr_current_pos

        #attention_scores.shape = (b, num_heads, q, k)
        #because its (b, num_heads, q, head_dim) @ (b, num_heads, head_dim, k)
        attention_scores = (queries @ new_keys.transpose(-2, -1)) * head_dim 
        q_tokens = queries.shape[-2]
        k_tokens = new_keys.shape[-2]

        # causal mask

        q_pos = torch.arange(q_tokens, k_tokens)
        k_pos = torch.arange(k_tokens)

        #mask.shape = (q, k)
        causal_bool_mask = q_pos.unsqueeze(-1) < k_pos

        #top_k_indexes.shape = (b, q, topk), T = q
        tok_k_indexes = self.indexer(x, ctx_x, self.topk, causal_mask_bool=causal_bool_mask)

        sparse_mask = torch.full(
            (b, num_tokens_Q, num_tokens_K), float("-inf"), device=device, dtype=attn_scores.dtype
        )        

        attention_scores = sparse_mask.scatter_(0, top_k_indexes, attention_scores)
        attention_weights = torch.softmax(attention_scores, dim=-1) * head_dim ** -0.5

        #new_values.shape = (b, num_heads, T, head_dim)
        #so (attention_weights * new_values).shape = (b, num_heads, T, head_dim)
        context_vectors = attention_weights @ new_values








# ── Model utilities ───────────────────────────────────────────────────────
def _get_base_model(model):
    m = model
    if isinstance(m, DDP):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m

# ── Training utilities ────────────────────────────────────────────────────
def calc_loss_batch(input_batch, target_batch, model, device, use_amp=False):
    input_batch = input_batch.to(device, non_blocking=True)
    target_batch = target_batch.to(device, non_blocking=True)
    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]

    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
        logits = model(input_batch)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None, use_amp=False, rank=0, world_size=1):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    num_batches = min(num_batches or len(data_loader), len(data_loader))
    count = 0

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device, use_amp=use_amp)
        total_loss += loss.item()
        count += 1

    if count == 0:
        return float("nan")

    avg_loss = total_loss / count
    if world_size > 1:
        loss_tensor = torch.tensor([avg_loss], device=device)
        all_reduce(loss_tensor, op=ReduceOp.AVG)
        avg_loss = loss_tensor.item()

    return avg_loss

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]

            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float("-inf"), device=logits.device), logits)

            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_id is not None and torch.any(idx_next == eos_id):
                break
            idx = torch.cat((idx, idx_next), dim=1)
    return idx

def generate_and_print_sample(model, tokenizer, device, start_context, rank, max_new_tokens=50):
    if rank != 0:
        return
    model.eval()
    base_model = _get_base_model(model)
    context_size = base_model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate(model, idx=encoded, max_new_tokens=max_new_tokens,
                             context_size=context_size, temperature=0.0)
        decoded = token_ids_to_text(token_ids, tokenizer)
        print("-" * 50)
        print(decoded.replace("\n", " "))
        print("-" * 50)
    model.train()

# ── Persistence ───────────────────────────────────────────────────────────
def get_experiment_dir(experiment_name, base_dir="/kaggle/working/experiment_results/"): # AJUSTE: Caminho absoluto do Kaggle
    result_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def _unwrap_model_state_dict(model):
    m = _get_base_model(model)
    return m.state_dict()

def save_experiment_artifacts(result_dir, model, optimizer, epoch, global_step, metrics, rank, final=False):
    if rank != 0:
        return

    checkpoint_name = "final_checkpoint.pth" if final else "latest_checkpoint.pth"
    checkpoint_path = os.path.join(result_dir, checkpoint_name)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": _unwrap_model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {"gpt": GPT_CONFIG_BASE, "training": TRAINING_CONFIG},
    }
    torch.save(checkpoint, checkpoint_path)

    for name, values in metrics.items():
        with open(os.path.join(result_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(values, f)
    return checkpoint_path

# ── DDP Training ──────────────────────────────────────────────────────────
def train_model_ddp(attention_module_class, rank, world_size, experiment_name=None):
    if experiment_name is None:
        experiment_name = attention_module_class.__name__

    device = torch.device("cuda", rank)
    device_type = device.type
    use_amp = TRAINING_CONFIG["use_amp"] and device_type == "cuda"
    accum_steps = TRAINING_CONFIG["accumulation_steps"]
    num_workers = TRAINING_CONFIG["num_workers"]
    per_gpu_batch = TRAINING_CONFIG["batch_size_per_gpu"]

    if rank == 0:
        print(f"Device: {device} | AMP: {use_amp} | Accum steps: {accum_steps}")
        print(f"Per-GPU batch: {per_gpu_batch} | Global batch: {per_gpu_batch * world_size}")
        print(f"Effective batch: {per_gpu_batch * world_size * accum_steps}")

    set_seed(123)
    tokenizer = load_tokenizer()

    with open(DATASET_TRAIN_PATH, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(DATASET_VAL_PATH, "r", encoding="utf-8") as f:
        val_text = f.read()

    ctx = GPT_CONFIG_BASE["context_length"]
    train_dataset = GPTDataset(train_text, tokenizer, ctx, ctx)
    val_dataset   = GPTDataset(val_text, tokenizer, ctx, ctx)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=per_gpu_batch, sampler=train_sampler,
        drop_last=True, num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=per_gpu_batch, sampler=val_sampler,
        drop_last=False, num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # Eval loaders with num_workers=0 for stability in DDP evaluation
    eval_train_loader = DataLoader(
        train_dataset, batch_size=per_gpu_batch,
        sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True),
        drop_last=True, num_workers=0, pin_memory=True,
    )
    eval_val_loader = DataLoader(
        val_dataset, batch_size=per_gpu_batch,
        sampler=DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False),
        drop_last=False, num_workers=0, pin_memory=True,
    )

    model = GPTModel(cfg=GPT_CONFIG_BASE, attention_module_class=attention_module_class)
    model = model.to(device)

    if device_type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            if rank == 0:
                print("torch.compile ativado")
        except Exception as e:
            if rank == 0:
                print(f"torch.compile indisponivel: {e}")

    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
    )

    total_optimizer_steps = (len(train_loader) // accum_steps) * TRAINING_CONFIG["num_epochs"]
    scheduler = get_lr_scheduler(optimizer, TRAINING_CONFIG["warmup_steps"], total_optimizer_steps)
    scaler = GradScaler(enabled=use_amp)

    result_dir = get_experiment_dir(experiment_name)
    train_losses, val_losses, lr_history = [], [], []
    global_step = 0

    for epoch in range(TRAINING_CONFIG["num_epochs"]):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}", disable=(rank != 0))

        for batch_idx, (input_batch, target_batch) in pbar:
            input_batch = input_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_batch)
                loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            is_last_batch = (batch_idx + 1) == len(train_loader)
            if (batch_idx + 1) % accum_steps == 0 or is_last_batch:
                scaler.unscale_(optimizer)
                if TRAINING_CONFIG["max_grad_norm"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            current_lr = scheduler.get_last_lr()[0]
            lr_history.append(current_lr)

            if rank == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item() * accum_steps:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": global_step,
                })

            if global_step % TRAINING_CONFIG["eval_freq"] == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = calc_loss_loader(
                        eval_train_loader, model, device,
                        num_batches=TRAINING_CONFIG["eval_iter"], use_amp=use_amp,
                        rank=rank, world_size=world_size,
                    )
                    val_loss = calc_loss_loader(
                        eval_val_loader, model, device,
                        num_batches=TRAINING_CONFIG["eval_iter"], use_amp=use_amp,
                        rank=rank, world_size=world_size,
                    )
                model.train()

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if rank == 0:
                    print(f"Ep {epoch+1} | Step {global_step:>5} | "
                          f"train={train_loss:.4f} | val={val_loss:.4f} | lr={current_lr:.2e}")
                    generate_and_print_sample(model, tokenizer, device, START_CONTEXT, rank)

        metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "lr_history": lr_history,
        }
        save_experiment_artifacts(
            result_dir, model, optimizer, epoch, global_step, metrics, rank,
            final=(epoch == TRAINING_CONFIG["num_epochs"] - 1),
        )

    if rank == 0:
        generate_and_print_sample(model, tokenizer, device, START_CONTEXT, rank)

    return train_losses, val_losses

# ── Analysis ──────────────────────────────────────────────────────────────
def analyze_results(experiment_names, result_base_dir="/kaggle/working/experiment_results/", rank=0): # AJUSTE: Caminho absoluto
    if rank != 0:
        return

    all_experiment_results = {}
    for name in experiment_names:
        result_dir = os.path.join(result_base_dir, name)
        train_file = os.path.join(result_dir, "train_losses.pkl")
        val_file   = os.path.join(result_dir, "val_losses.pkl")

        if os.path.exists(train_file):
            with open(train_file, "rb") as f:
                train_losses = pickle.load(f)
            with open(val_file, "rb") as f:
                val_losses = pickle.load(f)
            all_experiment_results[name] = {"train_losses": train_losses, "val_losses": val_losses}

    if not all_experiment_results:
        print("Nenhum resultado encontrado para analise.")
        return

    stability_window = 5
    summary_data = []
    for name, metrics in all_experiment_results.items():
        final_val_loss = metrics["val_losses"][-1] if metrics["val_losses"] else float("nan")
        final_perplexity = math.exp(final_val_loss) if not math.isnan(final_val_loss) else float("nan")
        recent = metrics["val_losses"][-stability_window:]
        stability_var = float(np.var(recent)) if len(recent) > 1 else 0.0

        summary_data.append({
            "Attention": name,
            "Val_Loss": round(final_val_loss, 4),
            "Perplexity": round(final_perplexity, 4),
            "Stability_Var": round(stability_var, 6),
            "Steps": len(metrics["train_losses"]),
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n--- Resumo ---")
    print(summary_df.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    ax1 = axes[0, 0]
    for name, m in all_experiment_results.items():
        perps = [math.exp(l) for l in m["val_losses"]]
        ax1.plot(perps, label=name, marker="o", markersize=4)
    ax1.set_title("Perplexidade de Validacao")
    ax1.set_xlabel("Eval step")
    ax1.set_ylabel("Perplexidade")
    ax1.legend()
    ax1.grid(True)

    ax2 = axes[0, 1]
    for name, m in all_experiment_results.items():
        ax2.plot(m["val_losses"], label=name, marker="s", markersize=4)
    ax2.set_title("Loss de Validacao")
    ax2.set_xlabel("Eval step")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    ax3 = axes[1, 0]
    for name, m in all_experiment_results.items():
        ax3.plot(m["train_losses"], label=name, marker="^", markersize=3)
    ax3.set_title("Loss de Treino")
    ax3.set_xlabel("Eval step")
    ax3.set_ylabel("Loss")
    ax3.legend()
    ax3.grid(True)

    ax4 = axes[1, 1]
    names_stab = list(all_experiment_results.keys())
    vars_stab = [np.var(m["val_losses"][-stability_window:]) if len(m["val_losses"]) > 1 else 0.0
                 for m in all_experiment_results.values()]
    ax4.bar(names_stab, vars_stab)
    ax4.set_title(f"Variancia (ultimas {stability_window})")
    ax4.set_ylabel("Variancia")
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(result_base_dir, "comparison.pdf"))
    plt.close()
    print(f"\nGrafico salvo em {os.path.join(result_base_dir, 'comparison.pdf')}")

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento DDP de GPTModel com variantes de atencao")
    parser.add_argument("--attention", type=str, default="PyTorchOptimized",
                        choices=["OriginalManual", "PyTorchOptimized", "all"])
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    ddp_setup(rank, world_size)
    set_seed(123)

    if rank == 0:
        print(f"PyTorch: {torch.__version__}")
        print(f"GPUs: {world_size}")
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 7:
                torch.set_float32_matmul_precision("high")
                print("Tensor cores ativados")
        print(f"AMP: {TRAINING_CONFIG['use_amp']}")
        print(f"Accum steps: {TRAINING_CONFIG['accumulation_steps']}")
        print(f"Effective batch: {TRAINING_CONFIG['batch_size_per_gpu'] * world_size * TRAINING_CONFIG['accumulation_steps']}")

    attention_map = {
        "OriginalManual": OriginalSelfAttention,
        "PyTorchOptimized": PyTorchMultiHeadAttention,
    }

    if args.attention == "all":
        experiments = list(attention_map.items())
    else:
        experiments = [(args.attention, attention_map[args.attention])]

    for name, attn_cls in experiments:
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Treinando: {name}")
            print(f"{'=' * 60}")
        train_model_ddp(attn_cls, rank, world_size, name)

    barrier()
    destroy_process_group()

    if rank == 0:
        analyze_results([n for n, _ in experiments])
