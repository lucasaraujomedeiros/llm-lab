import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
import requests
import os
import math

# ==========================================
# 1. CONFIGURAÇÕES 
# ==========================================
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size (GPT-2)
    "context_length": 1024,  # Reduzido para caber melhor na memória local (no notebook original era 1024)
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

# ==========================================
# 2. ARQUITETURA DO MODELO
# ==========================================
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))

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
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"]
        )
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
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)

# ==========================================
# 3. PREPARAÇÃO DE DADOS (DATASET E LOADER)
# ==========================================
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids, self.target_ids = [], []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1: i + max_length + 1]))

    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx): return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

# ==========================================
# 4. FUNÇÕES DE TREINAMENTO E GERAÇÃO
# ==========================================
def calc_loss_batch(input_batch, target_batch, model, device):
    logits = model(input_batch.to(device))
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.to(device).flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter=None):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature

            logits = logits - logits.max(dim=-1, keepdim=True).values

            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id: 
            break

        idx = torch.cat((idx, idx_next), dim=1) 

    return idx

# ==========================================
# 5. PIPELINE PRINCIPAL (EXECUÇÃO)
# ==========================================
def avaliar(model, device):
    url = "https://raw.githubusercontent.com/lucasaraujomedeiros/llm-lab/refs/heads/main/lucas/data/textos_wikipedia_relevantes.txt"
    file_path = "textos_wikipedia_relevantes.txt"
    if not os.path.exists(file_path):
        print("Baixando dataset...")
        with open(file_path, "wb") as f:
            f.write(requests.get(url, timeout=30).content)
    
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    
    train_loader = create_dataloader(text_data[:split_idx], batch_size=2, max_length=GPT_CONFIG_124M["context_length"], stride=128, shuffle=True)
    val_loader = create_dataloader(text_data[split_idx:], batch_size=2, max_length=GPT_CONFIG_124M["context_length"], stride=128, shuffle=False)

    train_loss, val_loss = evaluate_model(model=model, train_loader=train_loader,
                                          val_loader=val_loader, device=device, eval_iter=10    )
    return (train_loss, val_loss)

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def load_and_evaluate_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando Pipeline de Avaliação. Usando dispositivo: {device}")

    # 2. Inicializar a arquitetura "vazia"
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)
    
    # 3. Carregar os pesos do modelo treinado
    caminho_modelo = "wikipediaenglish.pth" # O arquivo que você salvou
    
    if os.path.exists(caminho_modelo):
        print(f"Carregando pesos do arquivo {caminho_modelo}...")
        # map_location garante que funcione mesmo se quem rodar não tiver GPU
        model.load_state_dict(torch.load(caminho_modelo, map_location=device, weights_only=True))
    else:
        print(f"Erro: Arquivo {caminho_modelo} não encontrado. Rodando com modelo não treinado.")
    
    model.to(device)
    model.eval() # Modo de avaliação (desliga dropout, etc)

    # 4. Avaliação (Opcional: Calcular Perplexidade num texto de teste)
    _, val_loss = avaliar(model, device)
    print("Loss = ", val_loss, " |||| Perplexity = ", math.exp(val_loss))

    # 5. Geração de Texto (O objetivo final)
    prompts = [
        "The long and",
        "What is the purpose of life?",
        "I must also observe"
    ]
    
    print("\n--- Gerando Textos ---")
    for prompt in prompts:
        encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
        out_tokens = generate(
            model=model, 
            idx=encoded, 
            temperature=2,
            max_new_tokens=30, 
            context_size=GPT_CONFIG_124M["context_length"]
        )
        texto_gerado = tokenizer.decode(out_tokens.squeeze(0).tolist())
        print(f"\nPrompt: '{prompt}'")
        print(f"Gerado: {texto_gerado}")

if __name__ == "__main__":
    os.system("pwd")
    load_and_evaluate_pipeline()

