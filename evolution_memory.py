# evolution_memory.py
# EVO-MEM — Model and Dataset definitions

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random


# ─────────────────────────────────────────────
# THEME DEFINITIONS  (shared across files)
# ─────────────────────────────────────────────
THEMES = {
    0: {
        "name": "Urban",
        "base": [0.3, 0.3, 0.3],
        "growth": [0.9, 0.9, 1.0],
        "caption": "Urban expansion seen in the northern sector",
    },
    1: {
        "name": "Forest",
        "base": [0.1, 0.4, 0.1],
        "growth": [0.6, 0.3, 0.1],
        "caption": "Deforestation occurring in the rainforest",
    },
    2: {
        "name": "Water",
        "base": [0.0, 0.1, 0.6],
        "growth": [0.3, 0.5, 0.2],
        "caption": "River levels rising due to seasonal floods",
    },
    3: {
        "name": "Desert",
        "base": [0.8, 0.7, 0.3],
        "growth": [0.6, 0.6, 0.6],
        "caption": "New residential complex built on sandy terrain",
    },
}


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class EvolutionSatelliteDataset(Dataset):
    """
    Simulates three dataset paradigms:
      - NWPU:      categorical land-cover themes
      - RSICD:     descriptive natural-language captions
      - Sentinel-2: temporal multi-frame image streams
    """
    def __init__(self, size: int = 1000):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx):
        theme_id = random.randint(0, 3)
        return make_stream(theme_id), torch.tensor(theme_id)


def make_stream(theme_id: int) -> torch.Tensor:
    """Generate a 5-frame temporal image stream for the given theme."""
    theme = THEMES[theme_id]
    stream = []
    for t in range(5):
        img = np.full((224, 224, 3), theme["base"], dtype=np.float32)
        for _ in range(2 + t * 3):
            x = random.randint(0, 180)
            y = random.randint(0, 180)
            s = random.randint(15, 40)
            img[x: x + s, y: y + s] = theme["growth"]
        img += np.random.uniform(-0.05, 0.05, (224, 224, 3))
        stream.append(torch.from_numpy(np.clip(img, 0, 1)).permute(2, 0, 1))
    return torch.stack(stream)


# ─────────────────────────────────────────────
# COLOR ENCODER  (replaces ViT)
# ─────────────────────────────────────────────
class ColorEncoder(nn.Module):
    """
    Lightweight stand-in for the Vision Transformer encoder.
    Projects the global mean RGB colour to a 768-d feature vector —
    the same dimensionality as the ViT CLS token.
    For synthetic satellite blobs, global colour is the dominant
    discriminative signal, making this encoder both fast and accurate.
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 768)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B, 3, H, W)
        return self.proj(img.mean(dim=[2, 3]))


# ─────────────────────────────────────────────
# EPISODIC MEMORY BANK
# ─────────────────────────────────────────────
class EpisodicMemoryBank(nn.Module):
    """Ring buffer storing feature embeddings. Write/Read via circular ptr."""
    def __init__(self, embed_dim: int = 768, capacity: int = 100):
        super().__init__()
        self.capacity = capacity
        self.register_buffer("memory", torch.randn(capacity, embed_dim))
        self.ptr = 0

    def write(self, feature: torch.Tensor) -> None:
        self.memory[self.ptr] = feature.mean(dim=0).detach()
        self.ptr = (self.ptr + 1) % self.capacity

    def read(self) -> torch.Tensor:
        return self.memory


# ─────────────────────────────────────────────
# EVOLUTIONARY SELECTOR
# ─────────────────────────────────────────────
class EvolutionarySelector(nn.Module):
    """Top-k cosine retrieval + Gaussian mutation."""
    def __init__(self, mutation_rate: float = 0.1):
        super().__init__()
        self.mutation_rate = mutation_rate

    def forward(self, current_feat: torch.Tensor, memory_bank: torch.Tensor) -> torch.Tensor:
        sim = F.cosine_similarity(
            current_feat.unsqueeze(1), memory_bank.unsqueeze(0), dim=-1
        )
        top_k_idx = torch.topk(sim, k=5).indices
        selected  = memory_bank[top_k_idx]
        mask = torch.rand(selected.shape, device=selected.device) < self.mutation_rate
        return selected + (mask * torch.randn_like(selected) * 0.05)


# ─────────────────────────────────────────────
# FULL MODEL
# ─────────────────────────────────────────────
class EvolutionMemoryModel(nn.Module):
    """
    EVO-MEM Pipeline:
      ColorEncoder -> Episodic Memory Bank -> Evolutionary Selector
      -> Memory-Augmented MLP Decoder -> Class logits
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        embed_dim     = 768
        self.encoder  = ColorEncoder()
        self.emb_bank = EpisodicMemoryBank(embed_dim)
        self.selector = EvolutionarySelector()
        # 1 current + 5 memories = 6 x 768
        self.decoder  = nn.Sequential(
            nn.Linear(embed_dim * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, image_stream: torch.Tensor) -> torch.Tensor:
        latest_img = image_stream[:, -1, :, :, :]
        enc_out    = self.encoder(latest_img)
        self.emb_bank.write(enc_out)
        memories   = self.selector(enc_out, self.emb_bank.read())
        combined   = torch.cat([enc_out.unsqueeze(1), memories], dim=1)
        return self.decoder(combined.view(combined.size(0), -1))
