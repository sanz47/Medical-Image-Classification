import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim

        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_emb
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.qkv = nn.Linear(emb_dim, emb_dim * 3, bias=False)
        self.fc_out = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        dots = torch.einsum('bhid,bhjd->bhij', q, k) / self.head_dim ** 0.5
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.reshape(B, N, C)
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(emb_dim, num_heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.mlp(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, img_size=224, patch_size=16, emb_dim=768, depth=12, num_heads=12, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)
        self.transformer = nn.Sequential(
            *[TransformerBlock(emb_dim, num_heads, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    img = torch.randn(1, 3, 224, 224) 
    vit = VisionTransformer(num_classes=10)  
    preds = vit(img)
    print(preds.shape) 
