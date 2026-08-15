from __future__ import annotations
import torch
from torch import nn
from torch.autograd import Function

class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, strength): ctx.strength = strength; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return -ctx.strength * grad_output, None

class PathEncoder(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1,32,15,4,7),nn.GroupNorm(4,32),nn.GELU(),nn.Conv1d(32,64,11,4,5),nn.GroupNorm(8,64),nn.GELU(),nn.Conv1d(64,128,9,5,4),nn.GroupNorm(8,128),nn.GELU(),nn.Conv1d(128,dim,5,5,2),nn.GroupNorm(12,dim),nn.GELU())
    def forward(self, x): return self.net(x).mean(-1)

class EnvWaveSSL(nn.Module):
    def __init__(self, dim=192, env_dim=6, layers=4, heads=6, dropout=0.1):
        super().__init__()
        self.path_encoder = PathEncoder(dim)
        self.path_embedding = nn.Parameter(torch.randn(1,8,dim)*0.02)
        a = torch.block_diag(torch.ones(4,4), torch.ones(4,4)); a = a/a.sum(1,keepdim=True)
        self.register_buffer("adjacency", a)
        self.graph_proj = nn.Linear(dim,dim,bias=False)
        layer = nn.TransformerEncoderLayer(dim, heads, dim*4, dropout, "gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.damage_head = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim))
        self.environment_head = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim))
        self.env_predictor = nn.Sequential(nn.GELU(),nn.Linear(dim,env_dim))
        self.env_adversary = nn.Sequential(nn.GELU(),nn.Linear(dim,env_dim))
        self.wave_decoder = nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,2000))
    def forward(self, wave, adversary_strength=1.0):
        b,p,s = wave.shape
        tokens = self.path_encoder(wave.reshape(b*p,1,s)).reshape(b,p,-1) + self.path_embedding
        tokens = tokens + self.graph_proj(torch.einsum("ij,bjd->bid",self.adjacency,tokens))
        tokens = self.transformer(tokens); pooled = tokens.mean(1)
        zd, ze = self.damage_head(pooled), self.environment_head(pooled)
        return {"z_damage":zd,"z_environment":ze,"environment_prediction":self.env_predictor(ze),"environment_adversarial":self.env_adversary(GradientReverse.apply(zd,adversary_strength)),"wave_reconstruction":self.wave_decoder(tokens),"path_tokens":tokens}

def masked_wave(wave, ratio=0.35, block=80):
    b,p,s = wave.shape; mask = torch.zeros_like(wave,dtype=torch.bool)
    for _ in range(max(1,int(s*ratio/block))):
        start=torch.randint(0,max(s-block,1),(b,p),device=wave.device); offsets=torch.arange(block,device=wave.device).view(1,1,-1); index=(start.unsqueeze(-1)+offsets).clamp_max(s-1); mask.scatter_(2,index,True)
    return wave.masked_fill(mask,0), mask

