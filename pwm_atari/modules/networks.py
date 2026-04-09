import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AgentLayer(nn.Module):
    def __init__(self, inp_size, size, hidden, act):
        super().__init__()
        self.layer = nn.Sequential(
            FeedForwardLayer(inp_size, hidden, act()),
            FeedForwardLayer(hidden, hidden, act()))
        self.head = nn.Linear(hidden, size)
        self.head.apply(uniform_weight_init(0.0))
        
    def forward(self, inp):
        x = self.layer(inp)
        x = self.head(x)
        return x


class Encoder(nn.Module):
    def __init__(self, width, in_ch, stem_ch, min_res, act, latent_dim=1024):
        super().__init__()
        feature_width = width // 2
        channels = stem_ch

        backbone = [ConvLayer(in_ch, channels, act())]
        while True:
            out_channels = channels * 2
            backbone += [ConvLayer(
                channels, out_channels, act())]
            channels = out_channels
            feature_width //= 2

            if feature_width == min_res:
                break

        self.backbone = nn.Sequential(*backbone)
        self.out_ch = channels
        
        # --- 新增：加入 Self-Attention 弥补局部感受野 ---
        self.attn = nn.MultiheadAttention(embed_dim=self.out_ch, num_heads=4, batch_first=True)
        # --- 新增：投影到纯潜空间维度 z_t ---
        self.head = nn.Linear(self.out_ch * (min_res ** 2), latent_dim)
        # 对外暴露的特征维度现在变成了纯潜向量的维度
        self.embed = latent_dim

    def forward(self, x):
        shape = x.shape[:2]
        x = x.flatten(0, 1)  # (B*L, C, H, W)
        x = self.backbone(x)
        
        # --- 新增：空间特征转序列，输入 Attention ---
        # 将空间维度 (H, W) 展平作为 Attention 的 Sequence 维度
        B_L, C, H, W = x.shape
        x = x.flatten(2, 3).transpose(1, 2)  # (B*L, H*W, C)
        
        # Self-Attention 处理空间全局关系
        attn_out, _ = self.attn(x, x, x)
        
        # --- 新增：展平并映射到潜空间向量 z_t ---
        z = attn_out.flatten(1, 2)  # (B*L, H*W*C)
        z = self.head(z)            # (B*L, latent_dim)
        
        return z.unflatten(0, shape)  # 还原为 (B, L, latent_dim)


class Decoder(nn.Module):
    def __init__(self, stoch, out_ch, in_ch, stem_ch, min_res, act):
        super().__init__()
        backbone = [Rearrange(stoch, out_ch, min_res, act())]

        channels = out_ch
        feat_width = min_res
        while True:
            if channels == stem_ch:
                break
            
            out_channels = channels // 2
            backbone += [TransposeConvLayer(
                channels, out_channels, act())]
            channels = out_channels
            feat_width *= 2

        backbone += [nn.ConvTranspose2d(
            channels, in_ch, 4, 2, 1)]
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        shape = sample.shape[:2]
        obs_hat = self.backbone(sample)
        obs_hat = obs_hat.unflatten(0, shape)
        return torch.sigmoid(obs_hat)


class Head(nn.Module):
    def __init__(self, inp_size, size, hidden, act):
        super().__init__()
        self.backbone = nn.Sequential(
            FeedForwardLayer(inp_size, hidden, act()),
            FeedForwardLayer(hidden, hidden, act()))
        self.head = nn.Linear(hidden, size)
        self.head.apply(uniform_weight_init(0.0))

    def forward(self, feat):
        x = self.backbone(feat)
        x = self.head(x)
        return x
    

class InpLayer(nn.Module):
    def __init__(self, inp_size, size, hidden, act):
        super().__init__()
        self.backbone = FeedForwardLayer(
            inp_size, hidden, act())
        self.head = nn.Sequential(
            nn.Linear(hidden, size, bias=False),
            nn.LayerNorm(size))

    def forward(self, inp):
        x = self.backbone(inp)
        x = self.head(x)
        return x
    

class ObsStatLayer(nn.Module):
    def __init__(self, inp_size, size, act):
        super().__init__()
        self.head = nn.Linear(inp_size, size, bias=False)
        self.norm = BatchNorm1d(size)

    def forward(self, inp):
        if inp.dtype != self.head.weight.dtype:
            inp = inp.to(self.head.weight.dtype)
        x = self.head(inp)
        x = self.norm(x)
        return x


class ImsStatLayer(nn.Module):
    def __init__(self, inp_size, size, act):
        super().__init__()
        self.backbone = GatingLayer(inp_size, act())
        self.head = nn.Linear(inp_size, size)
        self.norm = BatchNorm1d(size)

    def forward(self, inp):
        if inp.dtype != self.head.weight.dtype:
            inp = inp.to(self.head.weight.dtype)
        x = self.backbone(inp)
        x = self.head(x)
        x = self.norm(x)
        return x


class MixingLayer(nn.Module):
    def __init__(self, inp_size, hidden, bias=True):
        super().__init__()
        self.mix = nn.Parameter(init_weight(1, inp_size))
        self.register_parameter("mix_weight", self.mix)
        self.layer = nn.Linear(inp_size, hidden, bias=bias)

    def forward(self, inp, last):
        mix_w = torch.sigmoid(self.mix + 1)
        x = mix_w * inp + (1 - mix_w) * last
        x = self.layer(x)
        return x
    

class GatingLayer(nn.Module):
    def __init__(self, hidden, act, pdrop=0.1):
        super().__init__()
        self.layer = nn.Linear(hidden, 2 * hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(pdrop)
        self.act = act

    def forward(self, inp):
        p = self.layer(inp)
        x, g = p.chunk(2, dim=-1)
        x = F.mish(g) * x
        x = self.drop(x)
        x = x + inp
        x = self.norm(x)
        return x


class FeedForwardLayer(nn.Module):
    def __init__(self, inp_size, size, act):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(inp_size, size, bias=False),
            nn.LayerNorm(size))
        self.act = act

    def forward(self, inp):
        x = self.layer(inp)
        x = self.act(x)
        return x


class Rearrange(nn.Module):
    def __init__(self, in_dim, out_ch, min_res, act):
        super().__init__()
        out_dim = out_ch * (min_res ** 2)
        self.layer = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)

        self.channels = out_ch
        self.width = min_res
        self.size = (
            self.channels, self.width, self.width)

    def forward(self, x):
        x = self.layer(x)
        x = x.flatten(0, 1)
        x = x.unflatten(1, self.size)
        x = self.norm(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim, out_dim, 4, 2, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = act

    def forward(self, inp):
        x = self.conv(inp)
        x = self.norm(x)
        x = self.act(x)
        return x


class TransposeConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_dim, out_dim, 4, 2, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = act

    def forward(self, inp):
        x = self.conv(inp)
        x = self.norm(x)
        x = self.act(x)
        return x


class BatchNorm1d(nn.Module):
    def __init__(self, size, eps=1e-5):
        super().__init__()
        self.norm = nn.BatchNorm1d(size, eps=eps)

    def forward(self, inp):
        if inp.dim() > 2:
            shape = inp.shape[:2]
            x = inp.flatten(0, 1)
            if x.shape[0] == 1 and self.norm.training:
                x = F.batch_norm(
                    x,
                    self.norm.running_mean,
                    self.norm.running_var,
                    self.norm.weight,
                    self.norm.bias,
                    training=False,
                    eps=self.norm.eps,
                )
            else:
                x = self.norm(x)
            x = x.unflatten(0, shape)
        else:
            if inp.shape[0] == 1 and self.norm.training:
                x = F.batch_norm(
                    inp,
                    self.norm.running_mean,
                    self.norm.running_var,
                    self.norm.weight,
                    self.norm.bias,
                    training=False,
                    eps=self.norm.eps,
                )
            else:
                x = self.norm(inp)
        return x


class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, inp):
        x = inp.float()
        s = torch.square(x)
        s = s.mean(dim=-1, keepdim=True)
        s = torch.rsqrt(s + self.eps)
        x = (x * s).type_as(inp)
        x = x * self.weight
        return x
    

def init_weight(in_dim, out_dim):
    denoms = (in_dim + out_dim) / 2
    scale = 1 / denoms
    std = math.sqrt(scale) / 0.87962566103423978
    data = torch.randn(in_dim, out_dim) * std
    data = torch.clip(data, min=-2*std, max=2*std)
    return data


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
    return f