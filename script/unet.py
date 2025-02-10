import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np
from timeit import default_timer



def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(p=keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.condEmbedding(t)
        return emb

    
class DownSample_O(nn.Module):
    def __init__(self, in_ch):
        super(DownSample_O, self).__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x):
        x = self.c1(x) + self.c2(x)
        return x


class UpSample_O(nn.Module):
    def __init__(self, in_ch, ups):
        super(UpSample_O, self).__init__()
        self.c = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.t = nn.ConvTranspose2d(in_ch, in_ch, 5, 2, 2, 1)  # kernel, stride, padding, outpadding
        self.ups = ups

    def forward(self, x):
        _, _, H, W = x.shape
        if self.ups == 0:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
        elif self.ups == 1:
            x = self.t(x)
        x = self.c(x)

        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)
        
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))  # bmm tensor乘法，缩放点积求value的权重
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)
        
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        
        h = self.proj(h)
        
        return x + h

class Pre_UP(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, attn=False):
        super(Pre_UP, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(1, in_ch),  # 2
            Swish(),
            nn.ConvTranspose2d(in_ch, out_ch, 5, 2, 1, 0),  # 9->27: 4,3,1,1 / 10: 6,2,1,1 / 9: 5,2,1,0
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch), 
            Swish(),
            nn.Dropout(dropout),
            nn.ConvTranspose2d(out_ch, out_ch, 6, 2, 1, 0),  # 27->81:4,3,1,1 / 49: 6,2,1,1 / 41: 6,2,1,1
        )

        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.attn(h)
        return h


class ResBlock_for_O(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, attn=True):
        super(ResBlock_for_O, self).__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:  
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, in_ch=9, output_ch=3, ch=32, ch_mult=[1,2,2,2], attn=False, num_res_blocks=2, dropout=0.01, sample_gap=5, ups=0): 
        super(UNet, self).__init__()

        self.pre_up = Pre_UP(in_ch, ch, dropout=dropout, attn=attn)  # atten
        self.sample_gap = sample_gap
        
        self.head = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock_for_O(in_ch=now_ch, out_ch=out_ch, dropout=dropout, attn=attn))  # atten
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample_O(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock_for_O(now_ch, now_ch, dropout, attn=True),  # atten
            ResBlock_for_O(now_ch, now_ch, dropout, attn=attn),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks+1):
                self.upblocks.append(ResBlock_for_O(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, dropout=dropout, attn=attn))  # atten
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample_O(now_ch, ups))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, output_ch, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x):

        x_ini = x
        x = self.pre_up(x)
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            h = layer(h)

        # Upsampling
        a = 0
        for layer in self.upblocks:
            if isinstance(layer, ResBlock_for_O):
                hh = hs.pop()
                h = torch.cat([h, hh], dim=1)
            
            h = layer(h)

        h = self.tail(h)

        assert len(hs) == 0

        return h

    def substituter(self, sample_gap, data, observe):
        out_data = data
        h_sub = torch.tensor(np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2], out_data.shape[3])), dtype=torch.float64).to(out_data.device)
        for i in range(out_data.shape[2]):
            for z in range(out_data.shape[3]):
                if z%sample_gap == 0 and i%sample_gap == 0:
                    h_sub[:,:,i,z] = observe[:,-3:,int(i/sample_gap),int(z/sample_gap)]
                else:
                    h_sub[:,:,i,z] = out_data[:,:,i,z]

        return h_sub