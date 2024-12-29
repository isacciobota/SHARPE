# This file is a modified version of https://github.com/MzeroMiko/VMamba/blob/main/classification/models/vmamba.py

import math
import torch
import torch.nn as nn

from model.modules.csm_triton import cross_scan_fn, cross_merge_fn
from model.modules.mamba2.ssd_minimal import selective_scan_chunk_fn


class SS2D(nn.Module):
    def __init__(self, dim_in, dt_rank, d_state):
        super().__init__()
        k_group = 4
        self.d_state = d_state
        self.x_proj = [
            nn.Linear(dim_in, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]

        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # simple init dt_projs, A_logs, Ds
        self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(dim_in // dt_rank))))
        self.A_logs = nn.Parameter(-0.1 * torch.ones((k_group, dt_rank)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, dt_rank)))

    def forward(self, x, chunk_size=64, selective_scan_backend=None, scan_mode="cross2d", scan_force_torch=False):
        assert scan_mode in ["unidi", "bidi", "cross2d"]
        assert selective_scan_backend in [None, "triton", "torch"]
        x_proj_bias = getattr(self, "x_proj_bias", None)

        N = self.d_state
        B, H, W, RD = x.shape
        K, R, D = self.Ds.shape
        assert RD == R * D
        L = H * W
        KR = K * R
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

        xs = cross_scan_fn(x, in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch)
        x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)
        xs = xs.contiguous().view(B, L, KR, D)
        dts = dts.contiguous().view(B, L, KR)
        Bs = Bs.contiguous().view(B, L, K, N)
        Cs = Cs.contiguous().view(B, L, K, N)

        As = -self.A_logs.exp().view(KR)
        Ds = self.Ds.view(KR, D)
        dt_bias = self.dt_projs_bias.view(KR)

        ys, final_state = selective_scan_chunk_fn(
            xs, dts, As, Bs, Cs, chunk_size=chunk_size, D=Ds, dt_bias=dt_bias,
            dt_softplus=True, return_final_states=True, backend=selective_scan_backend,
        )

        y = cross_merge_fn(ys.view(B, H, W, K, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=self.Ds,
                us=xs, dts=dts, delta_bias=self.dt_projs_bias,
                initial_state=self.initial_state, final_satte=final_state,
                ys=ys, y=y, H=H, W=W,
            ))

        y = y.view(B, H, W, -1)

        return y.to(x.dtype)


class SS2DBlock(nn.Module):
    def __init__(self, args, dim_in, mode):
        super().__init__()
        dim_hidden = int(args.ssm_ratio * dim_in)
        dt_rank = math.ceil(dim_in / 16)
        assert dim_hidden % dt_rank == 0

        self.in_proj = nn.Linear(dim_in, dim_hidden, bias=False)

        #dw conv
        pad = (args.conv_kernel - 1) // 2
        if mode == 'spatial':
            self.dw_conv = nn.Conv2d(dim_hidden, dim_hidden, groups=dim_hidden, bias=False, kernel_size=(args.conv_kernel, 1), padding=(pad, 0))
        elif mode == 'temporal':
            self.dw_conv = nn.Conv2d(dim_hidden, dim_hidden, groups=dim_hidden, bias=False, kernel_size=(1, args.conv_kernel), padding=(0, pad))
        else:
            self.dw_conv = nn.Conv2d(dim_hidden, dim_hidden, groups=dim_hidden, bias=False, kernel_size=args.conv_kernel, padding=pad)

        self.act_1 = nn.GELU()
        self.ss2d = SS2D(dim_hidden, dt_rank, args.d_state)
        self.norm = nn.LayerNorm(dim_hidden)
        self.act_2 = nn.GELU()
        self.out_proj = nn.Linear(dim_hidden, dim_in, bias=False)
        self.dropout = nn.Dropout(args.drop)

    def forward(self, x):
        x = self.in_proj(x)
        x = x.permute(0, 3, 1, 2) # [B, T, J, C] -> [B, C, T, J]
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1) # [B, C, T, J] -> [B, T, J, C]
        x = self.act_1(x)
        x = self.ss2d(x, selective_scan_backend='torch', scan_mode="cross2d", scan_force_torch=False)
        x = self.act_2(self.norm(x))
        out = self.dropout(self.out_proj(x))
        return out
