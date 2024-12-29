import torch
from torch import nn

from model.modules.mlp import MLP
from model.modules.ss2d import SS2DBlock


class MixFormer(nn.Module):
    def __init__(self, args, dim_in, mixer, mode):
        super().__init__()
        self.norm_1 = nn.LayerNorm(dim_in)

        if mixer == 'mamba':
            self.mixer = SS2DBlock(args, dim_in=dim_in, mode=mode)
        else:
            raise Exception("Mixer type is not implemented")

        self.drop_path = nn.Identity()
        self.norm_2 = nn.LayerNorm(dim_in)
        self.mlp = MLP(dim_in, hidden_features=int(dim_in * args.mlp_ratio), out_features=dim_in, drop=args.drop)

        self.layer_scale_1 = nn.Parameter(args.layer_scale_init_value * torch.ones(1, 1, dim_in), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(args.layer_scale_init_value * torch.ones(1, 1, dim_in), requires_grad=True)

    def forward(self, x):
        """
        :param x: tensor with shape [B, T, J, C] (B=12, T=27, J=17, C=64)
        """
        x = x + self.drop_path(self.layer_scale_1 * self.mixer(self.norm_1(x)))  # (Norm + SS2D + Drop_Path) + x
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm_2(x)))  # (Norm + FFN + Drop_Path) + x

        return x


class SHARPEBlock(nn.Module):
    """
    The block, which will be repeated N times.
    """
    def __init__(self, args, dim_in, mixer):
        super().__init__()
        self.mixer_1_spatial = MixFormer(args, dim_in=dim_in, mixer=mixer['1'], mode='spatial')
        self.mixer_1_temporal = MixFormer(args, dim_in=dim_in, mixer=mixer['1'], mode='temporal')
        self.mixer_2_spatial = MixFormer(args, dim_in=dim_in, mixer=mixer['2'], mode='spatial')
        self.mixer_2_temporal = MixFormer(args, dim_in=dim_in,mixer=mixer['2'], mode='temporal')

        self.fusion = nn.Linear(dim_in * 2, 2)
        self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def forward(self, x):
        """
        :param x: tensor with shape [B, T, J, C]
        """
        x_1 = self.mixer_2_spatial(self.mixer_1_spatial(x))
        x_2 = self.mixer_2_temporal(self.mixer_1_temporal(x))

        alpha = torch.cat((x_1, x_2), dim=-1)
        alpha = self.fusion(alpha)
        alpha = alpha.softmax(dim=-1)
        x = x_1 * alpha[..., 0:1] + x_2 * alpha[..., 1:2]

        return x


class SHARPE(nn.Module):
    """
    Main class of SHARPE
    """
    def __init__(self, args, dim_in=3, dim_feat=64, dim_rep=512, dim_out=3, mixer=None):
        """
        :param dim_in: Input dimension
        :param dim_feat: Feature dimension
        :param dim_rep: Motion representation dimension
        :param dim_out: Output dimension. For 3D pose lifting it is set to 3
        """
        super().__init__()

        self.in_proj = nn.Linear(dim_in, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, args.num_joints, dim_feat))

        layers = []
        for layer in range(args.n_layers):
            layers.append(SHARPEBlock(args, dim_in=dim_feat, mixer=mixer))
        self.sharpe_blocks = nn.Sequential(*layers)

        self.norm = nn.LayerNorm(dim_feat)
        self.motion_semantic_proj =  nn.Linear(dim_feat, dim_rep)
        self.act = nn.Tanh()
        self.regression_head = nn.Linear(dim_rep, dim_out)

    def forward(self, x):
        """
        :param x: tensor with shape [B, T, J, C]
        """
        x = self.in_proj(x)  # FC Layer: dim_in -> dim_feat. Projection to higher dim space
        x = x + self.pos_embed  # Add SPE (spatial positional embedding)

        x = self.sharpe_blocks(x)

        x = self.norm(x)  # normalization
        x = self.motion_semantic_proj(x)  # FC Layer: dim_feat -> dim_rep (512 for all variants).
        x = self.act(x)
        x = self.regression_head(x)  # FC Layer: dim_feat -> dim_out: 3 (3D Pose Estimation, estimating the x y z for each joint from each frame for every image from the batch: [B, T, J, C])

        return x  # [B, T, J, C]