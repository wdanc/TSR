# PiT
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

import torch
from einops import rearrange
from torch import nn
import math

from functools import partial
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from models.dtcwt_stem import DTCWTStem_bchw

def _cifar10_cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 10, 'input_size': (3, 32, 32),
        'interpolation': 'bicubic',
        'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010),

        **kwargs
    }

def _rs_cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 45, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.3680, 0.3810, 0.3436), 'std': (0.1454, 0.1356, 0.1320),
        **kwargs
    }

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TSRBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, f_decorrelation=True):
        super().__init__()
        self.f_decorrelation = f_decorrelation
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.f_decorrelation:
            self.f_x = x
        return x

# -----------------------------------------

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class TSRTransformer(nn.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio, drop_rate=.0, attn_drop_rate=.0,
                 drop_path_prob=None, f_decorrelation=True):
        super(TSRTransformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.ModuleList([
            TSRBlock(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                f_decorrelation=f_decorrelation
            )
            for i in range(depth)])

    def forward(self, x, cls_tokens):
        h, w = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')

        token_length = cls_tokens.shape[1]
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x, cls_tokens

class Transformer(nn.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(depth)])

    def forward(self, x, cls_tokens):
        h, w = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')

        token_length = cls_tokens.shape[1]
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x, cls_tokens


class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):

        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size,
                              stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class TSRPoolingTransformer(nn.Module):
    def __init__(self, img_size, patch_size, stride, base_dims, depth, heads,
                 mlp_ratio, num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0, f_decorrelation=True):
        super(TSRPoolingTransformer, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor(
            (img_size + 2 * padding - patch_size*2) / (stride*2) + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, base_dims[0] * heads[0], width, width),
            requires_grad=True
        )
        self.patch_embed = DTCWTStem_bchw(in_chans=in_chans, embed_dim=base_dims[0] * heads[0],
                                          patch_size=patch_size, stride=stride, padding=padding,
                                          norm_layer=nn.LayerNorm, f_decorrelation=True)

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, base_dims[0] * heads[0]),
            requires_grad=True
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            self.transformers.append(
                TSRTransformer(base_dims[stage], depth[stage], heads[stage],
                               mlp_ratio, drop_rate, attn_drop_rate, drop_path_prob,
                               f_decorrelation)
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
                                      )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = self.pools[stage](x, cls_tokens)
        x, cls_tokens = self.transformers[-1](x, cls_tokens)

        cls_tokens = self.norm(cls_tokens)

        return cls_tokens

    def forward(self, x):
        cls_token = self.forward_features(x)
        cls_token = self.head(cls_token[:, 0])
        return cls_token

class TSRPoolingTransformer_cifar(nn.Module):
    def __init__(self, img_size, base_dims, depth, heads,
                 mlp_ratio, num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0, f_decorrelation=True):
        super(TSRPoolingTransformer_cifar, self).__init__()

        total_block = sum(depth)
        block_idx = 0

        width = img_size // 2

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.pos_embed = nn.Parameter(
            torch.randn(1, base_dims[0] * heads[0], width, width),
            requires_grad=True
        )
        self.patch_embed = DTCWTStem_bchw(in_chans=in_chans, embed_dim=base_dims[0] * heads[0],
                                          patch_size=3, stride=1, padding=1,
                                          norm_layer=nn.LayerNorm, f_decorrelation=f_decorrelation)

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, base_dims[0] * heads[0]),
            requires_grad=True
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            self.transformers.append(
                TSRTransformer(base_dims[stage], depth[stage], heads[stage], mlp_ratio,
                               drop_rate, attn_drop_rate, drop_path_prob,
                               f_decorrelation)
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
                                      )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = self.pools[stage](x, cls_tokens)
        x, cls_tokens = self.transformers[-1](x, cls_tokens)

        cls_tokens = self.norm(cls_tokens)

        return cls_tokens

    def forward(self, x):
        cls_token = self.forward_features(x)
        cls_token = self.head(cls_token[:, 0])
        return cls_token


# @register_model
# def pit_b(pretrained, **kwargs):
#     model = PoolingTransformer(
#         image_size=224,
#         patch_size=14,
#         stride=7,
#         base_dims=[64, 64, 64],
#         depth=[3, 6, 4],
#         heads=[4, 8, 16],
#         mlp_ratio=4,
#         **kwargs
#     )
#     if pretrained:
#         state_dict = \
#         torch.load('weights/pit_b_820.pth', map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model

@register_model
def tsr_pit_b_nwpu(pretrained, **kwargs):
    model = TSRPoolingTransformer(
        patch_size=8,
        stride=4,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('weights/pit_b_820.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    model.default_cfg = _rs_cfg()
    return model


@register_model
def tsr_pit_b_cifar(pretrained, **kwargs):
    model = TSRPoolingTransformer_cifar(
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('weights/pit_b_820.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    model.default_cfg = _rs_cfg()
    return model


# @register_model
# def pit_s(pretrained, **kwargs):
#     model = PoolingTransformer(
#         image_size=224,
#         patch_size=16,
#         stride=8,
#         base_dims=[48, 48, 48],
#         depth=[2, 6, 4],
#         heads=[3, 6, 12],
#         mlp_ratio=4,
#         **kwargs
#     )
#     if pretrained:
#         state_dict = \
#         torch.load('weights/pit_s_809.pth', map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model


@register_model
def tsr_pit_s_nwpu(pretrained, **kwargs):
    model = TSRPoolingTransformer(
        patch_size=8,
        stride=4,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('weights/pit_s_809.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    model.default_cfg = _rs_cfg()
    return model



# @register_model
# def pit_ti_nwpu(pretrained, **kwargs):
#     model = PoolingTransformer(
#         patch_size=16,
#         stride=8,
#         base_dims=[32, 32, 32],
#         depth=[2, 6, 4],
#         heads=[2, 4, 8],
#         mlp_ratio=4,
#         **kwargs
#     )
#     if pretrained:
#         state_dict = \
#         torch.load('weights/pit_ti_730.pth', map_location='cpu')
#         model.load_state_dict(state_dict)
#     model.default_cfg = _NWPURESISC45_cfg()
#     return model

@register_model
def tsr_pit_ti_nwpu(pretrained, **kwargs):
    model = TSRPoolingTransformer(
        patch_size=8,
        stride=4,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('weights/pit_s_809.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    model.default_cfg = _rs_cfg()
    return model

@register_model
def tsr_pit_ti_cifar(pretrained, **kwargs):
    model = TSRPoolingTransformer_cifar(
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('weights/pit_s_809.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    model.default_cfg = _cifar10_cfg()
    return model