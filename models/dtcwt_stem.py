import torch
import torch.nn as nn
from einops import rearrange

from dtcwt.transform2d import DTCWTForward
from dtcwt.lowlevel import q2c

class DTCWTStem_bhwc(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size=3, stride=2, padding=1, norm_layer=None, f_decorrelation=True):
        super().__init__()
        self.f_decorrelation = f_decorrelation
        self.dtcwt_layer = DTCWTForward(J=1)

        self.modulate = nn.Sequential(
            nn.Conv2d(in_chans * 15, embed_dim,
                      kernel_size=patch_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True))

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _dtcwt(self, x):
        x_l, x_hlist = self.dtcwt_layer(x)
        (ll1, ll2), (ll3, ll4) = q2c(x_l)
        x_l = torch.stack([ll1, ll2, ll4], dim=1)
        x_h = x_hlist[-1]  # B C D H W RI
        x_h = rearrange(x_h, 'b c d h w r -> b (d r) c h w')
        x = torch.cat([x_l, x_h], dim=1)  # b 15 c h w
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        return x

    def forward(self, x):
        '''
        :param x: expected shape: b c h w
        :return: output shape: b h w c
        '''
        x = self._dtcwt(x)
        x = self.modulate(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        if self.f_decorrelation:
            self.f_x = x
        return x

class DTCWTStem_bchw(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size=3, stride=2, padding=1, norm_layer=None, f_decorrelation=True):
        super().__init__()
        self.f_decorrelation = f_decorrelation
        self.dtcwt_layer = DTCWTForward(J=1)

        self.modulate = nn.Sequential(
            nn.Conv2d(in_chans * 15, embed_dim,
                      kernel_size=patch_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True))

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _dtcwt(self, x):
        x_l, x_hlist = self.dtcwt_layer(x)
        (ll1, ll2), (ll3, ll4) = q2c(x_l)
        x_l = torch.stack([ll1, ll2, ll4], dim=1)
        x_h = x_hlist[-1]  # B C D H W RI
        x_h = rearrange(x_h, 'b c d h w r -> b (d r) c h w')
        x = torch.cat([x_l, x_h], dim=1)  # b 15 c h w
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        return x

    def forward(self, x):
        '''
        :param x: expected shape: b c h w
        :return: output shape: b c h w
        '''
        x = self._dtcwt(x)
        x = self.modulate(x)
        x = x.permute(0, 2, 3, 1)  # b h w c
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        if self.f_decorrelation:
            self.f_x = x
        return x


class DTCWTStem_bnc(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size=3, stride=2, padding=1, norm_layer=None, f_decorrelation=True):
        super().__init__()
        self.f_decorrelation = f_decorrelation
        self.dtcwt_layer = DTCWTForward(J=1)

        self.modulate = nn.Sequential(
            nn.Conv2d(in_chans * 15, embed_dim,
                      kernel_size=patch_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True))

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _dtcwt(self, x):
        x_l, x_hlist = self.dtcwt_layer(x)
        (ll1, ll2), (ll3, ll4) = q2c(x_l)
        x_l = torch.stack([ll1, ll2, ll4], dim=1)
        x_h = x_hlist[-1]  # B C D H W RI
        x_h = rearrange(x_h, 'b c d h w r -> b (d r) c h w')
        x = torch.cat([x_l, x_h], dim=1)  # b 15 c h w
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        return x

    def forward(self, x):
        '''
        :param x: expected shape: b c h w
        :return: output shape: b (h w) c
        '''
        x = self._dtcwt(x)
        x = self.modulate(x)
        x = x.permute(0, 2, 3, 1)  # b h w c
        x = x.view(x.shape[0], -1, x.shape[-1])
        x = self.norm(x)
        if self.f_decorrelation:
            self.f_x = x
        return x