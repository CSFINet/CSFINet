# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from einops import rearrange
import numpy as np
from torch.nn import init
from torch.autograd import Variable
import scipy.misc
from os.path import join as pjoin
from scipy import ndimage
import math
import cv2
import datetime
from PIL import Image
import time
from torchvision import transforms
from typing import Optional
from einops import rearrange

np.set_printoptions(threshold=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ['CSFINet']


class CSFINet(nn.Module):

    def __init__(self, input_channel, n_classes, kernel_size=3, feature_scale=4, decoder="vanilla", bias=True,
                 is_deconv=True, is_batchnorm=True, selfeat=False, shift_n=5, auxseg=False):
        super(CSFINet, self).__init__()
        self.is_deconv = is_deconv
        self.input_channel = input_channel
        self.n_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.decoder = decoder
        self.kernel_size = kernel_size
        self.bias = bias
        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        # H,W,C -> H/2,W/2,64
        self.conv1 = unetConv2(self.input_channel, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        # H/2,W/2,64 -> H/4,W/4,128
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        # H/4,W/4,128 ->H/8,W/8,256
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        # H/8,W/8,256 -> H/16,W/16,512
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        # H/16,W/16,512 -> H/16,W/16,1024
        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.p3m = PatchMerging(dim=256, hlong=56, wlong=56)
        self.p4m = PatchMerging(dim=512, hlong=56, wlong=56)
        self.norm3 = nn.LayerNorm(256)
        self.norm4 = nn.LayerNorm(512)
        self.norm5 = nn.LayerNorm(1024)
        self.piu_trans = PIU(dim=256)
        self.cont1 = nn.Conv2d(in_channels=256 * 2, out_channels=256, kernel_size=1)
        self.cont2 = nn.Conv2d(in_channels=512 * 2, out_channels=512, kernel_size=1)
        self.cont3 = nn.Conv2d(in_channels=1024 * 2, out_channels=1024, kernel_size=1)

        self.conh0 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
        self.conh1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.conh2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conh3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.conh4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.swin_former = SwinTransformer(patch_size=4, in_chans=3, num_classes=1000, embed=(256, 512, 1024, 1024),
                                           embed_dim=256,
                                           depths=(2, 2, 6, 2),
                                           num_heads=(4, 8, 16, 16), window_size=7, mlp_ratio=4., qkv_bias=True,
                                           drop_rate=0.,
                                           attn_drop_rate=0.,
                                           drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True,
                                           use_checkpoint=False,
                                           )

        self.flowconv1 = nn.Conv2d(1024 * 2, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv1 = nn.Conv2d(1024 * 2, 1024, kernel_size=3, stride=1, padding=1)
        self.flowconv2 = nn.Conv2d(512 + 512, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv2 = nn.Conv2d(1024 + 512, 512, kernel_size=3, stride=1, padding=1)
        self.flowconv3 = nn.Conv2d(256 + 256, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv3 = nn.Conv2d(256 + 512, 256, kernel_size=3, stride=1, padding=1)
        self.flowconv4 = nn.Conv2d(128 + 128, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv4 = nn.Conv2d(128 + 256, 128, kernel_size=3, stride=1, padding=1)
        self.flowconv5 = nn.Conv2d(64 + 64, 2, kernel_size=3, stride=1, padding=1)
        self.vanilla_conv5 = nn.Conv2d(64 + 128, 64, kernel_size=3, stride=1, padding=1)
        self.convp2 = nn.Sequential(
            nn.Conv2d(filters[0], self.n_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        ## -------------Score Block in Decoder--------------    
        self.score_block1 = nn.Sequential(
            nn.Conv2d(filters[0], n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block2 = nn.Sequential(
            nn.Conv2d(filters[1], n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block3 = nn.Sequential(
            nn.Conv2d(filters[2], n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block4 = nn.Sequential(
            nn.Conv2d(filters[3], n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        self.score_block5 = nn.Sequential(
            nn.Conv2d(filters[4], self.n_classes, 5, padding=2),
            nn.BatchNorm2d(self.n_classes),
            nn.ReLU(inplace=True)
        )

        # self.convp2 = nn.Conv2d(self.n_classes * 5, self.n_classes, kernel_size=3, stride=1, padding=1)
        self.bnp2 = nn.BatchNorm2d(self.n_classes)
        self.relup2 = nn.ReLU(inplace=True)
        self.convp3 = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1, stride=1, padding=0)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def _upsample(self, x, y, scale=1):  # the size of x is as y
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _init_cell_state(self, tensor):
        return torch.zeros(tensor.size())

    # FIU structure
    def FIU(self, featmap_front, featmap_latter, flow):
        B, C, H, W = featmap_front.size()
        flow = flow.permute(0, 2, 3, 1)
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid = torch.stack((grid_x, grid_y), 2).float()
        grid.requires_grad = False
        grid = grid.type_as(featmap_latter)
        vgrid = grid + flow
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(featmap_latter, vgrid_scaled, mode='bilinear', padding_mode='zeros')
        output = output + featmap_front
        return output

    def forward(self, inputs):
        trans_feature = []
        x5 = self.conv1(inputs)

        x4 = self.maxpool1(x5)
        x4 = self.conv2(x4)

        x3 = self.maxpool2(x4)
        x3 = self.conv3(x3)

        [b3, c3, h33, w33] = x3.size()
        x3_reshape = rearrange(x3, 'b c h w -> b (h w) c')
        sw3, h3_, w3_ = self.swin_former.layers[0](x3_reshape, h33, w33)

        sw3 = self.norm3(sw3)
        x3_trans = rearrange(sw3, 'b (h w ) c -> b c h w', h=h33, w=w33)

        x3_trans = F.upsample(x3_trans, size=(56, 56), mode='bilinear')
        trans_feature.append(x3_trans)
        sw32 = self.p3m(sw3, h33, w33)
        sw32_reashape = rearrange(sw32, 'b (h w) c -> b c h w', h=(h33 + 1) // 2, w=(w33 + 1) // 2)
        x2 = self.maxpool3(x3)
        x2 = self.conv4(x2)

        sk23 = sw32_reashape + x2
        [b2, c2, h22, w22] = x2.size()
        x2_reshape = rearrange(sk23, 'b c h w -> b (h w) c')
        sw2, h2_, w2_ = self.swin_former.layers[1](x2_reshape, h22, w22)
        sw2 = self.norm4(sw2)
        x2_trans = rearrange(sw2, 'b (h w ) c -> b c h w', h=h22, w=w22)
        x2_trans = F.upsample(x2_trans, size=(28, 28), mode='bilinear')
        trans_feature.append(x2_trans)
        sw21 = self.p4m(sw2, h22, w22)
        sw32_reashape = rearrange(sw21, 'b (h w) c -> b c h w', h=(h22 + 1) // 2, w=(w22 + 1) // 2)  # 1 1024
        x1 = self.maxpool4(x2)
        x1 = self.conv5(x1)
        [b1, c1, h11, w11] = x1.size()
        sw20 = x1 + sw32_reashape
        sw20 = rearrange(sw20, 'b c h w -> b (h w) c')
        sw200, h1_, w1_ = self.swin_former.layers[2](sw20, h11, w11)
        sw00 = self.norm5(sw200)
        x1_trans = rearrange(sw00, 'b (h w ) c -> b c h w', h=h11, w=w11)
        x1_trans = F.upsample(x1_trans, size=(14, 14), mode='bilinear')
        trans_feature.append(x1_trans)
        # PIU
        feature_trans = self.piu_trans(trans_feature)
        fea1 = feature_trans[0]
        fea2 = feature_trans[1]
        fea3 = feature_trans[2]

        t3 = torch.cat((fea1, x3), dim=1)
        t3 = self.cont1(t3)

        t2 = torch.cat((fea2, x2), dim=1)
        t2 = self.cont2(t2)

        t1 = torch.cat((fea3, x1), dim=1)
        t1 = self.cont3(t1)

        h0 = self._init_cell_state(t1).cuda()
        if self.decoder == "vanilla":
            h0 = self.conh0(h0)
            fuse = torch.cat((t1, self._upsample(h0, t1)), 1)
            flow = self.flowconv1(fuse)
            h1 = self.FIU(t1, h0, flow)
            h1 = torch.relu(h1)

            h1 = self.conh1(h1)
            fuse = torch.cat((t2, self._upsample(h1, t2)), 1)
            flow = self.flowconv2(fuse)
            h2 = self.FIU(t2, h1, flow)
            h2 = torch.relu(h2)

            h2 = self.conh2(h2)
            fuse = torch.cat((t3, self._upsample(h2, t3)), 1)
            flow = self.flowconv3(fuse)
            h3 = self.FIU(t3, h2, flow)
            h3 = torch.relu(h3)

            h3 = self.conh3(h3)
            fuse = torch.cat((x4, self._upsample(h3, x4)), 1)
            flow = self.flowconv4(fuse)
            h4 = self.FIU(x4, h3, flow)
            h4 = torch.relu(h4)

            h4 = self.conh4(h4)
            fuse = torch.cat((x5, self._upsample(h4, x5)), 1)
            flow = self.flowconv5(fuse)
            h5 = self.FIU(x5, h4, flow)
            h5 = torch.relu(h5)

        out = self.convp2(h5)
        return out


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )

                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            # x = torch.tensor(x, dtype=torch.float32)
            x = conv(x)

        return x


from torch.nn import init


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


# PIU structure
class PIU(nn.Module):
    def __init__(self, dim=256, num=1):
        super(PIU, self).__init__()
        self.ini_win_size = 2
        self.channels = [256, 512, 1024]
        self.dim = dim
        self.depth = 3
        self.fc_module = nn.ModuleList()
        self.fc_rever_module = nn.ModuleList()
        self.num = num
        for i in range(self.depth):
            self.fc_module.append(nn.Linear(self.channels[i], self.dim))

        for i in range(self.depth):
            self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))

        self.group_attention = []
        for i in range(self.num):
            self.group_attention.append(attentionBlock(dim))
        self.group_attention = nn.Sequential(*self.group_attention)
        self.split_list = [4 * 4, 2 * 2, 1 * 1]

    def forward(self, x):

        x = [self.fc_module[i](item.permute(0, 2, 3, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
        # Patch Matching
        for j, item in enumerate(x):
            B, H, W, C = item.shape
            win_size = self.ini_win_size ** (self.depth - j - 1)
            item = item.reshape(B, H // win_size, win_size, W // win_size, win_size, C).permute(0, 1, 3, 2, 4,
                                                                                                5).contiguous()
            item = item.reshape(B, H // win_size, W // win_size, win_size * win_size, C).contiguous()
            x[j] = item
        x = tuple(x)
        x = torch.cat(x, dim=-2)  # (B, H // win, W // win, N, C) // N = mh * mw
        # Scale fusion
        for i in range(self.num):
            x = self.group_attention[i](x)  # (B, H // win_size, W // win_size, win_size*win_size, C)

        x = torch.split(x, self.split_list, dim=-2)
        x = list(x)
        # patch reversion
        for j, item in enumerate(x):
            B, num_blocks, _, N, C = item.shape
            win_size = self.ini_win_size ** (self.depth - j - 1)
            item = item.reshape(B, num_blocks, num_blocks, win_size, win_size, C).permute(0, 1, 3, 2, 4,
                                                                                          5).contiguous().reshape(B,
                                                                                                                  num_blocks * win_size,
                                                                                                                  num_blocks * win_size,
                                                                                                                  C)
            item = self.fc_rever_module[j](item).permute(0, 3, 1, 2).contiguous()
            x[j] = item
        return x


class attentionBlock(nn.Module):
    def __init__(self, dim):
        super(attentionBlock, self).__init__()
        self.SlayerNorm_1 = nn.LayerNorm(dim, eps=1e-6)
        self.SlayerNorm_2 = nn.LayerNorm(dim, eps=1e-6)
        self.Attention = MultiScaleAtten(dim)
        self.FFN = MLP(dim)

    def forward(self, x):
        h = x  # (B, N, H) // b n c
        x = self.SlayerNorm_1(x)

        x = self.Attention(x)  # padding 到right_size
        x = h + x

        h = x
        x = self.SlayerNorm_2(x)

        x = self.FFN(x)
        x = h + x

        return x


class MLP(nn.Module):

    def __init__(self, dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.functional.gelu
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# swin_transformer structure , You can ignore the following code
def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mh, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    # 还原成feature map, H,W对应分割前的feature map 的H W
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            # 宽度方向的右侧 高度方向的底部 进行padding
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W  # 通过下采样后的宽度和高度


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, hlong, wlong, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.hlong = hlong
        self.wlong = wlong
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, hlong, wlong):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == hlong * wlong, "input feature has wrong size"

        x = x.view(B, hlong, wlong, C)

        # padding
        # 需要对特征图下采样两倍，如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (hlong % 2 == 1) or (wlong % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            # 前两个代表channel方向，中间代表宽度方向，最后代表高度方向
            # 代表在宽度方向的右侧，高度方向的下侧，从而保证是2的整数倍，然后可以下采样
            x = F.pad(x, (0, 0, 0, wlong % 2, 0, hlong % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]  蓝色的
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]  绿色的
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]  黄色的
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]  红色的
        # 在channel方向进行拼接
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        # 通过view方向在高宽的方向进行展平
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]
        x = self.norm(x)
        # 通过全连接层调整channel
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]
        # 生成位置偏移表
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])  # 假设windows=2的话，coords_h=0,1, coords_w=0,1
        coords_w = torch.arange(self.window_size[1])
        # meshgrid返回的是两个二tensor 通过stack方法进行拼接得到2
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords,
                                       1)  # [2, Mh*Mw]  [[0,0,1,1],[0,1,0,1]]第一行表示的feature对应的行标，第二行对应的是feature对应的列表
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # 分别获得qkv
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)  # 加一个batch维度

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:

            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        if self.shift_size > 0:

            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint

        self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):

        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]

        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]

        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, H, W):

        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=256, depths=(2, 2, 6, 2), num_heads=(4, 8, 16, 16), embed=(256, 512, 1024, 1024),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):

        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(  ##dim=int(embed_dim * 2 ** i_layer),
                dim=embed[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)  # 自适应的全局平均池化，把最后一个维度变成1
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class MultiScaleAtten(nn.Module):
    def __init__(self, dim):
        super(MultiScaleAtten, self).__init__()
        self.qkv_linear = nn.Linear(dim, dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.num_head = 8
        self.scale = (dim // self.num_head) ** 0.5

    def forward(self, x):
        B, num_blocks, _, _, C = x.shape  # (B, num_blocks, num_blocks, N, C)
        qkv = self.qkv_linear(x).reshape(B, num_blocks, num_blocks, -1, 3, self.num_head, C // self.num_head).permute(4,
                                                                                                                      0,
                                                                                                                      1,
                                                                                                                      2,
                                                                                                                      5,
                                                                                                                      3,
                                                                                                                      6).contiguous()  # (3, B, num_block, num_block, head, N, C)
        q, k, v = qkv[0], qkv[1], qkv[2]
        atten = q @ k.transpose(-1, -2).contiguous()
        atten = self.softmax(atten)
        atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks, num_blocks, -1, C)
        atten_value = self.proj(atten_value)  # (B, num_block, num_block, N, C)
        return atten_value
