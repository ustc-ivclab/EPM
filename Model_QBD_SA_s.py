import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
from Metrics import block_qtnode_norm

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, padding, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            # nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            # nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Luma_Q_Net(nn.Module):  # luma QT depth prediction
    def __init__(self, classification=True, c_ratio=1.0):
        super(Luma_Q_Net, self).__init__()
        from models.swin_transformer import SwinTransformer

        if classification:
            # classification
            if c_ratio < 0.0625:
                # 再降低维度已经不可能，将windows_size从8将为4，将depth将为2,2,2
                self.model = SwinTransformer(img_size=64, patch_size=2, in_chans=1, num_classes=5,
                        embed_dim=6, depths=[2, 2, 2], num_heads=[3, 3, 3],
                        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                        use_checkpoint=False, fused_window_process=False, input_size=64,
                        classification=True)
            else:
                self.model = SwinTransformer(img_size=64, patch_size=2, in_chans=1, num_classes=5,
                        embed_dim=int(96 * c_ratio), depths=[2, 2, 4], num_heads=[3, 6, 12],
                        window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                        use_checkpoint=False, fused_window_process=False, input_size=64,
                        classification=True)
        else:
            # regression
            self.model = SwinTransformer(img_size=64, patch_size=2, in_chans=1, num_classes=1,
                    embed_dim=96, depths=[2, 2, 2], num_heads=[3, 6, 12],
                    window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                    norm_layer=None, ape=False, patch_norm=False,
                    use_checkpoint=False, fused_window_process=False, input_size=64)

        self.classification = classification


    def forward(self, x):  # input 1*68*68
        x = x[:,:,2:66, 2:66]
        out = self.model(x)
        B, H, W, C =  out.shape
        if not self.classification:
            out = out.view(B, 1, H, W)
        return out  # qt depth map

class Luma_MSBD_Net(nn.Module):  # luma bt depth and direction prediction
    def __init__(self, classification=True, c_ratio=1.0):
        super(Luma_MSBD_Net, self).__init__()
        self.classification = classification
        from models.SwinT_BD import SwinTransformer_MSBD
        if classification:
            if c_ratio == 0.0625:
                self.depth_model = SwinTransformer_MSBD(img_size=64, patch_size=2, in_chans=2, num_classes=3,
                        embed_dim=int(96 * c_ratio), depths=[2, 4], num_heads=[3, 3],
                        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                        use_checkpoint=False, fused_window_process=False, input_size=64 ,classification=True)
            elif c_ratio < 0.0625:
                self.depth_model = SwinTransformer_MSBD(img_size=64, patch_size=2, in_chans=2, num_classes=3,
                        embed_dim=6, depths=[2, 2], num_heads=[3, 3],
                        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                        use_checkpoint=False, fused_window_process=False, input_size=64 ,classification=True)
            else:
                self.depth_model = SwinTransformer_MSBD(img_size=64, patch_size=2, in_chans=2, num_classes=3,
                        embed_dim=int(96 * c_ratio), depths=[2, 4], num_heads=[4, 8],
                        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                        use_checkpoint=False, fused_window_process=False, input_size=64 ,classification=True)

            if c_ratio == 0.125:
                # 更改num_heads
                self.dire_model = SwinTransformer_MSBD(img_size=64, patch_size=2, in_chans=2, num_classes=3,
                        embed_dim=int(48 * c_ratio), depths=[2, 4], num_heads=[3, 6],
                        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                        use_checkpoint=False, fused_window_process=False, input_size=64 ,classification=True)
            elif c_ratio == 0.0625:
                self.dire_model = SwinTransformer_MSBD(img_size=64, patch_size=2, in_chans=2, num_classes=3,
                        embed_dim=int(48 * c_ratio), depths=[2, 2], num_heads=[3, 3],
                        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                        use_checkpoint=False, fused_window_process=False, input_size=64 ,classification=True)
            elif c_ratio < 0.0625:
                self.dire_model = SwinTransformer_MSBD(img_size=64, patch_size=2, in_chans=2, num_classes=3,
                        embed_dim=3, depths=[2, 2], num_heads=[3, 3],
                        window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                        use_checkpoint=False, fused_window_process=False, input_size=64 ,classification=True)
            else:
                self.dire_model = SwinTransformer_MSBD(img_size=64, patch_size=2, in_chans=2, num_classes=3,
                        embed_dim=int(48 * c_ratio), depths=[2, 4], num_heads=[4, 8],
                        window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                        use_checkpoint=False, fused_window_process=False, input_size=64 ,classification=True)
        else:
            self.model = SwinTransformer_MSBD(img_size=64, patch_size=2, in_chans=2, num_classes=2,
                    embed_dim=96, depths=[4, 8], num_heads=[4, 8],
                    window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                    use_checkpoint=False, fused_window_process=False, input_size=64)
    def forward(self, x, x1):  # input image block + qt depth map
        x = x[:,:,2:66, 2:66]
        x1_1 = (F.interpolate(x1, scale_factor=8))
        input = torch.cat([x, x1_1], 1)
        if self.classification:
            M0, M1, M2  = self.depth_model(input)
            D0, D1, D2  = self.dire_model(input)
            return M0, M1, M2, D0, D1, D2
        else:
            M0, M1, M2  = self.model(input)  # (b,2,16,16)
            M1[:,0] = M0[:,0] + M1[:,0]
            M2[:,0] = M1[:,0] + M2[:,0]
        return M0, M1, M2

class Chroma_Q_Net(nn.Module):  # chroma QT depth prediction
    def __init__(self, classification=True, c_ratio=1.0):
        super(Chroma_Q_Net, self).__init__()
        from models.swin_transformer import SwinTransformer

        if classification:
            # classification
            self.model = SwinTransformer(img_size=32, patch_size=2, in_chans=3, num_classes=5,
                    embed_dim=int(96 * c_ratio), depths=[2, 4], num_heads=[3, 12],
                    window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                    use_checkpoint=False, fused_window_process=False, input_size=64,
                    classification=True, is_Luma=False)
        else:
            # regression
            self.model = SwinTransformer(img_size=32, patch_size=2, in_chans=3, num_classes=1,
                    embed_dim=96, depths=[4, 4, 4], num_heads=[3, 6, 12],
                    window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                    norm_layer=None, ape=False, patch_norm=False,
                    use_checkpoint=False, fused_window_process=False, input_size=64, is_Luma=False)

        self.classification = classification


    def forward(self, x):  # input 1*68*68
        x = x[:,:,1:33, 1:33]
        out = self.model(x)
        B, H, W, C =  out.shape
        if not self.classification:
            out = out.view(B, 1, H, W)
        return out  # qt depth map

class Chroma_MSBD_Net(nn.Module):  # chroma bt depth and direction prediction
    def __init__(self, classification=True, c_ratio=1.0):
        super(Chroma_MSBD_Net, self).__init__()
        self.classification = classification
        from models.SwinT_BD import SwinTransformer_MSBD
        if classification:
            self.depth_model = SwinTransformer_MSBD(img_size=32, patch_size=2, in_chans=4, num_classes=3,
                    embed_dim=int(96 * c_ratio), depths=[2, 4], num_heads=[4, 8],
                    window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                    use_checkpoint=False, fused_window_process=False, input_size=64 ,classification=True, is_Luma=False)
            self.dire_model = SwinTransformer_MSBD(img_size=32, patch_size=2, in_chans=4, num_classes=3,
                    embed_dim=int(96 * c_ratio), depths=[2, 4], num_heads=[4, 8],
                    window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                    use_checkpoint=False, fused_window_process=False, input_size=64 ,classification=True, is_Luma=False)
        else:
            self.model = SwinTransformer_MSBD(img_size=32, patch_size=2, in_chans=4, num_classes=2,
                    embed_dim=96, depths=[4, 8], num_heads=[4, 8],
                    window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                    norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                    use_checkpoint=False, fused_window_process=False, input_size=64, is_Luma=False)

    def forward(self, x, x1):  # input image block + qt depth map
        x = x[:,:,1:33, 1:33]
        x1_1 = (F.interpolate(x1, scale_factor=4))
        input = torch.cat([x, x1_1], 1)
        if self.classification:
            M0, M1, M2  = self.depth_model(input)
            D0, D1, D2  = self.dire_model(input)
            return M0, M1, M2, D0, D1, D2
        else:
            M0, M1, M2  = self.model(input)  # (b,2,16,16)
            M1[:,0] = M0[:,0] + M1[:,0]
            M2[:,0] = M1[:,0] + M2[:,0]
        return M0, M1, M2



if __name__ == "__main__":
    from thop import profile

    # model = Chroma_Q_Net(classification=True).cuda()
    # input = torch.zeros(1, 3, 34, 34).cuda()
    # flops, params = profile(model, (input,))
    # print('QTNet-SA Chroma')
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    # model = Chroma_MSBD_Net(classification=True).cuda()
    # input = torch.zeros(1, 3, 34, 34).cuda()
    # qt_map = torch.zeros(1, 1, 8, 8).cuda()
    # flops, params = profile(model, (input, qt_map))
    # print('QTNet-SA, Chroma')
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    c_ratio = 0.03125
    model = Luma_Q_Net(classification=True, c_ratio=c_ratio).cuda()
    input = torch.zeros(1, 1, 68, 68).cuda()
    flops, params = profile(model, (input,))
    qt_flops = flops
    qt_params = params
    print('QTNet-SA')
    print('flops: ', flops, 'params: ', params)
    print('flops: %.3f G, params: %.3f M' % (flops / 1000000000.0, params / 1000000.0))


    model = Luma_MSBD_Net(classification=True, c_ratio=c_ratio).cuda()
    input = torch.zeros(1, 1, 68, 68).cuda()
    qt_map = torch.zeros(1, 1, 8, 8).cuda()
    flops, params = profile(model, (input, qt_map))
    mt_flops = flops
    mt_params = params
    print('QTNet-SA')
    print('flops: ', flops, 'params: ', params)
    print('flops: %.3f G, params: %.3f M' % (flops / 1000000000.0, params / 1000000.0))
    
    print("total flops: %.4f"%((qt_flops+mt_flops) / 1000000000))
    print("total params: %.4f"%((qt_params+mt_params) / 1000000))