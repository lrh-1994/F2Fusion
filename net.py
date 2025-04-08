import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import math
import numpy as np
#from network_swinfusion1 import SwinFusion
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
EPSILON = 1e-10
import time
from functools import partial
from typing import Optional, Callable
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
        
class network_vr(nn.Module):
    def __init__(self):
        super(network_vr, self).__init__()
        self.vi_encoder=network_vi_encoder()
        self.ir_encoder=network_ir_encoder()
        self.decoder=network_decoder()
        
    def forward(self,vi,ir,dwt,idwt):
    
        vi64_l,vi32_l,vi16_l,vi64_h,vi32_h,vi16_h=self.vi_encoder(vi,dwt)	
        ir64_l,ir32_l,ir16_l,ir64_h,ir32_h,ir16_h=self.ir_encoder(ir,dwt)
        vi_out=self.decoder(vi64_l,vi32_l,vi16_l,vi64_h,vi32_h,vi16_h,idwt)
        ir_out=self.decoder(ir64_l,ir32_l,ir16_l,ir64_h,ir32_h,ir16_h,idwt)
        
        return vi_out,ir_out
        
class network_fusion(nn.Module):
    def __init__(self):
        super(network_fusion, self).__init__()
        self.feature_fusion =feature_fusion()
        self.vr=network_vr()
    def forward(self,vi64_l,ir64_l,vi32_l,ir32_l,vi16_l,ir16_l):
        return 0
         
class feature_fusion(nn.Module):
    def __init__(self):
        super(feature_fusion, self).__init__()

        self.fusion1_conv = ConvFusion1()
        self.fusion2_conv = ConvFusion2()
        self.fusion3_conv = ConvFusion3()
        
    def forward(self,vi64_l,ir64_l,vi32_l,ir32_l,vi16_l,ir16_l):
        
        f64_l,out_s64,out_i64,out_d64 = self.fusion1_conv(vi64_l,ir64_l,vi64_l.size(2),vi64_l.size(3))
        
        f32_l,out_s32,out_i32,out_d32 = self.fusion2_conv(vi32_l,ir32_l,vi32_l.size(2),vi32_l.size(3))
        
        f16_l,out_s16,out_i16,out_d16 = self.fusion3_conv(vi16_l,ir16_l,vi16_l.size(2),vi16_l.size(3))
        
        return f64_l,f32_l,f16_l,out_s64,out_i64,out_d64,out_s32,out_i32,out_d32,out_s16,out_i16,out_d16

class network_vi_encoder(nn.Module):
    def __init__(self):
        super(network_vi_encoder,self).__init__()
        #1\self.pos_drop = nn.Dropout(p=0.)

        self.swintrans_vi1=VSSLayer(dim=64)
        self.swintrans_vi2=VSSLayer(dim=64)
        self.swintrans_vi3=VSSLayer(dim=64)

        self.conv_vi1 = nn.Conv2d(1, 64, kernel_size=3, stride=1,padding=1)
        

    def forward(self, vi, dwt):
        
        vi1_s=self.conv_vi1(vi)
        
        vi1=self.swintrans_vi1(vi1_s)
        
        vi64_l,vi64_h = dwt(vi1)
        
        vi2=self.swintrans_vi2(vi64_l)
        
        vi32_l,vi32_h = dwt(vi2)
        
        vi3=self.swintrans_vi3(vi32_l)
        
        vi16_l,vi16_h = dwt(vi3)
        
        return vi64_l,vi32_l,vi16_l,vi64_h,vi32_h,vi16_h

class network_ir_encoder(nn.Module):
    def __init__(self):
        super(network_ir_encoder,self).__init__()

        #1\self.pos_drop = nn.Dropout(p=0.)
        self.swintrans_ir1=VSSLayer(dim=64)
        self.swintrans_ir2=VSSLayer(dim=64)
        self.swintrans_ir3=VSSLayer(dim=64)
        
        self.conv_ir1 = nn.Conv2d(1, 64, kernel_size=3, stride=1,padding=1)
        

    def forward(self, ir, dwt):
        ir1_s=self.conv_ir1(ir)
        ir1=self.swintrans_ir1(ir1_s)
        ir64_l,ir64_h = dwt(ir1)
        ir2=self.swintrans_ir2(ir64_l)
        ir32_l,ir32_h = dwt(ir2)
        ir3=self.swintrans_ir3(ir32_l)
        ir16_l,ir16_h = dwt(ir3)
        
        return ir64_l,ir32_l,ir16_l,ir64_h,ir32_h,ir16_h

class network_decoder(nn.Module):
    def __init__(self):
        super(network_decoder, self).__init__()
        
        self.sff_fusion1 = SKFF(64,2,8,bias=False)
        self.sff_fusion2 = SKFF(64,2,8,bias=False)
        self.sff_fusion3 = SKFF(64,2,8,bias=False)
        self.swintrans_d1=VSSLayer(dim=64)
        self.swintrans_d2=VSSLayer(dim=64)
        self.swintrans_d3=VSSLayer(dim=64)
        
        self.relu_d = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_d = nn.Conv2d(64, 32, kernel_size=3, stride=1,padding=1)
        self.conv_o = nn.Conv2d(32, 1, kernel_size=3, stride=1,padding=1)
        self.relu_o = nn.Sigmoid()


    def forward(self,f64_l,f32_l,f16_l,f64_h,f32_h,f16_h,idwt):
    
        d_16 = self.swintrans_d1(f16_l)
        d_32 = self.sff_fusion1([f32_l,idwt([d_16,f16_h])])#f32_l+idwt([d_16,f16_h])
        
        d_32 = self.swintrans_d2(d_32)
        d_64 = self.sff_fusion2([f64_l,idwt([d_32,f32_h])])
        
        d_64 =  self.swintrans_d3(d_64)
        d_o= idwt([d_64,f64_h])

        result_d = self.conv_d(d_o)
        result_d = self.relu_d(result_d)
        result_o = self.conv_o(result_d)
        result_o = self.relu_o(result_o)
        
        return result_o

class LocalFusion(nn.Module):
    def __init__(self):
        super(LocalFusion, self).__init__()
        
        self.relu_ir1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu_ir2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu_vi1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu_vi2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_ir1 = nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1)
        self.conv_ir2 = nn.Conv2d(64, 32, kernel_size=3, stride=1,padding=1)
        self.conv_vi1 = nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1)
        self.conv_vi2 = nn.Conv2d(64, 32, kernel_size=3, stride=1,padding=1)


    def forward(self,ir,vi):
    
        ir1 = self.conv_ir1(ir)
        ir1 = self.relu_ir1(ir1)
        
        vi1 = self.conv_vi1(vi)
        vi1 = self.relu_vi1(vi1)
        
        mix1 = torch.cat([ir1[:,0::2,:,:],vi1[:,0::2,:,:]],1)
        mix2 = torch.cat([ir1[:,1::2,:,:],vi1[:,1::2,:,:]],1)
        
        cat1 = self.conv_ir2(mix1)
        cat2 = self.conv_vi2(mix2)
        
        cat1 = self.relu_ir2(cat1)
        cat2 = self.relu_vi2(cat2)
        
        catf = torch.cat([cat1,cat2],1)
        
        return catf

class SKFF(nn.Module):
    def __init__(self, in_channels, height=2,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V 


class ConvFusion1(nn.Module):
    def __init__(self):
        super(ConvFusion1, self).__init__()
        self.sff_fusion = LocalFusion()#SKFF(64,2,8,bias=False)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)#ReLU()
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)#ReLU()
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)#ReLU()

        self.avgpool3 = torch.nn.AvgPool2d(3,3)
        self.avgpool5 = torch.nn.AvgPool2d(5,3)
        self.avgpool7 = torch.nn.AvgPool2d(7,5)
        self.conv_1_vi1 = nn.Conv2d(192, 64, kernel_size=3, stride=1,padding=1)
        self.conv_1_ir1 = nn.Conv2d(192, 64, kernel_size=3, stride=1,padding=1)
        self.conv_1_all_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1)
        #self.conv_bottle = nn.Conv2d(128, 64, kernel_size=3, stride=1,padding=1)
        #self.act_layer = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.VK=VSSLayer_cross(dim=64)

    def forward(self,vi,ir,m,n):
        vid_3=vi-F.interpolate(self.avgpool3(vi),size=(m,n),mode='bilinear',align_corners=False)
        vid_5=vi-F.interpolate(self.avgpool5(vi),size=(m,n),mode='bilinear',align_corners=False)
        vid_7=vi-F.interpolate(self.avgpool7(vi),size=(m,n),mode='bilinear',align_corners=False)
        ird_3=ir-F.interpolate(self.avgpool3(ir),size=(m,n),mode='bilinear',align_corners=False)
        ird_5=ir-F.interpolate(self.avgpool5(ir),size=(m,n),mode='bilinear',align_corners=False)
        ird_7=ir-F.interpolate(self.avgpool7(ir),size=(m,n),mode='bilinear',align_corners=False)

        vi_detail=self.conv_1_vi1(torch.cat([vid_3,vid_5,vid_7],1))
        vi_detail=self.relu1(vi_detail)
        ir_detail=self.conv_1_ir1(torch.cat([ird_3,ird_5,ird_7],1))
        ir_detail=self.relu2(ir_detail)
        out_detail=vi_detail+ir_detail
        out_detail=self.conv_1_all_1(out_detail)
        out_detail=self.relu3(out_detail)
        
        ins=self.sff_fusion(ir,vi)#[ir,vi]
        
        xir,xvi=self.VK(ir,vi)
        
        #out_salient=self.conv_bottle(torch.cat([xir,xvi],1))
        out_salient=xir+xvi#self.act_layer(out_salient)
        
        sumall=ins+out_detail+out_salient
           
        return sumall,out_salient,ins,out_detail

class ConvFusion2(nn.Module):
    def __init__(self):
        super(ConvFusion2, self).__init__()
        self.sff_fusion = LocalFusion()#SKFF(64,2,8,bias=False)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)#ReLU()
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)#ReLU()
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)#ReLU()

        self.avgpool3 = torch.nn.AvgPool2d(3,3)
        self.avgpool5 = torch.nn.AvgPool2d(5,3)
        self.avgpool7 = torch.nn.AvgPool2d(7,5)
        self.conv_1_vi1 = nn.Conv2d(192, 64, kernel_size=3, stride=1,padding=1)
        self.conv_1_ir1 = nn.Conv2d(192, 64, kernel_size=3, stride=1,padding=1)
        self.conv_1_all_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1)
        #self.conv_bottle = nn.Conv2d(128, 64, kernel_size=3, stride=1,padding=1)
        #self.act_layer = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.VK=VSSLayer_cross(dim=64)

    def forward(self,vi,ir,m,n):
        vid_3=vi-F.interpolate(self.avgpool3(vi),size=(m,n),mode='bilinear',align_corners=False)
        vid_5=vi-F.interpolate(self.avgpool5(vi),size=(m,n),mode='bilinear',align_corners=False)
        vid_7=vi-F.interpolate(self.avgpool7(vi),size=(m,n),mode='bilinear',align_corners=False)
        ird_3=ir-F.interpolate(self.avgpool3(ir),size=(m,n),mode='bilinear',align_corners=False)
        ird_5=ir-F.interpolate(self.avgpool5(ir),size=(m,n),mode='bilinear',align_corners=False)
        ird_7=ir-F.interpolate(self.avgpool7(ir),size=(m,n),mode='bilinear',align_corners=False)

        vi_detail=self.conv_1_vi1(torch.cat([vid_3,vid_5,vid_7],1))
        vi_detail=self.relu1(vi_detail)
        ir_detail=self.conv_1_ir1(torch.cat([ird_3,ird_5,ird_7],1))
        ir_detail=self.relu2(ir_detail)
        out_detail=vi_detail+ir_detail
        out_detail=self.conv_1_all_1(out_detail)
        out_detail=self.relu3(out_detail)
        ins=self.sff_fusion(ir,vi)
        xir,xvi=self.VK(ir,vi)
        
        #out_salient=self.conv_bottle(torch.cat([xir,xvi],1))
        out_salient=xir+xvi#self.act_layer(out_salient)
        
        sumall=ins+out_salient+out_detail
           
        return sumall,out_salient,ins,out_detail

class ConvFusion3(nn.Module):
    def __init__(self):
        super(ConvFusion3, self).__init__()
        self.sff_fusion = LocalFusion()#SKFF(64,2,8,bias=False)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)#ReLU()
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)#ReLU()
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)#ReLU()

        self.avgpool3 = torch.nn.AvgPool2d(3,3)
        self.avgpool5 = torch.nn.AvgPool2d(5,3)
        self.avgpool7 = torch.nn.AvgPool2d(7,5)
        self.conv_1_vi1 = nn.Conv2d(192, 64, kernel_size=3, stride=1,padding=1)
        self.conv_1_ir1 = nn.Conv2d(192, 64, kernel_size=3, stride=1,padding=1)
        self.conv_1_all_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1)
        
        #self.conv_bottle = nn.Conv2d(128, 64, kernel_size=3, stride=1,padding=1)
        #self.act_layer = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.VK=VSSLayer_cross(dim=64)

    def forward(self,vi,ir,m,n):
        vid_3=vi-F.interpolate(self.avgpool3(vi),size=(m,n),mode='bilinear',align_corners=False)
        vid_5=vi-F.interpolate(self.avgpool5(vi),size=(m,n),mode='bilinear',align_corners=False)
        vid_7=vi-F.interpolate(self.avgpool7(vi),size=(m,n),mode='bilinear',align_corners=False)
        ird_3=ir-F.interpolate(self.avgpool3(ir),size=(m,n),mode='bilinear',align_corners=False)
        ird_5=ir-F.interpolate(self.avgpool5(ir),size=(m,n),mode='bilinear',align_corners=False)
        ird_7=ir-F.interpolate(self.avgpool7(ir),size=(m,n),mode='bilinear',align_corners=False)

        vi_detail=self.conv_1_vi1(torch.cat([vid_3,vid_5,vid_7],1))
        vi_detail=self.relu1(vi_detail)
        ir_detail=self.conv_1_ir1(torch.cat([ird_3,ird_5,ird_7],1))
        ir_detail=self.relu2(ir_detail)
        out_detail=vi_detail+ir_detail
        out_detail=self.conv_1_all_1(out_detail)
        out_detail=self.relu3(out_detail)
        ins=self.sff_fusion(ir,vi)
        xir,xvi=self.VK(ir,vi)
        #out_salient=self.conv_bottle(torch.cat([xir,xvi],1))
        out_salient=xir+xvi#self.act_layer(out_salient)
        
        sumall=ins+out_salient+out_detail
           
        return sumall,out_salient,ins,out_detail

class SingleModule(nn.Module):
    def __init__(self, n_channels, n_blocks, act, attention):
        super(SingleModule, self).__init__()
        res_blocks = [rcab_block(n_channels=n_channels, kernel=3, activation=act) for _ in range(n_blocks)]
        self.body_block = nn.Sequential(*res_blocks)
        self.attention = attention
        if attention:
            self.coder = nn.Sequential(DiEnDec(3, act))
            self.dac = nn.Sequential(DAC(n_channels))
            self.hessian3 = nn.Sequential(MSHF(n_channels, kernel=3))
            self.hessian5 = nn.Sequential(MSHF(n_channels, kernel=5))
            self.hessian7 = nn.Sequential(MSHF(n_channels, kernel=7))

    def forward(self, x):
        sz = x.size()
        resin = self.body_block(x)

        if self.attention:
            hessian3 = self.hessian3(resin)
            hessian5 = self.hessian5(resin)
            hessian7 = self.hessian7(resin)
            hessian = torch.cat((torch.mean(hessian3, dim=1, keepdim=True),
                                 torch.mean(hessian5, dim=1, keepdim=True),
                                 torch.mean(hessian7, dim=1, keepdim=True))
                                , 1)
            hessian = self.coder(hessian)
            attention = torch.sigmoid(self.dac[0](hessian.expand(sz), x))
            resout = resin * attention
        else:
            resout = resin

        output = resout #+ x

        return output


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=3, in_chans=3, embed_dim=32, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=1, padding=1)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)#.permute(0, 2, 3, 1)
        #if self.norm is not None:
            #x = self.norm(x)
        return x


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4
        #print('start.....................................')
        #print(x.size())

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        
        #print(x_hwwh.size())
        
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        
        #print(xs.size())

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        
        #print(x_dbl.size())
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        #print(dts.size())
        #print(Bs.size())
        #print(Cs.size())
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        
        #print(dts.size())
        #print("end'.........")
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        
        #print(xs.size())
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        #print(dts.size())
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        #print(Bs.size())
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        #print(Cs.size())
        Ds = self.Ds.float().view(-1) # (k * d)
        #print(Ds.size())
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        #print(As.size())
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        
        #print('start.................................................')
        
        #print(x.size())

        xz = self.in_proj(x)
        
        #print(xz.size())
        
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
        
        #print(x.size())

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        #assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.conv_vss1 = nn.Conv2d(hidden_dim, hidden_dim,kernel_size=3, stride=1,padding=1)
        self.relu_vss1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #3\self.relu_vss2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #3\self.conv_vss2 = nn.Conv2d(hidden_dim, hidden_dim,kernel_size=3, stride=1,padding=1)
        

    def forward(self, input: torch.Tensor):
        #print(input.size())
        x_vss = self.drop_path(self.self_attention(self.ln_1(input.permute(0,2,3,1))))
        x_conv = self.relu_vss1(self.conv_vss1(input))
        x = x_conv+x_vss.permute(0,3,1,2)
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth=1, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        return x
        
class SS2D_cross(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj0 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj1 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d0 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2d1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj0 = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj1 = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight0 = nn.Parameter(torch.stack([t.weight for t in self.x_proj0], dim=0)) # (K=4, N, inner)
        self.x_proj_weight1 = nn.Parameter(torch.stack([t.weight for t in self.x_proj1], dim=0))
        del self.x_proj0
        del self.x_proj1

        self.dt_projs0 = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs1 = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight0 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs0], dim=0)) # (K=4, inner, rank)
        self.dt_projs_weight1 = nn.Parameter(torch.stack([t.weight for t in self.dt_projs1], dim=0)) # (K=4, inner, rank)
        
        self.dt_projs_bias0 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs0], dim=0)) # (K=4, inner)
        self.dt_projs_bias1 = nn.Parameter(torch.stack([t.bias for t in self.dt_projs1], dim=0)) 
        
        del self.dt_projs0
        del self.dt_projs1
        
        self.A_logs0 = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.A_logs1 = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        
        self.Ds0 = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds1 = self.D_init(self.d_inner, copies=4, merge=True)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm0 = nn.LayerNorm(self.d_inner)
        self.out_proj0 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_norm1 = nn.LayerNorm(self.d_inner)
        self.out_proj1 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x0,x1):
        self.selective_scan0 = selective_scan_fn
        self.selective_scan1 = selective_scan_fn
        
        B, C, H, W = x0.shape
        L = H * W
        K = 4

        x_hwwh0 = torch.stack([x0.view(B, -1, L), torch.transpose(x0, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs0 = torch.cat([x_hwwh0, torch.flip(x_hwwh0, dims=[-1])], dim=1) # (b, k, d, l)
        
        x_hwwh1 = torch.stack([x1.view(B, -1, L), torch.transpose(x1, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs1 = torch.cat([x_hwwh1, torch.flip(x_hwwh1, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl0 = torch.einsum("b k d l, k c d -> b k c l", xs0.view(B, K, -1, L), self.x_proj_weight0)
        x_dbl1 = torch.einsum("b k d l, k c d -> b k c l", xs1.view(B, K, -1, L), self.x_proj_weight1)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts0, Bs0, Cs0 = torch.split(x_dbl0, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts0 = torch.einsum("b k r l, k d r -> b k d l", dts0.view(B, K, -1, L), self.dt_projs_weight0)
        dts1, Bs1, Cs1 = torch.split(x_dbl0, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts1 = torch.einsum("b k r l, k d r -> b k d l", dts1.view(B, K, -1, L), self.dt_projs_weight1)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs0 = xs0.float().view(B, -1, L) # (b, k * d, l)
        dts0 = dts0.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs0 = Bs0.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs0 = Cs0.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds0 = self.Ds0.float().view(-1) # (k * d)
        As0 = -torch.exp(self.A_logs0.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias0 = self.dt_projs_bias0.float().view(-1) # (k * d)
        
        xs1 = xs1.float().view(B, -1, L) # (b, k * d, l)
        dts1 = dts1.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs1 = Bs1.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs1 = Cs1.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds1 = self.Ds1.float().view(-1) # (k * d)
        As1 = -torch.exp(self.A_logs1.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias1 = self.dt_projs_bias1.float().view(-1) # (k * d)

        out_y0 = self.selective_scan0(
            xs0, dts1, 
            As1, Bs1, Cs1, Ds1, z=None,
            delta_bias=dt_projs_bias1,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y0.dtype == torch.float
        
        out_y1 = self.selective_scan1(
            xs1, dts0, 
            As0, Bs0, Cs0, Ds0, z=None,
            delta_bias=dt_projs_bias0,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y1.dtype == torch.float

        inv_y0 = torch.flip(out_y0[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y0 = torch.transpose(out_y0[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y0 = torch.transpose(inv_y0[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        
        inv_y1 = torch.flip(out_y1[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y1 = torch.transpose(out_y1[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y1 = torch.transpose(inv_y1[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y0[:, 0], inv_y0[:, 0], wh_y0, invwh_y0, out_y1[:, 0], inv_y1[:, 0], wh_y1, invwh_y1

    def forward(self, x0, x1):
        B, H, W, C = x0.shape

        xz0 = self.in_proj0(x0)
        xz1 = self.in_proj1(x1)
               
        x0, z0 = xz0.chunk(2, dim=-1) # (b, h, w, d)
        x1, z1 = xz1.chunk(2, dim=-1)

        x0 = x0.permute(0, 3, 1, 2).contiguous()
        x0 = self.act(self.conv2d0(x0)) # (b, d, h, w)
        
        x1 = x0.permute(0, 3, 1, 2).contiguous()
        x1 = self.act(self.conv2d1(x0))
        
        
        y1, y2, y3, y4, y5, y6, y7, y8 = self.forward_core(x0,x1)
        
        
        assert y1.dtype == torch.float32
        
        yir = y1 + y2 + y3 + y4
        yvi = y5 + y6 + y7 + y8
        
        
        yir = torch.transpose(yir, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        yir = self.out_norm0(yir)
        
        yvi = torch.transpose(yvi, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        yvi = self.out_norm1(yvi)
        
        yir = yir * F.silu(z0)
        out0 = self.out_proj0(yir)
        
        yvi = yvi * F.silu(z1)
        out1 = self.out_proj1(yvi)
        
        return out0, out1


class VSSBlock_cross(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.self_attention = SS2D_cross(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        self.drop_path = DropPath(drop_path)
        self.conv_vss1 = nn.Conv2d(hidden_dim, hidden_dim,kernel_size=3, stride=1,padding=1)
        self.relu_vss1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu_vss2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_vss2 = nn.Conv2d(hidden_dim, hidden_dim,kernel_size=3, stride=1,padding=1)
        
        self.conv_vss3 = nn.Conv2d(hidden_dim, hidden_dim,kernel_size=3, stride=1,padding=1)
        self.relu_vss3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu_vss4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_vss4 = nn.Conv2d(hidden_dim, hidden_dim,kernel_size=3, stride=1,padding=1)
        

    def forward(self, x0,x1):
        x_vss0, x_vss1 = self.drop_path(self.self_attention(self.ln_1(x0.permute(0,2,3,1)),self.ln_2(x1.permute(0,2,3,1))))
        #x_conv0 = self.relu_vss2(self.conv_vss2(self.relu_vss1(self.conv_vss1(x0))))
        #x_conv1 = self.relu_vss4(self.conv_vss4(self.relu_vss3(self.conv_vss3(x1))))
        x0 = x_vss0.permute(0,3,1,2) #x0+
        x1 = x_vss1.permute(0,3,1,2) #x1+
        return x0,x1


class VSSLayer_cross(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth=1, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock_cross(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, xir, xvi):
        for blk in self.blocks:
            xir,xvi = blk(xir,xvi)

        return xir,xvi
        
        
        
