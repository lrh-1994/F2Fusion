import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import math
import numpy as np
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2)
        return self.tv_loss_weight * 2 * (h_tv[:, :, :h_x - 1, :w_x - 1] + w_tv[:, :, :h_x - 1, :w_x - 1])

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()#.repeat(1, 32, 1, 1)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()#.repeat(1, 32, 1, 1)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class gradient_sobel(nn.Module):
    def __init__(self,channels=64):
        super(gradient_sobel, self).__init__()
        # laplacian_kernel = torch.tensor([[1,1,1],[1,-8,1],[1,1,1]]).float()
        laplacian_kernel1 = torch.tensor([[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]).float()
        laplacian_kernel1 = laplacian_kernel1.view(1, 1, 3, 3)
        laplacian_kernel1 = laplacian_kernel1.repeat(channels, 1, 1, 1)
        #print(laplacian_kernel.size())
        #print(laplacian_kernel)
        #print(laplacian_kernel.size())
        self.laplacian_filter1 = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter1.weight.data = laplacian_kernel1
        self.laplacian_filter1.weight.requires_grad = False

        laplacian_kernel2 = torch.tensor([[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]).float()
        laplacian_kernel2 = laplacian_kernel2.view(1, 1, 3, 3)
        laplacian_kernel2 = laplacian_kernel2.repeat(channels, 1, 1, 1)
        #print(laplacian_kernel.size())
        #print(laplacian_kernel)
        #print(laplacian_kernel.size())
        self.laplacian_filter2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter2.weight.data = laplacian_kernel2
        self.laplacian_filter2.weight.requires_grad = False


    def forward(self,x):
        #return self.laplacian_filter(x) ** 2
        return torch.abs(self.laplacian_filter1(x))+torch.abs(self.laplacian_filter2(x))
class gradient_sobel32(nn.Module):
    def __init__(self,channels=32):
        super(gradient_sobel32, self).__init__()
        # laplacian_kernel = torch.tensor([[1,1,1],[1,-8,1],[1,1,1]]).float()
        laplacian_kernel1 = torch.tensor([[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]).float()
        laplacian_kernel1 = laplacian_kernel1.view(1, 1, 3, 3)
        laplacian_kernel1 = laplacian_kernel1.repeat(channels, 1, 1, 1)
        #print(laplacian_kernel.size())
        #print(laplacian_kernel)
        #print(laplacian_kernel.size())
        self.laplacian_filter1 = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter1.weight.data = laplacian_kernel1
        self.laplacian_filter1.weight.requires_grad = False

        laplacian_kernel2 = torch.tensor([[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]).float()
        laplacian_kernel2 = laplacian_kernel2.view(1, 1, 3, 3)
        laplacian_kernel2 = laplacian_kernel2.repeat(channels, 1, 1, 1)
        #print(laplacian_kernel.size())
        #print(laplacian_kernel)
        #print(laplacian_kernel.size())
        self.laplacian_filter2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter2.weight.data = laplacian_kernel2
        self.laplacian_filter2.weight.requires_grad = False


    def forward(self,x):
        #return self.laplacian_filter(x) ** 2
        return torch.abs(self.laplacian_filter1(x))+torch.abs(self.laplacian_filter2(x))
class gradient_sobel64(nn.Module):
    def __init__(self,channels=64):
        super(gradient_sobel64, self).__init__()
        # laplacian_kernel = torch.tensor([[1,1,1],[1,-8,1],[1,1,1]]).float()
        laplacian_kernel1 = torch.tensor([[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]).float()
        laplacian_kernel1 = laplacian_kernel1.view(1, 1, 3, 3)
        laplacian_kernel1 = laplacian_kernel1.repeat(channels, 1, 1, 1)
        #print(laplacian_kernel.size())
        #print(laplacian_kernel)
        #print(laplacian_kernel.size())
        self.laplacian_filter1 = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter1.weight.data = laplacian_kernel1
        self.laplacian_filter1.weight.requires_grad = False

        laplacian_kernel2 = torch.tensor([[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]).float()
        laplacian_kernel2 = laplacian_kernel2.view(1, 1, 3, 3)
        laplacian_kernel2 = laplacian_kernel2.repeat(channels, 1, 1, 1)
        #print(laplacian_kernel.size())
        #print(laplacian_kernel)
        #print(laplacian_kernel.size())
        self.laplacian_filter2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter2.weight.data = laplacian_kernel2
        self.laplacian_filter2.weight.requires_grad = False


    def forward(self,x):
        #return self.laplacian_filter(x) ** 2
        return torch.abs(self.laplacian_filter1(x))+torch.abs(self.laplacian_filter2(x))
class gradient_sobel128(nn.Module):
    def __init__(self,channels=128):
        super(gradient_sobel128, self).__init__()
        # laplacian_kernel = torch.tensor([[1,1,1],[1,-8,1],[1,1,1]]).float()
        laplacian_kernel1 = torch.tensor([[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]).float()
        laplacian_kernel1 = laplacian_kernel1.view(1, 1, 3, 3)
        laplacian_kernel1 = laplacian_kernel1.repeat(channels, 1, 1, 1)
        #print(laplacian_kernel.size())
        #print(laplacian_kernel)
        #print(laplacian_kernel.size())
        self.laplacian_filter1 = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter1.weight.data = laplacian_kernel1
        self.laplacian_filter1.weight.requires_grad = False

        laplacian_kernel2 = torch.tensor([[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]).float()
        laplacian_kernel2 = laplacian_kernel2.view(1, 1, 3, 3)
        laplacian_kernel2 = laplacian_kernel2.repeat(channels, 1, 1, 1)
        #print(laplacian_kernel.size())
        #print(laplacian_kernel)
        #print(laplacian_kernel.size())
        self.laplacian_filter2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter2.weight.data = laplacian_kernel2
        self.laplacian_filter2.weight.requires_grad = False


    def forward(self,x):
        #return self.laplacian_filter(x) ** 2
        return torch.abs(self.laplacian_filter1(x))+torch.abs(self.laplacian_filter2(x))

class gradient_feature(nn.Module):
    def __init__(self,channels=32):
        super(gradient_feature, self).__init__()
        # laplacian_kernel = torch.tensor([[1,1,1],[1,-8,1],[1,1,1]]).float()
        laplacian_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]]).float()
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        print(laplacian_kernel.size())
        #print(laplacian_kernel)
        #print(laplacian_kernel.size())
        self.laplacian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter.weight.data = laplacian_kernel
        self.laplacian_filter.weight.requires_grad = False
    def forward(self,x):
        #return self.laplacian_filter(x) ** 2
        return self.laplacian_filter(x)

class gradient_mean(nn.Module):
    def __init__(self,channels=1):
        super(gradient_mean, self).__init__()
        laplacian_kernel = torch.tensor([[1,1,1],[1,-8,1],[1,1,1]]).float()
        #laplacian_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]]).float()
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        #print(laplacian_kernel)
        #print(laplacian_kernel.size())
        self.laplacian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter.weight.data = laplacian_kernel
        self.laplacian_filter.weight.requires_grad = False
    def forward(self,x):
        #return self.laplacian_filter(x) ** 2
        return torch.abs(self.laplacian_filter(x))

class gradient_feature256(nn.Module):
    def __init__(self,channels=256):
        super(gradient_feature256, self).__init__()
        # laplacian_kernel = torch.tensor([[1,1,1],[1,-8,1],[1,1,1]]).float()
        laplacian_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]]).float()
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        #print(laplacian_kernel)
        #print(laplacian_kernel.size())
        self.laplacian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter.weight.data = laplacian_kernel
        self.laplacian_filter.weight.requires_grad = False
    def forward(self,x):
        #return self.laplacian_filter(x) ** 2
        return self.laplacian_filter(x)
class gradient_feature512(nn.Module):
    def __init__(self,channels=512):
        super(gradient_feature512, self).__init__()
        # laplacian_kernel = torch.tensor([[1,1,1],[1,-8,1],[1,1,1]]).float()
        laplacian_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]]).float()
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        #print(laplacian_kernel)
        #print(laplacian_kernel.size())
        self.laplacian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter.weight.data = laplacian_kernel
        self.laplacian_filter.weight.requires_grad = False
    def forward(self,x):
        #return self.laplacian_filter(x) ** 2
        return self.laplacian_filter(x)
class gradient_map1(nn.Module):
    def __init__(self,channels=1):
        super(gradient_map1, self).__init__()
        # laplacian_kernel = torch.tensor([[1,1,1],[1,-8,1],[1,1,1]]).float()
        laplacian_kernel = torch.tensor([[0,1/4,0],[1/4,-1,1/4],[0,1/4,0]]).float()
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        #print(laplacian_kernel)
        #print(laplacian_kernel.size())
        self.laplacian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter.weight.data = laplacian_kernel
        self.laplacian_filter.weight.requires_grad = False
    def forward(self,x):
        #return self.laplacian_filter(x) ** 2
        return self.laplacian_filter(x)
class gradient_map(nn.Module):
    def __init__(self,channels=64):
        super(gradient_map, self).__init__()
        # laplacian_kernel = torch.tensor([[1,1,1],[1,-8,1],[1,1,1]]).float()
        laplacian_kernel = torch.tensor([[0,1/4,0],[1/4,-1,1/4],[0,1/4,0]]).float()
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        #print(laplacian_kernel)
        #print(laplacian_kernel.size())
        self.laplacian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter.weight.data = laplacian_kernel
        self.laplacian_filter.weight.requires_grad = False
    def forward(self,x):
        #return self.laplacian_filter(x) ** 2
        return self.laplacian_filter(x)
class gradient_loss128(nn.Module):
    def __init__(self,channels=128):
        super(gradient_loss128, self).__init__()
        # laplacian_kernel = torch.tensor([[1,1,1],[1,-8,1],[1,1,1]]).float()
        laplacian_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]]).float()
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        #print(laplacian_kernel)
        #print(laplacian_kernel.size())
        self.laplacian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter.weight.data = laplacian_kernel
        self.laplacian_filter.weight.requires_grad = False
    def forward(self,x):
        #return self.laplacian_filter(x) ** 2
        return self.laplacian_filter(x)
