import torch
import torch.nn as nn

class Flat(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MV_CNN(nn.Module):
    def __init__(self, num_classes):
        super(MV_CNN, self).__init__()
        """
        LAYERS
        ------
        1) CL, 8x7x7, stride = 1, padding = 0. Based on the following layers, think whether you should enable the bias or not for this layer. 
        2) normalization layer (think about what the parameter normalized_shape should be). 
        3) leaky ReLu layer with 0.01 'leak' for negative input. 
        """

        # Layers Defintion
        self.cl_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=1, padding=0, bias=False)
        self.norm_1 = nn.LayerNorm(normalized_shape=[8, 218, 218])
        self.relu_1 = nn.LeakyReLU(negative_slope=0.01)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cl_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, stride=2, padding=0, groups=8, bias=False)
        self.norm_2 = nn.LayerNorm(normalized_shape=[8, 52, 52])
        self.relu_2 = nn.LeakyReLU(negative_slope=0.01)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cl_3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True)

        self.cl_4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=0, groups=16, bias=False)
        self.norm_3 = nn.LayerNorm(normalized_shape=[16, 46, 46])
        self.relu_3 = nn.LeakyReLU(negative_slope=0.01)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cl_5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)

        self.cl_5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=0, groups=16, bias=False)
        self.norm_4 = nn.LayerNorm(normalized_shape=[16, 17, 17])
        self.relu_4 = nn.LeakyReLU(negative_slope=0.01)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cl_6 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)

        self.flat = Flat()
        self.fc = nn.Linear(in_features=64, out_features=num_classes)


    def forward(self, x):
        """

        DEFINE YOUR FORWARD PASS HERE

        """

        cl_1 = self.cl_1(x)
        norm_1 = self.norm_1(cl_1)
        relu_1 = self.relu_1(norm_1)
        pool_1 = self.pool_1(relu_1)

        cl_2 = self.cl_2(pool_1)
        norm_2 = self.norm_2(cl_2)
        relu_2 = self.relu_2(norm_2)
        pool_2 = self.relu_2(relu_2)
        cl_3 = self.cl_3(pool_2)

        cl_4 = self.cl_4(cl_3)
        norm_3 = self.norm_3(cl_4)
        relu_3 = self.relu_3(norm_3)
        pool_3 = self.pool_3(relu_3)

        cl_5 = self.cl_4(pool_3)
        norm_4 = self.norm_4(cl_5)
        relu_4 = self.relu_4(norm_4)
        pool_4 = self.pool_4(relu_4)

        cl_5 = self.cl_5(pool_4)

        flat_out = self.flat(cl_5)
        out = self.fc(flat_out)
        return out