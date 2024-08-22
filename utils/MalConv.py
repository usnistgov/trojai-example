# Adapted from https://github.com/NeuromorphicComputationResearchProgram/MalConv2/blob/main/MalConv.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils.LowMemConv import LowMemConvBase




class MalConv(LowMemConvBase):
    
    def __init__(self, out_size=5, channels=128, window_size=512, stride=512,
                 embd_size=8, log_stride=None):
        super(MalConv, self).__init__()
        self.embd = nn.Embedding(257, embd_size, padding_idx=0)
        if not log_stride is None:
            stride = 2**log_stride

        self.conv_1 = nn.Conv1d(embd_size, channels, window_size,
                                stride=stride, bias=True)
        self.conv_2 = nn.Conv1d(embd_size, channels, window_size,
                                stride=stride, bias=True)

        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)
        
    
    def processRange(self, x):
        x = self.embd(x)
        x = torch.transpose(x,-1,-2)
        cnn_value = self.conv_1(x)
        gating_weight = torch.sigmoid(self.conv_2(x))        
        x = cnn_value * gating_weight
        return x
    
    def forward(self, x):
        post_conv = x = self.seq2fix(x)
        penult = x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x, penult, post_conv
