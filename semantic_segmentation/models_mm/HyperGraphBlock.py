import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from dhg import Hypergraph
from dhg.data import Cooking200
from dhg.models import HGNN, HGNNP
from dhg.random import set_seed
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

class HGNNPBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.net = HGNNP(in_channels, 32, out_channels, use_bn=True)
    def forward(self, x):
        B,C,H,W = x.shape
        L = H*W
        x = x.view(B,L,C)

        for i in range(B):
            # print("cal G...")
            ft = x[i]
            G = Hypergraph.from_feature_kNN(ft, k=30)
            # print("forward...")
            out = self.net(x[i], G)
            if i==0:
                outs = out
            else:
                outs = torch.cat((outs,out), dim=0)
            del ft
            del G
        outs = outs.view(B,-1,H,W)
        return outs


if __name__ == "__main__":
    x = torch.randn(4,3,224,224)
    model = HyperGraphBlock(3,23)
    y = model(x)
    print(y.shape)