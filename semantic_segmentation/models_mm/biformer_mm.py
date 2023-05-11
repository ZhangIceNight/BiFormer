
import torch
import torch.nn as nn
from mmcv_custom import load_checkpoint
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from models_cls.biformer import BiFormer 
# from models_cls.biformer_stl import BiFormerSTL as BiFormer 
from timm.models.layers import LayerNorm2d
from .SIM import SIM
from .PatchMerging import PatchMerging
from .HyperGraphBlock import HGNNPBlock

@BACKBONES.register_module()  
class BiFormer_mm(BiFormer):
    def __init__(self, pretrained=None, **kwargs):
        super().__init__(**kwargs)
        
        # step 1: remove unused segmentation head & norm
        del self.head # classification head
        del self.norm # head norm
        #del self.downsample_layers
        # step 2: add extra norms for dense tasks
        self.extra_norms = nn.ModuleList()
        self.hg_layers = nn.ModuleList()
        self.sims = nn.ModuleList()
        self.patchMergings = nn.ModuleList()
        for i in range(4):
            self.extra_norms.append(LayerNorm2d(self.embed_dim[i]))
            self.sims.append(SIM(self.embed_dim[i]))
            self.patchMergings.append(PatchMerging(self.embed_dim[i]//2))
            self.hg_layers.append(HGNNPBlock(self.embed_dim[i], self.embed_dim[i]))
        self.preEmb = nn.Sequential(
            nn.Conv2d(3, self.embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(self.embed_dim[0] // 2),
            nn.GELU()
        ) 



        # step 3: initialization & load ckpt
        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)

        # step 4: convert sync bn, as the batch size is too small in segmentation
        # TODO: check if this is correct
        nn.SyncBatchNorm.convert_sync_batchnorm(self)


    def init_weights(self, pretrained):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            print(f'Load pretrained model from {pretrained}')   
    
    def forward_features(self, x: torch.Tensor):
        out = []
        y = self.preEmb(x)
        for i in range(4):
            if i==0:
                y = self.patchMergings[i](y)
            else:
                y = self.patchMergings[i](x)
            x = self.downsample_layers[i](x)
            x = x + y
            short = self.sims[i](x)
            #short = x
            x = self.stages[i](x)
            short = self.hg_layers[i](short)
            x = x + short
            del y
            del short
            # DONE: check the inconsistency -> no effect on performance
            # in the version before submission:
            # x = self.extra_norms[i](x)
            # out.append(x)
            out.append(self.extra_norms[i](x))
        return tuple(out)
    
    def forward(self, x:torch.Tensor):
        return self.forward_features(x)


if __name__ == "__main__":
  b, h, w, c = 4, 224, 224, 48
  x = torch.randn([b,c,h,w]).cuda()

