from .simple_resnet import SimpleResnet_20, SimpleResnet_36, SimpleResnet_64
from .simple_resnet import SimpleResnet_split_64, SimpleResnet_split_36
from .simple_resnet import SimpleResnet_fullsplit_36, SimpleResnet_split_attention_36, SimpleResnet_36_multiTask
from .simple_resnet import SimpleResnet_split_abla_36,SimpleResnet_atten_split_abla_36, SimpleResnet_split_abla_64
from .simple_resnet import SimpleResnet_20_Se, SimpleResnet_36_Se, SimpleResnet_64_Se
from .simple_resnet import SimpleResnet_split_abla_36_fusion
from .classifier import *
from .resnest import *
from .senet import *
from .mapping_net import *
from .resnet import *
from .hpda import HPDA_res50
from .Module import MLB, PCM_AM
# from .resnet import ResNet_50 as resnet50
# from .resnet import ResNet_101 as resnet101
# from .resnet import ResNet_152 as resnet152
from .densenet import DenseNet_121, DenseNet_161, DenseNet_169, DenseNet_201
