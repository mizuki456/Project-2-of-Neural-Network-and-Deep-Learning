'''
Models implementation and training & evaluating functions
'''

from . import vgg
from . import basic
from .basic import BasicModel, SimpleCIFARModel, SimpleCIFARModel_LeakyReLU, DeepCIFARModel

from .vgg import VGG_A, VGG_A_Dropout, VGG_A_Light, VGG_A_BN