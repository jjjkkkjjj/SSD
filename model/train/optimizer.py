__all__ = ['Adam', 'SGD', 'Momentum']
from tensorflow.compat.v1.train import *

class Adam(AdamOptimizer):
    pass

class SGD(GradientDescentOptimizer):
    pass

class Momentum(MomentumOptimizer):
    pass