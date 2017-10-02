import mxnet as mx
import numpy as np
from mxnet.gluon import Block

class _CNNControl(Block):

    def __init__(self,
                 input_order =  3,
                 input_size  =100,
                 core_number =  4,
                 core_size   = 25):
        """
        input_order (int): Order of input-CNN filter 
        input_size  (int): Size of input vector
        core_number (int): Number of switching core
        core_size   (int): Size of switching core
        """

        self._io = input_order
        self._is = input_size
        self._cn = core_number
        self._cs = core_size

        with mx.AttrScope(group='Ctrl', data='input'):
            self.initials()

    def initials(self):
        mx.random.seed(0)
        
       



        

                 
                 
