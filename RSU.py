import mxnet as mx
import numpy as np

from Control import _CNNControl
from Switch import _SoftmaxSwitch

class RSU(object):

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

        self.ctrl = _CNNControl(self._io,
                                self._is,
                                self._cn,
                                self._cs)

        self.swch = _SoftmaxSwitch(self._io,
                                   self._is,
                                   self._cn,
                                   self._cs)

        with mx.AttrScope(group='RSU'):
            self.initials()

    def initials(self):
        mx.random.seed(0)
        
       



        

                 
                 
