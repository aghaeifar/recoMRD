import os
import ctypes
import torch
import numpy as np
from recoMRD import recoMRD
from math import pi as PI

class recoMRD_B0(recoMRD):
    dTE         = 0
    img_b0      = torch.empty([1])
    img_mag     = torch.empty([1])    
    img_mask    = torch.empty([1])

    def __init__(self, filename=None, device='cpu'):
        super().__init__(filename, device)            
        self.runReco()
        
    def runReco(self):
        if self.dim_info['eco']['len'] < 2:
            print(f"Error!\033[93mAt least two echoes are expected!\033[0m")
            return

        super().runReco()  
        self.img_mag = torch.abs(self.img)

        self.dTE = np.diff(self.xml_hdr.sequenceParameters.TE) * 1e-3
        print(f"Calculating B0 map. \u0394TE = {self.dTE[0]*1e3} ms")

        dim_eco = self.dim_info['eco']['ind']
        dim_rep = self.dim_info['rep']['ind']
        idx = torch.tensor([0, 1])
        if self.dim_info['rep']['len'] == 1 : #  % regular B0 mapping, b0map = (Eco2 - Eco1)                        
            self.img_b0 = self.img_mag.index_select(dim_eco, idx[1]) * self.img.index_select(dim_eco, idx[1]) * \
                          self.img_mag.index_select(dim_eco, idx[0]) * self.img.index_select(dim_eco, idx[0]).conj()  

        else: # shims basis-map, b0map = (Eco2Repn - Eco1Repn) - (Eco2Rep1 - Eco1Rep1)
            self.img_b0 = self.img_mag.index_select(dim_eco, idx[1]) * self.img.index_select(dim_rep, idx[1])        * \
                          self.img_mag.index_select(dim_eco, idx[0]) * self.img.index_select(dim_rep, idx[0]).conj() * \
                          self.img_mag.index_select(dim_eco, idx[1]).index_select(dim_rep, idx[0]) * self.img.index_select(dim_eco, idx[1]).index_select(dim_rep, idx[0]).conj() * \
                          self.img_mag.index_select(dim_eco, idx[0]).index_select(dim_rep, idx[0]) * self.img.index_select(dim_eco, idx[0]).index_select(dim_rep, idx[0])
            self.img_b0.moveaxis(dim_rep, 0)[0,...] = torch.tensor(complex(1,0))

        self.img_b0 = torch.angle(self.img_b0)

    ##########################################################
    def get_b0hz(self, b0_uw:torch.Tensor = None, offset = 0):
        if b0_uw is None:
            b0_uw = self._unwrap_b0()
        if b0_uw.shape != self.img_b0.shape:
            print(f"\033[93mUnwrapped image is not yet calculated. \033[0m")
            return None
        return (b0_uw + offset) / self.dTE[0] / (2*PI)

    ##########################################################
    def _unwrap_b0(self):
        print('Unwrapping B0...')
        b0_size = (*self.img_b0.squeeze().shape,) + (1,) # add singleton dimensions
        if len(b0_size) != 4 and len(b0_size) != 5:
            print(f'Only 3D or 4D data is supported for unwrapping. Input shape is {b0_size}')
            return None

        dir_path = os.path.dirname(os.path.realpath(__file__))
        handle   = ctypes.CDLL(os.path.join(dir_path, "..", "utils", "lib", "libunwrap_b0.so")) 
        handle.unwrap_b0.argtypes = [np.ctypeslib.ndpointer(np.float32, ndim=self.img_b0.ndim, flags='F'),
                                     np.ctypeslib.ndpointer(np.float32, ndim=self.img_b0.ndim, flags='F'),
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        b0 = self.img_b0.numpy().copy(order='F')        
        b0_uw = np.zeros_like(b0)
        handle.unwrap_b0(b0, b0_uw, *b0_size[:4]) # [:4] -> always 4D shape to unwrapper
        return (torch.from_numpy(b0_uw.copy(order='C'))) 
                     
    ##########################################################

    # def coil_combination(self, volume, method='sos', coil_sens=None):
    #     self.img_mag  = super().coil_combination(volume, method='sos')[:,:,:,:,:,0:1,0:1,0:1,0:1,0:1,0:1].copy() # I used 0:1 rather than  0 fro indexing to keep singleton dimensions. See https://stackoverflow.com/questions/3551242
    #     self.img_mask = self.img_mag.copy()
    #     self.img_b0   = np.angle(np.sum(self.img_b0, self.dim_info['cha']['ind'], keepdims=True))
        # self.img = [] # save memory
                    

    
    
