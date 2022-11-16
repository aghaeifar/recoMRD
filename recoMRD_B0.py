import ctypes
import os
import numpy as np
from recoMRD import recoMRD

class recoMRD_B0(recoMRD):
    dTE         = 0
    img_b0      = np.empty([1])
    img_mag     = np.empty([1])    
    img_mask    = np.empty([1])
    img_b0_uw   = np.empty([1]) # unwrapped 

    def __init__(self, filename=None):
        super().__init__(filename)
        super().runReco()
        
    def _custom_task(self):
        self.dTE = np.diff(self.xml_hdr.sequenceParameters.TE) * 1e-3
        print(f"Calculating B0 map. dTE = {self.dTE[0]*1e3} ms")

        if self.dim_info['rep']['len'] == 1 : #  % regular B0 mapping, b0map = (Eco2 - Eco1)
            self.img_b0 = abs(self.img[:,:,:,:,:,1]) * self.img[:,:,:,:,:,1] * \
                          abs(self.img[:,:,:,:,:,0]) * np.conj(self.img[:,:,:,:,:,0])   
            self.dim_info['eco']['len'] = 1
            self.dim_size[self.dim_info['eco']['ind']] = 1
        else: # shims basis-map, b0map = (Eco2Repn - Eco1Repn) - (Eco2Rep1 - Eco1Rep1)
            self.img_b0 = abs(self.img[:,:,:,:,:,1,:]) * self.img[:,:,:,:,:,1,:]          * \
                          abs(self.img[:,:,:,:,:,0,:]) * np.conj(self.img[:,:,:,:,:,0,:]) * \
                          abs(self.img[:,:,:,:,:,1,0]) * np.conj(self.img[:,:,:,:,:,1,0]) * \
                          abs(self.img[:,:,:,:,:,0,0]) * self.img[:,:,:,:,:,0,0]


    def get_b0hz(self, scale = 1, offset = 0):
        if self.img_b0_uw.shape != self.img_b0.shape:
            print(f"\033[93mUnwrapped image is not yet calculated. \033[0m")
            return None
        return (scale*self.img_b0_uw + offset) / self.dTE[0] / (2*np.pi)

    def sqz(self):
        super().sqz() # update boundries
        self.img_b0     = self.img_b0.squeeze()
        self.img_mag    = self.img_mag.squeeze()
        self.img_mask   = self.img_mask.squeeze()
        self.img_b0_uw  = self.img_b0_uw.squeeze()

    def unwrap_b0(self):
        self.img_b0_uw = np.zeros(self.img_b0.shape, dtype=np.float32, order='F')
        dir_path = os.path.dirname(os.path.realpath(__file__))
        handle   = ctypes.CDLL(os.path.join(dir_path, "lib", "libunwrap_b0.so")) 
        handle.unwrap_b0.argtypes = [np.ctypeslib.ndpointer(np.float32, ndim=self.img_b0.ndim, flags='F'),
                                     np.ctypeslib.ndpointer(np.float32, ndim=self.img_b0_uw.ndim, flags='F'),
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int]
        
        b0_size = [x for x in self.img_b0.shape if x > 1]
        handle.unwrap_b0(self.img_b0, self.img_b0_uw, *b0_size)

    def _coil_combination(self):
        super()._coil_combination()
        self.img_b0  = np.angle(np.sum(self.img_b0, self.dim_info['cha']['ind'], keepdims=True))
        self.img_mag = self.img[:,:,:,:,:,0,0,0]
        self.img = [] # save memory
                    

    
    
