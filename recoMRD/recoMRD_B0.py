import os
import ctypes
import numpy as np
from .recoMRD import recoMRD
from .utils import create_brain_mask

class recoMRD_B0(recoMRD):
    dTE         = 0
    img_b0      = np.empty([1])
    img_mag     = np.empty([1])    
    img_mask    = np.empty([1])

    def __init__(self, filename=None):
        super().__init__(filename)        
        self.runReco()
        
    def runReco(self):
        if self.dim_info['eco']['len'] < 2:
            print(f"Error!\033[93mAt least two echoes are expected!\033[0m")
            return
        
        kspace   = self.remove_oversampling(self.kspace['image_scan'], is_kspace = True)
        self.img = self.kspace_to_image(kspace)
                
        self.dTE = np.diff(self.xml_hdr.sequenceParameters.TE) * 1e-3
        print(f"Calculating B0 map. \u0394TE = {self.dTE[0]*1e3} ms")

        ind_eco = self.dim_info['eco']['ind']
        ind_rep = self.dim_info['rep']['ind']
        if self.dim_info['rep']['len'] == 1 : #  % regular B0 mapping, b0map = (Eco2 - Eco1)  
            # using .take, we used [1] rather than 1 to specify slice index. this let to keep that singleton dimension. Otherwise we should expand dims to keep consistency.          
            self.img_b0 = abs(self.img.take([1], axis=ind_eco)) * self.img.take([1], axis=ind_eco) * \
                          abs(self.img.take([0], axis=ind_eco)) * np.conj(self.img.take([0], axis=ind_eco))   
        else: # shims basis-map, b0map = (Eco2Repn - Eco1Repn) - (Eco2Rep1 - Eco1Rep1)
            self.img_b0 = abs(self.img.take([1], axis=ind_eco)) * self.img.take([1], axis=ind_eco)          * \
                          abs(self.img.take([0], axis=ind_eco)) * np.conj(self.img.take([0], axis=ind_eco)) * \
                          abs(self.img.take([1], axis=ind_eco).take([0], axis=ind_rep)) * np.conj(self.img.take([1], axis=ind_eco).take([0], axis=ind_rep)) * \
                          abs(self.img.take([0], axis=ind_eco).take([0], axis=ind_rep)) * self.img.take([0], axis=ind_eco).take([0], axis=ind_rep)
            self.img_b0[:,:,:,:,:,0,0] = complex(1,0)

        self.coil_combination(self.img, method='sos', coil_sens=None)

    ##########################################################
    def get_b0hz(self, b0_uw:np.ndarray = None, offset = 0):
        if b0_uw is None:
            b0_uw = self.unwrap_b0()
        if b0_uw.shape != self.img_b0.shape:
            print(f"\033[93mUnwrapped image is not yet calculated. \033[0m")
            return None
        return (b0_uw + offset) / self.dTE[0] / (2*np.pi)

    ##########################################################
    def unwrap_b0(self):
        print('Unwrapping B0...')
        b0_size = [x for x in self.img_b0.shape if x > 1]
        b0_size.append(1) # unwrapper assumes 4D inputs and loops over 4th dimensions. We set the 4th dim to 1, in case of 3D data.
        b0_size = b0_size[0:4]
        if len(b0_size) != 4 :
            print(f'Only 3D or 4D data is supported for unwrapping. Input shape is {b0_size}')
            return None

        dir_path = os.path.dirname(os.path.realpath(__file__))
        handle   = ctypes.CDLL(os.path.join(dir_path, "lib", "libunwrap_b0.so")) 
        handle.unwrap_b0.argtypes = [np.ctypeslib.ndpointer(np.float32, ndim=self.img_b0.ndim, flags='F'),
                                     np.ctypeslib.ndpointer(np.float32, ndim=self.img_b0.ndim, flags='F'),
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int]
        b0 = self.img_b0.copy(order='F')        
        b0_uw = np.zeros(b0.shape, dtype=b0.dtype, order='F')
        handle.unwrap_b0(b0, b0_uw, *b0_size) # 3D and 4D input  
        return (b0_uw.copy(order='C')) 
                     
    ##########################################################

    def coil_combination(self, volume, method='sos', coil_sens=None):
        self.img_mag  = super().coil_combination(volume, method='sos')[:,:,:,:,:,0:1,0:1,0:1,0:1,0:1,0:1].copy() # I used 0:1 rather than  0 fro indexing to keep singleton dimensions. See https://stackoverflow.com/questions/3551242
        self.img_mask = self.img_mag.copy()
        self.img_b0   = np.angle(np.sum(self.img_b0, self.dim_info['cha']['ind'], keepdims=True))
        # self.img = [] # save memory
                    

    
    
