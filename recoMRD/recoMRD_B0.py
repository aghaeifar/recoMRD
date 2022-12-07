import os
import ctypes
import numpy as np
from .recoMRD import recoMRD
from scipy import ndimage

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
        else: # shims basis-map, b0map = (Eco2Repn - Eco1Repn) - (Eco2Rep1 - Eco1Rep1)
            self.img_b0 = abs(self.img[:,:,:,:,:,1,:]) * self.img[:,:,:,:,:,1,:]          * \
                          abs(self.img[:,:,:,:,:,0,:]) * np.conj(self.img[:,:,:,:,:,0,:]) * \
                          abs(self.img[:,:,:,:,:,1,0]) * np.conj(self.img[:,:,:,:,:,1,0]) * \
                          abs(self.img[:,:,:,:,:,0,0]) * self.img[:,:,:,:,:,0,0]
            self.img_b0[:,:,:,:,:,0,0] = np.exp( 1j * 1)

        self.dim_info['eco']['len'] = 1
        self.dim_size[self.dim_info['eco']['ind']] = 1

    ##########################################################
    def get_b0hz(self, scale = 1, offset = 0):
        if self.img_b0_uw.shape != self.img_b0.shape:
            print(f"\033[93mUnwrapped image is not yet calculated. \033[0m")
            return None
        return (scale*self.img_b0_uw + offset) / self.dTE[0] / (2*np.pi)

    ##########################################################
    def unwrap_b0(self):
        print('Unwrapping B0...')
        b0_size = [x for x in self.img_b0.shape if x > 1]
        b0_size.append(1)
        b0_size = b0_size[0:4]
        if len(b0_size) != 4 :
            print(f'Only 3D or 4D data is supported for unwrapping. Input shape is {b0_size}')
            return

        dir_path = os.path.dirname(os.path.realpath(__file__))
        handle   = ctypes.CDLL(os.path.join(dir_path, "lib", "libunwrap_b0.so")) 
        handle.unwrap_b0.argtypes = [np.ctypeslib.ndpointer(np.float32, ndim=self.img_b0.ndim, flags='F'),
                                     np.ctypeslib.ndpointer(np.float32, ndim=self.img_b0.ndim, flags='F'),
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int]
        b0 = self.img_b0.copy(order='F')        
        b0_uw = np.zeros(b0.shape, dtype=b0.dtype, order='F')
        handle.unwrap_b0(b0, b0_uw, *b0_size) # 3D and 4D input  
        self.img_b0_uw =  b0_uw.copy(order='C') 
                     
    ##########################################################
    def create_mask(self, erode_size = 3):
        print('Creating mask...')
        mask_size = [x for x in self.img_mag.shape if x > 1]
        if len(mask_size) != 3 :
            print(f'Only 3D data is supported for masking. Input shape is {mask_size}')
            return

        dir_path = os.path.dirname(os.path.realpath(__file__))
        handle   = ctypes.CDLL(os.path.join(dir_path, "lib", "libbet2.so")) 
        handle.runBET.argtypes = [np.ctypeslib.ndpointer(np.float32, ndim=self.img_mag.ndim, flags='F'),
                                  np.ctypeslib.ndpointer(np.float32, ndim=len(mask_size), flags='F'),
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int]
                                  
        mag = self.img_mag.copy(order='F')
        mask = np.zeros(mask_size, dtype=mag.dtype, order='F')
        handle.runBET(mag, mask, *mask_size) # 3D input     
        if erode_size > 1:            
            es = erode_size
            mask = ndimage.binary_erosion(mask, structure=np.ones((es,es,es))).astype(self.img_mask.dtype)
            mask = np.asfortranarray(mask)  # binary_erosion changes order to C_CONTIGUOUS

        self.img_mask =  mask.reshape(self.img_mag.shape).copy(order='C') 

    def _coil_combination(self, volume, method='sos', coil_sens=None):
        self.img = super()._coil_combination(self.img, method='sos')
        self.img_b0   = np.angle(np.sum(self.img_b0, self.dim_info['cha']['ind'], keepdims=True))
        self.img_mag  = self.img[:,:,:,:,:,0,0,0].copy()
        self.img_mask = self.img_mag.copy()
        # self.img = [] # save memory
                    

    
    
