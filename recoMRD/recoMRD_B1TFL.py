import os
import ctypes
import numpy as np
from recoMRD import recoMRD
from scipy import ndimage

class recoMRD_B1TFL(recoMRD):
    dTE       = 0
    img_b1mag = np.empty([1])
    img_b1phs = np.empty([1])    
    img_mask  = np.empty([1])
    img_cp    = np.empty([1]) # unwrapped 

    def __init__(self, filename=None):
        super().__init__(filename)
        super().runReco()

    def _custom_task(self):
        pass

    def sqz(self):
        super().sqz() # update boundries
        self.img_b1mag = self.img_b1mag.squeeze()
        self.img_b1phs = self.img_b1phs.squeeze()
        self.img_mask  = self.img_mask.squeeze()
        self.img_cp    = self.img_cp.squeeze()
    
    def create_mask(self, erode_size = 3):
        print('Creating mask...')
        mask_size = [x for x in self.img_cp.shape if x > 1]
        if len(mask_size) != 3 :
            print(f'Only 3D data is supported for masking. Input shape is {mask_size}')
            return

        dir_path = os.path.dirname(os.path.realpath(__file__))
        handle   = ctypes.CDLL(os.path.join(dir_path, "lib", "libbet2.so")) 
        handle.runBET.argtypes = [np.ctypeslib.ndpointer(np.float32, ndim=self.img_cp.ndim, flags='F'),
                                  np.ctypeslib.ndpointer(np.float32, ndim=len(mask_size), flags='F'),
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int]
        mag = self.img_cp.copy(order='F')
        mask = np.zeros(mask_size, dtype=mag.dtype, order='F')
        handle.runBET(mag, mask, *mask_size) # 3D input     
        if erode_size > 1:            
            es = erode_size
            mask = ndimage.binary_erosion(mask, structure=np.ones((es,es,es))).astype(self.img_mask.dtype)
            mask = np.asfortranarray(mask)  # binary_erosion changes order to C_CONTIGUOUS

        self.img_mask =  mask.reshape(self.img_cp.shape).copy(order='C')