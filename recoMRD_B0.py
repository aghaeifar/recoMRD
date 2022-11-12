
from recoMRD import recoMRD
import numpy as np

class recoMRD_B0(recoMRD):
    dTE         = 0
    img_mag     = 0
    img_b0      = 0
    img_mask    = 0

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


    def get_b0hz(self, scale = 1, offset = 0):
        return (scale*self.img_b0 + offset) / self.dTE[0] / (2*np.pi)

    def _coil_combination(self):
        self.img_b0  = np.angle(np.sum(self.img_b0, self.dim_info['cha']['ind'], keepdims=True))
        self.img_mag = np.sqrt(np.sum(abs(self.img[:,:,:,:,:,0,0,0])**2, self.dim_info['cha']['ind'], keepdims=True))
        self.img     = []; # save memory
            
        # update dim info
        #   this.update_size(this.img_b0);

    
    