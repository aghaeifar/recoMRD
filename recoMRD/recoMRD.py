import os
import ctypes
import numpy as np
import nibabel as nib
from bart import bart
from tqdm import tqdm
from recoMRD import readMRD


# ifftnd and fftnd are taken from twixtools package
def ifftnd(kspace:np.ndarray, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img  = fftshift(ifftn(ifftshift(kspace, axes=axes), axes=axes), axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))
    return img

def fftnd(img:np.ndarray, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, fftn
    if axes is None:
        axes = range(img.ndim)
    kspace  = fftshift(fftn(ifftshift(img, axes=axes), axes=axes), axes=axes)
    kspace /= np.sqrt(np.prod(np.take(kspace.shape, axes)))
    return kspace


class recoMRD(readMRD):    
    img = None
  
    def __init__(self, filename=None):   
        super().__init__(filename)


    def runReco(self):
        self.img = self.kspace_to_image(self.kspace['image_scan'])
        self.img = self.remove_oversampling(self.img)
        self.img = self.coil_combination(self.img, method='sos')


    # applying iFFT to kspace and build image
    def kspace_to_image(self, kspace:np.ndarray):
        if kspace.ndim != len(self.dim_size):
            print(f'Error! shape is wrong. {kspace.shape} vs {self.dim_size}')
            return

        img = np.zeros_like(kspace, dtype=np.complex64)
        # this loop is slow because of order='F' and ind is in the first dimensions. see above, _create_kspace(). 
        for ind in tqdm(range(self.dim_info['cha']['len']), desc='Fourier transform'):
            img[ind,...] = ifftnd(kspace[ind,...], [0,1,2])        
        return img

    def image_to_kspace(self, img:np.ndarray):
        if img.ndim != len(self.dim_size):
            print(f'Error! shape is wrong. {img.shape} vs {self.dim_size}')
            return

        kspace = np.zeros_like(img, dtype=np.complex64)
        # this loop is slow because of order='F' and ind is in the first dimensions. see above, _create_kspace(). 
        for ind in tqdm(range(self.dim_info['cha']['len']), desc='Fourier transform'):
            kspace[ind,...] = fftnd(img[ind,...], [0,1,2])        
        return kspace

    ##########################################################
    def coil_combination(self, volume:np.ndarray, method='sos', coil_sens=None, update_diminfo=False):
        if volume.ndim != len(self.dim_size):
            print(f'Input size not valid. {volume.shape} != {self.dim_size}')
            return

        all_methods = ('sos', 'bart', 'adaptive')
        if method.lower() not in all_methods:
            print(f'Given method is not valid. Choose between {", ".join(all_methods)}')
            return
        

        volume_comb = np.sqrt(np.sum(abs(volume)**2, self.dim_info['cha']['ind'], keepdims=True))
        if method.lower() == 'bart' and coil_sens is not None:
            l2_reg    = 1e-4
            volume    = np.moveaxis(volume, 0, 3) # adapting to bart CFL format
            coil_sens = np.moveaxis(coil_sens, 0, 3) # adapting to bart CFL format
            n_extra1  = np.prod(volume.shape[4:]) # number of extra dims  
            n_extra2  = np.prod(volume.shape[5:]) # number of extra dims excluing slice
            kspace    = volume.reshape(volume.shape[:4] +(-1,)) # flatten extra dims
            volume_comb  = volume_comb.reshape(volume_comb.shape[:4] + (-1,)) # flatten extra dims
            scale_factor = [np.percentile(volume_comb[...,ind], 99).astype(np.float32) for ind in range(volume_comb.shape[-1])]
            recon  = [np.expand_dims(bart.bart(1, 'pics -w {} -R Q:{} -S'.format(scale_factor[ind], l2_reg), kspace[...,ind], coil_sens[...,ind%n_extra2]), axis=[0,3]) for ind in range(n_extra1)]
            shp = [1,] + self.dim_size[1:]
            volume_comb = np.stack(recon, axis=4).reshape(shp)

        elif method.lower() == 'adaptive' and coil_sens is not None:
            coil_sens   = np.expand_dims(coil_sens, axis=[*range(coil_sens.ndim, volume.ndim)]) # https://numpy.org/doc/stable/user/basics.broadcasting.html
            volume_comb = np.divide(volume, coil_sens, out=np.zeros_like(coil_sens), where=coil_sens!=0)
            volume_comb = np.sum(volume_comb, self.dim_info['cha']['ind'], keepdims=True)

        if update_diminfo:
            self.dim_info['cha']['len'] = volume_comb.shape[self.dim_info['cha']['ind']]
            self.dim_size[self.dim_info['cha']['ind']] = self.dim_info['cha']['len']

        return volume_comb

    ##########################################################
    def calc_coil_sensitivity(self, acs, method='caldir'):
        all_methods = ('espirit', 'caldir', 'walsh')
        if method.lower() not in all_methods:
            print(f'Given method is not valid. Choose between {", ".join(all_methods)}')
            return
        d = self.dim_info
        if d['cha']['ind']!=0 or d['ro']['ind']!=1 or d['pe1']['ind']!=2 or d['pe2']['ind']!=3 or d['slc']['ind']!=4:
            print('Error! Dimension order does not fit to the desired order.')
            return

        coils_sensitivity = np.zeros_like(acs[...,0,0,0,0,0,0])
        if method.lower() == 'espirit'.lower():
            for cslc in range(self.dim_info['slc']['len']):
                kspace      = np.moveaxis(acs[...,cslc,0,0,0,0,0,0], 0, 3)
                coil_sens   = bart.bart(1, 'ecalib -m 1', kspace)
                coils_sensitivity[...,cslc] = np.moveaxis(coil_sens, 3, 0)
        elif method.lower() == 'caldir'.lower():
            for cslc in range(self.dim_info['slc']['len']):
                kspace      = np.moveaxis(acs[...,cslc,0,0,0,0,0,0], 0, 3)
                cal_size    = np.max(kspace.shape[:3])//2
                coil_sens   = bart.bart(1, f'caldir {cal_size}', kspace)
                coils_sensitivity[...,cslc] = np.moveaxis(coil_sens, 3, 0)
        elif method.lower() == 'walsh'.lower():
            dir_path = os.path.dirname(os.path.realpath(__file__))
            handle   = ctypes.CDLL(os.path.join(dir_path, "lib", "libwalsh.so")) 
            handle.adaptive_combine.argtypes = [np.ctypeslib.ndpointer(np.complex64, ndim=4, flags='F'),
                                                np.ctypeslib.ndpointer(np.complex64, ndim=4, flags='F'),
                                                np.ctypeslib.ndpointer(np.float32, ndim=3, flags='F'),
                                                ctypes.POINTER(ctypes.c_int), 
                                                ctypes.POINTER(ctypes.c_int), 
                                                ctypes.POINTER(ctypes.c_int),
                                                ctypes.c_int, ctypes.c_bool]
            acs = acs[...,0,0,0,0,0,0].squeeze().copy(order='F')
            weights = np.zeros_like(acs, dtype=np.complex64, order='F')  
            norm = np.zeros_like(acs[0,...], dtype=np.float32, order='F')    
            n = list(acs.shape)
            ks = [7, 7, 3]
            st = [1, 1, 1]
            nc_svd = -1
            handle.adaptive_combine(acs, weights, norm, (ctypes.c_int*4)(*n), (ctypes.c_int*3)(*ks), (ctypes.c_int*3)(*st), nc_svd, False) # 3D and 4D input
            coils_sensitivity = weights.copy(order='C').reshape(coils_sensitivity.shape)
            
        return coils_sensitivity

    ##########################################################
    def compress_coil(self, *kspace:np.ndarray, virtual_channels=None): 
        # kspace[0] is the reference input to create compression matrix for all inputs, it should be GRAPPA scan for example.    
        print('Compressing Rx channels...')
        d = self.dim_info
        if d['cha']['ind']!=0 or d['ro']['ind']!=1 or d['pe1']['ind']!=2 or d['pe2']['ind']!=3 or d['slc']['ind']!=4:
            print('Error! Dimension order does not fit to the desired order.')
            return

        if virtual_channels == None:
            virtual_channels = int(kspace[0].shape[d['cha']['ind']] * 0.75)

        kspace_cfl = [np.moveaxis(kspc, 0, 3) for kspc in kspace] # adapting to bart CFL format
        cc_matrix  = [bart.bart(1, 'bart cc -A -M', kspace_cfl[0][...,cslc,0,0,0,0,0,0]) for cslc in range(d['slc']['len'])]

        kspace_compressed_cfl = []
        for kspc in kspace_cfl:
            n_extra1 = np.prod(kspc.shape[4:]) # number of extra dims  
            n_extra2 = np.prod(kspc.shape[5:]) # number of extra dims excluing slice
            kspc_r   = kspc.reshape(kspc.shape[:4] + (-1,)) # flatten extra dims
            kspc_cc  = [bart.bart(1, f'ccapply -p {virtual_channels}', k, cc_matrix[i%n_extra2]) for k, i in zip(kspc_r, range(n_extra1))]
            kspace_compressed_cfl.append(np.stack(kspc_cc, axis=4).reshape(kspc.shape))

        kspace_compressed = [np.moveaxis(kspc, 3, 0) for kspc in kspace_compressed_cfl]
        return (*kspace_compressed,)

    ##########################################################
    def remove_oversampling(self, img:np.ndarray, update_diminfo=False):
        if img.ndim != len(self.dim_size):
            print(f'Error! shape is wrong. {img.shape} vs {self.dim_size}')
            return
        
        if img.shape[self.dim_info['ro']['ind']] != self.dim_info['ro']['len']:
            print('Seems oversampling is already removed!')

        print('Remove oversampling...', end=' ')
        cutoff = (img.shape[self.dim_info['ro']['ind']] - self.matrix_size['image']['x']) // 2 # // -> integer division
        img = np.take(img, np.arange(cutoff, cutoff+self.matrix_size['image']['x']), axis=self.dim_info['ro']['ind']) # img[:,cutoff:-cutoff,...]

        if update_diminfo:
            self.dim_info['ro']['len'] = img.shape[self.dim_info['ro']['ind']]
            self.dim_size[self.dim_info['ro']['ind']] = self.dim_info['ro']['len']
        print('Done.')
        return img
        
    ##########################################################
    # Save a custom volume as nifti
    def make_nifti(self, volume, filename):
        
        if isinstance(volume, np.ndarray) == False: 
            print("Input volume must be a numpy array")
            return

        df = self.dim_info
        ds = [y for y in self.dim_size if y!=1]
        vs = [y for y in volume.shape  if y!=1]
        if vs != ds:
            print(f"Size mismatch! {vs} vs {ds}")
            return

        if len(vs) > 4 :
            print(f"{len(vs)}D data is not supported")
            return

        # bringing to a shape that fits to dim_info
        volume = volume.reshape(self.dim_size)
        volume = np.flip(volume, axis = [df['ro']['ind'], 
                                         df['pe1']['ind'],
                                         df['slc']['ind']])
        #
        # creating permute indices
        #
        prmt_ind = np.arange(0, len(df), 1, dtype=int)
        # swap ro and pe1
        prmt_ind[[df['ro']['ind'], df['pe1']['ind']]] = prmt_ind[[df['pe1']['ind'], df['ro']['ind']]] 
        # move cha to end
        icha = df['cha']['ind']
        prmt_ind = np.hstack([prmt_ind[:icha], prmt_ind[icha+1:], prmt_ind[icha]])

        volume = np.transpose(volume, prmt_ind)
        volume = volume.squeeze()
       
        #
        # build affine matrix, according to SPM notation
        #

        T = self.transformation['mat44'].copy()
        T[:,1:3] = -T[:,1:3] # experimentally discovered

        PixelSpacing = [self.fov['image']['x'] / self.matrix_size['image']['x'], 
                        self.fov['image']['y'] / self.matrix_size['image']['y']]
        R = T[:,0:2] @ np.diag(PixelSpacing)
        x1 = [1,1,1,1]
        x2 = [1,1,self.matrix_size['image']['z'],1]
        
        thickness = self.fov['image']['z'] / self.matrix_size['image']['z']
        zmax = (self.fov['image']['z'] - thickness) / 2
        y1_c = T @ [0, 0, -zmax, 1]
        y2_c = T @ [0, 0, +zmax, 1]
        # SBCS Position Vector points to slice center this must be recalculated for DICOM to point to the upper left corner.
        y1 = y1_c - T[:,0] * self.fov['image']['x']/2 - T[:,1] * self.fov['image']['y']/2
        y2 = y2_c - T[:,0] * self.fov['image']['x']/2 - T[:,1] * self.fov['image']['y']/2
        
        DicomToPatient = np.column_stack((y1, y2, R)) @ np.linalg.inv(np.column_stack((x1, x2, np.eye(4,2))))
        # Flip voxels in y
        
        AnalyzeToDicom = np.column_stack((np.diag([1,-1,1]), [0, (self.matrix_size['image']['y']+1), 0]))
        AnalyzeToDicom = np.row_stack((AnalyzeToDicom, [0,0,0,1]))
        # Flip mm coords in x and y directions
        PatientToTal   = np.diag([-1, -1, 1, 1]) 
        affine         = PatientToTal @ DicomToPatient @ AnalyzeToDicom
        affine         = affine @ np.column_stack((np.eye(4,3), [1,1,1,1])) # this part is implemented in SPM nifti.m
        
        #
        # save to file
        #
        img = nib.Nifti1Image(volume, affine)
        nib.save(img, filename)
