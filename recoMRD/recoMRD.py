import os
import ctypes
import numpy as np
import scipy as sp
import nibabel as nib
from bart import bart
from tqdm import tqdm
from .readMRD import readMRD
from .utils import ifftnd, fftnd




class recoMRD(readMRD):    
    img = None
  
    def __init__(self, filename=None):   
        super().__init__(filename)


    def runReco(self):
        self.img = self.kspace_to_image(self.kspace['image_scan'])
        self.img = self.remove_oversampling(self.img)
        self.img = self.coil_combination(self.img, method='sos')

    ##########################################################
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
    def coil_combination(self, volume:np.ndarray, method='sos', coil_sens=None):
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
            # print(recon[0].shape)
            # print(len(recon))
            # shp = [1,] + self.dim_size[1:]
            # print(shp)
            volume_comb = np.stack(recon, axis=4) #.reshape(shp)

        elif method.lower() == 'adaptive' and coil_sens is not None:
            coil_sens   = np.expand_dims(coil_sens, axis=[*range(coil_sens.ndim, volume.ndim)]) # https://numpy.org/doc/stable/user/basics.broadcasting.html
            volume_comb = np.divide(volume, coil_sens, out=np.zeros_like(coil_sens), where=coil_sens!=0)
            volume_comb = np.sum(volume_comb, self.dim_info['cha']['ind'], keepdims=True)

        return volume_comb

    ##########################################################
    def calc_coil_sensitivity(self, acs:np.ndarray, method='caldir'):
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
    def remove_oversampling(self, img:np.ndarray, is_kspace=False):
        if img.squeeze().ndim != len([i for i in self.dim_size if i>1]):
            print(f'Error! shape is wrong. {img.shape} vs {self.dim_size}')
            return
        
        if img.shape[self.dim_info['ro']['ind']] != self.dim_info['ro']['len']:
            print('Oversampling is already removed!')
            return

        print('Remove oversampling...', end=' ')
        if is_kspace:
            os_factor = self.dim_info['ro']['len'] / self.matrix_size['image']['x'] # must be divisible, otherwise I made a mistake somewhere
            ind = np.arange(0, self.dim_info['ro']['len'], os_factor, dtype=int)
        else:
            cutoff = (img.shape[self.dim_info['ro']['ind']] - self.matrix_size['image']['x']) // 2 # // -> integer division
            ind = np.arange(cutoff, cutoff+self.matrix_size['image']['x']) # img[:,cutoff:-cutoff,...]
        
        img = np.take(img, ind, axis=self.dim_info['ro']['ind'])

        print('Done.')
        return img


    ##########################################################   
    # update dimension info based on the input image
    def update_dim_info(self, img:np.ndarray):
        if img.ndim != len(self.dim_size):
            print(f'Error! shape is wrong. {img.shape} vs {self.dim_size}')
            return
        
        for tags in self.dim_info.keys():
            self.dim_info[tags]['len'] = img.shape[self.dim_info[tags]['ind']]
            self.dim_size[self.dim_info[tags]['ind']] = self.dim_info[tags]['len']


    ##########################################################
    # Partial Fourier using Projection onto Convex Sets
    def POCS(self, kspace:np.ndarray, dim_pf=1, number_of_iterations=5):

        dims_enc = [self.dim_info['ro']['ind'], self.dim_info['pe1']['ind'] , self.dim_info['pe2']['ind']]
        dims_nopocs = tuple([int(x) for x in range(kspace.ndim) if x != dim_pf])        
        dims_nopocs_enc = dims_enc.copy()
        dims_nopocs_enc.remove(dim_pf)

        n_full = kspace.shape[dim_pf]
        n_zpf  = n_full - (self.enc_minmax_ind[dim_pf][1] - self.enc_minmax_ind[dim_pf][0]) # number of zeros along partial fourier dimension 
        minmax_ind = [self.enc_minmax_ind[dim_pf][0], self.enc_minmax_ind[dim_pf][1]]

        mask_raw = np.sum(np.abs(kspace), dims_nopocs) > 0
        ind_one  = np.nonzero(mask_raw == True)[0].tolist()
        nopocs_range = np.arange(ind_one[-1]+1)
        if ind_one[0] > (mask_raw.size - ind_one[-1] - 1):
            nopocs_range = np.arange(ind_one[0], mask_raw.size)

        acc = ind_one[1] - ind_one[0]
        shift = acc * ((mask_raw.size - nopocs_range.size) // acc)
        if nopocs_range[-1] == mask_raw.size - 1:
            shift = -shift

        # mask for non pocs dimension
        mask_raw[nopocs_range] = True
        mask_nonpocs = np.broadcast_to(np.expand_dims(mask_raw, axis=dims_nopocs), kspace.shape)

        # mask for pocs dimensions
        mask_pocs = np.abs(kspace) > 0
        mask_pocs = mask_pocs | np.roll(mask_pocs, shift, axis=dim_pf) # expland mask along pocs direction
        
        # vector mask for symmetric region
        mask_sym = mask_raw & np.flip(mask_raw)

        gauss_pdf = sp.stats.norm.pdf(np.linspace(0, 1, n_full), 0.5, 0.05) * mask_sym
        kspace_symmetric = kspace.copy()
        kspace_symmetric = np.swapaxes(np.swapaxes(kspace_symmetric, dim_pf, -1) * gauss_pdf, -1, dim_pf)

        angle_kspace_symmetric = ifftnd(kspace_symmetric, axes = dims_enc) # along all encoding directions
        angle_kspace_symmetric = np.exp(1j * np.angle(angle_kspace_symmetric))

        kspace_full = ifftnd(kspace, axes=dims_nopocs_enc) # along non-pocs encoding directions
        kspace_full_clone = kspace_full.copy()
        for i in range(number_of_iterations):
            image_full  = ifftnd(kspace_full, axes=dim_pf)
            image_full  = np.abs(image_full) * angle_kspace_symmetric
            kspace_full = fftnd(image_full, axes=dim_pf)
            np.putmask(kspace_full, mask_nonpocs, kspace_full_clone) # replace elements of kspace_full from kspace_full_clone based on mask_pocs

        kspace_full = fftnd(kspace_full, axes=dims_nopocs_enc)
        np.putmask(kspace_full, np.logical_not(mask_pocs), 0)
        return kspace_full

    
    ##########################################################
    # Save sampling pattern as mat file
    def save_sampling_pattern(self, volume:np.ndarray, filename):
        sampling_pattern = np.abs(volume) > 0
        sp.io.savemat(filename, {'sampling_pattern': sampling_pattern})
        print(f'Sampling pattern is saved as {os.path.abspath(filename)}')


    ##########################################################
    # Save a custom volume as nifti
    def make_nifti(self, volume:np.ndarray, filename):
        
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
