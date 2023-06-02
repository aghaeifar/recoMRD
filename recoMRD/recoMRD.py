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


    def runReco(self, method_sensitivity='caldir', method_coilcomb='bart'):
        kspace_osremoved = self.remove_oversampling(self.kspace['image_scan'], is_kspace=True)
        # Partial Fourier?
        if self.isPartialFourierRO:
            kspace_osremoved = self.POCS(kspace_osremoved, dim_pf= self.dim_info['ro']['ind'])
        if self.isPartialFourierPE1:
            kspace_osremoved = self.POCS(kspace_osremoved, dim_pf= self.dim_info['pe1']['ind'])
        if self.isPartialFourierPE2:
            kspace_osremoved = self.POCS(kspace_osremoved, dim_pf= self.dim_info['pe2']['ind'])

        coils_sensitivity = None
        # Parallel Imaging?
        if self.isParallelImaging:
            acs_img = self.kspace['acs']
            acs_img = self.remove_oversampling(acs_img, is_kspace=True)
            coils_sensitivity = self.calc_coil_sensitivity(acs_img, method=method_sensitivity)
        else:
            coils_sensitivity = self.calc_coil_sensitivity(kspace_osremoved, method=method_sensitivity)

        self.img = self.coil_combination(kspace_osremoved, method=method_coilcomb, coil_sens=coils_sensitivity)

    ##########################################################
    # applying iFFT to kspace and build image
    def kspace_to_image(self, kspace:np.ndarray, axes=None):
        if kspace.ndim != len(self.dim_size):
            print(f'Error! shape is wrong. {kspace.shape} vs {self.dim_size}')
            return
        if axes is None:
            axes = self.dim_enc

        img = np.zeros_like(kspace, dtype=np.complex64)
        # this loop is slow because of order='F' and ind is in the first dimensions. see above, _create_kspace(). 
        for ind in tqdm(range(self.dim_info['cha']['len']), desc='Fourier transform'):
            img[ind,...] = ifftnd(kspace[[ind],...], axes=axes) # [ind] --> https://stackoverflow.com/questions/3551242/
        return img

    def image_to_kspace(self, img:np.ndarray, axes=None):
        if img.ndim != len(self.dim_size):
            print(f'Error! shape is wrong. {img.shape} vs {self.dim_size}')
            return
        if axes is None:
            axes = self.dim_enc

        kspace = np.zeros_like(img, dtype=np.complex64)
        # this loop is slow because of order='F' and ind is in the first dimensions. see above, _create_kspace(). 
        for ind in tqdm(range(self.dim_info['cha']['len']), desc='Fourier transform'):
            kspace[ind,...] = fftnd(img[[ind],...], axes=axes) # [ind] --> https://stackoverflow.com/questions/3551242/     
        return kspace

    ##########################################################
    def coil_combination(self, kspace:np.ndarray, method='sos', coil_sens=None):
        if kspace.ndim != len(self.dim_size):
            print(f'Input size is not valid. {kspace.shape} != {self.dim_size}')
            return
        if coil_sens is not None:
            if kspace.shape[:self.dim_info['slc']['ind']+1] != coil_sens.shape:
                print(f'Coils Sens. size is not valid. {kspace.shape[:self.dim_info["slc"]["ind"]+1]} != {coil_sens.shape}')
                return

        all_methods = ('sos', 'bart', 'adaptive')
        if method.lower() not in all_methods:
            print(f'Given method is not valid. Choose between {", ".join(all_methods)}')
            return
        
        shp = (1,) + kspace.shape[1:]
        # sos    
        volume       = self.kspace_to_image(kspace) # is needed in 'adaptive'
        volume_comb  = np.sqrt(np.sum(abs(volume)**2, self.dim_info['cha']['ind'], keepdims=True)) # is needed in 'bart' to calculate scale factor
        if method.lower() == 'bart' and coil_sens is not None:
            l2_reg       = 1e-4
            kspace       = np.moveaxis(kspace, 0, 3) # adapting to bart CFL format: [RO, PE1, PE2, CHA, ...] see https://bart-doc.readthedocs.io/en/latest/data.html or https://github.com/mrirecon/bart/blob/master/src/misc/mri.h
            coil_sens    = np.moveaxis(coil_sens, 0, 3) # adapting to bart CFL format
            n_nonBART1   = np.prod(kspace.shape[self.dim_info['slc']['ind']:]) # number of non-BART dims 
            n_nonBART2   = np.prod(kspace.shape[self.dim_info['slc']['ind']+1:]) # number of non-BART dims ecluding slice dim
            kspace       = kspace.reshape(kspace.shape[:4] +(-1,)) # flatten extra dims
            volume_comb  = volume_comb.reshape(volume_comb.shape[:4] + (-1,)) # flatten extra dims for output
            scale_factor = [np.percentile(volume_comb[...,ind], 99).astype(np.float32) for ind in range(volume_comb.shape[-1])]
            recon        = [bart.bart(1, 'pics -w {} -R Q:{} -S'.format(scale_factor[ind], l2_reg), kspace[...,ind], coil_sens[...,ind//n_nonBART2]) for ind in range(n_nonBART1)]
            volume_comb  = np.stack(recon, axis=recon[0].ndim).reshape(shp)

        elif method.lower() == 'adaptive' and coil_sens is not None:
            coil_sens    = np.expand_dims(coil_sens, axis=[*range(coil_sens.ndim, kspace.ndim)]) # https://numpy.org/doc/stable/user/basics.broadcasting.html
            volume_comb  = np.divide(volume, coil_sens, out=np.zeros_like(coil_sens), where=coil_sens!=0)
            volume_comb  = np.sum(volume_comb, self.dim_info['cha']['ind'], keepdims=True)

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
            
            acs = self.kspace_to_image(acs)
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
        if img.ndim != len(self.dim_size):
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
        print(f'POCS reconstruction along dim = {dim_pf} started...')
        dim_nonpf = tuple([int(x) for x in range(kspace.ndim) if x != dim_pf])        
        dim_nonpf_enc = tuple(set(self.dim_enc) & set(dim_nonpf))

        n_full = kspace.shape[dim_pf]
        # n_zpf  = n_full - (self.enc_minmax_ind[dim_pf][1] - self.enc_minmax_ind[dim_pf][0]) # number of zeros along partial fourier dimension 
        # minmax_ind = [self.enc_minmax_ind[dim_pf][0], self.enc_minmax_ind[dim_pf][1]]

        # mask for partial Fourier dimension taking accelleration into account
        mask_pf_acc = np.sum(np.abs(kspace), dim_nonpf) > 0
        ind_one  = np.nonzero(mask_pf_acc == True)[0].tolist()
        # partial Fourier is at the beginning or end of the dimension
        nopocs_range = np.arange(ind_one[-1]+1) # nopocs_range does not take accelleration into account, right?
        if ind_one[0] > (mask_pf_acc.size - ind_one[-1] - 1): # check which side has more zeros, beginning or end
            nopocs_range = np.arange(ind_one[0], mask_pf_acc.size)

        # accelleration in partial Fourier direction
        acc_pf = ind_one[1] - ind_one[0]
        shift = acc_pf * ((mask_pf_acc.size - nopocs_range.size) // acc_pf)
        if nopocs_range[-1] == mask_pf_acc.size - 1: # again, check which side partial Fourier is
            shift = -shift

        # mask if there was no accelleration in PF direction
        mask_pf = mask_pf_acc.copy()
        mask_pf[nopocs_range] = True 
        # vector mask for central region
        mask_sym = mask_pf & np.flip(mask_pf)
        # mask for entire kspace with no accelleration in PF direction
        mask_pf = np.broadcast_to(np.expand_dims(mask_pf, axis=dim_nonpf), kspace.shape)
        
        # mask that takes accelleration into account without partial Fourier
        mask_nonpf = np.abs(kspace) > 0
        mask_nonpf = mask_nonpf | np.roll(mask_nonpf, shift, axis=dim_pf) # expland mask along pocs direction
        
        # gaussian mask for central region in partial Fourier dimension
        gauss_pdf = sp.stats.norm.pdf(np.linspace(0, 1, n_full), 0.5, 0.05) * mask_sym
        # kspace smoothed with gaussian profile and masked central region
        kspace_symmetric = kspace.copy()
        kspace_symmetric = np.swapaxes(np.swapaxes(kspace_symmetric, dim_pf, -1) * gauss_pdf, -1, dim_pf)

        angle_image_symmetric = self.kspace_to_image(kspace_symmetric)
        angle_image_symmetric = np.exp(1j * np.angle(angle_image_symmetric))

        kspace_full = self.kspace_to_image(kspace, axes=dim_nonpf_enc) # along non-pf encoding directions
        kspace_full_clone = kspace_full.copy()
        for ind in range(number_of_iterations):
            image_full  = self.kspace_to_image(kspace_full, axes=[dim_pf])
            image_full  = np.abs(image_full) * angle_image_symmetric
            kspace_full = self.image_to_kspace(image_full, axes=[dim_pf])
            np.putmask(kspace_full, mask_pf, kspace_full_clone) # replace elements of kspace_full from kspace_full_clone based on mask_nonpf

        kspace_full = self.image_to_kspace(kspace_full, axes=dim_nonpf_enc)
        np.putmask(kspace_full, np.logical_not(mask_nonpf), 0)
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
        # input must have same dimension ordering as dim_info
        df = self.dim_info
        check_dims = [df['ro']['ind'], df['pe1']['ind'], df['pe2']['ind'], df['slc']['ind'], df['rep']['ind']]
        ds = [self.dim_size[y] for y in check_dims]
        vs = [volume.shape[y] for y in check_dims]
        if vs != ds and vs[0]*2 != ds[0]: # second condition is to account for oversampling
            print(f"Size mismatch (RO, PE1, PE2, SLC, REP)! {vs } vs {ds}")
            return

        vs = [y for y in volume.shape  if y!=1]
        if len(vs) > 4 :
            print(f"{len(vs)}D data is not supported")
            return

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
        # save to file
        #
        img = nib.Nifti1Image(volume, self.nii_affine)
        nib.save(img, filename)
