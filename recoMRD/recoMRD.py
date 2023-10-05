import os
import math
import ctypes
import torch
import numpy as np
import nibabel as nib
from bart import bart
from tqdm import tqdm
from .readMRD import readMRD

BART_COIL_DIM  = 3
BART_SLICE_DIM = 13
BART_BATCH_DIM = 15
BART_TE_DIM    = 5


def ifftnd(kspace:torch.Tensor, axes=[-1]):
    from torch.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img  = fftshift(ifftn(ifftshift(kspace, dim=axes), dim=axes), dim=axes)
    return img


def fftnd(img:torch.Tensor, axes=[-1]):
    from torch.fft import fftshift, ifftshift, fftn
    if axes is None:
        axes = range(img.ndim)
    kspace  = fftshift(fftn(ifftshift(img, dim=axes), dim=axes), dim=axes)
    return kspace

# adapting to bart CFL format, see https://bart-doc.readthedocs.io/en/latest/data.html or https://github.com/mrirecon/bart/blob/master/src/misc/mri.h
# Dimensions in BART (/src/misc/mri.h): [READ_DIM, PHS1_DIM, PHS2_DIM, COIL_DIM, MAPS_DIM, TE_DIM, COEFF_DIM, COEFF2_DIM, ITER_DIM, CSHIFT_DIM, TIME_DIM, TIME2_DIM, LEVEL_DIM, SLICE_DIM, AVG_DIM, BATCH_DIM]
def toBART(kspace:torch.Tensor, dim_info):
    kspace = kspace.reshape(kspace.shape[:dim_info['slc']['ind']+1] + (-1,)) # flatten extra dims
    ind_lastDim = kspace.ndim - 1 # index of last dimension
    kspace = kspace[(...,) + (None,)*(BART_BATCH_DIM - kspace.ndim + 1)] # add extra dims to match BART format size
    kspace = kspace.moveaxis((dim_info['cha']['ind'], dim_info['slc']['ind'], ind_lastDim), (BART_COIL_DIM, BART_SLICE_DIM, BART_TE_DIM)) # match BART dim order
    return kspace

# converting BART data format to recoMRD data format
def fromBART(kspace:torch.Tensor, dim_info, original_shape):
    kspace = kspace[(...,) + (None,)*(BART_BATCH_DIM - kspace.ndim + 1)] # add extra dims to match BART format size if is not already!
    kspace = kspace.moveaxis((BART_COIL_DIM, BART_SLICE_DIM, BART_TE_DIM), (dim_info['cha']['ind'], dim_info['slc']['ind'], dim_info['slc']['ind']+1))
    return kspace.reshape(original_shape)


class recoMRD(readMRD):    
    img = None
    device = 'cpu'
  
    def __init__(self, filename=None, device='cpu'):   
        super().__init__(filename)
        self.device = device
        torch.cuda.empty_cache()
        for keys in self.kspace:
            if isinstance(self.kspace[keys], np.ndarray):                
                self.kspace[keys] = torch.from_numpy(self.kspace[keys])

    def runReco(self, method_sensitivity='caldir', remove_os=True):     
        torch.cuda.empty_cache()   
        # Removing Oversampling?
        kspace = self.kspace['image_scan'] if not remove_os else self.remove_oversampling(self.kspace['image_scan'], is_kspace=True)

        # Partial Fourier?
        if self.isPartialFourierRO:
            kspace = self.POCS(kspace, dim_pf= self.dim_info['ro']['ind'])
        if self.isPartialFourierPE1:
            kspace = self.POCS(kspace, dim_pf= self.dim_info['pe1']['ind'])
        if self.isPartialFourierPE2:
            kspace = self.POCS(kspace, dim_pf= self.dim_info['pe2']['ind'])

        coils_sensitivity = None
        # Parallel Imaging?
        if self.isParallelImaging:
            acs = self.kspace['acs'] if not remove_os else self.remove_oversampling(self.kspace['acs'], is_kspace=True)
            coils_sensitivity = self.calc_coil_sensitivity(acs, method=method_sensitivity)
        else:
            coils_sensitivity = self.calc_coil_sensitivity(kspace, method=method_sensitivity)

        self.img = self.coil_combination(kspace, coil_sens=coils_sensitivity)

    ##########################################################
    # applying iFFT to kspace and build image
    def kspace_to_image(self, kspace:torch.Tensor, axes=None):
        if kspace.ndim != len(self.dim_size):
            print(f'Error! shape is wrong. {kspace.shape} vs {self.dim_size}')
            return
        if axes is None:
            axes = self.dim_enc

        img = torch.zeros_like(kspace, dtype=kspace.dtype)
        # this loop is slow because of order='F' and ind is in the first dimensions. see above, _create_kspace(). 
        for ind in tqdm(range(self.dim_info['cha']['len']), desc='Fourier transform'):
            img[ind,...] = ifftnd(kspace[[ind],...], axes=axes) # [ind] --> https://stackoverflow.com/questions/3551242/
        return img

    def image_to_kspace(self, img:torch.Tensor, axes=None):
        if img.ndim != len(self.dim_size):
            print(f'Error! shape is wrong. {img.shape} vs {self.dim_size}')
            return
        if axes is None:
            axes = self.dim_enc

        kspace = torch.zeros_like(img, dtype=img.dtype)
        # this loop is slow because of order='F' and ind is in the first dimensions. see above, _create_kspace(). 
        for ind in tqdm(range(self.dim_info['cha']['len']), desc='Fourier transform'):
            kspace[ind,...] = fftnd(img[[ind],...], axes=axes) # [ind] --> https://stackoverflow.com/questions/3551242/     
        return kspace

    ##########################################################
    def coil_combination(self, kspace:torch.Tensor, coil_sens:torch.Tensor, rss=False):
        print(f'Combining coils... ')
        torch.cuda.empty_cache()
        if kspace.ndim != len(self.dim_size):
            print(f'Input size is not valid. {kspace.shape} != {self.dim_size}')
            return
        
        if kspace.shape[:self.dim_info['slc']['ind']+1] != coil_sens.shape[:self.dim_info['slc']['ind']+1] :
            print(f'Coils Sensitivity size is not valid. {kspace.shape} != {coil_sens.shape}')
            return
        
        shp = (1,) + kspace.shape[1:] # coil is 1 after combining
        # sos    
        volume       = self.kspace_to_image(kspace)       
        volume_comb  = torch.sqrt(torch.sum(torch.abs(volume)**2, self.dim_info['cha']['ind'], keepdims=True)) # is needed in 'bart' to calculate scale factor
        
        if rss == False:
            GPU          = '-g' if self.device == 'cuda' else ''
            l2_reg       = 1e-4
            kspace       = toBART(kspace, self.dim_info)
            coil_sens    = toBART(coil_sens, self.dim_info)
            scale_factor = torch.quantile(volume_comb, 0.99).tolist()        
            recon        = bart.bart(1, f'-p {1<<BART_SLICE_DIM} -e {kspace.shape[BART_SLICE_DIM]} pics {GPU} -d4 -w {scale_factor} -R Q:{l2_reg} -S', kspace.numpy(), coil_sens.numpy())
            volume_comb  = fromBART(torch.from_numpy(recon), self.dim_info, shp)
            
        print('Done!')
        return volume_comb

    ##########################################################
    def calc_coil_sensitivity(self, acs:torch.Tensor, method='caldir'):
        print('Computing coil sensitivity...')
        torch.cuda.empty_cache()
        all_methods = ('espirit', 'caldir', 'walsh')
        
        if method.lower() not in all_methods:
            print(f'Given method is not valid. Choose between {", ".join(all_methods)}')
            return
        d = self.dim_info
        if d['cha']['ind']!=0 or d['ro']['ind']!=1 or d['pe1']['ind']!=2 or d['pe2']['ind']!=3 or d['slc']['ind']!=4:
            print('Error! Dimension order does not fit to the desired order.')
            return

        # picking the 0th element of the free dimensions
        for dim_free in self.dim_free:
            acs = acs.index_select(self.dim_info[dim_free]['ind'], torch.Tensor([0]).int()) 
        acs_bart = toBART(acs, self.dim_info).numpy() # adapting to bart CFL format

        if method.lower() == 'espirit'.lower():
            coil_sens = bart.bart(1, f'-p {1<<BART_SLICE_DIM} -e {acs_bart.shape[BART_SLICE_DIM]} ecalib -m 1 -d 4', acs_bart)
            coil_sens = fromBART(torch.from_numpy(coil_sens), self.dim_info, acs.shape)

        elif method.lower() == 'caldir'.lower():
            kernel_size = max([acs.shape[d] for d in self.dim_enc])//2
            coil_sens = bart.bart(1, '-p {} -e {} caldir {}'.format(1<<BART_SLICE_DIM, acs_bart.shape[BART_SLICE_DIM], kernel_size), acs_bart)
            coil_sens = fromBART(torch.from_numpy(coil_sens), self.dim_info, acs.shape)

        print('Done!')
        return coil_sens

    ##########################################################
    def compress_coil(self, *kspace:torch.Tensor, virtual_channels=None): 
        # kspace[0] is the reference input to create compression matrix for all inputs, it should be GRAPPA scan for example.    
        print('Compressing Rx channels...')
        torch.cuda.empty_cache()
        if virtual_channels == None:
            virtual_channels = int(kspace[0].shape[self.dim_info['cha']['ind']] * 0.75)

        kspace_cfl = [torch.moveaxis(kspc, self.dim_info['cha']['ind'], BART_COIL_DIM) for kspc in kspace] # adapting to bart CFL format
        cc_matrix  = [bart.bart(1, 'bart cc -A -M', kspace_cfl[0][...,cslc,0,0,0,0,0,0].numpy()) for cslc in range(self.dim_info['slc']['len'])]

        kspace_compressed_cfl = []
        for kspc in kspace_cfl:
            n_extra1 = torch.prod(kspc.shape[self.dim_info['slc']['ind']:])   # number of extra dims  
            n_extra2 = torch.prod(kspc.shape[self.dim_info['slc']['ind']+1:]) # number of extra dims excluing slice
            kspc_r   = kspc.reshape(kspc.shape[:self.dim_info['slc']['ind']] + (-1,)) # flatten extra dims
            kspc_cc  = [bart.bart(1, f'ccapply -p {virtual_channels}', k, cc_matrix[i%n_extra2]) for k, i in zip(kspc_r, range(n_extra1))]
            kspace_compressed_cfl.append(torch.from_numpy(np.stack(kspc_cc, axis=4)).reshape(kspc.shape))

        kspace_compressed = [torch.moveaxis(kspc, BART_COIL_DIM, self.dim_info['cha']['ind']) for kspc in kspace_compressed_cfl]
        return (*kspace_compressed,)

    ##########################################################
    def remove_oversampling(self, img:torch.Tensor, is_kspace=False):
        print('Removing oversampling...')
        torch.cuda.empty_cache()
        if img.ndim != len(self.dim_size):
            print(f'Error! not same dimensionality. {img.shape} vs {self.dim_size}')
            return
        
        if img.shape[self.dim_info['ro']['ind']] != self.dim_info['ro']['len']:
            print('Oversampling is already removed!')
            return
        
        if is_kspace:
            os_factor = self.dim_info['ro']['len'] / self.matrix_size['image']['x'] # must be divisible, otherwise I made a mistake somewhere
            ind = torch.arange(0, self.dim_info['ro']['len'], os_factor, dtype=torch.long)
        else:
            cutoff = (img.shape[self.dim_info['ro']['ind']] - self.matrix_size['image']['x']) // 2 # // -> integer division
            ind = torch.arange(cutoff, cutoff+self.matrix_size['image']['x'], dtype=torch.long) # img[:,cutoff:-cutoff,...]
        
        img = torch.index_select(img, dim=self.dim_info['ro']['ind'], index=ind)

        print('Done!')
        return img


    ##########################################################
    # Partial Fourier using Projection onto Convex Sets
    def POCS(self, kspace:torch.Tensor, dim_pf=1, number_of_iterations=5):
        print(f'POCS reconstruction along dim = {dim_pf} started...')
        torch.cuda.empty_cache()
        kspace = kspace.to(self.device)

        dim_nonpf     = tuple([int(x) for x in range(kspace.ndim) if x != dim_pf])        
        dim_nonpf_enc = tuple(set(self.dim_enc) & set(dim_nonpf))

        n_full = kspace.shape[dim_pf] 
        # mask for partial Fourier dimension taking accelleration into account
        mask    = torch.sum(torch.abs(kspace), dim_nonpf) > 0 # a mask along PF direction, considering acceleration, type: tensor
        mask_clone = mask.clone()
        ind_one = torch.nonzero(mask == True, as_tuple=True)[0] # index of mask_pf_acc, type: tensor
        acc_pf  = ind_one[1] - ind_one[0] # accelleration in partial Fourier direction
        # partial Fourier is at the beginning or end of the dimension
        ind_samples = torch.arange(ind_one[-1]+1) # index of samples in PF direction, without acceleration. ind_nopocs does not take accelleration into account, right?
        if ind_one[0] > (mask.numel() - ind_one[-1] - 1): # check which side has more zeros, beginning or end
            ind_samples = torch.arange(ind_one[0], mask.numel())

        # mask if there was no accelleration in PF direction
        mask[ind_samples] = True 
        # vector mask for central region
        mask_sym = mask & torch.flip(mask, dims=[0])      
        # gaussian mask for central region in partial Fourier dimension
        gauss_pdf = torch.signal.windows.gaussian(n_full, std=10, device=kspace.device) * mask_sym
        # kspace smoothed with gaussian profile and masked central region
        kspace_symmetric = kspace.clone()
        kspace_symmetric = torch.swapaxes(torch.swapaxes(kspace_symmetric, dim_pf, -1) * gauss_pdf, -1, dim_pf)
        angle_image_symmetric  = self.kspace_to_image(kspace_symmetric) 
        angle_image_symmetric /= torch.abs(angle_image_symmetric) # normalize to unit circle         

        kspace_full = self.kspace_to_image(kspace, axes=dim_nonpf_enc) # along non-pf encoding directions
        kspace_full_clone = kspace_full.clone()
        # free memory
        del kspace_symmetric 
        del kspace
        torch.cuda.empty_cache()

        for ind in range(number_of_iterations):
            image_full  = self.kspace_to_image(kspace_full, axes=[dim_pf])
            image_full  = torch.abs(image_full) * angle_image_symmetric
            kspace_full = self.image_to_kspace(image_full, axes=[dim_pf])
            torch.moveaxis(kspace_full, dim_pf, 0)[mask] = torch.moveaxis(kspace_full_clone, dim_pf, 0)[mask] # replace elements of kspace_full from original kspace_full_clone

        kspace_full = self.image_to_kspace(kspace_full, axes=dim_nonpf_enc)
        # remove all samples that was not part of the original dataset (e.g. acceleartion)        
        mask = mask_clone
        mask[ind_one[0]%acc_pf::acc_pf] = True
        torch.moveaxis(kspace_full, dim_pf, 0)[~mask] = 0       
        print('Done!')

        return kspace_full.to('cpu') 

    ##########################################################
    def noise_whitening(self, kspace:torch.Tensor, noise:torch.Tensor):
        pass

    ##########################################################
    def get_gfactor(self, kspace:torch.Tensor, coil_sens:torch.Tensor):
        pass

    ##########################################################
    # Save a custom volume as nifti
    def make_nifti(self, volume:torch.Tensor, filename):        
        # input must have same dimension ordering as dim_info
        check_dims = [self.dim_info['ro']['ind'], self.dim_info['pe1']['ind'], self.dim_info['pe2']['ind'], self.dim_info['slc']['ind'], self.dim_info['rep']['ind']] # the dimensions that must be checked
        ds = [self.dim_size[y] for y in check_dims]
        vs = [volume.shape[y] for y in check_dims]
        if vs != ds and vs[0]*2 != ds[0]: # second condition is to account for oversampling
            print(f"Size mismatch (RO, PE1, PE2, SLC, REP)! {vs } vs {ds}")
            return

        vs = [y for y in volume.shape if y!=1]
        if len(vs) > 4 :
            print(f"{len(vs)}D data is not supported")
            return

        volume = torch.flip(volume, dims=[self.dim_info['ro']['ind'], 
                                          self.dim_info['pe1']['ind'],
                                          self.dim_info['slc']['ind']])
        #
        # creating permute indices
        #
        # prmt_ind = np.arange(0, len(self.dim_info), 1, dtype=int)
        # # swap ro and pe1
        # prmt_ind[[self.dim_info['ro']['ind'], self.dim_info['pe1']['ind']]] = prmt_ind[[self.dim_info['pe1']['ind'], self.dim_info['ro']['ind']]] 
        # # move cha to end
        # icha = self.dim_info['cha']['ind']
        # prmt_ind = np.hstack([prmt_ind[:icha], prmt_ind[icha+1:], prmt_ind[icha]])
    
        # volume = np.transpose(volume, prmt_ind)

        volume = volume.swapaxes(self.dim_info['ro']['ind'], self.dim_info['pe1']['ind']).moveaxis(self.dim_info['cha']['ind'], -1).squeeze()
       
        #
        # save to file
        #
        img = nib.Nifti1Image(volume.numpy(), self.nii_affine)
        nib.save(img, filename)
