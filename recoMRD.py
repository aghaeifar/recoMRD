import os
import sys
import ismrmrd
import ismrmrd.xsd
import numpy as np
import h5py
import warnings
from tqdm import tqdm

# ifftnd and fftnd are taken from twixtools package
def ifftnd(kspace, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img = fftshift(ifftn(ifftshift(kspace, axes=axes), axes=axes), axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))
    return img

def fftnd(img, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, fftn
    if axes is None:
        axes = range(img.ndim)
    kspace = fftshift(fftn(ifftshift(img, axes=axes), axes=axes), axes=axes)
    kspace /= np.sqrt(np.prod(np.take(kspace.shape, axes)))
    return kspace

class recoMRD(object):    
    hdr     = None
    fov     = None
    xml     = None
    data    = None
    flags   = None
    affine  = None
    xml_hdr = None
    dim_info    = None
    dim_size    = None
    matrix_size = None
    img     = None
    kspace  = None
    
    def __init__(self, filename=None):   
        if sys.version_info < (3,7,0):
            raise SystemExit('Python version >= 3.7.x is required. Aborting...')

        self.filename  = filename                
        tags = ['cha', 'ro', 'pe1', 'pe2', 'slc', 'eco', 'rep', 'set', 'seg', 'ave', 'phs'] # order matters here
        self.dim_info = {}
        for i in range(len(tags)):
            self.dim_info[tags[i]] = {}
            self.dim_info[tags[i]]['len']  = 1
            self.dim_info[tags[i]]['ind'] = i

        self._import_mrd()
        self._extract_flags()
        self._create_kspace()

    def runReco(self):
        self._create_image()
        self._remove_oversamples()
        self._create_affine()
        self._reorder_slice()
        self._custom_task()
        self._coil_combination()
        self._create_mask()

    def _import_mrd(self):
        if not os.path.isfile(self.filename):
            print(f'file {self.filename} doesn\'t exist>')
            raise SystemExit('Goodbye')
        
        with h5py.File(self.filename, "r") as mrd:
            if len(mrd.keys()) > 1:
                print('MRD file has more than one group. The last group will be imported.')
            dataset_name = list(mrd.keys())[-1]
            data_struct  = mrd[dataset_name]['data'][:]
            self.xml     = mrd[dataset_name]['xml'][0]
            self.xml_hdr = ismrmrd.xsd.CreateFromDocument(mrd[dataset_name]['xml'][0])
            self.hdr     = data_struct['head']
            self.data    = data_struct['data']
    
    def _extract_flags(self):
        flags = {}
        flags['pc']         = np.bitwise_and( self.hdr['flags'] , 1 << ismrmrd.ACQ_IS_PHASECORR_DATA-1      ).astype(bool)
        flags['nav']        = np.bitwise_and( self.hdr['flags'] , 1 << ismrmrd.ACQ_IS_NAVIGATION_DATA-1     ).astype(bool)
        flags['iPAT']       = np.bitwise_and( self.hdr['flags'] , 1 << ismrmrd.ACQ_IS_PARALLEL_CALIBRATION-1).astype(bool)
        flags['noise']      = np.bitwise_and( self.hdr['flags'] , 1 << ismrmrd.ACQ_IS_NOISE_MEASUREMENT-1   ).astype(bool)
        flags['feedback']   = np.bitwise_and( self.hdr['flags'] , 1 << ismrmrd.ACQ_IS_RTFEEDBACK_DATA-1     ).astype(bool)
        flags['image_scan'] = np.logical_not( flags['iPAT'] | flags['nav'] | flags['pc'] | flags['noise'] | flags['feedback'])
        self.flags = flags

        enc     = self.xml_hdr.encoding[0]
        # Matrix size
        matrix_size                = {'kspace':{}, 'image':{}}
        matrix_size['image']['x']  = enc.reconSpace.matrixSize.x
        matrix_size['image']['y']  = enc.reconSpace.matrixSize.y
        matrix_size['image']['z']  = enc.reconSpace.matrixSize.z
        matrix_size['kspace']['x'] = enc.encodedSpace.matrixSize.x
        matrix_size['kspace']['y'] = enc.encodedSpace.matrixSize.y
        matrix_size['kspace']['z'] = enc.encodedSpace.matrixSize.z
        self.matrix_size = matrix_size

        # Field of View
        fov                = {'kspace':{}, 'image':{}}
        fov['image']['x']  = enc.reconSpace.fieldOfView_mm.x
        fov['image']['y']  = enc.reconSpace.fieldOfView_mm.y
        fov['image']['z']  = enc.reconSpace.fieldOfView_mm.z
        fov['kspace']['x'] = enc.encodedSpace.fieldOfView_mm.x
        fov['kspace']['y'] = enc.encodedSpace.fieldOfView_mm.y
        fov['kspace']['z'] = enc.encodedSpace.fieldOfView_mm.z
        self.fov = fov

        # Dimensions Size
        self.dim_info['cha']['len'] = self.hdr['active_channels'][0]
        self.dim_info['ro']['len']  = self.hdr['number_of_samples'][0]

        if enc.encodingLimits.kspace_encoding_step_1 != None:
            self.dim_info['pe1']['len'] = enc.encodingLimits.kspace_encoding_step_1.maximum + 1

        if enc.encodingLimits.kspace_encoding_step_2 != None:
            self.dim_info['pe2']['len'] = enc.encodingLimits.kspace_encoding_step_2.maximum + 1

        if enc.encodingLimits.slice != None:
            self.dim_info['slc']['len'] = enc.encodingLimits.slice.maximum + 1

        if enc.encodingLimits.contrast != None:
            self.dim_info['eco']['len'] = enc.encodingLimits.contrast.maximum + 1

        if enc.encodingLimits.repetition != None:
            self.dim_info['rep']['len'] = enc.encodingLimits.repetition.maximum + 1

        if enc.encodingLimits.set != None:
            self.dim_info['set']['len'] = enc.encodingLimits.set.maximum + 1

        if enc.encodingLimits.set != None:
            self.dim_info['seg']['len'] = enc.encodingLimits.segment.maximum + 1

        if enc.encodingLimits.average != None:
            self.dim_info['ave']['len'] = enc.encodingLimits.average.maximum + 1

        if enc.encodingLimits.phase != None:
            self.dim_info['phs']['len'] = enc.encodingLimits.phase.maximum + 1

        self.dim_size = [1] * len(self.dim_info)
        for i in self.dim_info.keys():
            self.dim_size[self.dim_info[i]['ind']] = self.dim_info[i]['len']

        if self.dim_info['ro']['len'] != matrix_size['kspace']['x']:
            print(f"\033[93m Number of RO samples ({self.dim_info['ro']['len']}) differs from expectation ({matrix_size['kspace']['x']})\033[0m")
            

    def _create_kspace(self):
        dif = self.dim_info
        dsz = self.dim_size
        # I moved 'cha' and 'ro' to end of numpy array. with this, filling the kspace is much faster in loop
        dsz_p = dsz[2:] + dsz[0:2]
        kspace = np.zeros(dsz_p, dtype=np.complex64)

        for ind in tqdm(list(*np.where(self.flags['image_scan'])), desc='Building k-space'):
            data_tr = (self.data[ind][0::2] + 1j*self.data[ind][1::2]).reshape((dif['cha']['len'], dif['ro']['len']))
            kspace[self.hdr['idx']['kspace_encode_step_1'][ind],
                   self.hdr['idx']['kspace_encode_step_2'][ind],
                   self.hdr['idx']['slice'][ind],
                   self.hdr['idx']['contrast'][ind],
                   self.hdr['idx']['repetition'][ind],
                   self.hdr['idx']['set'][ind],
                   self.hdr['idx']['segment'][ind],
                   self.hdr['idx']['average'][ind],
                   self.hdr['idx']['phase'][ind], :, :] = data_tr
        
        # correcting the dimensions order
        dsz_i = [*range(len(dsz))]
        self.kspace = np.transpose(kspace, dsz_i[-2:] + dsz_i[:-2] )


    def _create_image(self):
        self.img = np.zeros(self.dim_size, dtype=np.complex64)
        for ind in tqdm(range(self.dim_info['cha']['len']), desc='Fourier transform'):
            temp = self.kspace[ind,:,:,:,:,:,:,:,:,:,:]
            self.img[ind,:,:,:,:,:,:,:,:,:,:] = ifftnd(temp, [0,1,2])
        

    def _coil_combination(self):
        self.img = np.sqrt(np.sum(abs(self.img)**2, self.dim_info['cha']['ind'], keepdims=True))
        self.dim_info['cha']['len'] = 1


    def _remove_oversamples(self):
        cutoff = (self.dim_info['ro']['len'] - self.matrix_size['image']['x']) // 2
        self.img = self.img[:,cutoff:-cutoff,:,:,:,:,:]
        

    def _reorder_slice(self):
        unsorted_order = np.zeros((self.dim_info['slc']['len']))
        for cslc in range(self.dim_info['slc']['len']):
            p1 = np.linalg.solve(self.affine['mat44'], self.affine['soda'][cslc,:,3])
            unsorted_order[cslc] = p1[2]
        ind_sorted = np.argsort(unsorted_order)
        self.img   = self.img[:,:,:,:,ind_sorted,:,:,:,:,:,:]
        self.affine['soda'] = self.affine['soda'][ind_sorted,:,:]

    def _custom_task(self):
        pass
    
    def _create_mask(self):
        pass

    def _create_affine(self):
        hdr = self.hdr
        affine = {}
        affine['soda'] = np.zeros((self.dim_info['slc']['len'], 4, 4))
        offcenter = np.zeros(3)
        for cslc in range(self.dim_info['slc']['len']):
            ind = np.where((hdr['idx']['slice'] == cslc) & self.flags['image_scan'])[0]
            if len(ind) == 0:
                print(f"\033[91m slice index not found! aborting...\033[0m")
                raise SystemExit('Goodbye')
            
            dcm = np.column_stack((hdr['phase_dir'][ind[0],:], 
                                   hdr['read_dir'] [ind[0],:], 
                                   hdr['slice_dir'][ind[0],:]))

            affine['soda'][cslc,0:3,:] = np.column_stack((dcm, hdr['position'][ind[0],:]))
            affine['soda'][cslc,:,:]   = np.row_stack((affine['soda'][cslc,0:3,:], [0, 0, 0, 1]))           
            offcenter += hdr['position'][ind[0],:]

        offcenter /= self.dim_info['slc']['len']
        affine['offcenter']     = offcenter 
        affine['mat44']         = affine['soda'][0,:,:] 
        affine['mat44'][0:3,3]  = offcenter
        self.affine = affine


    def make_nifti(self):
        pass

# %%
# from recoMRD import recoMRD
# filename = '/DATA/aaghaeifar/rawdata/aa_B0Mapping_basismaps/meas_MID00047_FID27956_aa_B0Mapping_2ndorder_IECO.mrd'
# mrd = recoMRD(filename)
# mrd.runReco()
#