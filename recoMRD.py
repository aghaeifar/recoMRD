import os
import sys
import h5py
import ismrmrd
import numpy as np
import nibabel as nib
from tqdm import tqdm
from ismrmrd.xsd import CreateFromDocument as ismrmrd_xml_parser


# ifftnd and fftnd are taken from twixtools package
def ifftnd(kspace, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img  = fftshift(ifftn(ifftshift(kspace, axes=axes), axes=axes), axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))
    return img

def fftnd(img, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, fftn
    if axes is None:
        axes = range(img.ndim)
    kspace  = fftshift(fftn(ifftshift(img, axes=axes), axes=axes), axes=axes)
    kspace /= np.sqrt(np.prod(np.take(kspace.shape, axes)))
    return kspace

class recoMRD(object):    
    hdr     = None
    fov     = None
    xml     = None
    data    = None
    flags   = None    
    xml_hdr = None
    dim_info    = None
    dim_size    = None
    matrix_size = None
    img     = None
    kspace  = None
    transformation  = None
    
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
        self._extract_transformation()
        self._reorder_slice()
        self._custom_task()
        self._coil_combination()

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
            self.xml_hdr = ismrmrd_xml_parser(mrd[dataset_name]['xml'][0])
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
            print(f"\033[93mNumber of RO samples ({self.dim_info['ro']['len']}) differs from expectation ({matrix_size['kspace']['x']})\033[0m")
            

    def _create_kspace(self):
        dif = self.dim_info
        dsz = self.dim_size
        # I used here order='F' since for order='C', which is default, filling the kspace was slower. 
        # To make order='C' fast, we need to move 'cha' and 'ro' to end of numpy array. However, we have to
        # permute dimensions later then, which will break continuity in memory, i.e. self.kspace.flags is false
        # for C_CONTIGUOUS and F_CONTIGUOUS
        # dsz_p = dsz[2:] + dsz[0:2]
        kspace = np.zeros(dsz, dtype=np.complex64, order = 'F')

        for ind in tqdm(list(*np.where(self.flags['image_scan'])), desc='Filling k-space'):
            data_tr = (self.data[ind][0::2] + 1j*self.data[ind][1::2]).reshape((dif['cha']['len'], dif['ro']['len']))
            kspace[:,:, 
                   self.hdr['idx']['kspace_encode_step_1'][ind],
                   self.hdr['idx']['kspace_encode_step_2'][ind],
                   self.hdr['idx']['slice'][ind],
                   self.hdr['idx']['contrast'][ind],
                   self.hdr['idx']['repetition'][ind],
                   self.hdr['idx']['set'][ind],
                   self.hdr['idx']['segment'][ind],
                   self.hdr['idx']['average'][ind],
                   self.hdr['idx']['phase'][ind]] = data_tr
        
        # correcting the dimensions order
        dsz_i = [*range(len(dsz))]
        self.kspace = kspace # np.transpose(kspace, dsz_i[-2:] + dsz_i[:-2])

    # applying iFFT to kspace and build image
    def _create_image(self):
        self.img = np.zeros(self.dim_size, dtype=np.complex64, order='F')
        # this loop is slow because of order='F' and ind is in the first dimensions. see above, _create_kspace(). 
        for ind in tqdm(range(self.dim_info['cha']['len']), desc='Fourier transform'):
            temp = self.kspace[ind,:,:,:,:,:,:,:,:,:,:]
            self.img[ind,:,:,:,:,:,:,:,:,:,:] = ifftnd(temp, [0,1,2])        

    def _coil_combination(self):
        self.img = np.sqrt(np.sum(abs(self.img)**2, self.dim_info['cha']['ind'], keepdims=True))
        self.dim_info['cha']['len'] = 1
        self.dim_size[self.dim_info['cha']['ind']] = 1


    def _remove_oversamples(self):
        print('Remove oversampling...')
        cutoff = (self.dim_info['ro']['len'] - self.matrix_size['image']['x']) // 2 # // -> integer division
        self.img = self.img[:,cutoff:-cutoff,:,:,:,:,:]
        self.dim_info['ro']['len'] = self.img.shape[self.dim_info['ro']['ind']]
        self.dim_size[self.dim_info['ro']['ind']] = self.dim_info['ro']['len']
        

    def _reorder_slice(self):
        print('Reorder slice...')
        unsorted_order = np.zeros((self.dim_info['slc']['len']))
        for cslc in range(self.dim_info['slc']['len']):
            p1 = np.linalg.solve(self.transformation['mat44'], self.transformation['soda'][cslc,:,3])
            unsorted_order[cslc] = p1[2]
        ind_sorted = np.argsort(unsorted_order)
        self.img   = self.img[:,:,:,:,ind_sorted,:,:,:,:,:,:]
        self.transformation['soda'] = self.transformation['soda'][ind_sorted,:,:]

    # Tasks to be executed before coils combination
    def _custom_task(self):
        pass
    
    # Coordinate transformation
    def _extract_transformation(self):
        hdr = self.hdr
        transformation = {}
        transformation['soda'] = np.zeros((self.dim_info['slc']['len'], 4, 4))
        offcenter = np.zeros(3)
        for cslc in range(self.dim_info['slc']['len']):
            ind = np.where((hdr['idx']['slice'] == cslc) & self.flags['image_scan'])[0]
            if len(ind) == 0:
                print(f"\033[91mslice index not found! aborting...\033[0m")
                raise SystemExit('Goodbye')
            
            dcm = np.column_stack((hdr['phase_dir'][ind[0],:], 
                                   hdr['read_dir'] [ind[0],:], 
                                   hdr['slice_dir'][ind[0],:]))

            transformation['soda'][cslc,0:3,:] = np.column_stack((dcm, hdr['position'][ind[0],:]))
            transformation['soda'][cslc,:,:]   = np.row_stack((transformation['soda'][cslc,0:3,:], [0, 0, 0, 1]))           
            offcenter += hdr['position'][ind[0],:]

        offcenter /= self.dim_info['slc']['len']
        transformation['offcenter']     = offcenter 
        transformation['mat44']         = transformation['soda'][0,:,:] 
        transformation['mat44'][0:3,3]  = offcenter
        self.transformation = transformation

    # Squeezing image data
    def sqz(self):
        print('Squeezing...')
        for key in list(self.dim_info):
            if self.dim_info[key]['len'] == 1:                
                self.dim_info.pop(key, None)
        # refine dimensions index, assuimg sorted dictionary (Python > 3.7)
        l = list(self.dim_info.items())
        for i in range(len(self.dim_info)):
            self.dim_info[l[i][0]]['ind'] = i
        
        self.dim_size = [y for y in self.dim_size if y!=1]
        self.img = np.squeeze(self.img)

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
        T = self.transformation['mat44']
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
