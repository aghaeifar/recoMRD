import os
import sys
import h5py
import ismrmrd
import numpy as np
from tqdm import tqdm
from ismrmrd.xsd import CreateFromDocument as ismrmrd_xml_parser

class readMRD(object): 
    flags    = None  
    xml      = None
    hdr      = None
    fov      = None
    data     = None
    xml_hdr  = None
    dim_info = None
    dim_size = None    
    is3D     = False
    kspace   = {}         
    matrix_size         = None
    enc_minmax_ind      = {}
    readmrd_tags        = {}
    ismrmrd_tags        = {} 
    transformation      = None
    isParallelImaging   = False
    isRefScanSeparate   = False
    isPartialFourierRO  = False
    isPartialFourierPE1 = None
    isPartialFourierPE2 = None
    acceleration_factor = [1,1]

    
    def __init__(self, filename=None):   
        if sys.version_info < (3,7,0):
            raise SystemExit('Python version >= 3.7.x is required. Aborting...')

        self.filename  = filename                
        self.readmrd_tags = ['cha', 'ro', 'pe1', 'pe2', 'slc', 'eco', 'rep', 'set', 'seg', 'ave', 'phs'] # order matters here
        self.ismrmrd_tags = ['', '', 'kspace_encode_step_1', 'kspace_encode_step_2', 'slice', 'contrast', 'repetition', 'set', 'segment', 'average', 'phase']
        self.kspace = {}
        self.dim_info = {}
        for i in range(len(self.readmrd_tags)):
            self.dim_info[self.readmrd_tags[i]] = {}
            self.dim_info[self.readmrd_tags[i]]['len'] = 1
            self.dim_info[self.readmrd_tags[i]]['ind'] = i

        self._import_mrd()
        self._extract_flags()
        self._create_kspace()
        self._extract_transformation()
        if self.is3D == False:
            self._reorder_slice()


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
        flags['acs']        = np.bitwise_and( self.hdr['flags'] , 1 << ismrmrd.ACQ_IS_PARALLEL_CALIBRATION-1).astype(bool)
        flags['noise']      = np.bitwise_and( self.hdr['flags'] , 1 << ismrmrd.ACQ_IS_NOISE_MEASUREMENT-1   ).astype(bool)
        flags['feedback']   = np.bitwise_and( self.hdr['flags'] , 1 << ismrmrd.ACQ_IS_RTFEEDBACK_DATA-1     ).astype(bool)
        flags['image_scan'] = np.logical_not( flags['acs'] | flags['nav'] | flags['pc'] | flags['noise'] | flags['feedback'])
        # update acs, to include ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING
        flags['acs']       |= np.bitwise_and( self.hdr['flags'] , 1 << ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING-1).astype(bool)
        self.flags = flags
        print('Number of reference scans: {}'.format(np.count_nonzero(flags['acs'])))
        print('Number of image     scans: {}'.format(np.count_nonzero(flags['image_scan'])))

        enc = self.xml_hdr.encoding[0]
        # Matrix size
        matrix_size                = {'kspace':{}, 'image':{}}
        matrix_size['image']['x']  = enc.reconSpace.matrixSize.x
        matrix_size['image']['y']  = enc.reconSpace.matrixSize.y
        matrix_size['image']['z']  = enc.reconSpace.matrixSize.z
        matrix_size['kspace']['x'] = enc.encodedSpace.matrixSize.x
        matrix_size['kspace']['y'] = enc.encodedSpace.matrixSize.y
        matrix_size['kspace']['z'] = enc.encodedSpace.matrixSize.z
        self.matrix_size = matrix_size
        print(f'k-space size in protocol: {matrix_size["kspace"]["x"]} x {matrix_size["kspace"]["y"]} x {matrix_size["kspace"]["z"]}')
        print(f'image   size in protocol: {matrix_size["image"]["x"]} x {matrix_size["image"]["y"]} x {matrix_size["image"]["z"]}')

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
        self.dim_info['ro']['len']  = matrix_size['kspace']['x'] # self.hdr['number_of_samples'][0]

        if enc.encodingLimits.kspace_encoding_step_1 != None:
            self.dim_info['pe1']['len'] = matrix_size['kspace']['y'] # enc.encodingLimits.kspace_encoding_step_1.maximum + 1

        if enc.encodingLimits.kspace_encoding_step_2 != None:
            self.dim_info['pe2']['len'] = matrix_size['kspace']['z'] # enc.encodingLimits.kspace_encoding_step_2.maximum + 1
        
        if enc.encodingLimits.repetition != None:
            self.dim_info['rep']['len'] = enc.encodingLimits.repetition.maximum + 1

        if enc.encodingLimits.contrast != None:
            self.dim_info['eco']['len'] = enc.encodingLimits.contrast.maximum + 1

        if enc.encodingLimits.set != None:
            self.dim_info['seg']['len'] = enc.encodingLimits.segment.maximum + 1

        if enc.encodingLimits.average != None:
            self.dim_info['ave']['len'] = enc.encodingLimits.average.maximum + 1

        if enc.encodingLimits.slice != None:
            self.dim_info['slc']['len'] = enc.encodingLimits.slice.maximum + 1

        if enc.encodingLimits.phase != None:
            self.dim_info['phs']['len'] = enc.encodingLimits.phase.maximum + 1

        if enc.encodingLimits.set != None:
            self.dim_info['set']['len'] = enc.encodingLimits.set.maximum + 1

        self.dim_size = [1] * len(self.dim_info)
        for i in self.dim_info.keys():
            self.dim_size[self.dim_info[i]['ind']] = self.dim_info[i]['len']

        # index of minimum and maximum encoding
        image_scan_ind = list(*np.where(self.flags['image_scan']))
        first_sample_ind = matrix_size['kspace']['x']//2 - self.hdr['center_sample'][image_scan_ind[0]]
        self.enc_minmax_ind[self.dim_info['ro']['ind']]  = [first_sample_ind, first_sample_ind+self.hdr['number_of_samples'][image_scan_ind[0]]-1]
        self.enc_minmax_ind[self.dim_info['pe1']['ind']] = [enc.encodingLimits.kspace_encoding_step_1.minimum, enc.encodingLimits.kspace_encoding_step_1.maximum]
        self.enc_minmax_ind[self.dim_info['pe2']['ind']] = [enc.encodingLimits.kspace_encoding_step_2.minimum, enc.encodingLimits.kspace_encoding_step_2.maximum]

        if (self.dim_info['pe1']['len'] != enc.encodingLimits.kspace_encoding_step_1.maximum + 1 or 
            self.dim_info['pe2']['len'] != enc.encodingLimits.kspace_encoding_step_2.maximum + 1  ):
            print(f'\033[93mk-space encoding size ({self.dim_info["pe1"]["len"]} x {self.dim_info["pe2"]["len"]}) ' 
                  f'differs from max encoding step ({enc.encodingLimits.kspace_encoding_step_1.maximum + 1} x {enc.encodingLimits.kspace_encoding_step_2.maximum + 1})\033[0m')
            print('\033[93mThis can be due to parallel imaging, partial Fourier, etc.\033[0m')


        self.acceleration_factor = [enc.parallelImaging.accelerationFactor.kspace_encoding_step_1, enc.parallelImaging.accelerationFactor.kspace_encoding_step_2]
        if self.acceleration_factor[0] > 1 or self.acceleration_factor[1] > 1 :
            print(f'Acceleration factor: {self.acceleration_factor[0]} x {self.acceleration_factor[1]}')
            if np.bitwise_and( self.hdr['flags'] , 1 << ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING-1).astype(bool).any():
                self.isRefScanSeparate = False
                print('Reference scan type: integrated')
            else:
                self.isRefScanSeparate = True
                print('Reference scan type: separate')

        self.is3D = self.dim_info['pe2']['len'] > 1


    def _create_kspace(self):
        existing_scans = [scan for scan in self.flags if self.flags[scan].any()]
        print(f'Existing scans: {", ".join(existing_scans)}.')
        print(f'Fully sampled array size={self.dim_size}')
        for scan_type in existing_scans: 
            ind_scan = list(*np.where(self.flags[scan_type]))  # index of where scan_type is available          
            dim_size_local = self.dim_size[:] # copy a list to keep original dim_size unchanged
            dim_size_local[self.dim_info['ro']['ind']] = self.hdr['number_of_samples'][ind_scan[0]] # update readout size from first available scan in current scan_type --> for asymmetry readout or missed samples
            for tag in range(self.dim_info['ro']['ind']+1, len(self.readmrd_tags)): # update length of other dimensions --> for AccelFactor, partial Fourier, etc.
                dim_size_local[self.dim_info[self.readmrd_tags[tag]]['ind']] = len(np.unique(self.hdr['idx'][self.ismrmrd_tags[tag]][ind_scan]))

            ind_pe1_min = 0
            ind_pe2_min = 0
            ind_ro_zeropad = 0
            if scan_type != 'image_scan':
                self.kspace[scan_type] = np.zeros(dim_size_local, dtype=np.complex64)   
                if scan_type == 'acs': # standardize separate and integrated reference scan to the same size
                    ind_pe1_min = np.min(self.hdr['idx']['kspace_encode_step_1'][ind_scan])
                    ind_pe2_min = np.min(self.hdr['idx']['kspace_encode_step_2'][ind_scan])    
            else: # == 'image_scan'
                self.kspace[scan_type] = np.zeros(self.dim_size, dtype=np.complex64)  # I used dim_sz rather than dim_size_local to reset unsampled samples for possible asymmetric echo
                
                ro_diff = self.dim_size[self.dim_info['ro']['ind']] - dim_size_local[self.dim_info['ro']['ind']]
                if ro_diff > 4 : # this case is Asymmetric echo! zero padding in one side
                    self.isPartialFourierRO = True
                    ind_ro_zeropad = self.enc_minmax_ind[self.dim_info['ro']['ind']][0]
                    print(f'\033[93mHint! Asymmetric echo. RO zero pad index = {ind_ro_zeropad}\033[0m')
                elif ro_diff > 0 : # this case is not asymmetric echo! zero paddding in both sides 
                    ind_ro_zeropad = ro_diff//2
                    print(f'RO zero pad index = {ind_ro_zeropad}')

            for ind in tqdm(list(*np.where(self.flags[scan_type])), desc=f'Filling {scan_type:<10}, size={dim_size_local}'):
                data_tr = self.data[ind][0::2] + 1j*self.data[ind][1::2]
                data_tr = data_tr.reshape(dim_size_local[self.dim_info['cha']['ind']], dim_size_local[self.dim_info['ro']['ind']])
                self.kspace[scan_type][:, ind_ro_zeropad:dim_size_local[self.dim_info['ro']['ind']]+ind_ro_zeropad,  # dim_size_local[dim_inf['ro']['ind']] is needed for asymmetric echo
                                       self.hdr['idx'][self.ismrmrd_tags[2]][ind]-ind_pe1_min,
                                       self.hdr['idx'][self.ismrmrd_tags[3]][ind]-ind_pe2_min,
                                       self.hdr['idx'][self.ismrmrd_tags[4]][ind],
                                       self.hdr['idx'][self.ismrmrd_tags[5]][ind],
                                       self.hdr['idx'][self.ismrmrd_tags[6]][ind],
                                       self.hdr['idx'][self.ismrmrd_tags[7]][ind],
                                       self.hdr['idx'][self.ismrmrd_tags[8]][ind],
                                       self.hdr['idx'][self.ismrmrd_tags[9]][ind],
                                       self.hdr['idx'][self.ismrmrd_tags[10]][ind]] = data_tr


    def _reorder_slice(self):
        print('Reorder slice...', end=' ')
        unsorted_order = np.zeros((self.dim_info['slc']['len']))
        for cslc in range(self.dim_info['slc']['len']):
            p1 = np.linalg.solve(self.transformation['mat44'], self.transformation['soda'][cslc,:,3])
            unsorted_order[cslc] = p1[2]
        ind_sorted = np.argsort(unsorted_order)
        for scan in self.kspace:
            if scan == 'image_scan' or scan == 'acs':
                self.kspace[scan] = self.kspace[scan][:,:,:,:,ind_sorted,...]

        self.transformation['soda'] = self.transformation['soda'][ind_sorted,...]
        print('Done.')

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

