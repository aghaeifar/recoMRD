import torch
from recoMRD import recoMRD

#
# This class implementes reconstruction of the B1 mapping technique as described in https://onlinelibrary.wiley.com/doi/10.1002/mrm.29459
#

GAMMA_RAD       = 267.52218744e6  # rad/(sT)
GAMMA_HZ        = 42.577478518e6  # Hz/T

OPERTATING_MODE = 'alICEProgramPara[10]'
NUM_TX_ABS      = 'alICEProgramPara[11]'
ABSOLUTE_MODE   = 'alICEProgramPara[12]'
PULSE_DURATION  = 'alICEProgramPara[14]'
RELATIVE_MODE   = 'alICEProgramPara[15]'
NUM_TX_REL      = 'alICEProgramPara[16]'
PULSE_INTEGRAL  = 'alICEProgramPara[17]'
B0_DTE          = 'alICEProgramPara[18]'

HAS_ABSOLUTE_MAP = 1<<0
HAS_RELATIVE_MAP = 1<<1
HAS_B0_MAP       = 1<<3
HAS_SINC_SAT     = 1<<4

PTX_MODE_CP     = 1
PTX_MODE_ONEON  = 2
PTX_MODE_ONEOFF = 3
PTX_MODE_ONEINV = 4

mapping_mode = {PTX_MODE_CP       :{'diag':complex( 1.0, 0.0), 'offdiag':complex(1.0, 0.0)}, \
                PTX_MODE_ONEON    :{'diag':complex( 1.0, 0.0), 'offdiag':complex(0.0, 0.0)}, \
                PTX_MODE_ONEOFF   :{'diag':complex( 0.0, 0.0), 'offdiag':complex(1.0, 0.0)}, \
                PTX_MODE_ONEINV   :{'diag':complex(-1.0, 0.0), 'offdiag':complex(1.0, 0.0)}}

class recoMRD_B1TFL(recoMRD):
    nTx      = 0
    img_cp   = torch.empty([1]) # CP map
    img_fa   = torch.empty([1]) # FA map
    img_b    = torch.empty([1]) # B1 map, nT per Volt
    img_mask = torch.empty([1])
    params   = {OPERTATING_MODE:0, NUM_TX_ABS:0, ABSOLUTE_MODE:0, PULSE_DURATION:0, RELATIVE_MODE:0, NUM_TX_REL:0, PULSE_INTEGRAL:0, B0_DTE:0}
    seqTxScaleFactor = torch.empty([1])

    def __init__(self, filename=None, device='cpu'):
        super().__init__(filename, device)        
        self.parseHeader()
        self.runReco()    

    def parseHeader(self):
        for pl in self.xml_hdr.userParameters.userParameterLong:
            if pl.name in self.params:
                self.params[pl.name] = pl.value
        
        if self.params[NUM_TX_ABS] != self.params[NUM_TX_REL]:
            print(f'\033[93mNumber of absolute and relative transmit channels do not match: {self.params[NUM_TX_ABS]} vs {self.params[NUM_TX_REL]}\033[0m')
            return
        if self.params[OPERTATING_MODE] & HAS_ABSOLUTE_MAP == 0 or self.params[OPERTATING_MODE] & HAS_RELATIVE_MAP == 0:
            print(f'\033[93mAbsolute and relative maps must present.\033[0m')
            return  
        
        self.params[PULSE_INTEGRAL] /= 1e6 # convert to Volt * second
        self.params[PULSE_DURATION] /= 1e6 # convert to seconds
        self.params[B0_DTE] /= 1e6 # convert to seconds
        self.nTx = self.params[NUM_TX_ABS]

        self.seqTxScaleFactor = torch.complex(torch.zeros(self.nTx), torch.zeros(self.nTx)) # filling with zero, just in case some channels are missing in the header
        for i in range(self.nTx):
            for pl in self.xml_hdr.userParameters.userParameterDouble:
                if pl.name == f'aTxScaleFactor[{i}].dRe':
                    self.seqTxScaleFactor[i].real = pl.value
                elif pl.name == f'aTxScaleFactor[{i}].dIm':
                    self.seqTxScaleFactor[i].imag = pl.value

        print(f'Operating Mode: {self.params[OPERTATING_MODE]:05b}\n' +
              f'Absolute Mode : {self.params[ABSOLUTE_MODE]}\n' + 
              f'Relative Mode : {self.params[RELATIVE_MODE]}\n' +
              f'Num. Tx (Abs, Rel) : {self.params[NUM_TX_ABS]}, {self.params[NUM_TX_REL]}\n' +
              f'Pulse Duration(sec.) and Integral(volt*sec.): {self.params[PULSE_DURATION]}, {self.params[PULSE_INTEGRAL]}\n')
        print(f'Sequence TxScaleFactor = {self.seqTxScaleFactor}')

    def runReco(self, method_sensitivity='caldir', remove_os=True):
        super().runReco(method_sensitivity=method_sensitivity, remove_os=remove_os)
        rep_ind = self.dim_info['rep']['ind']
        sat     = self.img.index_select(rep_ind, torch.arange(0, self.nTx))
        ref     = self.img.index_select(rep_ind, torch.tensor([self.nTx]))
        rel_map = self.img.index_select(rep_ind, torch.arange(self.nTx+1, 2*self.nTx+1))
        fa_map  = torch.arccos( sat / ref).abs().rad2deg()        
        fa_map  = fa_map * rel_map / rel_map.abs() 
        fa_map  = torch.reshape(fa_map.moveaxis(rep_ind, 0), (self.nTx, -1))  # reshape to 2D matrix, [nTx, nVoxels]

        # create the matrix of scales factors
        scale_factor = torch.full((self.nTx, self.nTx), mapping_mode[self.params[ABSOLUTE_MODE]]['offdiag'])
        scale_factor.fill_diagonal_(mapping_mode[self.params[ABSOLUTE_MODE]]['diag'])
        scale_factor = scale_factor * self.seqTxScaleFactor
        
        self.img_fa = torch.linalg.solve(scale_factor, fa_map).moveaxis(0,-1).reshape(sat.shape) # solve the linear system of equations
        self.img_b1 = self.img_fa / (GAMMA_HZ * 360.0 * self.params[PULSE_INTEGRAL]) * 1e9 # convert to nT/Volt 
        self.img_cp = torch.sum(self.img_fa * self.seqTxScaleFactor.view((-1,)+(1,)*(self.img_fa.dim()-rep_ind-1)), dim=rep_ind, keepdim=True) # sum over all transmit channels
        