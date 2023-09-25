import os
import ctypes
import numpy as np
from recoMRD import recoMRD

#
# see: https://onlinelibrary.wiley.com/doi/10.1002/mrm.29459
#

OPERTATING_MODE  = 'alICEProgramPara[10]'
ABSOLUTE_MODE    = 'alICEProgramPara[12]'
RELATIVE_MODE    = 'alICEProgramPara[15]'
PULSE_INTEGRAL   = 'alICEProgramPara[17]'
PULSE_DURATION   = 'alICEProgramPara[14]'
NUM_TX_ABS       = 'alICEProgramPara[11]'
NUM_TX_REL       = 'alICEProgramPara[16]'

HAS_ABSOLUTE_MAP = 1<<0
HAS_RELATIVE_MAP = 1<<1
HAS_B0_MAP       = 1<<3
HAS_SINC_SAT     = 1<<4

PTX_MODE_CP     = 1
PTX_MODE_ONEON  = 2
PTX_MODE_ONEOFF = 3
PTX_MODE_ONEINV = 4


class recoMRD_B1TFL(recoMRD):
    nTx       = 0
    img_b1mag = np.empty([1])
    img_b1phs = np.empty([1])    
    img_mask  = np.empty([1])
    img_cp    = np.empty([1]) # unwrapped 
    params    = {OPERTATING_MODE:0, ABSOLUTE_MODE:0, RELATIVE_MODE:0, PULSE_INTEGRAL:0, PULSE_DURATION:0, NUM_TX_ABS:0, NUM_TX_REL:0}

    def __init__(self, filename=None, device='cpu'):
        super().__init__(filename, device)        
        self.parseHeader()
        self.runReco()    

    def parseHeader(self):
        for pl in self.xml_hdr.userParameters.userParameterLong:
            if pl.name in self.params:
                self.params[pl.name] = pl.value

        print(f'Operating Mode: {self.params[OPERTATING_MODE]}\n' +
              f'Absolute Mode: {self.params[ABSOLUTE_MODE]}\n' + 
              f'Relative Mode: {self.params[RELATIVE_MODE]}\n' +
              f'Num Tx (Abs, Rel) : {self.params[NUM_TX_ABS]}, {self.params[NUM_TX_REL]}\n' +
              f'Pulse Duration and Inegral: {self.params[PULSE_DURATION]}, {self.params[PULSE_INTEGRAL]}\n')
        
        if self.params[NUM_TX_ABS] != self.params[NUM_TX_REL]:
            print(f'\033[93mNumber of absolute and relative transmit channels do not match: {self.params[NUM_TX_ABS]} vs {self.params[NUM_TX_REL]}\033[0m')
            return
        if self.params[OPERTATING_MODE] & HAS_ABSOLUTE_MAP == 0 or self.params[OPERTATING_MODE] & HAS_RELATIVE_MAP == 0:
            print(f'\033[93mAbsolute and relative maps must present.\033[0m')
            return  
        nTx = self.params[NUM_TX_ABS] 
        

    def runReco(self):
        super().runReco()
        scale_factor_abs = self.getTxScaleFactor(self.params[ABSOLUTE_MODE])
        scale_factor_abs = self.getTxScaleFactor(self.params[RELATIVE_MODE])

    def getTxScaleFactor(self, mode):
        rfmode = {PTX_MODE_CP:{}, PTX_MODE_ONEON:{}, PTX_MODE_ONEOFF:{}, PTX_MODE_ONEINV:{}}
        rfmode[PTX_MODE_CP]     = {'diag':complex( 1.0, 0.0), 'offdiag':complex(1.0, 0.0)}
        rfmode[PTX_MODE_ONEON]  = {'diag':complex( 1.0, 0.0), 'offdiag':complex(0.0, 0.0)}
        rfmode[PTX_MODE_ONEOFF] = {'diag':complex( 0.0, 0.0), 'offdiag':complex(1.0, 0.0)}
        rfmode[PTX_MODE_ONEINV] = {'diag':complex(-1.0, 0.0), 'offdiag':complex(1.0, 0.0)}

        if mode not in rfmode:
            print(f'\033[93mUnknown RF mode: {mode}\033[0m')
            return

        scale_factor = np.full((self.nTx, self.nTx), rfmode[mode]['offdiag'] )
        np.fill_diagonal(scale_factor, rfmode[mode]['diag'])

        return scale_factor