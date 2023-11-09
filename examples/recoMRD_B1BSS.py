import torch
from recoMRD import recoMRD

#
# This class implementes reconstruction of the Bloch-Siegert shift B1 mapping technique as described in https://onlinelibrary.wiley.com/doi/10.1002/mrm.22357
#

GAMMA_RAD       = 267.52218744e6  # rad/(sT)
GAMMA_HZ        = 42.577478518e6  # Hz/T

BSSTransmitVoltage              = 'alICEProgramPara[12]' # Volt
BSSAmplitudeIntegralVs          = 'alICEProgramPara[13]' # Volt*micro-second
BSSAmplitudeIntegralNormalized  = 'alICEProgramPara[14]' # 
BSSPowerIntegralNormalized      = 'alICEProgramPara[15]' # us
BSSDuration                     = 'alICEProgramPara[16]' # Hz
BSSOffResonance                 = 'alICEProgramPara[17]' # Hz
# FermiAmplitudeIntegral          = 'alICEProgramPara[16]' # Hz
# FermiPowerIntegral              = 'alICEProgramPara[17]' # Hz
RefVoltage                      = 'asNucleusInfo[0].flReferenceAmplitude' # Volt


class recoMRD_B1BSS(recoMRD):
    img_b1nTv = torch.empty([1]) # B1 map, nT/V
    img_mag = torch.empty([1]) # magnitude image
    KBS = torch.empty([1]) # B1 mapping constant
    seqTxScaleFactor = torch.empty([1])
    params   = {BSSAmplitudeIntegralVs:0, BSSAmplitudeIntegralNormalized:0, BSSPowerIntegralNormalized:0, BSSDuration:0, BSSOffResonance:0, BSSTransmitVoltage:0, RefVoltage:0}

    def __init__(self, filename=None, device='cpu'):
        super().__init__(filename, device)        
        self.parseHeader()
        self.runReco()    

    def parseHeader(self):
        for pl in self.xml_hdr.userParameters.userParameterLong + self.xml_hdr.userParameters.userParameterDouble:
            if pl.name in self.params:
                self.params[pl.name] = pl.value

        self.params[BSSTransmitVoltage]             /= 1e3  # remove scaling factor of 1e3
        self.params[BSSAmplitudeIntegralVs]         /= 1e6  # convert to Volt * second         
        self.params[BSSAmplitudeIntegralNormalized] /= 1e3  # remove scaling factor of 1e3
        self.params[BSSPowerIntegralNormalized]     /= 1e3  # remove scaling factor of 1e3

        print(f'BSS Pulse Integral [V*Sec]:     {self.params[BSSAmplitudeIntegralVs]}\n' +
              f'BSS Pulse Integral Normalized:  {self.params[BSSAmplitudeIntegralNormalized]}\n' +
              f'BSS Pulse Power Normalized:     {self.params[BSSPowerIntegralNormalized]}\n' +
              f'BSS Pulse Duration [us]:        {self.params[BSSDuration]}\n' +
              f'BSS Pulse Off Resonance [Hz]:   {self.params[BSSOffResonance]}\n' + 
              f'BSS Pulse Voltage [V]:          {self.params[BSSTransmitVoltage]}\n' + 
              f'Ref Pulse Voltage [V]:          {self.params[RefVoltage]}')

    def runReco(self, method_sensitivity='caldir', remove_os=True):
        set_ind = self.dim_info['set']['ind']
        idx     = torch.tensor([0, 1])
        kspace  = self.kspace['image_scan'] if not remove_os else self.remove_oversampling(self.kspace['image_scan'], is_kspace=True)
        self.kspace['image_scan'] = None # free memory
        img     = self.kspace_to_image(kspace)
        self.img_mag    = torch.sqrt(torch.sum(torch.abs(img)**2, self.dim_info['cha']['ind'], keepdims=True)) 

        b1_temp = self.img_mag.index_select(set_ind, idx[0]) * img.index_select(set_ind, idx[0]) * \
                  self.img_mag.index_select(set_ind, idx[1]) * img.index_select(set_ind, idx[1]).conj()  
        
        b1_temp = -torch.angle(b1_temp.sum(dim=self.dim_info['cha']['ind'], keepdims=True)) # some one tell me why we need to add a minus sign here
        b1_temp /= 2 # divide by 2 since we combined phases from BSS+ and BSS- pulses

        dT       = 1e-6
        Wrf      = 2 * torch.pi * self.params[BSSOffResonance]
        self.KBS = GAMMA_RAD*GAMMA_RAD*self.params[BSSPowerIntegralNormalized]*dT / (2 * Wrf)
        print(f'KBS = {self.KBS} rad/T²')
        # compute B1 map in percentage of the nominal flip angle
        self.img_b1nTv = 1e9 * torch.sqrt(b1_temp / self.KBS) / self.params[BSSTransmitVoltage] # nT/V
        self.img_b1nTv [torch.isnan(self.img_b1nTv )] = 0

        