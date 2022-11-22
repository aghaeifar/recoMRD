# %%
from recoMRD import recoMRD_B0

import numpy as np 
import vini
import os


filename = './../example_data/meas_MID00575_FID29449_aa_B0Phantom.mrd'
mrd = recoMRD_B0(filename)

mrd.unwrap_b0()
mrd.create_mask()
# vini.show(mrd.img_b0_uw.squeeze())
#vini.show(mrd.img_mask.squeeze())
#mrd.make_nifti(mrd.img_b0_uw, 'mrd.nii')


# %%
# %%
