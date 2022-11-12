# %%
import recoMRD_B0 
import importlib
import numpy as np 


importlib.reload(recoMRD_B0)

from recoMRD_B0 import recoMRD_B0

filename = '/DATA/aaghaeifar/rawdata/silent_shimming/data/mrd/meas_MID00575_FID29449_aa_B0Phantom.mrd'
filename = '/home/ali/Nextcloud/Temp/meas_MID00575_FID29449_aa_B0Phantom.mrd'
mrd = recoMRD_B0(filename)


# %%
