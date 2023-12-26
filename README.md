# recoMRD
**This project is deprecated. The ongoing developments have been transferred to the [recotwix](https://github.com/aghaeifar/recotwix) project.**

Scripts to read and reconstruct ismrmrd (mrd) format

There are a few jupyter notebooks in the example folder to demo how to use recoMRD. Testing MRD files can be downloaded from [here](https://keeper.mpdl.mpg.de/d/77583249089d470183f2/).

## Install

## Dependecies
This program uses following (optional) tools. Compiled binaries are provided in the [lib](./recoMRD/lib/) folder.
- Standalone brain extraction tool (BET2) for brain masking [+](https://github.com/aghaeifar/bet2)
- SRNCP 3D phase unwrapping [+](https://github.com/ivoreus/phase_unwrap) 

[BART](https://github.com/mrirecon/bart) toolbox is used for some of coil combination methods (optional).


## Class overview
### Inheritance tree:

    |--readMRD
        |-- recoMRD
            |-- recoMRD_B0
            |-- recoMRD_B1TFL
            |-- recoMRD_custom

### Important class members
#### readMRD
*kspace* : n dimension data storing different scan types, e.g., kspace, navigator, feedback, auto calibration scan, etc.
*dim_size* : length of fully sampled acquired data in each dimension (no accelration, no partial fourier, no asymmetric echo, no ...?)\
*matrix_size* : size of fully sampled 3D volume in k-space and image space\
*fov* : FoV in k-space and image space\
*enc_minmax_ind* : index of the first and the last sample/line for all encoding directions

in progress ...


