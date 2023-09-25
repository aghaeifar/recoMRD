import os
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import montage
from scipy import ndimage


def create_brain_mask(volume: np.ndarray, erode_size=3):
    mask_size = [x for x in volume.shape if x > 1]
    if len(mask_size) != 3:
        print(f'Only 3D data is supported for masking. Input shape is {mask_size}')
        return

    dir_path = os.path.dirname(os.path.realpath(__file__))
    handle   = ctypes.CDLL(os.path.join(dir_path, "lib", "libbet2.so"))
    handle.runBET.argtypes = [np.ctypeslib.ndpointer(np.float32, ndim=volume.ndim, flags='F'),
                              np.ctypeslib.ndpointer(
                              np.float32, ndim=len(mask_size), flags='F'),
                              ctypes.c_int, ctypes.c_int, ctypes.c_int]

    mag = volume.copy(order='F')
    mask = np.zeros(mask_size, dtype=mag.dtype, order='F')
    handle.runBET(mag, mask, *mask_size)  # 3D input
    if erode_size > 1:
        es = erode_size
        mask = ndimage.binary_erosion(mask, structure=np.ones((es, es, es))).astype(volume.dtype)
        # binary_erosion changes order to C_CONTIGUOUS
        mask = np.asfortranarray(mask)

    return mask.reshape(volume.shape).copy(order='C')



def plot_mosaic(img:np.ndarray, cmap='turbo', clim=None, grid_shape=None, title=None, transpose=False):
    img = np.squeeze(np.abs(img)) 
    if img.ndim > 3:
        print(f'Error! plot_mosaic expects 3D data but it is {img.ndim}D')
        return
    img = np.moveaxis(img,-1,0)
    img = montage(img, fill=0, grid_shape=grid_shape)
    if transpose:
        img = img.T
    
    plt.figure()
    plt.imshow(img, cmap=cmap, clim=clim)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.colorbar()
    plt.tight_layout() 
    return img

