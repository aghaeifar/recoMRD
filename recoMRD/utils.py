import os
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import montage
from scipy import ndimage

# ifftnd and fftnd are taken from twixtools package
def ifftnd(kspace:np.ndarray, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img  = fftshift(ifftn(ifftshift(kspace, axes=axes), axes=axes), axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))
    return img


def fftnd(img:np.ndarray, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, fftn
    if axes is None:
        axes = range(img.ndim)
    kspace  = fftshift(fftn(ifftshift(img, axes=axes), axes=axes), axes=axes)
    kspace /= np.sqrt(np.prod(np.take(kspace.shape, axes)))
    return kspace


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



def plot3D(img:np.ndarray, cmap='turbo', clim=None, pos = None):
    img = img.squeeze()
    if pos is None:
        pos = [x//2 for x in img.shape[0:3]]
        
    plt.figure()
    plt.subplot(131)
    plt.imshow(img[:,:,pos[2]], cmap=cmap, origin='lower',clim=clim)
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(img[:,pos[1],:], cmap=cmap, origin='lower',clim=clim)
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(img[pos[0],:,:], cmap=cmap, origin='lower',clim=clim)
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout() 


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

