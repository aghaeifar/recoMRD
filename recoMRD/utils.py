
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import montage

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


def plot3D(img:np.ndarray, cmap='turbo', clim=None, pos = None):
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
    img = np.squeeze(img) 
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

