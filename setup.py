# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 07:49:16 2017

@author: Ali Aghaeifar
"""


from setuptools import setup


setup(name='recoMRD', # this will be name of package in packages list : pip list 
      version='1.0',
      description='Reconstruction utilities for ismrmrd (MRD) files',
      keywords='ismrmrd,reconstruction,mri,nifti,ismrmrd2nifti',
      author='Ali Aghaeifar',
      author_email='ali.aghaeifar [at] tuebingen.mpg [dot] de',
      license='MIT License',
      packages=['recoMRD'],
      install_requires = ['tqdm','numpy','nibabel','ismrmrd','h5py','scipy', 'scikit-image','matplotlib']
     )
