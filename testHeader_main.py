#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:13:09 2017

@author: valsecchi
"""
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
import astropy
from os import listdir,rename, makedirs
from os.path import isfile, join, exists, isdir,splitext
from astropy.io import fits 
import pyfits
import fitsio
import h5py

# Function to read the data and return it as 3D arrays
def read_data(path_im,path_ob,path_dc):
    """
    read()
    """
    if path_dc:
        # Load DCs and average them
        filenames_dc = os.listdir(path_dc)
        filenames_dc.sort()
        f_name = path_dc + '/' + i
        
        im_a1 = []
        for i in f_name:

            if i.lower().endswith('.fits'):
                print('fits')
                try:
                    im_a1.append(fits.open(i)[0].data)
                except OSError:
                    im_a1.append(fitsio.read(i))
            elif i.lower().endswith(('.tiff','.tif')) :
                print('tif or tiff')
                im_a1.append(Image.open(i))
            elif i.lower().endswith(('.hdf','.h4','.hdf4','.he2','h5','.hdf5','.he5')): 
                print('hdf')
                im_a1.append(h5py.File(i,'r'))
            else:
                print(splitext(i)[-1],'file extension not yet implemented....Do it your own way!')
        
        
        
        f_name = path_dc + '/' + filenames_dc[0]    # Load first DC in folder
        print(f_name)
        im = Image.open(f_name)
        im_a1 = np.array(im)
        
        for i in filenames_dc:              # Iterate through filenames in DC folder
            f_name = path_dc + '/' + i
            im = Image.open(f_name)
            im_a = np.array(im)
            im_a1 = (im_a1 + im_a)/2        # Add to previous DC and divide by 2 - average 
    else:
        print('...no dark current...')
        
    # Load Projectionss
    filenames_im = os.listdir(path_im)  # Create list of filenames in projection folder
    filenames_im.sort()                 # Sort the lsit (just in case)
    
    stack_im = list()                   # Generate empty list for projections 
    
    for i in filenames_im:              # Iterate through filenames in projection folder
        f_name = path_im + '/' + i
        im = Image.open(f_name)         # Open image
        if path_dc:
            im_a = np.array(im-im_a1)             # Convert image to array
        else:
            im_a = np.array(im)
        stack_im.append(im_a)           # Append array to list
    
    stack_im_ar = np.asarray(stack_im)  # Convert list to numpy array
    
    # Load Open Beams
    filenames_ob = os.listdir(path_ob)
    filenames_ob.sort()
    
    stack_ob = list()
    
    for i in filenames_ob:
        f_name = path_ob + '/' + i
        im = Image.open(f_name)
        if path_dc:
            im_a = np.array(im-im_a1)             # Convert image to array
        else:
            im_a = np.array(im)
        stack_ob.append(im_a)
    
    stack_ob_ar = np.asarray(stack_ob)            
    if  stack_im_ar.shape != stack_ob_ar.shape:
        print('!!!WARNING!!! SHAPE OF PROJECTIONS AND OPEN BEAMS ARE NOT THE SAME')
    return(stack_im_ar,stack_ob_ar)
    
path_ob = 'data/data_OB'
path_im = 'data/data_smp'
path_dc = 'data/DCs'

#(im,ob) = read_data(path_im,path_ob,path_dc)
im_a1 = []
i = '/Users/valsecchi/Documents/python2_7/nGI_reconstrucrion/nGITrello/testnGI/boa2014n004376.hdf'
if i.lower().endswith('.fits'):
    print('fits')
    try:
        im_a1.append(fits.open(i)[0].data)
    except OSError:
        im_a1.append(fitsio.read(i))
elif i.lower().endswith(('.tiff','.tif')) :
    print('tif or tiff')
    im_a1.append(Image.open(i))
elif i.lower().endswith(('.hdf','.h4','.hdf4','.he2','h5','.hdf5','.he5')): 
    print('hdf')
    im_a1.append(h5py.File(i,'r+'))
else:
    print(splitext(i)[-1],'file extension not yet implemented....Do it your own way!')