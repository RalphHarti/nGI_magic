#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 08:07:33 2017
"""
from functions import read_data,cropped,createIm,normalization,saveIm,binning,oscillation,createIm_fft,med_filt_z
from pixelwiseDPC import pixelWiseDPC,pixelWisePC
import numpy as np
path_ob = '/Users/valsecchi/Dropbox/B0A/OBspin_OFF'
path_im = '/Users/valsecchi/Dropbox/B0A/Sample_spin_OFF'
path_dc = '/Users/valsecchi/Documents/PSI/BOA_FOLDER/BOA_DICEMBRE/Data/01_firstSlot/LASER_IN/04_brp_dark'#'data/DCs'


path_dc = ''#'data/DCs'
bin_fac = None                 # no binning either 1 or None, 2x2 binning: bin_fac = 2
norm_param = [750,320,70,300]
crop_param = [10,15,80,60]
oscillationParam = [300,300,50,50]

im,ob = read_data(path_im,path_ob,path_dc)
#im,ob=normalization(im,ob,*norm_param)
#crop_param = [170,170,600,600]
#print(np.shape(im))
oscillation(im,ob,*oscillationParam,repeatedPeriod=False)
im,ob=normalization(im,ob,*norm_param)
#im,ob=med_filt_z(im,ob,3)
#%%
#oscillation(im,ob,*oscillationParam,repeatedPeriod=False)
oscillation(im,ob,*oscillationParam,repeatedPeriod=True)


#im,ob = cropped(im,ob,*crop_param)
#print(np.shape(im))

#norm_param = [0,0,512,512]
#im,ob=normalization(im,ob,*norm_param)
#print(np.shape(im))

#im, ob = binning(im,ob,bin_fac)
#im, ob = med_filt_z(im,ob,3)
#ti, dpci, dfi, vis_map = createIm(im,ob)
#ti, dpci, dfi, vis_map = createIm_fft(im,ob)
#saveIm(ti, dpci, dfi, vis_map,name='name',folder='med_filt',overWrite=True)"""

#%%
import numpy as np
import scipy.optimize as opt
import pylab as plt


def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple                                                        
    xo = float(xo)                                                              
    yo = float(yo)                                                              
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)   
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)    
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)   
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)         
                        + c*((y-yo)**2)))    
    return g.ravel()


def example_twoD_Gaussian():    
    # Create x and y indices
    x = np.linspace(0, 200, 201)
    y = np.linspace(0, 200, 201)
    x, y = np.meshgrid(x, y)
    
    #create data
    data = twoD_Gaussian((x, y), 3, 100, 100, 20, 100, 0, 10)
    
    # plot twoD_Gaussian data generated above
    plt.figure()
    plt.imshow(data.reshape(201, 201))
    plt.colorbar()
    
    # add some noise to the data and try to fit the data generated beforehand
    initial_guess = (3,100,100,20,40,0,10)
    
    data_noisy = data + 0.2*np.random.normal(size=data.shape)
    
    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data_noisy, p0=initial_guess)
    
    data_fitted = twoD_Gaussian((x, y), *popt)
    print(popt)
    
    fig, ax = plt.subplots(1, 1)
    ax.hold(True)
    ax.imshow(data_noisy.reshape(201, 201), cmap=plt.cm.jet, origin='bottom',
        extent=(x.min(), x.max(), y.min(), y.max()))
    ax.contour(x, y, data_fitted.reshape(201, 201), 8, colors='w')
    plt.show()
    plt.close()
    return
example_twoD_Gaussian()