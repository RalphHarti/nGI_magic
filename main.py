#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:42:58 2017

@author: harti and valsecchi
"""
from functions import read_data,cropped,createIm,normalization,saveIm,binning,oscillation,createIm_fft,win_filt_z
from pixelwiseDPC import pixelWiseDPC,pixelWisePC

path_ob = 'data/data_OB_noise_snr5'
path_im = 'data/data_smp_noise_snr5'
path_dc = ''#'data/DCs'

norm_param = [3,5,20,40]
crop_param = [10,15,80,60]
bin_fac = None                 # no binning either 1 or None, 2x2 binning: bin_fac = 2

(im,ob) = read_data(path_im,path_ob,path_dc)
#im,ob=normalization(im,ob,*norm_param)
#im,ob = cropped(im,ob,*crop_param)
#im, ob = binning(im,ob,bin_fac)
im, ob = win_filt_z(im,ob)
ti, dpci, dfi, vis_map = createIm(im,ob)
#ti, dpci, dfi, vis_map = createIm_fft(im,ob)
oscillation(im,ob,5,5,2,2)
saveIm(ti, dpci, dfi, vis_map,name='snr_5',folder='noisy_snr5_win_z3',overWrite=True)



