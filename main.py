#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:42:58 2017

@author: harti and valsecchi
"""
import matplotlib.pyplot as plt
from functions import read_data,cropped,createIm,normalization,saveIm,binning,oscillation

path_ob = 'data/data_OB'
path_im = 'data/data_smp'
path_dc = ''#'data/DCs'

norm_param = [3,5,20,40]
crop_param = [10,15,500,500]
bin_fac = None                 # no binning either 1 or None, 2x2 binning: bin_fac = 2

(im,ob) = read_data(path_im,path_ob,path_dc)
im,ob=normalization(im,ob,*norm_param)
#im,ob = cropped(im,ob,*crop_param)
#im, ob = binning(im,ob,bin_fac)
#ti, dpci, dfi, vis_map = createIm(im,ob)
#saveIm(ti, dpci, dfi, vis_map,'name','folder',overWrite=True)

