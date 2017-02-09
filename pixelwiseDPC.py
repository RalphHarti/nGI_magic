#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:13:09 2017

@author: harti and valsecchi
"""
import scipy
import numpy as np

def pixelWiseDPC(dpci,p2um=4,d1cm=1.94,lambdaAmstr=4.1):
    """
    formula:
        d Φ / d x  =       (p2 / λ * d1) φ           
          according to SI 
    
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----        
        
    """
    
    dphi_over_dxPixel = (p2um*1e-3)/(lambdaAmstr*1e-7*d1cm*10*2)*dpci
#    dphi_over_dxPixel = scipy.integrate.simps()

    return dphi_over_dxPixel

def pixelWisePC(dpciUnit,pixelConversion=9.2222,p2um=4,d1cm=1.94,lambdaAmstr=4.1):
    """
    formula:
        ∫ d Φ   =     ∫  (p2 / λ * d1) φ  dx         
    according to SI 
    
    Parameters
    ----------
    
    Returns
    -------
    
    Notes
    -----    
    """
    dphi_over_dxPixel = np.array([scipy.integrate.cumtrapz(line*pixelConversion,initial=0) for line in dpciUnit])
    dphi_over_dxPixelREV = np.fliplr(np.array([scipy.integrate.cumtrapz(line*pixelConversion,initial=0) for line in np.fliplr(dpciUnit)]))

    return (np.abs(dphi_over_dxPixel)+np.abs(dphi_over_dxPixelREV))/2

print(pixelWiseDPC(1))