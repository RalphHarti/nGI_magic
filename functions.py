#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:10:52 2017

@author: harti and valsecchi
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
import pyfits
from os import makedirs
from astropy.io import fits 
import h5py
from pathlib import Path



def readRead(path,dc=0):
    """
    readRead()
    
    """
    my_file = Path(path)
    if my_file.is_file():
        im_a1 = []
        if path.lower().endswith('.fits'):
            im_a1.append(fits.open(path,ignore_missing_end=True)[0].data)
        elif path.lower().endswith(('.tiff','.tif')) :
            im_a1.append(np.asarray(Image.open(path)))
        elif path.lower().endswith(('.hdf','.h4','.hdf4','.he2','h5','.hdf5','.he5')): 
            hdf = h5py.File(path,'r')['entry']['data']['data'].value
            for iScan in hdf:
                im_a1.append(iScan)
        else:
    ##        if you don't want to break the program just comment the raise and eventually uncomment the print for having a feedback
    #        print(os.path.splitext(path)[-1],'file extension not yet implemented....Do it your own way!')
            raise OSError('file extension not yet implemented....Do it your own way!')     
#        if dc:
        im_a1 = np.asarray(im_a1)-dc
#        else:
#            im_a1 = np.asarray(im_a1)
        return im_a1
    else:
        print(path,'does not exist')
    
def read_data(path_im,path_ob,path_dc):
    """
    read()
    """
#    Dark current
    imExt = ['.fits','.tiff','.tif','.hdf','.h4','.hdf4','.he2','h5','.hdf5','.he5']
    if path_dc:
        im_a1 = []
        filenames_dc = [name for name in os.listdir(path_dc) if name.lower().endswith(tuple(imExt))]
        filenames_dc.sort()
        for name in filenames_dc:
            full_path_name = path_dc+'/'+name
            print(full_path_name)
            im_a1.append(readRead(full_path_name))
        im_a1 = np.asarray(im_a1)
        im_a1 = np.sum(im_a1,axis=0)/np.shape(im_a1)[0]
    
#    Open beam
    filenames_ob = [name for name in os.listdir(path_ob) if name.lower().endswith(tuple(imExt))]
    filenames_ob.sort()
    stack_ob = []
    for name in filenames_ob:
        full_path_name = path_ob+'/'+name
        print(full_path_name)
        if path_dc:
            stack_ob.append(readRead(full_path_name,im_a1)) #with dc
        else:
            stack_ob.append(readRead(full_path_name))   #without dc
    stack_ob = np.concatenate(stack_ob)

    
#    Projections
    filenames_im = [name for name in os.listdir(path_im) if name.lower().endswith(tuple(imExt))]
    filenames_im.sort()
    stack_im_ar = []
    for name in filenames_im:
        full_path_name = path_im+'/'+name
        print(full_path_name)
        if path_dc:
            stack_im_ar.append(readRead(full_path_name,im_a1)) #with dc
        else:
            stack_im_ar.append(readRead(full_path_name))   #without dc
    stack_im_ar = np.concatenate(stack_im_ar)
    
    if np.shape(stack_im_ar) != np.shape(stack_ob):
            raise ValueError('Data and open beam have different shapes')
        
    return stack_im_ar,stack_ob




xROI,yROI,thickROI,heightROI=10,10,35,35 #parameter for roi
   
def roi(im,xROI,yROI,thickROI,heightROI,show=False,titleOne='Original image with selected ROI',titleTwo='ROI'):
    """
    roi() takes a SINGLE image and crops it 
    (xROI,yROI) is the upper left-hand corner of the cropping rectangle 
    """
    if (0<=xROI<=im.shape[0] and 0<=xROI+thickROI<=im.shape[0] and 0<=yROI<=im.shape[1] and 0<=yROI+heightROI<=im.shape[1]):
        imROI = im[yROI:yROI+heightROI,xROI:xROI+thickROI]
        if show:
            vmin,vmax=im.min(),im.max()
            cmap='gray'
            fig = plt.figure(figsize=(15,10)) 
            gs = gridspec.GridSpec(1, 2,width_ratios=[4,1],height_ratios=[1,1]) 
            ax = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
#            fig,(ax,ax2) = plt.subplots(1,2,sharex=False,sharey=False,figsize=(15,10))
            ax.imshow(im,vmin=vmin, vmax=vmax,interpolation='nearest',cmap=cmap)
            rectNorm = patches.Rectangle((xROI,yROI),thickROI,heightROI,linewidth=1,edgecolor='m',facecolor='none')
            ax.add_patch(rectNorm)
            ax.set_title(titleOne)
            ax2.imshow(im,vmin=vmin, vmax=vmax,interpolation='nearest',cmap=cmap)
            ax2.set_title(titleTwo)
            ax2.set_xlim([xROI,xROI+thickROI])
            ax2.set_ylim([yROI+heightROI,yROI])
            plt.tight_layout()
            plt.show()
            plt.close('all')
        return(imROI)
    else:
        print('!!!WARNING!!! \nROI out of range')


def cropped(stack_im,stack_ob,xROI=xROI,yROI=yROI,thickROI=thickROI,heightROI=heightROI):
    """
    cropped() takes a stack of data,ob and dark currenr and crops them 
    (xROI,yROI) is the upper left-hand corner of the cropping rectangle 
    """
    stack_im_ar = [roi(im=stack_im[0],xROI=xROI,yROI=yROI,thickROI=thickROI,heightROI=heightROI,show=True,titleTwo='Cropped region')]
    for i in stack_im[1:]:
        stack_im_ar.append(roi(im=i,xROI=xROI,yROI=yROI,thickROI=thickROI,heightROI=heightROI,show=False))
#    stack_im_ar = [roi(im=i,xROI=xROI,yROI=yROI,thickROI=thickROI,heightROI=heightROI,show=True) for i in stack_im]
    
    stack_ob_ar = [roi(im=i,xROI=xROI,yROI=yROI,thickROI=thickROI,heightROI=heightROI,show=False) for i in stack_ob]
    
    
    return(np.asarray(stack_im_ar),np.asarray(stack_ob_ar))

    
def normalization(stack_im,stack_ob,xROI=xROI,yROI=yROI,thickROI=thickROI,heightROI=heightROI,show=True):
    """
    normalization()
    """
    Area = abs(thickROI*heightROI)  
    
    stack_im_ar = []    
    
    roi(stack_im[0],xROI,yROI,thickROI,heightROI,show,titleTwo='Area for normalization')

    stack_im_ar = [l/(l[yROI:yROI+heightROI+1,xROI:xROI+thickROI+1].sum()/Area) for l in stack_im]   
    for i in stack_im_ar:
        print(i.mean())
        
    stack_ob_ar = [l/(l[yROI:yROI+heightROI+1,xROI:xROI+thickROI+1].sum()/Area) for l in stack_ob] 
    return(np.asarray(stack_im_ar),np.asarray(stack_ob_ar))

def oscillation(stack_im,stack_ob,xROI=xROI,yROI=yROI,thickROI=thickROI,heightROI=heightROI,repeatedPeriod=False,folder=False):
    """
   stack_im is the data stack
   stack_ob is the open beam stack
   xROI,yROI,thickROI,heightROI is the rectangle of the ROI for the oscillation plot and (xROI,yROI) is the upper left corner
   repeatedPeriod=bool  double the period of the stack appendind to the end of the first
   folder is the folder where you want to save the plot if False it does not save
    
    """
    titleOne='Area for oscillation'
    titleTwo='Oscillation plot'
    area = abs(thickROI*heightROI)
    stack_ob_ar = [l[yROI:yROI+heightROI+1,xROI:xROI+thickROI+1].sum()/area for l in stack_ob] 
    stack_im_ar = [l[yROI:yROI+heightROI+1,xROI:xROI+thickROI+1].sum()/area for l in stack_im] 
    if repeatedPeriod:
        stack_ob_ar += stack_ob_ar
        stack_im_ar += stack_im_ar
        titleTwo += ' repetead period'
    
#    PLOT oscillation
    im = stack_im[0]
    vmin,vmax=im.min(),im.max()
    cmap='gray'
    fig = plt.figure(figsize=(15,10)) 
    gs = gridspec.GridSpec(1,2,width_ratios=[2,1],height_ratios=[1,1]) 
    ax = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax.imshow(im,vmin=vmin, vmax=vmax,interpolation='nearest',cmap=cmap)
    rectNorm = patches.Rectangle((xROI,yROI),thickROI,heightROI,linewidth=1,edgecolor='m',facecolor='none')
    ax.add_patch(rectNorm)
    ax.set_title(titleOne)
    
    ax2.plot(range(1,len(stack_im_ar)+1),stack_im_ar,color='g',label='data')
    ax2.scatter(range(1,len(stack_im_ar)+1),stack_im_ar,marker='*',color='g')
    ax2.plot(range(1,len(stack_ob_ar)+1),stack_ob_ar,color='b',label='ob')
    ax2.scatter(range(1,len(stack_ob_ar)+1),stack_ob_ar,color='b')
    ax2.legend(loc=1, shadow=True)
    ax2.set_title(titleTwo)
    ax2.set_xlim((0,len(stack_ob_ar)+2))
#    ax2.set_ylim([yROI+heightROI,yROI])
    plt.tight_layout()
    plt.show()
    if folder:
        if not os.path.exists('data/'+folder):
            makedirs('data/'+folder) 
            print('files saved in folder: ','data/'+folder)
        fig.savefig('data/'+folder+'/oscillationPlot.png', bbox_inches='tight')
    plt.close('all')
    

def matrix(stack_im):
    """
    """
    shapeStack = np.shape(stack_im)
    B = np.zeros((shapeStack[0],3))  
    numberPeriods = 1
    ###TODO: function for number of periods
    
    stack_imReshaped = np.reshape(stack_im,[shapeStack[0],shapeStack[1]*shapeStack[2]])
    rangeStack = range(shapeStack[0])
    
    for j in rangeStack:
        B[j][0] = 1.0
        B[j][1] = np.cos(2*np.pi*rangeStack[j]*numberPeriods/(shapeStack[0]-1))
        B[j][2] = np.sin(2*np.pi*rangeStack[j]*numberPeriods/(shapeStack[0]-1))
    B = np.matrix(B)
    
    G = (B.T * B).I * B.T
    print(np.shape(G),np.shape(stack_imReshaped))
    A = (G*stack_imReshaped)
    offSet,absoluteAmpl,absPhase = A[0,:],A[1,:],A[2,:]
    a0 = np.reshape(np.asarray(offSet),[shapeStack[1],shapeStack[2]])
    a1 = np.reshape(np.sqrt(np.asarray(absoluteAmpl)**2+np.asarray(absPhase)**2),[shapeStack[1],shapeStack[2]])
    phi = np.reshape(np.arctan((np.asarray(absPhase)/np.asarray(absoluteAmpl))),[shapeStack[1],shapeStack[2]])
    return a0,a1,phi
     
def reductionMatrix(stack_im,stack_ob):
    """
    reductionMatrix(): it applies matrix() to both stacks im and ob
    """
    return (matrix(stack_im),matrix(stack_ob))
    
def createIm(stack_im,stack_ob):
    """
    """
    imParam,obParam = reductionMatrix(stack_im,stack_ob)
    
    TI = np.divide(imParam[0],obParam[0])
    DPCI = imParam[2]-obParam[2]
    DFI = np.divide(np.divide(imParam[1],imParam[0]),np.divide(obParam[1],obParam[0]))
    VIS_map = np.divide(obParam[1],obParam[0])
    return TI, DPCI, DFI, VIS_map
    
def saveIm(ti,dpci,dfi,vis_map,name='name',folder='folder',overWrite=False):
    """
    """
    if not os.path.exists('data/'+folder):
        makedirs('data/'+folder) 
        print('files saved in folder: ','data/'+folder)
    fits.writeto('data/'+folder+'/ti_'+str(name)+'.fits',ti,clobber=overWrite)
    fits.writeto('data/'+folder+'/dpci_'+str(name)+'.fits',dpci,clobber=overWrite)
    fits.writeto('data/'+folder+'/dfi_'+str(name)+'.fits',dfi,clobber=overWrite)
    fits.writeto('data/'+folder+'/visi_'+str(name)+'.fits',vis_map,clobber=overWrite)
    
def binning(stack_im,stack_ob,bin_fac=None):
    
    num_im,x_im,y_im = np.shape(stack_im)
    stack_im_tmp = list()
    if bin_fac:
        for i in range(num_im):
            t = stack_im[i]
            t_resh = t.reshape(int(x_im/bin_fac), bin_fac, int(y_im/bin_fac), bin_fac)
            stack_im_tmp.append(t_resh.mean(axis=3).mean(axis=1))
    
        stack_im_bin = np.asarray(stack_im_tmp) 
    else:
        stack_im_bin = stack_im
        
    
    num_ob,x_ob,y_ob = np.shape(stack_ob)
    stack_ob_tmp = list()
    if bin_fac:
        for i in range(num_ob):
            t = stack_ob[i]
            t_resh = t.reshape(int(x_ob/bin_fac), bin_fac, int(y_ob/bin_fac), bin_fac)
            stack_ob_tmp.append(t_resh.mean(axis=3).mean(axis=1))
    
        stack_ob_bin = np.asarray(stack_ob_tmp) 
    else:
        stack_ob_bin = stack_ob
        
    return stack_im_bin, stack_ob_bin
        
