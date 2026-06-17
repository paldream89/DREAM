# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 14:24:06 2022

@author: Admin
"""

import numpy as np
from math import ceil,sqrt,e,floor
from scipy import optimize
# import matplotlib.pyplot as plt

def twoD_diff_dis_func(x,a,b,c):
    
    return 2*c*x/a*np.exp(-x**2/a)+b*x

def oneD_diff_dis_func(x,a,b,c):
    
    return c*np.exp(-x**2/a)+b

def oneD_kinesin_dis_func(x,a,b,c,d):
    
    return c*np.exp(-(x-d)**2/a)+b

def oneD_diff_EP_func(x,a,b,c,x0):
    
    return c*np.exp(-(x-x0)**2/a)+b

def drate_fitter(hist,disp_thre,hist_bin_num,pixel_size,min_points_percell,time_interval,fitting_error):
    
    hist_bin_size = disp_thre/hist_bin_num 
    hist_bin_size_1D = 1.4*disp_thre/hist_bin_num 
    bins_x = np.arange(0.5*hist_bin_size, disp_thre, hist_bin_size, dtype=np.float32)
    
    bins_x_1D = np.arange(-0.7*disp_thre+0.5*hist_bin_size_1D, 0.7*disp_thre, hist_bin_size_1D, dtype=np.float32)
    
    # for guessed initial fitting parameters
    bin_thre = int(floor((disp_thre*0.6)/hist_bin_size)) 
    hist_start = hist[5:5+hist_bin_num].astype(np.int32)
    if np.sum(hist_start)>min_points_percell:
        
        if hist[3] > 5:
            
            guessed_a = (np.average(bins_x[0:bin_thre],weights=hist_start[0:bin_thre]))**2
            guessed_b = (hist_start[-1]/bins_x[-1] + hist_start[-2]/bins_x[-2])/2
            guessed_c = np.amax(hist_start[0:bin_thre])*sqrt(guessed_a)*e/2
            
            p0 = [guessed_a,guessed_b,guessed_c]
            bounds = (0,[disp_thre**2,np.inf,np.inf])
            
            try: 
                
                popt,_ = optimize.curve_fit(twoD_diff_dis_func,bins_x,hist_start,p0=p0,
                                            bounds=bounds)
                
            except RuntimeError or RuntimeWarning:
                
                print('RuntimeError')
                popt = p0
                popt[0] = 0
            
            drate_start = popt[0] * (pixel_size/1000)**2 /4/time_interval
            
        else:
            
            guessed_a = 2*(0.3*disp_thre)**2
            guessed_b = hist_start[-1]
            guessed_c = np.amax(hist_start)
            # c*np.exp(-x**2/a)+b
            p0 = [guessed_a,guessed_b,guessed_c]
            bounds = (0,[2*disp_thre**2,np.inf,np.inf])
            
            try: 
                
                popt,_ = optimize.curve_fit(oneD_diff_dis_func,bins_x_1D,hist_start,p0=p0,
                                            bounds=bounds)

                drate_start = popt[0] * (pixel_size/1000)**2 /4/time_interval
                
            except RuntimeError  or RuntimeWarning:
                
                print('RuntimeError')
                drate_start = 0
        
    else:
        drate_start = 0
        
    hist_end = hist[10+hist_bin_num:10+2*hist_bin_num].astype(np.int32)
    if np.sum(hist_end)>min_points_percell:
        
        if hist[8+hist_bin_num] > 5:
        
            guessed_a = (np.average(bins_x[0:bin_thre],weights=hist_end[0:bin_thre]))**2
            guessed_b = (hist_end[-1]/bins_x[-1] + hist_end[-2]/bins_x[-2])/2
            guessed_c = np.amax(hist_end[0:bin_thre])*sqrt(guessed_a)*e/2
            
            p0 = [guessed_a,guessed_b,guessed_c]
            bounds = (0,[disp_thre**2,np.inf,np.inf])
            
            try: 
                
                popt,_ = optimize.curve_fit(twoD_diff_dis_func,bins_x,hist_end,p0=p0,
                                            bounds=bounds)
                
            except RuntimeError or RuntimeWarning:
                
                print('RuntimeError')
                popt = p0
                popt[0] = 0
            
            drate_end = popt[0] * (pixel_size/1000)**2 /4/time_interval
            
        else:
            
            guessed_a = 2*(0.3*disp_thre)**2
            guessed_b = hist_end[-1]
            guessed_c = np.amax(hist_end)
            # c*np.exp(-x**2/a)+b
            p0 = [guessed_a,guessed_b,guessed_c]
            bounds = (0,[2*disp_thre**2,np.inf,np.inf])
            
            try: 
                
                popt,_ = optimize.curve_fit(oneD_diff_dis_func,bins_x_1D,hist_end,p0=p0,
                                            bounds=bounds)
                drate_end = popt[0] * (pixel_size/1000)**2 /4/time_interval

            except RuntimeError  or RuntimeWarning:
                
                print('RuntimeError')
                drate_end = 0
    
    else:
        drate_end = 0
        
    drate = np.hstack((drate_start,drate_end))
    return drate

def drate_fitter_kinesin(hist,disp_thre,hist_bin_num,pixel_size,min_points_percell,time_interval,fitting_error):
    
    hist_bin_size = disp_thre/hist_bin_num 
    hist_bin_size_1D = 1.4*disp_thre/hist_bin_num 
    bins_x = np.arange(0.5*hist_bin_size, disp_thre, hist_bin_size, dtype=np.float32)
    
    bins_x_1D = np.arange(-0.7*disp_thre+0.5*hist_bin_size_1D, 0.7*disp_thre, hist_bin_size_1D, dtype=np.float32)
    
    # for guessed initial fitting parameters
    bin_thre = int(floor((disp_thre*0.6)/hist_bin_size)) 
    hist_start = hist[5:5+hist_bin_num].astype(np.int32)
    if np.sum(hist_start)>min_points_percell:
        
        if hist[3] > 5:  # because for 2D, manually set angle = 100
            
            # guessed_a = (np.average(bins_x[0:bin_thre],weights=hist_start[0:bin_thre]))**2
            # guessed_b = (hist_start[-1]/bins_x[-1] + hist_start[-2]/bins_x[-2])/2
            # guessed_c = np.amax(hist_start[0:bin_thre])*sqrt(guessed_a)*e/2
            
            # p0 = [guessed_a,guessed_b,guessed_c]
            # bounds = (0,[disp_thre**2,np.inf,np.inf])
            
            # try: 
                
            #     popt,_ = optimize.curve_fit(twoD_diff_dis_func,bins_x,hist_start,p0=p0,
            #                                 bounds=bounds)
                
            # except RuntimeError or RuntimeWarning:
                
            #     print('RuntimeError')
            #     popt = p0
            #     popt[0] = 0
            
            drate_start = -1
            
        else:
            
            guessed_a = 2*(0.3*disp_thre)**2
            guessed_b = hist_start[-1]
            guessed_c = np.amax(hist_start)
            guessed_d = np.average(bins_x_1D,weights=hist_start)
            # c*np.exp(-x**2/a)+b
            p0 = [guessed_a,guessed_b,guessed_c,guessed_d]
            bounds = ([0,0,0,-0.7*disp_thre],[2*disp_thre**2,np.inf,np.inf,0.7*disp_thre])
            
            try: 
                
                popt,_ = optimize.curve_fit(oneD_kinesin_dis_func,bins_x_1D,hist_start,p0=p0,
                                            bounds=bounds)

                drate_start = abs(popt[3] * pixel_size / time_interval)
                
            except RuntimeError  or RuntimeWarning:
                
                print('RuntimeError')
                drate_start = -2
        
    else:
        drate_start = -3
        
    hist_end = hist[10+hist_bin_num:10+2*hist_bin_num].astype(np.int32)
    if np.sum(hist_end)>min_points_percell:
        
        if hist[8+hist_bin_num] > 5:
        
            # guessed_a = (np.average(bins_x[0:bin_thre],weights=hist_end[0:bin_thre]))**2
            # guessed_b = (hist_end[-1]/bins_x[-1] + hist_end[-2]/bins_x[-2])/2
            # guessed_c = np.amax(hist_end[0:bin_thre])*sqrt(guessed_a)*e/2
            
            # p0 = [guessed_a,guessed_b,guessed_c]
            # bounds = (0,[disp_thre**2,np.inf,np.inf])
            
            # try: 
                
            #     popt,_ = optimize.curve_fit(twoD_diff_dis_func,bins_x,hist_end,p0=p0,
            #                                 bounds=bounds)
                
            # except RuntimeError or RuntimeWarning:
                
            #     print('RuntimeError')
            #     popt = p0
            #     popt[0] = 0
            
            drate_end = -1
            
        else:
            
            guessed_a = 2*(0.3*disp_thre)**2
            guessed_b = hist_end[-1]
            guessed_c = np.amax(hist_end)
            guessed_d = np.average(bins_x_1D,weights=hist_end)
            # c*np.exp(-(x-d)**2/a)+b
            p0 = [guessed_a,guessed_b,guessed_c,guessed_d]
            bounds = ([0,0,0,-0.7*disp_thre],[2*disp_thre**2,np.inf,np.inf,0.7*disp_thre])
            
            try: 
                
                popt,_ = optimize.curve_fit(oneD_kinesin_dis_func,bins_x_1D,hist_end,p0=p0,
                                            bounds=bounds)
                drate_end = abs(popt[3] * pixel_size / time_interval)

            except RuntimeError  or RuntimeWarning:
                
                print('RuntimeError')
                drate_end = -2
    
    else:
        drate_end = -3
        
    drate = np.hstack((drate_start,drate_end))
    return drate

def drate_fitter_keepZcI(hist,disp_thre,hist_bin_num,pixel_size,min_points_percell,time_interval,fitting_error):
    
    hist_bin_size = disp_thre/hist_bin_num 
    hist_bin_size_1D = 1.4*disp_thre/hist_bin_num 
    bins_x = np.arange(0.5*hist_bin_size, disp_thre, hist_bin_size, dtype=np.float32)
    
    bins_x_1D = np.arange(-0.7*disp_thre+0.5*hist_bin_size_1D, 0.7*disp_thre, hist_bin_size_1D, dtype=np.float32)
    
    # for guessed initial fitting parameters
    bin_thre = int(floor((disp_thre*0.6)/hist_bin_size)) 
    hist_start = hist[5:5+hist_bin_num].astype(np.int32)
    if np.sum(hist_start)>min_points_percell:
        
        if hist[3] > 5:
            
            guessed_a = (np.average(bins_x[0:bin_thre],weights=hist_start[0:bin_thre]))**2
            guessed_b = (hist_start[-1]/bins_x[-1] + hist_start[-2]/bins_x[-2])/2
            guessed_c = np.amax(hist_start[0:bin_thre])*sqrt(guessed_a)*e/2
            
            p0 = [guessed_a,guessed_b,guessed_c]
            bounds = (0,[disp_thre**2,np.inf,np.inf])
            
            try: 
                
                popt,_ = optimize.curve_fit(twoD_diff_dis_func,bins_x,hist_start,p0=p0,
                                            bounds=bounds)
                
            except RuntimeError or RuntimeWarning:
                
                print('RuntimeError')
                popt = p0
                popt[0] = 0
            
            drate_start = popt[0] * (pixel_size/1000)**2 /4/time_interval
            
        else:
            
            guessed_a = 2*(0.3*disp_thre)**2
            guessed_b = hist_start[-1]
            guessed_c = np.amax(hist_start)
            # c*np.exp(-x**2/a)+b
            p0 = [guessed_a,guessed_b,guessed_c]
            bounds = (0,[2*disp_thre**2,np.inf,np.inf])
            
            try: 
                
                popt,_ = optimize.curve_fit(oneD_diff_dis_func,bins_x_1D,hist_start,p0=p0,
                                            bounds=bounds)

                drate_start = popt[0] * (pixel_size/1000)**2 /4/time_interval
                
            except RuntimeError  or RuntimeWarning:
                
                print('RuntimeError')
                drate_start = 0
        
    else:
        drate_start = 0
        
    hist_end = hist[10+hist_bin_num:10+2*hist_bin_num].astype(np.int32)
    if np.sum(hist_end)>min_points_percell:
        
        if hist[8+hist_bin_num] > 5:
        
            guessed_a = (np.average(bins_x[0:bin_thre],weights=hist_end[0:bin_thre]))**2
            guessed_b = (hist_end[-1]/bins_x[-1] + hist_end[-2]/bins_x[-2])/2
            guessed_c = np.amax(hist_end[0:bin_thre])*sqrt(guessed_a)*e/2
            
            p0 = [guessed_a,guessed_b,guessed_c]
            bounds = (0,[disp_thre**2,np.inf,np.inf])
            
            try: 
                
                popt,_ = optimize.curve_fit(twoD_diff_dis_func,bins_x,hist_end,p0=p0,
                                            bounds=bounds)
                
            except RuntimeError or RuntimeWarning:
                
                print('RuntimeError')
                popt = p0
                popt[0] = 0
            
            drate_end = popt[0] * (pixel_size/1000)**2 /4/time_interval
            
        else:
            
            guessed_a = 2*(0.3*disp_thre)**2
            guessed_b = hist_end[-1]
            guessed_c = np.amax(hist_end)
            # c*np.exp(-x**2/a)+b
            p0 = [guessed_a,guessed_b,guessed_c]
            bounds = (0,[2*disp_thre**2,np.inf,np.inf])
            
            try: 
                
                popt,_ = optimize.curve_fit(oneD_diff_dis_func,bins_x_1D,hist_end,p0=p0,
                                            bounds=bounds)
                drate_end = popt[0] * (pixel_size/1000)**2 /4/time_interval

            except RuntimeError  or RuntimeWarning:
                
                print('RuntimeError')
                drate_end = 0
    
    else:
        drate_end = 0
        
    drate = np.hstack((drate_start,drate_end))
    return drate

def drate_fitter_Z(hist,disp_thre,hist_bin_num,pixel_size,min_points_percell,time_interval,fitting_error):
    
    hist_bin_size = disp_thre/hist_bin_num 
    hist_bin_size_1D = 1.4*disp_thre/hist_bin_num 
    bins_x = np.arange(0.5*hist_bin_size, disp_thre, hist_bin_size, dtype=np.float32)
    
    bins_x_1D = np.arange(-0.7*disp_thre+0.5*hist_bin_size_1D, 0.7*disp_thre, hist_bin_size_1D, dtype=np.float32)
    
    # for guessed initial fitting parameters
    bin_thre = int(floor((disp_thre*0.6)/hist_bin_size)) 
    hist_start = hist[5:5+hist_bin_num].astype(np.int32)
    if np.sum(hist_start)>min_points_percell:
            
        guessed_a = 2*(0.3*disp_thre)**2
        guessed_b = hist_start[-1]
        guessed_c = np.amax(hist_start)
        # c*np.exp(-x**2/a)+b
        p0 = [guessed_a,guessed_b,guessed_c]
        bounds = (0,[2*disp_thre**2,np.inf,np.inf])
        
        try: 
            
            popt,_ = optimize.curve_fit(oneD_diff_dis_func,bins_x_1D,hist_start,p0=p0,
                                        bounds=bounds)

            drate_start = popt[0] * (pixel_size/1000)**2 /4/time_interval
            
        except RuntimeError  or RuntimeWarning:
            
            print('RuntimeError')
            drate_start = 0
        
    else:
        drate_start = 0
        
    hist_end = hist[10+hist_bin_num:10+2*hist_bin_num].astype(np.int32)
    if np.sum(hist_end)>min_points_percell:
            
        guessed_a = 2*(0.3*disp_thre)**2
        guessed_b = hist_end[-1]
        guessed_c = np.amax(hist_end)
        # c*np.exp(-x**2/a)+b
        p0 = [guessed_a,guessed_b,guessed_c]
        bounds = (0,[2*disp_thre**2,np.inf,np.inf])
        
        try: 
            
            popt,_ = optimize.curve_fit(oneD_diff_dis_func,bins_x_1D,hist_end,p0=p0,
                                        bounds=bounds)
            drate_end = popt[0] * (pixel_size/1000)**2 /4/time_interval

        except RuntimeError  or RuntimeWarning:
            
            print('RuntimeError')
            drate_end = 0
    
    else:
        drate_end = 0
        
    drate = np.hstack((drate_start,drate_end))
    return drate

def drate_fitter_x_y(hist,disp_threX,disp_threY,hist_bin_numX,hist_bin_numY,
                     pixel_size,min_points_percell,
                     time_interval,fitting_error):
    
    hist_bin_size_1D_X = 2*disp_threX/hist_bin_numX
    bins_1D_x = np.arange(-disp_threX+0.5*hist_bin_size_1D_X, disp_threX, hist_bin_size_1D_X, dtype=np.float32)
    hist_bin_size_1D_Y = 2*disp_threY/hist_bin_numY
    bins_1D_y = np.arange(-disp_threY+0.5*hist_bin_size_1D_Y, disp_threY, hist_bin_size_1D_Y, dtype=np.float32)
    
    # ========== 高斯拟合 ==========
    def fit_hist(bin_centers, counts):
        # 初始参数: a, b, c
        a_init = np.var(bin_centers)
        b_init = np.min(counts)
        c_init = np.max(counts) - b_init
        p0 = [a_init, b_init, c_init]
        bounds = ([1e-6, 0, 0], [np.inf, np.inf, np.inf])
    
        try:
            popt, _ = optimize.curve_fit(
                oneD_diff_dis_func, bin_centers, counts,
                p0=p0, bounds=bounds)
        except:
            popt = [0, 0, 0]
        return popt
    
    def fit_hist_EP(bin_centers, counts):
        # 初始参数: a, b, c
        a_init = np.var(bin_centers)
        b_init = np.min(counts)
        c_init = np.max(counts) - b_init
        x0_init = 0
        p0 = [a_init, b_init, c_init, x0_init]
        bounds = ([1e-6, 0, 0, -disp_threX], [np.inf, np.inf, np.inf, disp_threX])
    
        try:
            popt, _ = optimize.curve_fit(
                oneD_diff_EP_func, bin_centers, counts,
                p0=p0, bounds=bounds)
        except:
            popt = [0, 0, 0, 0]
        return popt
    
    hist_Y = hist[4+hist_bin_numX:4+hist_bin_numX+hist_bin_numY]+hist[4+hist_bin_numX*2+hist_bin_numY:4+hist_bin_numX*2+hist_bin_numY*2]
    
    if np.sum(hist_Y)>min_points_percell:
        
        popt_Y = fit_hist(bins_1D_y,hist_Y)
        drate_Y = popt_Y[0] * (pixel_size/1000)**2 /4/time_interval
        
        hist_X_odd = hist[4:4+hist_bin_numX]
        hist_X_even = hist[4+hist_bin_numX+hist_bin_numY:4+hist_bin_numX*2+hist_bin_numY]
        hist_X = hist_X_odd[::-1]+hist_X_even  # flip and add on
        popt_X = fit_hist_EP(bins_1D_x,hist_X)
        drate_X = popt_X[0] * (pixel_size/1000)**2 /4/time_interval
        EP_X = popt_X[3]* (pixel_size/1000)/time_interval
    
    else:
        
        drate_Y=0
        drate_X=0
        EP_X=0
        
    drate_array=np.array([drate_Y,drate_X,EP_X])
    
    return drate_array