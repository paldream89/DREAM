# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 14:23:31 2022

@author: Admin
"""

import numpy as np
from math import ceil,sqrt,e,floor,atan2,atan,pi
from scipy import optimize
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit

# import matplotlib.pyplot as plt

def twoD_diff_dis_func(x,a,b,c):
    
    return 2*c*x/a*np.exp(-x**2/a)+b*x

def gaussian_with_background(x, a1, x1, w1, y0):
    
    return a1 * np.exp(-((x - x1) ** 2) / (2 * w1 ** 2)) + y0

def gaussian_with_background_invert(x, a1, x1, w1, y0):
    
    return (a1-y0) * np.exp(-((x - x1) ** 2) / (2 * w1 ** 2)) + y0

def hist_maker(STORM_npdata,disp_thre,hist_bin_num,image_bin_size,
               xinter_yinter_frame_disp_angle_xcyc,
               indice_map,counter_map,xinter_max,yinter_max,min_points_percell,
               pca_ratio_thre):
    
    own_xc = STORM_npdata['x_start']/image_bin_size
    own_yc = STORM_npdata['y_start']/image_bin_size
    own_frame = STORM_npdata['frame']
    own_angle = 100
    own_x_inter = own_xc.astype(np.int16)
    own_y_inter = own_yc.astype(np.int16)
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map[own_x_inter-1,own_y_inter-1,0:counter_map[own_x_inter-1,own_y_inter-1]],
                                      indice_map[own_x_inter,own_y_inter-1,0:counter_map[own_x_inter,own_y_inter-1]],
                                      indice_map[own_x_inter+1,own_y_inter-1,0:counter_map[own_x_inter+1,own_y_inter-1]],
                                      indice_map[own_x_inter-1,own_y_inter,0:counter_map[own_x_inter-1,own_y_inter]],
                                      indice_map[own_x_inter,own_y_inter,0:counter_map[own_x_inter,own_y_inter]],
                                      indice_map[own_x_inter+1,own_y_inter,0:counter_map[own_x_inter+1,own_y_inter]],
                                      indice_map[own_x_inter-1,own_y_inter+1,0:counter_map[own_x_inter-1,own_y_inter+1]],
                                      indice_map[own_x_inter,own_y_inter+1,0:counter_map[own_x_inter,own_y_inter+1]],
                                      indice_map[own_x_inter+1,own_y_inter+1,0:counter_map[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc['yc'][indice_foraverage]
    foraverage_disp = xinter_yinter_frame_disp_angle_xcyc['disp'][indice_foraverage]
    foraverage_distance_bool = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))<0.5
    foraverage_disp_final = foraverage_disp[foraverage_distance_bool]          
    own_points = np.size(foraverage_disp_final)
    
    if own_points>min_points_percell:
        
        foraverage_angle_final = xinter_yinter_frame_disp_angle_xcyc['angle'][indice_foraverage][foraverage_distance_bool]
        disp_y = foraverage_disp_final*np.sin(foraverage_angle_final)
        disp_x = foraverage_disp_final*np.cos(foraverage_angle_final)
        disp_x_y = np.transpose(np.vstack((disp_x,disp_y)))
        pca = PCA(n_components=1).fit(disp_x_y)
        comp_extracted = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        if explained_variance_ratio[0] > pca_ratio_thre:
            
            preferred_angle = atan2(comp_extracted[0,1],comp_extracted[0,0])
            own_angle = preferred_angle
            foraverage_projected_disp = foraverage_disp_final*np.cos(foraverage_angle_final-preferred_angle)
            
            averaged_histogram_start,_ = np.histogram(foraverage_projected_disp, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre))
        else:
            
            averaged_histogram_start,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (0,disp_thre))
        
    else:
        averaged_histogram_start = np.zeros(hist_bin_num)
    
    averaged_xc_start = (own_xc*image_bin_size)
    averaged_yc_start = (own_yc*image_bin_size)
    averaged_frame_start = own_frame
    averaged_angle_start = own_angle
    averaged_points_start = (own_points)
    
    # repeat for x_end and y_end, only difference is frame+1: start
    
    own_xc = STORM_npdata['x_end']/image_bin_size
    own_yc = STORM_npdata['y_end']/image_bin_size
    own_frame = STORM_npdata['frame']+1
    own_angle = 100
    own_x_inter = own_xc.astype(np.int16)
    own_y_inter = own_yc.astype(np.int16)
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map[own_x_inter-1,own_y_inter-1,0:counter_map[own_x_inter-1,own_y_inter-1]],
                                      indice_map[own_x_inter,own_y_inter-1,0:counter_map[own_x_inter,own_y_inter-1]],
                                      indice_map[own_x_inter+1,own_y_inter-1,0:counter_map[own_x_inter+1,own_y_inter-1]],
                                      indice_map[own_x_inter-1,own_y_inter,0:counter_map[own_x_inter-1,own_y_inter]],
                                      indice_map[own_x_inter,own_y_inter,0:counter_map[own_x_inter,own_y_inter]],
                                      indice_map[own_x_inter+1,own_y_inter,0:counter_map[own_x_inter+1,own_y_inter]],
                                      indice_map[own_x_inter-1,own_y_inter+1,0:counter_map[own_x_inter-1,own_y_inter+1]],
                                      indice_map[own_x_inter,own_y_inter+1,0:counter_map[own_x_inter,own_y_inter+1]],
                                      indice_map[own_x_inter+1,own_y_inter+1,0:counter_map[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc['yc'][indice_foraverage]
    foraverage_disp = xinter_yinter_frame_disp_angle_xcyc['disp'][indice_foraverage]
    foraverage_distance_bool = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))<0.5
    foraverage_disp_final = foraverage_disp[foraverage_distance_bool]          
    own_points = np.size(foraverage_disp_final)
    
    if own_points>min_points_percell:
        
        foraverage_angle_final = xinter_yinter_frame_disp_angle_xcyc['angle'][indice_foraverage][foraverage_distance_bool]
        disp_y = foraverage_disp_final*np.sin(foraverage_angle_final)
        disp_x = foraverage_disp_final*np.cos(foraverage_angle_final)
        disp_x_y = np.transpose(np.vstack((disp_x,disp_y)))
        pca = PCA(n_components=1).fit(disp_x_y)
        comp_extracted = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        if explained_variance_ratio[0] > pca_ratio_thre:
            
            preferred_angle = atan2(comp_extracted[0,1],comp_extracted[0,0])
            own_angle = preferred_angle
            foraverage_projected_disp = foraverage_disp_final*np.cos(foraverage_angle_final-preferred_angle)
            
            averaged_histogram_end,_ = np.histogram(foraverage_projected_disp, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre))
        else:
            
            averaged_histogram_end,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (0,disp_thre))
        
    else:
        averaged_histogram_end = np.zeros(hist_bin_num)
    
    averaged_xc_end = (own_xc*image_bin_size)
    averaged_yc_end = (own_yc*image_bin_size)
    averaged_frame_end = own_frame
    averaged_angle_end = own_angle
    averaged_points_end = (own_points)
    
    # repeat for x_end and y_end, only difference is frame+1: finished
    
    return_array = np.hstack((averaged_xc_start,averaged_yc_start,averaged_frame_start,
                             averaged_angle_start,averaged_points_start,averaged_histogram_start,
                             averaged_xc_end,averaged_yc_end,averaged_frame_end,
                            averaged_angle_end,averaged_points_end,averaged_histogram_end))
    
    return return_array

def hist_maker_bleed(STORM_npdata,disp_thre,hist_bin_num,image_bin_size,
               xinter_yinter_frame_disp_angle_xcyc,
               indice_map,counter_map,xinter_max,yinter_max,min_points_percell,
               pca_ratio_thre):
    
    FWHM = 0.5
    sigma = FWHM/2.3548
    own_xc = STORM_npdata['x_start']/image_bin_size
    own_yc = STORM_npdata['y_start']/image_bin_size
    own_frame = STORM_npdata['frame']
    own_angle = 100
    own_x_inter = own_xc.astype(np.int16)
    own_y_inter = own_yc.astype(np.int16)
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map[own_x_inter-1,own_y_inter-1,0:counter_map[own_x_inter-1,own_y_inter-1]],
                                      indice_map[own_x_inter,own_y_inter-1,0:counter_map[own_x_inter,own_y_inter-1]],
                                      indice_map[own_x_inter+1,own_y_inter-1,0:counter_map[own_x_inter+1,own_y_inter-1]],
                                      indice_map[own_x_inter-1,own_y_inter,0:counter_map[own_x_inter-1,own_y_inter]],
                                      indice_map[own_x_inter,own_y_inter,0:counter_map[own_x_inter,own_y_inter]],
                                      indice_map[own_x_inter+1,own_y_inter,0:counter_map[own_x_inter+1,own_y_inter]],
                                      indice_map[own_x_inter-1,own_y_inter+1,0:counter_map[own_x_inter-1,own_y_inter+1]],
                                      indice_map[own_x_inter,own_y_inter+1,0:counter_map[own_x_inter,own_y_inter+1]],
                                      indice_map[own_x_inter+1,own_y_inter+1,0:counter_map[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc['yc'][indice_foraverage]
    foraverage_disp = xinter_yinter_frame_disp_angle_xcyc['disp'][indice_foraverage]
    foraverage_distance = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))
    foraverage_distance_bool = foraverage_distance<0.5
    foraverage_disp_final = foraverage_disp[foraverage_distance_bool]
    foraverage_distance_final = foraverage_distance[foraverage_distance_bool]
    own_points = np.size(foraverage_disp_final)
    
    if own_points>min_points_percell:
        
        foraverage_angle_final = xinter_yinter_frame_disp_angle_xcyc['angle'][indice_foraverage][foraverage_distance_bool]
        disp_y = foraverage_disp_final*np.sin(foraverage_angle_final)
        disp_x = foraverage_disp_final*np.cos(foraverage_angle_final)
        disp_x_y = np.transpose(np.vstack((disp_x,disp_y)))
        pca = PCA(n_components=1).fit(disp_x_y)
        comp_extracted = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        weights_bleed = np.floor(e**(foraverage_distance_final*foraverage_distance_final/-2/sigma/sigma)/sigma/sqrt(2*pi)*100)
        if explained_variance_ratio[0] > pca_ratio_thre:
            
            preferred_angle = atan2(comp_extracted[0,1],comp_extracted[0,0])
            own_angle = preferred_angle
            foraverage_projected_disp = foraverage_disp_final*np.cos(foraverage_angle_final-preferred_angle)
            
            averaged_histogram_start,_ = np.histogram(foraverage_projected_disp, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre),
                                  weights=weights_bleed)
        else:
            
            averaged_histogram_start,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (0,disp_thre),
                                  weights=weights_bleed)
        
    else:
        averaged_histogram_start = np.zeros(hist_bin_num)
    
    averaged_xc_start = (own_xc*image_bin_size)
    averaged_yc_start = (own_yc*image_bin_size)
    averaged_frame_start = own_frame
    averaged_angle_start = own_angle
    averaged_points_start = (own_points)
    
    # repeat for x_end and y_end, only difference is frame+1: start
    
    own_xc = STORM_npdata['x_end']/image_bin_size
    own_yc = STORM_npdata['y_end']/image_bin_size
    own_frame = STORM_npdata['frame']+1
    own_angle = 100
    own_x_inter = own_xc.astype(np.int16)
    own_y_inter = own_yc.astype(np.int16)
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map[own_x_inter-1,own_y_inter-1,0:counter_map[own_x_inter-1,own_y_inter-1]],
                                      indice_map[own_x_inter,own_y_inter-1,0:counter_map[own_x_inter,own_y_inter-1]],
                                      indice_map[own_x_inter+1,own_y_inter-1,0:counter_map[own_x_inter+1,own_y_inter-1]],
                                      indice_map[own_x_inter-1,own_y_inter,0:counter_map[own_x_inter-1,own_y_inter]],
                                      indice_map[own_x_inter,own_y_inter,0:counter_map[own_x_inter,own_y_inter]],
                                      indice_map[own_x_inter+1,own_y_inter,0:counter_map[own_x_inter+1,own_y_inter]],
                                      indice_map[own_x_inter-1,own_y_inter+1,0:counter_map[own_x_inter-1,own_y_inter+1]],
                                      indice_map[own_x_inter,own_y_inter+1,0:counter_map[own_x_inter,own_y_inter+1]],
                                      indice_map[own_x_inter+1,own_y_inter+1,0:counter_map[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc['yc'][indice_foraverage]
    foraverage_disp = xinter_yinter_frame_disp_angle_xcyc['disp'][indice_foraverage]
    foraverage_distance = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))
    foraverage_distance_bool = foraverage_distance<0.5
    foraverage_disp_final = foraverage_disp[foraverage_distance_bool]
    foraverage_distance_final = foraverage_distance[foraverage_distance_bool]
    own_points = np.size(foraverage_disp_final)
    
    if own_points>min_points_percell:
        
        foraverage_angle_final = xinter_yinter_frame_disp_angle_xcyc['angle'][indice_foraverage][foraverage_distance_bool]
        disp_y = foraverage_disp_final*np.sin(foraverage_angle_final)
        disp_x = foraverage_disp_final*np.cos(foraverage_angle_final)
        disp_x_y = np.transpose(np.vstack((disp_x,disp_y)))
        pca = PCA(n_components=1).fit(disp_x_y)
        comp_extracted = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        weights_bleed = np.floor(e**(foraverage_distance_final*foraverage_distance_final/-2/sigma/sigma)/sigma/sqrt(2*pi)*100)
        if explained_variance_ratio[0] > pca_ratio_thre:
            
            preferred_angle = atan2(comp_extracted[0,1],comp_extracted[0,0])
            own_angle = preferred_angle
            foraverage_projected_disp = foraverage_disp_final*np.cos(foraverage_angle_final-preferred_angle)
            
            averaged_histogram_end,_ = np.histogram(foraverage_projected_disp, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre),
                                  weights=weights_bleed)
        else:
            
            averaged_histogram_end,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (0,disp_thre),
                                  weights=weights_bleed)
        
    else:
        averaged_histogram_end = np.zeros(hist_bin_num)
    
    averaged_xc_end = (own_xc*image_bin_size)
    averaged_yc_end = (own_yc*image_bin_size)
    averaged_frame_end = own_frame
    averaged_angle_end = own_angle
    averaged_points_end = (own_points)
    
    # repeat for x_end and y_end, only difference is frame+1: finished
    
    return_array = np.hstack((averaged_xc_start,averaged_yc_start,averaged_frame_start,
                             averaged_angle_start,averaged_points_start,averaged_histogram_start,
                             averaged_xc_end,averaged_yc_end,averaged_frame_end,
                            averaged_angle_end,averaged_points_end,averaged_histogram_end))
    
    return return_array
    
def hist_maker_bleed_keepZcI(STORM_npdata,disp_thre,hist_bin_num,image_bin_size,
               xinter_yinter_frame_disp_angle_xcyc,
               indice_map,counter_map,xinter_max,yinter_max,min_points_percell,
               pca_ratio_thre):
    
    FWHM = 0.5
    sigma = FWHM/2.3548
    own_xc = STORM_npdata['x_start']/image_bin_size
    own_yc = STORM_npdata['y_start']/image_bin_size
    own_frame = STORM_npdata['frame']
    own_angle = 100
    own_x_inter = own_xc.astype(np.int16)
    own_y_inter = own_yc.astype(np.int16)
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map[own_x_inter-1,own_y_inter-1,0:counter_map[own_x_inter-1,own_y_inter-1]],
                                      indice_map[own_x_inter,own_y_inter-1,0:counter_map[own_x_inter,own_y_inter-1]],
                                      indice_map[own_x_inter+1,own_y_inter-1,0:counter_map[own_x_inter+1,own_y_inter-1]],
                                      indice_map[own_x_inter-1,own_y_inter,0:counter_map[own_x_inter-1,own_y_inter]],
                                      indice_map[own_x_inter,own_y_inter,0:counter_map[own_x_inter,own_y_inter]],
                                      indice_map[own_x_inter+1,own_y_inter,0:counter_map[own_x_inter+1,own_y_inter]],
                                      indice_map[own_x_inter-1,own_y_inter+1,0:counter_map[own_x_inter-1,own_y_inter+1]],
                                      indice_map[own_x_inter,own_y_inter+1,0:counter_map[own_x_inter,own_y_inter+1]],
                                      indice_map[own_x_inter+1,own_y_inter+1,0:counter_map[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc['yc'][indice_foraverage]
    foraverage_disp = xinter_yinter_frame_disp_angle_xcyc['disp'][indice_foraverage]
    foraverage_distance = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))
    foraverage_distance_bool = foraverage_distance<0.5
    foraverage_disp_final = foraverage_disp[foraverage_distance_bool]
    foraverage_distance_final = foraverage_distance[foraverage_distance_bool]
    own_points = np.size(foraverage_disp_final)
    
    if own_points>min_points_percell:
        
        foraverage_angle_final = xinter_yinter_frame_disp_angle_xcyc['angle'][indice_foraverage][foraverage_distance_bool]
        disp_y = foraverage_disp_final*np.sin(foraverage_angle_final)
        disp_x = foraverage_disp_final*np.cos(foraverage_angle_final)
        disp_x_y = np.transpose(np.vstack((disp_x,disp_y)))
        pca = PCA(n_components=1).fit(disp_x_y)
        comp_extracted = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        weights_bleed = np.floor(e**(foraverage_distance_final*foraverage_distance_final/-2/sigma/sigma)/sigma/sqrt(2*pi)*100)
        if explained_variance_ratio[0] > pca_ratio_thre:
            
            preferred_angle = atan2(comp_extracted[0,1],comp_extracted[0,0])
            own_angle = preferred_angle
            foraverage_projected_disp = foraverage_disp_final*np.cos(foraverage_angle_final-preferred_angle)
            
            averaged_histogram_start,_ = np.histogram(foraverage_projected_disp, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre),
                                  weights=weights_bleed)
        else:
            
            averaged_histogram_start,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (0,disp_thre),
                                  weights=weights_bleed)
        
    else:
        averaged_histogram_start = np.zeros(hist_bin_num)
    
    # data_type = np.dtype([('x_inter', np.int16),('y_inter', np.int16),('frame', np.int32),
    #                       ('disp', np.float32), ('angle', np.float32), 
    #                       ('xc', np.float32), ('yc', np.float32)],
    #                      ('RawZc', np.float32),('I', np.float32))
    
    # saved_data_type = np.dtype([('xc', np.float32), ('yc', np.float32),('frame', np.int32),
    #                             ('diffusion', np.float32), ('angle', np.float32), ('points', np.int32)])
    
    # data_type_STORMnp = np.dtype([('x_start', np.float32), ('y_start', np.float32),
    #                               ('x_end', np.float32), ('y_end', np.float32),
    #                               ('frame', np.int32),('disp', np.float32), ('angle', np.float32),
    #                               ('RawZc_start', np.float32), ('RawZc_end', np.float32),
    #                               ('I_start', np.float32), ('I_end', np.float32)])
    
    averaged_xc_start = (own_xc*image_bin_size)
    averaged_yc_start = (own_yc*image_bin_size)
    averaged_frame_start = own_frame
    averaged_angle_start = own_angle
    averaged_points_start = (own_points)
    averaged_RawZc_start = STORM_npdata['RawZc_start']
    averaged_I_start = STORM_npdata['I_start']
    
    # repeat for x_end and y_end, only difference is frame+1: start
    
    own_xc = STORM_npdata['x_end']/image_bin_size
    own_yc = STORM_npdata['y_end']/image_bin_size
    own_frame = STORM_npdata['frame']+1
    own_angle = 100
    own_x_inter = own_xc.astype(np.int16)
    own_y_inter = own_yc.astype(np.int16)
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map[own_x_inter-1,own_y_inter-1,0:counter_map[own_x_inter-1,own_y_inter-1]],
                                      indice_map[own_x_inter,own_y_inter-1,0:counter_map[own_x_inter,own_y_inter-1]],
                                      indice_map[own_x_inter+1,own_y_inter-1,0:counter_map[own_x_inter+1,own_y_inter-1]],
                                      indice_map[own_x_inter-1,own_y_inter,0:counter_map[own_x_inter-1,own_y_inter]],
                                      indice_map[own_x_inter,own_y_inter,0:counter_map[own_x_inter,own_y_inter]],
                                      indice_map[own_x_inter+1,own_y_inter,0:counter_map[own_x_inter+1,own_y_inter]],
                                      indice_map[own_x_inter-1,own_y_inter+1,0:counter_map[own_x_inter-1,own_y_inter+1]],
                                      indice_map[own_x_inter,own_y_inter+1,0:counter_map[own_x_inter,own_y_inter+1]],
                                      indice_map[own_x_inter+1,own_y_inter+1,0:counter_map[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc['yc'][indice_foraverage]
    foraverage_disp = xinter_yinter_frame_disp_angle_xcyc['disp'][indice_foraverage]
    foraverage_distance = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))
    foraverage_distance_bool = foraverage_distance<0.5
    foraverage_disp_final = foraverage_disp[foraverage_distance_bool]
    foraverage_distance_final = foraverage_distance[foraverage_distance_bool]
    own_points = np.size(foraverage_disp_final)
    
    if own_points>min_points_percell:
        
        foraverage_angle_final = xinter_yinter_frame_disp_angle_xcyc['angle'][indice_foraverage][foraverage_distance_bool]
        disp_y = foraverage_disp_final*np.sin(foraverage_angle_final)
        disp_x = foraverage_disp_final*np.cos(foraverage_angle_final)
        disp_x_y = np.transpose(np.vstack((disp_x,disp_y)))
        pca = PCA(n_components=1).fit(disp_x_y)
        comp_extracted = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        weights_bleed = np.floor(e**(foraverage_distance_final*foraverage_distance_final/-2/sigma/sigma)/sigma/sqrt(2*pi)*100)
        if explained_variance_ratio[0] > pca_ratio_thre:
            
            preferred_angle = atan2(comp_extracted[0,1],comp_extracted[0,0])
            own_angle = preferred_angle
            foraverage_projected_disp = foraverage_disp_final*np.cos(foraverage_angle_final-preferred_angle)
            
            averaged_histogram_end,_ = np.histogram(foraverage_projected_disp, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre),
                                  weights=weights_bleed)
        else:
            
            averaged_histogram_end,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (0,disp_thre),
                                  weights=weights_bleed)
        
    else:
        averaged_histogram_end = np.zeros(hist_bin_num)
    
    averaged_xc_end = (own_xc*image_bin_size)
    averaged_yc_end = (own_yc*image_bin_size)
    averaged_frame_end = own_frame
    averaged_angle_end = own_angle
    averaged_points_end = (own_points)
    averaged_RawZc_end = STORM_npdata['RawZc_end']
    averaged_I_end = STORM_npdata['I_end']
    
    # repeat for x_end and y_end, only difference is frame+1: finished
    
    return_array = np.hstack((averaged_xc_start,averaged_yc_start,averaged_frame_start,
                             averaged_angle_start,averaged_points_start,averaged_histogram_start,
                             averaged_xc_end,averaged_yc_end,averaged_frame_end,
                            averaged_angle_end,averaged_points_end,averaged_histogram_end,
                            averaged_RawZc_start,averaged_RawZc_end,averaged_I_start,averaged_I_end))
    
    return return_array

def hist_maker_Z(STORM_npdata,disp_thre,hist_bin_num,image_bin_size,
               xinter_yinter_frame_disp_angle_xcyc,
               indice_map,counter_map,xinter_max,yinter_max,min_points_percell,
               pca_ratio_thre):
    
    FWHM = 0.5
    sigma = FWHM/2.3548
    own_xc = STORM_npdata['x_start']/image_bin_size
    own_yc = STORM_npdata['y_start']/image_bin_size
    own_frame = STORM_npdata['frame']
    own_angle = 100
    own_x_inter = own_xc.astype(np.int16)
    own_y_inter = own_yc.astype(np.int16)
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map[own_x_inter-1,own_y_inter-1,0:counter_map[own_x_inter-1,own_y_inter-1]],
                                      indice_map[own_x_inter,own_y_inter-1,0:counter_map[own_x_inter,own_y_inter-1]],
                                      indice_map[own_x_inter+1,own_y_inter-1,0:counter_map[own_x_inter+1,own_y_inter-1]],
                                      indice_map[own_x_inter-1,own_y_inter,0:counter_map[own_x_inter-1,own_y_inter]],
                                      indice_map[own_x_inter,own_y_inter,0:counter_map[own_x_inter,own_y_inter]],
                                      indice_map[own_x_inter+1,own_y_inter,0:counter_map[own_x_inter+1,own_y_inter]],
                                      indice_map[own_x_inter-1,own_y_inter+1,0:counter_map[own_x_inter-1,own_y_inter+1]],
                                      indice_map[own_x_inter,own_y_inter+1,0:counter_map[own_x_inter,own_y_inter+1]],
                                      indice_map[own_x_inter+1,own_y_inter+1,0:counter_map[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc['yc'][indice_foraverage]
    foraverage_disp = xinter_yinter_frame_disp_angle_xcyc['disp'][indice_foraverage]
    foraverage_disp_xy = xinter_yinter_frame_disp_angle_xcyc['disp_xy'][indice_foraverage]
    foraverage_distance = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))
    foraverage_distance_bool = foraverage_distance<0.5
    foraverage_disp_final = foraverage_disp[foraverage_distance_bool]
    foraverage_disp_xy_final = foraverage_disp_xy[foraverage_distance_bool]
    foraverage_distance_final = foraverage_distance[foraverage_distance_bool]
    own_points = np.size(foraverage_disp_final)
    
    if own_points>min_points_percell:
        
        foraverage_angle_final = xinter_yinter_frame_disp_angle_xcyc['angle'][indice_foraverage][foraverage_distance_bool]
        disp_y = foraverage_disp_xy_final*np.sin(foraverage_angle_final)
        disp_x = foraverage_disp_xy_final*np.cos(foraverage_angle_final)
        disp_x_y = np.transpose(np.vstack((disp_x,disp_y)))
        pca = PCA(n_components=1).fit(disp_x_y)
        comp_extracted = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        if explained_variance_ratio[0] > pca_ratio_thre:
            
            preferred_angle = atan2(comp_extracted[0,1],comp_extracted[0,0])
            own_angle = preferred_angle
            # foraverage_projected_disp = foraverage_disp_final*np.cos(foraverage_angle_final-preferred_angle)
            
            averaged_histogram_start,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre))
        else:
            
            averaged_histogram_start,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre))
        
    else:
        averaged_histogram_start = np.zeros(hist_bin_num)
    
    averaged_xc_start = (own_xc*image_bin_size)
    averaged_yc_start = (own_yc*image_bin_size)
    averaged_frame_start = own_frame
    averaged_angle_start = own_angle
    averaged_points_start = (own_points)
    
    # repeat for x_end and y_end, only difference is frame+1: start
    
    own_xc = STORM_npdata['x_end']/image_bin_size
    own_yc = STORM_npdata['y_end']/image_bin_size
    own_frame = STORM_npdata['frame']+1
    own_angle = 100
    own_x_inter = own_xc.astype(np.int16)
    own_y_inter = own_yc.astype(np.int16)
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map[own_x_inter-1,own_y_inter-1,0:counter_map[own_x_inter-1,own_y_inter-1]],
                                      indice_map[own_x_inter,own_y_inter-1,0:counter_map[own_x_inter,own_y_inter-1]],
                                      indice_map[own_x_inter+1,own_y_inter-1,0:counter_map[own_x_inter+1,own_y_inter-1]],
                                      indice_map[own_x_inter-1,own_y_inter,0:counter_map[own_x_inter-1,own_y_inter]],
                                      indice_map[own_x_inter,own_y_inter,0:counter_map[own_x_inter,own_y_inter]],
                                      indice_map[own_x_inter+1,own_y_inter,0:counter_map[own_x_inter+1,own_y_inter]],
                                      indice_map[own_x_inter-1,own_y_inter+1,0:counter_map[own_x_inter-1,own_y_inter+1]],
                                      indice_map[own_x_inter,own_y_inter+1,0:counter_map[own_x_inter,own_y_inter+1]],
                                      indice_map[own_x_inter+1,own_y_inter+1,0:counter_map[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc['yc'][indice_foraverage]
    foraverage_disp = xinter_yinter_frame_disp_angle_xcyc['disp'][indice_foraverage]
    foraverage_disp_xy = xinter_yinter_frame_disp_angle_xcyc['disp_xy'][indice_foraverage]
    foraverage_distance = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))
    foraverage_distance_bool = foraverage_distance<0.5
    foraverage_disp_final = foraverage_disp[foraverage_distance_bool]
    foraverage_disp_xy_final = foraverage_disp_xy[foraverage_distance_bool]
    foraverage_distance_final = foraverage_distance[foraverage_distance_bool]
    own_points = np.size(foraverage_disp_final)
    
    if own_points>min_points_percell:
        
        foraverage_angle_final = xinter_yinter_frame_disp_angle_xcyc['angle'][indice_foraverage][foraverage_distance_bool]
        disp_y = foraverage_disp_xy_final*np.sin(foraverage_angle_final)
        disp_x = foraverage_disp_xy_final*np.cos(foraverage_angle_final)
        disp_x_y = np.transpose(np.vstack((disp_x,disp_y)))
        pca = PCA(n_components=1).fit(disp_x_y)
        comp_extracted = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        weights_bleed = np.floor(e**(foraverage_distance_final*foraverage_distance_final/-2/sigma/sigma)/sigma/sqrt(2*pi)*100)
        if explained_variance_ratio[0] > pca_ratio_thre:
            
            preferred_angle = atan2(comp_extracted[0,1],comp_extracted[0,0])
            own_angle = preferred_angle
            # foraverage_projected_disp = foraverage_disp_final*np.cos(foraverage_angle_final-preferred_angle)
            
            averaged_histogram_end,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre),
                                  weights=weights_bleed)
        else:
            
            averaged_histogram_end,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre),
                                  weights=weights_bleed)
        
    else:
        averaged_histogram_end = np.zeros(hist_bin_num)
    
    averaged_xc_end = (own_xc*image_bin_size)
    averaged_yc_end = (own_yc*image_bin_size)
    averaged_frame_end = own_frame
    averaged_angle_end = own_angle
    averaged_points_end = (own_points)
    
    # repeat for x_end and y_end, only difference is frame+1: finished
    
    return_array = np.hstack((averaged_xc_start,averaged_yc_start,averaged_frame_start,
                             averaged_angle_start,averaged_points_start,averaged_histogram_start,
                             averaged_xc_end,averaged_yc_end,averaged_frame_end,
                            averaged_angle_end,averaged_points_end,averaged_histogram_end))
    
    return return_array

def hist_maker_Z_Bleed(STORM_npdata,disp_thre,hist_bin_num,image_bin_size,
               xinter_yinter_frame_disp_angle_xcyc,
               indice_map,counter_map,xinter_max,yinter_max,min_points_percell,
               pca_ratio_thre):
    
    FWHM = 0.5
    sigma = FWHM/2.3548
    own_xc = STORM_npdata['x_start']/image_bin_size
    own_yc = STORM_npdata['y_start']/image_bin_size
    own_frame = STORM_npdata['frame']
    own_angle = 100
    own_x_inter = own_xc.astype(np.int16)
    own_y_inter = own_yc.astype(np.int16)
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map[own_x_inter-1,own_y_inter-1,0:counter_map[own_x_inter-1,own_y_inter-1]],
                                      indice_map[own_x_inter,own_y_inter-1,0:counter_map[own_x_inter,own_y_inter-1]],
                                      indice_map[own_x_inter+1,own_y_inter-1,0:counter_map[own_x_inter+1,own_y_inter-1]],
                                      indice_map[own_x_inter-1,own_y_inter,0:counter_map[own_x_inter-1,own_y_inter]],
                                      indice_map[own_x_inter,own_y_inter,0:counter_map[own_x_inter,own_y_inter]],
                                      indice_map[own_x_inter+1,own_y_inter,0:counter_map[own_x_inter+1,own_y_inter]],
                                      indice_map[own_x_inter-1,own_y_inter+1,0:counter_map[own_x_inter-1,own_y_inter+1]],
                                      indice_map[own_x_inter,own_y_inter+1,0:counter_map[own_x_inter,own_y_inter+1]],
                                      indice_map[own_x_inter+1,own_y_inter+1,0:counter_map[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc['yc'][indice_foraverage]
    foraverage_disp = xinter_yinter_frame_disp_angle_xcyc['disp'][indice_foraverage]
    foraverage_disp_xy = xinter_yinter_frame_disp_angle_xcyc['disp_xy'][indice_foraverage]
    foraverage_distance = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))
    foraverage_distance_bool = foraverage_distance<0.5
    foraverage_disp_final = foraverage_disp[foraverage_distance_bool]
    foraverage_disp_xy_final = foraverage_disp_xy[foraverage_distance_bool]
    foraverage_distance_final = foraverage_distance[foraverage_distance_bool]
    own_points = np.size(foraverage_disp_final)
    weights_bleed = np.floor(e**(foraverage_distance_final*foraverage_distance_final/-2/sigma/sigma)/sigma/sqrt(2*pi)*100)
    
    if own_points>min_points_percell:
        
        foraverage_angle_final = xinter_yinter_frame_disp_angle_xcyc['angle'][indice_foraverage][foraverage_distance_bool]
        disp_y = foraverage_disp_xy_final*np.sin(foraverage_angle_final)
        disp_x = foraverage_disp_xy_final*np.cos(foraverage_angle_final)
        disp_x_y = np.transpose(np.vstack((disp_x,disp_y)))
        pca = PCA(n_components=1).fit(disp_x_y)
        comp_extracted = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        if explained_variance_ratio[0] > pca_ratio_thre:
            
            preferred_angle = atan2(comp_extracted[0,1],comp_extracted[0,0])
            own_angle = preferred_angle
            # foraverage_projected_disp = foraverage_disp_final*np.cos(foraverage_angle_final-preferred_angle)
            
            averaged_histogram_start,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre),
                                  weights=weights_bleed)
        else:
            
            averaged_histogram_start,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre),
                                  weights=weights_bleed)
        
    else:
        averaged_histogram_start = np.zeros(hist_bin_num)
    
    averaged_xc_start = (own_xc*image_bin_size)
    averaged_yc_start = (own_yc*image_bin_size)
    averaged_frame_start = own_frame
    averaged_angle_start = own_angle
    averaged_points_start = (own_points)
    
    # repeat for x_end and y_end, only difference is frame+1: start
    
    own_xc = STORM_npdata['x_end']/image_bin_size
    own_yc = STORM_npdata['y_end']/image_bin_size
    own_frame = STORM_npdata['frame']+1
    own_angle = 100
    own_x_inter = own_xc.astype(np.int16)
    own_y_inter = own_yc.astype(np.int16)
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map[own_x_inter-1,own_y_inter-1,0:counter_map[own_x_inter-1,own_y_inter-1]],
                                      indice_map[own_x_inter,own_y_inter-1,0:counter_map[own_x_inter,own_y_inter-1]],
                                      indice_map[own_x_inter+1,own_y_inter-1,0:counter_map[own_x_inter+1,own_y_inter-1]],
                                      indice_map[own_x_inter-1,own_y_inter,0:counter_map[own_x_inter-1,own_y_inter]],
                                      indice_map[own_x_inter,own_y_inter,0:counter_map[own_x_inter,own_y_inter]],
                                      indice_map[own_x_inter+1,own_y_inter,0:counter_map[own_x_inter+1,own_y_inter]],
                                      indice_map[own_x_inter-1,own_y_inter+1,0:counter_map[own_x_inter-1,own_y_inter+1]],
                                      indice_map[own_x_inter,own_y_inter+1,0:counter_map[own_x_inter,own_y_inter+1]],
                                      indice_map[own_x_inter+1,own_y_inter+1,0:counter_map[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc['yc'][indice_foraverage]
    foraverage_disp = xinter_yinter_frame_disp_angle_xcyc['disp'][indice_foraverage]
    foraverage_disp_xy = xinter_yinter_frame_disp_angle_xcyc['disp_xy'][indice_foraverage]
    foraverage_distance = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))
    foraverage_distance_bool = foraverage_distance<0.5
    foraverage_disp_final = foraverage_disp[foraverage_distance_bool]
    foraverage_disp_xy_final = foraverage_disp_xy[foraverage_distance_bool]
    foraverage_distance_final = foraverage_distance[foraverage_distance_bool]
    own_points = np.size(foraverage_disp_final)
    weights_bleed = np.floor(e**(foraverage_distance_final*foraverage_distance_final/-2/sigma/sigma)/sigma/sqrt(2*pi)*100)
    
    if own_points>min_points_percell:
        
        foraverage_angle_final = xinter_yinter_frame_disp_angle_xcyc['angle'][indice_foraverage][foraverage_distance_bool]
        disp_y = foraverage_disp_xy_final*np.sin(foraverage_angle_final)
        disp_x = foraverage_disp_xy_final*np.cos(foraverage_angle_final)
        disp_x_y = np.transpose(np.vstack((disp_x,disp_y)))
        pca = PCA(n_components=1).fit(disp_x_y)
        comp_extracted = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        weights_bleed = np.floor(e**(foraverage_distance_final*foraverage_distance_final/-2/sigma/sigma)/sigma/sqrt(2*pi)*100)
        if explained_variance_ratio[0] > pca_ratio_thre:
            
            preferred_angle = atan2(comp_extracted[0,1],comp_extracted[0,0])
            own_angle = preferred_angle
            # foraverage_projected_disp = foraverage_disp_final*np.cos(foraverage_angle_final-preferred_angle)
            
            averaged_histogram_end,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre),
                                  weights=weights_bleed)
        else:
            
            averaged_histogram_end,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre),
                                  weights=weights_bleed)
        
    else:
        averaged_histogram_end = np.zeros(hist_bin_num)
    
    averaged_xc_end = (own_xc*image_bin_size)
    averaged_yc_end = (own_yc*image_bin_size)
    averaged_frame_end = own_frame
    averaged_angle_end = own_angle
    averaged_points_end = (own_points)
    
    # repeat for x_end and y_end, only difference is frame+1: finished
    
    return_array = np.hstack((averaged_xc_start,averaged_yc_start,averaged_frame_start,
                             averaged_angle_start,averaged_points_start,averaged_histogram_start,
                             averaged_xc_end,averaged_yc_end,averaged_frame_end,
                            averaged_angle_end,averaged_points_end,averaged_histogram_end))
    
    return return_array

def hist_maker_radial(excel,disp_thre,hist_bin_num,image_bin_size,
               xinter_yinter_frame_disp_angle_xcyc,
               indice_map,counter_map,xinter_max,yinter_max,min_points_percell,
               pca_ratio_thre,bin_num_radial,pixel_size,time_interval,fixed_D_fast,
               fixed_D_fast_torr):

    own_xc = excel[1]/image_bin_size
    own_yc = excel[3]/image_bin_size
    own_x_inter = own_xc.astype(np.int16)
    own_y_inter = own_yc.astype(np.int16)
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map[own_x_inter-1,own_y_inter-1,0:counter_map[own_x_inter-1,own_y_inter-1]],
                                      indice_map[own_x_inter,own_y_inter-1,0:counter_map[own_x_inter,own_y_inter-1]],
                                      indice_map[own_x_inter+1,own_y_inter-1,0:counter_map[own_x_inter+1,own_y_inter-1]],
                                      indice_map[own_x_inter-1,own_y_inter,0:counter_map[own_x_inter-1,own_y_inter]],
                                      indice_map[own_x_inter,own_y_inter,0:counter_map[own_x_inter,own_y_inter]],
                                      indice_map[own_x_inter+1,own_y_inter,0:counter_map[own_x_inter+1,own_y_inter]],
                                      indice_map[own_x_inter-1,own_y_inter+1,0:counter_map[own_x_inter-1,own_y_inter+1]],
                                      indice_map[own_x_inter,own_y_inter+1,0:counter_map[own_x_inter,own_y_inter+1]],
                                      indice_map[own_x_inter+1,own_y_inter+1,0:counter_map[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc['yc'][indice_foraverage]
    foraverage_disp = xinter_yinter_frame_disp_angle_xcyc['disp'][indice_foraverage]
    
    #================注意：image_bin_size起作用，纳入考虑范围的点======================
    foraverage_distance = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))
    foraverage_distance_bool = foraverage_distance<0.5
    foraverage_distance_final = foraverage_distance[foraverage_distance_bool]*image_bin_size
    foraverage_disp_final = foraverage_disp[foraverage_distance_bool]
    distance_max = image_bin_size/2
    distance_min = 0
    distance_step = (distance_max-distance_min)/bin_num_radial
    ranges = [(range_i, range_i + distance_step) for range_i in np.arange(distance_min, distance_max, distance_step)]
    drate_radius_binx = np.arange(0.5*distance_step, distance_max, distance_step, dtype=np.float32)
    
    drate_start = np.zeros(bin_num_radial)
    range_i=0
    for range_i, (r_min, r_max) in enumerate(ranges):
        
    # 找出在该范围内的索引
        mask = (foraverage_distance_final >= r_min) & (foraverage_distance_final < r_max)
        filtered_disp = foraverage_disp_final[mask]
    
        # 跳过空数据段
        if len(filtered_disp) > min_points_percell:
    
            hist_start,_ = np.histogram(filtered_disp,
                                        bins = hist_bin_num, range = (0,disp_thre))
            hist_bin_size = disp_thre/hist_bin_num 
            bins_x = np.arange(0.5*hist_bin_size, disp_thre, hist_bin_size, dtype=np.float32)
            
            # for guessed initial fitting parameters
            bin_thre = int(floor((disp_thre*0.6)/hist_bin_size)) 
                    
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
                popt[0] = -1
            
            drate_start[range_i] = popt[0] * (pixel_size/1000)**2 /4/time_interval
                
        else:
            hist_start = np.zeros(hist_bin_num)
            drate_start[range_i] = 0
    
    drate_mask = drate_start>0
    filter_drate = drate_start[drate_mask]
    filter_radius_binx = drate_radius_binx[drate_mask]
    
    initial_params = [fixed_D_fast/2, 0, 0.5, fixed_D_fast]
    
    if len(filter_radius_binx) > 0:
        # 进行曲线拟合
        # 设定初始参数：a1, x1, w1, y0
        
        # 设定参数范围，限制 x1 在 [0, 390]，x2 在 [410, 800]
        param_bounds = ([0, -0.1, 0, fixed_D_fast-fixed_D_fast_torr],  # 最小值
                        [fixed_D_fast, 0.1, 10, fixed_D_fast+fixed_D_fast_torr])  # 最大值
    
        # 进行曲线拟合
        params, _ = curve_fit(gaussian_with_background_invert, filter_radius_binx, filter_drate,
                              p0=initial_params, bounds=param_bounds)
    
        # params, _ = curve_fit(gaussian_with_background, bin_xaxis_plot_Fit, drate_array_Fit, p0=initial_params)
        Fit_result = np.array(params)  # 存储拟合参数
        
        # def gaussian_with_background_invert(x, a1, x1, w1, y0):
            
        #     return (a1-y0) * np.exp(-((x - x1) ** 2) / (2 * w1 ** 2)) + y0
        
    else:
        
        Fit_result = [0,0,0,0]
        
    
    averaged_xc_start = excel[1]
    averaged_yc_start = excel[3]
    averaged_points_start = excel[7]
    averaged_zc_start = excel[5]
    fitted_zc_start = Fit_result[0]
    fitted_diameter = Fit_result[2]*2.355*pixel_size
    fitted_zc_bg = Fit_result[3]
    averaged_frame = excel[8]
    
    # repeat for x_end and y_end, only difference is frame+1: finished
    
    return_array = np.hstack((averaged_xc_start,averaged_yc_start,averaged_zc_start,
                              averaged_points_start,fitted_zc_start,fitted_diameter,fitted_zc_bg,
                              averaged_frame))
    
    return return_array
    
def hist_maker_bleed_HighDen(STORM_npdata,disp_thre,hist_bin_num,image_bin_size,
               xinter_yinter_frame_disp_angle_xcyc,
               indice_map,counter_map,xinter_max,yinter_max,min_points_percell,
               pca_ratio_thre,HighDen_thre,frame_num):
    
    FWHM = 0.5
    sigma = FWHM/2.3548
    own_xc = STORM_npdata['x_start']/image_bin_size
    own_yc = STORM_npdata['y_start']/image_bin_size
    own_frame = STORM_npdata['frame']
    own_angle = 100
    own_x_inter = own_xc.astype(np.int16)
    own_y_inter = own_yc.astype(np.int16)
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map[own_x_inter-1,own_y_inter-1,0:counter_map[own_x_inter-1,own_y_inter-1]],
                                      indice_map[own_x_inter,own_y_inter-1,0:counter_map[own_x_inter,own_y_inter-1]],
                                      indice_map[own_x_inter+1,own_y_inter-1,0:counter_map[own_x_inter+1,own_y_inter-1]],
                                      indice_map[own_x_inter-1,own_y_inter,0:counter_map[own_x_inter-1,own_y_inter]],
                                      indice_map[own_x_inter,own_y_inter,0:counter_map[own_x_inter,own_y_inter]],
                                      indice_map[own_x_inter+1,own_y_inter,0:counter_map[own_x_inter+1,own_y_inter]],
                                      indice_map[own_x_inter-1,own_y_inter+1,0:counter_map[own_x_inter-1,own_y_inter+1]],
                                      indice_map[own_x_inter,own_y_inter+1,0:counter_map[own_x_inter,own_y_inter+1]],
                                      indice_map[own_x_inter+1,own_y_inter+1,0:counter_map[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc['yc'][indice_foraverage]
    foraverage_disp = xinter_yinter_frame_disp_angle_xcyc['disp'][indice_foraverage]
    foraverage_frame = xinter_yinter_frame_disp_angle_xcyc['frame'][indice_foraverage]
    foraverage_angle = xinter_yinter_frame_disp_angle_xcyc['angle'][indice_foraverage]
    foraverage_distance = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))
    foraverage_distance_bool = foraverage_distance<0.5
    foraverage_disp_med = foraverage_disp[foraverage_distance_bool]
    foraverage_distance_med = foraverage_distance[foraverage_distance_bool]
    foraverage_frame_med = foraverage_frame[foraverage_distance_bool]
    foraverage_angle_med = foraverage_angle[foraverage_distance_bool]
    
    own_points_med = np.size(foraverage_disp_med)
    
    # fitler if the density is too high
    if own_points_med>HighDen_thre:
    
        frame_ratio = HighDen_thre/own_points_med
        frame_bool = (foraverage_frame_med > (frame_num*(1-frame_ratio)))
        foraverage_disp_final = foraverage_disp_med[frame_bool]
        foraverage_distance_final = foraverage_distance_med[frame_bool]
        foraverage_angle_final = foraverage_angle_med[frame_bool]
        
    else:
    
        foraverage_disp_final = foraverage_disp_med
        foraverage_distance_final = foraverage_distance_med
        foraverage_angle_final = foraverage_angle_med
    
    own_points = np.size(foraverage_disp_final)
    
    if own_points>min_points_percell:
        
        disp_y = foraverage_disp_final*np.sin(foraverage_angle_final)
        disp_x = foraverage_disp_final*np.cos(foraverage_angle_final)
        disp_x_y = np.transpose(np.vstack((disp_x,disp_y)))
        pca = PCA(n_components=1).fit(disp_x_y)
        comp_extracted = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        weights_bleed = np.floor(e**(foraverage_distance_final*foraverage_distance_final/-2/sigma/sigma)/sigma/sqrt(2*pi)*100)
        if explained_variance_ratio[0] > pca_ratio_thre:
            
            preferred_angle = atan2(comp_extracted[0,1],comp_extracted[0,0])
            own_angle = preferred_angle
            foraverage_projected_disp = foraverage_disp_final*np.cos(foraverage_angle_final-preferred_angle)
            
            averaged_histogram_start,_ = np.histogram(foraverage_projected_disp, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre),
                                  weights=weights_bleed)
        else:
            
            averaged_histogram_start,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (0,disp_thre),
                                  weights=weights_bleed)
        
    else:
        averaged_histogram_start = np.zeros(hist_bin_num)
    
    averaged_xc_start = (own_xc*image_bin_size)
    averaged_yc_start = (own_yc*image_bin_size)
    averaged_frame_start = own_frame
    averaged_angle_start = own_angle
    averaged_points_start = (own_points_med)
    
    # repeat for x_end and y_end, only difference is frame+1: start
    
    own_xc = STORM_npdata['x_end']/image_bin_size
    own_yc = STORM_npdata['y_end']/image_bin_size
    own_frame = STORM_npdata['frame']+1
    own_angle = 100
    own_x_inter = own_xc.astype(np.int16)
    own_y_inter = own_yc.astype(np.int16)
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map[own_x_inter-1,own_y_inter-1,0:counter_map[own_x_inter-1,own_y_inter-1]],
                                      indice_map[own_x_inter,own_y_inter-1,0:counter_map[own_x_inter,own_y_inter-1]],
                                      indice_map[own_x_inter+1,own_y_inter-1,0:counter_map[own_x_inter+1,own_y_inter-1]],
                                      indice_map[own_x_inter-1,own_y_inter,0:counter_map[own_x_inter-1,own_y_inter]],
                                      indice_map[own_x_inter,own_y_inter,0:counter_map[own_x_inter,own_y_inter]],
                                      indice_map[own_x_inter+1,own_y_inter,0:counter_map[own_x_inter+1,own_y_inter]],
                                      indice_map[own_x_inter-1,own_y_inter+1,0:counter_map[own_x_inter-1,own_y_inter+1]],
                                      indice_map[own_x_inter,own_y_inter+1,0:counter_map[own_x_inter,own_y_inter+1]],
                                      indice_map[own_x_inter+1,own_y_inter+1,0:counter_map[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc['yc'][indice_foraverage]
    foraverage_disp = xinter_yinter_frame_disp_angle_xcyc['disp'][indice_foraverage]
    foraverage_frame = xinter_yinter_frame_disp_angle_xcyc['frame'][indice_foraverage]
    foraverage_angle = xinter_yinter_frame_disp_angle_xcyc['angle'][indice_foraverage]
    foraverage_distance = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))
    foraverage_distance_bool = foraverage_distance<0.5
    foraverage_disp_med = foraverage_disp[foraverage_distance_bool]
    foraverage_distance_med = foraverage_distance[foraverage_distance_bool]
    foraverage_frame_med = foraverage_frame[foraverage_distance_bool]
    foraverage_angle_med = foraverage_angle[foraverage_distance_bool]
    
    own_points_med = np.size(foraverage_disp_med)
    
    # fitler if the density is too high
    if own_points_med>HighDen_thre:
    
        frame_ratio = HighDen_thre/own_points_med
        frame_bool = (foraverage_frame_med > (frame_num*(1-frame_ratio)))
        foraverage_disp_final = foraverage_disp_med[frame_bool]
        foraverage_distance_final = foraverage_distance_med[frame_bool]
        foraverage_angle_final = foraverage_angle_med[frame_bool]
    else:
        foraverage_disp_final = foraverage_disp_med
        foraverage_distance_final = foraverage_distance_med
        foraverage_angle_final = foraverage_angle_med
    
    own_points = np.size(foraverage_disp_final)
    
    if own_points>min_points_percell:
        
        disp_y = foraverage_disp_final*np.sin(foraverage_angle_final)
        disp_x = foraverage_disp_final*np.cos(foraverage_angle_final)
        disp_x_y = np.transpose(np.vstack((disp_x,disp_y)))
        pca = PCA(n_components=1).fit(disp_x_y)
        comp_extracted = pca.components_
        explained_variance_ratio = pca.explained_variance_ratio_
        weights_bleed = np.floor(e**(foraverage_distance_final*foraverage_distance_final/-2/sigma/sigma)/sigma/sqrt(2*pi)*100)
        if explained_variance_ratio[0] > pca_ratio_thre:
            
            preferred_angle = atan2(comp_extracted[0,1],comp_extracted[0,0])
            own_angle = preferred_angle
            foraverage_projected_disp = foraverage_disp_final*np.cos(foraverage_angle_final-preferred_angle)
            
            averaged_histogram_end,_ = np.histogram(foraverage_projected_disp, 
                                  bins = hist_bin_num, range = (-0.7*disp_thre,
                                                                0.7*disp_thre),
                                  weights=weights_bleed)
        else:
            
            averaged_histogram_end,_ = np.histogram(foraverage_disp_final, 
                                  bins = hist_bin_num, range = (0,disp_thre),
                                  weights=weights_bleed)
        
    else:
        averaged_histogram_end = np.zeros(hist_bin_num)
    
    averaged_xc_end = (own_xc*image_bin_size)
    averaged_yc_end = (own_yc*image_bin_size)
    averaged_frame_end = own_frame
    averaged_angle_end = own_angle
    averaged_points_end = (own_points_med)
    
    # repeat for x_end and y_end, only difference is frame+1: finished
    
    return_array = np.hstack((averaged_xc_start,averaged_yc_start,averaged_frame_start,
                             averaged_angle_start,averaged_points_start,averaged_histogram_start,
                             averaged_xc_end,averaged_yc_end,averaged_frame_end,
                            averaged_angle_end,averaged_points_end,averaged_histogram_end))
    
    return return_array

def hist_maker_x_y(STORM_npdata,disp_threX,disp_threY,hist_bin_numX,
                   hist_bin_numY,image_bin_size,
               xinter_yinter_frame_disp_angle_xcyc_odd,
               xinter_yinter_frame_disp_angle_xcyc_even,
               indice_map_odd,counter_map_odd,
               indice_map_even,counter_map_even,
               xinter_max,yinter_max,min_points_percell):
    
    # start point, odd dataset
    
    own_xc = STORM_npdata['x_start']/image_bin_size
    own_yc = STORM_npdata['y_start']/image_bin_size
    own_frame = STORM_npdata['frame']
    own_x_inter = own_xc.astype(np.int16)
    own_y_inter = own_yc.astype(np.int16)
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map_odd[own_x_inter-1,own_y_inter-1,0:counter_map_odd[own_x_inter-1,own_y_inter-1]],
                                      indice_map_odd[own_x_inter,own_y_inter-1,0:counter_map_odd[own_x_inter,own_y_inter-1]],
                                      indice_map_odd[own_x_inter+1,own_y_inter-1,0:counter_map_odd[own_x_inter+1,own_y_inter-1]],
                                      indice_map_odd[own_x_inter-1,own_y_inter,0:counter_map_odd[own_x_inter-1,own_y_inter]],
                                      indice_map_odd[own_x_inter,own_y_inter,0:counter_map_odd[own_x_inter,own_y_inter]],
                                      indice_map_odd[own_x_inter+1,own_y_inter,0:counter_map_odd[own_x_inter+1,own_y_inter]],
                                      indice_map_odd[own_x_inter-1,own_y_inter+1,0:counter_map_odd[own_x_inter-1,own_y_inter+1]],
                                      indice_map_odd[own_x_inter,own_y_inter+1,0:counter_map_odd[own_x_inter,own_y_inter+1]],
                                      indice_map_odd[own_x_inter+1,own_y_inter+1,0:counter_map_odd[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc_odd['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc_odd['yc'][indice_foraverage]
    foraverage_distance_bool = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))<0.5          
    own_points = np.sum(foraverage_distance_bool)
    
    if own_points>min_points_percell:
        
        foraverage_dispY_final = xinter_yinter_frame_disp_angle_xcyc_odd['angle'][indice_foraverage][foraverage_distance_bool]
        foraverage_dispX_final = xinter_yinter_frame_disp_angle_xcyc_odd['disp'][indice_foraverage][foraverage_distance_bool]
        averaged_histogram_start_oddX,_ = np.histogram(foraverage_dispX_final, 
                              bins = hist_bin_numX, range = (-disp_threX,disp_threX))
        averaged_histogram_start_oddY,_ = np.histogram(foraverage_dispY_final, 
                              bins = hist_bin_numY, range = (-disp_threY,disp_threY))
        
    else:
        averaged_histogram_start_oddX = np.zeros(hist_bin_numX)
        averaged_histogram_start_oddY = np.zeros(hist_bin_numY)
    
    averaged_xc_start = (own_xc*image_bin_size)
    averaged_yc_start = (own_yc*image_bin_size)
    averaged_frame_start = own_frame
    averaged_points_start_odd = (own_points)
    
    # start point, even dataset
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map_even[own_x_inter-1,own_y_inter-1,0:counter_map_even[own_x_inter-1,own_y_inter-1]],
                                      indice_map_even[own_x_inter,own_y_inter-1,0:counter_map_even[own_x_inter,own_y_inter-1]],
                                      indice_map_even[own_x_inter+1,own_y_inter-1,0:counter_map_even[own_x_inter+1,own_y_inter-1]],
                                      indice_map_even[own_x_inter-1,own_y_inter,0:counter_map_even[own_x_inter-1,own_y_inter]],
                                      indice_map_even[own_x_inter,own_y_inter,0:counter_map_even[own_x_inter,own_y_inter]],
                                      indice_map_even[own_x_inter+1,own_y_inter,0:counter_map_even[own_x_inter+1,own_y_inter]],
                                      indice_map_even[own_x_inter-1,own_y_inter+1,0:counter_map_even[own_x_inter-1,own_y_inter+1]],
                                      indice_map_even[own_x_inter,own_y_inter+1,0:counter_map_even[own_x_inter,own_y_inter+1]],
                                      indice_map_even[own_x_inter+1,own_y_inter+1,0:counter_map_even[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc_even['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc_even['yc'][indice_foraverage]
    foraverage_distance_bool = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))<0.5          
    own_points = np.sum(foraverage_distance_bool)
    
    if own_points>min_points_percell:
        
        foraverage_dispY_final = xinter_yinter_frame_disp_angle_xcyc_even['angle'][indice_foraverage][foraverage_distance_bool]
        foraverage_dispX_final = xinter_yinter_frame_disp_angle_xcyc_even['disp'][indice_foraverage][foraverage_distance_bool]
        averaged_histogram_start_evenX,_ = np.histogram(foraverage_dispX_final, 
                              bins = hist_bin_numX, range = (-disp_threX,disp_threX))
        averaged_histogram_start_evenY,_ = np.histogram(foraverage_dispY_final, 
                              bins = hist_bin_numY, range = (-disp_threY,disp_threY))
        
    else:
        averaged_histogram_start_evenX = np.zeros(hist_bin_numX)
        averaged_histogram_start_evenY = np.zeros(hist_bin_numY)
    
    averaged_points_start_even = (own_points)
    
    start_array = np.hstack((averaged_xc_start,averaged_yc_start,
                   averaged_frame_start,averaged_points_start_odd+averaged_points_start_even,
                   averaged_histogram_start_oddX,averaged_histogram_start_oddY,
                   averaged_histogram_start_evenX,averaged_histogram_start_evenY))
    
    # repeat for x_end and y_end, only difference is frame+1: start
    
    own_xc = STORM_npdata['x_end']/image_bin_size
    own_yc = STORM_npdata['y_end']/image_bin_size
    own_frame = STORM_npdata['frame']+1
    own_x_inter = own_xc.astype(np.int16)
    own_y_inter = own_yc.astype(np.int16)
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map_odd[own_x_inter-1,own_y_inter-1,0:counter_map_odd[own_x_inter-1,own_y_inter-1]],
                                      indice_map_odd[own_x_inter,own_y_inter-1,0:counter_map_odd[own_x_inter,own_y_inter-1]],
                                      indice_map_odd[own_x_inter+1,own_y_inter-1,0:counter_map_odd[own_x_inter+1,own_y_inter-1]],
                                      indice_map_odd[own_x_inter-1,own_y_inter,0:counter_map_odd[own_x_inter-1,own_y_inter]],
                                      indice_map_odd[own_x_inter,own_y_inter,0:counter_map_odd[own_x_inter,own_y_inter]],
                                      indice_map_odd[own_x_inter+1,own_y_inter,0:counter_map_odd[own_x_inter+1,own_y_inter]],
                                      indice_map_odd[own_x_inter-1,own_y_inter+1,0:counter_map_odd[own_x_inter-1,own_y_inter+1]],
                                      indice_map_odd[own_x_inter,own_y_inter+1,0:counter_map_odd[own_x_inter,own_y_inter+1]],
                                      indice_map_odd[own_x_inter+1,own_y_inter+1,0:counter_map_odd[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc_odd['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc_odd['yc'][indice_foraverage]
    foraverage_distance_bool = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))<0.5          
    own_points = np.sum(foraverage_distance_bool)
    
    if own_points>min_points_percell:
        
        foraverage_dispY_final = xinter_yinter_frame_disp_angle_xcyc_odd['angle'][indice_foraverage][foraverage_distance_bool]
        foraverage_dispX_final = xinter_yinter_frame_disp_angle_xcyc_odd['disp'][indice_foraverage][foraverage_distance_bool]
        averaged_histogram_end_oddX,_ = np.histogram(foraverage_dispX_final, 
                              bins = hist_bin_numX, range = (-disp_threX,disp_threX))
        averaged_histogram_end_oddY,_ = np.histogram(foraverage_dispY_final, 
                              bins = hist_bin_numY, range = (-disp_threY,disp_threY))
        
    else:
        averaged_histogram_end_oddX = np.zeros(hist_bin_numX)
        averaged_histogram_end_oddY = np.zeros(hist_bin_numY)
    
    averaged_xc_end = (own_xc*image_bin_size)
    averaged_yc_end = (own_yc*image_bin_size)
    averaged_frame_end = own_frame
    averaged_points_end_odd = (own_points)
    
    # end point, even dataset
    
    indice_foraverage = 0
    if own_x_inter>0 and own_x_inter+1<xinter_max and own_y_inter>0 and own_y_inter+1<yinter_max:
        indice_foraverage = np.hstack((indice_map_even[own_x_inter-1,own_y_inter-1,0:counter_map_even[own_x_inter-1,own_y_inter-1]],
                                      indice_map_even[own_x_inter,own_y_inter-1,0:counter_map_even[own_x_inter,own_y_inter-1]],
                                      indice_map_even[own_x_inter+1,own_y_inter-1,0:counter_map_even[own_x_inter+1,own_y_inter-1]],
                                      indice_map_even[own_x_inter-1,own_y_inter,0:counter_map_even[own_x_inter-1,own_y_inter]],
                                      indice_map_even[own_x_inter,own_y_inter,0:counter_map_even[own_x_inter,own_y_inter]],
                                      indice_map_even[own_x_inter+1,own_y_inter,0:counter_map_even[own_x_inter+1,own_y_inter]],
                                      indice_map_even[own_x_inter-1,own_y_inter+1,0:counter_map_even[own_x_inter-1,own_y_inter+1]],
                                      indice_map_even[own_x_inter,own_y_inter+1,0:counter_map_even[own_x_inter,own_y_inter+1]],
                                      indice_map_even[own_x_inter+1,own_y_inter+1,0:counter_map_even[own_x_inter+1,own_y_inter+1]]))
    
    foraverage_xc = xinter_yinter_frame_disp_angle_xcyc_even['xc'][indice_foraverage]
    foraverage_yc = xinter_yinter_frame_disp_angle_xcyc_even['yc'][indice_foraverage]
    foraverage_distance_bool = np.sqrt(np.square(foraverage_xc-own_xc)+np.square(foraverage_yc-own_yc))<0.5          
    own_points = np.sum(foraverage_distance_bool)
    
    if own_points>min_points_percell:
        
        foraverage_dispY_final = xinter_yinter_frame_disp_angle_xcyc_even['angle'][indice_foraverage][foraverage_distance_bool]
        foraverage_dispX_final = xinter_yinter_frame_disp_angle_xcyc_even['disp'][indice_foraverage][foraverage_distance_bool]
        averaged_histogram_end_evenX,_ = np.histogram(foraverage_dispX_final, 
                              bins = hist_bin_numX, range = (-disp_threX,disp_threX))
        averaged_histogram_end_evenY,_ = np.histogram(foraverage_dispY_final, 
                              bins = hist_bin_numY, range = (-disp_threY,disp_threY))
        
    else:
        averaged_histogram_end_evenX = np.zeros(hist_bin_numX)
        averaged_histogram_end_evenY = np.zeros(hist_bin_numY)
    
    averaged_points_end_even = (own_points)
    
    end_array = np.hstack((averaged_xc_end,averaged_yc_end,
                   averaged_frame_end,averaged_points_end_odd+averaged_points_end_even,
                   averaged_histogram_end_oddX,averaged_histogram_end_oddY,
                   averaged_histogram_end_evenX,averaged_histogram_end_evenY))
    
    # repeat for x_end and y_end, only difference is frame+1: finished
    
    return_array = np.stack((start_array,end_array),axis=1)
    
    return return_array
    
    