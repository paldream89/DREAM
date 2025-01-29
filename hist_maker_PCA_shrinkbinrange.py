# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 14:23:31 2022

@author: Admin
"""

import numpy as np
from math import ceil,sqrt,e,floor,atan2,atan,pi
from scipy import optimize
from sklearn.decomposition import PCA

# import matplotlib.pyplot as plt


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
    