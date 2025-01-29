# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 11:49:10 2020

@author: xlm69
"""


# disp_angle_calculator

import numpy as np

def disp_angle_calculator(odd_frame_array_Xc,odd_frame_array_Yc,even_frame_array_Xc,
                          even_frame_array_Yc,len_odd,len_even,disp_thre,frame_number):
    
    distance_matrix = np.sqrt((odd_frame_array_Xc[None,:]-even_frame_array_Xc[:,None])**2
    +(odd_frame_array_Yc[None,:]-even_frame_array_Yc[:,None])**2)
    
    disp_thre_bool = distance_matrix < disp_thre
    
    even_indice,odd_indice = np.mgrid[:len_even,:len_odd]
    
    return_array_disp_all = distance_matrix[disp_thre_bool]

    rng = np.random.default_rng()
    random_index = rng.permutation(len(return_array_disp_all))
    
    return_array_x_start_all = odd_frame_array_Xc[odd_indice[disp_thre_bool]][random_index]
    
    _,unique_index = np.unique(return_array_x_start_all,return_index=True)
    
    return_array_x_start = return_array_x_start_all[unique_index]
    return_array_x_end = even_frame_array_Xc[even_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_y_start = odd_frame_array_Yc[odd_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_y_end = even_frame_array_Yc[even_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_disp = return_array_disp_all[random_index][unique_index]
    
    correlate_num = len(return_array_x_start)
    return_array_frame = np.full((correlate_num,),frame_number)
    return_array_angle = np.arctan2((return_array_y_end-return_array_y_start),(return_array_x_end-return_array_x_start))
    
    return correlate_num, return_array_x_start,return_array_y_start,\
        return_array_x_end,return_array_y_end, return_array_frame,\
            return_array_disp,return_array_angle
            
def disp_angle_calculator_keepRawZcI(odd_frame_array_Xc,odd_frame_array_Yc,even_frame_array_Xc,
                          even_frame_array_Yc,len_odd,len_even,disp_thre,frame_number,
                          odd_frame_array_RawZc,even_frame_array_RawZc,
                          odd_frame_array_I,even_frame_array_I):
    
    distance_matrix = np.sqrt((odd_frame_array_Xc[None,:]-even_frame_array_Xc[:,None])**2
    +(odd_frame_array_Yc[None,:]-even_frame_array_Yc[:,None])**2)
    
    disp_thre_bool = distance_matrix < disp_thre
    
    even_indice,odd_indice = np.mgrid[:len_even,:len_odd]
    
    return_array_disp_all = distance_matrix[disp_thre_bool]

    rng = np.random.default_rng()
    random_index = rng.permutation(len(return_array_disp_all))
    
    return_array_x_start_all = odd_frame_array_Xc[odd_indice[disp_thre_bool]][random_index]
    
    _,unique_index = np.unique(return_array_x_start_all,return_index=True)
    
    return_array_x_start = return_array_x_start_all[unique_index]
    return_array_x_end = even_frame_array_Xc[even_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_y_start = odd_frame_array_Yc[odd_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_y_end = even_frame_array_Yc[even_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_RawZc_start = odd_frame_array_RawZc[odd_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_RawZc_end = even_frame_array_RawZc[even_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_I_start = odd_frame_array_I[odd_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_I_end = even_frame_array_I[even_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_disp = return_array_disp_all[random_index][unique_index]
    
    correlate_num = len(return_array_x_start)
    return_array_frame = np.full((correlate_num,),frame_number)
    return_array_angle = np.arctan2((return_array_y_end-return_array_y_start),(return_array_x_end-return_array_x_start))
    
    return correlate_num, return_array_x_start,return_array_y_start,\
        return_array_x_end,return_array_y_end, return_array_frame,\
            return_array_disp,return_array_angle,return_array_RawZc_start,\
            return_array_RawZc_end,return_array_I_start,return_array_I_end
            
def disp_angle_calculator_Z(odd_frame_array_Xc,odd_frame_array_Yc,
                            odd_frame_array_Zc,even_frame_array_Xc,
                            even_frame_array_Yc,even_frame_array_Zc,
                            len_odd,len_even,disp_thre,frame_number,pixel_size):
    
    distance_matrix = np.sqrt((odd_frame_array_Xc[None,:]-even_frame_array_Xc[:,None])**2
    +(odd_frame_array_Yc[None,:]-even_frame_array_Yc[:,None])**2)
    distance_matrix_Zc = (odd_frame_array_Zc[None,:]-even_frame_array_Zc[:,None])/pixel_size
    
    disp_thre_bool = distance_matrix < disp_thre
    
    even_indice,odd_indice = np.mgrid[:len_even,:len_odd]
    
    return_array_disp_all = distance_matrix[disp_thre_bool]
    return_array_disp_Z_all = distance_matrix_Zc[disp_thre_bool]

    rng = np.random.default_rng()
    random_index = rng.permutation(len(return_array_disp_all))
    
    return_array_x_start_all = odd_frame_array_Xc[odd_indice[disp_thre_bool]][random_index]
    
    _,unique_index = np.unique(return_array_x_start_all,return_index=True)
    
    return_array_x_start = return_array_x_start_all[unique_index]
    return_array_x_end = even_frame_array_Xc[even_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_y_start = odd_frame_array_Yc[odd_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_y_end = even_frame_array_Yc[even_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_disp = return_array_disp_Z_all[random_index][unique_index]
    return_array_disp_xy = return_array_disp_all[random_index][unique_index]
    return_array_angle = np.arctan2((return_array_y_end-return_array_y_start),(return_array_x_end-return_array_x_start))
    
    correlate_num = len(return_array_x_start)
    return_array_frame = np.full((correlate_num,),frame_number)
    
    return correlate_num, return_array_x_start,return_array_y_start,\
        return_array_x_end,return_array_y_end, return_array_frame,\
            return_array_disp,return_array_angle,return_array_disp_xy
            
def disp_angle_calculator_Z_Zvalue(odd_frame_array_Xc,odd_frame_array_Yc,
                            odd_frame_array_Zc,even_frame_array_Xc,
                            even_frame_array_Yc,even_frame_array_Zc,
                            len_odd,len_even,disp_thre,frame_number,pixel_size,
                            Zvalue):
    
    Zvalue_bool1 = odd_frame_array_Zc < Zvalue 
    Zvalue_bool2 = odd_frame_array_Zc > -Zvalue
    Zvalue_bool = Zvalue_bool1 & Zvalue_bool2
    odd_frame_array_Xc = odd_frame_array_Xc[Zvalue_bool]
    odd_frame_array_Yc = odd_frame_array_Yc[Zvalue_bool]
    odd_frame_array_Zc = odd_frame_array_Zc[Zvalue_bool]
    len_odd = len(odd_frame_array_Xc)
    distance_matrix = np.sqrt((odd_frame_array_Xc[None,:]-even_frame_array_Xc[:,None])**2
    +(odd_frame_array_Yc[None,:]-even_frame_array_Yc[:,None])**2)
    distance_matrix_Zc = (odd_frame_array_Zc[None,:]-even_frame_array_Zc[:,None])/pixel_size
    
    disp_thre_bool = distance_matrix < disp_thre
    
    even_indice,odd_indice = np.mgrid[:len_even,:len_odd]
    
    return_array_disp_all = distance_matrix[disp_thre_bool]
    return_array_disp_Z_all = distance_matrix_Zc[disp_thre_bool]

    rng = np.random.default_rng()
    random_index = rng.permutation(len(return_array_disp_all))
    
    return_array_x_start_all = odd_frame_array_Xc[odd_indice[disp_thre_bool]][random_index]
    
    _,unique_index = np.unique(return_array_x_start_all,return_index=True)
    
    return_array_x_start = return_array_x_start_all[unique_index]
    return_array_x_end = even_frame_array_Xc[even_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_y_start = odd_frame_array_Yc[odd_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_y_end = even_frame_array_Yc[even_indice[disp_thre_bool]][random_index]\
        [unique_index]
    return_array_disp = return_array_disp_Z_all[random_index][unique_index]
    return_array_angle = return_array_disp_all[random_index][unique_index]
    
    correlate_num = len(return_array_x_start)
    return_array_frame = np.full((correlate_num,),frame_number)
    
    return correlate_num, return_array_x_start,return_array_y_start,\
        return_array_x_end,return_array_y_end, return_array_frame,\
            return_array_disp,return_array_angle