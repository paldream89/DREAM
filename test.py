import numpy as np
import matplotlib.pyplot as plt

image_bin_size = 7
distance_max = image_bin_size/2
distance_min = 0
bin_num_radial = 20
distance_step = (distance_max-distance_min)/bin_num_radial
ranges = [(range_i, range_i + distance_step) for range_i in np.arange(distance_min, distance_max, distance_step)] 
drate_radius_binx = np.arange(0.5*distance_step, distance_max, distance_step, dtype=np.float32)

drate_start = np.zeros(bin_num_radial)
for range_i, (r_min, r_max) in enumerate(ranges):
    
    range_i+=1

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 15:01:29 2025

@author: Admin
"""

odd_frame_array_Xc = np.array([1, 3, 5])
even_frame_array_Xc = np.array([2, 4])
odd_frame_array_Yc = np.array([7, 5, 9])
even_frame_array_Yc = np.array([13, 16])

disp_threX = 2
disp_threY = 10

distance_X = np.abs(odd_frame_array_Xc[None, :] - even_frame_array_Xc[:, None])
distance_Y = np.abs(odd_frame_array_Yc[None, :] - even_frame_array_Yc[:, None])

disp_thre_bool = (distance_X < disp_threX) & (distance_Y < disp_threY)
even_idx, odd_idx = np.where(disp_thre_bool)

print(even_idx, odd_idx)
