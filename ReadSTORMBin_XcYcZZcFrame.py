# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 11:24:49 2020

@author: xlm69
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:38:41 2020

@author: xlm69
"""


# read STORM data

import numpy as np

# file_path = r'C:\Users\xlm69\Desktop\python_work\threeD\S1_EM150_f300cyl_15k_sec1234567_zstep2_increase_cat_xyzmatch_cat1_by_bowen.bin'

# Get how many data points the bin file has
# Seems like the bin use the first 16 bytes, as the first line, to store the name
# The last 4 bytes of the first 16 bytes are the number of data points

# A = 892482637
# B = frame number
# C = 6
# D = total molecule number

def read_storm_bin_XcYcZZcFrame(file_path):
 
    header_type = np.dtype([('A',np.int32),('B',np.int32),('C',np.int32),('D',np.int32)])
    number_of_points_extractor = np.fromfile(file_path, dtype=header_type ,count=1)
    number_of_points = number_of_points_extractor['D'][0]
    frame_number = number_of_points_extractor['B'][0]
    
    # According to the number of data points, determine the end when reading the bin file
    # The order is quite different when you read a txt file saved from insight 3, be careful
    
    data_type = np.dtype([('x_start', np.float32), ('y_start', np.float32),
                      ('x_end', np.float32), ('y_end', np.float32),
                      ('frame', np.int32),
                      ('disp', np.float32), ('angle', np.float32)])   
    
    STORM_npdata = np.fromfile(file_path, dtype=data_type, count=number_of_points, offset=16)
    
    return frame_number, number_of_points, STORM_npdata

def read_storm_bin_XcYcZZcFrame_RawZcI(file_path):
 
    header_type = np.dtype([('A',np.int32),('B',np.int32),('C',np.int32),('D',np.int32)])
    number_of_points_extractor = np.fromfile(file_path, dtype=header_type ,count=1)
    number_of_points = number_of_points_extractor['D'][0]
    frame_number = number_of_points_extractor['B'][0]
    
    # According to the number of data points, determine the end when reading the bin file
    # The order is quite different when you read a txt file saved from insight 3, be careful
    
    data_type = np.dtype([('x_start', np.float32), ('y_start', np.float32),
                      ('x_end', np.float32), ('y_end', np.float32),
                      ('frame', np.int32),
                      ('disp', np.float32), ('angle', np.float32),
                         ('RawZc_start', np.float32), ('RawZc_end', np.float32),
                         ('I_start', np.float32), ('I_end', np.float32)])
    
    STORM_npdata = np.fromfile(file_path, dtype=data_type, count=number_of_points, offset=16)
    
    return frame_number, number_of_points, STORM_npdata

def read_storm_bin_XcYcFramedisAngledisxy(file_path):
 
    header_type = np.dtype([('A',np.int32),('B',np.int32),('C',np.int32),('D',np.int32)])
    number_of_points_extractor = np.fromfile(file_path, dtype=header_type ,count=1)
    number_of_points = number_of_points_extractor['D'][0]
    frame_number = number_of_points_extractor['B'][0]
    
    # According to the number of data points, determine the end when reading the bin file
    # The order is quite different when you read a txt file saved from insight 3, be careful
    
    data_type = np.dtype([('x_start', np.float32), ('y_start', np.float32),
                      ('x_end', np.float32), ('y_end', np.float32),
                      ('frame', np.int32),
                      ('disp', np.float32), ('angle', np.float32),('disp_xy', np.float32)])   
    
    STORM_npdata = np.fromfile(file_path, dtype=data_type, count=number_of_points, offset=16)
    
    return frame_number, number_of_points, STORM_npdata

