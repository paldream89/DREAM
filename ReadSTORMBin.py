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

def read_storm_bin(file_path):
 
    header_type = np.dtype([('A',np.int32),('B',np.int32),('C',np.int32),('D',np.int32)])
    number_of_points_extractor = np.fromfile(file_path, dtype=header_type ,count=1)
    number_of_points = number_of_points_extractor['D'][0]
    frame_number = number_of_points_extractor['B'][0]
    
    # According to the number of data points, determine the end when reading the bin file
    # The order is quite different when you read a txt file saved from insight 3, be careful
    
    data_type = np.dtype([('X', np.float32), ('Y', np.float32), ('Xc', np.float32), ('Yc', np.float32),
                   ('Height', np.float32), ('Area', np.float32), ('Width', np.float32), ('Phi', np.float32), 
                   ('Ax', np.float32), ('BG', np.float32), ('I', np.float32), ('Category', np.int32), 
                   ('Valid', np.int32), ('Frame', np.int32), ('Length', np.int32), ('Link', np.int32), 
                   ('Z', np.float32), ('Zc', np.float32)])
    STORM_npdata = np.fromfile(file_path, dtype=data_type,count=number_of_points,offset=16)
    
    return frame_number, number_of_points, STORM_npdata

