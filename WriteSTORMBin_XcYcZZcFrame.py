# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 10:38:44 2020

@author: xlm69
"""


# write arrays into STORM bin data

# assume there are inputs at least from Xc and Yc

import numpy as np

def write_storm_bin_XcYcZZcFrame(file_path, frame_num, total_number, x_start, y_start, x_end, y_end,
                                 frame, disp, angle):
        
    data_type = np.dtype([('x_start', np.float32), ('y_start', np.float32),
                      ('x_end', np.float32), ('y_end', np.float32),
                      ('frame', np.int32),
                      ('disp', np.float32), ('angle', np.float32)])    

    new_A = np.zeros(total_number,dtype=data_type)
    new_A['x_start'] = x_start
    new_A['y_start'] = y_start
    new_A['x_end'] = x_end
    new_A['y_end'] = y_end
    new_A['frame'] = frame
    new_A['disp'] = disp
    new_A['angle'] = angle
    
    write_header = np.array([892482637,frame_num,6,total_number],dtype=np.int32)
    
    f = open(file_path,"wb")
    arr = bytearray(new_A)
    arr_2 = bytearray(write_header)
    f.write(arr_2)
    f.write(arr)
    f.close()
    
def write_storm_bin_XcYcZZcFrameRawZcI(file_path, frame_num, total_number, x_start, y_start, x_end, y_end,
                                 frame, disp, angle,RawZc_start,RawZc_end,I_start,I_end):
        
    data_type = np.dtype([('x_start', np.float32), ('y_start', np.float32),
                      ('x_end', np.float32), ('y_end', np.float32),
                      ('frame', np.int32),
                      ('disp', np.float32), ('angle', np.float32),
                      ('RawZc_start', np.float32), ('RawZc_end', np.float32),
                      ('I_start', np.float32), ('I_end', np.float32)])    

    new_A = np.zeros(total_number,dtype=data_type)
    new_A['x_start'] = x_start
    new_A['y_start'] = y_start
    new_A['x_end'] = x_end
    new_A['y_end'] = y_end
    new_A['frame'] = frame
    new_A['disp'] = disp
    new_A['angle'] = angle
    new_A['RawZc_start'] = RawZc_start
    new_A['RawZc_end'] = RawZc_end
    new_A['I_start'] = I_start
    new_A['I_end'] = I_end
    
    write_header = np.array([892482637,frame_num,6,total_number],dtype=np.int32)
    
    f = open(file_path,"wb")
    arr = bytearray(new_A)
    arr_2 = bytearray(write_header)
    f.write(arr_2)
    f.write(arr)
    f.close()
    
def write_storm_bin_XcYcFramedisAngledisxy(file_path, frame_num, total_number, x_start, y_start, x_end, y_end,
                                 frame, disp, angle, disp_xy):
        
    data_type = np.dtype([('x_start', np.float32), ('y_start', np.float32),
                      ('x_end', np.float32), ('y_end', np.float32),
                      ('frame', np.int32),
                      ('disp', np.float32), ('angle', np.float32),('disp_xy',np.float32)])    

    new_A = np.zeros(total_number,dtype=data_type)
    new_A['x_start'] = x_start
    new_A['y_start'] = y_start
    new_A['x_end'] = x_end
    new_A['y_end'] = y_end
    new_A['frame'] = frame
    new_A['disp'] = disp
    new_A['angle'] = angle
    new_A['disp_xy'] = disp_xy
    
    write_header = np.array([892482637,frame_num,6,total_number],dtype=np.int32)
    
    f = open(file_path,"wb")
    arr = bytearray(new_A)
    arr_2 = bytearray(write_header)
    f.write(arr_2)
    f.write(arr)
    f.close()