# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 19:34:06 2022

@author: Admin
"""

import numpy as np

def Write_STORMbin(file_path, total_number, frame_num, x, y, xc, yc, height, area, width, phi, ax, bg, I, 
                   category, valid, frame, length, link, z, zc):
    
    data_type = np.dtype([('X', np.float32), ('Y', np.float32), ('Xc', np.float32), ('Yc', np.float32),
                   ('Height', np.float32), ('Area', np.float32), ('Width', np.float32), ('Phi', np.float32), 
                   ('Ax', np.float32), ('BG', np.float32), ('I', np.float32), ('Category', np.int32), 
                   ('Valid', np.int32), ('Frame', np.int32), ('Length', np.int32), ('Link', np.int32), 
                   ('Z', np.float32), ('Zc', np.float32)])

    new_A = np.zeros(total_number,dtype=data_type)
    new_A['X'] = x
    new_A['Y'] = y
    new_A['Xc'] = xc
    new_A['Yc'] = yc
    new_A['Height'] = height
    new_A['Area'] = area
    new_A['Width'] = width
    new_A['Phi'] = phi
    new_A['Ax'] = ax
    new_A['BG'] = bg
    new_A['I'] = I
    new_A['Category'] = category
    new_A['Valid'] = valid
    new_A['Frame'] = frame
    new_A['Length'] = length
    new_A['Link'] = link
    new_A['Z'] = z
    new_A['Zc'] = zc
    
    write_header = np.array([892482637,frame_num,6,total_number],dtype=np.int32)
    
    f = open(file_path,"wb")
    arr = bytearray(new_A)
    arr_2 = bytearray(write_header)
    f.write(arr_2)
    f.write(arr)
    f.close()