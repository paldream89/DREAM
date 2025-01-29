# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 15:28:10 2022

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 10:31:38 2020

@author: xlm69
"""

# data_type = np.dtype([('X', np.float32), ('Y', np.float32), ('Xc', np.float32), ('Yc', np.float32),
#                    ('Height', np.float32), ('Area', np.float32), ('Width', np.float32), ('Phi', np.float32), 
#                    ('Ax', np.float32), ('BG', np.float32), ('I', np.float32), ('Category', np.int32), 
#                    ('Valid', np.int32), ('Frame', np.int32), ('Length', np.int32), ('Link', np.int32), 
#                    ('Z', np.float32), ('Zc', np.float32)])

import numpy as np
from ReadSTORMBin import read_storm_bin
from WriteSTORMBin_XcYcZZcFrame import write_storm_bin_XcYcZZcFrame
from disp_angle_calculator_pickone2 import disp_angle_calculator
import time
import wx

app = wx.App()
frame = wx.Frame(None, -1, 'win.py')
frame.SetSize(0,0,200,50)

# Create open file dialog
openFileDialog = wx.FileDialog(frame, "Select Multiple STORM Bin files", "", "", 
      "Bin files (*.bin)|*.bin", 
       wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)

openFileDialog.ShowModal()
file_path_list = openFileDialog.GetPaths()
openFileDialog.Destroy()

disp_thre = 5

for file_path in file_path_list:

    # read the data
    start_time = time.time()
    total_frame, total_mol_num, original_mol_list = read_storm_bin(file_path)
    total_frame = total_frame.item()
    total_mol_num = total_mol_num.item()
    
    estimated_total_num = 25*total_frame
    result_array_x_start = np.zeros(estimated_total_num,dtype=np.float32)
    result_array_y_start = np.zeros(estimated_total_num,dtype=np.float32)
    result_array_x_end = np.zeros(estimated_total_num,dtype=np.float32)
    result_array_y_end = np.zeros(estimated_total_num,dtype=np.float32)
    result_array_frame = np.zeros(estimated_total_num,dtype=np.int32)
    result_array_disp= np.zeros(estimated_total_num,dtype=np.float32)
    result_array_angle = np.zeros(estimated_total_num,dtype=np.float32)
    
    result_array_start_index = 0
            
    frame_cursor = 1
    frame_index_cursor_A = 0
    frame_index_cursor_B = 0
    frame_index_cursor_C = 0
    frame_index_cursor_D = 0
    
    original_mol_Xc = np.float32(original_mol_list['Xc'])
    original_mol_Yc = np.float32(original_mol_list['Yc'])
    original_mol_Frame = np.int32(original_mol_list['Frame'])
    
    # correlate for odd and even frame
    
    while frame_cursor < total_frame and frame_index_cursor_D <= total_mol_num-1:
            
        if original_mol_Frame[frame_index_cursor_D]==frame_cursor:
            
            frame_index_cursor_A = frame_index_cursor_D
            
            while original_mol_Frame[frame_index_cursor_D]==frame_cursor:
                
                frame_index_cursor_D += 1
                if frame_index_cursor_D > total_mol_num-1:
                    break
                
            frame_index_cursor_B = frame_index_cursor_D
            frame_cursor += 1
            
            if frame_index_cursor_D > total_mol_num-1:
                
                break
    
            if original_mol_Frame[frame_index_cursor_D]==frame_cursor:
                
                frame_index_cursor_C = frame_index_cursor_D
                
                while original_mol_Frame[frame_index_cursor_D]==frame_cursor:
                
                    frame_index_cursor_D += 1
                    if frame_index_cursor_D > total_mol_num-1:
                        break
            
            frame_cursor += 1
            
            if frame_cursor%1000 == 1:
                print(frame_cursor/total_frame)
            
            if frame_index_cursor_B == frame_index_cursor_C:
                
                correlate_num, return_array_x_start, return_array_y_start,return_array_x_end, \
                    return_array_y_end,return_array_frame, return_array_disp, return_array_angle \
                        = disp_angle_calculator(original_mol_Xc[frame_index_cursor_A:frame_index_cursor_B],
                                      original_mol_Yc[frame_index_cursor_A:frame_index_cursor_B],
                                      original_mol_Xc[frame_index_cursor_C:frame_index_cursor_D],
                                      original_mol_Yc[frame_index_cursor_C:frame_index_cursor_D],
                                      frame_index_cursor_B-frame_index_cursor_A,
                                      frame_index_cursor_D-frame_index_cursor_C,
                                      disp_thre,frame_cursor-2)
                
                result_array_x_start[result_array_start_index:result_array_start_index+correlate_num] \
                    = return_array_x_start
                result_array_y_start[result_array_start_index:result_array_start_index+correlate_num] \
                    = return_array_y_start
                result_array_x_end[result_array_start_index:result_array_start_index+correlate_num] \
                    = return_array_x_end
                result_array_y_end[result_array_start_index:result_array_start_index+correlate_num] \
                    = return_array_y_end
                result_array_frame[result_array_start_index:result_array_start_index+correlate_num] \
                    = return_array_frame
                result_array_disp[result_array_start_index:result_array_start_index+correlate_num] \
                    = return_array_disp
                result_array_angle[result_array_start_index:result_array_start_index+correlate_num] \
                    = return_array_angle
                
                result_array_start_index += correlate_num
        
        elif original_mol_Frame[frame_index_cursor_D]<frame_cursor:
            
            frame_index_cursor_D += 1
            
        elif original_mol_Frame[frame_index_cursor_D]>frame_cursor:
            
            frame_cursor += 2
    
    # repeat again for correlating even and odd
    
    result_array_x_start_2 = np.zeros(estimated_total_num,dtype=np.float32)
    result_array_y_start_2 = np.zeros(estimated_total_num,dtype=np.float32)
    result_array_x_end_2 = np.zeros(estimated_total_num,dtype=np.float32)
    result_array_y_end_2 = np.zeros(estimated_total_num,dtype=np.float32)
    result_array_frame_2 = np.zeros(estimated_total_num,dtype=np.int32)
    result_array_disp_2 = np.zeros(estimated_total_num,dtype=np.float32)
    result_array_angle_2 = np.zeros(estimated_total_num,dtype=np.float32)
    
    result_array_start_index_2 = 0
            
    frame_cursor = 2
    frame_index_cursor_A = 0
    frame_index_cursor_B = 0
    frame_index_cursor_C = 0
    frame_index_cursor_D = 0
    
    while frame_cursor < total_frame and frame_index_cursor_D <= total_mol_num-1:
            
        if original_mol_Frame[frame_index_cursor_D]==frame_cursor:
            
            frame_index_cursor_A = frame_index_cursor_D
            
            while original_mol_Frame[frame_index_cursor_D]==frame_cursor:
                
                frame_index_cursor_D += 1
                if frame_index_cursor_D > total_mol_num-1:
                    break
                
            frame_index_cursor_B = frame_index_cursor_D
            frame_cursor += 1
            
            if frame_index_cursor_D > total_mol_num-1:
                
                break
            
            if original_mol_Frame[frame_index_cursor_D]==frame_cursor:
                
                frame_index_cursor_C = frame_index_cursor_D
                
                while original_mol_Frame[frame_index_cursor_D]==frame_cursor:
                
                    frame_index_cursor_D += 1
                    if frame_index_cursor_D > total_mol_num-1:
                        break
            
            frame_cursor += 1
            
            if frame_cursor%1000 == 1:
                print(frame_cursor/total_frame)
            
            if frame_index_cursor_B == frame_index_cursor_C:
                
                correlate_num, return_array_x_start_2, return_array_y_start_2,return_array_x_end_2, \
                    return_array_y_end_2,return_array_frame_2, return_array_disp_2, return_array_angle_2 \
                        = disp_angle_calculator(original_mol_Xc[frame_index_cursor_A:frame_index_cursor_B],
                                      original_mol_Yc[frame_index_cursor_A:frame_index_cursor_B],
                                      original_mol_Xc[frame_index_cursor_C:frame_index_cursor_D],
                                      original_mol_Yc[frame_index_cursor_C:frame_index_cursor_D],
                                      frame_index_cursor_B-frame_index_cursor_A,
                                      frame_index_cursor_D-frame_index_cursor_C,
                                      disp_thre,frame_cursor-2)
                
                result_array_x_start_2[result_array_start_index_2:result_array_start_index_2+correlate_num] \
                    = return_array_x_start_2
                result_array_y_start_2[result_array_start_index_2:result_array_start_index_2+correlate_num] \
                    = return_array_y_start_2
                result_array_x_end_2[result_array_start_index_2:result_array_start_index_2+correlate_num] \
                    = return_array_x_end_2
                result_array_y_end_2[result_array_start_index_2:result_array_start_index_2+correlate_num] \
                    = return_array_y_end_2
                result_array_frame_2[result_array_start_index_2:result_array_start_index_2+correlate_num] \
                    = return_array_frame_2
                result_array_disp_2[result_array_start_index_2:result_array_start_index_2+correlate_num] \
                    = return_array_disp_2
                result_array_angle_2[result_array_start_index_2:result_array_start_index_2+correlate_num] \
                    = return_array_angle_2
                
                result_array_start_index_2 += correlate_num
        
        elif original_mol_Frame[frame_index_cursor_D]<frame_cursor:
            
            frame_index_cursor_D += 1
            
        elif original_mol_Frame[frame_index_cursor_D]>frame_cursor:
            
            frame_cursor += 2
    
    # repeat again for correlating even and odd, finished
    
    save_file_path = file_path.replace('.bin','-dis'+str(disp_thre)+'.bin')
    
    write_storm_bin_XcYcZZcFrame(save_file_path, total_frame, result_array_start_index+result_array_start_index_2, 
                                 np.hstack((result_array_x_start[0:result_array_start_index],result_array_x_start_2[0:result_array_start_index_2])), 
                                 np.hstack((result_array_y_start[0:result_array_start_index],result_array_y_start_2[0:result_array_start_index_2])), 
                                 np.hstack((result_array_x_end[0:result_array_start_index],result_array_x_end_2[0:result_array_start_index_2])), 
                                 np.hstack((result_array_y_end[0:result_array_start_index],result_array_y_end_2[0:result_array_start_index_2])),
                                 np.hstack((result_array_frame[0:result_array_start_index],result_array_frame_2[0:result_array_start_index_2])), 
                                 np.hstack((result_array_disp[0:result_array_start_index],result_array_disp_2[0:result_array_start_index_2])), 
                                 np.hstack((result_array_angle[0:result_array_start_index],result_array_angle_2[0:result_array_start_index_2])))
    
    print("--- %s seconds ---" % (time.time() - start_time))



