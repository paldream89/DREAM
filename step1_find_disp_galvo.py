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

disp_thre = 9

# Create open file dialog
openFileDialog = wx.FileDialog(frame, "Select 1st odd STORM bin", "", "", 
      "bin files (*.bin)|*.bin", 
       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

openFileDialog.ShowModal()
file_path_1stBin = openFileDialog.GetPath()
openFileDialog.Destroy()

openFileDialog = wx.FileDialog(frame, "Select 2nd even STORM bin", "", "", 
      "bin files (*.bin)|*.bin", 
       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

openFileDialog.ShowModal()
file_path_2ndBin = openFileDialog.GetPath()
openFileDialog.Destroy()

# read the data
start_time = time.time()
total_frame_1, total_mol_num_1, original_mol_list_1 = read_storm_bin(file_path_1stBin)
total_frame_2, total_mol_num_2, original_mol_list_2 = read_storm_bin(file_path_2ndBin)
total_frame = total_frame_1.item()+total_frame_2.item()
total_mol_num = total_mol_num_1.item()+total_mol_num_2.item()

original_mol_list_1['Frame']=np.int32(original_mol_list_1['Frame']*2-1)
original_mol_list_2['Frame']=np.int32(original_mol_list_2['Frame']*2)
original_mol_list = np.hstack((original_mol_list_1,original_mol_list_2))
frame_indice = np.argsort(original_mol_list['Frame'])
original_mol_list=original_mol_list[frame_indice]

original_mol_Xc = np.float32(original_mol_list['Xc'])
original_mol_Yc = np.float32(original_mol_list['Yc'])
original_mol_Frame = np.int32(original_mol_list['Frame'])

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

while frame_cursor < total_frame and frame_index_cursor_D <= total_mol_num-1:
        
    if original_mol_Frame[frame_index_cursor_D]==frame_cursor:
        
        frame_index_cursor_A = frame_index_cursor_D
        
        while original_mol_Frame[frame_index_cursor_D]==frame_cursor:
            
            frame_index_cursor_D += 1
            if frame_index_cursor_D > total_mol_num-1:
                break
            
        frame_index_cursor_B = frame_index_cursor_D
        frame_cursor += 1
        
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

save_file_path = file_path_1stBin.replace('.bin','-dis'+str(disp_thre)+'.bin')

write_storm_bin_XcYcZZcFrame(save_file_path, total_frame, result_array_start_index, 
                             result_array_x_start[0:result_array_start_index], 
                             result_array_y_start[0:result_array_start_index], 
                             result_array_x_end[0:result_array_start_index], 
                             result_array_y_end[0:result_array_start_index],
                             result_array_frame[0:result_array_start_index], 
                             result_array_disp[0:result_array_start_index], 
                             result_array_angle[0:result_array_start_index])

print("--- %s seconds ---" % (time.time() - start_time))


# write_storm_bin_XcYcZZcFrame(save_file_path, Xc = result_array_format['Xc'],
#                              Yc = result_array_format['Yc'], 
#                     I = result_array_format['I'], Frame = result_array_format['Frame'], 
#                     Z = result_array_format['Z'], Zc = result_array_format['Zc'])

# frame_num_counter = np.zeros((total_frame,), dtype=np.int32)

# for frame_num in original_mol_list['Frame']:
    
#     frame_num_counter[frame_num-1] += 1
    
# frame_cursor = np.hstack(([1],np.cumsum(frame_num_counter)))
    
# print("--- %s seconds ---" % (time.time() - start_time))


# odd_frame_array = np.stack((original_mol_Xc[frame_index_cursor_A:frame_index_cursor_B+1],
#                             original_mol_Yc[frame_index_cursor_A:frame_index_cursor_B+1],
#                             original_mol_I[frame_index_cursor_A:frame_index_cursor_B+1]),axis=1)
# even_frame_array = np.stack((original_mol_Xc[frame_index_cursor_C:frame_index_cursor_D+1],
#                             original_mol_Yc[frame_index_cursor_C:frame_index_cursor_D+1],
#                             original_mol_I[frame_index_cursor_C:frame_index_cursor_D+1]),axis=1)



