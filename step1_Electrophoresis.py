# -*- coding: utf-8 -*- 
"""
Created on Fri Apr 22 21:14:52 2022

@author: limin xiang
"""

import numpy as np
from ReadSTORMBin import read_storm_bin
from WriteSTORMBin_XcYcZZcFrame import write_storm_bin_XcYcZZcFrame
from disp_angle_calculator_pickone2 import disp_angle_calculator_XY
import time
import wx


def optimize_frame_indices_incremental(original_mol_Frame, total_mol_num, frame_start, sep,
                                       cursor_A, cursor_B, cursor_C, cursor_D):
    frame1 = frame_start
    frame2 = frame_start + sep

    while cursor_A < total_mol_num and original_mol_Frame[cursor_A] < frame1:
        cursor_A += 1
    cursor_B = cursor_A
    while cursor_B < total_mol_num and original_mol_Frame[cursor_B] == frame1:
        cursor_B += 1

    while cursor_C < total_mol_num and original_mol_Frame[cursor_C] < frame2:
        cursor_C += 1
    cursor_D = cursor_C
    while cursor_D < total_mol_num and original_mol_Frame[cursor_D] == frame2:
        cursor_D += 1

    if cursor_A == cursor_B or cursor_C == cursor_D:
        return cursor_A, cursor_B, cursor_C, cursor_D, False

    return cursor_A, cursor_B, cursor_C, cursor_D, True


if __name__ == "__main__":
    
    app = wx.App()
    frame = wx.Frame(None, -1, 'win.py')
    frame.SetSize(0, 0, 200, 50)

    openFileDialog = wx.FileDialog(frame, "Select STORM Bin file", "", "", 
                                   "Bin files (*.bin)|*.bin", 
                                   wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

    openFileDialog.ShowModal()
    file_path = openFileDialog.GetPath()
    openFileDialog.Destroy()

    total_frame, total_mol_num, original_mol_list = read_storm_bin(file_path)
    total_frame = total_frame.item()
    total_mol_num = total_mol_num.item()

    original_mol_Xc = np.float32(original_mol_list['Xc'])
    original_mol_Yc = np.float32(original_mol_list['Yc'])
    original_mol_Frame = np.int32(original_mol_list['Frame'])

    sep_array = [1]  # seperated by XXX frame
    dispthreX_array = [4]  # displacement in x
    dispthreY_array = [4]  # displacement in y

    estimated_total_num = 100
    
    start_time = time.time()

    for sep, disp_threX, disp_threY in zip(sep_array, dispthreX_array, dispthreY_array):
        result_array_x_start = np.zeros(estimated_total_num, dtype=np.float32)
        result_array_y_start = np.zeros(estimated_total_num, dtype=np.float32)
        result_array_x_end = np.zeros(estimated_total_num, dtype=np.float32)
        result_array_y_end = np.zeros(estimated_total_num, dtype=np.float32)
        result_array_frame = np.zeros(estimated_total_num, dtype=np.int32)
        result_array_disp = np.zeros(estimated_total_num, dtype=np.float32)
        result_array_angle = np.zeros(estimated_total_num, dtype=np.float32)

        result_array_start_index = 0
        cursor_A = cursor_B = cursor_C = cursor_D = 0

        for frame_start in range(1, total_frame - sep):
            cursor_A, cursor_B, cursor_C, cursor_D, valid = optimize_frame_indices_incremental(
                original_mol_Frame, total_mol_num, frame_start, sep,
                cursor_A, cursor_B, cursor_C, cursor_D
            )

            if not valid:
                continue

            correlate_num, return_array_x_start, return_array_y_start, return_array_x_end, \
                return_array_y_end, return_array_frame, return_array_disp, return_array_angle \
                = disp_angle_calculator_XY(
                    original_mol_Xc[cursor_A:cursor_B],
                    original_mol_Yc[cursor_A:cursor_B],
                    original_mol_Xc[cursor_C:cursor_D],
                    original_mol_Yc[cursor_C:cursor_D],
                    cursor_B - cursor_A,
                    cursor_D - cursor_C,
                    disp_threX,disp_threY,
                    frame_start
            )
            # return_array_disp is displacement in X
            # return_array_angle is displacement in Y

            if result_array_start_index + correlate_num > estimated_total_num:
                expand_size = max(estimated_total_num, correlate_num)
                result_array_x_start = np.concatenate([result_array_x_start, np.zeros(expand_size, dtype=np.float32)])
                result_array_y_start = np.concatenate([result_array_y_start, np.zeros(expand_size, dtype=np.float32)])
                result_array_x_end = np.concatenate([result_array_x_end, np.zeros(expand_size, dtype=np.float32)])
                result_array_y_end = np.concatenate([result_array_y_end, np.zeros(expand_size, dtype=np.float32)])
                result_array_frame = np.concatenate([result_array_frame, np.zeros(expand_size, dtype=np.int32)])
                result_array_disp = np.concatenate([result_array_disp, np.zeros(expand_size, dtype=np.float32)])
                result_array_angle = np.concatenate([result_array_angle, np.zeros(expand_size, dtype=np.float32)])
                estimated_total_num += expand_size

            result_array_x_start[result_array_start_index:result_array_start_index + correlate_num] = return_array_x_start
            result_array_y_start[result_array_start_index:result_array_start_index + correlate_num] = return_array_y_start
            result_array_x_end[result_array_start_index:result_array_start_index + correlate_num] = return_array_x_end
            result_array_y_end[result_array_start_index:result_array_start_index + correlate_num] = return_array_y_end
            result_array_frame[result_array_start_index:result_array_start_index + correlate_num] = return_array_frame
            result_array_disp[result_array_start_index:result_array_start_index + correlate_num] = return_array_disp
            result_array_angle[result_array_start_index:result_array_start_index + correlate_num] = return_array_angle

            result_array_start_index += correlate_num
        
            if frame_start%10000 == 1:
                print(f'{frame_start/total_frame*100:.1f}%,sep is {sep}')

        save_file_path = file_path.replace('.bin', f'-dX{disp_threX}-dY{disp_threY}-sep{sep}.bin')

        write_storm_bin_XcYcZZcFrame(save_file_path, total_frame, result_array_start_index,
                                     result_array_x_start[:result_array_start_index],
                                     result_array_y_start[:result_array_start_index],
                                     result_array_x_end[:result_array_start_index],
                                     result_array_y_end[:result_array_start_index],
                                     result_array_frame[:result_array_start_index],
                                     result_array_disp[:result_array_start_index],
                                     result_array_angle[:result_array_start_index])
            # return_array_disp is displacement in X
            # return_array_angle is displacement in Y
        
    print("--- %s seconds ---" % (time.time() - start_time))
