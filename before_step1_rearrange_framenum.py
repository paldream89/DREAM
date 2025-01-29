# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:29:12 2023

@author: Admin
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

disp_thre = 7

for file_path in file_path_list:

    # read the data
    start_time = time.time()
    total_frame, total_mol_num, original_mol_list = read_storm_bin(file_path)
    total_frame = total_frame.item()
    total_mol_num = total_mol_num.item()
    
    