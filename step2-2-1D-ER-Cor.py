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
from WriteSTORMBin import Write_STORMbin_Packed
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

OneD_ratio = 0.9
Valid_thre = 400

file_path = file_path_list[0]

# read the data
start_time = time.time()
total_frame, total_mol_num, original_mol_list = read_storm_bin(file_path)
total_frame = total_frame.item()
total_mol_num = total_mol_num.item()

# 创建逻辑掩码，找到同时满足 Category == 0 且 Valid > Valid_thre 的位置
mask = (original_mol_list['Category'] == 0) & (original_mol_list['Valid'] > Valid_thre)

# 在这些位置上对 Zc 进行处理
original_mol_list['Zc'][mask] = original_mol_list['Zc'][mask]**OneD_ratio

save_file_path = file_path.replace('.bin','-R%.2f-V%.0f.bin' %(OneD_ratio,Valid_thre))

Write_STORMbin_Packed(save_file_path, total_mol_num, total_frame+2, original_mol_list)

