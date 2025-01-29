# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:36:14 2020

@author: xlm69
"""

import numpy as np
from ReadSTORMBin_XcYcZZcFrame import read_storm_bin_XcYcZZcFrame
from WriteSTORMBin_XcYcZZcFrame import write_storm_bin_XcYcZZcFrame
import wx

app = wx.App()
frame = wx.Frame(None, -1, 'win.py')
frame.SetSize(0,0,200,50)

# Create open file dialog
openFileDialog = wx.FileDialog(frame, "Select Multiple Bin with dis files", "", "", 
      "Bin files (*.bin)|*.bin", 
       wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)

openFileDialog.ShowModal()
file_path_list = openFileDialog.GetPaths()
openFileDialog.Destroy()

data_type = np.dtype([('x_start', np.float32), ('y_start', np.float32),
                  ('x_end', np.float32), ('y_end', np.float32),
                  ('frame', np.int32),
                  ('disp', np.float32), ('angle', np.float32)])   

comb_frame_number = np.array([])
comb_total_number = 0
comb_STORM_npdata = np.zeros(0,dtype=data_type)

for file_path in file_path_list:
    
    frame_num, number_of_points, STORM_npdata = read_storm_bin_XcYcZZcFrame(file_path)
    comb_STORM_npdata = np.append(comb_STORM_npdata,STORM_npdata,axis=0)
    comb_frame_number = np.append(comb_frame_number,frame_num)
    comb_total_number += number_of_points
    
save_file_path = file_path.replace('.bin','-'+str(len(comb_frame_number))+'comb.bin')
write_storm_bin_XcYcZZcFrame(save_file_path, np.amax(comb_frame_number), comb_total_number,
                             comb_STORM_npdata['x_start'],comb_STORM_npdata['y_start'],
                             comb_STORM_npdata['x_end'],comb_STORM_npdata['y_end'],
                             comb_STORM_npdata['frame'],comb_STORM_npdata['disp'],
                             comb_STORM_npdata['angle'])