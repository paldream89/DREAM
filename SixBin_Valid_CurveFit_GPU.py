# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 21:14:52 2022

@author: limin xiang
"""

import numpy as np
from ReadSTORMBin import read_storm_bin
from math import ceil,sqrt,e,floor
from scipy import optimize
# import matplotlib.pyplot as plt
import time
import wx
from WriteSTORMBin import Write_STORMbin_Packed
from scipy.spatial import cKDTree
# from multiprocessing import Pool, cpu_count
# from functools import partial

if __name__=="__main__":
    
    average_radius = 0.5 # unit is pixel
    max_mol = 1000000 # max total mol count in each bin
    keyword = 'I'
    
    app = wx.App()
    frame = wx.Frame(None, -1, 'win.py')
    frame.SetSize(0,0,200,50)
    
    # Create open file dialog
    # Ctrl + Mouse Left Button
    # Make sure select 1ms->2ms->3ms->4ms->5ms
    openFileDialog = wx.FileDialog(frame, "Select Multiple Bin", "", "", 
          "Bin files (*.bin)|*.bin", 
           wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)
    
    openFileDialog.ShowModal()
    file_path_list = openFileDialog.GetPaths()
    openFileDialog.Destroy()
    
    num_of_bin = np.size(file_path_list)
    # read the data
    start_time = time.time()
    
    data_type = np.dtype([('X', np.float32), ('Y', np.float32), ('Xc', np.float32), ('Yc', np.float32),
               ('Height', np.float32), ('Area', np.float32), ('Width', np.float32), ('Phi', np.float32), 
               ('Ax', np.float32), ('BG', np.float32), ('I', np.float32), ('Category', np.int32), 
               ('Valid', np.int32), ('Frame', np.int32), ('Length', np.int32), ('Link', np.int32), 
               ('Z', np.float32), ('Zc', np.float32)])
    
    total_frame = np.zeros(num_of_bin,dtype=np.int32)
    total_mol_num = np.zeros(num_of_bin,dtype=np.int32)
    original_mol_list = np.zeros((num_of_bin,max_mol),dtype=data_type)
    bin_index = 0
    
    for file_path in file_path_list:
        
        total_frame_temp, total_mol_num_temp, original_mol_list_temp = read_storm_bin(file_path)
        total_frame[bin_index] = int(total_frame_temp.item())
        # total_frame = 120000
        total_mol_num[bin_index] = int(total_mol_num_temp.item())
        original_mol_list[bin_index,:total_mol_num_temp] = original_mol_list_temp
        bin_index+=1
        
    # start processing
    # 根据第一个bin的每个分子，建立矩阵
    valid_array_eachmol = np.zeros((num_of_bin,total_mol_num[0]),dtype=np.float32)
    
    # 为每个 bin 构建 KD 树索引
    trees = []
    for j in range(num_of_bin):
        coords = np.stack((
            original_mol_list[j, :total_mol_num[j]]['Xc'],
            original_mol_list[j, :total_mol_num[j]]['Yc']
        ), axis=1)
        trees.append(cKDTree(coords))
    
    # 初始化结果数组
    valid_array_eachmol = np.zeros((num_of_bin, total_mol_num[0]), dtype=np.float32)
    
    # 以第一个 bin 的每个点作为中心，在每个 bin 中查找邻域平均值
    for i in range(total_mol_num[0]):
        x = original_mol_list[0, i]['Xc']
        y = original_mol_list[0, i]['Yc']
        center = np.array([x, y])
    
        for j in range(num_of_bin):
            # 在第 j 个 bin 的 KD 树中查找邻域内的索引
            indices = trees[j].query_ball_point(center, r=average_radius)
    
            if indices:  # 如果找到邻近点
                valid_values = original_mol_list[j, indices][keyword]
                valid_array_eachmol[j, i] = np.mean(valid_values)
            else:
                valid_array_eachmol[j, i] = 0  # 没找到邻近点则赋为 np.nan
                
    print("--- %s seconds ---" % (time.time() - start_time))
