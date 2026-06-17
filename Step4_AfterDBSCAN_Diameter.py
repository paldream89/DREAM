# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 21:14:52 2022

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:47:32 2020

@author: limin
"""

import numpy as np
import pandas as pd
from ReadSTORMBin_XcYcZZcFrame import read_storm_bin_XcYcZZcFrame
from ReadSTORMBin import read_storm_bin
from WriteSTORMBin import Write_STORMbin_Packed
from math import ceil,sqrt,e,floor
import matplotlib.pyplot as plt
import time
import wx
# from WriteSTORMBin import Write_STORMbin
from functools import partial
from multiprocessing import Pool, freeze_support,cpu_count
from hist_maker_PCA_shrinkbinrange import hist_maker_radial
from scipy.optimize import curve_fit
import os

# ================注意===================
# 这里的是radial profiling计算直径

def twoD_diff_dis_func(x,a,b,c):
    
    return 2*c*x/a*np.exp(-x**2/a)+b*x

def gaussian_with_background(x, a1, x1, w1, y0):
    
    return a1 * np.exp(-((x - x1) ** 2) / (2 * w1 ** 2)) + y0

if __name__=="__main__":
    
    app = wx.App()
    frame = wx.Frame(None, -1, 'win.py')
    frame.SetSize(0,0,200,50)
    
    # Create open file dialog
    openFileDialog = wx.FileDialog(frame, "Select Multiple Excel files", "", "", 
          "Bin files (*.xlsx)|*.xlsx", 
           wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)
    
    openFileDialog.ShowModal()
    Excel_path_list = openFileDialog.GetPaths()
    openFileDialog.Destroy()

    app2 = wx.App()
    frame2 = wx.Frame(None, -1, 'win.py')
    frame2.SetSize(0,0,200,50)
    
    # Create open file dialog
    openFileDialog = wx.FileDialog(frame2, "Select Multiple Bin with dis files", "", "", 
          "Bin files (*.bin)|*.bin", 
           wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)
    
    openFileDialog.ShowModal()
    file_path_list = openFileDialog.GetPaths()
    openFileDialog.Destroy()
    
    app3 = wx.App()
    frame3 = wx.Frame(None, -1, 'win.py')
    frame3.SetSize(0,0,200,50)
    
    # Create open file dialog
    openFileDialog = wx.FileDialog(frame3, "Select Multiple F.Bin files", "", "", 
          "Bin files (*.bin)|*.bin", 
           wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)
    
    openFileDialog.ShowModal()
    Bin_path_list = openFileDialog.GetPaths()
    openFileDialog.Destroy()
    
    # image bin size setting
    pixel_size = 114 # unit is nm
    image_bin_size = 4 # unit is pixel,This is radial profile plot,diameter
    # true_bin_size = int(pixel_size*image_bin_size/2)
    bin_num_radial = 12
    fixed_D_fast = 1.7 # radial profiling fixed background
    fixed_D_fast_torr = 0.3 # background torrelation, fixed_D_fast+-fixed_D_fast_torr
    valid_ave = 120  # the average valid value during DBScan
    diameter_thre = 50 # if diameter < diameter_thre, filter it, unit nm
    diameter_thre_large = 800 # if diameter > diameter_thre_large, filter it, unit nm
    
    # minimal point number to perform fitting
    min_points_percell = 30
    
    # sortig into an arry first to speed up, maximum points per cell
    sorting_thre = 30000
    
    # 0, do not add; 1, add one center point; 3, add three points
    mode_trigger = 1
    disp_thre = 5
    time_interval = 0.00613 # unit is s, so 1 ms
    
    # histogram binning setting
    hist_bin_num = 4*disp_thre # must be an interger
    hist_bin_size = disp_thre/hist_bin_num
    bins_x = np.arange(0.5*hist_bin_size, disp_thre, hist_bin_size, dtype=np.float32)
    bin_thre = int(floor((disp_thre*0.6)/hist_bin_size))  # for guessed initial fitting parameters
    
    # pca analysis threshold
    pca_ratio_thre = 0.99
    
    # fitting error threshold
    fitting_error = 0.5
    
    # CPU count
    cpu_used = int(cpu_count()/2)
    
    data_type = np.dtype([('x_inter', np.int16),('y_inter', np.int16),('frame', np.int32),
                          ('disp', np.float32), ('angle', np.float32), 
                          ('xc', np.float32), ('yc', np.float32)])
    
    saved_data_type = np.dtype([('xc', np.float32), ('yc', np.float32),('frame', np.int32),
                          ('diffusion', np.float32), ('angle', np.float32), ('points', np.int32)])
    
    file_index = 0
    
    for file_path in file_path_list:
    
        start_time = time.time()
        df = pd.read_excel(Excel_path_list[file_index], header=None)  # 不读取列名
        excel_array = df.to_numpy()
        excel_array = np.transpose(excel_array)
        # 1-9列
        # out_radius;out_avgx;out_sdx;out_avgy;out_sdy;out_avgZ;out_stdZ;out_num;frame
       
        # read the data
        frame_num, number_of_points, STORM_npdata = read_storm_bin_XcYcZZcFrame(file_path)
        
        # data_type = np.dtype([('x_start', np.float32), ('y_start', np.float32),
        #                   ('x_end', np.float32), ('y_end', np.float32),
        #                   ('frame', np.int32),
        #                   ('disp', np.float32), ('angle', np.float32)])    
        
        frame_num = frame_num.item()
        number_of_points = number_of_points.item()
        
        image_size_x = max(np.amax(STORM_npdata['x_start']),np.amax(STORM_npdata['x_end']))
        image_size_y = max(np.amax(STORM_npdata['y_start']),np.amax(STORM_npdata['y_end']))
        
        xinter_max = int(ceil(image_size_x/image_bin_size))
        yinter_max = int(ceil(image_size_y/image_bin_size))
        
        counter_map = np.zeros((xinter_max,yinter_max),dtype=np.uint16)
        indice_map = np.zeros((xinter_max,yinter_max,sorting_thre),dtype=np.int32)-1
            
        xinter_yinter_frame_disp_angle_xcyc = np.zeros((mode_trigger+2)*number_of_points,
                                                       dtype=data_type)
        temp_index = np.arange(mode_trigger+2)
        temp_index2 = np.stack((np.flip(temp_index)/(mode_trigger+1),\
                                temp_index/(mode_trigger+1)),axis=1)
        xinter_yinter_frame_disp_angle_xcyc['frame'] = np.tile(STORM_npdata['frame'], \
                                                            mode_trigger+2)
        xinter_yinter_frame_disp_angle_xcyc['disp'] = np.tile(STORM_npdata['disp'], \
                                                            mode_trigger+2)
        xinter_yinter_frame_disp_angle_xcyc['angle'] = np.tile(STORM_npdata['angle'], \
                                                            mode_trigger+2)
        xinter_yinter_frame_disp_angle_xcyc['xc'] = \
            ((np.dot(temp_index2,np.stack((STORM_npdata['x_start'],STORM_npdata['x_end']),axis=0))\
              .reshape(-1,1).squeeze(axis=1))/image_bin_size)
        xinter_yinter_frame_disp_angle_xcyc['yc'] = \
            ((np.dot(temp_index2,np.stack((STORM_npdata['y_start'],STORM_npdata['y_end']),axis=0))\
              .reshape(-1,1).squeeze(axis=1))/image_bin_size)
        xinter_yinter_frame_disp_angle_xcyc['x_inter'] = \
            xinter_yinter_frame_disp_angle_xcyc['xc'].astype(np.int16)
        xinter_yinter_frame_disp_angle_xcyc['y_inter'] = \
            xinter_yinter_frame_disp_angle_xcyc['yc'].astype(np.int16)
        
        # sorting start
        current_sort = 0
        while current_sort < (mode_trigger+2)*number_of_points:
            x = xinter_yinter_frame_disp_angle_xcyc['x_inter'][current_sort]
            y = xinter_yinter_frame_disp_angle_xcyc['y_inter'][current_sort]
            if counter_map[x,y] < sorting_thre:
                indice_map[x,y,counter_map[x,y]] = current_sort
                counter_map[x,y] = counter_map[x,y] + 1
                current_sort = current_sort + 1
            else:
                current_sort = current_sort + 1
        # sorting finished
        
        # order_indice = np.lexsort((xinter_yinter_frame_disp_angle_xcyc['y_inter'],\
        #                            xinter_yinter_frame_disp_angle_xcyc['x_inter']))
            
        # xinter_yinter_frame_disp_angle_xcyc = xinter_yinter_frame_disp_angle_xcyc[order_indice]
        
        # all the x,y unit is pixel, and the pixel size is defined as above:
        # if pixel_size = 160 # unit is nm
        # image_bin_size = 0.625 # unit is pixel
        # then here the unit is 0.625*160 = 100 nm
        # averaged_xc_yc_frame_diffusion_angle = np.zeros(2*number_of_points,dtype=saved_data_type)
        
        # 1-8列
        # out_radius;out_avgx;out_sdx;out_avgy;out_sdy;out_avgZ;out_stdZ;out_num
            
        print('Multi-processes histogram_making...')
        with Pool(processes = cpu_used) as pool:
            cluster_array = pool.map(partial(hist_maker_radial,disp_thre=disp_thre,
                                                  hist_bin_num=hist_bin_num,
                                                  image_bin_size=image_bin_size,
                                                  xinter_yinter_frame_disp_angle_xcyc=xinter_yinter_frame_disp_angle_xcyc,
                                                  indice_map=indice_map,counter_map=counter_map,
                                                  xinter_max=xinter_max,yinter_max=yinter_max,
                                                  min_points_percell=min_points_percell,
                                                  pca_ratio_thre=pca_ratio_thre,
                                                  bin_num_radial=bin_num_radial,
                                                  pixel_size=pixel_size,
                                                  time_interval=time_interval,
                                                  fixed_D_fast=fixed_D_fast,
                                                  fixed_D_fast_torr=fixed_D_fast_torr),excel_array)
            pool.close()
            
        cluster_array = np.array(cluster_array)
        # diameter_thre
        mask = ((cluster_array[:,5]>diameter_thre)&(cluster_array[:,5]<diameter_thre_large))
        cluster_array = cluster_array[mask,:]
        # mask = cluster_array[:,4]>0
        # cluster_array = cluster_array[mask,:]
        # averaged_xc_start 0,averaged_yc_start 1,averaged_zc_start 2,
        # averaged_points_start 3,fitted_zc_start 4,fitted_diameter 5,fitted_zc_bg 6,
        # averaged_frame 7
        
        #===========根据直径和Dslow进一步过滤===============
        # ratio threshold <1-e**(-0.00366*diameter), 100 nm 1/3, 200 nm 1/2
        mask2 = (cluster_array[:,4]/cluster_array[:,6])<(1-e**(-0.005*cluster_array[:,5]))
        cluster_array = cluster_array[mask2,:]
        
        # 添加valid_ave
        new_column = np.full((cluster_array.shape[0], 1), valid_ave)  # 创建新的一列
        cluster_array = np.hstack((cluster_array, new_column))        # 水平拼接
        #==============作图部分==================#

        # 设定参数
        lower_4 = 0       # 第4列（索引3）下限 扩散速率
        upper_4 = 2       # 第4列上限
        bins_4 = 20       # 第4列 bin 数
        
        lower_5 = 0      # 第5列（索引4）下限 直径
        upper_5 = 1000       # 第5列上限
        bins_5 = 20       # 第5列 bin 数
        
        # 提取列
        data_4 = cluster_array[:, 4]
        data_5 = cluster_array[:, 5]
        
        # 绘制第4列直方图
        plt.figure(figsize=(6, 4))
        plt.hist(data_4, bins=bins_4, range=(lower_4, upper_4), color='skyblue', edgecolor='black')
        plt.title("Histogram of cluster_array[:, 4] D_slow")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        
        # 绘制第5列直方图
        plt.figure(figsize=(6, 4))
        plt.hist(data_5, bins=bins_5, range=(lower_5, upper_5), color='salmon', edgecolor='black')
        plt.title("Histogram of cluster_array[:, 5] Diameter")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        #========保存数据==============#
        print('Data Saving...')
        df = pd.DataFrame(cluster_array)
        # file_path_excel = file_path[0:-4]
        # 1. 提取目录、文件名和扩展名
        folder, filename = os.path.split(file_path)
        base_name, ext = os.path.splitext(filename)
        
        # 2. 构建新的文件名，拼接变量
        new_name = f"{base_name}_s{image_bin_size:.1f}_n{bin_num_radial}_D{fixed_D_fast:.1f}.xlsx"
        
        # 3. 拼接完整路径
        new_path = os.path.join(folder, new_name)
        
        # 4. 保存 Excel 文件
        df = pd.DataFrame(cluster_array)
        df.to_excel(new_path, index=False, header=False)
        
        # image_bin_size = 4 # unit is pixel,This is radial profile plot,diameter
        # bin_num_radial = 12
        # fixed_D_fast = 2.1 # radial profiling fixed background
        
        #========读取Bin文件并进一步过滤Bin文件==============#
        # read the data
        total_frame, total_mol_num, original_mol_list = read_storm_bin(Bin_path_list[file_index])
        total_frame = total_frame.item()
        total_mol_num = total_mol_num.item()
        
        # 提取 cluster_array 中的 frame 值，确保类型一致
        excluded_frames = cluster_array[:, 7].astype(original_mol_list['Frame'].dtype)
        
        # 更新 mask：Category==0，Frame 不在 cluster_array 中，且 Frame ≠ 0
        mask = (
            (original_mol_list['Category'] != 3) &
            (original_mol_list['Frame'] != 1) &
            (~np.isin(original_mol_list['Frame'], excluded_frames))
        )
        
        # 将符合条件的 Category 设置为 2
        original_mol_list['Category'][mask] = 2
        
        newbin_file_path = Bin_path_list[file_index].replace('.bin','-F2.bin')
        Write_STORMbin_Packed(newbin_file_path, total_mol_num, total_frame-2, original_mol_list)
        
        file_index+=1

        
        
        
        
    