# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 09:06:15 2025

@author: Admin
"""

import numpy as np
from ReadSTORMBin_XcYcZZcFrame import read_storm_bin_XcYcZZcFrame
from math import ceil,sqrt,e,floor
from scipy import optimize
import matplotlib.pyplot as plt
import time
import wx

def oneD_diff_dis_func(x,a,b,c):
    
    return c*np.exp(-x**2/a)+b

def oneD_diff_EP_func(x,a,b,c,x0):
    
    return c*np.exp(-(x-x0)**2/a)+b

if __name__=="__main__":
    
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
    
    # image bin size setting
    pixel_size = 114 # unit is nm
    disp_threX = 4 # displacement in x
    disp_threY = 4 # displacement in y
    time_interval = 0.00769 # unit is s, so 1 ms
    period_frame = 650 # every 650 switch the voltage
    cut_frame = 200 # cut the first XX frame in the period frame
    
    # histogram binning setting
    multiplier = 3
    hist_bin_numY = int(2*multiplier*disp_threY) # must be an interger
    hist_bin_numX = int(2*multiplier*disp_threX) # must be an interger
    hist_bin_size_1D = 1/multiplier
    bins_1D_x = np.arange(-disp_threX+0.5*hist_bin_size_1D, disp_threX, hist_bin_size_1D, dtype=np.float32)
    bins_1D_y = np.arange(-disp_threY+0.5*hist_bin_size_1D, disp_threY, hist_bin_size_1D, dtype=np.float32)
    
    # fitting error threshold
    fitting_error = 0.5
    
    data_type = np.dtype([('x_inter', np.int16),('y_inter', np.int16),('frame', np.int32),
                          ('disp', np.float32), ('angle', np.float32), 
                          ('xc', np.float32), ('yc', np.float32)])
    
    saved_data_type = np.dtype([('xc', np.float32), ('yc', np.float32),('frame', np.int32),
                          ('diffusion', np.float32), ('angle', np.float32), ('points', np.int32)])
    
    start_time = time.time()
   
    # read the data
    frame_num, number_of_points, STORM_npdata = read_storm_bin_XcYcZZcFrame(file_path_list[0])
    
    # data_type = np.dtype([('x_start', np.float32), ('y_start', np.float32),
    #                   ('x_end', np.float32), ('y_end', np.float32),
    #                   ('frame', np.int32),
    #                   ('disp', np.float32), ('angle', np.float32)])    
    # disp is displacement in X, angle is displacement in Y
    
    frame_num = frame_num.item()
    number_of_points = number_of_points.item()
    
    # -------- 根据 period_frame 和 cut_frame 划分索引 --------
    frames = STORM_npdata['frame']
    
    # 计算周期编号（从0开始）
    period_idx = (frames - 1) // period_frame  # 整除，0,1,2,3...
    
    # 在周期内的位置
    frame_in_period = (frames - 1) % period_frame + 1
    
    # 筛选掉 cut_frame 前的帧
    valid_mask = (frame_in_period > cut_frame) & (frame_in_period < period_frame)
    
    # 奇数组（周期编号为奇数：1,3,5...）和偶数组（0,2,4...）
    odd_mask = valid_mask & (period_idx % 2 == 1)
    even_mask = valid_mask & (period_idx % 2 == 0)
    
    # 获得索引
    odd_indices = np.where(odd_mask)[0]
    even_indices = np.where(even_mask)[0]
    
    # print(f"✅ Odd group: {len(odd_indices)} points, Even group: {len(even_indices)} points")
    
    # 如果需要，你可以直接访问数据：
    odd_group_data = STORM_npdata[odd_indices]
    even_group_data = STORM_npdata[even_indices]
    
    # ========== 直方图统计 ==========
    # Odd Group Disp Histogram
    odd_disp = odd_group_data['disp']
    even_disp = even_group_data['disp']
    angle_all = STORM_npdata['angle']
    
    # 统计 Odd Disp
    counts_odd, bins_odd = np.histogram(
        odd_disp, bins=hist_bin_numX, range=(-disp_threX, disp_threX))
    bin_centers_odd = (bins_odd[:-1] + bins_odd[1:]) / 2
    
    # 统计 Even Disp
    counts_even, bins_even = np.histogram(
        even_disp, bins=hist_bin_numX, range=(-disp_threX, disp_threX))
    bin_centers_even = (bins_even[:-1] + bins_even[1:]) / 2
    
    # 统计 Angle
    counts_angle, bins_angle = np.histogram(
        angle_all, bins=hist_bin_numY, range=(-disp_threY, disp_threY))
    bin_centers_angle = (bins_angle[:-1] + bins_angle[1:]) / 2
    
    # ========== 高斯拟合 ==========
    def fit_hist(bin_centers, counts):
        # 初始参数: a, b, c
        a_init = np.var(bin_centers)
        b_init = np.min(counts)
        c_init = np.max(counts) - b_init
        p0 = [a_init, b_init, c_init]
        bounds = ([1e-6, 0, 0], [np.inf, np.inf, np.inf])
    
        try:
            popt, _ = optimize.curve_fit(
                oneD_diff_dis_func, bin_centers, counts,
                p0=p0, bounds=bounds)
        except:
            popt = [0, 0, 0]
        return popt
    
    def fit_hist_EP(bin_centers, counts):
        # 初始参数: a, b, c
        a_init = np.var(bin_centers)
        b_init = np.min(counts)
        c_init = np.max(counts) - b_init
        x0_init = 0
        p0 = [a_init, b_init, c_init, x0_init]
        bounds = ([1e-6, 0, 0, -disp_threX], [np.inf, np.inf, np.inf, disp_threX])
    
        try:
            popt, _ = optimize.curve_fit(
                oneD_diff_EP_func, bin_centers, counts,
                p0=p0, bounds=bounds)
        except:
            popt = [0, 0, 0, 0]
        return popt
    
    popt_odd = fit_hist_EP(bin_centers_odd, counts_odd)
    popt_even = fit_hist_EP(bin_centers_even, counts_even)
    popt_angle = fit_hist(bin_centers_angle, counts_angle)
    
    # 拟合曲线
    x_fit_odd = np.linspace(-disp_threX, disp_threX, 300)
    y_fit_odd = oneD_diff_EP_func(x_fit_odd, *popt_odd)
    y_fit_even = oneD_diff_EP_func(x_fit_odd, *popt_even)
    
    x_fit_angle = np.linspace(-disp_threY, disp_threY, 300)
    y_fit_angle = oneD_diff_dis_func(x_fit_angle, *popt_angle)
    
    # ========== 计算结果 ==========
    
    drate_odd = popt_odd[0] * (pixel_size/1000)**2 /4/time_interval
    drate_even = popt_even[0] * (pixel_size/1000)**2 /4/time_interval
    drate_angle = popt_angle[0] * (pixel_size/1000)**2 /4/time_interval
    EP_speed = (popt_odd[3] - popt_even[3]) * (pixel_size/1000)/time_interval/2
    print(f"✅ Odd D={drate_odd:.3f} um2/s\n Even D={drate_even:.3f} \n Y D={drate_angle:.3f}")
    print(f"✅ EP S={EP_speed:.3f} um/s")
    
    # ========== 绘图 ==========
    # --- 图1：Odd & Even Disp ---
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers_odd, counts_odd, width=bin_centers_odd[1]-bin_centers_odd[0],
            color='skyblue', alpha=0.5, label='Odd Disp')
    plt.bar(bin_centers_even, counts_even, width=bin_centers_even[1]-bin_centers_even[0],
            color='salmon', alpha=0.5, label='Even Disp')
    
    plt.plot(x_fit_odd, y_fit_odd, 'b--', linewidth=2, label='Odd Fit')
    plt.plot(x_fit_odd, y_fit_even, 'r--', linewidth=2, label='Even Fit')
    plt.legend(fontsize=12)
    plt.title("Disp Histogram & Gaussian Fit")
    plt.xlabel("Disp")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.show()
    
    # --- 图2：Angle ---
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers_angle, counts_angle, width=bin_centers_angle[1]-bin_centers_angle[0],
            color='gray', alpha=0.5, label='Y-dis')
    plt.plot(x_fit_angle, y_fit_angle, 'k--', linewidth=2, label='Y-dis Fit')
    plt.legend(fontsize=12)
    plt.title("Y-dis Histogram & Gaussian Fit")
    plt.xlabel("Y-dis")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.show()
    

    
    

        
        
        
        
        