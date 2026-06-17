# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 09:06:15 2025

@author: Admin
"""
import os   # 用于创建文件夹和路径操作
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
    
    # 获取第一个 bin 文件所在的文件夹路径（假设所有 bin 在同一目录）
    bin_dir = os.path.dirname(file_path_list[0])
    output_dir = bin_dir   # 输出文件夹就是 bin 文件所在文件夹
    os.makedirs(output_dir, exist_ok=True)   # 确保文件夹存在（一般已存在）
    
    # image bin size setting
    pixel_size = 114 # unit is nm
    disp_threX = 5 # displacement in x
    disp_threY = 5 # displacement in y 
    pulse= 25     #加电压脉冲时间，单位ms
    fps= 280  #相机每秒采集帧数数
    time_interval = 1/fps*5 # unit is s, so 1 ms=0.001s 
    # 需考虑sep_array = [5] 的设置
    period_frame = int(pulse/1000*fps) # 反转电压的帧数周期
    cut_frame = 0 # cut the first XX frame in the period frame
    cut_frame_end = 5 # cut the last XX frame in the period frame
    # 需考虑sep_array = [5] 的设置
    
    # histogram binning setting
    multiplier = 4
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
    valid_mask = (frame_in_period > cut_frame) & (frame_in_period < period_frame-cut_frame_end)
    
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
    Y_odd = odd_group_data['angle']
    Y_even = even_group_data['angle']
    
    # 统计 Odd Disp
    counts_odd, bins_odd = np.histogram(
        odd_disp, bins=hist_bin_numX, range=(-disp_threX, disp_threX))
    bin_centers_odd = (bins_odd[:-1] + bins_odd[1:]) / 2
    
    # 统计 Even Disp
    counts_even, bins_even = np.histogram(
        even_disp, bins=hist_bin_numX, range=(-disp_threX, disp_threX))
    bin_centers_even = (bins_even[:-1] + bins_even[1:]) / 2
    
    # 统计 Y_odd
    counts_Y_odd, bins_Y_odd = np.histogram(
        Y_odd, bins=hist_bin_numY, range=(-disp_threY, disp_threY))
    bin_centers_Y_odd = (bins_Y_odd[:-1] + bins_Y_odd[1:]) / 2
    
    # 统计 Y_even
    counts_Y_even, bins_Y_even = np.histogram(
        Y_even, bins=hist_bin_numY, range=(-disp_threY, disp_threY))
    bin_centers_Y_even = (bins_Y_even[:-1] + bins_Y_even[1:]) / 2
    
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
    popt_Y_odd = fit_hist(bin_centers_Y_odd, counts_Y_odd)
    popt_Y_even = fit_hist(bin_centers_Y_even, counts_Y_even)
    
    # 拟合曲线
    x_fit_odd = np.linspace(-disp_threX, disp_threX, 300)
    y_fit_odd = oneD_diff_EP_func(x_fit_odd, *popt_odd)
    y_fit_even = oneD_diff_EP_func(x_fit_odd, *popt_even)
    
    x_fit_Y_odd = np.linspace(-disp_threY, disp_threY, 300)
    y_fit_Y_odd = oneD_diff_dis_func(x_fit_Y_odd, *popt_Y_odd)
    y_fit_Y_even = oneD_diff_dis_func(x_fit_Y_odd, *popt_Y_even)
    
    # ========== 计算结果 ==========
    
    drate_odd = popt_odd[0] * (pixel_size/1000)**2 /4/time_interval
    drate_even = popt_even[0] * (pixel_size/1000)**2 /4/time_interval
    drate_Y_odd = popt_Y_odd[0] * (pixel_size/1000)**2 /4/time_interval
    drate_Y_even = popt_Y_even[0] * (pixel_size/1000)**2 /4/time_interval
    EP_speed = (popt_odd[3] - popt_even[3]) * (pixel_size/1000)/time_interval/2
    print(f"✅ Odd D={drate_odd:.3f} μm2/s\n Even D={drate_even:.3f} \n Y Odd D={drate_Y_odd:.3f} \n Y Even D={drate_Y_even:.3f}")
    print(f"✅ EP S={EP_speed:.3f} μm/s")
    
    # ==========保存结果到 TXT 文件============
    txt_path = os.path.join(output_dir, "Fitting_results.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=== SMdM & EP Speed Results ===\n")
        f.write(f"Odd D = {drate_odd:.3f} μm²/s\n")
        f.write(f"Even D = {drate_even:.3f} μm²/s\n")
        f.write(f"Y_odd D = {drate_Y_odd:.3f} μm²/s\n")
        f.write(f"Y_even D = {drate_Y_even:.3f} μm²/s\n")
        f.write(f"EP speed = {EP_speed:.3f} μm/s\n")
        f.write(f"Values: {drate_odd:.3f} {drate_even:.3f} {drate_Y_odd:.3f} {drate_Y_even:.3f} {EP_speed:.3f}\n\n")
        
        f.write("=== Input parameters ===\n")
        f.write(f"pixel_size = {pixel_size} nm\n")
        f.write(f"disp_threX = {disp_threX}\n")
        f.write(f"disp_threY = {disp_threY}\n")
        f.write(f"pulse = {pulse}\n")
        f.write(f"fps = {fps}\n")
        f.write(f"time_interval = {time_interval} s\n")
        f.write(f"period_frame = {period_frame}\n")
        f.write(f"cut_frame = {cut_frame}\n")
    print(f"文本结果已保存至：{txt_path}")
    
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
    plt.title("X-disp Histogram & Gaussian Fit")
    plt.xlabel("X-disp")
    plt.ylabel("Counts")
    plt.tight_layout()
    # 保存图1
    fig1_path = os.path.join(output_dir, "X-disp_Histogram.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"图1已保存至：{fig1_path}")
    plt.show()
    
    # --- 图2：Angle ---
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers_Y_odd, counts_Y_odd, width=bin_centers_Y_odd[1]-bin_centers_Y_odd[0],
            color='skyblue', alpha=0.5, label='Y-dis')
    plt.bar(bin_centers_Y_even, counts_Y_even, width=bin_centers_Y_even[1]-bin_centers_Y_even[0],
            color='salmon', alpha=0.5, label='Y-dis')
    plt.plot(x_fit_Y_odd, y_fit_Y_odd, 'b--', linewidth=2, label='Y odd-dis Fit')
    plt.plot(x_fit_Y_odd, y_fit_Y_even, 'r--', linewidth=2, label='Y even-dis Fit')
    plt.legend(fontsize=12)
    plt.title("Y-dis Histogram & Gaussian Fit")
    plt.xlabel("Y-dis")
    plt.ylabel("Counts")
    plt.tight_layout()
    # 保存图2
    fig2_path = os.path.join(output_dir, "Y-dis_Histogram.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"图2已保存至：{fig2_path}")
    plt.show()
    

    
    

        
        
        
        
        