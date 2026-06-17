# -*- coding: utf-8 -*-

"""
对所有周期内同顺序帧进行分析
Created on Tue Jul 22 09:06:15 2025
@author: Admin
"""

import os
import numpy as np
from ReadSTORMBin_XcYcZZcFrame import read_storm_bin_XcYcZZcFrame
from math import ceil, sqrt, e, floor
from scipy import optimize
import matplotlib.pyplot as plt
import time
import wx

def oneD_diff_dis_func(x, a, b, c):
    return c * np.exp(-x**2 / a) + b

def oneD_diff_EP_func(x, a, b, c, x0):
    return c * np.exp(-(x - x0)**2 / a) + b

if __name__ == "__main__":

    app = wx.App()
    frame = wx.Frame(None, -1, 'win.py')
    frame.SetSize(0, 0, 200, 50)

    # 选择 bin 文件（只选择一次）
    openFileDialog = wx.FileDialog(frame, "Select Multiple Bin with dis files", "", "",
                                   "Bin files (*.bin)|*.bin",
                                   wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)
    openFileDialog.ShowModal()
    file_path_list = openFileDialog.GetPaths()
    openFileDialog.Destroy()

    if not file_path_list:
        print("No file selected. Exiting.")
        exit()

    # 输出目录
    bin_dir = os.path.dirname(file_path_list[0])
    output_dir = bin_dir
    os.makedirs(output_dir, exist_ok=True)

    # 固定参数（不随 cut_frame 变化）
    pixel_size = 114          # nm
    disp_threX = 1            # displacement in x
    disp_threY = 1            # displacement in y
    pulse= 200               #施加电压脉冲时间，单位ms
    fps= 400                  #相机每秒采集帧数
    time_interval = 1/fps # unit is s, so 1 ms=0.001s
    period_frame = int(pulse*fps/1000) # 反转电压的帧数周期
    cut_frame = 1 # cut the first XX frame in the period frame

    # 直方图参数
    multiplier = 16
    hist_bin_numY = int(2 * multiplier * disp_threY)
    hist_bin_numX = int(2 * multiplier * disp_threX)
    hist_bin_size_1D = 1 / multiplier
    bins_1D_x = np.arange(-disp_threX + 0.5 * hist_bin_size_1D, disp_threX, hist_bin_size_1D, dtype=np.float32)
    bins_1D_y = np.arange(-disp_threY + 0.5 * hist_bin_size_1D, disp_threY, hist_bin_size_1D, dtype=np.float32)

    fitting_error = 0.5

    data_type = np.dtype([('x_inter', np.int16), ('y_inter', np.int16), ('frame', np.int32),
                          ('disp', np.float32), ('angle', np.float32),
                          ('xc', np.float32), ('yc', np.float32)])

    saved_data_type = np.dtype([('xc', np.float32), ('yc', np.float32), ('frame', np.int32),
                                ('diffusion', np.float32), ('angle', np.float32), ('points', np.int32)])

    start_time = time.time()

    # 读取数据（只读一次）
    frame_num, number_of_points, STORM_npdata = read_storm_bin_XcYcZZcFrame(file_path_list[0])
    frame_num = frame_num.item()
    number_of_points = number_of_points.item()
    frames = STORM_npdata['frame']
    angle_all = STORM_npdata['angle']   # 注意：原代码 angle 统计未加筛选，这里保持不变

    # 定义拟合函数（内部使用）
    def fit_hist(bin_centers, counts):
        a_init = np.var(bin_centers)
        b_init = np.min(counts)
        c_init = np.max(counts) - b_init
        p0 = [a_init, b_init, c_init]
        bounds = ([1e-6, 0, 0], [np.inf, np.inf, np.inf])
        try:
            popt, _ = optimize.curve_fit(oneD_diff_dis_func, bin_centers, counts, p0=p0, bounds=bounds)
        except:
            popt = [0, 0, 0]
        return popt

    def fit_hist_EP(bin_centers, counts):
        a_init = np.var(bin_centers)
        b_init = np.min(counts)
        c_init = np.max(counts) - b_init
        x0_init = 0
        p0 = [a_init, b_init, c_init, x0_init]
        bounds = ([1e-6, 0, 0, -disp_threX], [np.inf, np.inf, np.inf, disp_threX])
        try:
            popt, _ = optimize.curve_fit(oneD_diff_EP_func, bin_centers, counts, p0=p0, bounds=bounds)
        except:
            popt = [0, 0, 0, 0]
        return popt

    # 用于保存所有 cut_frame 的结果
    results = []

    # ========== 主循环：cut_frame 从 0 到 72 ==========
    i=period_frame-2
    for cut_frame in range(0, i):
      
        # 根据当前 cut_frame 重新计算 cut_frame_end
        cut_frame_end = period_frame-2 - cut_frame

        # 计算周期编号和在周期内的位置
        period_idx = (frames - 1) // period_frame
        frame_in_period = (frames - 1) % period_frame + 1

        # 筛选有效帧
        valid_mask = (frame_in_period > cut_frame) & (frame_in_period < (period_frame - cut_frame_end))
        odd_mask = valid_mask & (period_idx % 2 == 1)
        even_mask = valid_mask & (period_idx % 2 == 0)

        odd_indices = np.where(odd_mask)[0]
        even_indices = np.where(even_mask)[0]

        odd_group_data = STORM_npdata[odd_indices]
        even_group_data = STORM_npdata[even_indices]

        # X直方图统计
        odd_disp = odd_group_data['disp']
        even_disp = even_group_data['disp']


        counts_odd, bins_odd = np.histogram(odd_disp, bins=hist_bin_numX, range=(-disp_threX, disp_threX))
        bin_centers_odd = (bins_odd[:-1] + bins_odd[1:]) / 2

        counts_even, bins_even = np.histogram(even_disp, bins=hist_bin_numX, range=(-disp_threX, disp_threX))
        bin_centers_even = (bins_even[:-1] + bins_even[1:]) / 2

        # Y直方图
        Y_odd = odd_group_data['angle']
        Y_even = even_group_data['angle']
        counts_Y_odd, bins_Y_odd = np.histogram(
            Y_odd, bins=hist_bin_numY, range=(-disp_threY, disp_threY))
        bin_centers_Y_odd = (bins_Y_odd[:-1] + bins_Y_odd[1:]) / 2
    
        counts_Y_even, bins_Y_even = np.histogram(
            Y_even, bins=hist_bin_numY, range=(-disp_threY, disp_threY))
        bin_centers_Y_even = (bins_Y_even[:-1] + bins_Y_even[1:]) / 2

        # 拟合
        popt_odd = fit_hist_EP(bin_centers_odd, counts_odd)
        popt_even = fit_hist_EP(bin_centers_even, counts_even)
        popt_Y_odd = fit_hist(bin_centers_Y_odd, counts_Y_odd)
        popt_Y_even = fit_hist(bin_centers_Y_even, counts_Y_even)

        # 计算结果
        drate_odd = popt_odd[0] * (pixel_size/1000)**2 /4/time_interval
        drate_even = popt_even[0] * (pixel_size/1000)**2 /4/time_interval
        drate_Y_odd = popt_Y_odd[0] * (pixel_size/1000)**2 /4/time_interval
        drate_Y_even = popt_Y_even[0] * (pixel_size/1000)**2 /4/time_interval
        EP_odd = popt_odd[3] * (pixel_size/1000)/time_interval/2
        EP_even = popt_even[3] * (pixel_size/1000)/time_interval/2
        
        frame_time= cut_frame*time_interval

       # print(f" {drate_odd:.3f} {drate_even:.3f} {drate_Y_odd:.3f} {drate_Y_even:.3f} {EP_odd:.3f} {EP_even:.3f}")
        # 保存结果
        results.append((cut_frame,frame_time, drate_odd, drate_even, drate_Y_odd, drate_Y_even, EP_odd, EP_even))
    import csv
    csv_path = os.path.join(output_dir, "frame_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame","frame_time", "X_odd", "X_even", "Y_odd", "Y_even","EP_odd","EP_even"])
        writer.writerows(results)
    print(f"\nAll results saved to: {csv_path}")

    elapsed_time = time.time() - start_time
    print(f"Total time: {elapsed_time:.2f} seconds")
  
    
    # ========== 绘图 ==========
    # 提取第1列（索引0）作为 x
    t = [row[1] for row in results]

    # 提取第6列（索引5）作为 y1
    EP1 = [row[6] for row in results]

    # 提取第7列（索引6）作为 y2
    EP2 = [row[7] for row in results]
 
    # --- 图1：Odd & Even EP ---
    plt.figure(figsize=(8, 6))
    plt.plot(t, EP1, 'b-o', linewidth=2, markersize=6, label='EP-odd')   # 蓝色实线+圆点
    plt.plot(t, EP2, 'r-o', linewidth=2, markersize=6, label='Ep-even')  # 红色实线+圆点
    plt.legend(fontsize=12)
    plt.title("EP-Time")
    plt.xlabel("Time(s)")
    plt.ylabel("EP(μm/s)")
    plt.tight_layout()
    plt.show()
    
    
    fig1_path = os.path.join(output_dir, "EP-Time")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"EP-Time已保存至：{fig1_path}")
    plt.show()
    
    

        
        
        
        
        
