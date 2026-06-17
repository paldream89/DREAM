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
from functools import partial
from multiprocessing import Pool,freeze_support,cpu_count
from hist_maker_PCA_shrinkbinrange import hist_maker_x_y
from drate_fitter_PCA_shrinkbinrange import drate_fitter_x_y
from WriteSTORMBin import Write_STORMbin

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
    cut_frame = 0 # cut the first XX frame in the period frame
    image_bin_size = 2 # unit is pixel
    true_bin_size = int(pixel_size*image_bin_size/2) 
    # it is also the diameter for bleeding_Gaussian
    # image_bin_size is designed to be 2*FWHM
    mode_trigger = 1 # 0, do not add; 1, add one center point; 3, add three points
    
    # minimal point number to perform fitting
    min_points_percell = 30
    
    # sortig into an arry first to speed up, maximum points per cell
    sorting_thre = 1000
    
    # histogram binning setting
    multiplierX = 3
    hist_bin_numX = int(2*multiplierX*disp_threX) # must be an interger
    multiplierY = 3
    hist_bin_numY = int(2*multiplierY*disp_threY) # must be an interger

    hist_bin_size_1D_X = 1/multiplierX
    bins_1D_x = np.arange(-disp_threX+0.5*hist_bin_size_1D_X, disp_threX, hist_bin_size_1D_X, dtype=np.float32)
    hist_bin_size_1D_Y = 1/multiplierY
    bins_1D_y = np.arange(-disp_threY+0.5*hist_bin_size_1D_Y, disp_threY, hist_bin_size_1D_Y, dtype=np.float32)
    
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
    number_of_points_odd = np.size(odd_indices)
    number_of_points_even = np.size(even_indices)
    
    image_size_x = max(np.amax(STORM_npdata['x_start']),np.amax(STORM_npdata['x_end']))
    image_size_y = max(np.amax(STORM_npdata['y_start']),np.amax(STORM_npdata['y_end']))

    xinter_max = int(ceil(image_size_x/image_bin_size))
    yinter_max = int(ceil(image_size_y/image_bin_size))
    
    # need four counter_map, indice_map.
    # For points in STOMnp_data, needs its two histograms in odd frames and even frames respectively

    # sorting odd and even respectively, odd first
    counter_map_odd = np.zeros((xinter_max,yinter_max),dtype=np.uint16)
    indice_map_odd = np.zeros((xinter_max,yinter_max,sorting_thre),dtype=np.int32)-1
        
    xinter_yinter_frame_disp_angle_xcyc_odd = np.zeros((mode_trigger+2)*number_of_points_odd,
                                                   dtype=data_type)
    temp_index = np.arange(mode_trigger+2)
    temp_index2 = np.stack((np.flip(temp_index)/(mode_trigger+1),\
                            temp_index/(mode_trigger+1)),axis=1)
    xinter_yinter_frame_disp_angle_xcyc_odd['frame'] = np.tile(odd_group_data['frame'], \
                                                        mode_trigger+2)
    xinter_yinter_frame_disp_angle_xcyc_odd['disp'] = np.tile(odd_group_data['disp'], \
                                                        mode_trigger+2)
    xinter_yinter_frame_disp_angle_xcyc_odd['angle'] = np.tile(odd_group_data['angle'], \
                                                        mode_trigger+2)
    xinter_yinter_frame_disp_angle_xcyc_odd['xc'] = \
        ((np.dot(temp_index2,np.stack((odd_group_data['x_start'],odd_group_data['x_end']),axis=0))\
          .reshape(-1,1).squeeze(axis=1))/image_bin_size)
    xinter_yinter_frame_disp_angle_xcyc_odd['yc'] = \
        ((np.dot(temp_index2,np.stack((odd_group_data['y_start'],odd_group_data['y_end']),axis=0))\
          .reshape(-1,1).squeeze(axis=1))/image_bin_size)
    xinter_yinter_frame_disp_angle_xcyc_odd['x_inter'] = \
        xinter_yinter_frame_disp_angle_xcyc_odd['xc'].astype(np.int16)
    xinter_yinter_frame_disp_angle_xcyc_odd['y_inter'] = \
        xinter_yinter_frame_disp_angle_xcyc_odd['yc'].astype(np.int16)
    
    current_sort = 0
    while current_sort < (mode_trigger+2)*number_of_points_odd:
        x = xinter_yinter_frame_disp_angle_xcyc_odd['x_inter'][current_sort]
        y = xinter_yinter_frame_disp_angle_xcyc_odd['y_inter'][current_sort]
        if counter_map_odd[x,y] < sorting_thre:
            indice_map_odd[x,y,counter_map_odd[x,y]] = current_sort
            counter_map_odd[x,y] = counter_map_odd[x,y] + 1
            current_sort = current_sort + 1
        else:
            current_sort = current_sort + 1
            
    # sorting odd and even respectively, then even

    counter_map_even = np.zeros((xinter_max,yinter_max),dtype=np.uint16)
    indice_map_even = np.zeros((xinter_max,yinter_max,sorting_thre),dtype=np.int32)-1
        
    xinter_yinter_frame_disp_angle_xcyc_even = np.zeros((mode_trigger+2)*number_of_points_even,
                                                   dtype=data_type)
    temp_index = np.arange(mode_trigger+2)
    temp_index2 = np.stack((np.flip(temp_index)/(mode_trigger+1),\
                            temp_index/(mode_trigger+1)),axis=1)
    xinter_yinter_frame_disp_angle_xcyc_even['frame'] = np.tile(even_group_data['frame'], \
                                                        mode_trigger+2)
    xinter_yinter_frame_disp_angle_xcyc_even['disp'] = np.tile(even_group_data['disp'], \
                                                        mode_trigger+2)
    xinter_yinter_frame_disp_angle_xcyc_even['angle'] = np.tile(even_group_data['angle'], \
                                                        mode_trigger+2)
    xinter_yinter_frame_disp_angle_xcyc_even['xc'] = \
        ((np.dot(temp_index2,np.stack((even_group_data['x_start'],even_group_data['x_end']),axis=0))\
          .reshape(-1,1).squeeze(axis=1))/image_bin_size)
    xinter_yinter_frame_disp_angle_xcyc_even['yc'] = \
        ((np.dot(temp_index2,np.stack((even_group_data['y_start'],even_group_data['y_end']),axis=0))\
          .reshape(-1,1).squeeze(axis=1))/image_bin_size)
    xinter_yinter_frame_disp_angle_xcyc_even['x_inter'] = \
        xinter_yinter_frame_disp_angle_xcyc_even['xc'].astype(np.int16)
    xinter_yinter_frame_disp_angle_xcyc_even['y_inter'] = \
        xinter_yinter_frame_disp_angle_xcyc_even['yc'].astype(np.int16)
    
    current_sort = 0
    while current_sort < (mode_trigger+2)*number_of_points_even:
        x = xinter_yinter_frame_disp_angle_xcyc_even['x_inter'][current_sort]
        y = xinter_yinter_frame_disp_angle_xcyc_even['y_inter'][current_sort]
        if counter_map_even[x,y] < sorting_thre:
            indice_map_even[x,y,counter_map_even[x,y]] = current_sort
            counter_map_even[x,y] = counter_map_even[x,y] + 1
            current_sort = current_sort + 1
        else:
            current_sort = current_sort + 1

    # even sorting finished
    
    # disp is displacement in X, angle is displacement in Y
    
    # histogram making started
    # averaged_hist_array includes:
    # histogram at 4-24 (if bin number is 20) in x-axis for odd frame
    # histogram at 24-44 (if bin number is 20) in y-axis for odd frame
    # histogram at 44-64 (if bin number is 20) in x-axis for even frame
    # histogram at 64-84 (if bin number is 20) in y-axis for even frame
    # averaged_xc,averaged_yc,
    # averaged_frame,averaged_points_odd+averaged_points_even,
    # averaged_histogram_oddX(4:4+hist_bin_numX),
    # averaged_histogram_oddY(4+hist_bin_numX:4+hist_bin_numX+hist_bin_numY),
    # averaged_histogram_evenX(4+hist_bin_numX+hist_bin_numY:4+hist_bin_numX*2+hist_bin_numY),
    # averaged_histogram_evenY(4+hist_bin_numX*2+hist_bin_numY:4+hist_bin_numX*2+hist_bin_numY*2)
    
    print('Multi-processes histogram_making...')
    with Pool(processes = cpu_count()) as pool:
        averaged_hist_array = pool.map(partial(hist_maker_x_y,disp_threX=disp_threX,
                                               disp_threY=disp_threY,
                                               hist_bin_numX=hist_bin_numX,
                                               hist_bin_numY=hist_bin_numY,
                                               image_bin_size=image_bin_size,
                                              xinter_yinter_frame_disp_angle_xcyc_odd=xinter_yinter_frame_disp_angle_xcyc_odd,
                                              xinter_yinter_frame_disp_angle_xcyc_even=xinter_yinter_frame_disp_angle_xcyc_even,
                                              indice_map_odd=indice_map_odd,counter_map_odd=counter_map_odd,
                                              indice_map_even=indice_map_even,counter_map_even=counter_map_even,
                                              xinter_max=xinter_max,yinter_max=yinter_max,
                                              min_points_percell=min_points_percell/2),STORM_npdata)
        pool.close()
    
    stacked = np.stack(averaged_hist_array, axis=0)  # shape: (n, j, 2)
    averaged_hist_array_R = stacked.transpose(0, 2, 1).reshape(-1, stacked.shape[1])  # shape: (2n, j)
    
    print('Multi-processes drate fitting...')
    with Pool(processes = cpu_count()) as pool:
        averaged_drate = pool.map(partial(drate_fitter_x_y,disp_threX=disp_threX,
                                          disp_threY=disp_threY,
                                                hist_bin_numX=hist_bin_numX,
                                                hist_bin_numY=hist_bin_numY,
                                                pixel_size=pixel_size,
                                                min_points_percell=min_points_percell,
                                                time_interval=time_interval,
                                                fitting_error=fitting_error), averaged_hist_array_R)
        pool.close()
        
    # drate_Y,drate_X,EP_X
    print('Data Saving...')
    
    # convert list to np array
    averaged_drate = np.array(averaged_drate,dtype=np.float32)
    
    remove_unuseful_indice = averaged_drate[:,0] > 0
    averaged_drate = averaged_drate[remove_unuseful_indice]
    useful_number_of_points = np.sum(remove_unuseful_indice)
    averaged_hist_array_R = averaged_hist_array_R[remove_unuseful_indice]
    
    xc_array = averaged_hist_array_R[:,0]
    yc_array = averaged_hist_array_R[:,1]
    height_array = np.ones(useful_number_of_points, dtype=np.float32)*206.031
    area_array = np.ones(useful_number_of_points, dtype=np.float32)*968.889
    width_array = np.ones(useful_number_of_points, dtype=np.float32)*279.095
    phi_array = np.zeros(useful_number_of_points, dtype=np.float32)
    ax_array = np.ones(useful_number_of_points, dtype=np.float32)
    bg_array = averaged_drate[:,1]   # diffusion rate on X
    I_array = np.ones(useful_number_of_points, dtype=np.float32)*1476.02
    category_array = np.zeros(useful_number_of_points, dtype=np.int32)
    valid_array = np.array(averaged_hist_array_R[:,3], dtype=np.int32)
    frame_array = np.array(averaged_hist_array_R[:,2], dtype=np.int32)
    length_array = np.ones(useful_number_of_points, dtype=np.int32)
    link_array = -np.ones(useful_number_of_points, dtype=np.int32)
    z_array = averaged_drate[:,0]  # diffusion rate on Y
    zc_array = averaged_drate[:,2]  # EP speed along X
    
    file_path=file_path_list[0]
    save_file_path = file_path.replace('.bin','-pc'+str(true_bin_size)+'nm'+'.bin')

    Write_STORMbin(save_file_path, np.int32(useful_number_of_points), int(frame_num+2),
                    xc_array,
                    yc_array, 
                    xc_array,
                    yc_array,
                    height_array, area_array, width_array, phi_array, ax_array, 
                    bg_array, 
                    I_array, category_array, 
                    valid_array,
                    frame_array, 
                    length_array, link_array, 
                    z_array, 
                    zc_array)
    
    save_file_hist_map = file_path.replace('.bin','-Hist'+'-Add'+str(mode_trigger)+\
                                                'pt-'+'bsize'+str(true_bin_size)+'nm'+\
                                                    '-bnum'+str(hist_bin_numX)+\
                                                        '-'+str(hist_bin_numY)+'.npy')
    np.save(save_file_hist_map,averaged_hist_array_R)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    print("--- %5.3f useful points percentage---" % (useful_number_of_points/2/number_of_points))
    
    
    
    #============plot test====================#
    # stp=1
    # edp=1000
    # hist = averaged_hist_array_R[stp:edp+1,:].sum(axis=0)
    # hist_bin_size_1D_X = 2*disp_threX/hist_bin_numX
    # bins_1D_x = np.arange(-disp_threX+0.5*hist_bin_size_1D_X, disp_threX, hist_bin_size_1D_X, dtype=np.float32)
    # hist_bin_size_1D_Y = 2*disp_threY/hist_bin_numY
    # bins_1D_y = np.arange(-disp_threY+0.5*hist_bin_size_1D_Y, disp_threY, hist_bin_size_1D_Y, dtype=np.float32)
    
    # def fit_hist(bin_centers, counts):
    #     # 初始参数: a, b, c
    #     a_init = np.var(bin_centers)
    #     b_init = np.min(counts)
    #     c_init = np.max(counts) - b_init
    #     p0 = [a_init, b_init, c_init]
    #     bounds = ([1e-6, 0, 0], [np.inf, np.inf, np.inf])
    
    #     try:
    #         popt, _ = optimize.curve_fit(
    #             oneD_diff_dis_func, bin_centers, counts,
    #             p0=p0, bounds=bounds)
    #     except:
    #         popt = [0, 0, 0]
    #     return popt
    
    # def fit_hist_EP(bin_centers, counts):
    #     # 初始参数: a, b, c
    #     a_init = np.var(bin_centers)
    #     b_init = np.min(counts)
    #     c_init = np.max(counts) - b_init
    #     x0_init = 0
    #     p0 = [a_init, b_init, c_init, x0_init]
    #     bounds = ([1e-6, 0, 0, -disp_threX], [np.inf, np.inf, np.inf, disp_threX])
    
    #     try:
    #         popt, _ = optimize.curve_fit(
    #             oneD_diff_EP_func, bin_centers, counts,
    #             p0=p0, bounds=bounds)
    #     except:
    #         popt = [0, 0, 0, 0]
    #     return popt
    
    # hist_Y = hist[4+hist_bin_numX:4+hist_bin_numX+hist_bin_numY]+hist[4+hist_bin_numX*2+hist_bin_numY:4+hist_bin_numX*2+hist_bin_numY*2]
    # popt_Y = fit_hist(bins_1D_y,hist_Y)
    # drate_Y = popt_Y[0] * (pixel_size/1000)**2 /4/time_interval
    
    # hist_X_odd = hist[4:4+hist_bin_numX]
    # hist_X_even = hist[4+hist_bin_numX+hist_bin_numY:4+hist_bin_numX*2+hist_bin_numY]
    # hist_X = hist_X_odd[::-1]+hist_X_even
    # popt_X = fit_hist_EP(bins_1D_x,hist_X)
    # drate_X = popt_X[0] * (pixel_size/1000)**2 /4/time_interval
    # EP_X = popt_X[3]* (pixel_size/1000)/time_interval
    
    # drate_array=np.array([drate_Y,drate_X,EP_X])
    
    # plt.figure(figsize=(8, 6))
    # plt.bar(bins_1D_y, hist_Y, width=bins_1D_y[1]-bins_1D_y[0],
    #         color='gray', alpha=0.5, label='Y-dis')
    # plt.legend(fontsize=12)
    # plt.title("Y-dis Histogram & Gaussian Fit")
    # plt.xlabel("Y-dis")
    # plt.ylabel("Counts")
    # plt.tight_layout()
    # plt.show()
    
    # plt.figure(figsize=(8, 6))
    # plt.bar(bins_1D_x, hist_X, width=bins_1D_x[1]-bins_1D_x[0],
    #         color='gray', alpha=0.5, label='X-dis')
    # plt.legend(fontsize=12)
    # plt.title("X-dis Histogram & Gaussian Fit")
    # plt.xlabel("X-dis")
    # plt.ylabel("Counts")
    # plt.tight_layout()
    # plt.show()
    
    # print(f"✅ X D={drate_X:.3f} um2/s\n Y D={drate_Y:.3f}")
    # print(f"✅ EP S={EP_X:.3f} um/s")
    #============plot test ends====================#


    
    

        
        
        
        
        