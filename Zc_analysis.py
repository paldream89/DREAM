# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 20:51:39 2026

@author: Admin
"""

# read bin, plot Zc distribution, fit with Gaussian, Get peak position and Sigma
import numpy as np
from ReadSTORMBin import read_storm_bin
from scipy import optimize
import matplotlib.pyplot as plt
import time
import wx

def gaussian_func(x, A, mu, sigma, bg):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + bg

def gaussian_func_bg0(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

if __name__=="__main__":
    
    keyword = 'Zc'
    bin_number = 80
    
    # True: 自动设置上下限；False: 手动设置
    auto_limit = False
    lower_limit = 0
    upper_limit = 5
    
    # Gaussian fitting range (must be inside histogram range)
    fit_lower = 1
    fit_upper = 4
    
    # Gaussian fitting options
    fix_bg_zero = False   # True: bg固定为0, False: bg参与拟合
    
    app = wx.App()
    frame = wx.Frame(None, -1, 'win.py')
    frame.SetSize(0,0,200,50)
    
    openFileDialog = wx.FileDialog(frame, "Select Bin", "", "", 
          "Bin files (*.bin)|*.bin", 
           wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)
    
    openFileDialog.ShowModal()
    file_path_list = openFileDialog.GetPaths()
    openFileDialog.Destroy()
    
    num_of_bin = np.size(file_path_list)
    start_time = time.time()
    
    data_type = np.dtype([('X', np.float32), ('Y', np.float32), ('Xc', np.float32), ('Yc', np.float32),
               ('Height', np.float32), ('Area', np.float32), ('Width', np.float32), ('Phi', np.float32), 
               ('Ax', np.float32), ('BG', np.float32), ('I', np.float32), ('Category', np.int32), 
               ('Valid', np.int32), ('Frame', np.int32), ('Length', np.int32), ('Link', np.int32), 
               ('Z', np.float32), ('Zc', np.float32)])
    
    for file_path in file_path_list:
        
        total_frame, total_mol_num, original_mol_list = read_storm_bin(file_path)
        total_frame = int(total_frame.item())
        total_mol_num = int(total_mol_num.item())
        
        # 提取数据并去掉 nan / inf
        data_array = np.array(original_mol_list[keyword], dtype=np.float64)
        data_array = data_array[np.isfinite(data_array)]
        
        if data_array.size == 0:
            print(f'No valid data found in {file_path}')
            continue
        
        # 设置上下限
        if auto_limit:
            lower_limit = np.min(data_array)
            upper_limit = np.max(data_array)
        
        # 限制到作图范围内
        data_in_range = data_array[(data_array >= lower_limit) & (data_array <= upper_limit)]
        
        if data_in_range.size == 0:
            print(f'No data in selected range for {file_path}')
            continue
        
        # 直方图
        counts, bin_edges = np.histogram(data_in_range, bins=bin_number, range=(lower_limit, upper_limit))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 设置拟合范围
        fit_lower_use = lower_limit if fit_lower is None else fit_lower
        fit_upper_use = upper_limit if fit_upper is None else fit_upper
        
        # 只取拟合范围内的直方图数据
        fit_mask = (bin_centers >= fit_lower_use) & (bin_centers <= fit_upper_use)
        
        x_fit_data = bin_centers[fit_mask]
        y_fit_data = counts[fit_mask]
        
        # 初始值
        A_init = np.max(counts) - np.min(counts)
        mu_init = bin_centers[np.argmax(counts)]
        sigma_init = np.std(data_in_range)
        bg_init = np.min(counts)
        p0 = [A_init, mu_init, sigma_init, bg_init]
        
        # 参数范围
        lower_bounds = [0, lower_limit, 1e-6, 0]
        upper_bounds = [np.inf, upper_limit, np.inf, np.inf]
        
        try:
            popt, pcov = optimize.curve_fit(
                gaussian_func,
                x_fit_data,
                y_fit_data,
                p0=p0,
                bounds=(lower_bounds, upper_bounds)
            )
            
            A_fit, mu_fit, sigma_fit, bg_fit = popt
            FWHM = 2 * np.sqrt(2 * np.log(2)) * sigma_fit
            
            print(f'File: {file_path}')
            print(f'Peak position = {mu_fit:.6f}')
            print(f'Sigma = {sigma_fit:.6f}')
            print(f'FWHM = {FWHM:.6f}')
            
            # 作图
            x_fit = np.linspace(lower_limit, upper_limit, 1000)
            y_fit = gaussian_func(x_fit, *popt)
            
            plt.figure(figsize=(8, 6))
            plt.hist(data_in_range, bins=bin_number, range=(lower_limit, upper_limit),
                     color='skyblue', edgecolor='black', alpha=0.7)
            plt.plot(x_fit, y_fit, 'r-', linewidth=2)
            plt.xlabel(keyword)
            plt.ylabel('Counts')
            plt.title(f'{keyword} Distribution: {file_path}')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f'Gaussian fitting failed for {file_path}: {e}')
    
    print("--- %s seconds ---" % (time.time() - start_time))