# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 09:02:46 2025

@author: Admin
"""

import numpy as np
import pandas as pd
from math import ceil,sqrt,e,floor
import matplotlib.pyplot as plt
import time
import wx
# from WriteSTORMBin import Write_STORMbin
from scipy.optimize import curve_fit
import os
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.integrate import simpson

# ================注意===================
# 这里的是radial profiling计算直径

def twoD_diff_dis_func(x,a,b,c):
    
    return 2*c*x/a*np.exp(-x**2/a)+b*x

def gaussian_with_background(x, a1, x1, w1, y0):
    
    return a1 * np.exp(-((x - x1) ** 2) / (2 * w1 ** 2)) + y0

def exp_plus_gauss(x, a1, x1, w1, a2, b2):
    
    return a1 * np.exp(-((x - x1) ** 2) / (2 * w1 ** 2)) + a2 * np.exp(-b2 * x)

def gauss0_plus_gauss(x, a1, x1, w1, a2, w2):
    
    return a1 * np.exp(-((x - x1) ** 2) / (2 * w1 ** 2)) + a2 * np.exp(-(x ** 2) / (2 * w2 ** 2))

def gauss_only(x, a1, x1, w1):
    return a1 * np.exp(-((x - x1)**2) / (2 * w1**2))

def exp_only(x, a2, b2):
    return a2 * np.exp(-b2 * x)

def gauss0_only(x, a2, w2):
    return a2 * np.exp(-(x ** 2) / (2 * w2 ** 2))

if __name__=="__main__":
    
    D_statistics = 2.12 # 统计得到的扩散速率，针对温度影响测定结果的问题进行归一化 2.12
    filter_ratio = 1.5 # 如果觉得cluster count阈值设置太低了，还可以进一步筛选过滤
    
    # 设定参数
    lower_4 = 0      # 第4列（索引3）下限 扩散速率
    upper_4 = 2.0     # 第4列上限
    interval_4 = 0.4
    bins_4 = 20       # 第4列 bin 数
    interval_ycount = 100
    
    lower_5 = 0      # 第5列（索引4）下限 直径
    upper_5 = 800       # 第5列上限
    interval_5 = 200
    bins_5 = 20       # 第5列 bin 数
    
    lower_cluster_density = 0
    high_cluster_density = 6
    ytick_cluster_density = np.linspace(lower_cluster_density,high_cluster_density,5)
    
    color_deepblue = (61/255, 89/255, 171/255)
    
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
    
    df = pd.read_excel(Excel_path_list[0], header=None)  # 不读取列名
    excel_array = df.to_numpy()
    excel_array = np.transpose(excel_array)
    
    # 假设 excel_array 是原始二维数组 (至少有 8 行)
    data_raw = np.array(excel_array[:9, :],dtype=np.float32)  # 取前8行
    
    # 删除包含 NaN 的列
    valid_mask = ~np.isnan(data_raw).any(axis=0)
    clean_data = data_raw[:, valid_mask]
    
    # 构建结构化数据类型
    data_type = np.dtype([
        ('Xc', np.float32),
        ('Yc', np.float32),
        ('D_ave', np.float32),
        ('Count', np.int32),
        ('D_fit', np.float32),
        ('Diameter', np.float32),
        ('D_bg', np.float32),
        ('Frame', np.int32),
        ('Count_ave', np.int32)
    ])
    
    # 创建结构化数组并填入数据
    Data_array = np.zeros(clean_data.shape[1], dtype=data_type)
    Data_array['Xc']       = clean_data[0, :]
    Data_array['Yc']       = clean_data[1, :]
    Data_array['D_ave']    = clean_data[2, :]
    Data_array['Count']    = clean_data[3, :].astype(np.int32)
    Data_array['D_fit']    = clean_data[4, :]
    Data_array['Diameter'] = clean_data[5, :]
    Data_array['D_bg']     = clean_data[6, :]
    Data_array['Frame']    = clean_data[7, :].astype(np.int32)
    Data_array['Count_ave'] = clean_data[8, :].astype(np.int32)
    
    # 进一步筛选过滤
    mask = Data_array['Count']>filter_ratio*Data_array['Count_ave']
    Data_array = Data_array[mask]
    filter_fraction = np.sum(mask)/np.size(mask)
    
    #==========cluster density分析部分==========
    mask_cluster = []

    for idx, val in enumerate(excel_array[15,:]):
        if isinstance(val, str) and "length" in val:
            try:
                # next_val = excel_array[15,idx+1]
                mask_cluster.append(idx+1)
            except IndexError:
                continue
    # 构建结构化数据类型
    data_type_cluster = np.dtype([
        ('length', np.float32),
        ('width', np.float32),
        ('number', np.int32)
    ])
    
    # 创建结构化数组并填入数据
    Cluster_array = np.zeros(np.size(mask_cluster), dtype=data_type_cluster)
    Cluster_array['length'] = excel_array[15,mask_cluster]
    Cluster_array['width'] = excel_array[16,mask_cluster]
    Cluster_array['number'] = excel_array[17,mask_cluster].astype(np.int32)
    
    Cluster_density_array = Cluster_array['number']/Cluster_array['length']/Cluster_array['width']*filter_fraction
    
    #==============作图部分==================#
    color_blue = (0/255, 114/255, 178/255)
    color_red = (213/255, 94/255, 0/255)
    color_red2 = (255/255, 51/255, 51/255)
    color_purple = (204/255, 121/255, 167/255)
    color_orange = (230/255, 159/255, 0/255)
    color_green = (0/255, 158/255, 115/255)
    color_skyblue = (86/255, 180/255, 233/255)
    color_skyblue2 = (135/255, 206/255, 235/255)
    color_grey = (160/255, 160/255, 160/255)
    light_slate_gray = (230/255, 235/255, 245/255)
    very_light_pink = (255/255, 245/255, 250/255)
    lavender_gray = (220/255, 225/255, 235/255)
    mint_cream = (245/255, 255/255, 250/255)
    
    # 提取列
    data_4 = Data_array['D_fit']/Data_array['D_bg']*D_statistics
    # data_4 = Data_array['D_fit']
    data_5 = Data_array['Diameter'] # [Data_array['Diameter']>0]

    # 通用样式配置
    tick_fontsize = 30
    frame_linewidth = 5
    bar_edgecolor = 'black'
    bar_linewidth = 2
    box_color = 'salmon'
    dot_color = 'black'
    dot_size = 12
    tick_fontsize = 30
    
    # -------- 直方图 1 -------- D_fit, data_4
    fig, ax = plt.subplots(figsize=(9, 4))
    counts, bins, patches = ax.hist(data_4, bins=bins_4, range=(lower_4, upper_4),
                                 color=color_skyblue2, edgecolor='black', linewidth=2)
    
    #========对第4列直方图Data_4, Dfit_slow进行双组分拟合==============#
    # 计算 bin 中心
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # bin_centers[0] = 0.02
    # a1 * np.exp(-((x - x1) ** 2) / (2 * w1 ** 2)) + a2 * np.exp(-b2 * x)
    # a1 * np.exp(-((x - x1) ** 2) / (2 * w1 ** 2)) + a2 * np.exp(-(x ** 2) / (2 * w2 ** 2))
    
    # 初始值估计
    a1_init = np.median(counts)
    a2_init = np.max(counts)
    # a2_init = 0.0001
    x1_init = np.mean(bin_centers)
    w1_init = 0.2
    w2_init = 0.01
    p0 = [a1_init, x1_init, w1_init, a2_init, w2_init]
    
    # 拟合范围约束：所有参数大于 0
    bounds = ([0, 0, 0, a1_init, 0], [ np.inf, np.inf, 1, np.inf, np.inf])
    # bounds = ([0, 0, 0, 0, 0], [ np.inf, np.inf, 1, 0.0002, np.inf])
    
    # bin_centers[0] = 0.01
    # 执行拟合
    params, _ = curve_fit(gauss0_plus_gauss, bin_centers, counts, p0=p0, bounds=bounds)
    a1_fit, x1_fit, w1_fit, a2_fit, w2_fit = params
    
    # def gauss0_plus_gauss(x, a1, x1, w1, a2, w2):
    # return a1 * np.exp(-((x - x1) ** 2) / (2 * w1 ** 2)) + a2 * np.exp(-(x ** 2) / (2 * w2 ** 2))
    
    # 拟合曲线
    x_fit = np.linspace(bins[0], bins[-1], 500)
    y_total = gauss0_plus_gauss(x_fit, *params)
    y_gauss = gauss_only(x_fit, a1_fit, x1_fit, w1_fit)
    y_exp = gauss0_only(x_fit, a2_fit, w2_fit)
    
    # 绘制
    # ax.plot(x_fit, y_total, '--', color='black', linewidth=frame_linewidth)
    ax.plot(x_fit, y_gauss, '--', color='red', linewidth=frame_linewidth)
    ax.plot(x_fit, y_exp, '--', color='blue', linewidth=frame_linewidth)
    
    # 计算面积
    area_gauss = simpson(y=y_gauss, x=x_fit)
    area_exp = simpson(y=y_exp, x=x_fit)
    total_area = area_gauss + area_exp
    
    percent_gauss = 100 * area_gauss / total_area
    percent_exp = 100 * area_exp / total_area
    
    print(f"Gaussian area = {area_gauss:.2f} ({percent_gauss:.1f}%)")
    print(f"Exponential area = {area_exp:.2f} ({percent_exp:.1f}%)")
    
    # 各类参数设置
    # 设置整数 y 轴：使用 np.arange 明确指定整数刻度
    max_count = int(np.max(counts))
    yticks = np.arange(0, max_count + 1 + interval_ycount, interval_ycount)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks], fontsize=40, fontname='Arial')
    ax.set_ylim([0, max_count*1.1])
    
    # x 轴也一样处理（可选）
    xticks = np.linspace(lower_4, upper_4, int((upper_4 - lower_4) / interval_4) + 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks], fontsize=40, fontname='Arial')
    ax.set_xlim([lower_4-0.02, upper_4])
    
    # 去除文字标题、标签
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='both', width=frame_linewidth)
    
    # 控制边框：只保留左和下
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(frame_linewidth)
    
    plt.tight_layout(pad=0)
    plt.show()
    
    # -------- 直方图 2 --------  Diameter data_5
    fig, ax = plt.subplots(figsize=(9, 4))
    counts, bins, patches = ax.hist(data_5, bins=bins_5, range=(lower_5, upper_5), color='salmon',
            edgecolor=bar_edgecolor, linewidth=bar_linewidth)
    
    # 设置整数 y 轴：使用 np.arange 明确指定整数刻度
    max_count = int(np.max(counts))
    yticks = np.arange(0, max_count + 1, interval_ycount)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks], fontsize=30, fontname='Arial')
    
    # x 轴也一样处理（可选）
    xticks = np.arange(lower_5, upper_5 + 1, interval_5)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], fontsize=30, fontname='Arial')
    ax.set_xlim([lower_5-1, upper_5])
    
    # 美化坐标轴
    ax.set_xticklabels(ax.get_xticks(), fontsize=tick_fontsize, fontname='Arial')
    ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize, fontname='Arial')
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='both', width=frame_linewidth)
    
    # 边框控制
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(frame_linewidth)
    
    plt.tight_layout(pad=0)
    plt.show()
    
    # -------- Box plot --------
    fig, ax = plt.subplots(figsize=(3, 8))
    
    # 画 Box Plot，设置线条样式
    sns.boxplot(
        y=Cluster_density_array,
        color=box_color,
        width=0.6,
        fliersize=0,
        ax=ax,
        boxprops=dict(edgecolor='black', linewidth=5),
        whiskerprops=dict(color='black', linewidth=5),
        capprops=dict(color='black', linewidth=5),
        medianprops=dict(color='black', linewidth=5)
    )
    
    # 画散点图
    sns.stripplot(
        y=Cluster_density_array,
        color=dot_color,
        size=dot_size*1.5,
        jitter=True,
        ax=ax
    )
    
    # 设置刻度
    # ax.set_xticks([])
    ax.set_yticks(ytick_cluster_density)
    ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize*1.5, fontname='Arial')
    ax.tick_params(axis='both', width=frame_linewidth*1.5)
    
    # 设置边框
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(frame_linewidth*1.5)
    
    # 去除坐标轴标题
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    
    plt.tight_layout(pad=0)
    plt.show()
    
    mean_density = np.mean(Cluster_density_array)
    std_density = np.std(Cluster_density_array)
    
    print(f"Mean: {mean_density:.2f}")
    print(f"Standard Deviation: {std_density:.2f}")
    
    # --------横坐标直径，纵坐标Dslow散点分布图 --------

    # # 参数设置
    # dot_color = 'black'
    # dot_size = 80
    # tick_fontsize = 30
    # frame_linewidth = 5
    
    # # 创建散点图
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.scatter(data_5, data_4, color=dot_color, s=dot_size)
    
    # # 设置坐标轴刻度字体和大小
    # ax.set_xticklabels(ax.get_xticks(), fontsize=tick_fontsize, fontname='Arial')
    # ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize, fontname='Arial')
    
    # # 去除标题与标签
    # ax.set_xlabel("")
    # ax.set_ylabel("")
    # ax.set_title("")
    
    # # 设置边框样式
    # for spine in ['top', 'right']:
    #     ax.spines[spine].set_visible(False)
    # for spine in ['left', 'bottom']:
    #     ax.spines[spine].set_linewidth(frame_linewidth)
    
    # # 紧凑布局
    # plt.tight_layout(pad=0)
    # plt.show()

    
    
