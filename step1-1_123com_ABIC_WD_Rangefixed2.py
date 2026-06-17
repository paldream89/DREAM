# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:02:52 2024

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 21:57:23 2022

@author: Admin
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
import numpy as np
from math import floor,sqrt,e,pi
from scipy import optimize
import matplotlib as mpl
import matplotlib.path as mpltPath
import wx
from ReadSTORMBin_XcYcZZcFrame import read_storm_bin_XcYcZZcFrame
from ReadSTORMBin import read_storm_bin

def twoD_diff_dis_func(x,a,b,c):
    
    return 2*c*x/a*np.exp(-x**2/a)+b*x

def twoD_diff_dis_func_double(x,a,b,c,a2,c2):
    
    return 2*c*x/a*np.exp(-x**2/a)+2*c2*x/a2*np.exp(-x**2/a2)+b*x

def twoD_diff_dis_func_triple(x,a,b,c,a2,c2,a3,c3):
    
    return 2*c*x/a*np.exp(-x**2/a)+2*c2*x/a2*np.exp(-x**2/a2)+2*c3*x/a3*np.exp(-x**2/a3)+b*x


def calculate_aic_bic(y_obs, y_fit, k):
    """
    Calculate AIC and BIC based on residual sum of squares.
    Here n is the number of histogram bins and k is the number of fitted parameters.
    Assumption: least-squares fitting with Gaussian residuals.
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    n = y_obs.size
    rss = np.sum((y_obs - y_fit)**2)

    # Avoid log(0) when the fit is nearly perfect.
    rss_per_point = max(rss/n, 1e-12)

    aic = n*np.log(rss_per_point) + 2*k
    bic = n*np.log(rss_per_point) + k*np.log(n)
    return aic, bic, rss


def D_to_a(D):
    """
    Convert diffusion coefficient D (um^2/s) to fitted displacement parameter a.
    D = a * (pixel_size/1000)^2 / (4*time_interval)
    Therefore, a = D * 4*time_interval / (pixel_size/1000)^2
    """
    return D * 4 * time_interval / (pixel_size/1000)**2


def a_to_D(a):
    """
    Convert fitted displacement parameter a to diffusion coefficient D (um^2/s).
    """
    return a * (pixel_size/1000)**2 / 4 / time_interval


def get_triple_a_bounds():
    """
    Build lower and upper bounds for a, a2, a3 according to the user-defined
    D center values and tolerances.
    """
    D1_low = max(D1_R - Tor_1, 1e-12)
    D1_high = max(D1_R + Tor_1, D1_low + 1e-12)

    D2_low = max(D2_R - Tor_2, 1e-12)
    D2_high = max(D2_R + Tor_2, D2_low + 1e-12)

    D3_low = max(D3_R - Tor_3, 1e-12)
    D3_high = max(D3_R + Tor_3, D3_low + 1e-12)

    a1_low = max(D_to_a(D1_low), 1e-12)
    a1_high = min(D_to_a(D1_high), disp_thre**2)

    a2_low = max(D_to_a(D2_low), 1e-12)
    a2_high = min(D_to_a(D2_high), disp_thre**2)

    a3_low = max(D_to_a(D3_low), 1e-12)
    a3_high = min(D_to_a(D3_high), disp_thre**2)

    if not (a1_low < a1_high and a2_low < a2_high and a3_low < a3_high):
        raise ValueError('Invalid D range. Please check D1_R/Tor_1, D2_R/Tor_2, and D3_R/Tor_3.')

    return (a1_low, a1_high, a2_low, a2_high, a3_low, a3_high)



def get_double_a_bounds():
    """
    Build lower and upper bounds for a and a2 according to the user-defined
    D center values and tolerances for 2-component fitting.
    """
    Dd1_low = max(Dd_1 - Tord_1, 1e-12)
    Dd1_high = max(Dd_1 + Tord_1, Dd1_low + 1e-12)

    Dd2_low = max(Dd_2 - Tord_2, 1e-12)
    Dd2_high = max(Dd_2 + Tord_2, Dd2_low + 1e-12)

    ad1_low = max(D_to_a(Dd1_low), 1e-12)
    ad1_high = min(D_to_a(Dd1_high), disp_thre**2)

    ad2_low = max(D_to_a(Dd2_low), 1e-12)
    ad2_high = min(D_to_a(Dd2_high), disp_thre**2)

    if not (ad1_low < ad1_high and ad2_low < ad2_high):
        raise ValueError('Invalid double-component D range. Please check Dd_1/Tord_1 and Dd_2/Tord_2.')

    return (ad1_low, ad1_high, ad2_low, ad2_high)

def clip_value(value, low, high):
    """
    Clip initial guessed value into the allowed bound range.
    This is necessary because scipy.optimize.curve_fit requires p0 to be inside bounds.
    """
    return min(max(value, low), high)


def make_histogram_plot(hist):
    """
    Draw the selected-area histogram with exactly the same visual settings used below.
    This helper is used when component == 23, because 2-component and 3-component
    fits need to be shown in two separate figures.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xticks([])
    ax.set_yticks([])

    x_min = 0
    x_max = disp_thre
    x_ticks = np.arange(x_min, x_max+0.1, 1)
    ax.set_xlim([x_min, x_max])
    ax.set_xticks(x_ticks)
    ax.yaxis.set_tick_params(labelsize=0, size=4, width=4, color='black')
    ax.xaxis.set_tick_params(labelsize=0, size=4, width=4, color='black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)
    plt.bar(bins_x,hist,width=bin_size*0.8, color=color_green,alpha=0.5)
    return fig, ax


# displacement threshold
disp_thre = 5

# diffusion rate map upper lower limit
d_high = 4
d_low = 0
pixel_size = 114 # unit is nm
time_interval = 0.00515# unit is s, so 1 ms
component = 1 # 1 or 2 or 3 component? Use 23 to fit 2-component and 3-component models together.

# ===== D range constraints for 2-component fitting =====
# Unit of D is um^2/s.
# During 2-component fitting, the corresponding parameters a and a2 are constrained
# so that:
# D1 is within Dd_1 +/- Tord_1
# D2 is within Dd_2 +/- Tord_2
Dd_1 = 2.6
Tord_1 = 2.4
Dd_2 = 0.1
Tord_2 = 0.001

# ===== D range constraints for 3-component fitting =====
# Unit of D is um^2/s.
# D = a * (pixel_size/1000)^2 / (4*time_interval)
# During 3-component fitting, the corresponding parameters a, a2, a3 are constrained
# so that:
# D1 is within D1_R +/- Tor_1
# D2 is within D2_R +/- Tor_2
# D3 is within D3_R +/- Tor_3
D1_R = 2.6
Tor_1 = 2.4
D2_R = 1.6
Tor_2 = 1.4
D3_R = 0.1
Tor_3 = 0.001

# point cloud point size
point_size = 0.2

# accelerate figure speed
num_points_showedRatio = 50

# histogram binning setting
bin_num = disp_thre*4 # must be an interger
bin_size = disp_thre/bin_num 
bins_x = np.arange(0.5*bin_size, disp_thre, bin_size, dtype=np.float32)
bin_thre = int(floor((disp_thre*0.6)/bin_size))

# angle binning setting
angle_bin_num = 1  # from -pi to pi
angle_bin_size = 2*pi/angle_bin_num
angle_bins_x = np.arange(-pi+0.5*angle_bin_size, pi, angle_bin_size, dtype=np.float32)

angle_bin_size_plot = 360/angle_bin_num
angle_bins_x_plot = np.arange(-180+0.5*angle_bin_size_plot, 
                              180, angle_bin_size_plot, dtype=np.float32)

color_blue = (0/255, 114/255, 178/255)
color_red = (213/255, 94/255, 0/255)
color_blue2 = (51/255, 51/255, 255/255)
color_red2 = (255/255, 51/255, 51/255)
color_purple = (204/255, 121/255, 167/255)
color_orange = (230/255, 159/255, 0/255)
color_green = (0/255, 158/255, 115/255)
color_skyblue = (86/255, 180/255, 233/255)
color_grey = (160/255, 160/255, 160/255)

# initialize saved data
drate_hist_array = np.zeros((angle_bin_num,bin_num),dtype=np.uint32) 
drate_array = np.zeros(angle_bin_num,dtype=np.float32)
drate_array_2 = np.zeros(angle_bin_num,dtype=np.float32)

app = wx.App()
frame = wx.Frame(None, -1, 'win.py')
frame.SetSize(0,0,200,50)

# Create open file dialog
openFileDialog = wx.FileDialog(frame, "Select -dis bin", "", "", 
      "bin files (*.bin)|*.bin", 
       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

openFileDialog.ShowModal()
file_path_STORMBin = openFileDialog.GetPath()
openFileDialog.Destroy()

frame_num, number_of_points, STORM_npdata = read_storm_bin_XcYcZZcFrame(file_path_STORMBin)
        # data_type = np.dtype([('x_start', np.float32), ('y_start', np.float32),
        #                   ('x_end', np.float32), ('y_end', np.float32),
        #                   ('frame', np.int32),
        #                   ('disp', np.float32), ('angle', np.float32)])   
        
app2 = wx.App()
frame2 = wx.Frame(None, -1, 'win.py')
frame2.SetSize(0,0,200,50)

# Create open file dialog
openFileDialog2 = wx.FileDialog(frame2, "Select Diffusion bin", "", "", 
      "bin files (*.bin)|*.bin", 
       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

openFileDialog2.ShowModal()
file_path_STORMBin2 = openFileDialog2.GetPath()
openFileDialog2.Destroy()

frame_num_D, number_of_points_D, STORM_npdata_D = read_storm_bin(file_path_STORMBin2)
        # data_type = np.dtype([('x_start', np.float32), ('y_start', np.float32),
        #                   ('x_end', np.float32), ('y_end', np.float32),
        #                   ('frame', np.int32),
        #                   ('disp', np.float32), ('angle', np.float32)])   
        
num_showed = int(number_of_points_D/num_points_showedRatio)
        
cmap = mpl.colormaps["jet"]
cmap.set_under(color='black')
cmap.set_bad(color='black')

y_max = np.amax(STORM_npdata_D['Yc'])
STORM_npdata_D['Yc'] = y_max-STORM_npdata_D['Yc']
x_max = np.amax(STORM_npdata_D['Xc'])
plt.figure(figsize=(15/250*x_max,15/250*y_max))
plt.scatter(STORM_npdata_D['Xc'][0:num_showed],STORM_npdata_D['Yc'][0:num_showed],cmap=cmap,s=point_size,
            c=STORM_npdata_D['Zc'][0:num_showed],vmin = d_low,vmax = d_high)

plt.show()

fig = plt.gcf()
ax = plt.gca()
x_main_min,x_main_max = ax.get_xlim()
y_main_min,y_main_max = ax.get_ylim()

# IMPORTANT:
# The lasso is drawn on the displayed diffusion map generated from STORM_npdata_D.
# In that displayed map, Yc has been transformed as y_plot = y_max - Yc.
# Therefore, the points from STORM_npdata must be transformed into the same
# display coordinate system before calling contains_points().
# STORM_npdata_D and STORM_npdata have no index correspondence; only their
# physical/display coordinates should be used for region selection.
STORM_x_start_plot = STORM_npdata['x_start'].copy()
STORM_y_start_plot = y_max - STORM_npdata['y_start']

pix = np.transpose(np.vstack((STORM_x_start_plot, STORM_y_start_plot)))

def onSelect(x):
    
    p = mpltPath.Path(x)
    ind = p.contains_points(pix, radius=0)
    
    selected = np.zeros_like(STORM_npdata['disp'])-0.001
    selected = STORM_npdata['disp'][ind]
    selected_angle = STORM_npdata['angle'][ind]
    plt.figure()
    cmap = mpl.colormaps["jet"]
    cmap.set_under(color='black')
    cmap.set_bad(color='black')
    
    plt.scatter(STORM_x_start_plot[ind],STORM_y_start_plot[ind],cmap=cmap,s=point_size,
                c=selected,vmin = d_low,vmax = d_high)
    ax_select = plt.gca()
    ax_select.set_xlim(xmin=x_main_min,xmax=x_main_max)
    ax_select.set_ylim(ymin=y_main_min,ymax=y_main_max)
    
    order_indice = np.argsort(selected_angle).astype(np.uint32)
    selected = selected[order_indice]
    selected_angle = selected_angle[order_indice]
    size_selected = np.size(selected)
    
    cursor_A = 0
    cursor_B = 0
    cursor_angle = 1
    
    while cursor_B < size_selected:
        
        cursor_A = cursor_B
        
        while selected_angle[cursor_B]<-pi+cursor_angle*angle_bin_size:
            
            cursor_B += 1
            
            if cursor_B >= size_selected:
                break
        
        hist,_ = np.histogram(selected[cursor_A:cursor_B], 
                                       bins = bin_num, range = (0,disp_thre))
        
        total_mol_select_1 = np.sum(hist)
        total_mol_select_2 = np.size(STORM_npdata['disp'][ind])
        
        # drate_hist_array[cursor_angle-1,:] = np.uint32(hist)
        
        fig, ax = plt.subplots(figsize=(9, 4))
        # fig.patch.set_alpha(0.0)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 显示刻度线，但不显示数值
        
        x_min = 0
        x_max = disp_thre # 确保包含数据的最大范围
        x_ticks = np.arange(x_min, x_max+0.1, 1)  # 生成间隔为 0.5 的刻度
        # # 设置 x 轴刻度
        ax.set_xlim([x_min, x_max])
        ax.set_xticks(x_ticks)# 显示刻度线，但不显示数值
        ax.yaxis.set_tick_params(labelsize=0, size=4, width=4, color='black')  # 仅显示刻度线

        # 显示刻度线，但不显示数值
        ax.xaxis.set_tick_params(labelsize=0, size=4, width=4, color='black')  # 仅显示刻度线        ax.xaxis.set_tick_params(labelsize=0, size=5, width=1, color='black')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(4)
        ax.spines['bottom'].set_linewidth(4)
        plt.bar(bins_x,hist,width=bin_size*0.8, color=color_green,alpha=0.5)
        plt.show()
        
        if component == 1:
            
            guessed_a = (0.3*disp_thre)**2
            guessed_b = (hist[-1]/bins_x[-1] + hist[-2]/bins_x[-2])/2
            guessed_c = np.amax(hist[0:bin_thre])*sqrt(guessed_a)*e/2
                    
            p0 = [guessed_a,guessed_b,guessed_c]
                    
            try: 
                        
                popt,_ = optimize.curve_fit(twoD_diff_dis_func,bins_x,hist,p0=p0)
                fit_hist = twoD_diff_dis_func(bins_x,*popt)
                aic, bic, rss = calculate_aic_bic(hist,fit_hist,k=3)
                selected_drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
                selected_slope = popt[1]/np.sum(hist)
                print('Selected Area Diffusion Rate-2D: '+'%.2f' % selected_drate)
                print('Selected Area background-2D: '+'%.6f' % selected_slope)
                print('AIC-1component: '+'%.3f' % aic)
                print('BIC-1component: '+'%.3f' % bic)
                print('RSS-1component: '+'%.3f' % rss)
                print('see bin_x for x axis; Hist(y axis) is saved as txt')
                save_hist = np.concatenate((bins_x,hist),axis = 0)
                fname = file_path_STORMBin.replace('.bin','_Sel'+str(total_mol_select_2)+'.txt')
                np.savetxt(fname,save_hist,fmt='%.5f',delimiter=' ',newline='\n')
                print('Fitted Curve:')
                print('2*%.0f*x/%.5f*exp(-x^2/%.5f)+%.0f*x' %(popt[2],popt[0],popt[0],popt[1]))
                print('Totol Mol Selected:'+str(total_mol_select_2))
                
                smooth_x = np.linspace(0,disp_thre,num=int(disp_thre*100))
                fitted_y = 2*popt[2]*smooth_x/popt[0]*np.exp(-smooth_x**2/popt[0])+popt[1]*smooth_x
                distribution_y = 2*popt[2]*smooth_x/popt[0]*np.exp(-smooth_x**2/popt[0])
                background_y = popt[1]*smooth_x
                plt.plot(smooth_x, fitted_y,'k')
                plt.plot(smooth_x, distribution_y,'r--')
                plt.plot(smooth_x, background_y,'b--')
                plt.show()
                        
            except RuntimeError:
                        
                print ("RuntimeError")
                
        elif component == 2:
        
            try: 
                
                guessed_a = (0.3*disp_thre)**2
                guessed_b = (hist[-1]/bins_x[-1] + hist[-2]/bins_x[-2])/2
                guessed_c = np.amax(hist[0:bin_thre])*sqrt(guessed_a)*e/2
                guessed_a2 = (0.3*disp_thre)**2/6
                guessed_c2 = np.amax(hist[0:bin_thre])*sqrt(guessed_a)*e/2
                        
                ad1_low, ad1_high, ad2_low, ad2_high = get_double_a_bounds()

                guessed_a = clip_value(guessed_a, ad1_low, ad1_high)
                guessed_a2 = clip_value(guessed_a2, ad2_low, ad2_high)

                p0 = [guessed_a,guessed_b,guessed_c,guessed_a2,guessed_c2]
                lower_bounds = [ad1_low, 0, 0, ad2_low, 0]
                upper_bounds = [ad1_high, np.inf, np.inf, ad2_high, np.inf]
                bounds = (lower_bounds, upper_bounds)

                print('2-component D bounds:')
                print('D1 range: %.3f to %.3f um^2/s' % (a_to_D(ad1_low), a_to_D(ad1_high)))
                print('D2 range: %.3f to %.3f um^2/s' % (a_to_D(ad2_low), a_to_D(ad2_high)))

                popt,_ = optimize.curve_fit(twoD_diff_dis_func_double,bins_x,hist,p0=p0,bounds=bounds)
                fit_hist = twoD_diff_dis_func_double(bins_x,*popt)
                aic, bic, rss = calculate_aic_bic(hist,fit_hist,k=5)
                drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
                drate_2 = popt[3] * (pixel_size/1000)**2 /4/time_interval
                
                
                # popt,_ = optimize.curve_fit(twoD_diff_dis_func,bins_x,selected_hist_sum,p0=p0)
                # selected_drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
                slope = popt[1]/np.sum(hist)
                print('Selected Area Diff Rate-2D-1 first D:'+'%.2f' % drate)
                print('Selected Area Diff Rate-2D-2 second D:'+'%.2f' % drate_2)
                print('Selected Area background-2D background: '+'%.6f' % slope)
                print('AIC-2component: '+'%.3f' % aic)
                print('BIC-2component: '+'%.3f' % bic)
                print('RSS-2component: '+'%.3f' % rss)
                # print('Selected Area b-2D: '+str(bins_x))
                # print('Selected Area b-2D: '+str(hist))
                # print('Selected Area b-2D: '+str(popt))
                smooth_x = np.linspace(0,disp_thre,num=int(disp_thre*100))
                fitted_y = 2*popt[2]*smooth_x/popt[0]*np.exp(-smooth_x**2/popt[0])+popt[1]*smooth_x+2*popt[4]*smooth_x/popt[3]*np.exp(-smooth_x**2/popt[3])
                distribution_y = 2*popt[2]*smooth_x/popt[0]*np.exp(-smooth_x**2/popt[0])
                distribution_y2 = 2*popt[4]*smooth_x/popt[3]*np.exp(-smooth_x**2/popt[3])
                background_y = popt[1]*smooth_x
                D1_percentage = (np.sum(distribution_y)/(np.sum(distribution_y2)+np.sum(distribution_y)))*100
                D2_percentage = (np.sum(distribution_y2)/(np.sum(distribution_y2)+np.sum(distribution_y)))*100
                print('Diff Rate first D ratio: '+'%.2f' % D1_percentage)
                print('Diff Rate second D ratio: '+'%.2f' % D2_percentage)
                print('see bin_x for x axis; Hist(y axis) is saved as txt')
                save_hist = np.concatenate((bins_x,hist),axis = 0)
                fname = file_path_STORMBin.replace('.bin','_Sel'+str(total_mol_select_2)+'.txt')
                np.savetxt(fname,save_hist,fmt='%.5f',delimiter=' ',newline='\n')
                print('Fitted Curve:')
                print('2*%.0f*x/%.5f*exp(-x^2/%.5f)+%.0f*x+2*%.0f*x/%.5f*exp(-x^2/%.5f)' %(popt[2],popt[0],popt[0],popt[1],popt[4],popt[3],popt[3]))
                
                plt.plot(smooth_x, fitted_y,color=color_grey,
                         linestyle='-', linewidth=5)
                plt.plot(smooth_x, distribution_y,color=color_red2,
                         linestyle='--', linewidth=5)
                plt.plot(smooth_x, distribution_y2,color=color_blue2,
                         linestyle='--', linewidth=5)
                plt.plot(smooth_x, background_y,color=color_purple,
                         linestyle='--', linewidth=5)
                
                
                ##===========绘图设置================
                
                
                plt.show()
                print('Totol Mol Num='+str(total_mol_select_1)+'='+str(total_mol_select_2))
                
                
                
            except RuntimeError:
                
                print('RuntimeError')
                drate = 0

        elif component == 3:

            try:

                guessed_a = (0.3*disp_thre)**2
                guessed_b = (hist[-1]/bins_x[-1] + hist[-2]/bins_x[-2])/2
                guessed_c = np.amax(hist[0:bin_thre])*sqrt(guessed_a)*e/2
                guessed_a2 = (0.3*disp_thre)**2/6
                guessed_c2 = np.amax(hist[0:bin_thre])*sqrt(guessed_a)*e/2
                guessed_a3 = (0.3*disp_thre)**2/20
                guessed_c3 = np.amax(hist[0:bin_thre])*sqrt(guessed_a)*e/2

                a1_low, a1_high, a2_low, a2_high, a3_low, a3_high = get_triple_a_bounds()

                guessed_a = clip_value(guessed_a, a1_low, a1_high)
                guessed_a2 = clip_value(guessed_a2, a2_low, a2_high)
                guessed_a3 = clip_value(guessed_a3, a3_low, a3_high)

                p0 = [guessed_a,guessed_b,guessed_c,
                      guessed_a2,guessed_c2,
                      guessed_a3,guessed_c3]
                lower_bounds = [a1_low, 0, 0,
                                a2_low, 0,
                                a3_low, 0]
                upper_bounds = [a1_high, np.inf, np.inf,
                                a2_high, np.inf,
                                a3_high, np.inf]
                bounds = (lower_bounds, upper_bounds)

                print('3-component D bounds:')
                print('D1 range: %.3f to %.3f um^2/s' % (a_to_D(a1_low), a_to_D(a1_high)))
                print('D2 range: %.3f to %.3f um^2/s' % (a_to_D(a2_low), a_to_D(a2_high)))
                print('D3 range: %.3f to %.3f um^2/s' % (a_to_D(a3_low), a_to_D(a3_high)))

                popt,_ = optimize.curve_fit(twoD_diff_dis_func_triple,bins_x,hist,
                                            p0=p0,bounds=bounds,maxfev=20000)
                fit_hist = twoD_diff_dis_func_triple(bins_x,*popt)
                aic, bic, rss = calculate_aic_bic(hist,fit_hist,k=7)

                drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
                drate_2 = popt[3] * (pixel_size/1000)**2 /4/time_interval
                drate_3 = popt[5] * (pixel_size/1000)**2 /4/time_interval
                slope = popt[1]/np.sum(hist)

                print('Selected Area Diff Rate-2D-1 first D:'+'%.2f' % drate)
                print('Selected Area Diff Rate-2D-2 second D:'+'%.2f' % drate_2)
                print('Selected Area Diff Rate-2D-3 third D:'+'%.2f' % drate_3)
                print('Selected Area background-2D background: '+'%.6f' % slope)
                print('AIC-3component: '+'%.3f' % aic)
                print('BIC-3component: '+'%.3f' % bic)
                print('RSS-3component: '+'%.3f' % rss)

                smooth_x = np.linspace(0,disp_thre,num=int(disp_thre*100))
                distribution_y = 2*popt[2]*smooth_x/popt[0]*np.exp(-smooth_x**2/popt[0])
                distribution_y2 = 2*popt[4]*smooth_x/popt[3]*np.exp(-smooth_x**2/popt[3])
                distribution_y3 = 2*popt[6]*smooth_x/popt[5]*np.exp(-smooth_x**2/popt[5])
                background_y = popt[1]*smooth_x
                fitted_y = distribution_y+distribution_y2+distribution_y3+background_y

                distribution_sum = np.sum(distribution_y)+np.sum(distribution_y2)+np.sum(distribution_y3)
                D1_percentage = (np.sum(distribution_y)/distribution_sum)*100
                D2_percentage = (np.sum(distribution_y2)/distribution_sum)*100
                D3_percentage = (np.sum(distribution_y3)/distribution_sum)*100

                print('Diff Rate first D ratio: '+'%.2f' % D1_percentage)
                print('Diff Rate second D ratio: '+'%.2f' % D2_percentage)
                print('Diff Rate third D ratio: '+'%.2f' % D3_percentage)
                print('see bin_x for x axis; Hist(y axis) is saved as txt')
                save_hist = np.concatenate((bins_x,hist),axis = 0)
                fname = file_path_STORMBin.replace('.bin','_Sel'+str(total_mol_select_2)+'.txt')
                np.savetxt(fname,save_hist,fmt='%.5f',delimiter=' ',newline='\n')
                print('Fitted Curve:')
                print('2*%.0f*x/%.5f*exp(-x^2/%.5f)+%.0f*x+2*%.0f*x/%.5f*exp(-x^2/%.5f)+2*%.0f*x/%.5f*exp(-x^2/%.5f)' %
                      (popt[2],popt[0],popt[0],popt[1],
                       popt[4],popt[3],popt[3],
                       popt[6],popt[5],popt[5]))

                plt.plot(smooth_x, fitted_y,color=color_grey,
                         linestyle='-', linewidth=5)
                plt.plot(smooth_x, distribution_y,color=color_red2,
                         linestyle='--', linewidth=5)
                plt.plot(smooth_x, distribution_y2,color=color_blue2,
                         linestyle='--', linewidth=5)
                plt.plot(smooth_x, distribution_y3,color=color_orange,
                         linestyle='--', linewidth=5)
                plt.plot(smooth_x, background_y,color=color_purple,
                         linestyle='--', linewidth=5)


                ##===========绘图设置================


                plt.show()
                print('Totol Mol Num='+str(total_mol_select_1)+'='+str(total_mol_select_2))



            except RuntimeError:

                print('RuntimeError')
                drate = 0

        elif component == 23:

            print('================ 2-component fitting ================')
            try:
                guessed_a = (0.3*disp_thre)**2
                guessed_b = (hist[-1]/bins_x[-1] + hist[-2]/bins_x[-2])/2
                guessed_c = np.amax(hist[0:bin_thre])*sqrt(guessed_a)*e/2
                guessed_a2 = (0.3*disp_thre)**2/6
                guessed_c2 = np.amax(hist[0:bin_thre])*sqrt(guessed_a)*e/2

                ad1_low, ad1_high, ad2_low, ad2_high = get_double_a_bounds()

                guessed_a = clip_value(guessed_a, ad1_low, ad1_high)
                guessed_a2 = clip_value(guessed_a2, ad2_low, ad2_high)

                p0_2 = [guessed_a,guessed_b,guessed_c,guessed_a2,guessed_c2]
                lower_bounds_2 = [ad1_low, 0, 0, ad2_low, 0]
                upper_bounds_2 = [ad1_high, np.inf, np.inf, ad2_high, np.inf]
                bounds_2 = (lower_bounds_2, upper_bounds_2)

                print('2-component D bounds:')
                print('D1 range: %.3f to %.3f um^2/s' % (a_to_D(ad1_low), a_to_D(ad1_high)))
                print('D2 range: %.3f to %.3f um^2/s' % (a_to_D(ad2_low), a_to_D(ad2_high)))

                popt_2,_ = optimize.curve_fit(twoD_diff_dis_func_double,bins_x,hist,
                                               p0=p0_2,bounds=bounds_2,maxfev=20000)
                fit_hist_2 = twoD_diff_dis_func_double(bins_x,*popt_2)
                aic_2, bic_2, rss_2 = calculate_aic_bic(hist,fit_hist_2,k=5)

                drate_1_2c = popt_2[0] * (pixel_size/1000)**2 /4/time_interval
                drate_2_2c = popt_2[3] * (pixel_size/1000)**2 /4/time_interval
                slope_2 = popt_2[1]/np.sum(hist)

                print('Selected Area Diff Rate-2D-1 first D:'+'%.2f' % drate_1_2c)
                print('Selected Area Diff Rate-2D-2 second D:'+'%.2f' % drate_2_2c)
                print('Selected Area background-2D background: '+'%.6f' % slope_2)
                print('AIC-2component: '+'%.3f' % aic_2)
                print('BIC-2component: '+'%.3f' % bic_2)
                print('RSS-2component: '+'%.3f' % rss_2)

                smooth_x = np.linspace(0,disp_thre,num=int(disp_thre*100))
                distribution_y_2c_1 = 2*popt_2[2]*smooth_x/popt_2[0]*np.exp(-smooth_x**2/popt_2[0])
                distribution_y_2c_2 = 2*popt_2[4]*smooth_x/popt_2[3]*np.exp(-smooth_x**2/popt_2[3])
                background_y_2c = popt_2[1]*smooth_x
                fitted_y_2c = distribution_y_2c_1+distribution_y_2c_2+background_y_2c

                distribution_sum_2c = np.sum(distribution_y_2c_1)+np.sum(distribution_y_2c_2)
                D1_percentage_2c = (np.sum(distribution_y_2c_1)/distribution_sum_2c)*100
                D2_percentage_2c = (np.sum(distribution_y_2c_2)/distribution_sum_2c)*100
                print('Diff Rate first D ratio: '+'%.2f' % D1_percentage_2c)
                print('Diff Rate second D ratio: '+'%.2f' % D2_percentage_2c)
                print('Fitted Curve-2component:')
                print('2*%.0f*x/%.5f*exp(-x^2/%.5f)+%.0f*x+2*%.0f*x/%.5f*exp(-x^2/%.5f)' %
                      (popt_2[2],popt_2[0],popt_2[0],popt_2[1],popt_2[4],popt_2[3],popt_2[3]))

                make_histogram_plot(hist)
                plt.plot(smooth_x, fitted_y_2c,color=color_grey,
                         linestyle='-', linewidth=5)
                plt.plot(smooth_x, distribution_y_2c_1,color=color_red2,
                         linestyle='--', linewidth=5)
                plt.plot(smooth_x, distribution_y_2c_2,color=color_blue2,
                         linestyle='--', linewidth=5)
                plt.plot(smooth_x, background_y_2c,color=color_purple,
                         linestyle='--', linewidth=5)
                plt.show()

            except RuntimeError:
                print('RuntimeError in 2-component fitting')
            except ValueError as err:
                print('ValueError in 2-component fitting: '+str(err))

            print('================ 3-component fitting ================')
            try:
                guessed_a = (0.3*disp_thre)**2
                guessed_b = (hist[-1]/bins_x[-1] + hist[-2]/bins_x[-2])/2
                guessed_c = np.amax(hist[0:bin_thre])*sqrt(guessed_a)*e/2
                guessed_a2 = (0.3*disp_thre)**2/6
                guessed_c2 = np.amax(hist[0:bin_thre])*sqrt(guessed_a)*e/2
                guessed_a3 = (0.3*disp_thre)**2/20
                guessed_c3 = np.amax(hist[0:bin_thre])*sqrt(guessed_a)*e/2

                a1_low, a1_high, a2_low, a2_high, a3_low, a3_high = get_triple_a_bounds()

                guessed_a = clip_value(guessed_a, a1_low, a1_high)
                guessed_a2 = clip_value(guessed_a2, a2_low, a2_high)
                guessed_a3 = clip_value(guessed_a3, a3_low, a3_high)

                p0_3 = [guessed_a,guessed_b,guessed_c,
                        guessed_a2,guessed_c2,
                        guessed_a3,guessed_c3]
                lower_bounds_3 = [a1_low, 0, 0,
                                  a2_low, 0,
                                  a3_low, 0]
                upper_bounds_3 = [a1_high, np.inf, np.inf,
                                  a2_high, np.inf,
                                  a3_high, np.inf]
                bounds_3 = (lower_bounds_3, upper_bounds_3)

                print('3-component D bounds:')
                print('D1 range: %.3f to %.3f um^2/s' % (a_to_D(a1_low), a_to_D(a1_high)))
                print('D2 range: %.3f to %.3f um^2/s' % (a_to_D(a2_low), a_to_D(a2_high)))
                print('D3 range: %.3f to %.3f um^2/s' % (a_to_D(a3_low), a_to_D(a3_high)))

                popt_3,_ = optimize.curve_fit(twoD_diff_dis_func_triple,bins_x,hist,
                                               p0=p0_3,bounds=bounds_3,maxfev=20000)
                fit_hist_3 = twoD_diff_dis_func_triple(bins_x,*popt_3)
                aic_3, bic_3, rss_3 = calculate_aic_bic(hist,fit_hist_3,k=7)

                drate_1_3c = popt_3[0] * (pixel_size/1000)**2 /4/time_interval
                drate_2_3c = popt_3[3] * (pixel_size/1000)**2 /4/time_interval
                drate_3_3c = popt_3[5] * (pixel_size/1000)**2 /4/time_interval
                slope_3 = popt_3[1]/np.sum(hist)

                print('Selected Area Diff Rate-2D-1 first D:'+'%.2f' % drate_1_3c)
                print('Selected Area Diff Rate-2D-2 second D:'+'%.2f' % drate_2_3c)
                print('Selected Area Diff Rate-2D-3 third D:'+'%.2f' % drate_3_3c)
                print('Selected Area background-2D background: '+'%.6f' % slope_3)
                print('AIC-3component: '+'%.3f' % aic_3)
                print('BIC-3component: '+'%.3f' % bic_3)
                print('RSS-3component: '+'%.3f' % rss_3)

                smooth_x = np.linspace(0,disp_thre,num=int(disp_thre*100))
                distribution_y_3c_1 = 2*popt_3[2]*smooth_x/popt_3[0]*np.exp(-smooth_x**2/popt_3[0])
                distribution_y_3c_2 = 2*popt_3[4]*smooth_x/popt_3[3]*np.exp(-smooth_x**2/popt_3[3])
                distribution_y_3c_3 = 2*popt_3[6]*smooth_x/popt_3[5]*np.exp(-smooth_x**2/popt_3[5])
                background_y_3c = popt_3[1]*smooth_x
                fitted_y_3c = distribution_y_3c_1+distribution_y_3c_2+distribution_y_3c_3+background_y_3c

                distribution_sum_3c = np.sum(distribution_y_3c_1)+np.sum(distribution_y_3c_2)+np.sum(distribution_y_3c_3)
                D1_percentage_3c = (np.sum(distribution_y_3c_1)/distribution_sum_3c)*100
                D2_percentage_3c = (np.sum(distribution_y_3c_2)/distribution_sum_3c)*100
                D3_percentage_3c = (np.sum(distribution_y_3c_3)/distribution_sum_3c)*100
                print('Diff Rate first D ratio: '+'%.2f' % D1_percentage_3c)
                print('Diff Rate second D ratio: '+'%.2f' % D2_percentage_3c)
                print('Diff Rate third D ratio: '+'%.2f' % D3_percentage_3c)
                print('Fitted Curve-3component:')
                print('2*%.0f*x/%.5f*exp(-x^2/%.5f)+%.0f*x+2*%.0f*x/%.5f*exp(-x^2/%.5f)+2*%.0f*x/%.5f*exp(-x^2/%.5f)' %
                      (popt_3[2],popt_3[0],popt_3[0],popt_3[1],
                       popt_3[4],popt_3[3],popt_3[3],
                       popt_3[6],popt_3[5],popt_3[5]))

                make_histogram_plot(hist)
                plt.plot(smooth_x, fitted_y_3c,color=color_grey,
                         linestyle='-', linewidth=5)
                plt.plot(smooth_x, distribution_y_3c_1,color=color_red2,
                         linestyle='--', linewidth=5)
                plt.plot(smooth_x, distribution_y_3c_2,color=color_blue2,
                         linestyle='--', linewidth=5)
                plt.plot(smooth_x, distribution_y_3c_3,color=color_orange,
                         linestyle='--', linewidth=5)
                plt.plot(smooth_x, background_y_3c,color=color_purple,
                         linestyle='--', linewidth=5)
                plt.show()

                print('================ Model comparison ================')
                if 'aic_2' in locals() and 'bic_2' in locals():
                    print('Delta AIC = AIC-3component - AIC-2component: '+'%.3f' % (aic_3-aic_2))
                    print('Delta BIC = BIC-3component - BIC-2component: '+'%.3f' % (bic_3-bic_2))
                    print('Delta RSS = RSS-3component - RSS-2component: '+'%.3f' % (rss_3-rss_2))
                print('Totol Mol Num='+str(total_mol_select_1)+'='+str(total_mol_select_2))

            except RuntimeError:
                print('RuntimeError in 3-component fitting')
            except ValueError as err:
                print('ValueError in 3-component fitting: '+str(err))

            print('see bin_x for x axis; Hist(y axis) is saved as txt')
            save_hist = np.concatenate((bins_x,hist),axis = 0)
            fname = file_path_STORMBin.replace('.bin','_Sel'+str(total_mol_select_2)+'.txt')
            np.savetxt(fname,save_hist,fmt='%.5f',delimiter=' ',newline='\n')

        else:

            print('component should be 1, 2, 3, or 23')
        
        # drate_array[cursor_angle-1] = drate
        # drate_array_2[cursor_angle-1] = drate_2
        cursor_angle += 1
        
    # plt.figure()
    # plt.plot(angle_bins_x_plot,drate_array,'b')
    # plt.show()
        
        # drate_hist_array = np.zeros((angle_bin_num,bin_num),dtype=np.uint32) 
        # drate_array = np.zeros(angle_bin_num,dtype=np.float32)
    
    # selected_hist = np.zeros_like(hist_bin,dtype=np.uint16)
    # selected_hist[ind] = hist_bin[ind]
    # selected_hist_sum = np.sum(selected_hist, axis=0)
    # plt.figure()
    # plt.bar(bins_x,selected_hist_sum,width=bin_size*0.8)
    # plt.show()
    
    # guessed_a = (np.average(bins_x[0:bin_thre],weights=selected_hist_sum[0:bin_thre]))**2
    # guessed_b = (selected_hist_sum[-1]/bins_x[-1] + selected_hist_sum[-2]/bins_x[-2])/2
    # guessed_c = np.amax(selected_hist_sum[0:bin_thre])*sqrt(guessed_a)*e/2
            
    # p0 = [guessed_a,guessed_b,guessed_c]
            
    # try: 
                
    #     popt,_ = optimize.curve_fit(twoD_diff_dis_func,bins_x,selected_hist_sum,p0=p0)
    #     selected_drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
    #     selected_slope = popt[1]/np.sum(selected_hist_sum)
    #     print('Selected Area D: '+str(selected_drate))
    #     print('Selected Area b: '+str(selected_slope))
    #     # smooth_x = np.linspace(0,disp_thre,num=int(disp_thre*100))
    #     # fitted_y = 2*popt[2]*smooth_x/popt[0]*np.exp(-smooth_x**2/popt[0])+popt[1]*smooth_x
    #     # distribution_y = 2*popt[2]*smooth_x/popt[0]*np.exp(-smooth_x**2/popt[0])
    #     # background_y = popt[1]*smooth_x
    #     # plt.plot(smooth_x, fitted_y,'k')
    #     # plt.plot(smooth_x, distribution_y,'r--')
    #     # plt.plot(smooth_x, background_y,'b--')
    #     # plt.show()
                
    # except RuntimeError:
                
    #     print ("RuntimeError")
    
def onPress(event):
    print('Pressed')
    
def onRelease(event):
    print('Release')
    
lineprops = {'color':'red','linewidth':1,'alpha':0.6}
lsso = LassoSelector(ax=ax,onselect=onSelect,props = lineprops,button=1)

fig.canvas.mpl_connect('button_press_event',onPress)
fig.canvas.mpl_connect('button_release_event',onRelease)

plt.show()








