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
import matplotlib.path as mpltPath
import wx
from ReadSTORMBin_XcYcZZcFrame import read_storm_bin_XcYcZZcFrame

def twoD_diff_dis_func(x,a,b,c):
    
    return 2*c*x/a*np.exp(-x**2/a)+b*x


# displacement threshold
disp_thre = 5

# diffusion rate map upper lower limit
d_high = disp_thre
d_low = 0
pixel_size = 115 # unit is nm
time_interval = 0.00532 # unit is s, so 1 ms

# point cloud point size
point_size = 0.01

# histogram binning setting
bin_num = disp_thre*4 # must be an interger
bin_size = disp_thre/bin_num 
bins_x = np.arange(0.5*bin_size, disp_thre, bin_size, dtype=np.float32)
bin_thre = int(floor((disp_thre*0.6)/bin_size))

# angle binning setting
angle_bin_num = 20  # from -pi to pi
angle_bin_size = 2*pi/angle_bin_num
angle_bins_x = np.arange(-pi+0.5*angle_bin_size, pi, angle_bin_size, dtype=np.float32)

# initialize saved data
drate_hist_array = np.zeros((angle_bin_num,bin_num),dtype=np.uint32) 
drate_array = np.zeros(angle_bin_num,dtype=np.float32)

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
        
cmap = plt.cm.jet
cmap.set_under(color='black')
cmap.set_bad(color='black')
plt.scatter(STORM_npdata['x_start'],STORM_npdata['y_start'],cmap=cmap,s=point_size,
            c=STORM_npdata['disp'],vmin = d_low,vmax = d_high)

plt.show()

fig = plt.gcf()
ax = plt.gca()
x_main_min,x_main_max = ax.get_xlim()
y_main_min,y_main_max = ax.get_ylim()

pix = np.transpose(np.vstack((STORM_npdata['x_start'],STORM_npdata['y_start'])))

def onSelect(x):
    p = mpltPath.Path(x)
    ind = p.contains_points(pix, radius=0)
    
    selected = np.zeros_like(STORM_npdata['disp'])-0.001
    selected = STORM_npdata['disp'][ind]
    selected_angle = STORM_npdata['angle'][ind]
    plt.figure()
    cmap = plt.cm.jet
    cmap.set_under(color='black')
    cmap.set_bad(color='black')
    plt.scatter(STORM_npdata['x_start'][ind],STORM_npdata['y_start'][ind],cmap=cmap,s=point_size,
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
    
    while cursor_B < size_selected and cursor_angle<angle_bin_num+1:
        
        cursor_A = cursor_B
        
        while selected_angle[cursor_B]<-pi+cursor_angle*angle_bin_size:
            
            cursor_B += 1
            
            if cursor_B >= size_selected:
                break
        
        hist,_ = np.histogram(selected[cursor_A:cursor_B], 
                                       bins = bin_num, range = (0,disp_thre))
        
        drate_hist_array[cursor_angle-1,:] = np.uint32(hist)
        
        guessed_a = (0.3*disp_thre)**2
        guessed_b = (hist[-1]/bins_x[-1] + hist[-2]/bins_x[-2])/2
        guessed_c = np.amax(hist[0:bin_thre])*sqrt(guessed_a)*e/2
                
        p0 = [guessed_a,guessed_b,guessed_c]
        bounds = (0,[disp_thre**2,np.inf,np.inf])
        
        try: 
                    
            popt,_ = optimize.curve_fit(twoD_diff_dis_func,bins_x,hist,p0=p0,bounds=bounds)
            drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
            
        except RuntimeError:
            
            print('RuntimeError')
            drate = 0
        
        drate_array[cursor_angle-1] = drate
        cursor_angle += 1
        
    plt.figure()
    plt.plot(angle_bins_x,drate_array,'b')
    plt.show()
        
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
    
props = {'color':'red','linewidth':1,'alpha':0.6}
lsso = LassoSelector(ax=ax,onselect=onSelect,props = props,button=1)

fig.canvas.mpl_connect('button_press_event',onPress)
fig.canvas.mpl_connect('button_release_event',onRelease)

plt.show()








