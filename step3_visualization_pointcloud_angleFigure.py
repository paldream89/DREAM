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
from ReadSTORMBin import read_storm_bin

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors

# displacement threshold
disp_thre = 5

# diffusion rate map upper lower limit
d_high = pi
d_low = 0
pixel_size = 115 # unit is nm
time_interval = 0.00532 # unit is s, so 1 ms

# point cloud point size
point_size = 0.01

app = wx.App()
frame = wx.Frame(None, -1, 'win.py')
frame.SetSize(0,0,200,50)

# Create open file dialog
openFileDialog = wx.FileDialog(frame, "Select STORM bin", "", "", 
      "bin files (*.bin)|*.bin", 
       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

openFileDialog.ShowModal()
file_path_STORMBin = openFileDialog.GetPath()
openFileDialog.Destroy()

frame_number, number_of_points, STORM_npdata = read_storm_bin(file_path_STORMBin)

indice = STORM_npdata['Category']==0

scale = np.linspace(0, 255, 256).astype(np.int16)
scale_reverrse = scale[::-1]
zero_256 = np.zeros(256).astype(np.int16)
ones_768 = np.ones(768).astype(np.int16)*256

red_scale = np.hstack((scale_reverrse,zero_256,scale))
green_scale = np.hstack((scale,scale_reverrse,zero_256))
blue_scale = np.hstack((zero_256,scale,scale_reverrse))

angle_scale_bar = np.transpose(np.vstack((red_scale,green_scale,blue_scale,ones_768)))/256


plt.figure(figsize=(12,8))
ax = plt.axes()
ax.set_facecolor("black")
ax.set_aspect('equal','box')

cmap = ListedColormap(angle_scale_bar)

cmap.set_under(color='black')
cmap.set_bad(color='black')

plt.scatter(STORM_npdata['Xc'][indice],STORM_npdata['Yc'][indice],cmap=cmap,s=point_size,alpha=0.5,
            c=STORM_npdata['Zc'][indice],vmin = d_low,vmax = d_high)
plt.colorbar()
plt.show()



# fig = plt.gcf()
# ax = plt.gca()
# x_main_min,x_main_max = ax.get_xlim()
# y_main_min,y_main_max = ax.get_ylim()

# pix = np.transpose(np.vstack((STORM_npdata['Xc'],STORM_npdata['Yc'])))

# def onSelect(x):
#     p = mpltPath.Path(x)
#     ind = p.contains_points(pix, radius=0)
    
#     selected = np.zeros_like(STORM_npdata['Zc'])-0.001
#     selected[ind] = STORM_npdata['Zc'][ind]
#     plt.figure()
#     cmap = plt.cm.jet
#     cmap.set_under(color='black')
#     cmap.set_bad(color='black')
#     plt.scatter(STORM_npdata['Xc'][ind],STORM_npdata['Yc'][ind],cmap=cmap,s=point_size,
#                 c=selected[ind],vmin = d_low,vmax = d_high)
#     ax_select = plt.gca()
#     ax_select.set_xlim(xmin=x_main_min,xmax=x_main_max)
#     ax_select.set_ylim(ymin=y_main_min,ymax=y_main_max)
    
#     selected_hist = np.zeros_like(hist_bin,dtype=np.uint16)
#     selected_hist[ind] = hist_bin[ind]
#     selected_hist_sum = np.sum(selected_hist, axis=0)
#     plt.figure()
#     plt.bar(bins_x,selected_hist_sum,width=bin_size*0.8)
#     plt.show()
    
#     guessed_a = (np.average(bins_x[0:bin_thre],weights=selected_hist_sum[0:bin_thre]))**2
#     guessed_b = (selected_hist_sum[-1]/bins_x[-1] + selected_hist_sum[-2]/bins_x[-2])/2
#     guessed_c = np.amax(selected_hist_sum[0:bin_thre])*sqrt(guessed_a)*e/2
            
#     p0 = [guessed_a,guessed_b,guessed_c]
            
#     try: 
                
#         popt,_ = optimize.curve_fit(twoD_diff_dis_func,bins_x,selected_hist_sum,p0=p0)
#         selected_drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
#         selected_slope = popt[1]/np.sum(selected_hist_sum)
#         print('Selected Area D: '+str(selected_drate))
#         print('Selected Area b: '+str(selected_slope))
#         smooth_x = np.linspace(0,disp_thre,num=int(disp_thre*100))
#         fitted_y = 2*popt[2]*smooth_x/popt[0]*np.exp(-smooth_x**2/popt[0])+popt[1]*smooth_x
#         distribution_y = 2*popt[2]*smooth_x/popt[0]*np.exp(-smooth_x**2/popt[0])
#         background_y = popt[1]*smooth_x
#         plt.plot(smooth_x, fitted_y,'k')
#         plt.plot(smooth_x, distribution_y,'r--')
#         plt.plot(smooth_x, background_y,'b--')
#         plt.show()
                
#     except RuntimeError:
                
#         print ("RuntimeError")
    
# def onPress(event):
#     print('Pressed')
    
# def onRelease(event):
#     print('Release')
    
# lineprops = {'color':'red','linewidth':1,'alpha':0.6}
# lsso = LassoSelector(ax=ax,onselect=onSelect,lineprops = lineprops,button=1)

# fig.canvas.mpl_connect('button_press_event',onPress)
# fig.canvas.mpl_connect('button_release_event',onRelease)

# plt.show()








