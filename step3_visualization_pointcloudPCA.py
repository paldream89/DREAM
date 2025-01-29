# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:50:44 2022

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
import matplotlib.path as mpltPath
import wx
from ReadSTORMBin import read_storm_bin

# displacement threshold
disp_thre = 9

# diffusion rate map upper lower limit
d_high = 30
d_low = 0
pixel_size = 110 # unit is nm
time_interval = 0.001  # unit is s, so 1 ms

# point cloud point size
point_size = 1

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

openFileDialog = wx.FileDialog(frame, "Select Histogram npy", "", "", 
      "numpy files (*.npy)|*.npy", 
       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

openFileDialog.ShowModal()
file_path_Hist = openFileDialog.GetPath()
openFileDialog.Destroy()

frame_number, number_of_points, STORM_npdata = read_storm_bin(file_path_STORMBin)
hist_bin = np.load(file_path_Hist)
bin_number = len(hist_bin[0,:])

bin_size = disp_thre/bin_number
bin_size_1D = 1.4*disp_thre/bin_number
bins_x = np.arange(0.5*bin_size, disp_thre, bin_size, dtype=np.float32)

bins_x_1D = np.arange(-0.7*disp_thre+0.5*bin_size_1D, 0.7*disp_thre, bin_size_1D, dtype=np.float32)

# for guessed initial fitting parameters
bin_thre = int(floor((disp_thre*0.6)/bin_size)) 

def oneD_diff_dis_func(x,a,b,c):
    
    return c*np.exp(-x**2/a)+b

def twoD_diff_dis_func(x,a,b,c):
    
    return 2*c*x/a*np.exp(-x**2/a)+b*x

cmap = plt.cm.jet
cmap.set_under(color='black')
cmap.set_bad(color='black')
plt.scatter(STORM_npdata['Xc'],STORM_npdata['Yc'],cmap=cmap,s=point_size,
            c=STORM_npdata['Zc'],vmin = d_low,vmax = d_high)

plt.show()

fig = plt.gcf()
ax = plt.gca()
x_main_min,x_main_max = ax.get_xlim()
y_main_min,y_main_max = ax.get_ylim()

pix = np.transpose(np.vstack((STORM_npdata['Xc'],STORM_npdata['Yc'])))

def onSelect(x):
    p = mpltPath.Path(x)
    ind = p.contains_points(pix, radius=0)
    
    selected = np.zeros_like(STORM_npdata['Zc'])-0.001
    selected[ind] = STORM_npdata['Zc'][ind]
    plt.figure()
    cmap = plt.cm.jet
    cmap.set_under(color='black')
    cmap.set_bad(color='black')
    plt.scatter(STORM_npdata['Xc'][ind],STORM_npdata['Yc'][ind],cmap=cmap,s=point_size,
                c=selected[ind],vmin = d_low,vmax = d_high)
    ax_select = plt.gca()
    ax_select.set_xlim(xmin=x_main_min,xmax=x_main_max)
    ax_select.set_ylim(ymin=y_main_min,ymax=y_main_max)
    
    selected_hist = np.zeros_like(hist_bin,dtype=np.uint16)
    selected_hist = hist_bin[ind]
    # selected_hist_sum = np.sum(selected_hist, axis=0)
    
    selected_STORMnp = STORM_npdata['Z'][ind]
    indice_2D = selected_STORMnp>5
    indice_1D = selected_STORMnp<5
    
    selected_hist_sum = np.sum(selected_hist[indice_2D], axis=0)
    selected_hist_sum_1D = np.sum(selected_hist[indice_1D], axis=0)
    
    plt.figure()
    plt.bar(bins_x,selected_hist_sum,width=bin_size*0.8)
    plt.show()
    
    guessed_a = (np.average(bins_x[0:bin_thre]))**2
    guessed_b = (selected_hist_sum[-1]/bins_x[-1] + selected_hist_sum[-2]/bins_x[-2])/2
    guessed_c = np.amax(selected_hist_sum[0:bin_thre])*sqrt(guessed_a)*e/2
            
    p0 = [guessed_a,guessed_b,guessed_c]
            
    try: 
                
        popt,_ = optimize.curve_fit(twoD_diff_dis_func,bins_x,selected_hist_sum,p0=p0)
        selected_drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
        selected_slope = popt[1]/np.sum(selected_hist_sum)
        print('Selected Area D-2D: '+str(selected_drate))
        print('Selected Area b-2D: '+str(selected_slope))
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
    
    plt.figure()
    plt.bar(bins_x_1D,selected_hist_sum_1D,width=bin_size_1D*0.8)
    plt.show()
    
    guessed_a = 2*(0.3*disp_thre)**2
    guessed_b = selected_hist_sum_1D[-1]
    guessed_c = np.amax(selected_hist_sum_1D)
            
    p0 = [guessed_a,guessed_b,guessed_c]
            
    try: 
                
        popt,_ = optimize.curve_fit(oneD_diff_dis_func,bins_x_1D,selected_hist_sum_1D,p0=p0)
        selected_drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
        selected_slope = popt[1]
        print('Selected Area D-1D: '+str(selected_drate))
        print('Selected Area b-1D: '+str(selected_slope))
        smooth_x = np.linspace(-0.7*disp_thre,0.7*disp_thre,num=int(1.4*disp_thre*100))
        fitted_y = popt[2]*np.exp(-smooth_x**2/popt[0])+popt[1]
        # distribution_y = 2*popt[2]*smooth_x/popt[0]*np.exp(-smooth_x**2/popt[0])
        # background_y = popt[1]*smooth_x
        plt.plot(smooth_x, fitted_y,'k')
        # plt.plot(smooth_x, distribution_y,'r--')
        # plt.plot(smooth_x, background_y,'b--')
        plt.show()
                
    except RuntimeError:
                
        print ("RuntimeError")
        
    angle_hist,angle_bins = np.histogram(selected_STORMnp[indice_1D],bins=24,range=(-pi,pi))
    angle_bin_size = 2*pi/24
    angle_bins = np.arange(-pi+0.5*angle_bin_size, pi, angle_bin_size, dtype=np.float32)
    
    plt.figure()
    plt.bar(angle_bins,angle_hist,width=angle_bin_size*0.8)
    plt.show()

def onPress(event):
    print('Pressed')
    
def onRelease(event):
    print('Release')
    
lineprops = {'color':'red','linewidth':1,'alpha':0.6}
lsso = LassoSelector(ax=ax,onselect=onSelect,lineprops = lineprops,button=1)

fig.canvas.mpl_connect('button_press_event',onPress)
fig.canvas.mpl_connect('button_release_event',onRelease)

plt.show()








