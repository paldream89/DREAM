# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:37:22 2020

@author: xlm69
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
import numpy as np
from math import floor,sqrt,e
from scipy import optimize
import matplotlib.path as mpltPath
import wx

# displacement threshold
disp_thre = 3

# diffusion rate map upper lower limit
d_high = 10
d_low = 0
pixel_size = 160 # unit is nm
time_interval = 0.001 # unit is s, so 1 ms

app = wx.App()
frame = wx.Frame(None, -1, 'win.py')
frame.SetSize(0,0,200,50)

# Create open file dialog
openFileDialog = wx.FileDialog(frame, "Select Drate Map txt", "", "", 
      "txt files (*.txt)|*.txt", 
       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

openFileDialog.ShowModal()
file_path_Drate = openFileDialog.GetPath()
openFileDialog.Destroy()

openFileDialog = wx.FileDialog(frame, "Select Histogram npy", "", "", 
      "numpy files (*.npy)|*.npy", 
       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

openFileDialog.ShowModal()
file_path_Hist = openFileDialog.GetPath()
openFileDialog.Destroy()

drate_map = np.loadtxt(file_path_Drate,dtype=np.float32,delimiter='\t')
hist_map = np.load(file_path_Hist)
bin_number = len(hist_map[0,0,:])

bin_size = disp_thre/bin_number
bins_x = np.arange(0.5*bin_size, disp_thre, bin_size, dtype=np.float32)
bin_thre = int(floor((disp_thre*0.6)/bin_size))

def twoD_diff_dis_func(x,p):
    
    a,b,c = p
    return 2*c*x/a*np.exp(-x**2/a)+b*x

def func_error(p,y,x):
    
    return np.sum((y-twoD_diff_dis_func(x,p))**2)

plt.figure()
cmap = plt.cm.jet
cmap.set_under(color='black')
cmap.set_bad(color='black')
plt.imshow(drate_map, cmap=cmap, interpolation='nearest', 
           vmin = d_low, vmax = d_high)

fig = plt.gcf()
ax = plt.gca()

x, y = np.meshgrid(np.arange(drate_map.shape[1]), np.arange(drate_map.shape[0]))
pix = np.stack((x.flatten(), y.flatten()),axis=1)

def onSelect(x):
    p = mpltPath.Path(x)
    ind = p.contains_points(pix, radius=0)
    
    selected = np.zeros_like(drate_map)-0.001
    selected.flat[ind] = drate_map.flat[ind]
    plt.figure()
    cmap = plt.cm.jet
    cmap.set_under(color='black')
    cmap.set_bad(color='black')
    plt.imshow(selected, cmap=cmap, interpolation='nearest', 
            vmin = d_low, vmax = d_high)
    
    selected_hist = np.zeros_like(hist_map,dtype=np.uint16)
    selected_hist.reshape(-1,bin_number)[ind,:] = hist_map.reshape(-1,bin_number)[ind,:]
    selected_hist_sum = np.sum(np.sum(selected_hist, axis=0), axis=0)
    plt.figure()
    plt.bar(bins_x,selected_hist_sum,width=bin_size*0.8)
    plt.show()
    
    guessed_a = (np.average(bins_x[0:bin_thre],weights=selected_hist_sum[0:bin_thre]))**2
    guessed_b = (selected_hist_sum[-1]/bins_x[-1] + selected_hist_sum[-2]/bins_x[-2])/2
    guessed_c = np.amax(selected_hist_sum[0:bin_thre])*sqrt(guessed_a)*e/2
            
    p0 = [guessed_a,guessed_b,guessed_c]
            
    try: 
                
        popt = optimize.basinhopping(func_error,p0,niter=10,
                                     minimizer_kwargs={'method':'L-BFGS-B',
                                                       'args':(selected_hist_sum,bins_x)})
        
        selected_drate = popt.x[0] * (pixel_size/1000)**2 /4/time_interval
        selected_slope = popt.x[1]/np.sum(selected_hist_sum)
        print('Selected Area D: '+str(selected_drate))
        print('Selected Area b: '+str(selected_slope))
        smooth_x = np.linspace(0,disp_thre,num=int(disp_thre*100))
        fitted_y = 2*popt.x[2]*smooth_x/popt.x[0]*np.exp(-smooth_x**2/popt.x[0])\
            +popt.x[1]*smooth_x
        distribution_y = 2*popt.x[2]*smooth_x/popt.x[0]*np.exp(-smooth_x**2/popt.x[0])
        background_y = popt.x[1]*smooth_x
        plt.plot(smooth_x, fitted_y,'k')
        plt.plot(smooth_x, distribution_y,'r--')
        plt.plot(smooth_x, background_y,'b--')
        plt.show()
        # 2*c*x/a*np.exp(-x**2/a)+b*x
                
    except RuntimeError:
                
        print ("RuntimeError")
    
def onPress(event):
    print('Pressed')
    
def onRelease(event):
    print('Release')
    
lineprops = {'color':'red','linewidth':1,'alpha':0.6}
lsso = LassoSelector(ax=ax,onselect=onSelect,lineprops = lineprops,button=1)

fig.canvas.mpl_connect('button_press_event',onPress)
fig.canvas.mpl_connect('button_release_event',onRelease)

plt.show()



