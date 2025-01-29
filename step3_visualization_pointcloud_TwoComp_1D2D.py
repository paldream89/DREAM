import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
import numpy as np
from math import floor,sqrt,e,pi
from scipy import optimize
import matplotlib as mpl
import matplotlib.path as mpltPath
import wx
from ReadSTORMBin import read_storm_bin


# 请注意！！！！！！！！！！！
# 这个程序一维和二维都考虑在内
# displacement threshold
disp_thre = 5

# diffusion rate map upper lower limit
d_high = 4
d_low = 0
pixel_size = 115 # unit is nm
time_interval = 0.0061349  # unit is s, so 1 ms

# point cloud point size
point_size = 0.1

# switch between one component and two component fitting
component = 2  # 1 one component, 2 two component

# percentage of molecules shown, to accelerate
per_mol_shown = 0.2 # 0 to 1

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
bin_thre = int(floor((disp_thre*0.8)/bin_size)) 

def oneD_diff_dis_func(x,a,b,c):
    
    return c*np.exp(-x**2/a)+b

def oneD_diff_dis_func_double(x,a,b,c,a2,c2):
    
    return c*np.exp(-x**2/a)+c2*np.exp(-x**2/a2)+b

def twoD_diff_dis_func(x,a,b,c):
    
    return 2*c*x/a*np.exp(-x**2/a)+b*x

def twoD_diff_dis_func_double(x,a,b,c,a2,c2):
    
    return 2*c*x/a*np.exp(-x**2/a)+2*c2*x/a2*np.exp(-x**2/a2)+b*x

shown_mol_num = int(number_of_points*per_mol_shown)
cmap = mpl.colormaps["jet"]
cmap.set_under(color='black')
cmap.set_bad(color='black')
yc_max = np.amax(STORM_npdata['Yc'])
STORM_npdata['Yc'] = yc_max-STORM_npdata['Yc'] # flip yc
plt.scatter(STORM_npdata['Xc'][0:shown_mol_num],STORM_npdata['Yc'][0:shown_mol_num],
            cmap=cmap,s=point_size,
            c=STORM_npdata['Zc'][0:shown_mol_num],vmin = d_low,vmax = d_high)

plt.show()

fig = plt.gcf()
ax = plt.gca()
x_main_min,x_main_max = ax.get_xlim()
y_main_min,y_main_max = ax.get_ylim()

pix = np.transpose(np.vstack((STORM_npdata['Xc'],STORM_npdata['Yc'])))

selected = np.zeros_like(STORM_npdata['Zc'])-0.001
selected_hist = np.zeros_like(hist_bin,dtype=np.uint16)
selected_hist_sum = np.zeros_like(bins_x,dtype=np.uint16)
selected_hist_sum_1D = np.zeros_like(bins_x,dtype=np.uint16)

def onSelect(x):
    p = mpltPath.Path(x)
    ind = p.contains_points(pix, radius=0)
    
    selected[ind] = STORM_npdata['Zc'][ind]
    plt.figure()
    cmap = mpl.colormaps["jet"]
    cmap.set_under(color='black')
    cmap.set_bad(color='black')
    plt.scatter(STORM_npdata['Xc'][ind],STORM_npdata['Yc'][ind],cmap=cmap,s=point_size,
                c=selected[ind],vmin = d_low,vmax = d_high)
    ax_select = plt.gca()
    ax_select.set_xlim(xmin=x_main_min,xmax=x_main_max)
    ax_select.set_ylim(ymin=y_main_min,ymax=y_main_max)
    
    selected_hist = hist_bin[ind]
    # selected_hist_sum = np.sum(selected_hist, axis=0)
    
    selected_STORMnp = STORM_npdata['Z'][ind]
    indice_2D = selected_STORMnp>5
    indice_1D = selected_STORMnp<5
    
    num_point_2D = np.sum(indice_2D)
    num_point_1D = np.sum(indice_1D)
    
    print('selected 2D: %d 1D: %d' %(num_point_2D,num_point_1D))
    
    selected_hist_sum = np.sum(selected_hist[indice_2D], axis=0)
    selected_hist_sum_1D = np.sum(selected_hist[indice_1D], axis=0)
    total_mol_select = np.size(selected_hist[indice_2D][:,0])
    
    plt.figure()
    plt.bar(bins_x,selected_hist_sum,width=bin_size*0.8)
    plt.show()
    plt.xlim(-0.1,disp_thre+0.1)
    
    if component == 1:
        
        # For 2D fitting, but only 1 component
        guessed_a = (np.average(bins_x[0:bin_thre]))**2
        guessed_b = (selected_hist_sum[-1]/bins_x[-1] + selected_hist_sum[-2]/bins_x[-2])/2
        guessed_c = np.amax(selected_hist_sum[0:bin_thre])*sqrt(guessed_a)*e/2
                
        p0 = [guessed_a,guessed_b,guessed_c]
                
        try: 
                    
            popt,_ = optimize.curve_fit(twoD_diff_dis_func,bins_x,selected_hist_sum,p0=p0)
            selected_drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
            selected_slope = popt[1]/np.sum(selected_hist_sum)
            print('Selected Area Diffusion Rate-2D: '+'%.2f' % selected_drate)
            print('Selected Area background-2D: '+'%.6f' % selected_slope)
            print('see bin_x for x axis; Hist(y axis) is saved as txt')
            save_hist = np.concatenate((bins_x,selected_hist_sum),axis = 0)
            fname = file_path_Hist.replace('.npy','_Sel'+str(total_mol_select)+'2D.txt')
            np.savetxt(fname,save_hist,fmt='%.5f',delimiter=' ',newline='\n')
            print('2D Fitted Curve:')
            print('2*%.0f*x/%.5f*exp(-x^2/%.5f)+%.0f*x' %(popt[2],popt[0],popt[0],popt[1]))
            print('Totol Mol Selected:'+str(total_mol_select))
            
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
            
        # For 1D fitting, but only 1 component
        plt.figure()
        plt.bar(bins_x_1D,selected_hist_sum_1D,width=bin_size_1D*0.8)
        plt.show()
        plt.xlim(-0.7*disp_thre-0.1,0.7*disp_thre+0.1)
        
        guessed_a = 2*(0.3*disp_thre)**2
        guessed_b = selected_hist_sum_1D[-1]
        guessed_c = np.amax(selected_hist_sum_1D)
                
        p0 = [guessed_a,guessed_b,guessed_c]
                
        try: 
                    
            popt,_ = optimize.curve_fit(oneD_diff_dis_func,bins_x_1D,selected_hist_sum_1D,p0=p0)
            selected_drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
            selected_slope = popt[1]/np.sum(selected_hist_sum_1D)
            print('Selected Area Diffusion Rate-1D: '+'%.2f' % selected_drate)
            print('Selected Area background-1D: '+'%.6f' % selected_slope)
            print('see bin_x_1D for x axis; Hist(y axis) is saved as txt')
            save_hist = np.concatenate((bins_x_1D,selected_hist_sum_1D),axis = 0)
            fname = file_path_Hist.replace('.npy','_Sel'+str(total_mol_select)+'1D.txt')
            np.savetxt(fname,save_hist,fmt='%.5f',delimiter=' ',newline='\n')
            print('1D Fitted Curve:')
            print('2*%.0f*exp(-x^2/%.5f)+%.0f' %(popt[2],popt[0],popt[1]))
            print('Totol Mol Selected:'+str(total_mol_select))
            
            smooth_x = np.linspace(-0.7*disp_thre,0.7*disp_thre,num=int(1.4*disp_thre*100))
            fitted_y = popt[2]*np.exp(-smooth_x**2/popt[0])+popt[1]
            distribution_y = popt[2]*np.exp(-smooth_x**2/popt[0])
            background_y = popt[1]+0*smooth_x
            plt.plot(smooth_x, fitted_y,'k')
            plt.plot(smooth_x, distribution_y,'r--')
            plt.plot(smooth_x, background_y,'b--')
            plt.show()
                    
        except RuntimeError:
                    
            print ("RuntimeError")
            
    else:
        
        # For 2D fitting, but 2 components
        guessed_a = (0.3*disp_thre)**2
        guessed_b = (selected_hist_sum[-1]/bins_x[-1] + selected_hist_sum[-2]/bins_x[-2])/2
        guessed_c = np.amax(selected_hist_sum[0:bin_thre])*sqrt(guessed_a)*e/2
        guessed_a2 = (0.3*disp_thre)**2/6
        guessed_c2 = np.amax(selected_hist_sum[0:bin_thre])*sqrt(guessed_a)*e/2
                
        p0 = [guessed_a,guessed_b,guessed_c,guessed_a2,guessed_c2]
        bounds = (0,[disp_thre**2,np.inf,np.inf,disp_thre**2,np.inf])
        
        try: 
                    
            popt,_ = optimize.curve_fit(twoD_diff_dis_func_double,bins_x,selected_hist_sum,p0=p0,bounds=bounds)
            drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
            drate_2 = popt[3] * (pixel_size/1000)**2 /4/time_interval
            
            
            # popt,_ = optimize.curve_fit(twoD_diff_dis_func,bins_x,selected_selected_hist_sum_sum,p0=p0)
            # selected_drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
            slope = popt[1]/np.sum(selected_hist_sum)
            print('Selected Area Diff Rate-2D-1 first D:'+'%.2f' % drate)
            print('Selected Area Diff Rate-2D-2 second D:'+'%.2f' % drate_2)
            print('Selected Area background-2D background: '+'%.6f' % slope)
            smooth_x = np.linspace(0,disp_thre,num=int(disp_thre*100))
            fitted_y = 2*popt[2]*smooth_x/popt[0]*np.exp(-smooth_x**2/popt[0])+popt[1]*smooth_x+2*popt[4]*smooth_x/popt[3]*np.exp(-smooth_x**2/popt[3])
            distribution_y = 2*popt[2]*smooth_x/popt[0]*np.exp(-smooth_x**2/popt[0])
            distribution_y2 = 2*popt[4]*smooth_x/popt[3]*np.exp(-smooth_x**2/popt[3])
            background_y = popt[1]*smooth_x
            D1_percentage = (np.sum(distribution_y)/(np.sum(distribution_y2)+np.sum(distribution_y)))*100
            D2_percentage = (np.sum(distribution_y2)/(np.sum(distribution_y2)+np.sum(distribution_y)))*100
            print('2D Diff Rate first D ratio: '+'%.2f' % D1_percentage)
            print('2D Diff Rate second D ratio: '+'%.2f' % D2_percentage)
            print('see bin_x for x axis; Hist(y axis) is saved as txt')
            save_hist = np.concatenate((bins_x,selected_hist_sum),axis = 0)
            fname = file_path_Hist.replace('.npy','_Sel'+str(total_mol_select)+'2D.txt')
            np.savetxt(fname,save_hist,fmt='%.5f',delimiter=' ',newline='\n')
            print('2D Fitted Curve:')
            print('2*%.0f*x/%.5f*exp(-x^2/%.5f)+%.0f*x+2*%.0f*x/%.5f*exp(-x^2/%.5f)' %(popt[2],popt[0],popt[0],popt[1],popt[4],popt[3],popt[3]))
            print('Totol Mol Selected:'+str(total_mol_select))
            plt.plot(smooth_x, fitted_y,'k')
            plt.plot(smooth_x, distribution_y,'r--')
            plt.plot(smooth_x, distribution_y2,'m--')
            plt.plot(smooth_x, background_y,'b--')
            plt.show()
            
        except RuntimeError:
            
            print('RuntimeError')
            
        # For 1D fitting ,but 2 components
        plt.figure()
        plt.bar(bins_x_1D,selected_hist_sum_1D,width=bin_size_1D*0.8)
        plt.show()
        plt.xlim(-0.7*disp_thre-0.1,0.7*disp_thre+0.1)
        
        guessed_a = 2*(0.3*disp_thre)**2
        guessed_b = selected_hist_sum_1D[-1]
        guessed_c = np.amax(selected_hist_sum_1D)
        guessed_a2 = 2*(0.3*disp_thre)**2/6
        guessed_c2 = np.amax(selected_hist_sum_1D)/3
                
        p0 = [guessed_a,guessed_b,guessed_c,guessed_a2,guessed_c2]
        
        try:
            
            popt,_ = optimize.curve_fit(oneD_diff_dis_func_double,bins_x_1D,selected_hist_sum_1D,p0=p0)
            drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
            drate_2 = popt[3] * (pixel_size/1000)**2 /4/time_interval
            
            slope = popt[1]/np.sum(selected_hist_sum_1D)
            print('Selected Area Diff Rate-1D-1 first D:'+'%.2f' % drate)
            print('Selected Area Diff Rate-1D-2 second D:'+'%.2f' % drate_2)
            print('Selected Area background-1D background: '+'%.6f' % slope)
            smooth_x = np.linspace(-0.7*disp_thre,0.7*disp_thre,num=int(1.4*disp_thre*100))
            fitted_y = popt[2]*np.exp(-smooth_x**2/popt[0])+popt[4]*np.exp(-smooth_x**2/popt[3])+popt[1]
            distribution_y = popt[2]*np.exp(-smooth_x**2/popt[0])
            distribution_y2 = popt[4]*np.exp(-smooth_x**2/popt[3])
            background_y = popt[1]+0*smooth_x
            D1_percentage = (np.sum(distribution_y)/(np.sum(distribution_y2)+np.sum(distribution_y)))*100
            D2_percentage = (np.sum(distribution_y2)/(np.sum(distribution_y2)+np.sum(distribution_y)))*100
            print('1D Diff Rate first D ratio: '+'%.2f' % D1_percentage)
            print('1D Diff Rate second D ratio: '+'%.2f' % D2_percentage)
            print('see bin_x_1D for x axis; Hist(y axis) is saved as txt')
            save_hist = np.concatenate((bins_x_1D,selected_hist_sum_1D),axis = 0)
            fname = file_path_Hist.replace('.npy','_Sel'+str(total_mol_select)+'1D.txt')
            np.savetxt(fname,save_hist,fmt='%.5f',delimiter=' ',newline='\n')
            print('1D Fitted Curve:')
            print('%.0f*exp(-x^2/%.5f)+%.0f*exp(-x^2/%.5f)+%.5f' %(popt[2],popt[0],popt[4],popt[3],popt[1]))
            print('Totol Mol Selected:'+str(total_mol_select))
            plt.plot(smooth_x, fitted_y,'k')
            plt.plot(smooth_x, distribution_y,'r--')
            plt.plot(smooth_x, distribution_y2,'m--')
            plt.plot(smooth_x, background_y,'b--')
            plt.show()
            
        except RuntimeError:
            
            print('RuntimeError')
            
    return selected_hist_sum
    
    # plt.figure()
    # plt.bar(bins_x_1D,selected_hist_sum_1D,width=bin_size_1D*0.8)
    # plt.show()
    
    # guessed_a = 2*(0.3*disp_thre)**2
    # guessed_b = selected_hist_sum_1D[-1]
    # guessed_c = np.amax(selected_hist_sum_1D)
            
    # p0 = [guessed_a,guessed_b,guessed_c]
            
    # try: 
                
    #     popt,_ = optimize.curve_fit(oneD_diff_dis_func,bins_x_1D,selected_hist_sum_1D,p0=p0)
    #     selected_drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
    #     selected_slope = popt[1]
    #     print('Selected Area D-1D: '+str(selected_drate))
    #     print('Selected Area b-1D: '+str(selected_slope))
    #     smooth_x = np.linspace(-0.7*disp_thre,0.7*disp_thre,num=int(1.4*disp_thre*100))
    #     fitted_y = popt[2]*np.exp(-smooth_x**2/popt[0])+popt[1]
    #     # distribution_y = 2*popt[2]*smooth_x/popt[0]*np.exp(-smooth_x**2/popt[0])
    #     # background_y = popt[1]*smooth_x
    #     plt.plot(smooth_x, fitted_y,'k')
    #     # plt.plot(smooth_x, distribution_y,'r--')
    #     # plt.plot(smooth_x, background_y,'b--')
    #     plt.show()
                
    # except RuntimeError:
                
    #     print ("RuntimeError")
        
    # angle_hist,angle_bins = np.histogram(selected_STORMnp[indice_1D],bins=24,range=(-pi,pi))
    # angle_bin_size = 2*pi/24
    # angle_bins = np.arange(-pi+0.5*angle_bin_size, pi, angle_bin_size, dtype=np.float32)
    
    # plt.figure()
    # plt.bar(angle_bins,angle_hist,width=angle_bin_size*0.8)
    # plt.show()

def onPress(event):
    print('Pressed')
    
def onRelease(event):
    print('Release')
    
lineprops = {'color':'red','linewidth':1,'alpha':0.6}
lsso = LassoSelector(ax=ax,onselect=onSelect,props = lineprops,button=1)

fig.canvas.mpl_connect('button_press_event',onPress)
fig.canvas.mpl_connect('button_release_event',onRelease)

plt.show()








