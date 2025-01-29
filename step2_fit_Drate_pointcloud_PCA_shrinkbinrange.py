# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 21:14:52 2022

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:47:32 2020

@author: limin
"""

import numpy as np
from ReadSTORMBin_XcYcZZcFrame import read_storm_bin_XcYcZZcFrame
from math import ceil,sqrt,e,floor
from scipy import optimize
# import matplotlib.pyplot as plt
import time
import wx
from WriteSTORMBin import Write_STORMbin
from functools import partial
from multiprocessing import Pool, freeze_support,cpu_count
from hist_maker_PCA_shrinkbinrange import hist_maker
from drate_fitter_PCA_shrinkbinrange import drate_fitter

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
    pixel_size = 115 # unit is nm
    image_bin_size = 0.86 # unit is pixel
    true_bin_size = int(pixel_size*image_bin_size) 
    # it is also the diameter for averaging
    
    # minimal point number to perform fitting
    min_points_percell = 30
    
    # sortig into an arry first to speed up, maximum points per cell
    sorting_thre = 500
    
    # 0, do not add; 1, add one center point; 3, add three points
    mode_trigger = 1
    disp_thre = 5
    time_interval = 0.00597  # unit is s, so 1 ms
    
    # histogram binning setting
    hist_bin_num = 4*disp_thre # must be an interger
    hist_bin_size = disp_thre/hist_bin_num 
    bins_x = np.arange(0.5*hist_bin_size, disp_thre, hist_bin_size, dtype=np.float32)
    bin_thre = int(floor((disp_thre*0.6)/hist_bin_size))  # for guessed initial fitting parameters
    
    # pca analysis threshold
    pca_ratio_thre = 0.6
    
    # fitting error threshold
    fitting_error = 0.5
    
    # cpu ratio
    cpu_ratio = 2
    
    data_type = np.dtype([('x_inter', np.int16),('y_inter', np.int16),('frame', np.int32),
                          ('disp', np.float32), ('angle', np.float32), 
                          ('xc', np.float32), ('yc', np.float32)])
    
    saved_data_type = np.dtype([('xc', np.float32), ('yc', np.float32),('frame', np.int32),
                          ('diffusion', np.float32), ('angle', np.float32), ('points', np.int32)])
    
    for file_path in file_path_list:
    
        start_time = time.time()
       
        # read the data
        frame_num, number_of_points, STORM_npdata = read_storm_bin_XcYcZZcFrame(file_path)
        
        # data_type = np.dtype([('x_start', np.float32), ('y_start', np.float32),
        #                   ('x_end', np.float32), ('y_end', np.float32),
        #                   ('frame', np.int32),
        #                   ('disp', np.float32), ('angle', np.float32)])    
        
        frame_num = frame_num.item()
        number_of_points = number_of_points.item()
        
        image_size_x = max(np.amax(STORM_npdata['x_start']),np.amax(STORM_npdata['x_end']))
        image_size_y = max(np.amax(STORM_npdata['y_start']),np.amax(STORM_npdata['y_end']))
        
        xinter_max = int(ceil(image_size_x/image_bin_size))
        yinter_max = int(ceil(image_size_y/image_bin_size))
        
        counter_map = np.zeros((xinter_max,yinter_max),dtype=np.uint16)
        indice_map = np.zeros((xinter_max,yinter_max,sorting_thre),dtype=np.int32)-1
            
        xinter_yinter_frame_disp_angle_xcyc = np.zeros((mode_trigger+2)*number_of_points,
                                                       dtype=data_type)
        temp_index = np.arange(mode_trigger+2)
        temp_index2 = np.stack((np.flip(temp_index)/(mode_trigger+1),\
                                temp_index/(mode_trigger+1)),axis=1)
        xinter_yinter_frame_disp_angle_xcyc['frame'] = np.tile(STORM_npdata['frame'], \
                                                            mode_trigger+2)
        xinter_yinter_frame_disp_angle_xcyc['disp'] = np.tile(STORM_npdata['disp'], \
                                                            mode_trigger+2)
        xinter_yinter_frame_disp_angle_xcyc['angle'] = np.tile(STORM_npdata['angle'], \
                                                            mode_trigger+2)
        xinter_yinter_frame_disp_angle_xcyc['xc'] = \
            ((np.dot(temp_index2,np.stack((STORM_npdata['x_start'],STORM_npdata['x_end']),axis=0))\
              .reshape(-1,1).squeeze(axis=1))/image_bin_size)
        xinter_yinter_frame_disp_angle_xcyc['yc'] = \
            ((np.dot(temp_index2,np.stack((STORM_npdata['y_start'],STORM_npdata['y_end']),axis=0))\
              .reshape(-1,1).squeeze(axis=1))/image_bin_size)
        xinter_yinter_frame_disp_angle_xcyc['x_inter'] = \
            xinter_yinter_frame_disp_angle_xcyc['xc'].astype(np.int16)
        xinter_yinter_frame_disp_angle_xcyc['y_inter'] = \
            xinter_yinter_frame_disp_angle_xcyc['yc'].astype(np.int16)
        
        # sorting start
        current_sort = 0
        while current_sort < (mode_trigger+2)*number_of_points:
            x = xinter_yinter_frame_disp_angle_xcyc['x_inter'][current_sort]
            y = xinter_yinter_frame_disp_angle_xcyc['y_inter'][current_sort]
            if counter_map[x,y] < sorting_thre:
                indice_map[x,y,counter_map[x,y]] = current_sort
                counter_map[x,y] = counter_map[x,y] + 1
                current_sort = current_sort + 1
            else:
                current_sort = current_sort + 1
        # sorting finished
        
        # order_indice = np.lexsort((xinter_yinter_frame_disp_angle_xcyc['y_inter'],\
        #                            xinter_yinter_frame_disp_angle_xcyc['x_inter']))
            
        # xinter_yinter_frame_disp_angle_xcyc = xinter_yinter_frame_disp_angle_xcyc[order_indice]
        
        # all the x,y unit is pixel, and the pixel size is defined as above:
        # if pixel_size = 160 # unit is nm
        # image_bin_size = 0.625 # unit is pixel
        # then here the unit is 0.625*160 = 100 nm
        # averaged_xc_yc_frame_diffusion_angle = np.zeros(2*number_of_points,dtype=saved_data_type)
        
        print('Multi-processes histogram_making...')
        with Pool(processes = int(cpu_count()/cpu_ratio)) as pool:
            averaged_hist_array = pool.map(partial(hist_maker,disp_thre=disp_thre,
                                                  hist_bin_num=hist_bin_num,
                                                  image_bin_size=image_bin_size,
                                                  xinter_yinter_frame_disp_angle_xcyc=xinter_yinter_frame_disp_angle_xcyc,
                                                  indice_map=indice_map,counter_map=counter_map,
                                                  xinter_max=xinter_max,yinter_max=yinter_max,
                                                  min_points_percell=min_points_percell,
                                                  pca_ratio_thre=pca_ratio_thre),STORM_npdata)
            pool.close()
        
        # averaged_hist_array includes: histogram at 5-24 and 30-49 (bin_number = 20)
        # averaged_xc_start,averaged_yc_start,averaged_frame_start,
        #                          averaged_angle_start,averaged_points_start,averaged_histogram_start,
        #                          averaged_xc_end,averaged_yc_end,averaged_frame_end,
        #                         averaged_angle_end,averaged_points_end,averaged_histogram_end
        
        print('Multi-processes drate fitting...')
        with Pool(processes = int(cpu_count()/cpu_ratio)) as pool:
            averaged_drate = pool.map(partial(drate_fitter,disp_thre=disp_thre,
                                                   hist_bin_num=hist_bin_num,
                                                   pixel_size=pixel_size,
                                                   min_points_percell=min_points_percell,
                                                   time_interval=time_interval,
                                                   fitting_error=fitting_error), averaged_hist_array)
            pool.close()
        
        print('Data Saving...')
        
        # convert list to np array
        averaged_hist_array = np.array(averaged_hist_array,dtype=np.float32)
        averaged_drate = np.array(averaged_drate,dtype=np.float32)
        
        averaged_drate_stack = np.hstack((averaged_drate[:,0],averaged_drate[:,1]))
        remove_unuseful_indice = averaged_drate_stack > 0
        averaged_drate_stack = averaged_drate_stack[remove_unuseful_indice]
        useful_number_of_points = np.size(averaged_drate_stack)
        
        height_array = np.ones(useful_number_of_points, dtype=np.float32)*206.031
        area_array = np.ones(useful_number_of_points, dtype=np.float32)*968.889
        width_array = np.ones(useful_number_of_points, dtype=np.float32)*279.095
        phi_array = np.zeros(useful_number_of_points, dtype=np.float32)
        ax_array = np.ones(useful_number_of_points, dtype=np.float32)
        bg_array = np.ones(useful_number_of_points, dtype=np.float32)*394.159
        I_array = np.ones(useful_number_of_points, dtype=np.float32)*1476.02
        category_array = np.zeros(useful_number_of_points, dtype=np.int32)
        length_array = np.ones(useful_number_of_points, dtype=np.int32)
        link_array = -np.ones(useful_number_of_points, dtype=np.int32)
        z_array_angle = (np.hstack((averaged_hist_array[:,3].astype(np.float32),averaged_hist_array[:,8+hist_bin_num].astype(np.float32))))[remove_unuseful_indice]
        category_array[z_array_angle > 10] = 1
        save_file_path = file_path.replace('.bin','-avePCA'+str(true_bin_size)+'nm'+'.bin')

        Write_STORMbin(save_file_path, np.int32(useful_number_of_points), 1000,
                        (np.hstack((averaged_hist_array[:,0].astype(np.float32),averaged_hist_array[:,5+hist_bin_num].astype(np.float32))))[remove_unuseful_indice],
                        (np.hstack((averaged_hist_array[:,1].astype(np.float32),averaged_hist_array[:,6+hist_bin_num].astype(np.float32))))[remove_unuseful_indice], 
                        (np.hstack((averaged_hist_array[:,0].astype(np.float32),averaged_hist_array[:,5+hist_bin_num].astype(np.float32))))[remove_unuseful_indice],
                        (np.hstack((averaged_hist_array[:,1].astype(np.float32),averaged_hist_array[:,6+hist_bin_num].astype(np.float32))))[remove_unuseful_indice],
                        height_array, area_array, width_array, phi_array, ax_array, bg_array, 
                        I_array, category_array, 
                        (np.hstack((averaged_hist_array[:,4].astype(np.int32),averaged_hist_array[:,9+hist_bin_num].astype(np.int32))))[remove_unuseful_indice],
                        (np.hstack((averaged_hist_array[:,2].astype(np.float32),averaged_hist_array[:,7+hist_bin_num].astype(np.float32))))[remove_unuseful_indice], 
                        length_array, link_array, 
                        z_array_angle, 
                        averaged_drate_stack)
        
# averaged_xc_start,averaged_yc_start,averaged_frame_start,
# averaged_angle_start,averaged_points_start,averaged_histogram_start,

    # data_type = np.dtype([('X', np.float32), ('Y', np.float32), ('Xc', np.float32), ('Yc', np.float32),
    #                ('Height', np.float32), ('Area', np.float32), ('Width', np.float32), ('Phi', np.float32), 
    #                ('Ax', np.float32), ('BG', np.float32), ('I', np.float32), ('Category', np.int32), 
    #                ('Valid', np.int32), ('Frame', np.int32), ('Length', np.int32), ('Link', np.int32), 
    #                ('Z', np.float32), ('Zc', np.float32)])
        
        hist_map = (np.vstack((averaged_hist_array[:,5:5+hist_bin_num].astype(np.int32),averaged_hist_array[:,10+hist_bin_num:10+2*hist_bin_num].astype(np.int32))))[remove_unuseful_indice]
        save_file_hist_map = file_path.replace('.bin','-HistPCA'+'-Add'+str(mode_trigger)+\
                                                   'pt-'+'bsize'+str(true_bin_size)+'nm'+\
                                                        '-bnum'+str(hist_bin_num)+'.npy')
        np.save(save_file_hist_map,hist_map)
        
        print("--- %s seconds ---" % (time.time() - start_time))
        print("--- %5.3f useful points percentage---" % (useful_number_of_points/2/number_of_points))
        
        # saved_data_type = np.dtype([('xc', np.float32), ('yc', np.float32),('frame', np.int32),
        #                       ('diffusion', np.float32), ('angle', np.float32), ('points', np.int32)])
    
    