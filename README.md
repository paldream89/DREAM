1. Run step1_find_disp.py 
Input: This python script read a binary file (.bin), containing the following data type for identified single molecules:
data_type = np.dtype([('X', np.float32), ('Y', np.float32), ('Xc', np.float32), ('Yc', np.float32), ('Height', np.float32), ('Area', np.float32), ('Width', np.float32), ('Phi', np.float32),  ('Ax', np.float32), ('BG', np.float32), ('I', np.float32), ('Category', np.int32),  ('Valid', np.int32), ('Frame', np.int32), ('Length', np.int32), ('Link', np.int32),  ('Z', np.float32), ('Zc', np.float32)]).
 
Parameter setting: The disp_thre means the searching radius for paired single molecules between consecutive frames. Unit is pixel.
Also, another script step1_find_disp_1-2and2-3.py can be used instead of the current one, if all consecutive frames are paired to obtain displacement
Output: Generate a file ending with dis7.bin (if “disp_thre” was set to be 7). containing the following data type for paired single molecules:
data_type = np.dtype([('x_start', np.float32), ('y_start', np.float32), ('x_end', np.float32), ('y_end', np.float32), ('frame', np.int32), ('disp', np.float32), ('angle', np.float32)])   

2.Run step2_fit_Drate_pcBleed_PCA_shrinkbinrange.py
Input: The file generated via Step 1 ending with dis7.bin.
Parameter setting: 
 
Pixel_szie: pixel size for the camera.
image_bin_size: radius used for the Gaussian Convolution Kernel
 
Min_points_percell: Only performed fitting if total molecule count is larger than this
  
Disp_thre: the number you used in the step1
Time_interval: the separation time of two consecutive frames
 
Hist_bin_num: histogram bin number for constructing displacement histogram
 
Pca_ratio_thre: for PCA analysis threshold. If all 2D, use 0.99. However, if there are some 1D motion. Use ~0.6
 
Cpu_used: How many CPU will you use for running this program
Output: a binary file (.bin), containing the following data type for identified single molecules. The diffusion coefficients were stored in Zc, and the diffusion angles were stored in Z:
data_type = np.dtype([('X', np.float32), ('Y', np.float32), ('Xc', np.float32), ('Yc', np.float32), ('Height', np.float32), ('Area', np.float32), ('Width', np.float32), ('Phi', np.float32),  ('Ax', np.float32), ('BG', np.float32), ('I', np.float32), ('Category', np.int32),  ('Valid', np.int32), ('Frame', np.int32), ('Length', np.int32), ('Link', np.int32),  ('Z', np.float32), ('Zc', np.float32)]).
The DREAM image can be reconstructed by opening the binary file with open-source SMLM image-processing software, such as Insight3 developed by Bo Huang and Xiaowei Zhuang. In the reconstructed image, color encodes the Zc value, which represents the local diffusion coefficient.
Another output file ending with “.npy” stores all the displacement histograms for each single molecules.
Examples of Raw data is also included in this folder, named with:
 

