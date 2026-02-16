import matplotlib.pyplot as plt 
import numpy as np

base_dir = "/home/geuba03p/weta_project/weta_videos_cropped/processed_trajectories"
file_base_name = "hcrass1f_trial_redo_20250203_120147_center_mm"
tra_pos = f"{base_dir}/{file_base_name}_raw.npy" 
tra_f_pos = f"{base_dir}/{file_base_name}.npy" 

tra = np.load(tra_pos)
tra_f = np.load(tra_f_pos)

plt.plot(tra[:,0],tra[:,1],label='raw detection', linewidth=2)
plt.plot(tra_f[:,0],tra_f[:,1],label='filtered detection',linewidth=1)
plt.xlabel('coordinate, mm')
plt.ylabel('coordinate, mm')
plt.axis('scaled') 
plt.legend()
plt.show()