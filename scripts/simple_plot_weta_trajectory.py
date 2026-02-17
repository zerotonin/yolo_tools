import matplotlib.pyplot as plt 
import numpy as np

base_dir = "/home/geuba03p/weta_project/weta_videos_cropped/processed_trajectories"
file_base_name = "hcrass1f_trial_redo_20250203_120147"
tra_pos = f"{base_dir}/{file_base_name}_center_mm_raw.npy" 
tra_f_pos = f"{base_dir}/{file_base_name}_center_mm.npy" 
tra_temp_pos = f"{base_dir}/{file_base_name}_animal_temperature.npy"

tra = np.load(tra_pos)
tra_f = np.load(tra_f_pos)
tra_temp = np.load(tra_temp_pos)

plt.plot(tra[:,0],tra[:,1],label='raw detection', linewidth=2, color='blue')
plt.plot(tra_f[:,0],tra_f[:,1],label='filtered detection',linewidth=1,color='green')
plt.xlabel('coordinate, mm')
plt.ylabel('coordinate, mm')
plt.axis('scaled') 
plt.legend()

plt.figure()
time_axis = np.linspace(0,len(tra_temp)/(25*60),len(tra_temp))
plt.plot(time_axis,tra_temp,label='animal temperature', linewidth=2, color='orange')
plt.xlabel('time, min')
plt.ylabel('temperature, °C')


plt.show()