

import h5py
import librosa
import numpy as np
import os
import tqdm

def read_data(file_path):
    SIGNAL = {}
    f = h5py.File(file_path)
    for k, v in f.items():
        SIGNAL[k] = np.array(v)
    file_name = os.path.basename(file_path)

    SIGNAL=SIGNAL[file_name.split(".")[0]]
    return SIGNAL
    

st_factor = [0.8, 0.9, 1.0, 1.1, 1.2]
sh_factor = [0.2, 0.4, 1.0, 0.6]
def data_augmentation(data_root, save_path):
    os.makedirs(save_path, exist_ok=True)
    class_dir_names = os.listdir(data_root)

    phase_data_save_path = os.path.join(save_path, "phase")
    intensity_data_save_path = os.path.join(save_path, "intensity")
    os.makedirs(phase_data_save_path, exist_ok=True)
    os.makedirs(intensity_data_save_path, exist_ok=True)

    for class_dir_name in class_dir_names:
        print("processing{}".format(class_dir_name))

        class_phase_save_path = os.path.join(phase_data_save_path, class_dir_name)
        class_intensity_save_path = os.path.join(intensity_data_save_path, class_dir_name)
        os.makedirs(class_phase_save_path, exist_ok=True)
        os.makedirs(class_intensity_save_path, exist_ok=True)

        class_dir_path = os.path.join(data_root, class_dir_name)
        file_names = os.listdir(class_dir_path)
        for file_name in tqdm.tqdm(file_names):
            file_path = os.path.join(class_dir_path, file_name)
            SIGNAL = read_data(file_path)
            for st in st_factor:
                SIGNAL0_stretch = librosa.effects.time_stretch(SIGNAL[0,:],rate=st) 
                SIGNAL1_stretch = librosa.effects.time_stretch(SIGNAL[1,:],rate=st) 
                SIGNAL0_stretch_save_path = os.path.join(class_phase_save_path, file_name.replace(".mat", f"_stretch_{st}.npy"))
                SIGNAL1_stretch_save_path = os.path.join(class_intensity_save_path, file_name.replace(".mat", f"_stretch_{st}.npy"))
                np.save(SIGNAL0_stretch_save_path, SIGNAL0_stretch)
                np.save(SIGNAL1_stretch_save_path, SIGNAL1_stretch)
            for sh in sh_factor:
                SIGNAL0_roll = np.roll(SIGNAL[0,:], int(sh*len(SIGNAL[0,:]))) 
                SIGNAL1_roll = np.roll(SIGNAL[1,:], int(sh*len(SIGNAL[1,:]))) 
                SIGNAL0_roll_save_path = os.path.join(class_phase_save_path, file_name.replace(".mat", f"_roll_{sh}.npy"))
                SIGNAL1_roll_save_path = os.path.join(class_intensity_save_path, file_name.replace(".mat", f"_roll_{sh}.npy"))
                np.save(SIGNAL0_roll_save_path, SIGNAL0_roll)
                np.save(SIGNAL1_roll_save_path, SIGNAL1_roll)

data_augmentation(data_root="./data/DAS1K", save_path="./data/DAS1K_data_augmentation")
                

    
    

