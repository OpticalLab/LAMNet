import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


fs = 10000

data_root_dir = "./data/DAS1K_data_augmentation"
save_root_dir = "./data/DAS1K_data_augmentation_melspec"
os.makedirs(save_root_dir,exist_ok=True)
mode = ["phase", "intensity"]

for m in mode:
    data_dir = os.path.join(data_root_dir, m)
    save_mode_dir = os.path.join(save_root_dir, m)
    os.makedirs(save_mode_dir, exist_ok=True)
    for class_name in os.listdir(data_dir):
        save_class_name_dir = os.path.join(save_mode_dir, class_name)
        os.makedirs(save_class_name_dir, exist_ok=True)
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            data = np.load(file_path)
            plt.figure(figsize=(224/300, 224/300),dpi=300)
            melspec = librosa.feature.melspectrogram(y=data, sr=fs,
                                         n_fft=1024, hop_length=512, n_mels=40)
            logmelspec = librosa.power_to_db(melspec)
            librosa.display.specshow(logmelspec, sr=fs)
            plt.axis('off') 
            plt.title('')   
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1) 
            save_file_path = os.path.join(save_class_name_dir, file_name.replace(".npy", ".png"))
            plt.savefig(save_file_path)
            plt.close()
            # print(save_file_path)




