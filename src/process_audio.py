import csv
import math
import pandas as pd
from scipy.io import wavfile
from scipy import signal
import librosa
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
import sys
import random
import shutil
from tqdm import tqdm 

def display(f):
    sample_rate, samples = wavfile.read(f)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    
    plt.figure(figsize=(0, 5))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(times, frequencies, np.log(spectrogram))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Full Spectrogram')
    
    plt.subplot(1, 2, 2)
    
    n = 5
    # Find the index of the time point closest to the nth second
    index_ns = np.argmin(np.abs(times - n))
    plt.pcolormesh(times[:index_ns+1], frequencies, np.log(spectrogram[:, :index_ns+1]))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'Spectrogram of {n}th Second')
    
    plt.tight_layout()
    plt.show()


def clean(l):
    labels_path = '/mnt/Data1/Acoustics/labels/REPMUS_2021/2081/'+l
    output_file = '/mnt/Data2/jvanneste/Spectrogram-Segmentation/data/labels/'+l

    with open(labels_path, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = infile.readlines()
        writer = csv.writer(outfile)

        # Write CSV header
        writer.writerow(['class', 'x1', 'x2', 'y1', 'y2'])

        # Process each pair of lines
        for i in range(0, len(reader), 2):
            x_line = reader[i].strip()
            try:
                y_line = reader[i + 1].strip()
                x_parts = x_line.split('\t')
                x1, x2, class_label = float(x_parts[0]), float(x_parts[1]), x_parts[2]
                y_parts = y_line.split('\t')[1:]
           
                y1, y2 = float(y_parts[0]), float(y_parts[1])
                
                
                if 'fv' in class_label.lower():
                    writer.writerow([class_label.lower(), x1, x2, y1, y2])
            
                
                if 'whistle' in class_label.lower():
                    writer.writerow(['whistle', x1, x2, y1, y2])
                    
            except:
                pass

def coords_2_yolo(bbox):
    center_x = (bbox[0]+bbox[1])/2
    center_y = (bbox[2]+bbox[3])/2
    w = bbox[1]-bbox[0]
    h = bbox[3]-bbox[2]
    return round(center_x,2), round(center_y,2), round(w,2), round(h,2)

def normalise_coords(row,x1,sr,S_dB):
    x2 = row['x2']-x1
    if x2>1:
        x2=1   
    y1, y2 = row['y1'], row['y2']
    
    mel_frequencies = librosa.mel_frequencies(n_mels=60, fmin=0.0, fmax=sr/2)    
    y1_bin_index = np.argmin(np.abs(mel_frequencies - y1))
    y2_bin_index = np.argmin(np.abs(mel_frequencies - y2))
    
    return (round(row['x1']-x1,2),round(x2,2), round(y1_bin_index/60, 2), round(y2_bin_index/60, 2))


def save_spectrogram_whistle(audio_file, row, idx, df): 
    print(row)
    directory = '../data/spectrograms/'
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    if not txt_files:
        save_as = 1
    else:
        save_as = max([int(filename.split('.')[0]) for filename in txt_files])+1

    y, sr = librosa.load(audio_file, sr=52137)
    x1 = row['x1']
    x2 = row['x2']
    start = int(x1*sr)
    end = int(x2*sr)

    segment = y[start:end]
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=60, n_fft=2048, hop_length=1024)
    Xdb = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='mel')   
    #$plt.colorbar(format='%+2.0f dB')
    plt.axis('off')
    x,y,w,h = coords_2_yolo(normalise_coords(row, x1, sr, Xdb))

    
    plt.savefig(f"../data/spectrograms/{save_as}.png", bbox_inches = 'tight', pad_inches=0)
    
    vessel = False
    if 'fv' in df.iloc[0]['class'] and x1 > df.iloc[0]['x1'] and x2 < df.iloc[0]['x2']:
        #this spectrogram also has a vessel
        vessel = True
        mel_frequencies = librosa.mel_frequencies(n_mels=60, fmin=0.0, fmax=sr/2) 
        y1_bin_index = round(np.argmin(np.abs(mel_frequencies - df.iloc[0]['y1']))/60, 2)
        y2_bin_index = round(np.argmin(np.abs(mel_frequencies - df.iloc[0]['y2']))/60, 2)
        y_center_vessel = (y1_bin_index+y2_bin_index)/2
        h_vessel = y2_bin_index-y1_bin_index
        
        vessel_class = df.iloc[0]['class']
        if vessel_class=='lfv':
            vessel_class=1
        elif vessel_class=='mfv':
            vessel_class=2
        elif vessel_class=='hfv':
            vessel_class=3

      
    with open(f'../data/spectrograms/{save_as}.txt', 'w') as f:
        f.write(f"0 {x} {1-y} {w} {h}")
        if vessel:
            f.write("\n")
            f.write(f"{vessel_class} {0.5} {1-y_center_vessel} {1} {h_vessel}")
    
def split_data():
    d = '../data/spectrograms/'
    files = [f for f in os.listdir(d) if f.endswith('.png')]
    random.shuffle(files)
    train_len = int(len(files)*0.85)
    test_len = int(len(files)*0.05)

    train_split, test_split, val_split = files[:train_len], files[train_len:train_len+test_len], files[train_len+test_len:]
    
    for f in train_split:
        txt_file = os.path.splitext(f)[0] + '.txt'
        shutil.move(d+f, d+'train/'+f)
        shutil.move(d+txt_file, d+'train/'+txt_file)
  
    for f in test_split:
        txt_file = os.path.splitext(f)[0] + '.txt'
        shutil.move(d+f, d+'test/'+f)
        shutil.move(d+txt_file, d+'test/'+txt_file)
        
    for f in val_split:
        txt_file = os.path.splitext(f)[0] + '.txt'
        shutil.move(d+f, d+'val/'+f)
        shutil.move(d+txt_file, d+'val/'+txt_file)
    
    
    
def process_audio(f):
    print("NEW FILE -- ", f)
    audio_file = ('/mnt/Data1/Acoustics/raw_data/REPMUS_2021/PT07_1208492081/'+f)[:-3]+'wav'
    csv_file = '/mnt/Data2/jvanneste/Spectrogram-Segmentation/data/labels/'+f
    df = pd.read_csv(csv_file)
    for idx, row in tqdm(df.iterrows()):
        if idx!=0:
            save_spectrogram_whistle(audio_file, row, idx, df)



if __name__=='__main__':
   
    file_list = [
   # "5927.230919235004.txt"
   # "5927.230909092958.txt"
   "6335.230909124958.txt",
    "6335.230909091958.txt"  
   ]

    fl = [
    "1208492081.210910111003.txt", "1208492081.210914095005.txt",
    "1208492081.210910115003.txt", "1208492081.210914151005.txt",
    "1208492081.210910123003.txt", "1208492081.210915103006.txt",
    "1208492081.210910131003.txt", "1208492081.210915111006.txt",
    "1208492081.210910151003.txt", "1208492081.210915123006.txt",
    "1208492081.210910155003.txt", "1208492081.210915131006.txt",
    "1208492081.210910211003.txt", "1208492081.210915135006.txt",
    "1208492081.210910215003.txt", "1208492081.210915143006.txt",
    "1208492081.210910223003.txt", "1208492081.210915151006.txt",
    "1208492081.210910231003.txt", "1208492081.210915155006.txt",
    "1208492081.210910235003.txt", "1208492081.210915163006.txt",
    "1208492081.210911003003.txt", "1208492081.210915171006.txt",
    "1208492081.210915175006.txt",
    "1208492081.210911011003.txt", "1208492081.210915183006.txt",
    "1208492081.210911015003.txt", "1208492081.210915191006.txt",
    "1208492081.210911023003.txt", "1208492081.210915195006.txt",
    "1208492081.210911031003.txt", "1208492081.210915203006.txt",
    "1208492081.210911035003.txt", "1208492081.210915211006.txt",
    "1208492081.210911043003.txt", "1208492081.210915215006.txt",
    "1208492081.210911051003.txt", "1208492081.210915223006.txt",
    "1208492081.210911055003.txt", "1208492081.210915231006.txt",
    "1208492081.210911063003.txt", "1208492081.210915235006.txt",
    "1208492081.210911071003.txt", "1208492081.210916003006.txt",
    "1208492081.210911075003.txt", "1208492081.210916011006.txt",
    "1208492081.210916015006.txt",
    "1208492081.210911083003.txt", "1208492081.210916023006.txt",
    "1208492081.210911091003.txt", "1208492081.210916151006.txt",
    "1208492081.210911095003.txt", "1208492081.210916155006.txt",
    "1208492081.210911103003.txt", "1208492081.210916163006.txt",
    "1208492081.210911111003.txt", "1208492081.210916171006.txt",
    "1208492081.210911115003.txt", "1208492081.210916175006.txt",
    "1208492081.210911123003.txt", "1208492081.210916183006.txt",
    "1208492081.210911131003.txt", "1208492081.210916191006.txt",
    "1208492081.210911135003.txt", "1208492081.210916195006.txt",
    "1208492081.210911143003.txt", "1208492081.210916235007.txt",
    "1208492081.210911151003.txt",
    "1208492081.210911155003.txt", "1208492081.210917003007.txt",
    "1208492081.210911163003.txt", "1208492081.210917011007.txt",
    "1208492081.210911171003.txt", "1208492081.210917015007.txt",
    "1208492081.210911175003.txt", "1208492081.210917023007.txt",
    "1208492081.210911183003.txt", "1208492081.210917035007.txt",
    "1208492081.210911191003.txt", "1208492081.210917051007.txt",
    "1208492081.210912071004.txt", "1208492081.210917131007.txt"
]


    [clean(l) for l in fl]
    [process_audio(f) for f in fl]
   # split_data()

    