import csv
import math
import pandas as pd
from scipy.io import wavfile
from scipy import signal
import librosa
import matplotlib.pyplot as plt
import numpy as np
import sys

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


def clean(input_file, output_file):
    # Open input and output files
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = infile.readlines()
        writer = csv.writer(outfile)

        # Write CSV header
        writer.writerow(['class', 'x1', 'x2', 'y1', 'y2'])

        # Process each pair of lines
        for i in range(0, len(reader), 2):
            x_line = reader[i].strip()
            y_line = reader[i + 1].strip()
            
            # Extract x coordinates and class label
            x_parts = x_line.split('\t')
            x1, x2, class_label = float(x_parts[0]), float(x_parts[1]), x_parts[2]
            print(x1, x2, class_label)
            
            # Extract y coordinates
            y_parts = y_line.split('\t')[1:]
            print(y_parts)
            y1, y2 = float(y_parts[0]), float(y_parts[1])
            
            if class_label == 'whistle':  # Process only lines with 'whistle' class
                writer.writerow(['whistle', x1, x2, y1, y2])

def coords_2_yolo(bbox):
    center_x = (bbox[0]+bbox[1])/2
    center_y = (bbox[2]+bbox[3])/2
    w = bbox[1]-bbox[0]
    h = bbox[3]-bbox[2]
    return center_x, center_y, w, h

def normalise_coords(row,x1,sr,S_dB):
    height = S_dB.shape[0]
    x2 = row['x2']-x1
    if x2>1:
        x2=1   
    y1, y2 = row['y1'], row['y2']
    
    mel_frequencies = librosa.core.mel_frequencies(n_mels=60, fmin=0.0, fmax=sr/2)

    # Find the index of the frequency bin corresponding to y1
    for i, freq in enumerate(mel_frequencies):
        if freq > y1:
            y1_bin_index = i
            break
    for i, freq in enumerate(mel_frequencies):
        if freq > y2:
            y2_bin_index = i
            break

    return (round(row['x1']-x1,2),round(x2,2), round(y1_bin_index/height, 2), round(y2_bin_index/height, 2))


def save_spectrogram(audio_file, row, labels, idx): 
    print(row)
    y, sr = librosa.load(audio_file, sr=None)
    x1 = math.floor(row['x1'])
    x2=x1+1
   
    start = int(x1*sr)
    end = int(x2*sr)

    segment = y[start:end]
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=60)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')   
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    bbox = normalise_coords(row, x1, sr, S_dB)
    print(S_dB.shape[1])
    print(bbox)
    sys.exit()
    
    plt.savefig(f"data/spectrograms/{idx}.png")
    plt.close()


def process_audio(csv_file, audio_file):
    df = pd.read_csv(csv_file)
    labels = []
    for idx, row in df.iterrows():
        print("ID", idx)
        save_spectrogram(audio_file, row, labels, idx)
    with open('data/spectrograms/labels.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(labels)
   


   
#clean('data/labels/whi2.txt', 'data/labels/whi2.csv')
process_audio('data/labels/whi.csv', 'data/raw_data/whi.wav')