import csv
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np

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

#clean('data/labels/whi.txt', 'data/labels/whi.csv')



def save_spectrogram(audio_file, row):
    print("Row", row)
    y, sr = librosa.load(audio_file, sr=None)
    x1, x2 = row['x1'], row['x2']
    start = int(x1*sr)
    end = int(x2*sr)
    segment = y[start:end]
    D = librosa.stft(segment)
    spectrogram = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(x2-x1)
    plt.show()
    #plt.savefig("data/spectrogram_{row.name}.png")
    #plt.close()


def process_audio(csv_file, audio_file):
    df = pd.read_csv(csv_file)
    
    for idx, row in df.iterrows():
        save_spectrogram(audio_file, row)
        
        
        
process_audio('data/labels/whi.csv', 'data/raw_data/whi.wav')