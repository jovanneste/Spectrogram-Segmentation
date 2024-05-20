import cv2
import numpy as np
import glob
import os


def denoise(path):
    spectrogram = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    height, width = spectrogram.shape
    top_percentage = 0.45
    top_region_height = int(height * top_percentage)
    
    top_region = spectrogram[:top_region_height]
    bottom_region = spectrogram[top_region_height:]

    alpha = 1.7  
    beta = 0    
    enhanced_top_region = cv2.convertScaleAbs(top_region, alpha=alpha, beta=beta)
    
    brightness_scale = 0.5  
    darkened_bottom_region = cv2.convertScaleAbs(bottom_region, alpha=brightness_scale, beta=0)
    
    enhanced_spectrogram = np.vstack((enhanced_top_region, darkened_bottom_region))
    cv2.imwrite(path, enhanced_spectrogram)


if __name__=='__main__':
    for f in glob.glob(os.path.join('data/spectrograms/val', '*')):
        if f.endswith('.png'):
            denoise(f)