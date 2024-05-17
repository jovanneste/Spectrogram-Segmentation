import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
import shutil

def random_augmentation(input_dir, output_dir, num_augmentations=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    save_as = max([int(filename.split('.')[0]) for filename in txt_files])+1
    
    augmentation = iaa.Sequential([
        iaa.GammaContrast(gamma=(0.5, 2.0)),  # Random gamma correction
        iaa.MultiplyBrightness((0.5, 1.5)),  # Random brightness adjustment
        iaa.AddToHueAndSaturation((-20, 20)),  # Random color shifts
    ], random_order=True)  # Apply augmentations in random order

    # Loop through images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png')):
            file_name = os.path.splitext(filename)[0]
            image = cv2.imread(os.path.join(input_dir, filename))

            for i in range(num_augmentations):
                # Apply augmentation
                augmented_image = augmentation.augment_image(image)

                # Write augmented image to output directory
                output_path = os.path.join(output_dir, f"{save_as}.png")
                shutil.copyfile(input_dir+file_name+'.txt', input_dir+str(save_as)+'.txt')
                cv2.imwrite(output_path, augmented_image)
                save_as += 1
                print(f"Augmented {filename} (iteration {i+1}) saved successfully as {save_as}.")

# Example usage
input_directory = "data/spectrograms/train/"
output_directory = "data/spectrograms/train/"
num_augmentations = 10

random_augmentation(input_directory, output_directory, num_augmentations)
