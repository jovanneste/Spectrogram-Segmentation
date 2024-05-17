import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from PIL import Image, ImageDraw

# Function to load and preprocess data
def load_data(data_dir):
    images = []
    labels = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(data_dir, filename)
            image = Image.open(image_path).resize((334, 217))  # Resize image
            images.append(np.array(image))

            label_path = os.path.join(data_dir, filename[:-4] + ".txt")
            with open(label_path, 'r') as file:
                bbox_data = file.readline().strip().split()
                _, x, y, w, h = map(float, bbox_data)
                labels.append([x,y,w,h])

    return np.array(images), np.array(labels)


def plot_image_with_bbox(image, bbox):
    # Plot image
    plt.imshow(image)
    
    # Unpack bbox coordinates
    x_center, y_center, width, height = bbox
    
    # Convert normalized coordinates to pixel coordinates
    image_height, image_width, _ = image.shape
    xmin = int((x_center - width / 2) * image_width)
    xmax = int((x_center + width / 2) * image_width)
    ymin = int((y_center - height / 2) * image_height)
    ymax = int((y_center + height / 2) * image_height)
    
    # Draw bounding box
    plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                      edgecolor='r', facecolor='none', linewidth=2))
    
    # Show plot
    plt.show()

# Example usage:
# plot_image_with_bbox(image, label)

def iou(y_true, y_pred):
    y_true=y_true.numpy()
    y_pred = y_pred.numpy()
    intersect_x1 = np.maximum(y_true[0] - y_true[2] / 2, y_pred[0] - y_pred[2] / 2)
    intersect_y1 = np.maximum(y_true[1] - y_true[3] / 2, y_pred[1] - y_pred[3] / 2)
    intersect_x2 = np.minimum(y_true[0] + y_true[2] / 2, y_pred[0] + y_pred[2] / 2)
    intersect_y2 = np.minimum(y_true[1] + y_true[3] / 2, y_pred[1] + y_pred[3] / 2)

    intersect_area = np.maximum(0.0, intersect_x2 - intersect_x1) * np.maximum(0.0, intersect_y2 - intersect_y1)
    true_area = y_true[2] * y_true[3]
    pred_area = y_pred[2] * y_pred[3]

    union_area = true_area + pred_area - intersect_area
    iou = intersect_area / (union_area + 1e-10)  # Adding a small epsilon to avoid division by zero
    print(iou)
    return iou

# Define the CNN model
def create_model():
    inputs = tf.keras.Input(shape=(217, 334, 4))
    x = layers.Conv2D(16, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)  # Output layer for bounding box coordinates
    print('Outputs - ', outputs)

    model = Model(inputs, outputs)
    return model

# Load data
train_dir = "data/spectrograms/train"
val_dir = "data/spectrograms/val"
x_train, y_train = load_data(train_dir)
x_val, y_val = load_data(val_dir)
plot_image_with_bbox(x_train[3], y_train[3])

# Create and compile the model
model = create_model()
model.compile(optimizer='adam', loss='mse', metrics=[iou], run_eagerly=True)

# Train the model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100)  # Adjust epochs as needed

# Once the model is trained, you can use it for prediction
# predicted_boxes = model.predict(x_val)

# Visualize the predicted bounding boxes
# for i in range(len(x_val)):
#     image_with_bbox = visualize_bbox(x_val[i], predicted_boxes[i])
#     plt.imshow(image_with_bbox)
#     plt.show()
