import cv2

def plot_bbox_on_image(image_path, bbox_text_path, output_image_path):
    # Load the PNG image
    image = cv2.imread(image_path)

    # Read bounding box coordinates from the text file
    with open(bbox_text_path, 'r') as file:
        bbox_data = file.readline().strip().split()
    
    # Parse bounding box coordinates
    _, x, y, w, h = map(float, bbox_data)
    

    
    # Draw a point on the image at the specified pixel coordinates
    # Here we are drawing a red point with a thickness of 2 pixels
    color = (0, 0, 255)  # Red color in BGR format
    thickness = 2
    
    
    # Convert normalized coordinates to pixel coordinates
    image_height, image_width = image.shape[:2]
    print(image_height, image_width)
    print(x,y,w,h)
    x_pixel = int(x * image_width)
    y_pixel = int(y * image_height)
    w_pixel = int(w * image_width)
    h_pixel = int(h * image_height)
    
    
    # Draw bounding box on the image
    cv2.rectangle(image, (x_pixel - w_pixel//2, y_pixel - h_pixel//2),
                  (x_pixel + w_pixel//2, y_pixel + h_pixel//2), (0, 255, 0), 2)
    cv2.circle(image, (x_pixel, y_pixel), radius=2, color=color, thickness=thickness)

    

    cv2.imwrite(output_image_path, image)

i=60
plot_bbox_on_image(f'data/spectrograms/test/{i}.png', f'data/spectrograms/test/{i}.txt', 'annotated_image.png')
