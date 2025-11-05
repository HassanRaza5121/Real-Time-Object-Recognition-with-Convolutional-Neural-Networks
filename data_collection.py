import cv2
import os

def collect_images(object_name, num_images=35):
    """
    Captures a specified number of images for a given object category.

    Args:
        object_name (str): The name of the object category (e.g., 'pen', 'mug').
        num_images (int): The number of images to capture.
    """
    
    # Create a directory to save images if it doesn't exist
    directory = f'./dataset/{object_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    # Initialize the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"\nPress 'c' to capture an image for '{object_name}'. Press 'q' to quit.")
    
    image_count = 0
    while image_count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the live video feed
        cv2.putText(frame, f"Capturing for: {object_name} ({image_count}/{num_images})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Live Camera Feed', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Capture image on 'c' key press
        if key == ord('c'):
            filename = os.path.join(directory, f'image_{image_count}.jpg')
            cv2.imwrite(filename, frame)
            print(f"Captured {filename}")
            image_count += 1
        
        # Quit on 'q' key press
        elif key == ord('q'):
            print("Quitting image capture.")
            break
    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Define your four distinct object categories
    object_categories = ['object1', 'object2', 'object3', 'object4']
    
    # Loop through each object category to collect images
    for obj in object_categories:
        collect_images(obj, num_images=35) # Collect 35 images per object
    
    print("\nData collection complete!")