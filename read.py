import cv2 as cv
import os
import numpy as np
from typing import Optional, Union, str, bool

# Print working directory for debugging
print(f"Current working directory: {os.getcwd()}")

def load_and_save_image(img_path: str, output_path: str = 'output.jpg') -> bool:
    """
    Load an image from the specified path and save it to the output path.
    
    Args:
        img_path: Path to the input image
        output_path: Path where the image will be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Attempting to load image from: {os.path.abspath(img_path)}")
    
    # Check if file exists
    if not os.path.exists(img_path):
        print(f"ERROR: The file does not exist at {img_path}")
        return False
    
    # Read the image
    img = cv.imread(img_path)
    
    # Check if image was loaded successfully
    if img is None:
        print(f"ERROR: Image could not be loaded from {img_path}")
        return False
    
    # Save the image
    print(f"Image loaded successfully! Saving to {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save the image
    cv.imwrite(output_path, img)
    print(f"Image saved to {output_path}")
    return True

# Try to load and save the image
img_path = './Resources/Photos/cats.jpg'
load_and_save_image(img_path)

def process_video(input_path: str, output_path: str, frame_output_dir: Optional[str] = None) -> bool:
    """
    Process a video and save the result. Optionally save individual frames.
    
    Args:
        input_path: Path to the input video
        output_path: Path where the processed video will be saved
        frame_output_dir: Directory where individual frames will be saved (if provided)
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(input_path):
        print(f"ERROR: Input file doesn't exist at {input_path}")
        return False
    
    # Open the video
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Couldn't open video: {input_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {width}x{height}, {fps} fps, {frame_count} frames")
    
    # Create output directories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if frame_output_dir:
        os.makedirs(frame_output_dir, exist_ok=True)
    
    # Create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # ----------------------------------
        # Your frame processing code here
        # For example:
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # blurred = cv.GaussianBlur(frame, (5, 5), 0)
        # processed_frame = blurred  # or any other processing
        # ----------------------------------
        
        # For now, just use the original frame
        processed_frame = frame
        
        # Write to video
        out.write(processed_frame)
        
        # Save individual frame if requested
        if frame_output_dir:
            frame_path = os.path.join(frame_output_dir, f"frame_{frame_number:04d}.jpg")
            cv.imwrite(frame_path, processed_frame)
        
        frame_number += 1
        if frame_number % 100 == 0:
            print(f"Processed {frame_number}/{frame_count} frames")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Video processing complete. Output saved to: {output_path}")
    return True

def process_image(input_path: str, output_path: str) -> bool:
    """
    Process an image and save the result.
    
    Args:
        input_path: Path to the input image
        output_path: Path where the processed image will be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"ERROR: Input file doesn't exist at {input_path}")
        return False
    
    # Read the image
    img = cv.imread(input_path)
    if img is None:
        print(f"ERROR: Couldn't load image from {input_path}")
        return False
    
    print(f"Successfully loaded image: {input_path}")
    
    # ----------------------------------
    # Your image processing code here
    # For example:
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # blurred = cv.GaussianBlur(img, (5, 5), 0)
    # edges = cv.Canny(gray, 100, 200)
    # ----------------------------------
    
    # For now, just use the original image
    result = img
    
    # Save processed image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv.imwrite(output_path, result)
    print(f"Saved processed image to: {output_path}")
    return True

# Example usage for video processing
def process_sample_video():
    input_video = "./Resources/Videos/sample.mp4"
    output_video = "./output/processed_video.mp4"
    frames_dir = "./output/frames"  # Set to None if you don't want to save frames
    
    process_video(input_video, output_video, frames_dir)

# Example usage for image processing
def process_sample_image():
    input_file = "./Resources/Photos/cats.jpg"
    output_file = "./output/processed_cats.jpg"
    
    process_image(input_file, output_file)

# Run the examples
if __name__ == "__main__":
    # Process and save an image
    process_sample_image()
    
    # Process and save a video
    process_sample_video()
    
    # Clean up
    cv.destroyAllWindows()
