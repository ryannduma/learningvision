import cv2 as cv
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from typing import Tuple, Any, Optional, Union, List

# Define a type alias for our image type to avoid repeating the complex type
ImageType = np.ndarray[Tuple[int, ...], np.dtype[np.uint8]]

def apply_grayscale(image: ImageType) -> ImageType:
    """
    Convert an image to grayscale.
    
    Args:
        image: The input image in BGR format
        
    Returns:
        The grayscale version of the image
    """
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def apply_blur(image: ImageType, kernel_size: Tuple[int, int]=(7,7)) -> ImageType:
    """
    Apply Gaussian blur to an image.
    
    Args:
        image: The input image
        kernel_size: Tuple representing blur kernel size (default: (7,7))
        
    Returns:
        The blurred image
    """
    return cv.GaussianBlur(image, kernel_size, 0)

def detect_edges(image: ImageType, threshold1: int=125, threshold2: int=175) -> ImageType:
    """
    Detect edges in an image using Canny edge detector.
    
    Args:
        image: The input image
        threshold1: First threshold for the hysteresis procedure (default: 125)
        threshold2: Second threshold for the hysteresis procedure (default: 175)
        
    Returns:
        Image with detected edges
    """
    return cv.Canny(image, threshold1, threshold2)

def dilate_image(image: ImageType, kernel_size: Optional[Tuple[int, int]]=None, iterations: int=3) -> ImageType:
    """
    Dilate an image to enhance edges.
    
    Args:
        image: The input image (typically an edge image)
        kernel_size: Tuple representing dilation kernel size (default: None)
        iterations: Number of times dilation is applied (default: 3)
        
    Returns:
        The dilated image
    """
    kernel = None if kernel_size is None else np.ones(kernel_size, dtype=np.uint8)
    return cv.dilate(image, kernel, iterations=iterations)

def erode_image(image: ImageType, kernel_size: Optional[Tuple[int, int]]=None, iterations: int=3) -> ImageType:
    """
    Erode an image (opposite of dilation).
    
    Args:
        image: The input image
        kernel_size: Tuple representing erosion kernel size (default: None)
        iterations: Number of times erosion is applied (default: 3)
        
    Returns:
        The eroded image
    """
    kernel = None if kernel_size is None else np.ones(kernel_size, dtype=np.uint8)
    return cv.erode(image, kernel, iterations=iterations)

def resize_image(image: ImageType, dimensions: Tuple[int, int]=(500,500)) -> ImageType:
    """
    Resize an image to specified dimensions.
    
    Args:
        image: The input image
        dimensions: Tuple of (width, height) for the output image (default: (500,500))
        
    Returns:
        The resized image
    """
    return cv.resize(image, dimensions, interpolation=cv.INTER_CUBIC)

def crop_image(image: ImageType, y_range: Tuple[int, int]=(50,200), x_range: Tuple[int, int]=(200,400)) -> ImageType:
    """
    Crop a region from an image.
    
    Args:
        image: The input image
        y_range: Tuple of (start, end) for y-coordinates (default: (50,200))
        x_range: Tuple of (start, end) for x-coordinates (default: (200,400))
        
    Returns:
        The cropped image
    """
    return image[y_range[0]:y_range[1], x_range[0]:x_range[1]]

class ImageProcessorApp:
    def __init__(self, root: tk.Tk, image_path: str):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("800x600")
        
        # Load the image
        self.original_img = cv.imread(image_path)
        if self.original_img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create image display area
        self.img_label = ttk.Label(self.main_frame)
        self.img_label.pack(pady=10)
        
        # Display the original image
        self.display_image(self.original_img, "Original Image")
        
        # Create buttons frame
        self.buttons_frame = ttk.Frame(self.main_frame)
        self.buttons_frame.pack(pady=10)
        
        # Create buttons for each operation
        self.create_buttons()
        
        # Create reset button
        self.reset_button = ttk.Button(self.main_frame, text="Reset to Original", command=self.reset_image)
        self.reset_button.pack(pady=5)
        
    def create_buttons(self):
        operations = [
            ("Grayscale", self.apply_grayscale),
            ("Blur", self.apply_blur),
            ("Edges", self.detect_edges),
            ("Dilate", self.apply_dilate),
            ("Erode", self.apply_erode),
            ("Resize", self.apply_resize),
            ("Crop", self.apply_crop),
            ("All Effects", self.apply_all)
        ]
        
        # Create a button for each operation
        for text, command in operations:
            btn = ttk.Button(self.buttons_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=5)
    
    def display_image(self, image: np.ndarray, title: str):
        # Convert from BGR to RGB for PIL
        if len(image.shape) == 3:  # Color image
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        else:  # Grayscale image
            image_rgb = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        
        # Resize for display if too large
        h, w = image.shape[:2]
        max_size = 500
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image_rgb = cv.resize(image_rgb, (new_w, new_h))
        
        # Convert to PIL Image and then to PhotoImage
        img_pil = Image.fromarray(image_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Update the image in the label
        self.img_label.configure(image=img_tk)
        self.img_label.image = img_tk  # Keep a reference
        
        # Update window title
        self.root.title(f"Image Processor - {title}")
    
    def reset_image(self):
        self.display_image(self.original_img, "Original Image")
    
    def apply_grayscale(self):
        result = apply_grayscale(self.original_img)
        self.display_image(result, "Grayscale")
    
    def apply_blur(self):
        result = apply_blur(self.original_img)
        self.display_image(result, "Blurred")
    
    def detect_edges(self):
        blurred = apply_blur(self.original_img)
        result = detect_edges(blurred)
        self.display_image(result, "Edges")
    
    def apply_dilate(self):
        blurred = apply_blur(self.original_img)
        edges = detect_edges(blurred)
        result = dilate_image(edges)
        self.display_image(result, "Dilated")
    
    def apply_erode(self):
        blurred = apply_blur(self.original_img)
        edges = detect_edges(blurred)
        dilated = dilate_image(edges)
        result = erode_image(dilated)
        self.display_image(result, "Eroded")
    
    def apply_resize(self):
        result = resize_image(self.original_img)
        self.display_image(result, "Resized")
    
    def apply_crop(self):
        result = crop_image(self.original_img)
        self.display_image(result, "Cropped")
    
    def apply_all(self):
        # Process the image with all transformations one by one
        # Each will update the display
        self.apply_grayscale()
        self.root.after(1000, self.apply_blur)
        self.root.after(2000, self.detect_edges)
        self.root.after(3000, self.apply_dilate)
        self.root.after(4000, self.apply_erode)
        self.root.after(5000, self.apply_resize)
        self.root.after(6000, self.apply_crop)
        self.root.after(7000, self.reset_image)

def process_image():
    """
    Main function to process an image using a GUI interface.
    """
    # Set up the image path
    img_path = '/Users/rynduma/learningvision/learningvision/Resources/Photos/park.jpg'
    
    # Create the Tkinter root window
    root = tk.Tk()
    
    # Create the application
    app = ImageProcessorApp(root, img_path)
    
    # Start the Tkinter event loop
    root.mainloop()

# Execute the main function if this script is run directly
if __name__ == "__main__":
    process_image()