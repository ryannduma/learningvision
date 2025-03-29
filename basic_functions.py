import cv2 as cv
import tkinter as tk
from tkinter import ttk, messagebox
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
        
        # Current processed image (starts as original)
        self.current_img = self.original_img.copy()
        
        # Track applied operations
        self.applied_operations = []
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create image display area
        self.img_label = ttk.Label(self.main_frame)
        self.img_label.pack(pady=10)
        
        # Create status label to show applied transformations
        self.status_label = ttk.Label(self.main_frame, text="No transformations applied")
        self.status_label.pack(pady=5)
        
        # Create buttons frame
        self.buttons_frame = ttk.Frame(self.main_frame)
        self.buttons_frame.pack(pady=10)
        
        # Create buttons for each operation
        self.create_buttons()
        
        # Create reset button
        self.reset_button = ttk.Button(self.main_frame, text="Reset to Original", command=self.reset_image)
        self.reset_button.pack(pady=5)
        
        # Display the original image (do this last after all UI elements are created)
        self.display_image(self.original_img, "Original Image")
        
    def create_buttons(self):
        operations = [
            ("Grayscale", self.apply_grayscale),
            ("Blur", self.apply_blur),
            ("Edges", self.detect_edges),
            ("Dilate", self.apply_dilate),
            ("Erode", self.apply_erode),
            ("Resize", self.apply_resize),
            ("Crop", self.apply_crop)
        ]
        
        # Create a button for each operation
        for text, command in operations:
            btn = ttk.Button(self.buttons_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=5)
    
    def show_warning(self, message):
        """Show a warning dialog but allow the user to proceed."""
        return messagebox.askokcancel("Operation Warning", message)
    
    def check_operation_conflicts(self, operation):
        """Check if the new operation conflicts with previously applied operations."""
        conflicting_pairs = {
            "Dilate": ["Erode"],
            "Erode": ["Dilate"],
            "Blur": ["Edges", "Dilate", "Erode"],
            "Edges": ["Blur"]
        }
        
        if operation in conflicting_pairs:
            for conflict in conflicting_pairs[operation]:
                if conflict in self.applied_operations:
                    return f"Applying {operation} after {conflict} might not produce optimal results. Proceed anyway?"
        return None
    
    def display_image(self, image, title: str):
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
        
        # Update status label
        if self.applied_operations:
            self.status_label.config(text=f"Applied: {' → '.join(self.applied_operations)}")
        else:
            self.status_label.config(text="Original image")
    
    def reset_image(self):
        self.current_img = self.original_img.copy()
        self.applied_operations = []
        self.display_image(self.current_img, "Original Image")
    
    def apply_transformation(self, operation_name, result):
        """Apply a transformation result and update tracking."""
        # Update the current image
        self.current_img = result
        
        # Add to applied operations list
        self.applied_operations.append(operation_name)
        
        # Update display
        self.display_image(self.current_img, f"After {operation_name}")
    
    def apply_grayscale(self):
        # Check for conflicts
        warning = self.check_operation_conflicts("Grayscale")
        if warning and not self.show_warning(warning):
            return
            
        result = apply_grayscale(self.current_img)
        self.apply_transformation("Grayscale", result)
    
    def apply_blur(self):
        # Check for conflicts
        warning = self.check_operation_conflicts("Blur")
        if warning and not self.show_warning(warning):
            return
            
        result = apply_blur(self.current_img)
        self.apply_transformation("Blur", result)
    
    def detect_edges(self):
        # Check for conflicts
        warning = self.check_operation_conflicts("Edges")
        if warning and not self.show_warning(warning):
            return
            
        # For edge detection, we typically want to blur first
        if "Blur" not in self.applied_operations:
            blurred = apply_blur(self.current_img)
            result = detect_edges(blurred)
            self.apply_transformation("Edges", result)
        else:
            result = detect_edges(self.current_img)
            self.apply_transformation("Edges", result)
    
    def apply_dilate(self):
        # Check for conflicts
        warning = self.check_operation_conflicts("Dilate")
        if warning and not self.show_warning(warning):
            return
            
        # Typically dilate works on edge images
        if "Edges" not in self.applied_operations:
            if self.show_warning("Dilation works best on edge images. Apply edge detection first?"):
                blurred = apply_blur(self.current_img)
                edges = detect_edges(blurred)
                result = dilate_image(edges)
                self.applied_operations.append("Edges")  # Add intermediate step
                self.apply_transformation("Dilate", result)
            else:
                result = dilate_image(self.current_img)
                self.apply_transformation("Dilate", result)
        else:
            result = dilate_image(self.current_img)
            self.apply_transformation("Dilate", result)
    
    def apply_erode(self):
        # Check for conflicts
        warning = self.check_operation_conflicts("Erode")
        if warning and not self.show_warning(warning):
            return
            
        result = erode_image(self.current_img)
        self.apply_transformation("Erode", result)
    
    def apply_resize(self):
        # Resizing has little conflict with other operations
        result = resize_image(self.current_img)
        self.apply_transformation("Resize", result)
    
    def apply_crop(self):
        # Cropping has little conflict with other operations
        result = crop_image(self.current_img)
        self.apply_transformation("Crop", result)

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