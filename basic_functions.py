import cv2 as cv
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
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

def adjust_brightness_contrast(image: ImageType, alpha: float=1.5, beta: int=0) -> ImageType:
    """
    Adjust brightness and contrast of an image.
    
    Formula: new_image = alpha * original_image + beta
    
    Args:
        image: The input image
        alpha: Contrast control (1.0 means no change) (default: 1.5)
        beta: Brightness control (0 means no change) (default: 0)
        
    Returns:
        Adjusted image
    """
    return cv.convertScaleAbs(image, alpha=alpha, beta=beta)

def apply_threshold(image: ImageType, threshold: int=127, max_val: int=255, 
                   threshold_type: int=cv.THRESH_BINARY) -> ImageType:
    """
    Apply thresholding to an image.
    
    Args:
        image: The input image (should be grayscale)
        threshold: Threshold value (default: 127)
        max_val: Maximum value to use with THRESH_BINARY (default: 255)
        threshold_type: Type of thresholding (default: cv.THRESH_BINARY)
        
    Returns:
        Thresholded image
    """
    # Convert to grayscale if image is color
    if len(image.shape) > 2:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    
    _, thresholded = cv.threshold(gray, threshold, max_val, threshold_type)
    return thresholded

def rotate_image(image: ImageType, angle: float=45) -> ImageType:
    """
    Rotate an image by a specified angle.
    
    Args:
        image: The input image
        angle: Angle of rotation in degrees (default: 45)
        
    Returns:
        Rotated image
    """
    # Get the image size
    height, width = image.shape[:2]
    
    # Calculate the rotation matrix
    center = (width // 2, height // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation
    rotated = cv.warpAffine(image, rotation_matrix, (width, height))
    return rotated

def flip_image(image: ImageType, flip_code: int=1) -> ImageType:
    """
    Flip an image.
    
    Args:
        image: The input image
        flip_code: Flip direction (0: flip around x-axis, 1: flip around y-axis, -1: both) (default: 1)
        
    Returns:
        Flipped image
    """
    return cv.flip(image, flip_code)

def sharpen_image(image: ImageType) -> ImageType:
    """
    Apply sharpening filter to an image.
    
    Args:
        image: The input image
        
    Returns:
        Sharpened image
    """
    # Create sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    # Apply kernel to the image
    sharpened = cv.filter2D(image, -1, kernel)
    return sharpened

def extract_color_channel(image: ImageType, channel: int=0) -> ImageType:
    """
    Extract a specific color channel from an image (BGR format).
    
    Args:
        image: The input image (must be color)
        channel: Channel to extract (0: blue, 1: green, 2: red) (default: 0)
        
    Returns:
        Image with only the selected channel preserved
    """
    # Create a zero array of same shape as input image
    zero_channels = np.zeros_like(image)
    
    # Create a copy of the image
    output = zero_channels.copy()
    
    # Set the specified channel to the original value
    output[:, :, channel] = image[:, :, channel]
    
    return output

def add_text(image: ImageType, text: str="Sample Text", position: Tuple[int, int]=(50, 50), 
            font: int=cv.FONT_HERSHEY_SIMPLEX, font_scale: float=1, 
            color: Tuple[int, int, int]=(255, 255, 255), thickness: int=2) -> ImageType:
    """
    Add text to an image.
    
    Args:
        image: The input image
        text: Text to add (default: "Sample Text")
        position: Position (x, y) to place text (default: (50, 50))
        font: Font type (default: cv.FONT_HERSHEY_SIMPLEX)
        font_scale: Font scale (default: 1)
        color: Text color in BGR format (default: white)
        thickness: Text thickness (default: 2)
        
    Returns:
        Image with text
    """
    # Create a copy of the image to avoid modifying the original
    img_copy = image.copy()
    
    # Put text on the image
    cv.putText(img_copy, text, position, font, font_scale, color, thickness)
    
    return img_copy

class ImageProcessorApp:
    def __init__(self, root: tk.Tk, image_path: str):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("1000x650")
        
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
        
        # Create buttons frame - using two rows for better organization
        self.buttons_frame1 = ttk.Frame(self.main_frame)
        self.buttons_frame1.pack(pady=5)
        
        self.buttons_frame2 = ttk.Frame(self.main_frame)
        self.buttons_frame2.pack(pady=5)
        
        # Create buttons for each operation
        self.create_buttons()
        
        # Create reset button
        self.reset_button = ttk.Button(self.main_frame, text="Reset to Original", command=self.reset_image)
        self.reset_button.pack(pady=5)
        
        # Display the original image (do this last after all UI elements are created)
        self.display_image(self.original_img, "Original Image")
        
    def create_buttons(self):
        # First row of operations
        operations_row1 = [
            ("Grayscale", self.apply_grayscale),
            ("Blur", self.apply_blur),
            ("Edges", self.detect_edges),
            ("Dilate", self.apply_dilate),
            ("Erode", self.apply_erode),
            ("Resize", self.apply_resize),
            ("Crop", self.apply_crop)
        ]
        
        # Second row of operations (new functions)
        operations_row2 = [
            ("Brightness", self.apply_brightness),
            ("Threshold", self.apply_threshold),
            ("Rotate", self.apply_rotate),
            ("Flip", self.apply_flip),
            ("Sharpen", self.apply_sharpen),
            ("Add Text", self.apply_add_text),
            ("Color Filter", self.apply_color_filter)
        ]
        
        # Create buttons for the first row
        for text, command in operations_row1:
            btn = ttk.Button(self.buttons_frame1, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=5)
        
        # Create buttons for the second row
        for text, command in operations_row2:
            btn = ttk.Button(self.buttons_frame2, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=5)
    
    def show_warning(self, message):
        """Show a warning dialog but allow the user to proceed."""
        return messagebox.askokcancel("Operation Warning", message)
    
    def check_operation_conflicts(self, operation):
        """Check if the new operation conflicts with previously applied operations."""
        conflicting_pairs = {
            "Dilate": ["Erode"],
            "Erode": ["Dilate"],
            "Blur": ["Edges", "Dilate", "Erode", "Sharpen"],
            "Edges": ["Blur"],
            "Sharpen": ["Blur"]
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
        
    def apply_brightness(self):
        # Prompt user for alpha (contrast) and beta (brightness) values
        alpha = simpledialog.askfloat("Adjust Contrast", "Enter contrast multiplier (1.0 = no change):", 
                                     initialvalue=1.5, minvalue=0.1, maxvalue=3.0)
        if alpha is None:  # User cancelled
            return
            
        beta = simpledialog.askinteger("Adjust Brightness", "Enter brightness value to add (0 = no change):", 
                                      initialvalue=0, minvalue=-100, maxvalue=100)
        if beta is None:  # User cancelled
            return
        
        result = adjust_brightness_contrast(self.current_img, alpha=alpha, beta=beta)
        self.apply_transformation(f"Brightness/Contrast (α={alpha:.1f}, β={beta})", result)
    
    def apply_threshold(self):
        # Convert to grayscale if needed
        if len(self.current_img.shape) > 2 and "Grayscale" not in self.applied_operations:
            if self.show_warning("Thresholding works best on grayscale images. Convert to grayscale first?"):
                img_to_threshold = apply_grayscale(self.current_img)
                self.applied_operations.append("Grayscale")  # Add intermediate step
            else:
                img_to_threshold = self.current_img
        else:
            img_to_threshold = self.current_img
        
        # Prompt user for threshold value
        threshold = simpledialog.askinteger("Threshold", "Enter threshold value (0-255):", 
                                           initialvalue=127, minvalue=0, maxvalue=255)
        if threshold is None:  # User cancelled
            return
        
        result = apply_threshold(img_to_threshold, threshold=threshold)
        self.apply_transformation(f"Threshold ({threshold})", result)
    
    def apply_rotate(self):
        # Prompt user for rotation angle
        angle = simpledialog.askfloat("Rotate", "Enter rotation angle in degrees:", 
                                     initialvalue=45, minvalue=-180, maxvalue=180)
        if angle is None:  # User cancelled
            return
        
        result = rotate_image(self.current_img, angle=angle)
        self.apply_transformation(f"Rotate ({angle}°)", result)
    
    def apply_flip(self):
        # Create a popup to select flip direction
        flip_window = tk.Toplevel(self.root)
        flip_window.title("Flip Direction")
        flip_window.geometry("250x150")
        flip_window.resizable(False, False)
        
        flip_type = tk.IntVar(value=1)  # Default to horizontal flip
        
        ttk.Radiobutton(flip_window, text="Horizontal Flip", variable=flip_type, value=1).pack(pady=5)
        ttk.Radiobutton(flip_window, text="Vertical Flip", variable=flip_type, value=0).pack(pady=5)
        ttk.Radiobutton(flip_window, text="Both (180° Rotation)", variable=flip_type, value=-1).pack(pady=5)
        
        def apply_selected_flip():
            flip_code = flip_type.get()
            result = flip_image(self.current_img, flip_code=flip_code)
            
            flip_names = {1: "Horizontal", 0: "Vertical", -1: "Both"}
            self.apply_transformation(f"Flip ({flip_names[flip_code]})", result)
            flip_window.destroy()
        
        ttk.Button(flip_window, text="Apply", command=apply_selected_flip).pack(pady=10)
    
    def apply_sharpen(self):
        # Check for conflicts
        warning = self.check_operation_conflicts("Sharpen")
        if warning and not self.show_warning(warning):
            return
            
        result = sharpen_image(self.current_img)
        self.apply_transformation("Sharpen", result)
    
    def apply_add_text(self):
        # Create a dialog to get text details
        text_window = tk.Toplevel(self.root)
        text_window.title("Add Text")
        text_window.geometry("300x250")
        
        ttk.Label(text_window, text="Text:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        text_var = tk.StringVar(value="Sample Text")
        ttk.Entry(text_window, textvariable=text_var, width=25).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(text_window, text="Position X:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        pos_x_var = tk.IntVar(value=50)
        ttk.Entry(text_window, textvariable=pos_x_var, width=10).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(text_window, text="Position Y:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        pos_y_var = tk.IntVar(value=50)
        ttk.Entry(text_window, textvariable=pos_y_var, width=10).grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(text_window, text="Font Size:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        font_size_var = tk.DoubleVar(value=1.0)
        ttk.Entry(text_window, textvariable=font_size_var, width=10).grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(text_window, text="Color:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        color_frame = ttk.Frame(text_window)
        color_frame.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
        color_options = {
            "White": (255, 255, 255),
            "Black": (0, 0, 0),
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Blue": (255, 0, 0),
            "Yellow": (0, 255, 255)
        }
        
        color_var = tk.StringVar(value="White")
        color_combo = ttk.Combobox(color_frame, textvariable=color_var, values=list(color_options.keys()), width=10)
        color_combo.pack()
        
        def apply_text():
            text = text_var.get()
            position = (pos_x_var.get(), pos_y_var.get())
            font_scale = font_size_var.get()
            color = color_options[color_var.get()]
            
            result = add_text(self.current_img, text=text, position=position, 
                             font_scale=font_scale, color=color)
            
            self.apply_transformation(f"Add Text ({text})", result)
            text_window.destroy()
        
        ttk.Button(text_window, text="Apply", command=apply_text).grid(row=5, column=0, columnspan=2, pady=10)
    
    def apply_color_filter(self):
        # Only works on color images
        if len(self.current_img.shape) < 3:
            messagebox.showwarning("Warning", "Color filtering only works on color images.")
            return
        
        # Create a dialog to select color channel
        filter_window = tk.Toplevel(self.root)
        filter_window.title("Extract Color Channel")
        filter_window.geometry("200x150")
        filter_window.resizable(False, False)
        
        channel_var = tk.IntVar(value=2)  # Default to Red channel
        
        ttk.Radiobutton(filter_window, text="Red Channel", variable=channel_var, value=2).pack(pady=5)
        ttk.Radiobutton(filter_window, text="Green Channel", variable=channel_var, value=1).pack(pady=5)
        ttk.Radiobutton(filter_window, text="Blue Channel", variable=channel_var, value=0).pack(pady=5)
        
        def apply_filter():
            channel = channel_var.get()
            result = extract_color_channel(self.current_img, channel=channel)
            
            channel_names = {0: "Blue", 1: "Green", 2: "Red"}
            self.apply_transformation(f"Extract {channel_names[channel]} Channel", result)
            filter_window.destroy()
        
        ttk.Button(filter_window, text="Apply", command=apply_filter).pack(pady=10)

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