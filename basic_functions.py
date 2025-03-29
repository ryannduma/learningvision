import cv2 as cv

def apply_grayscale(image):
    """
    Convert an image to grayscale.
    
    Args:
        image: The input image in BGR format
        
    Returns:
        The grayscale version of the image
    """
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def apply_blur(image, kernel_size=(7,7)):
    """
    Apply Gaussian blur to an image.
    
    Args:
        image: The input image
        kernel_size: Tuple representing blur kernel size (default: (7,7))
        
    Returns:
        The blurred image
    """
    return cv.GaussianBlur(image, kernel_size, cv.BORDER_DEFAULT)

def detect_edges(image, threshold1=125, threshold2=175):
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

def dilate_image(image, kernel_size=(7,7), iterations=3):
    """
    Dilate an image to enhance edges.
    
    Args:
        image: The input image (typically an edge image)
        kernel_size: Tuple representing dilation kernel size (default: (7,7))
        iterations: Number of times dilation is applied (default: 3)
        
    Returns:
        The dilated image
    """
    return cv.dilate(image, kernel_size, iterations=iterations)

def erode_image(image, kernel_size=(7,7), iterations=3):
    """
    Erode an image (opposite of dilation).
    
    Args:
        image: The input image
        kernel_size: Tuple representing erosion kernel size (default: (7,7))
        iterations: Number of times erosion is applied (default: 3)
        
    Returns:
        The eroded image
    """
    return cv.erode(image, kernel_size, iterations=iterations)

def resize_image(image, dimensions=(500,500)):
    """
    Resize an image to specified dimensions.
    
    Args:
        image: The input image
        dimensions: Tuple of (width, height) for the output image (default: (500,500))
        
    Returns:
        The resized image
    """
    return cv.resize(image, dimensions, interpolation=cv.INTER_CUBIC)

def crop_image(image, y_range=(50,200), x_range=(200,400)):
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

def process_image():
    """
    Main function to process an image based on user input.
    Loads an image and applies transformations based on user selection.
    """
    # Read in an image
    img_path = '/Users/rynduma/learningvision/learningvision/Resources/Photos/park.jpg'
    img = cv.imread(img_path)
    
    if img is None:
        print(f"Error: Could not read image at {img_path}")
        return
    
    # Show original image
    cv.imshow('Original Image', img)
    
    # Display options to the user
    print("\nImage Processing Options:")
    print("1. Convert to grayscale")
    print("2. Apply Gaussian blur")
    print("3. Detect edges (Canny)")
    print("4. Dilate the image")
    print("5. Erode the image")
    print("6. Resize the image")
    print("7. Crop the image")
    print("8. Apply all transformations")
    print("0. Exit")
    
    # Get user choice
    choice = input("\nEnter your choice (0-8): ")
    
    # Process based on user choice
    if choice == '1':
        result = apply_grayscale(img)
        cv.imshow('Grayscale', result)
    elif choice == '2':
        result = apply_blur(img)
        cv.imshow('Blurred', result)
    elif choice == '3':
        blurred = apply_blur(img)  # Blur first for better edge detection
        result = detect_edges(blurred)
        cv.imshow('Edges', result)
    elif choice == '4':
        blurred = apply_blur(img)
        edges = detect_edges(blurred)
        result = dilate_image(edges)
        cv.imshow('Dilated', result)
    elif choice == '5':
        blurred = apply_blur(img)
        edges = detect_edges(blurred)
        dilated = dilate_image(edges)
        result = erode_image(dilated)
        cv.imshow('Eroded', result)
    elif choice == '6':
        result = resize_image(img)
        cv.imshow('Resized', result)
    elif choice == '7':
        result = crop_image(img)
        cv.imshow('Cropped', result)
    elif choice == '8':
        # Apply all transformations and show each step
        gray = apply_grayscale(img)
        cv.imshow('Grayscale', gray)
        
        blur = apply_blur(img)
        cv.imshow('Blur', blur)
        
        canny = detect_edges(blur)
        cv.imshow('Canny Edges', canny)
        
        dilated = dilate_image(canny)
        cv.imshow('Dilated', dilated)
        
        eroded = erode_image(dilated)
        cv.imshow('Eroded', eroded)
        
        resized = resize_image(img)
        cv.imshow('Resized', resized)
        
        cropped = crop_image(img)
        cv.imshow('Cropped', cropped)
    elif choice == '0':
        print("Exiting program.")
        return
    else:
        print("Invalid choice. Please try again.")
    
    # Wait for a key press to close all windows
    cv.waitKey(0)
    cv.destroyAllWindows()

# Execute the main function if this script is run directly
if __name__ == "__main__":
    process_image()