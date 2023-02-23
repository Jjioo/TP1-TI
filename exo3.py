import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_histogram(image):
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Gray Level")
    plt.ylabel("Frequency")
    plt.bar(bin_edges[:-1], histogram, width=1)
    plt.show()

def process_image(image):
    # Calculate histogram
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))
    
    # Calculate cumulative histogram
    cumulative_hist = np.cumsum(histogram)
    
    # Equalize histogram
    num_pixels = image.shape[0] * image.shape[1]
    equalized = ((cumulative_hist[image] * 255) / num_pixels).astype(np.uint8)
    
    return equalized

# Load the input image
color_image = Image.open("/content/pont.pgm")
array = np.array(color_image)
gray_image = np.array(color_image.convert('L'))

# Plot the original image and its histogram
plt.imshow(gray_image, cmap="gray")
plt.title("Original Image")
plt.show()
plot_histogram(gray_image)

# Process the image and plot the result and its histogram
output = process_image(gray_image)
plt.imshow(output, cmap="gray")
plt.title("Equalized Image")
plt.show()
plot_histogram(output)
