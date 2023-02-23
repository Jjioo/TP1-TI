"""
input_image = np.array([[4, 4, 4, 4, 4],
                        [3, 4, 5, 4, 3],
                        [3, 5, 5, 5, 3],
                        [3, 4, 5, 4, 3],
                        [4, 4, 4, 4, 4]])

"""

import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
from image_processing import plot_histogram
color_image = Image.open("/content/tiger.jpg")
gray_image = np.array(color_image.convert('L'))
plot_histogram(gray_image)
output = process_image(gray_image)
plot_histogram(output)
output = (output * 255).astype(np.uint8)

result_image = Image.fromarray(output)
result_image.save("/content/result.jpg")