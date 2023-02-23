import numpy as np
import matplotlib.pyplot as plt
import math

def process_image(input_image):
    # Compute the histogram of the input image
    counts = np.bincount(input_image.flatten(), minlength=8)
    count_dict = dict(enumerate(counts))
    
    # Compute the PDF and CDF of the input image
    sum_val = np.size(input_image)
    pdf_dict = {i: count / sum_val for i, count in count_dict.items()}
    total_count = sum(pdf_dict.values())
    cdf_dict = {}
    cumulative_count = 0.0
    for value in sorted(pdf_dict.keys()):
        cumulative_count += pdf_dict[value] / total_count
        cdf_dict[value] = cumulative_count
    
    # Compute the replacement dictionary
    cdf_total = {i: cdf_dict[i] * 7 for i in range(len(cdf_dict))}
    replace_dict = {i: round(cdf_total[i]) for i in range(len(cdf_total))}
    
    # Replace the pixel values in the input image using the replacement dictionary
    replace_func = np.vectorize(lambda x: replace_dict[x])
    output_image = replace_func(input_image)
    
    return output_image

def plot_histogram(image):
    values = image.flatten()
    
    # Create a histogram of the pixel values
    hist, bins = np.histogram(values, bins=np.arange(0, 9), density=False)

    # Plot the histogram
    plt.bar(bins[:-1], hist, align='center')
    plt.xticks(bins[:-1])
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    plt.title('Histogram of Input Image')
    plt.show()

def histogram_equalization(image):
    # Convert the image to grayscale
    #gray_image = np.array(color_image.convert('L'))
    if len(image.shape) == 3:
        gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray_image = image.astype(np.uint8)
    
    # Compute the PDF and CDF of the image
    sum = np.size(gray_image)
    counts = np.bincount(gray_image.flatten(), minlength=8)
    count_dict = dict(enumerate(counts))
    pdf_dict = dict(enumerate(counts/sum))

    # Compute the total count (sum of all values in the PDF)
    total_count = sum(pdf_dict.values())

    # Compute the CDF
    cdf_dict = {}
    cumulative_count = 0.0
    for value in sorted(pdf_dict.keys()):
        cumulative_count += pdf_dict[value] / total_count
        cdf_dict[value] = cumulative_count

    # Compute the new pixel values based on the CDF
    cdf_total = {}
    for i in range(len(cdf_dict)):
        cdf_total[i] = cdf_dict[i] * 7
    replace_dict = {}
    for key in cdf_total:
        replace_dict[key] = round(cdf_total[key])

    # Define a function to replace pixel values using the replace_dict
    def replace_value(x):
        return replace_dict[x]

    # Use NumPy's vectorize function to apply the replacement function to every pixel in the image
    replace_func = np.vectorize(replace_value)
    output_image = replace_func(gray_image)

    # Return the output image
    return output_image
