
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image

INPUT_PATH = "./THE1_Images/"
OUTPUT_PATH = "./Outputs/"

def read_image(img_path: str, rgb = True):
    img_data = Image.open(img_path)
    return np.asarray(img_data)

"""def write_image(img, output_path, rgb = True):

def extract_save_histogram(img, path):


def rotate_image(img,  degree:float = 0, interpolation_type:str = "linear"):
    #interpolation type: "linear" or "cubic"
    #degree: 45 or 90

    return rotated

def histogram_equalization(img):

    return img_hist_eq"""

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    #PART1
    img = read_image(INPUT_PATH + "a1.png")
    print(img)

    """output = rotate_image(img, 45, "linear")
    write_image(output, OUTPUT_PATH + "a1_45_linear.png")

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 45, "cubic")
    write_image(output, OUTPUT_PATH + "a1_45_cubic.png")

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 90, "linear")
    write_image(output, OUTPUT_PATH + "a1_90_linear.png")

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 90, "cubic")
    write_image(output, OUTPUT_PATH + "a1_90_cubic.png")

    img = read_image(INPUT_PATH + "a2.png")
    output = rotate_image(img, 45, "linear")
    write_image(output, OUTPUT_PATH + "a2_45_linear.png")

    img = read_image(INPUT_PATH + "a2.png")
    output = rotate_image(img, 45, "cubic")
    write_image(output, OUTPUT_PATH + "a2_45_cubic.png")

    #PART2
    img = read_image(INPUT_PATH + "b1.png", rgb = False)
    extract_save_histogram(img, OUTPUT_PATH + "original_histogram.png")
    equalized = histogram_equalization(img)
    extract_save_histogram(equalized, OUTPUT_PATH + "equalized_histogram.png")
    write_image(output, OUTPUT_PATH + "enhanced_image.png")

    # BONUS
    # Define the following function
    # equalized = adaptive_histogram_equalization(img)
    # extract_save_histogram(equalized, OUTPUT_PATH + "adaptive_equalized_histogram.png")
    # write_image(output, OUTPUT_PATH + "adaptive_enhanced_image.png")"""





