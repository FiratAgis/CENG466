"""
Submission by
Fırat Ağış, e2236867
Robin Koç, e246871
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
from PIL import Image
import math
import scipy.fftpack as fp
from scipy.linalg import hadamard

INPUT_PATH = "./THE_2 images/"
OUTPUT_PATH = "./Outputs/"


def read_image(img_path: str, rgb: bool = True) -> np.ndarray:
    img_data: Image
    if rgb:
        img_data = Image.open(img_path)
    else:
        img_data = Image.open(img_path).convert('L')
    return np.asarray(img_data)
    

def get_rgb(img_func: np.ndarray, channel: str, full_image: bool = True) -> np.ndarray:
    if channel == 'R':
        if full_image:
            return_val = np.zeros(img_func.shape)
            return_val[:, :, 0] = img_func[:, :, 0]
            return return_val
        else:
            return img_func[:, :, 0]
    if channel == 'G':
        if full_image:
            return_val = np.zeros(img_func.shape)
            return_val[:, :, 1] = img_func[:, :, 1]
            return return_val
        else:
            return img_func[:, :, 1]
    if channel == 'B':
        if full_image:
            return_val = np.zeros(img_func.shape)
            return_val[:, :, 2] = img_func[:, :, 2]
            return return_val
        else:
            return img_func[:, :, 2]


def get_fast_fourier(img_func: np.ndarray):
    # Image in Fourier domain
    return fp.fftn(img_func)


def fold_image_to_center(img_func: np.ndarray):
    shape = img_func.shape
    print(shape)
    img_func[0:(shape[0]//2), 0:(shape[1]//2)] = np.flip(img_func[0:(shape[0]//2), 0:(shape[1]//2)], (0, 1))  # flip upper_left
    img_func[(shape[0]//2):, 0:(shape[1]//2)] = np.flip(img_func[(shape[0]//2):, 0:(shape[1]//2)], (0, 1))  # flip upper_right
    img_func[0:(shape[0]//2), (shape[1]//2):] = np.flip(img_func[0:(shape[0]//2), (shape[1]//2):], (0, 1))    # flip bottom_left
    img_func[(shape[0]//2):, (shape[1]//2):] = np.flip(img_func[(shape[0]//2):, (shape[1]//2):], (0, 1))  # flip bottom_right
    return img_func


def zero_padding_to_square(img_func: np.ndarray):
    shape = img_func.shape

    maxh = shape[0]
    i = 2
    while maxh > i:
        i = i*2
    maxh = i
    if maxh > shape[0]+1000:
        maxh = maxh/2

    maxv = shape[1]
    i = 2
    while maxv > i:
        i = i*2
    maxv = i
    if maxv > shape[1]+1000:
        maxv = maxv/2
    
    padded_img = np.zeros((maxh, maxv, 3))
    for i in range(0, maxh):
        for j in range(0, maxv):
            if i < shape[0] and j < shape[1]:
                padded_img[i][j][0] = img_func[i][j][0]
                padded_img[i][j][1] = img_func[i][j][1]
                padded_img[i][j][2] = img_func[i][j][2]
            else:
                padded_img[i][j][0] = 0
                padded_img[i][j][1] = 0
                padded_img[i][j][2] = 0
    
    return padded_img


def hadamart_from_image(img_func: np.ndarray):
    img_func = zero_padding_to_square(img_func)
    print(img_func.shape)
    hadamard_matrixn = hadamard(img_func.shape[0])
    hadamard_matrixm = hadamard(img_func.shape[1])
    red_channel = get_rgb(img_func, 'R', False) 
    ara_basamak = np.matmul(hadamard_matrixn, red_channel)
    return_val = np.matmul(ara_basamak, hadamard_matrixm)

    print("ret before normalize ", return_val)
    # normalize
    return_val = normalize(return_val)

    print("ara ", ara_basamak)
    print("red ", red_channel)
    print("hadan ", hadamard_matrixn.shape)
    print("hadam ", hadamard_matrixm.shape)
    print("ret ", return_val)
    return return_val


def get_cosine(img_func: np.ndarray):
    # Image in Fourier domain
    return fp.dct(img_func)


def write_image(img_func: np.ndarray, output_path: str, rgb: bool = True) -> None:
    shape = img_func.shape
    if len(shape) == 2:
        rgb = False
    output_file = np.zeros(shape)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            if rgb:
                for z in range(0, shape[2]):
                    output_file[x][y][z] = float(img_func[x][y][z]) / 255
            else:
                output_file[x][y] = float(img_func[x][y]) / 255
    matplotlib.image.imsave(output_path, output_file, cmap="gray")


def normalize(arr: np.ndarray) -> np.ndarray:
    max_val = arr.max()
    min_val = arr.min()
    return (arr - min_val) * (254 / (max_val - min_val))


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    ##### get channel and test
    img = read_image(INPUT_PATH + "1.png")
    red_channel = get_rgb(img, 'R', False)

    output = get_cosine(red_channel)

    magnitude = abs(output)
    # normalize
    magnitude = normalize(magnitude)
    """max = magnitude.max()
    min = magnitude.min()
    ratio = 254/(max-min)
    magnitude = np.multiply(magnitude, ratio)"""
    output = np.log(1 + magnitude)

    write_image(output, OUTPUT_PATH + "deneme.png")

    ##### get channel and test
    img = read_image(INPUT_PATH + "1.png")
    red_channel = get_rgb(img, 'R', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    ### get fourier transformation of image
    fouriered_image = get_fast_fourier(red_channel)

    ### show fourier transform
    # Magnitude spectrum
    magnitude = abs(fouriered_image)
    # normalize
    magnitude = normalize(magnitude)
    """max = magnitude.max()
    min = magnitude.min()
    ratio = 254/(max-min)
    magnitude = np.multiply(magnitude, ratio)"""
    output = np.log(1 + magnitude)
    write_image(output, OUTPUT_PATH + "fourier_magnitude_1.png")

    # Kowalsky enhance >:(
    output = fold_image_to_center(magnitude)
    output = np.log(1 + output)
    write_image(output, OUTPUT_PATH + "folded_fourier_magnitude_1.png")

    # angle spectrum
    angle = np.angle(fouriered_image)
    write_image(output, OUTPUT_PATH + "fourier_angle_1.png")

    # reconstruct the image
    output = np.real(fp.ifftn(fouriered_image))
    write_image(output, OUTPUT_PATH + "reconstructed_1.png")
