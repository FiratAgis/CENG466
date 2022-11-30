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

def visualize_fourier(fouriered_image: np.ndarray):
    ### show fourier transform
    # Magnitude spectrum
    magnitude = abs(fouriered_image)
    # normalize
    magnitude = normalize(magnitude)
    # Kowalsky enhance >:(
    output = fold_image_to_center(magnitude)
    output = np.log(1 + output)
    return output

def everything_fourier(image_name: str):
    ##### get channel and test
    img = read_image(INPUT_PATH + image_name)
    final_image = np.zeros(img.shape)

    red_channel = get_rgb(img, 'R', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    ### get fourier transformation of image
    fouriered_image = get_fast_fourier(red_channel)
    output = visualize_fourier(fouriered_image)

    write_image(output, OUTPUT_PATH + "folded_fourier_magnitude_red"+image_name)

    # angle spectrum
    angle = np.angle(fouriered_image)
    write_image(output, OUTPUT_PATH + "fourier_angle_red_"+image_name)

    # reconstruct the image
    output = np.real(fp.ifftn(fouriered_image))
    final_image[:, :, 0] = output[:, :]
    write_image(output, OUTPUT_PATH + "reconstructed_red_"+image_name)


    ##### get channel and test
    red_channel = get_rgb(img, 'G', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    ### get fourier transformation of image
    fouriered_image = get_fast_fourier(red_channel)
    output = visualize_fourier(fouriered_image)

    write_image(output, OUTPUT_PATH + "folded_fourier_magnitude_green_"+image_name)

    # angle spectrum
    angle = np.angle(fouriered_image)
    write_image(output, OUTPUT_PATH + "fourier_angle_green_"+image_name)

    # reconstruct the image
    output = np.real(fp.ifftn(fouriered_image))
    final_image[:, :, 1] = output[:, :]
    write_image(output, OUTPUT_PATH + "reconstructed_green_"+image_name)


    ##### get channel and test
    red_channel = get_rgb(img, 'B', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    ### get fourier transformation of image
    fouriered_image = get_fast_fourier(red_channel)
    output = visualize_fourier(fouriered_image)

    write_image(output, OUTPUT_PATH + "folded_fourier_magnitude_blue_"+image_name)

    # angle spectrum
    angle = np.angle(fouriered_image)
    write_image(output, OUTPUT_PATH + "fourier_angle_blue_"+image_name)

    # reconstruct the image
    output = np.real(fp.ifftn(fouriered_image))
    final_image[:, :, 2] = output[:, :]
    write_image(output, OUTPUT_PATH + "reconstructed_blue_"+image_name)

    final_image = normalize(final_image)
    write_image(final_image, OUTPUT_PATH + "reconstructed_final_"+image_name)

def fold_image_to_center(img_func: np.ndarray):
    shape = img_func.shape
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
    
    padded_img = np.zeros((maxh, maxv))
    for i in range(0, maxh):
        for j in range(0, maxv):
            if i < shape[0] and j < shape[1]:
                padded_img[i][j] = img_func[i][j]
            else:
                padded_img[i][j] = 0
    
    return padded_img

def hadamart_from_image(img_func: np.ndarray):
    img_func = zero_padding_to_square(img_func)
    hadamard_matrixn = hadamard(img_func.shape[0])
    hadamard_matrixm = hadamard(img_func.shape[1])
    ara_basamak = np.matmul(hadamard_matrixn, img_func)
    return_val = np.matmul(ara_basamak, hadamard_matrixm)
    return return_val

def reverse_hadamard(hadamarded_img: np.ndarray):
    # print(img_func.shape)
    hadamard_matrixn = hadamard(hadamarded_img.shape[0])
    hadamard_matrixm = hadamard(hadamarded_img.shape[1])
    inv_hadamard_matrixn = np.linalg.inv(hadamard_matrixn)
    inv_hadamard_matrixm = np.linalg.inv(hadamard_matrixm)
    ara_basamak = np.matmul(inv_hadamard_matrixn, hadamarded_img)
    return_val = np.matmul(ara_basamak, inv_hadamard_matrixm)

    return return_val

def visualize_hadamard(hadamarded_img: np.ndarray):
    magnitude = abs(hadamarded_img)
    magnitude = normalize(magnitude)
    return_val = np.log(1 + magnitude)
    return return_val

def everything_hadamard(image_name: str):
    ##### get channel and test
    img = read_image(INPUT_PATH + image_name)
    original_shape = img.shape

    red_channel = get_rgb(img, 'R', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    ### get fourier transformation of image
    fouriered_image = hadamart_from_image(red_channel)
    output = visualize_hadamard(fouriered_image)

    write_image(output, OUTPUT_PATH + "folded_hadamard_magnitude_red"+image_name)

    # reconstruct the image
    output = reverse_hadamard(fouriered_image)
    final_image = np.zeros((output.shape[0], output.shape[1], 3))
    final_image[:, :, 0] = output[:, :]
    write_image(output, OUTPUT_PATH + "hadamard_reconstructed_red_"+image_name)


    ##### get channel and test
    red_channel = get_rgb(img, 'G', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    ### get fourier transformation of image
    fouriered_image = hadamart_from_image(red_channel)
    output = visualize_hadamard(fouriered_image)

    write_image(output, OUTPUT_PATH + "folded_hadamard_magnitude_green_"+image_name)


    # reconstruct the image
    output = reverse_hadamard(fouriered_image)
    final_image[:, :, 1] = output[:, :]
    write_image(output, OUTPUT_PATH + "hadamard_reconstructed_green_"+image_name)


    ##### get channel and test
    red_channel = get_rgb(img, 'B', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    ### get fourier transformation of image
    fouriered_image = hadamart_from_image(red_channel)
    output = visualize_hadamard(fouriered_image)

    write_image(output, OUTPUT_PATH + "folded_hadamard_magnitude_blue_"+image_name)


    # reconstruct the image
    output = reverse_hadamard(fouriered_image)
    final_image[:, :, 2] = output[:, :]
    write_image(output, OUTPUT_PATH + "hadamard_reconstructed_blue_"+image_name)

    final_image = normalize(final_image)
    final_image = final_image[0:original_shape[0], 0:original_shape[1], :]
    print(original_shape)
    print(final_image.shape)
    write_image(final_image, OUTPUT_PATH + "hadamard_reconstructed_final_"+image_name)



def get_cosine(img_func: np.ndarray):
    # Image in Fourier domain
    return fp.dct(img_func)

def visualize_cosine(fouriered_image: np.ndarray):
    ### show fourier transform
    # Magnitude spectrum
    magnitude = abs(fouriered_image)
    # normalize
    magnitude = normalize(magnitude)
    # Kowalsky enhance >:(
    output = np.log(1 + magnitude)
    return output

def everything_cosine(image_name: str):
    ##### get channel and test
    img = read_image(INPUT_PATH + image_name)
    final_image = np.zeros(img.shape)

    red_channel = get_rgb(img, 'R', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    ### get fourier transformation of image
    fouriered_image = get_cosine(red_channel)
    output = visualize_cosine(fouriered_image)

    write_image(output, OUTPUT_PATH + "folded_cosine_magnitude_red"+image_name)

    # angle spectrum
    angle = np.angle(fouriered_image)
    write_image(output, OUTPUT_PATH + "cosine_angle_red_"+image_name)

    # reconstruct the image
    output = np.real(fp.idct(fouriered_image))
    final_image[:, :, 0] = output[:, :]
    write_image(output, OUTPUT_PATH + "cosine_reconstructed_red_"+image_name)


    ##### get channel and test
    red_channel = get_rgb(img, 'G', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    ### get fourier transformation of image
    fouriered_image = get_cosine(red_channel)
    output = visualize_cosine(fouriered_image)

    write_image(output, OUTPUT_PATH + "folded_cosine_magnitude_green_"+image_name)

    # angle spectrum
    angle = np.angle(fouriered_image)
    write_image(output, OUTPUT_PATH + "cosine_angle_green_"+image_name)

    # reconstruct the image
    output = np.real(fp.idct(fouriered_image))
    final_image[:, :, 1] = output[:, :]
    write_image(output, OUTPUT_PATH + "cosine_reconstructed_green_"+image_name)


    ##### get channel and test
    red_channel = get_rgb(img, 'B', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    ### get fourier transformation of image
    fouriered_image = get_cosine(red_channel)
    output = visualize_cosine(fouriered_image)

    write_image(output, OUTPUT_PATH + "folded_cosine_magnitude_blue_"+image_name)

    # angle spectrum
    angle = np.angle(fouriered_image)
    write_image(output, OUTPUT_PATH + "cosine_angle_blue_"+image_name)

    # reconstruct the image
    output = np.real(fp.idct(fouriered_image))
    final_image[:, :, 2] = output[:, :]
    write_image(output, OUTPUT_PATH + "cosine_reconstructed_blue_"+image_name)

    final_image = normalize(final_image)
    write_image(final_image, OUTPUT_PATH + "cosine_reconstructed_final_"+image_name)


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


def generate_ideal_filter(size_x: int, size_y: int, r: float) -> np.ndarray:
    return_val = np.zeros((size_x, size_y))
    centerx = size_x/2
    centery = size_y/2
    for x in range(0, size_x):
        for y in range(0, size_y):
            if( math.sqrt((x-centerx)*(x-centerx) + (y-centery)*(y-centery)) <= r):
                return_val[x][y] = 1
            else:
                return_val[x][y] = 0
    return return_val


def generate_gaussian_filter(size_x: int, size_y: int, sd: float) -> np.ndarray:
    ret_val = np.zeros((size_x, size_y))
    for x in range(0, size_x):
        for y in range(0, size_y):
            ret_val[x][y] = pow(np.e, -2 * np.pi * (x*x + y*y) * sd * sd)
    return ret_val


def generate_butterworth_filter(size_x: int, size_y: int, n: float, r: float) -> np.ndarray:
    ret_val = np.zeros((size_x, size_y))
    for x in range(0, size_x):
        for y in range(0, size_y):
            ret_val[x][y] = 1 / (1 + pow((x*x + y*y) / r*r, n))
    return ret_val


def apply_filter(arr: np.ndarray, filter_type: str, pass_type: str, cut_off: float) -> np.ndarray:
    if filter_type == "ideal":
        fil = generate_ideal_filter(arr.shape[0], arr.shape[1], cut_off)
        if pass_type == "low":
            return np.multiply(arr, fil)
        if pass_type == "high":
            return np.multiply(1 - fil, arr)
    elif filter_type == "gaussian":
        fil = generate_gaussian_filter(arr.shape[0], arr.shape[1], cut_off)
        if pass_type == "low":
            return np.multiply(arr, fil)
        elif pass_type == "high":
            return np.multiply(1 - fil, arr)
    elif filter_type == "butterworth":
        fil = generate_butterworth_filter(arr.shape[0], arr.shape[1], 2, cut_off)
        if pass_type == "low":
            return np.multiply(arr, fil)
        elif pass_type == "high":
            return np.multiply(1 - fil, arr)


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    

    
    #everything_fourier("1.png")  #takes fourier transformation of an image for every channel then recunstructs them and creates final image -> total 10 images

    #everything_cosine("1.png")   #takes cosine transformation of an image for every channel then recunstructs them and creates final image  -> total 10 images
   
    #everything_hadamard("1.png") #takes hadamard transformation of an image for every channel then recunstructs them and creates final image  -> total 10 images
    
    

    img = read_image(INPUT_PATH + "3.png")
    final_image = np.zeros(img.shape)

    red_channel = get_rgb(img, 'R', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    ### get fourier transformation of image
    fouriered_image = get_fast_fourier(red_channel)
    output = visualize_fourier(fouriered_image)
    output = apply_filter(output, "ideal", "low", 50)

    write_image(output, OUTPUT_PATH + "filtered_fourier_magnitude_red_"+"3.png")


    # reconstruct the image
    fouriered_image = fold_image_to_center(fouriered_image)
    fouriered_image = apply_filter(fouriered_image, "ideal", "low", 50)
    fouriered_image = fold_image_to_center(fouriered_image)
    output = np.real(fp.ifftn(fouriered_image))
    final_image[:, :, 0] = output[:, :]
    write_image(output, OUTPUT_PATH + "filtered_reconstructed_red_"+"3.png")
