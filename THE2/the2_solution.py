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


def visualize_fourier(img_func: np.ndarray) -> np.ndarray:
    # show fourier transform
    # Magnitude spectrum
    magnitude = abs(img_func)
    # normalize
    magnitude = normalize(magnitude)
    # Kowalsky enhance >:(
    ret_val = fold_image_to_center(magnitude)
    return np.log(1 + ret_val)


def everything_fourier(image_name: str):
    # get channel and test
    img = read_image(INPUT_PATH + image_name)
    final_image = np.zeros(img.shape)

    red_channel = get_rgb(img, 'R', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    # get fourier transformation of image
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

    # get channel and test
    red_channel = get_rgb(img, 'G', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    # get fourier transformation of image
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

    # get channel and test
    red_channel = get_rgb(img, 'B', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    # get fourier transformation of image
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


def hadamart_from_image(img_func: np.ndarray) -> np.ndarray:
    img_func = zero_padding_to_square(img_func)
    hadamard_matrixn = hadamard(img_func.shape[0])
    hadamard_matrixm = hadamard(img_func.shape[1])
    ara_basamak = np.matmul(hadamard_matrixn, img_func)
    return np.matmul(ara_basamak, hadamard_matrixm)


def reverse_hadamard(hadamarded_img: np.ndarray) -> np.ndarray:
    # print(img_func.shape)
    hadamard_matrixn = hadamard(hadamarded_img.shape[0])
    hadamard_matrixm = hadamard(hadamarded_img.shape[1])
    inv_hadamard_matrixn = np.linalg.inv(hadamard_matrixn)
    inv_hadamard_matrixm = np.linalg.inv(hadamard_matrixm)
    ara_basamak = np.matmul(inv_hadamard_matrixn, hadamarded_img)
    return np.matmul(ara_basamak, inv_hadamard_matrixm)


def visualize_hadamard(hadamarded_img: np.ndarray) -> np.ndarray:
    magnitude = abs(hadamarded_img)
    magnitude = normalize(magnitude)
    return np.log(1 + magnitude)


def everything_hadamard(image_name: str):
    # get channel and test
    img = read_image(INPUT_PATH + image_name)
    original_shape = img.shape

    red_channel = get_rgb(img, 'R', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    # get fourier transformation of image
    fouriered_image = hadamart_from_image(red_channel)
    output = visualize_hadamard(fouriered_image)

    write_image(output, OUTPUT_PATH + "folded_hadamard_magnitude_red"+image_name)

    # reconstruct the image
    output = reverse_hadamard(fouriered_image)
    final_image = np.zeros((output.shape[0], output.shape[1], 3))
    final_image[:, :, 0] = output[:, :]
    write_image(output, OUTPUT_PATH + "hadamard_reconstructed_red_"+image_name)

    # get channel and test
    red_channel = get_rgb(img, 'G', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    # get fourier transformation of image
    fouriered_image = hadamart_from_image(red_channel)
    output = visualize_hadamard(fouriered_image)

    write_image(output, OUTPUT_PATH + "folded_hadamard_magnitude_green_"+image_name)

    # reconstruct the image
    output = reverse_hadamard(fouriered_image)
    final_image[:, :, 1] = output[:, :]
    write_image(output, OUTPUT_PATH + "hadamard_reconstructed_green_"+image_name)

    # get channel and test
    red_channel = get_rgb(img, 'B', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    # get fourier transformation of image
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


def visualize_cosine(img_func: np.ndarray):
    # show fourier transform
    # Magnitude spectrum
    magnitude = abs(img_func)
    # normalize
    magnitude = normalize(magnitude)
    # Kowalsky enhance >:(
    return np.log(1 + magnitude)


def everything_cosine(image_name: str):
    # get channel and test
    img = read_image(INPUT_PATH + image_name)
    final_image = np.zeros(img.shape)

    red_channel = get_rgb(img, 'R', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    # get fourier transformation of image
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

    # get channel and test
    red_channel = get_rgb(img, 'G', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    # get fourier transformation of image
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

    # get channel and test
    red_channel = get_rgb(img, 'B', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    # get fourier transformation of image
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
    matplotlib.image.imsave(output_path, output_file)


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
            if math.sqrt((x - centerx) * (x - centerx) + (y - centery) * (y - centery)) <= r:
                return_val[x][y] = 1
            else:
                return_val[x][y] = 0
    return return_val


def generate_gaussian_filter(size_x: int, size_y: int, sd: float, r: float) -> np.ndarray:
    ret_val = np.zeros((size_x, size_y))
    centerx = size_x / 2
    centery = size_y / 2
    for x in range(0, size_x):
        for y in range(0, size_y):
            tx = x-centerx
            ty = y-centery
            ret_val[x][y] = pow(np.e, -2 * np.pi * ((tx*tx + ty*ty) / (r * r)) * sd * sd)
    return ret_val


def generate_butterworth_filter(size_x: int, size_y: int, n: float, r: float) -> np.ndarray:
    ret_val = np.zeros((size_x, size_y))
    centerx = size_x / 2
    centery = size_y / 2
    for x in range(0, size_x):
        for y in range(0, size_y):
            tx = x - centerx
            ty = y - centery
            ret_val[x][y] = 1 / (1 + pow((tx*tx + ty*ty) / r, n))
    return ret_val


def apply_filter(img_func: np.ndarray, filter_type: str, pass_type: str, cut_off: float, optional: float = 2.0) -> np.ndarray:
    if filter_type == "I":
        fil = generate_ideal_filter(img_func.shape[0], img_func.shape[1], cut_off)
    elif filter_type == "G":
        fil = generate_gaussian_filter(img_func.shape[0], img_func.shape[1], optional, cut_off)
    elif filter_type == "B":
        fil = generate_butterworth_filter(img_func.shape[0], img_func.shape[1], optional, cut_off)
    else:
        fil = generate_ideal_filter(img_func.shape[0], img_func.shape[1], cut_off)
    if pass_type == "LP":
        return np.multiply(img_func, fil)
    if pass_type == "HP":
        return np.multiply(1 - fil, img_func)


def time_domain_to_frequency(image_func: np.ndarray, channel: str):
    rgb_channel = get_rgb(image_func, channel, False)
    transformed_image = get_fast_fourier(rgb_channel)
    return fold_image_to_center(transformed_image)


def frequency_domain_to_time_domain(image_func: np.ndarray) -> np.ndarray:
    return np.real(fp.ifftn(fold_image_to_center(image_func)))


def process_image(image_func: np.ndarray, filter_type: str, pass_type: str, cut_off: float, channel: str) -> np.ndarray:
    transformed_image = time_domain_to_frequency(image_func, channel)
    filtered_image = apply_filter(transformed_image, filter_type, pass_type, cut_off)
    return frequency_domain_to_time_domain(filtered_image)


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    # everything_fourier("1.png")  #takes fourier transformation of an image for every channel then reconstructs them and creates final image -> total 10 images

    # everything_cosine("1.png")   #takes cosine transformation of an image for every channel then reconstructs them and creates final image  -> total 10 images
   
    # everything_hadamard("1.png") #takes hadamard transformation of an image for every channel then reconstructs them and creates final image  -> total 10 images

    img = read_image(INPUT_PATH + "3.png")
    for fil in ("I", "G", "B",):
        for pas in ("LP", "HP"):
            for cut in (1, 5, 25):
                final_image = np.zeros(img.shape)
                final_image[:, :, 0] = normalize(process_image(img, fil, pas, cut, "R"))
                final_image[:, :, 1] = normalize(process_image(img, fil, pas, cut, "G"))
                final_image[:, :, 2] = normalize(process_image(img, fil, pas, cut, "B"))
                write_image(final_image, OUTPUT_PATH + f"{fil}{pas}_{cut}.png")

    """
    red_channel = get_rgb(img, 'R', False)
    # write_image(output, OUTPUT_PATH + "1_red.png")

    # get fourier transformation of image
    fouriered_image = get_fast_fourier(red_channel)
    output = visualize_fourier(fouriered_image)

    write_image(apply_filter(output, "ideal", "low", 50), OUTPUT_PATH + "ffmr_ideal_low"+"3.png")
    write_image(apply_filter(output, "ideal", "high", 50), OUTPUT_PATH + "ffmr_ideal_high" + "3.png")
    write_image(apply_filter(output, "gaussian", "low", 50), OUTPUT_PATH + "ffmr_gaussian_low" + "3.png")
    write_image(apply_filter(output, "gaussian", "high", 50), OUTPUT_PATH + "ffmr_gaussian_high" + "3.png")
    write_image(apply_filter(output, "butterworth", "low", 50), OUTPUT_PATH + "ffmr_butterworth_low" + "3.png")
    write_image(apply_filter(output, "butterworth", "high", 50), OUTPUT_PATH + "ffmr_butterworth_high" + "3.png")

    # reconstruct the image
    fouriered_image = fold_image_to_center(fouriered_image)
    write_image(np.real(fp.ifftn(fold_image_to_center(apply_filter(fouriered_image, "ideal", "low", 50)))),
                OUTPUT_PATH + "ffcr_ideal_low" + "3.png")
    write_image(np.real(fp.ifftn(fold_image_to_center(apply_filter(fouriered_image, "ideal", "high", 50)))),
                OUTPUT_PATH + "ffcr_ideal_high" + "3.png")
    write_image(np.real(fp.ifftn(fold_image_to_center(apply_filter(fouriered_image, "gaussian", "low", 50)))),
                OUTPUT_PATH + "ffcr_gaussian_low" + "3.png")
    write_image(np.real(fp.ifftn(fold_image_to_center(apply_filter(fouriered_image, "gaussian", "high", 50)))),
                OUTPUT_PATH + "ffcr_gaussian_high" + "3.png")
    write_image(np.real(fp.ifftn(fold_image_to_center(apply_filter(fouriered_image, "butterworth", "low", 50)))),
                OUTPUT_PATH + "ffcr_butterworth_low" + "3.png")
    write_image(np.real(fp.ifftn(fold_image_to_center(apply_filter(fouriered_image, "butterworth", "high", 50)))),
                OUTPUT_PATH + "ffcr_butterworth_high" + "3.png")

    output = np.real(fp.ifftn(fold_image_to_center(apply_filter(fouriered_image, "ideal", "low", 50))))

    final_image[:, :, 0] = output[:, :]"""

