"""
Submission by
Fırat Ağış, e2236867
Robin Koç, e246871
"""

import os
import numpy as np
import matplotlib.image
from PIL import Image
import math
import scipy.fftpack as fp
from scipy.linalg import hadamard

INPUT_PATH = "./THE_2 images/"
OUTPUT_PATH = "./Outputs/"
R1 = 10
R2 = 20
R3 = 50
POW = 3


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


def hadamart_from_image(img_func: np.ndarray) -> np.ndarray:
    img_func = zero_padding_to_square(img_func)
    hadamard_matrix_n = hadamard(img_func.shape[0])
    hadamard_matrix_m = hadamard(img_func.shape[1])
    ara_basamak = np.matmul(hadamard_matrix_n, img_func)
    return np.matmul(ara_basamak, hadamard_matrix_m)


def visualize_fourier(img_func: np.ndarray) -> np.ndarray:
    # show fourier transform
    # Magnitude spectrum
    magnitude = abs(img_func)
    # normalize
    magnitude = normalize(magnitude)
    # Kowalsky enhance >:(
    ret_val = fold_image_to_center(magnitude)
    return np.log(1 + ret_val)


def fold_image_to_center(img_func: np.ndarray):
    shape = img_func.shape
    # flip upper_left
    img_func[0:(shape[0]//2), 0:(shape[1]//2)] = np.flip(img_func[0:(shape[0]//2), 0:(shape[1]//2)], (0, 1))
    # flip upper_right
    img_func[(shape[0]//2):, 0:(shape[1]//2)] = np.flip(img_func[(shape[0]//2):, 0:(shape[1]//2)], (0, 1))
    # flip bottom_left
    img_func[0:(shape[0]//2), (shape[1]//2):] = np.flip(img_func[0:(shape[0]//2), (shape[1]//2):], (0, 1))
    # flip bottom_right
    img_func[(shape[0]//2):, (shape[1]//2):] = np.flip(img_func[(shape[0]//2):, (shape[1]//2):], (0, 1))
    return img_func


def zero_padding_to_square(img_func: np.ndarray):
    shape = img_func.shape

    maxh = shape[0]
    i = 2
    while maxh > i:
        i = i*2
    maxh = i
    if maxh > shape[0]+1000:
        maxh = maxh//2

    max_v = shape[1]
    i = 2
    while max_v > i:
        i = i*2
    max_v = i
    if max_v > shape[1]+1000:
        max_v = max_v//2
    
    padded_img = np.zeros((maxh, max_v))
    for i in range(0, maxh):
        for j in range(0, max_v):
            if i < shape[0] and j < shape[1]:
                padded_img[i][j] = img_func[i][j]
            else:
                padded_img[i][j] = 0
    
    return padded_img


def hadamard_from_image(img_func: np.ndarray) -> np.ndarray:
    img_func = zero_padding_to_square(img_func)
    hadamard_matrix_n = hadamard(img_func.shape[0])
    hadamard_matrix_m = hadamard(img_func.shape[1])
    ara_basamak = np.matmul(hadamard_matrix_n, img_func)
    return np.matmul(ara_basamak, hadamard_matrix_m)


def reverse_hadamard(hadamarded_img: np.ndarray) -> np.ndarray:
    # print(img_func.shape)
    hadamard_matrix_n = hadamard(hadamarded_img.shape[0])
    hadamard_matrix_m = hadamard(hadamarded_img.shape[1])
    inv_hadamard_matrix_n = np.linalg.inv(hadamard_matrix_n)
    inv_hadamard_matrix_m = np.linalg.inv(hadamard_matrix_m)
    ara_basamak = np.matmul(inv_hadamard_matrix_n, hadamarded_img)
    return np.matmul(ara_basamak, inv_hadamard_matrix_m)


def visualize_hadamard(hadamarded_img: np.ndarray) -> np.ndarray:
    magnitude = abs(hadamarded_img)
    magnitude = normalize(magnitude)
    return np.log(1 + magnitude)


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
    center_x = size_x/2
    center_y = size_y/2
    for x in range(0, size_x):
        for y in range(0, size_y):
            if math.sqrt((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)) <= r:
                return_val[x][y] = 1
            else:
                return_val[x][y] = 0
    return return_val


def generate_gaussian_filter(size_x: int, size_y: int, sd: float, r: float) -> np.ndarray:
    ret_val = np.zeros((size_x, size_y))
    center_x = size_x / 2
    center_y = size_y / 2
    for x in range(0, size_x):
        for y in range(0, size_y):
            tx = x-center_x
            ty = y-center_y
            ret_val[x][y] = pow(np.e, -2 * np.pi * ((tx*tx + ty*ty) / (r * r)) * sd * sd)
    return ret_val


def generate_butterworth_filter(size_x: int, size_y: int, n: float, r: float) -> np.ndarray:
    ret_val = np.zeros((size_x, size_y))
    center_x = size_x / 2
    center_y = size_y / 2
    for x in range(0, size_x):
        for y in range(0, size_y):
            tx = x - center_x
            ty = y - center_y
            ret_val[x][y] = 1 / (1 + pow((tx*tx + ty*ty) / r, n))
    return ret_val


def apply_filter(img_func: np.ndarray,
                 filter_type: str,
                 pass_type: str,
                 cut_off: float,
                 optional: float = 1.0) -> np.ndarray:
    if filter_type == "I":
        filter_mask = generate_ideal_filter(img_func.shape[0], img_func.shape[1], cut_off)
    elif filter_type == "G":
        filter_mask = generate_gaussian_filter(img_func.shape[0], img_func.shape[1], optional, cut_off)
    elif filter_type == "B":
        filter_mask = generate_butterworth_filter(img_func.shape[0], img_func.shape[1], optional, cut_off)
    else:
        filter_mask = generate_ideal_filter(img_func.shape[0], img_func.shape[1], cut_off)
    if pass_type == "LP":
        return np.multiply(img_func, filter_mask)
    if pass_type == "HP":
        return np.multiply(1 - filter_mask, img_func)


def time_domain_to_frequency(image_func: np.ndarray, channel: str):
    rgb_channel = get_rgb(image_func, channel, False)
    transformed_image = get_fast_fourier(rgb_channel)
    return fold_image_to_center(transformed_image)


def frequency_domain_to_time_domain(image_func: np.ndarray) -> np.ndarray:
    return np.real(fp.ifftn(fold_image_to_center(image_func)))


def process_image(image_func: np.ndarray,
                  filter_type: str,
                  pass_type: str,
                  cut_off: float,
                  channel: str,
                  optional: float = 1.0) -> np.ndarray:
    transformed_image = time_domain_to_frequency(image_func, channel)
    filtered_image = apply_filter(transformed_image, filter_type, pass_type, cut_off, optional)
    return frequency_domain_to_time_domain(filtered_image)


def power_transformation(image_func: np.ndarray, c: float, t: float):
    ret_val = np.zeros(image_func.shape)
    for x in range(image_func.shape[0]):
        for y in range(image_func.shape[1]):
            for z in range(image_func.shape[2]):
                ret_val[x][y][z] = c * pow(image_func[x][y][z], t)
    return ret_val


def power_transformation_by_channel(image_func: np.ndarray, c: float, t: float, channel: str):
    image_func_actual = get_rgb(image_func, channel, False)
    ret_val = np.zeros(image_func_actual.shape)
    for x in range(image_func_actual.shape[0]):
        for y in range(image_func_actual.shape[1]):
            ret_val[x][y] = c * pow(image_func_actual[x][y], t)
    return ret_val


def extract_histogram(img_func: np.ndarray) -> np.ndarray:
    ret_val = np.zeros(256)
    for x in img_func:
        for y in x:
            ret_val[int(y)] = ret_val[int(y)] + 1
    return ret_val


def create_cum_histogram(histogram: np.ndarray) -> np.ndarray:
    ret_val = np.zeros(256)
    ret_val[0] = histogram[0]
    for x in range(1, 256):
        ret_val[x] = ret_val[x - 1] + histogram[x]
    return ret_val


def histogram_equalization(img_func: np.array):
    shape = img_func.shape
    img_hist_eq = np.zeros(shape, dtype=np.uint)
    histogram = extract_histogram(img_func)
    cum = create_cum_histogram(histogram)
    size = float(shape[0] * shape[1])
    coefficient = 255.0 / size
    for x in range(shape[0]):
        for y in range(shape[1]):
            img_hist_eq[x][y] = math.floor(coefficient * cum[img_func[x][y]])
    return img_hist_eq


def generate_band_pass_filter(size_x: int, size_y: int, r_s: float, r_l: float) -> np.ndarray:
    return_val = np.zeros((size_x, size_y))
    center_x = size_x/2
    center_y = size_y/2
    for x in range(0, size_x):
        for y in range(0, size_y):
            if ((math.sqrt((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)) <= r_l) and
                    (math.sqrt((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)) >= r_s)):
                return_val[x][y] = 1
            else:
                return_val[x][y] = 0
    return return_val


def generate_band_reject_filter(size_x: int, size_y: int, r_s: float, r_l: float) -> np.ndarray:
    return_val = np.zeros((size_x, size_y))
    center_x = size_x/2
    center_y = size_y/2
    for x in range(0, size_x):
        for y in range(0, size_y):
            if ((math.sqrt((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)) <= r_l) and
                    (math.sqrt((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)) >= r_s)):
                return_val[x][y] = 0
            else:
                return_val[x][y] = 1
    return return_val


def band_reject(image_func: np.ndarray, cut_off1: float, cut_off2: float, channel: str) -> np.ndarray:
    transformed_image = time_domain_to_frequency(image_func, channel)
    band_fil = generate_band_reject_filter(transformed_image.shape[0], transformed_image.shape[1], cut_off1, cut_off2)
    filtered_image = np.multiply(transformed_image, band_fil)  # HP takes the outside LP takes the inside
    return frequency_domain_to_time_domain(filtered_image)


def band_pass(image_func: np.ndarray, cut_off1: float, cut_off2: float, channel: str) -> np.ndarray:
    transformed_image = time_domain_to_frequency(image_func, channel)
    band_fil = generate_band_pass_filter(transformed_image.shape[0], transformed_image.shape[1], cut_off1, cut_off2)
    filtered_image = np.multiply(transformed_image, band_fil)  # HP takes the outside LP takes the inside
    return frequency_domain_to_time_domain(filtered_image)


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Fourier Transform of 1.png
    img = read_image(INPUT_PATH + "1.png")
    fouriered_image = np.log(np.abs(fp.fftshift(get_fast_fourier(img))) ** 2)
    write_image(normalize(fouriered_image), OUTPUT_PATH + "F1.png")

    # Fourier Transform of 2.png
    img = read_image(INPUT_PATH + "2.png")
    fouriered_image = np.log(np.abs(fp.fftshift(get_fast_fourier(img))) ** 2)
    write_image(normalize(fouriered_image), OUTPUT_PATH + "F2.png")

    # Hadamard Transform of 1.png
    img = read_image(INPUT_PATH + "1.png")
    original_shape = img.shape
    red_channel = get_rgb(img, 'R', False)
    hadamarded_image = hadamart_from_image(red_channel)
    final_image = np.zeros((hadamarded_image.shape[0], hadamarded_image.shape[1], 3))
    final_image[:, :, 0] = visualize_hadamard(hadamarded_image)[:, :]

    green_channel = get_rgb(img, 'G', False)
    hadamarded_image = hadamart_from_image(green_channel)
    final_image[:, :, 1] = visualize_hadamard(hadamarded_image)[:, :]

    blue_channel = get_rgb(img, 'B', False)
    hadamarded_image = hadamart_from_image(blue_channel)
    final_image[:, :, 2] = visualize_hadamard(hadamarded_image)[:, :]

    final_image = normalize(final_image)
    final_image = final_image[0:original_shape[0], 0:original_shape[1], :]
    write_image(final_image, OUTPUT_PATH + "H1.png")

    # Hadamard Transform of 2.png
    img = read_image(INPUT_PATH + "2.png")
    original_shape = img.shape
    red_channel = get_rgb(img, 'R', False)
    hadamarded_image = hadamart_from_image(red_channel)
    final_image = np.zeros((hadamarded_image.shape[0], hadamarded_image.shape[1], 3))
    final_image[:, :, 0] = visualize_hadamard(hadamarded_image)[:, :]

    green_channel = get_rgb(img, 'G', False)
    hadamarded_image = hadamart_from_image(green_channel)
    final_image[:, :, 1] = visualize_hadamard(hadamarded_image)[:, :]

    blue_channel = get_rgb(img, 'B', False)
    hadamarded_image = hadamart_from_image(blue_channel)
    final_image[:, :, 2] = visualize_hadamard(hadamarded_image)[:, :]

    final_image = normalize(final_image)
    final_image = final_image[0:original_shape[0], 0:original_shape[1], :]
    write_image(final_image, OUTPUT_PATH + "H2.png")

    # Cosine Transform of 1.png
    img = read_image(INPUT_PATH + "1.png")
    cosined_image = fp.dct(fp.dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')
    cosined_image = cosined_image[:200, 0:200]
    write_image(normalize(cosined_image), OUTPUT_PATH + "C1.png")

    # Cosine Transform of 2.png
    img = read_image(INPUT_PATH + "2.png")
    cosined_image = fp.dct(fp.dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')
    cosined_image = cosined_image[:200, 0:200]
    write_image(normalize(cosined_image), OUTPUT_PATH + "C2.png")

    # High and Low Pass Filters for 3.png
    img = read_image(INPUT_PATH + "3.png")
    for fil in ("I", "G", "B",):
        for pas in ("LP", "HP"):
            for cut in (R1, R2, R3,):
                final_image = np.zeros(img.shape)
                final_image[:, :, 0] = normalize(process_image(img, fil, pas, cut, "R"))
                final_image[:, :, 1] = normalize(process_image(img, fil, pas, cut, "G"))
                final_image[:, :, 2] = normalize(process_image(img, fil, pas, cut, "B"))
                write_image(final_image, OUTPUT_PATH + f"{fil}{pas}_{cut}.png")

    # Band Filtering for 4.png
    img = read_image(INPUT_PATH + "4.png")
    final_img = np.zeros(img.shape)

    final_img[:, :, 0] = normalize(band_reject(img, 80, 200, "R"))
    final_img[:, :, 1] = normalize(band_reject(img, 80, 200, "G"))
    final_img[:, :, 2] = normalize(band_reject(img, 80, 200, "B"))

    write_image(final_img, OUTPUT_PATH + "BR1.png")
    final_img = np.zeros(img.shape)
    final_img[:, :, 0] = normalize(band_pass(img, 80, 200, "R"))
    final_img[:, :, 1] = normalize(band_pass(img, 80, 200, "G"))
    final_img[:, :, 2] = normalize(band_pass(img, 80, 200, "B"))

    write_image(final_img, OUTPUT_PATH + "BP1.png")

    # Band Filtering for 5.png
    img = read_image(INPUT_PATH + "5.png")
    final_img = np.zeros(img.shape)
    final_img[:, :, 0] = normalize(band_reject(img, 85, 1500, "R"))
    final_img[:, :, 1] = normalize(band_reject(img, 85, 1500, "G"))
    final_img[:, :, 2] = normalize(band_reject(img, 85, 1500, "B"))

    write_image(final_img, OUTPUT_PATH + "BR2.png")
    final_img = np.zeros(img.shape)

    final_img[:, :, 0] = normalize(band_pass(img, 85, 1500, "R"))
    final_img[:, :, 1] = normalize(band_pass(img, 85, 1500, "G"))
    final_img[:, :, 2] = normalize(band_pass(img, 85, 1500, "B"))

    write_image(final_img, OUTPUT_PATH + "BP2.png")

    # Image Enhancement for 6.png
    img = read_image(INPUT_PATH + "6.png")
    final_image = np.zeros(img.shape)
    final_image[:, :, 0] = normalize(histogram_equalization(get_rgb(img, 'R', False)))
    final_image[:, :, 1] = normalize(histogram_equalization(get_rgb(img, 'G', False)))
    final_image[:, :, 2] = normalize(histogram_equalization(get_rgb(img, 'B', False)))
    final_image[:, :, 0] = normalize(power_transformation_by_channel(final_image, 1, pow(1.1, POW), 'R'))
    final_image[:, :, 1] = normalize(power_transformation_by_channel(final_image, 1, pow(1.2, POW), 'G'))
    final_image = normalize(power_transformation(final_image, 1, pow(0.9, POW)))
    write_image(final_image, OUTPUT_PATH + f"Space6.png")

    # Image Enhancement for 7.png
    img = read_image(INPUT_PATH + "7.png")
    final_image = normalize(power_transformation(img, 1, pow(0.9, POW)))
    write_image(final_image, OUTPUT_PATH + f"Space7.png")
