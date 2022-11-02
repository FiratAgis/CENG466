import os

import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
from PIL import Image
import math

INPUT_PATH = "./THE1_Images/"
OUTPUT_PATH = "./Outputs/"


def read_image(img_path: str, rgb: bool = True) -> np.ndarray:
    img_data = Image.open(img_path)
    return np.asarray(img_data)


def write_image(img_func: np.ndarray, output_path: str, rgb: bool = True) -> None:
    """    if rgb:
        img_data = Image.fromarray(img_func)
    else:
        img_data = Image.fromarray(img_func)"""

    shape = img_func.shape
    output_file = np.zeros(shape)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            for z in range(0, shape[2]):
                output_file[x][y][z] = float(img_func[x][y][z]) / 255
    matplotlib.image.imsave(output_path, output_file)


def extract_save_histogram(img_func: np.ndarray, path: str):
    pass


def find_rotated_pos(x: float, y: float, o: tuple[float, float], degree: float = 0) -> tuple[float, float]:
    sin_val = math.sin((degree * math.pi) / 180)
    cos_val = math.cos((degree * math.pi) / 180)
    t_x = x - o[0]
    t_y = y - o[1]
    r_x = t_x * cos_val - t_y * sin_val
    r_y = t_x * sin_val + t_y * cos_val
    return (r_x + o[0], r_y + o[1], )


def rotate_image_linear(img_func: np.ndarray, degree: float = 0) -> np.ndarray:
    shape = img_func.shape
    originPoint = (float(shape[0]) / 2.0, float(shape[1]) / 2.0,)
    ret_val = np.zeros(shape)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            new_point = find_rotated_pos(float(x), float(y), originPoint, degree)
            # print(f"original point -> {x}, {y}, new point -> {new_point[0]}, {new_point[1]}")
            if new_point[0] < 0 or new_point[0] > shape[0] - 1 or new_point[1] < 0 or new_point[1] > shape[1] - 1:
                for z in range(0, shape[2]):
                    ret_val[x][y][z] = 0
            else:
                for z in range(0, shape[2]):
                    ret_val[x][y][z] = numpy.uint(
                        (int(img_func[math.floor(new_point[0])][math.floor(new_point[1])][z]) +
                         int(img_func[math.ceil(new_point[0])][math.floor(new_point[1])][z]) +
                         int(img_func[math.floor(new_point[0])][math.ceil(new_point[1])][z]) +
                         int(img_func[math.ceil(new_point[0])][math.ceil(new_point[1])][z])) / 4)
    return ret_val


def rotate_image_cubic(img_func: np.ndarray, degree: float = 0) -> np.ndarray:
    shape = img_func.shape
    originPoint = (float(shape[0]) / 2.0, float(shape[1]) / 2.0,)
    ret_val = np.zeros(shape)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            new_point = find_rotated_pos(float(x), float(y), originPoint, degree)
            for z in range(0, shape[2]):
                pass
    pass


def rotate_image(img_func: np.ndarray, degree: float = 0, interpolation_type: str = "linear") -> np.ndarray:
    if interpolation_type == "linear":
        return rotate_image_linear(img_func, degree)
    else:
        return rotate_image_cubic(img_func, degree)


"""def histogram_equalization(img_func: np.array):
    return img_hist_eq"""

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    # PART1
    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 45, "linear")
    write_image(output, OUTPUT_PATH + "a1_45_linear.png")

    """
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