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
    img_data: Image
    if rgb:
        img_data = Image.open(img_path)
    else:
        img_data = Image.open(img_path).convert('L')
    return np.asarray(img_data)


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


def extract_save_histogram(img_func: np.ndarray, path: str):
    histogram = extract_histogram(img_func)
    plt.stairs(histogram, range(257))
    plt.savefig(path)
    plt.clf()


def find_rotated_pos(x: float, y: float, o: tuple[float, float], degree: float = 0) -> tuple[float, float]:
    sin_val = math.sin((degree * math.pi) / 180)
    cos_val = math.cos((degree * math.pi) / 180)
    t_x = x - o[0]
    t_y = y - o[1]
    r_x = t_x * cos_val - t_y * sin_val
    r_y = t_x * sin_val + t_y * cos_val
    return r_x + o[0], r_y + o[1],


def rotate_image_linear(img_func: np.ndarray, degree: float = 0) -> np.ndarray:
    shape = img_func.shape
    originPoint = (float(shape[0]) / 2.0, float(shape[1]) / 2.0,)
    ret_val = np.zeros(shape)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            new_point = find_rotated_pos(float(x), float(y), originPoint, degree)
            if new_point[0] < 0 or new_point[0] > shape[0] - 1 or new_point[1] < 0 or new_point[1] > shape[1] - 1:
                for z in range(0, shape[2]):
                    ret_val[x][y][z] = 0
            else:
                x_floor = math.floor(new_point[0])
                y_floor = math.floor(new_point[1])
                x_ceil = math.ceil(new_point[0])
                y_ceil = math.ceil(new_point[1])
                if x_floor == x_ceil and y_floor == y_ceil:
                    for z in range(0, shape[2]):
                        ret_val[x][y][z] = img_func[x_floor][y_floor][z]
                elif x_floor == x_ceil:
                    for z in range(0, shape[2]):
                        ret_val[x][y][z] = numpy.uint(
                            float(img_func[x_floor][y_floor][z]) * (y_ceil - new_point[1]) +
                            float(img_func[x_floor][y_ceil][z]) * (new_point[1] - y_floor))
                elif y_floor == y_ceil:
                    for z in range(0, shape[2]):
                        ret_val[x][y][z] = numpy.uint(
                            float(img_func[x_floor][y_floor][z]) * (x_ceil - new_point[0]) +
                            float(img_func[x_ceil][y_floor][z]) * (new_point[0] - x_floor))
                else:
                    for z in range(0, shape[2]):
                        ret_val[x][y][z] = numpy.uint(
                            float(img_func[x_floor][y_floor][z]) * (x_ceil - new_point[0]) * (y_ceil - new_point[1]) +
                            float(img_func[x_floor][y_ceil][z]) * (x_ceil - new_point[0]) * (new_point[1] - y_floor) +
                            float(img_func[x_ceil][y_floor][z]) * (new_point[0] - x_floor) * (y_ceil - new_point[1]) +
                            float(img_func[x_ceil][y_ceil][z]) * (new_point[0] - x_floor) * (new_point[1] - y_floor))
    return ret_val


def rotate_image_cubic(img_func: np.ndarray, degree: float = 0) -> np.ndarray:
    shape = img_func.shape
    originPoint = (float(shape[0]) / 2.0, float(shape[1]) / 2.0,)
    ret_val = np.zeros(shape)
    m_left = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [-3, 3, -2, -1], [2, -2, 1, 1]])
    m_right = np.array([[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]])
    f_x = np.zeros(shape)
    f_y = np.zeros(shape)

    f_x[0] = img_func[1] / 2.0
    f_y[0][0] = img_func[0][1] / 2.0
    f_y[0][shape[1] - 1] = (- img_func[0][shape[1] - 2]) / 2.0
    for y in range(1, shape[1] - 1):
        f_y[0][y] = (img_func[0][y + 1] - img_func[0][y - 1]) / 2.0

    f_x[shape[0] - 1] = (- img_func[shape[0] - 1 - 1]) / 2.0
    f_y[shape[0] - 1][0] = img_func[shape[0] - 1][1] / 2.0
    f_y[shape[0] - 1][shape[1] - 1] = (- img_func[shape[0] - 1][shape[1] - 2]) / 2.0
    for y in range(1, shape[1] - 1):
        f_y[shape[0] - 1][y] = (img_func[shape[0] - 1][y + 1] - img_func[shape[0] - 1][y - 1]) / 2.0
    for x in range(1, shape[0] - 1):
        f_x[x] = (img_func[x + 1] - img_func[x - 1]) / 2.0

        f_y[x][0] = img_func[x][1] / 2.0
        f_y[x][shape[1] - 1] = (- img_func[x][shape[1] - 2]) / 2.0
        for y in range(1, shape[1] - 1):
            f_y[x][y] = (img_func[x][y + 1] - img_func[x][y - 1]) / 2.0

    f_xy = (f_x + f_y) / 2.0
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            new_point = find_rotated_pos(float(x), float(y), originPoint, degree)
            if 0 <= new_point[0] <= shape[0] - 1 and 0 <= new_point[1] <= shape[1] - 1:
                x_floor = math.floor(new_point[0])
                y_floor = math.floor(new_point[1])
                x_ceil = math.ceil(new_point[0])
                y_ceil = math.ceil(new_point[1])
                x_dist = new_point[0] - x_floor
                y_dist = new_point[1] - y_floor
                m_left_new = np.matmul(np.array([1, x_dist, x_dist ** 2, x_dist ** 3]), m_left)
                m_right_new = np.matmul(m_right, np.array([1, y_dist, y_dist ** 2, y_dist ** 3]).transpose())

                for z in range(0, shape[2]):
                    ret_val[x][y][z] = np.matmul(
                        np.matmul(m_left_new,
                                  np.array([[float(img_func[x_floor][y_floor][z]), float(img_func[x_floor][y_ceil][z]),
                                             f_y[x_floor][y_floor][z], f_y[x_floor][y_ceil][z]],
                                            [float(img_func[x_ceil][y_floor][z]), float(img_func[x_ceil][y_ceil][z]),
                                             f_y[x_ceil][y_floor][z], f_y[x_ceil][y_ceil][z]],
                                            [float(f_x[x_floor][y_floor][z]), float(f_x[x_floor][y_ceil][z]),
                                             f_xy[x_floor][y_floor][z], f_xy[x_floor][y_ceil][z]],
                                            [float(f_x[x_ceil][y_floor][z]), float(f_x[x_ceil][y_ceil][z]),
                                             f_xy[x_ceil][y_floor][z], f_xy[x_ceil][y_ceil][z]]])), m_right_new).max()

    min_val = ret_val.min()
    if min_val < 0.0:
        ret_val -= min_val
    max_val = ret_val.max()
    if max_val > 255.0:
        ret_val *= (255.0 / max_val)
    return ret_val


def rotate_image(img_func: np.ndarray, degree: float = 0, interpolation_type: str = "linear") -> np.ndarray:
    if interpolation_type == "linear":
        return rotate_image_linear(img_func, degree)
    else:
        return rotate_image_cubic(img_func, degree)


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


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    # PART1

    """img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 45, "linear")
    write_image(output, OUTPUT_PATH + "a1_45_linear.png")"""

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 45, "cubic")
    write_image(output, OUTPUT_PATH + "a1_45_cubic.png")

    """img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 90, "linear")
    write_image(output, OUTPUT_PATH + "a1_90_linear.png")"""

    img = read_image(INPUT_PATH + "a1.png")
    output = rotate_image(img, 90, "cubic")
    write_image(output, OUTPUT_PATH + "a1_90_cubic.png")

    """img = read_image(INPUT_PATH + "a2.png")
    output = rotate_image(img, 45, "linear")
    write_image(output, OUTPUT_PATH + "a2_45_linear.png")"""

    img = read_image(INPUT_PATH + "a2.png")
    output = rotate_image(img, 45, "cubic")
    write_image(output, OUTPUT_PATH + "a2_45_cubic.png")

    # PART2
    img = read_image(INPUT_PATH + "b1.png", rgb=False)
    extract_save_histogram(img, OUTPUT_PATH + "original_histogram.png")
    equalized = histogram_equalization(img)
    extract_save_histogram(equalized, OUTPUT_PATH + "equalized_histogram.png")
    write_image(equalized, OUTPUT_PATH + "enhanced_image.png")

    # BONUS
    # Define the following function
    # equalized = adaptive_histogram_equalization(img)
    # extract_save_histogram(equalized, OUTPUT_PATH + "adaptive_equalized_histogram.png")
    # write_image(output, OUTPUT_PATH + "adaptive_enhanced_image.png")"""
