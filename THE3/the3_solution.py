"""
Submission by
Fırat Ağış, e2236867
Robin Koç, e246871
"""
import math
import os
import numpy as np
import matplotlib.image
from PIL import Image


INPUT_PATH = "./THE3_Images/"
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
    matplotlib.image.imsave(output_path, output_file)


def normalize(arr: np.ndarray) -> np.ndarray:
    max_val = arr.max()
    min_val = arr.min()
    return (arr - min_val) * (254 / (max_val - min_val))


def extract_color_space(img_func: np.ndarray) -> list[tuple[int, int, int]]:
    ret_val = []
    for row in img_func:
        for cell in row:
            if (cell[0], cell[1], cell[2]) not in ret_val:
                ret_val.append((cell[0], cell[1], cell[2]))
    return ret_val


def get_distance(color1: tuple[int, int, int], color2: tuple[int, int, int]) -> float:
    return math.sqrt(pow(float(color1[0] - color2[0]), 2) +
                     pow(float(color1[1] - color2[1]), 2) +
                     pow(float(color1[2] - color2[2]), 2))


def get_distance_vector(color_space: list[tuple[int, int, int]], color: tuple[int, int, int]) -> np.ndarray:
    ret_val = np.zeros(len(color_space))
    for i in range(len(color_space)):
        ret_val[i] = get_distance(color_space[i], color)
    return ret_val


def get_average_color(colors: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    r = 0
    g = 0
    b = 0
    for color in colors:
        r += color[0]
        g += color[1]
        b += color[2]
    r = r//len(colors)
    g = g // len(colors)
    b = b // len(colors)
    return r, g, b,


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
