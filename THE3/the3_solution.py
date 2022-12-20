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
COLOR_CONSTANTS = [(255, 255, 255,),
                   (255, 0, 0),
                   (0, 255, 0),
                   (0, 0, 255),
                   (255, 255, 0),
                   (255, 0, 255),
                   (0, 255, 255),
                   (125, 125, 125),
                   (125, 0, 0),
                   (0, 125, 0),
                   (0, 0, 125),
                   (125, 125, 0),
                   (125, 0, 125),
                   (0, 125, 125)]


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
            if ((int(cell[0]), int(cell[1]), int(cell[2])) not in ret_val) and (int(cell[0]) != 0 or int(cell[1]) != 0 or int(cell[2]) != 0):
                ret_val.append((int(cell[0]), int(cell[1]), int(cell[2])))
    return ret_val


def get_distance(color1: tuple[int, int, int], color2: tuple[int, int, int]) -> float:
    return math.sqrt(float(pow(color1[0] - color2[0], 2) +
                           pow(color1[1] - color2[1], 2) +
                           pow(color1[2] - color2[2], 2)))


def get_distance_vector(color_space: list[tuple[int, int, int]], color: tuple[int, int, int]) -> np.ndarray:
    ret_val = np.zeros(len(color_space))
    for i in range(len(color_space)):
        ret_val[i] = get_distance(color_space[i], color)
    return ret_val


def get_average_color(colors: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    col = np.zeros(3)
    for color in colors:
        col = col + color
    col = col / len(colors)
    return int(col[0]), int(col[1]), int(col[2]),


def get_initial_means(color_space: list[tuple[int, int, int]], k: int) -> list[tuple[int, int, int]]:
    ret_val = []
    max_val = len(color_space) - 1
    for _ in range(k):
        ret_val.append(color_space[np.random.randint(0, max_val)])
    return ret_val


def get_next_means(color_space: list[tuple[int, int, int]],
                   means: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    dist_vects = []
    groups: list[list[tuple[int, int, int]]] = []
    ret_val = []
    for mean in means:
        dist_vects.append(get_distance_vector(color_space, mean))
        groups.append([])
    arr = np.array(dist_vects)
    for i in range(len(color_space)):
        groups[arr[:, i].argmin()].append(color_space[i])
    for group in groups:
        ret_val.append(get_average_color(group))
    return ret_val


def check_conversion(means1: list[tuple[int, int, int]], means2: list[tuple[int, int, int]]) -> bool:
    for x in range(len(means1)):
        if means1[x] not in means2:
            return False
    return True


def perform_k_means(color_space: list[tuple[int, int, int]], k: int) -> list[tuple[int, int, int]]:
    means = get_initial_means(color_space, k)
    next_means = get_next_means(color_space, means)
    count = 0
    while (not check_conversion(means, next_means)) and count < 1000:
        means = next_means
        next_means = get_next_means(color_space, means)
        count += 1
    return means


def group_by_mean(color_space: list[tuple[int, int, int]],
                  means: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    dist_vects = []
    groups: list[list[tuple[int, int, int]]] = []
    for mean in means:
        dist_vects.append(get_distance_vector(color_space, mean))
        groups.append([])
    arr = np.array(dist_vects)
    for i in range(len(color_space)):
        groups[arr[:, i].argmin()].append(color_space[i])
    return groups


def color_image_by_groups(img_func: np.ndarray, groups: list[list[tuple[int, int, int]]]) -> np.ndarray:
    ret_val = np.zeros(img_func.shape)
    for x in range(img_func.shape[0]):
        for y in range(img_func.shape[1]):
            color = (int(img_func[x][y][0]), int(img_func[x][y][1]), int(img_func[x][y][2]),)
            for i in range(len(groups)):
                if color in groups[i]:
                    ret_val[x][y][0] = COLOR_CONSTANTS[i][0]
                    ret_val[x][y][1] = COLOR_CONSTANTS[i][1]
                    ret_val[x][y][2] = COLOR_CONSTANTS[i][2]
                    if img_func.shape[2] == 4:
                        ret_val[x][y][3] = 255
                    break
    return ret_val


def down_sample_image(img_func: np.ndarray, x_fact: int, y_fact: int) -> np.ndarray:
    x_limit = img_func.shape[0] // x_fact
    y_limit = img_func.shape[1] // y_fact
    ret_val = np.zeros((x_limit, y_limit, 3), int)
    for x in range(x_limit):
        for y in range(y_limit):
            colors = []
            for xk in range(x_fact):
                for yk in range(y_fact):
                    cell = img_func[x * x_fact + xk][y * y_fact + yk]
                    colors.append((int(cell[0]), int(cell[1]), int(cell[2])))
            color = get_average_color(colors)
            ret_val[x][y][0] = color[0]
            ret_val[x][y][1] = color[1]
            ret_val[x][y][2] = color[2]
    return ret_val


def get_rgb_pixels(img_func: np.ndarray) -> np.ndarray:
    ret_val = np.zeros(img_func.shape)
    for x in range(img_func.shape[0]):
        for y in range(img_func.shape[1]):
            cell = img_func[x][y]
            if cell[0] > cell[1] > cell[2]:
                ret_val[x][y] = cell
    return ret_val


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    img = read_image(INPUT_PATH + "1_source.png")
    img = get_rgb_pixels(img)
    img = img / 2
    col_space = extract_color_space(img)
    for k_val in range(4, 15):
        img_means = perform_k_means(col_space, k_val)
        img_groups = group_by_mean(col_space, img_means)
        output_image = color_image_by_groups(img, img_groups)
        write_image(output_image, OUTPUT_PATH + f"1_faces_{k_val}.png")

    img = read_image(INPUT_PATH + "2_source.png")
    image_sampled = down_sample_image(img, 9, 9)
    image_sampled = get_rgb_pixels(image_sampled)
    image_sampled = image_sampled / 2
    col_space = extract_color_space(image_sampled)
    for k_val in range(4, 15):
        img_means = perform_k_means(col_space, k_val)
        img_groups = group_by_mean(col_space, img_means)
        output_image = color_image_by_groups(image_sampled, img_groups)
        write_image(output_image, OUTPUT_PATH + f"2_faces_{k_val}.png")

    img = read_image(INPUT_PATH + "3_source.png")
    img = get_rgb_pixels(img)
    img = img / 2
    col_space = extract_color_space(img)
    for k_val in range(4, 15):
        img_means = perform_k_means(col_space, k_val)
        img_groups = group_by_mean(col_space, img_means)
        output_image = color_image_by_groups(img, img_groups)
        write_image(output_image, OUTPUT_PATH + f"3_faces_{k_val}.png")
