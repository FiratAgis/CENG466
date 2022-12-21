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
            if (int(cell[0]), int(cell[1]), int(cell[2])) not in ret_val:
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


def color_image_by_groups(img_func: np.ndarray, groups: list[list[tuple[int, int, int]]],
                  black_non_rgb: bool = False) -> np.ndarray:
    ret_val = np.zeros(img_func.shape)
    for x in range(img_func.shape[0]):
        for y in range(img_func.shape[1]):
            color = (int(img_func[x][y][0]), int(img_func[x][y][1]), int(img_func[x][y][2]),)
            if img_func.shape[2] == 4:
                ret_val[x][y][3] = 255
            if black_non_rgb and not (color[0] > color[1] and color[0] > color[2]):
                ret_val[x][y][0] = 0
                ret_val[x][y][1] = 0
                ret_val[x][y][2] = 0
            else:
                for i in range(len(groups)):
                    if color in groups[i]:
                        ret_val[x][y][0] = COLOR_CONSTANTS[i][0]
                        ret_val[x][y][1] = COLOR_CONSTANTS[i][1]
                        ret_val[x][y][2] = COLOR_CONSTANTS[i][2]
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


def histogram_equalization(img_func: np.ndarray) -> np.ndarray:
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


def get_equal_image(img_func: np.ndarray) -> np.ndarray:
    final_image = np.zeros(img_func.shape)
    final_image[:, :, 0] = normalize(histogram_equalization(get_rgb(img_func, 'R', False)))
    final_image[:, :, 1] = normalize(histogram_equalization(get_rgb(img_func, 'G', False)))
    final_image[:, :, 2] = normalize(histogram_equalization(get_rgb(img_func, 'B', False)))
    if img_func.shape[2] == 4:
        final_image[:, :, 3] = 255
    return  final_image


def erode_image(img_func: np.ndarray, degree: int = 1) -> np.ndarray:
    if degree == 0:
        return img_func
    final_image = np.zeros(img_func.shape)
    for x in range(1, img_func.shape[0] - 1):
        for y in range(1, img_func.shape[1] -1):
            if (img_func[x-1][y-1][0] > 40 and img_func[x][y-1][0] > 40 and img_func[x+1][y-1][0] > 40 and
                img_func[x-1][y][0] > 40 and img_func[x][y][0] > 40 and img_func[x+1][y][0] > 40 and
                img_func[x-1][y+1][0] > 40 and img_func[x][y+1][0] > 40 and img_func[x+1][y+1][0] > 40):
                final_image[x][y][0] = 255
                final_image[x][y][1] = 255
                final_image[x][y][2] = 255
            if img_func.shape[2] == 4:
                final_image[x][y][3] = 255
    return erode_image(final_image, degree - 1)

def dilute_image(img_func: np.ndarray, degree: int = 1) -> np.ndarray:
    if degree == 0:
        return img_func
    final_image = np.zeros(img_func.shape)
    for x in range(1, img_func.shape[0] - 1):
        for y in range(1, img_func.shape[1] -1):
            if (img_func[x-1][y-1][0] > 40 or img_func[x][y-1][0] > 40 or img_func[x+1][y-1][0] > 40 or
                img_func[x-1][y][0] > 40 or img_func[x][y][0] > 40 or img_func[x+1][y][0] > 40 or
                img_func[x-1][y+1][0] > 40 or img_func[x][y+1][0] > 40 or img_func[x+1][y+1][0] > 40):
                final_image[x][y][0] = 255
                final_image[x][y][1] = 255
                final_image[x][y][2] = 255
            if img_func.shape[2] == 4:
                final_image[x][y][3] = 255
    return dilute_image(final_image, degree - 1)


def open_image(img_func: np.ndarray, degree: int = 1):
    ret_val = erode_image(img_func, degree)
    ret_val = dilute_image(ret_val, degree)
    return  ret_val


def close_image(img_func: np.ndarray, degree: int = 1):
    ret_val = dilute_image(img_func, degree)
    ret_val = erode_image(ret_val, degree)
    return  ret_val


def frame_cull(img_func: np.ndarray, factor_x: float, factor_y: float) -> np.ndarray:
    ret_val = np.array(img_func)
    for x in range(img_func.shape[0]):
        for y in range(img_func.shape[1]):
            if (x < (img_func.shape[0] * factor_x) or
                x > (img_func.shape[0] * (1- factor_x)) or
                y < (img_func.shape[1] * factor_y) or
                y > (img_func.shape[1] * (1-factor_y))):
                ret_val[x][y][0] = 0
                ret_val[x][y][1] = 0
                ret_val[x][y][2] = 0
    return ret_val


def convolve_image(img_func: np.ndarray, mask: np.ndarray, padding_x: int  = 1, padding_y: int = 1) -> np.ndarray:
    mask = np.flipud(np.fliplr(mask))

    mask_shape_x = mask.shape[0]
    mask_shape_y = mask.shape[1]
    img_shape_x = img_func.shape[0]
    img_shape_y = img_func.shape[1]

    output = np.zeros((int((img_shape_x - mask_shape_x + 2 * padding_x)), int((img_shape_y - mask_shape_y + 2 * padding_y))))

    imagePadded = np.zeros((img_shape_x + padding_x * 2, img_shape_y + padding_y * 2))
    imagePadded[int(padding_x):int(-1 * padding_x), int(padding_y):int(-1 * padding_y)] = img_func

    for y in range(img_shape_y):
        if y > img_shape_y - mask_shape_y:
            break
        for x in range(img_shape_x):
            if x > img_shape_x - mask_shape_x:
                break
            try:
                output[x, y] = (mask * imagePadded[x: x + mask_shape_x, y: y + mask_shape_y]).sum()
            except:
                break

    return output


def binarize_image(img_func: np.ndarray) -> np.ndarray:
    ret_val = np.zeros((img_func.shape[0], img_func.shape[1]))
    for x in range(img_func.shape[0]):
        for y in range(img_func.shape[1]):
            if img_func[x][y][0] > 40:
                ret_val[x][y] = 1
    return ret_val


def get_ellipse_mask(x_length: int, y_length: int) -> np.ndarray:
    ret_val = np.zeros((x_length * 2, y_length *2 ))
    for x in range(x_length * 2):
        for y in range(y_length * 2):
            if (pow(x-x_length, 2)/pow(x_length,2)) + (pow(y-y_length, 2)/pow(y_length,2)) <= 1:
                ret_val[x][y] = 1
    return ret_val


def apply_convolution_result(img_func: np.ndarray, x_length: int, y_length: int, dim: int) -> np.ndarray:
    ret_val = np.zeros((img_func.shape[0], img_func.shape[1], dim))
    cut_off = x_length * y_length
    for x in range(img_func.shape[0]):
        for y in range(img_func.shape[1]):
            if img_func[x][y] >= cut_off:
                for x1 in range(img_func.shape[0]):
                    for y1 in range(img_func.shape[1]):
                        if ret_val[x1][y1][0] == 0:
                            if abs(x1 - x) < x_length and abs(y1 - y) < y_length:
                                if (pow(x1-x, 2)/pow(x_length,2)) + (pow(y1-y, 2)/pow(y_length,2)) <= 1:
                                    for index in range(dim):
                                        ret_val[x1][y1][index] = 1
    return ret_val

def detect_faces(input_name: str,
                 output_name: str,
                 means_no: int,
                 equalize: bool = True,
                 factor_x: int = 1,
                 factor_y: int = 1,
                 bit_slice: int = 1,
                 red_mask: bool = True,
                 cull_factor_x: float = 0.1,
                 cull_factor_y: float = 0.1,
                 open_amount: int = 2,
                 close_amount: int = 2,
                 mask_x: int = 20,
                 mask_y: int = 15,
                 print_intermediate: bool = False):
    img_func = read_image(f"{INPUT_PATH}{input_name}.png")
    if equalize:
        img_func = get_equal_image(img_func)
        if print_intermediate:
            write_image(img_func, f"{OUTPUT_PATH}{output_name}_equalized.png")
    if factor_x > 1 or factor_y > 1:
        img_func = down_sample_image(img_func, factor_x, factor_y)
        if print_intermediate:
            write_image(img_func, f"{OUTPUT_PATH}{output_name}_down_sampled.png")
    if bit_slice > 0:
        img_func = img_func / pow(2, bit_slice)
        if print_intermediate:
            write_image(img_func, f"{OUTPUT_PATH}{output_name}_sliced.png")
    color_space = extract_color_space(img_func)
    image_means = perform_k_means(color_space, means_no)
    image_groups = group_by_mean(color_space, image_means)
    red_group = 0
    for index in range(len(image_means)):
        if image_means[index][0] > image_means[red_group][0]:
            red_group = index
    img_func = color_image_by_groups(img_func, [image_groups[red_group]], red_mask)
    if print_intermediate:
        write_image(img_func, f"{OUTPUT_PATH}{output_name}_k_means.png")

    img_func = frame_cull(img_func, cull_factor_x, cull_factor_y)
    if print_intermediate:
        write_image(img_func, f"{OUTPUT_PATH}{output_name}_culled_{cull_factor_x}_{cull_factor_y}.png")

    img_func = open_image(img_func, open_amount)
    if print_intermediate:
        write_image(img_func, f"{OUTPUT_PATH}{output_name}_opened_{open_amount}.png")

    img_func = close_image(img_func, close_amount)
    if print_intermediate:
        write_image(img_func, f"{OUTPUT_PATH}{output_name}_closed_{close_amount}.png")

    img_func = convolve_image(binarize_image(img_func), get_ellipse_mask(mask_x, mask_y), mask_x, mask_y)
    if print_intermediate:
        write_image(normalize(img_func), f"{OUTPUT_PATH}{output_name}_convolution.png")

    original = read_image(f"{INPUT_PATH}{input_name}.png")
    if factor_x > 1 or factor_y > 1:
        original = down_sample_image(original, factor_x, factor_y)

    img_func = original * apply_convolution_result(img_func, mask_x, mask_y, original.shape[2])
    write_image(img_func, f"{OUTPUT_PATH}{output_name}.png")

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    detect_faces("1_source", "1_faces", 6, print_intermediate=True)
    detect_faces("2_source", "2_faces", 6, factor_x=9, factor_y=9, print_intermediate=True, cull_factor_x=0.001, cull_factor_y=0.001, equalize=False)
    detect_faces("3_source", "3_faces", 6, print_intermediate=True)