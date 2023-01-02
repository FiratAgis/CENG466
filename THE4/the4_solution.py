"""
Submission by
Fırat Ağış, e2236867
Robin Koç, e246871
"""
import os
import numpy as np
import matplotlib.image
from PIL import Image


INPUT_PATH = "./THE4_Images/"
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
    max_val = max(arr.max(), 255)
    min_val = min(arr.min(), 0)
    return (arr - min_val) * (255 / (max_val - min_val))


def grayscale_image(img_func: np.ndarray) -> np.ndarray:
    return normalize((img_func[:,:,0] + img_func[:,:,1] + img_func[:,:,2])/3)


def grayscale_morphological_operation(img_func: np.ndarray, structuring_element: np.ndarray, operation_type: str) -> np.ndarray:
    if operation_type.lower() == "o":
        return grayscale_morphological_operation(grayscale_morphological_operation(img_func, structuring_element, "e"),
                                                 structuring_element, "d")
    elif operation_type.lower() == "c":
        return grayscale_morphological_operation(grayscale_morphological_operation(img_func, structuring_element, "d"),
                                                 structuring_element, "e")
    elif operation_type.lower() == "t":
        return img_func - grayscale_morphological_operation(img_func, structuring_element, "o")
    element_shape_x = structuring_element.shape[0]
    element_shape_y = structuring_element.shape[1]
    padding_x = int(element_shape_x // 2)
    padding_y = int(element_shape_y // 2)
    img_shape_x = img_func.shape[0]
    img_shape_y = img_func.shape[1]

    output = np.zeros((img_shape_x, img_shape_y))

    imagePadded = np.zeros((img_shape_x + padding_x * 2, img_shape_y + padding_y * 2))
    imagePadded[int(padding_x):int(-1 * padding_x), int(padding_y):int(-1 * padding_y)] = img_func

    for y in range(img_shape_y):
        if y > img_shape_y - element_shape_y:
            break
        for x in range(img_shape_x):
            if x > img_shape_x - element_shape_x:
                break
            try:
                if operation_type.lower() == "d":
                    output[x, y] = (structuring_element * imagePadded[x: x + element_shape_x, y: y + element_shape_y]).max()
                elif operation_type.lower() == "e":
                    output[x, y] = (structuring_element * imagePadded[x: x + element_shape_x, y: y + element_shape_y]).min()
            except:
                break

    return output


def generate_circular_structuring_element(radius: int, value: float = 1.0) -> np.ndarray:
    ret_val = np.zeros((radius*2, radius*2))
    for x in range(radius*2):
        for y in range(radius*2):
            if x**2 + y**2 < radius**2:
                ret_val[x][y] = value
    return ret_val


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)



