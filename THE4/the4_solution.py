"""
Submission by
Fırat Ağış, e2236867
Robin Koç, e246871
"""
import os
import numpy as np
import matplotlib.image
from matplotlib import pyplot as plt
from PIL import Image
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
import networkx as nx


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


def add_rgb_channel(img_func: np.ndarray) -> np.ndarray:
    ret_val = np.zeros((img_func.shape[0], img_func.shape[1], 3))
    ret_val[:, :, 0] = img_func
    ret_val[:, :, 1] = img_func
    ret_val[:, :, 2] = img_func
    return ret_val

def normalize(arr: np.ndarray) -> np.ndarray:
    max_val = max(arr.max(), 255)
    min_val = min(arr.min(), 0)
    return (arr - min_val) * (255 / (max_val - min_val))


def grayscale_image(img_func: np.ndarray) -> np.ndarray:
    ret_val = np.zeros((img_func.shape[0], img_func.shape[1],))
    ret_val += img_func[:, :, 0]
    ret_val += img_func[:, :, 1]
    ret_val += img_func[:, :, 2]
    return ret_val / 3


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

def mean_shift_and_its_friends(file, min_bin_freq):

    plt.figure(figsize=(20,5))
    img = cv.imread("/content/B" + str(file) + "3.jpg")

    # filter to reduce noise
    img = cv.medianBlur(img, 9)

    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    plt.subplot(2,3,1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    # flatten the image
    flat_image = img.reshape((-1,3))
    flat_image = np.float32(flat_image)

    # meanshift
    bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
    ms = MeanShift(bandwidth = bandwidth, max_iter=800, min_bin_freq=100, bin_seeding=True)
    ms.fit(flat_image)
    labeled=ms.labels_


    # get number of segments
    segments = np.unique(labeled)
    print('Number of segments: ', segments.shape[0])
    print(segments)
    # get the average color of each segment
    total = np.zeros((segments.shape[0], 3), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label] = total[label] + flat_image[i]
        count[label] += 1
    avg = total/count
    avg = np.uint8(avg)

    # cast the labeled image into the corresponding average color
    res = avg[labeled]
    result = res.reshape((img.shape))


    plt.subplot(2,3,2)
    plt.imshow(result)
    plt.title('Segmentation Map')
    plt.axis('off')


    result_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(result_gray, 30, 80)

    boundary_over = img

    for x in range(1, edges.shape[0]-1):
        for y in range(1, edges.shape[1]-1):
            if edges[x][y] > 127:
                boundary_over[x][y][0] = 255
                boundary_over[x][y][1] = 0
                boundary_over[x][y][2] = 0

                boundary_over[x+1][y][0] = 255
                boundary_over[x+1][y][1] = 0
                boundary_over[x+1][y][2] = 0

                boundary_over[x-1][y][0] = 255
                boundary_over[x-1][y][1] = 0
                boundary_over[x-1][y][2] = 0

                boundary_over[x][y+1][0] = 255
                boundary_over[x][y+1][1] = 0
                boundary_over[x][y][2] = 0

                boundary_over[x][y-1][0] = 255
                boundary_over[x][y-1][1] = 0
                boundary_over[x][y-1][2] = 0

    for x in range(0, edges.shape[0]):
        boundary_over[x][2][0] = 255
        boundary_over[x][2][1] = 0
        boundary_over[x][2][2] = 0  

        boundary_over[x][3][0] = 255
        boundary_over[x][3][1] = 0
        boundary_over[x][3][2] = 0  

        boundary_over[x][edges.shape[1]-3][0] = 255
        boundary_over[x][edges.shape[1]-3][1] = 0
        boundary_over[x][edges.shape[1]-3][2] = 0 

        boundary_over[x][edges.shape[1]-2][0] = 255
        boundary_over[x][edges.shape[1]-2][1] = 0
        boundary_over[x][edges.shape[1]-2][2] = 0 

    for y in range(0, edges.shape[1]):
        boundary_over[2][y][0] = 255
        boundary_over[2][y][1] = 0
        boundary_over[2][y][2] = 0  

        boundary_over[3][y][0] = 255
        boundary_over[3][y][1] = 0
        boundary_over[3][y][2] = 0  

        boundary_over[edges.shape[0]-2][y][0] = 255
        boundary_over[edges.shape[0]-2][y][1] = 0
        boundary_over[edges.shape[0]-2][y][2] = 0 

        boundary_over[edges.shape[0]-3][y][0] = 255
        boundary_over[edges.shape[0]-3][y][1] = 0
        boundary_over[edges.shape[0]-3][y][2] = 0 


    plt.subplot(2,3,3)
    plt.imshow(boundary_over)
    plt.title('Boundary Overlay')
    plt.axis('off')


    labeled_img = labeled.reshape((img.shape[0], img.shape[1]))
    adjacencies = np.zeros((segments.shape[0],segments.shape[0],5))

    for i in range(0,segments.shape[0]):
        adjacencies[i][i][0] = 1


    #four neighbourhood
    for x in range(1, labeled_img.shape[0]-1):
        for y in range(1, labeled_img.shape[1]-1):
            if labeled_img[x][y] != labeled_img[x-1][y]:
                adjacencies[labeled_img[x][y]][labeled_img[x-1][y]][0] = 1
                adjacencies[labeled_img[x][y]][labeled_img[x-1][y]][1] = 1 #right neighbour
            if labeled_img[x][y] != labeled_img[x+1][y]:
                adjacencies[labeled_img[x][y]][labeled_img[x+1][y]][0] = 1
                adjacencies[labeled_img[x][y]][labeled_img[x+1][y]][2] = 1 #left neighbour
            if labeled_img[x][y] != labeled_img[x][y-1]:
                adjacencies[labeled_img[x][y]][labeled_img[x][y-1]][0] = 1
                adjacencies[labeled_img[x][y]][labeled_img[x][y-1]][3] = 1 #up neighbour
            if labeled_img[x][y] != labeled_img[x][y+1]:
                adjacencies[labeled_img[x][y]][labeled_img[x][y+1]][0] = 1
                adjacencies[labeled_img[x][y]][labeled_img[x][y+1]][4] = 1 #down neighbour

    adjacencie_graph = nx.Graph()

    for i in segments:
        adjacencie_graph.add_node(i)

    for i in range(0,segments.shape[0]):
        for j in range(0,segments.shape[0]):
            if adjacencies[i][j][0] == 1:
                adjacencie_graph.add_edge(i,j)



    plt.subplot(2,3,4)
    nx.draw_circular(adjacencie_graph, with_labels = True)
    plt.title('Adjacency Graph')
    plt.axis('off')


    tree_rep = nx.Graph()

    tree_rep.add_node("root", size=10)

    label_counts = np.bincount(labeled)
    label_counts = np.nan_to_num(label_counts, nan=0)

    max_label = 0
    second_max = 1

    for i in range(0, len(label_counts)):
        if label_counts[i] > label_counts[max_label]:
            second_max = max_label
            max_label = i
    
    if max_label == 0 and second_max == 0:
        for i in range(1, len(label_counts)):
            if label_counts[i] > label_counts[second_max]:
                second_max = i


    tree_rep.add_node(max_label)
    tree_rep.add_edge(max_label,"root")

    tree_rep.add_node(second_max)
    tree_rep.add_edge(second_max,"root")


    # most_frequent = labeled[indices]

    for i in range(0,segments.shape[0]):
        for j in range(0,segments.shape[0]):
            if adjacencies[i][j][0] == 1 and i != max_label and i != second_max:
                if adjacencies[i][j][1] == 1 and adjacencies[i][j][2] == 1 and adjacencies[i][j][3] == 1 and adjacencies[i][j][4] == 1 :
                    tree_rep.add_node(i)
                    tree_rep.add_edge(i,j)
                    continue





    plt.subplot(2,3,5)
    pos=nx.kamada_kawai_layout(tree_rep)
    nx.draw_circular(tree_rep, with_labels = True)
    plt.title('Tree Representation')
    plt.axis('off')



    plt.savefig(OUTPUT_PATH + "B" + str(file) + "_algorithm_meanshift_parameterset_" + str(min_bin_freq), bbox_inches='tight')



if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    img = grayscale_image(read_image(INPUT_PATH + "A1.png"))
    write_image(add_rgb_channel(img), OUTPUT_PATH + "A1_gray.png", True)


