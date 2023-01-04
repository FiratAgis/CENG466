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
    return ((arr - min_val) * (255 / (max_val - min_val))).astype(np.uint8)


def grayscale_image(img_func: np.ndarray) -> np.ndarray:
    ret_val = np.zeros((img_func.shape[0], img_func.shape[1],))
    ret_val += img_func[:, :, 0]
    ret_val += img_func[:, :, 1]
    ret_val += img_func[:, :, 2]
    return (ret_val / 3).astype(np.uint8)


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

    limit_y = img_shape_y - element_shape_y
    limit_x = img_shape_x - element_shape_x

    output = np.zeros((img_shape_x, img_shape_y), dtype=np.uint8)

    if operation_type.lower() == "d":
        imagePadded = np.zeros((img_shape_x + padding_x * 2, img_shape_y + padding_y * 2), dtype=np.uint8)
        imagePadded[int(padding_x):int(-1 * padding_x), int(padding_y):int(-1 * padding_y)] = img_func.astype(np.uint8)
        for y in range(img_shape_y):
            if y > limit_y:
                break
            for x in range(img_shape_x):
                if x > limit_x:
                    break
                try:
                    output[x, y] = (structuring_element * imagePadded[x: x + element_shape_x, y: y + element_shape_y]).max()
                except:
                    break
    elif operation_type.lower() == "e":
        imagePadded = np.full((img_shape_x + padding_x * 2, img_shape_y + padding_y * 2), dtype=np.uint8, fill_value=255)
        imagePadded[int(padding_x):int(-1 * padding_x), int(padding_y):int(-1 * padding_y)] = img_func.astype(np.uint8)
        for y in range(img_shape_y):
            if y > limit_y:
                break
            for x in range(img_shape_x):
                if x > limit_x:
                    break
                try:
                    output[x, y] = (structuring_element * imagePadded[x: x + element_shape_x,  y: y + element_shape_y]).min()
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


def mean_shift_and_its_friends(file, cluster_freq: int = 100):

    plt.figure(figsize=(20,5))
    img_data = Image.open(INPUT_PATH + "B" + str(file) + ".jpg")
    img = np.asarray(img_data)
    img = down_sample_image(img, 16, 16)

    # filter to reduce noise
    img = cv.medianBlur(img, 9)

    # img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    plt.subplot(2,3,1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    # flatten the image
    flat_image = img.reshape((-1,3))
    flat_image = np.float32(flat_image)

    # meanshift
    bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
    ms = MeanShift(bandwidth = bandwidth, max_iter=800, min_bin_freq=cluster_freq, bin_seeding=True)
    ms.fit(flat_image)
    labeled=ms.labels_


    # get number of segments
    segments = np.unique(labeled)
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


    plt.savefig(OUTPUT_PATH + "B" + str(file) + "_algorithm_meanshift_parameterset_" + str(cluster_freq), bbox_inches='tight')

def n_cut_and_its_friends(file, n_segments: int = 400):

  plt.figure(figsize=(20,5))
  img_data = Image.open(INPUT_PATH + "B" + str(file) + ".jpg")
  img = np.asarray(img_data)
  img = down_sample_image(img, 16, 16)

  # img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

  plt.subplot(1,3,1)
  plt.imshow(img)
  plt.title('Original Image')
  plt.axis('off')

      # flatten the image
  flat_image = img.reshape((-1,3))
  flat_image = np.float32(flat_image)

  labels1 = segmentation.slic(img, compactness=30, n_segments=400)
  out1 = color.label2rgb(labels1, img, kind='avg')

  g = graph.rag_mean_color(img, labels1, mode='similarity')
  labels2 = graph.cut_normalized(labels1, g)
  result = color.label2rgb(labels2, img, kind='avg')

  result = result.astype(np.uint8)
  labeled = np.zeros((result.shape[0], result.shape[1]))
  labeled = np.sum(result, axis=2)//3
  labels = labeled
  labels = np.unique(labels)
  segments = labels


  plt.subplot(1,3,2)
  plt.imshow(result)
  plt.title('Segmentation Map')
  plt.axis('off')


  result_gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
  edges = cv.Canny(result_gray, 30, 80)

  boundary_over = img
  boundary_over_copy = boundary_over.copy()

  for x in range(1, edges.shape[0]-1):
    for y in range(1, edges.shape[1]-1):
      if edges[x][y] > 127:
        boundary_over_copy[x][y][0] = 255
        boundary_over_copy[x][y][1] = 0
        boundary_over_copy[x][y][2] = 0

        boundary_over_copy[x+1][y][0] = 255
        boundary_over_copy[x+1][y][1] = 0
        boundary_over_copy[x+1][y][2] = 0

        boundary_over_copy[x-1][y][0] = 255
        boundary_over_copy[x-1][y][1] = 0
        boundary_over_copy[x-1][y][2] = 0

        boundary_over_copy[x][y+1][0] = 255
        boundary_over_copy[x][y+1][1] = 0
        boundary_over_copy[x][y][2] = 0

        boundary_over_copy[x][y-1][0] = 255
        boundary_over_copy[x][y-1][1] = 0
        boundary_over_copy[x][y-1][2] = 0

  for x in range(0, edges.shape[0]):
        boundary_over_copy[x][2][0] = 255
        boundary_over_copy[x][2][1] = 0
        boundary_over_copy[x][2][2] = 0  

        boundary_over_copy[x][3][0] = 255
        boundary_over_copy[x][3][1] = 0
        boundary_over_copy[x][3][2] = 0  

        boundary_over_copy[x][edges.shape[1]-3][0] = 255
        boundary_over_copy[x][edges.shape[1]-3][1] = 0
        boundary_over_copy[x][edges.shape[1]-3][2] = 0 

        boundary_over_copy[x][edges.shape[1]-2][0] = 255
        boundary_over_copy[x][edges.shape[1]-2][1] = 0
        boundary_over_copy[x][edges.shape[1]-2][2] = 0 

  for y in range(0, edges.shape[1]):
        boundary_over_copy[2][y][0] = 255
        boundary_over_copy[2][y][1] = 0
        boundary_over_copy[2][y][2] = 0  

        boundary_over_copy[3][y][0] = 255
        boundary_over_copy[3][y][1] = 0
        boundary_over_copy[3][y][2] = 0  

        boundary_over_copy[edges.shape[0]-2][y][0] = 255
        boundary_over_copy[edges.shape[0]-2][y][1] = 0
        boundary_over_copy[edges.shape[0]-2][y][2] = 0 

        boundary_over_copy[edges.shape[0]-3][y][0] = 255
        boundary_over_copy[edges.shape[0]-3][y][1] = 0
        boundary_over_copy[edges.shape[0]-3][y][2] = 0 


  plt.subplot(1,3,3)
  plt.imshow(boundary_over_copy)
  plt.title('Boundary Overlay')
  plt.axis('off')



  plt.savefig(OUTPUT_PATH + "B" + str(file) + "_algorithm_ncut_parameterset_" + str(n_segments), bbox_inches='tight')


def get_average_color(colors: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    col = np.zeros(3)
    for color in colors:
        col = col + color
    col = col / len(colors)
    return int(col[0]), int(col[1]), int(col[2]),


def down_sample_image(img_func: np.ndarray, x_fact: int, y_fact: int, type: str = "rgb") -> np.ndarray:
    x_limit = img_func.shape[0] // x_fact
    y_limit = img_func.shape[1] // y_fact
    if type.lower() == "rgb":
        ret_val = np.zeros((x_limit, y_limit, 3), dtype=np.uint8)
    else:
        ret_val = np.zeros((x_limit, y_limit), dtype=np.uint8)
    for x in range(x_limit):
        for y in range(y_limit):
            colors = []
            if type.lower() == "rgb":
                for xk in range(x_fact):
                    for yk in range(y_fact):
                        cell = img_func[x * x_fact + xk][y * y_fact + yk]
                        colors.append((int(cell[0]), int(cell[1]), int(cell[2])))
                color = get_average_color(colors)
                ret_val[x][y][0] = color[0]
                ret_val[x][y][1] = color[1]
                ret_val[x][y][2] = color[2]
            if type.lower() == "binary":
                for xk in range(x_fact):
                    for yk in range(y_fact):
                        if img_func[x * x_fact + xk][y * y_fact + yk] > 200:
                            colors.append(1)
                        else:
                            colors.append(0)
                if sum(colors) >= (x_fact * y_fact) / 2:
                    ret_val[x][y] = 255
            if type.lower() == "gray":
                for xk in range(x_fact):
                    for yk in range(y_fact):
                        colors.append(img_func[x * x_fact + xk][y * y_fact + yk])
                ret_val[x][y] = sum(colors) / (x_fact * y_fact)
    return ret_val.astype(np.uint8)


def binarize_image(img_func: np.ndarray, threshold: int= 200) -> np.ndarray:
    ret_val = np.zeros((img_func.shape[0], img_func.shape[1]), np.uint8)
    for x in range(img_func.shape[0]):
        for y in range(img_func.shape[1]):
            if img_func[x][y] >= threshold:
                ret_val[x][y] = 255
    return ret_val


def erode_image(img_func: np.ndarray, degree: int = 1) -> np.ndarray:
    if degree == 0:
        return img_func
    final_image = np.zeros(img_func.shape)
    for x in range(1, img_func.shape[0] - 1):
        for y in range(1, img_func.shape[1] -1):
            if (img_func[x-1][y-1] > 40 and img_func[x][y-1] > 40 and img_func[x+1][y-1] > 40 and
                img_func[x-1][y] > 40 and img_func[x][y] > 40 and img_func[x+1][y] > 40 and
                img_func[x-1][y+1] > 40 and img_func[x][y+1] > 40 and img_func[x+1][y+1] > 40):
                final_image[x][y] = 255
    return erode_image(final_image, degree - 1)


def dilute_image(img_func: np.ndarray, degree: int = 1) -> np.ndarray:
    if degree == 0:
        return img_func
    final_image = np.zeros(img_func.shape)
    for x in range(1, img_func.shape[0] - 1):
        for y in range(1, img_func.shape[1] -1):
            if (img_func[x-1][y-1] > 40 or img_func[x][y-1]> 40 or img_func[x+1][y-1] > 40 or
                img_func[x-1][y] > 40 or img_func[x][y] > 40 or img_func[x+1][y] > 40 or
                img_func[x-1][y+1] > 40 or img_func[x][y+1] > 40 or img_func[x+1][y+1] > 40):
                final_image[x][y] = 255
    return dilute_image(final_image, degree - 1)


def open_image(img_func: np.ndarray, degree: int = 1):
    ret_val = erode_image(img_func, degree)
    ret_val = dilute_image(ret_val, degree)
    return  ret_val


def close_image(img_func: np.ndarray, degree: int = 1):
    ret_val = dilute_image(img_func, degree)
    ret_val = erode_image(ret_val, degree)
    return  ret_val


def delete_flower(img_func: np.ndarray, x: int, y: int) -> np.ndarray:
    if x < 0 or x >= img_func.shape[0] or y < 0 or y >= img_func.shape[1]:
        return img_func
    if img_func[x][y] < 200:
        return img_func
    else:
        ret_val = np.array(img_func)
        ret_val[x][y] = 0
        try:
            ret_val = delete_flower(ret_val, x - 1, y)
            ret_val = delete_flower(ret_val, x + 1, y)
            ret_val = delete_flower(ret_val, x, y - 1)
            ret_val = delete_flower(ret_val, x, y + 1)
        except RecursionError:
            return ret_val
        return ret_val


def count_flowers(img_func: np.ndarray) -> int:
    flower_count = 0
    temp = np.array(img_func)
    for x in range(img_func.shape[0]):
        for y in range(img_func.shape[1]):
            if temp[x][y] > 200:
                flower_count += 1
                temp = delete_flower(temp, x, y)
    return flower_count


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    img = read_image(f"{INPUT_PATH}A1.png") # Read image
    img = grayscale_image(down_sample_image(img, 16, 16)) # Down-sample and gray-scale the image
    img = grayscale_morphological_operation(img, np.ones((50, 50)), "t") # Perform top hat operation
    img = binarize_image(img, 100) # Binarize the image
    img = close_image(open_image(img, 2), 2) # Filter out noise
    write_image(add_rgb_channel(img), f"{OUTPUT_PATH}A1.png", True)
    print(f"The number of flowers in image A1 is {count_flowers(img)}")

    img = read_image(f"{INPUT_PATH}A2.png") # Read image
    img = grayscale_image(down_sample_image(img, 16, 16)) # Down-sample and gray-scale the image
    img = grayscale_morphological_operation(img, np.ones((75, 75)), "t") # Perform top hat operation
    img = binarize_image(img, 100) # Binarize the image
    img = down_sample_image(img, 2, 2, "binary")
    img = close_image(open_image(img, 2), 2) # Filter out noise
    write_image(add_rgb_channel(img), f"{OUTPUT_PATH}A2.png", True)
    print(f"The number of flowers in image A2 is {count_flowers(img)}")

    img = read_image(f"{INPUT_PATH}A3.png") # Read image
    img = grayscale_image(down_sample_image(img, 16, 16)) # Down-sample and gray-scale the image
    img = grayscale_morphological_operation(img, np.ones((150, 150)), "t") # Perform top hat operation
    img = binarize_image(img, 100) # Binarize the image
    img = down_sample_image(img, 2, 2, "binary")
    img = close_image(open_image(img, 4), 4) # Filter out noise
    write_image(add_rgb_channel(img), f"{OUTPUT_PATH}A3.png", True)
    print(f"The number of flowers in image A3 is {count_flowers(img)}")

    mean_shift_and_its_friends(3, 50)
    mean_shift_and_its_friends(3, 100)
    mean_shift_and_its_friends(3, 150)
    mean_shift_and_its_friends(2, 50)
    mean_shift_and_its_friends(2, 100)
    mean_shift_and_its_friends(2, 150)
    mean_shift_and_its_friends(1, 50)
    mean_shift_and_its_friends(1, 100)
    mean_shift_and_its_friends(1, 150)
    mean_shift_and_its_friends(4, 50)
    mean_shift_and_its_friends(4, 100)
    mean_shift_and_its_friends(4, 150)

    n_cut_and_its_friends(3, 200)
    n_cut_and_its_friends(3, 400)
    n_cut_and_its_friends(3, 600)
    n_cut_and_its_friends(2, 200)
    n_cut_and_its_friends(2, 400)
    n_cut_and_its_friends(2, 600)
    n_cut_and_its_friends(1, 200)
    n_cut_and_its_friends(1, 400)
    n_cut_and_its_friends(1, 600)
    n_cut_and_its_friends(4, 200)
    n_cut_and_its_friends(4, 400)
    n_cut_and_its_friends(4, 600)
