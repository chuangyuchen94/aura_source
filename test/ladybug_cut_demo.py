from sklearn.cluster import KMeans
import numpy as np
from PIL import Image

def load_image_data(file_path):
    """
    从指定的文件路径（图像）读出数据
    :param file_path:
    :return:
    """
    img = Image.open(file_path)
    img_array = np.array(img)

    return img_array

def split_image(img_array, part_num=2):
    """
    将图像分割为多个部分
    :param img_array:
    :param part_num:
    :return:
    """
    img_model = KMeans(n_clusters=part_num, init="k-means++", n_init=10, random_state=0)
    img_part_label = img_model.fit_transform(img_array)

    img_parts = []
    labels = np.unique(img_model.labels_)
    for label in labels:
        single_part = img_array.copy()
        single_part[img_model.labels_ != label] = 255 # 其他区域置为白色
        img_parts.append(single_part)

    return img_parts

def save_image(image_data, file_shape, file_path):
    """
    将图像的二维数据，写回变成图像
    :param image_data: 二维数组
    :param file_path: 原始文件的路径
    :return:
    """
    for image_index, image in enumerate(image_data):
        image_file_name = get_new_file_path(file_path, image_index)
        image_3d_data = image.reshape(file_shape)
        Image.fromarray(image_3d_data).save(image_file_name)

def get_new_file_path(file_path, num=0):
    file_postfix = file_path.rfind(".")
    new_file_path = f"{file_path[:file_postfix]}{num}{file_path[file_postfix:]}"
    print(f"new_file_path: {new_file_path}")

    return new_file_path

if "__main__" == __name__:
    img_path = "../data/ladybug.png"
    img_data = load_image_data(img_path)
    image_shape = img_data.shape
    print(img_data.shape)
    get_new_file_path(img_path, 0)
    get_new_file_path(img_path, 1)

    img_data = img_data.reshape(-1, 3)

    img_part_all = split_image(img_data, part_num=3)
    print(len(img_part_all))
    save_image(img_part_all, image_shape, img_path)

