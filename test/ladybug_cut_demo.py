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
        single_part = img_part_label[img_model.labels_ == label].copy()
        img_parts.append(single_part)

    return img_parts

if "__main__" == __name__:
    img_path = "../data/ladybug.png"
    img_data = load_image_data(img_path)
    print(img_data.shape)

    img_data = img_data.reshape(-1, 3)

    img_part_all = split_image(img_data, part_num=2)
    print(len(img_part_all))
    print(len(img_part_all))

