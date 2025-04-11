from my_models.naive_bayes import MyNaiveBayes
import numpy as np
import os
import re
import chardet
from sklearn.model_selection import train_test_split

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)  # 读取前10000字节检测

    return chardet.detect(rawdata)['encoding']

def read_mail(path, class_name):
    content_list = []
    class_name_list = []

    for file_name in os.listdir(path):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(path, file_name)
        try:
            encoding = detect_encoding(file_path)
            with open(file_path, "r", encoding=encoding, errors='replace') as file:
                content = file.read()
                words = split_data_to_word(content)

                content_list.append(words)
                class_name_list.append(class_name)

        except Exception as e:
            print(f"Skipped {file_path} due to error: {str(e)}")

    return content_list, class_name_list

def split_data_to_word(content):
    """
    分词
    :param content:
    :return:
    """
    list_of_word = re.split(r'\W+', content)
    return list_of_word

def build_vocabulary(*word_list):
    """
    创建语料库
    :param content_list:
    :return:
    """
    vocabulary = set()
    for content in word_list:
        for word in content:
            vocabulary.update(word)

    return list(vocabulary)

def translate_content_to_vector(words_in_content_list, vocabulary):
    """
    将内容列表转换成向量的表示形式
    :param words_in_content_list:
    :param vocabulary:
    :return:
    """
    sample_num = len(words_in_content_list)
    feature_num = len(vocabulary)

    content_vector = np.zeros((sample_num, feature_num))

    for sample_index in range(sample_num):
        for word in words_in_content_list[sample_index]:
            feature_index = vocabulary.index(word)
            if feature_index >= 0:
                content_vector[sample_index][feature_index] = content_vector[sample_index][feature_index] + 1

    return content_vector


if "__main__" == __name__:
    spam_path = r"../data/email/spam/"
    ham_path = r"../data/email/ham/"

    spam_content_list, spam_class_list = read_mail(spam_path, 1) # "1"表示垃圾邮件
    ham_content_list, ham_class_list = read_mail(ham_path, 0) # "0"表示正常邮件

    content_list = spam_content_list + ham_content_list
    class_list = spam_class_list + ham_class_list

    word_library = build_vocabulary(content_list)
    print(f"word_library's len: {len(word_library)}")

    X_all = translate_content_to_vector(content_list, word_library)
    y_all = np.array(class_list)
    print(f"X_all shape: {X_all.shape}")
    print(f"y_all shape: {y_all.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    bayes_model = MyNaiveBayes()
    bayes_model.fit(X_train, y_train)
    y_pred = bayes_model.predict(X_test)
