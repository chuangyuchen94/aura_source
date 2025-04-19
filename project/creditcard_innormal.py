from utils.data_preprocessing import data_statistics
from utils.data_preprocessing import sample_processing
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, make_scorer

if "__main__" == __name__:
    # 加载数据
    data = pd.read_csv("../data/creditcard.csv")

    # 统计数据
    # big_var_column = data_statistics.analyze_data(data)
    # print(f"big_var_column: {big_var_column}")

    # 标准化处理
    standard_transformer = StandardScaler()
    data["Amount"] = standard_transformer.fit_transform(data[["Amount"]])

    # big_var_column = data_statistics.analyze_data(data)
    # print(f"big_var_column: {big_var_column}")

    # one-hot编码处理
    # 不需要，因为class的枚举值是数值

    # 数据样本采样
    new_data = sample_processing.sampling_data(data, "Class", "under")
    big_var_column = data_statistics.analyze_data(new_data)
    print(f"big_var_column: {big_var_column}")

    # 切割特征与结果
    data_X = np.array(new_data.drop(labels=["Time", "Class"], axis=1))
    data_y = np.array(new_data[["Class"]]).reshape(-1)

    # 数据切分
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=42)

    # 选择模型: 逻辑回归
    # 训练模型：交叉验证
    print(f"\n===== 训练模型：交叉验证--C =====")
    C = [0.01, 0.1, 1, 10, 100]

    # for c in C:
    #    model = LogisticRegression(penalty="l1", C=c, solver="liblinear", max_iter=1000, tol=1e-5, random_state=42)
    #    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=make_scorer(recall_score))
    #    mean = sum(scores) / len(scores)
    #    print(f"result of parameter C: {c}\n\tscores: {scores} \n\tmean: {mean} \n")

    print(f"\n===== 训练模型：交叉验证--max_iterint =====")
    max_iter_list = [10, 100, 500, 1000, 2000]
    C = 0.01 # 参数C的结论：scores: [0.96202532 0.97468354 0.93670886 0.94936709 0.91025641] mean: 0.9466082440765986
    for max_iter in max_iter_list:
        model = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=max_iter, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=make_scorer(recall_score))
        mean = sum(scores) / len(scores)
        print(f"result of parameter max_iter: {max_iter}\n\tscores: {scores} \n\tmean: {mean} \n")

    # 以最佳参数训练模型
    # 用测试集数据测试，评估结果
    print(f"===== 以最佳参数训练模型 =====")
    print(f"===== 用测试集数据测试，评估结果 =====")

    model_best = LogisticRegression(penalty="l1", C=C, solver="liblinear", random_state=42)
    model_best.fit(X_train, y_train)
    y_pred = model_best.predict(X_test)
    recall_value = recall_score(y_test, y_pred)

    print(f"===== 结果：{recall_value} =====")
