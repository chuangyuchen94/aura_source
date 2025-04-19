from utils.data_preprocessing import data_statistics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

if "__main__" == __name__:
    # 加载数据
    data = pd.read_csv("../data/temps.csv")
    # big_var_column = data_statistics.analyze_data(data)

    # print(f"大方差字段：{big_var_column}")

    # one-hot编码处理
    data = pd.get_dummies(data, dtype=int)
    column_list = data.drop(["actual",], axis=1).columns
    # big_var_column = data_statistics.analyze_data(data)

    # 数据样本采样：回归任务，不需要进行采样处理

    # 切割特征与结果
    # 将pandas格式转换为numpy格式
    X = np.array(data.drop(["actual",], axis=1))
    y = np.array(data["actual"]).reshape(-1)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # 数据切分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 选择模型
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape_score = mean_absolute_percentage_error(y_test, y_pred)

    print(f"mape_score: {mape_score}")

    # 确定重要特征
    feature_importance = model.feature_importances_
    print(f"important_features: {feature_importance}")
    accumulative_importance = 0
    for index in np.argsort(feature_importance)[::-1]:
        importance = feature_importance[index]
        accumulative_importance += importance
        print(f"column: {column_list[index]} ===> {importance} | accumulative_importance: {accumulative_importance}")

    # 确定最重要的特征：temp_1, average
    # 重新选择特征，训练模型，进行预测，评估结果
    best_feature = ["temp_1", "average", "friend"]
    X_best = np.array(data[best_feature])
    X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(X_best, y, test_size=0.2, random_state=42)

    model_best_feature = RandomForestRegressor(random_state=42)
    model_best_feature.fit(X_train_best, y_train_best)
    y_pred_best = model_best_feature.predict(X_test_best)

    mape_score_best = mean_absolute_percentage_error(y_test_best, y_pred_best)

    print(f"mape_score_best: {mape_score_best}")
