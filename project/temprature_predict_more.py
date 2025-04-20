from utils.data_preprocessing import data_statistics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def grid_search(X_train, y_train, grid_params, title):
    """
    定义网格搜索的方法
    :param X_train:
    :param y_train:
    :return:
    """
    print(f"\n====== {title} ======")

    model_grid_search = RandomForestRegressor(random_state=42)
    search_grid = GridSearchCV(estimator=model_grid_search,
                                 param_grid=grid_params,
                                 scoring="neg_mean_absolute_percentage_error",
                                 n_jobs=-1,
                                 cv=3,
                                 verbose=2)
    search_grid.fit(X_train, y_train)
    best_params = search_grid.best_params_
    print(f"====== {title}，best_params: {best_params}")

def fit_predit(X_train, y_train, X_test, y_test, n_estimators, max_depth, min_samples_split, min_samples_leaf, bootstrap, title):
    model_grid = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                         bootstrap=bootstrap, random_state=42, n_jobs=-1)
    model_grid.fit(X_train, y_train)
    y_pred_grid = model_grid.predict(X_test)
    mape_score_grid = mean_absolute_percentage_error(y_test, y_pred_grid)

    print(f"====== {title}，mape_score_best: {mape_score_grid:.5f}")

if "__main__" == __name__:
    # 加载数据
    data = pd.read_csv("../data/temps_extended.csv")

    # 统计数据
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

    # 选择模型，建立基础模型
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape_score = mean_absolute_percentage_error(y_test, y_pred)

    print(f"===== 基础模型 mape_score: {mape_score:.5f} =====\n")

    # 确定重要特征
    feature_importance = model.feature_importances_
    # print(f"important_features: {feature_importance}")
    accumulative_importance = 0
    for index in np.argsort(feature_importance)[::-1]:
        importance = feature_importance[index]
        accumulative_importance += importance
        print(f"column: {column_list[index]} ===> {importance:.3f} | accumulative_importance: {accumulative_importance:.3f}")

    # 确定最重要的特征：temp_1, average
    # 重新选择特征，训练模型，进行预测，评估结果
    best_feature = ["temp_1", "average", "ws_1", "friend", "temp_2", "day"]
    X_best = np.array(data[best_feature])
    X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(X_best, y, test_size=0.2, random_state=42)

    model_best_feature = RandomForestRegressor(random_state=42, n_jobs=-1)
    model_best_feature.fit(X_train_best, y_train_best)
    y_pred_best = model_best_feature.predict(X_test_best)

    mape_score_best = mean_absolute_percentage_error(y_test_best, y_pred_best)

    print(f"======确定最重要的特征后，mape_score_best: {mape_score_best:.5f}")

    # 确定模型最佳参数
    print(f"\n====== 确定模型最佳参数 ======")

    n_estimators = [int(value) for value in np.arange(start=100, stop=2000, step=10, dtype=int)]
    max_depth = [int(value) for value in np.arange(start=4, stop=20, step=2, dtype=int)]
    min_samples_split = [int(value) for value in np.arange(start=2, stop=20, step=2, dtype=int)]
    min_samples_leaf = [int(value) for value in np.arange(start=1, stop=20, step=1, dtype=int)]
    bootstrap = [True, False]
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }
    verbose = 2
    random_model = RandomForestRegressor(random_state=42)
    # 随机选择（随机搜索）
    print(f"\n====== 随机选择（随机搜索） ======")
    random_search = RandomizedSearchCV(estimator=random_model,
                                       param_distributions=params,
                                       n_iter=100,
                                       scoring="neg_mean_absolute_percentage_error",
                                       cv = 3,
                                       verbose = 2,
                                       n_jobs=-1,
                                       random_state=42)
    # random_search.fit(X_train, y_train)
    # best_params = random_search.best_params_
    # print(f"best_params: {best_params}")

    # 重新训练模型，验证结果
    model_1 = RandomForestRegressor(n_estimators=160, max_depth=18, min_samples_split=18, min_samples_leaf=5, bootstrap=True, random_state=42, n_jobs=-1)
    model_1.fit(X_train_best, y_train_best)
    y_pred_1 = model_1.predict(X_test_best)
    mape_score_best_1 = mean_absolute_percentage_error(y_test_best, y_pred_1)

    print(f"====== 随机选择（随机搜索），mape_score_best: {mape_score_best_1:.5f}")

    # 网格搜索（地毯式搜索）
    # 上一次的搜索结果：best_params: {'n_estimators': 160, 'min_samples_split': 18, 'min_samples_leaf': 5, 'max_depth': 18, 'bootstrap': True}
    print(f"\n====== 网格搜索（地毯式搜索），第1次 ======")
    grid_params_1 = {
        "n_estimators": [140, 160, 180],
        "max_depth": [16, 18, 20],
        "min_samples_split": [16, 18, 20],
        "min_samples_leaf": [4, 5, 6],
    }
    model_grid_search_1 = RandomForestRegressor(random_state=42)
    grid_search_1 = GridSearchCV(estimator=model_grid_search_1,
                                 param_grid=grid_params_1,
                                 scoring="neg_mean_absolute_percentage_error",
                                 n_jobs=-1,
                                 cv=3,
                                 verbose=2)
    # grid_search_1.fit(X_train, y_train)
    # best_params_1 = grid_search_1.best_params_
    # print(f"====== 网格搜索（地毯式搜索）第1次，best_params: {best_params_1}")

    # 网格搜索（地毯式搜索）第1次，best_params: {'max_depth': 20, 'min_samples_leaf': 5, 'min_samples_split': 20, 'n_estimators': 160}
    model_grid_1 = RandomForestRegressor(n_estimators=160, max_depth=20, min_samples_split=20, min_samples_leaf=5,
                                    bootstrap=True, random_state=42, n_jobs=-1)
    model_grid_1.fit(X_train_best, y_train_best)
    y_pred_grid_1 = model_grid_1.predict(X_test_best)
    mape_score_grid_1 = mean_absolute_percentage_error(y_test_best, y_pred_grid_1)

    print(f"====== 网格搜索（地毯式搜索）第1次，mape_score_best: {mape_score_grid_1:.5f}")

    print(f"\n====== 网格搜索（地毯式搜索），第2次 ======")
    grid_params_2 = {
        "n_estimators": [140, 160, 180],
        "max_depth": [18, 20, 22],
        "min_samples_split": [18, 20, 22],
        "min_samples_leaf": [4, 5, 6],
    }

    model_grid_search_2 = RandomForestRegressor(random_state=42)
    grid_search_2 = GridSearchCV(estimator=model_grid_search_2,
                                 param_grid=grid_params_2,
                                 scoring="neg_mean_absolute_percentage_error",
                                 n_jobs=-1,
                                 cv=3,
                                 verbose=2)
    # grid_search_2.fit(X_train_best, y_train_best)
    # best_params_2 = grid_search_2.best_params_
    # print(f"====== 网格搜索（地毯式搜索）第2次，best_params: {best_params_2}")

    # 网格搜索（地毯式搜索）第2次，best_params: {'max_depth': 18, 'min_samples_leaf': 5, 'min_samples_split': 22, 'n_estimators': 180}
    model_grid_2 = RandomForestRegressor(n_estimators=180, max_depth=18, min_samples_split=22, min_samples_leaf=5,
                                         bootstrap=True, random_state=42, n_jobs=-1)
    model_grid_2.fit(X_train_best, y_train_best)
    y_pred_grid_2 = model_grid_2.predict(X_test_best)
    mape_score_grid_2 = mean_absolute_percentage_error(y_test_best, y_pred_grid_2)

    print(f"====== 网格搜索（地毯式搜索）第2次，mape_score_best: {mape_score_grid_2:.5f}")

    print(f"\n====== 网格搜索（地毯式搜索），第3次 ======")
    grid_params_3 = {
        "n_estimators": [160, 180, 200],
        "max_depth": [16, 18, 20],
        "min_samples_split": [20, 22, 24],
        "min_samples_leaf": [4, 5, 6],
    }

    model_grid_search_3 = RandomForestRegressor(random_state=42)
    grid_search_3 = GridSearchCV(estimator=model_grid_search_3,
                                 param_grid=grid_params_3,
                                 scoring="neg_mean_absolute_percentage_error",
                                 n_jobs=-1,
                                 cv=3,
                                 verbose=2)
    # grid_search_3.fit(X_train_best, y_train_best)
    # best_params_3 = grid_search_3.best_params_
    # print(f"====== 网格搜索（地毯式搜索）第3次，best_params: {best_params_3}")

    # 网格搜索（地毯式搜索）第3次，best_params: {'max_depth': 16, 'min_samples_leaf': 5, 'min_samples_split': 24, 'n_estimators': 180}
    model_grid_3 = RandomForestRegressor(n_estimators=180, max_depth=16, min_samples_split=24, min_samples_leaf=5,
                                         bootstrap=True, random_state=42, n_jobs=-1)
    model_grid_3.fit(X_train_best, y_train_best)
    y_pred_grid_3 = model_grid_3.predict(X_test_best)
    mape_score_grid_3 = mean_absolute_percentage_error(y_test_best, y_pred_grid_3)

    print(f"====== 网格搜索（地毯式搜索）第3次，mape_score_best: {mape_score_grid_3:.5f}")

    print(f"\n====== 网格搜索（地毯式搜索），第4次 ======")
    grid_params_4 = {
        "n_estimators": [160, 180, 200],
        "max_depth": [14, 16, 18],
        "min_samples_split": [22, 24, 26],
        "min_samples_leaf": [4, 5, 6],
    }
    model_grid_search_4 = RandomForestRegressor(random_state=42)
    grid_search_4 = GridSearchCV(estimator=model_grid_search_4,
                                 param_grid=grid_params_4,
                                 scoring="neg_mean_absolute_percentage_error",
                                 n_jobs=-1,
                                 cv=3,
                                 verbose=2)
    # grid_search_4.fit(X_train_best, y_train_best)
    # best_params_4 = grid_search_4.best_params_
    # print(f"====== 网格搜索（地毯式搜索）第4次，best_params: {best_params_4}")

    # 网格搜索（地毯式搜索）第4次，best_params: {'max_depth': 14, 'min_samples_leaf': 5, 'min_samples_split': 26, 'n_estimators': 180}
    model_grid_4 = RandomForestRegressor(n_estimators=180, max_depth=14, min_samples_split=26, min_samples_leaf=5,
                                         bootstrap=True, random_state=42, n_jobs=-1)
    model_grid_4.fit(X_train_best, y_train_best)
    y_pred_grid_4 = model_grid_4.predict(X_test_best)
    mape_score_grid_4 = mean_absolute_percentage_error(y_test_best, y_pred_grid_4)

    print(f"====== 网格搜索（地毯式搜索）第4次，mape_score_best: {mape_score_grid_4:.5f}")

    print(f"\n====== 网格搜索（地毯式搜索），第5次 ======")
    grid_params_5 = {
        "n_estimators": [160, 180, 200],
        "max_depth": [12, 14, 16],
        "min_samples_split": [24, 26, 28],
        "min_samples_leaf": [4, 5, 6],
    }
    model_grid_search_5 = RandomForestRegressor(random_state=42)
    grid_search_5 = GridSearchCV(estimator=model_grid_search_5,
                                 param_grid=grid_params_5,
                                 scoring="neg_mean_absolute_percentage_error",
                                 n_jobs=-1,
                                 cv=3,
                                 verbose=2)
    # grid_search_5.fit(X_train_best, y_train_best)
    # best_params_5 = grid_search_5.best_params_
    # print(f"====== 网格搜索（地毯式搜索）第5次，best_params: {best_params_5}")

    # 网格搜索（地毯式搜索）第5次，best_params: {'max_depth': 12, 'min_samples_leaf': 6, 'min_samples_split': 28, 'n_estimators': 180}
    model_grid_5 = RandomForestRegressor(n_estimators=180, max_depth=12, min_samples_split=28, min_samples_leaf=6,
                                         bootstrap=True, random_state=42, n_jobs=-1)
    model_grid_5.fit(X_train_best, y_train_best)
    y_pred_grid_5 = model_grid_5.predict(X_test_best)
    mape_score_grid_5 = mean_absolute_percentage_error(y_test_best, y_pred_grid_5)

    print(f"====== 网格搜索（地毯式搜索）第5次，mape_score_best: {mape_score_grid_5:.5f}")

    title_6 = "网格搜索（地毯式搜索）第6次"
    grid_params_6 = {
        "n_estimators": [160, 180, 200],
        "max_depth": [10, 12, 14],
        "min_samples_split": [26, 28, 30],
        "min_samples_leaf": [4, 5, 6],
    }
    # grid_search(X_train_best, y_train_best, grid_params_6, title_6)

    # 网格搜索（地毯式搜索）第6次，best_params: {'max_depth': 12, 'min_samples_leaf': 5, 'min_samples_split': 30, 'n_estimators': 180}
    fit_predit(X_train_best, y_train_best, X_test_best, y_test_best, n_estimators=180, max_depth=12, min_samples_split=30, min_samples_leaf=5, bootstrap=True, title=title_6)
