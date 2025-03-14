import pandas as pd
from sklearn.model_selection import train_test_split

all_data = pd.read_csv("../data/melb_data.csv")
# print(all_data.head())
# print(all_data.columns)
# print(all_data.shape)

# 1. 清除价格列为空的行
all_data.dropna(axis=0, subset=["Price"], inplace=True)
print(f"all_data.shape: {all_data.shape}")

y_all = all_data.Price
X_all = all_data.drop(axis=1, labels=["Price"], inplace=False)
print(f"X_all's shape: {X_all.shape}")
print(f"y_all's shape: {y_all.shape}")

X_train_full, X_valid_full, y_train_full, y_valid_full = train_test_split(X_all, y_all, train_size=0.8, random_state=0)
print(f"X_train_full's shape: {X_train_full.shape}")
print(f"X_valid_full's shape: {X_valid_full.shape}")
print(f"y_train_full's shape: {y_train_full.shape}")
print(f"y_valid_full's shape: {y_valid_full.shape}")

# 2. 去掉存在空值的列
null_columns = X_train_full.columns[X_train_full[X_train_full.columns].isnull().any()]
# cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]

print(f"null_columns: {null_columns}")
# print(f"cols_with_missing: {cols_with_missing}")

X_train_full.drop(axis=1, labels=null_columns, inplace=True)
X_valid_full.drop(axis=1, labels=null_columns, inplace=True)

print(f"X_train_full's shape: {X_train_full.shape}")
print(f"X_valid_full's shape: {X_valid_full.shape}")

# 3. 挑出分类变量的列
low_categorical_columns = [col for col in X_train_full.columns if X_train_full[col].nunique() < 10 and X_train_full[col].dtype == "object"]
print(f"low_categorical_columns: {low_categorical_columns}")

# 4. 挑出数值列
numerical_columns = [col for col in X_train_full.columns if X_train_full[col].dtype in ["int64", "float64"]]
print(f"numerical_columns: {numerical_columns}")

# 5. 挑出可用于预测价格的列
my_columns = low_categorical_columns + numerical_columns
X_train = X_train_full[my_columns].copy()
X_valid = X_valid_full[my_columns].copy()

print("\n")
print(f"X_train's shape: {X_train.shape}")
print(f"X_valid's shape: {X_valid.shape}")

category_columns = [col for col in X_train.columns if X_train[col].dtype == "object"]
# print(f"\n category_columns: {category_columns}")

s = (X_train.dtypes == 'object')
# print(f"type of s: {type(s)}")
# print(f"s: {s}")

def score_dataset(X_train, X_valid, y_train, y_valid):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(preds, y_valid)

def drop_categorial_columns(X_train, X_valid, y_train, y_valid, categorial_columns):
    drop_X_train = X_train.drop(axis=1, labels=categorial_columns, inplace=False)
    drop_X_valid = X_valid.drop(axis=1, labels=categorial_columns, inplace=False)

    mea = score_dataset(drop_X_train, drop_X_valid, y_train, y_valid)
    print(f"drop_categorial_columns' score: {mea}")

def ordinal_encode_test(X_train, X_valid, y_train, y_valid, categorial_columns):
    from sklearn.preprocessing import OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()

    categorial_X_train = X_train.copy()
    categorial_X_valid = X_valid.copy()

    categorial_X_train[category_columns] = ordinal_encoder.fit_transform(categorial_X_train[categorial_columns])
    categorial_X_valid[categorial_columns] = ordinal_encoder.transform((categorial_X_valid[categorial_columns]))

    mea = score_dataset(categorial_X_train, categorial_X_valid, y_train, y_valid)
    print(f"ordinal_encode_test score: {mea}")

def one_hot_encode_test(X_train, X_valid, y_train, y_valid, categorial_columns):
    from sklearn.preprocessing import OneHotEncoder
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    categorial_columns_X_train = pd.DataFrame(one_hot_encoder.fit_transform(X_train))
    categorial_columns_X_valid = pd.DataFrame(one_hot_encoder.transform(X_valid))

    categorial_columns_X_train.index = X_train[categorial_columns].index
    categorial_columns_X_valid.index = X_valid[categorial_columns].index

    numberial_columns_X_train = X_train.drop(axis=1, labels=categorial_columns, inplace=False)
    numberial_columns_X_valid = X_valid.drop(axis=1, labels=categorial_columns, inplace=False)

    merged_X_train = pd.concat([categorial_columns_X_train, numberial_columns_X_train], axis=1)
    merged_X_valid = pd.concat([categorial_columns_X_valid, numberial_columns_X_valid], axis=1)

    merged_X_train.columns = merged_X_train.columns.astype(str)
    merged_X_valid.columns = merged_X_valid.columns.astype(str)

    mae = score_dataset(merged_X_train, merged_X_valid, y_train, y_valid)
    print(f"one_hot_encode_test score: {mae}")

def data_preprocess_modeling(num_cols, cat_cols):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    # 处理分类特征列: 填充 + 独热编码
    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_transformer = Pipeline(
        steps = [
            ("imputer", cat_imputer),
            ("one_hot_encoder", cat_one_hot_encoder)
        ]
    )

    # 处理数值型的缺失值：填充
    num_imputer = SimpleImputer(strategy="mean")

    # 汇总，定义列的预处理
    column_transformer = ColumnTransformer(
        transformers = [
            ("number_precessing", num_imputer, num_cols),
            ("cat_processing", cat_transformer, cat_cols)
        ]
    )

    # 定义模型
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=0)

    # 定义：数据预处理 + 模型
    preprocessing_model_pipeline = Pipeline(
        steps = [
            ("processing", column_transformer),
            ("model and predict", model)
        ]
    )

    return preprocessing_model_pipeline


if "__main__" == __name__:
    # ordinal_encode_test(X_train, X_valid, y_train_full, y_valid_full, category_columns)
    # drop_categorial_columns(X_train, X_valid, y_train_full, y_valid_full, category_columns)
    # ordinal_encode_test(X_train, X_valid, y_train_full, y_valid_full, category_columns)
    # one_hot_encode_test(X_train, X_valid, y_train_full, y_valid_full, category_columns)
    from sklearn.metrics import mean_absolute_error

    # 分别出去数值特征列和分类特征列
    num_cols = [col for col in X_train.columns if X_train[col].dtype in ["int64", "float64"]]
    cat_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

    pipeline = data_preprocess_modeling(num_cols, cat_cols)
    pipeline.fit(X_train, y_train_full)
    preds = pipeline.predict(X_valid)
    mea = mean_absolute_error(preds, y_valid_full)

    print(f"mea = {mea}")