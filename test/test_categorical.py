import pandas as pd
from sklearn.model_selection import train_test_split

all_data = pd.read_csv("../data/melb_data.csv")
print(all_data.head())
print(all_data.columns)
print(all_data.shape)

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
print(f"\n category_columns: {category_columns}")

s = (X_train.dtypes == 'object')
print(f"type of s: {type(s)}")
print(f"s: {s}")

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


if "__main__" == __name__:
    drop_categorial_columns(X_train, X_valid, y_train_full, y_valid_full, category_columns)
