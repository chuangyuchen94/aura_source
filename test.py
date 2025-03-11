import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

X_full = pd.read_csv("data/train.csv")
y_full = X_full[["SalePrice"]]

X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)
y = X_full.SalePrice
X_full.drop(labels=["SalePrice"], axis=1, inplace=True)

X = X_full.select_dtypes(exclude=["object"])
#print("X_full")
#print(X_full)
print(f"X.shape = {X_full.shape}")
#print("y_full")
#print(y_full)
print(f"y.shape = {y_full.shape}")

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=0)

print(f"X_train's type: {type(X_train)}\n")

missing_value_count_by_column = X.isna().sum()
print(f"type of missing_value_count_by_column: {type(missing_value_count_by_column)}")
# print(missing_value_count_by_column)
# print(missing_value_count_by_column[missing_value_count_by_column > 0])

name_of_missing_columns = missing_value_count_by_column[missing_value_count_by_column > 0]
print(f"\n name_of_missing_columns: \n{name_of_missing_columns}")

imputer = SimpleImputer()
# imputer.fit_transform()