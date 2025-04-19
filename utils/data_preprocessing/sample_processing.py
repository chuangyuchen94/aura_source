import pandas as pd
from imblearn.over_sampling import SMOTE
def sampling_data(data, column, method="over", X_label=None, X_exclude=None, y_label=None, random_state=42):
    """
    对数据进行采样处理 \n
    处理方法包括：\n
        下采样：under \n
        过采样：over \n
    :param data: 数据
    :param method: 采样所依据的列
    :return: 采样后的数据
    """
    if "under" == method:
        column_count = data[column].value_counts()
        min_count = column_count.min()
        unique_value = data[column].unique()

        new_data = []
        for column_value in unique_value:
            current_data = data[data[column] == column_value].sample(n=min_count, random_state=random_state)
            new_data.append(current_data)

        return pd.concat(new_data)

    if "over" == method:
        smote_sample = SMOTE(random_state=random_state)
        data_X, data_y = None, None
        if X_label is not None:
            data_X, data_y = smote_sample.fit_resample(data[X_label], data[y_label])
        elif X_exclude is not None:
            data_X, data_y = smote_sample.fit_resample(data.drop(X_exclude, axis=1), data[y_label])
        return pd.concat([data_X, data_y], axis=1)

    return None