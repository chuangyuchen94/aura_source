import pandas as pd

def analyze_data(data, min_var=100):
    """
    分析数据，得到以下统计信息，进行打印输出：\n
    特征个数 \n

    1. 数值型 \n
    1）最大值 \n
    2）最小值 \n
    3）均值 \n
    4）方差 \n

    2. 文本型 \n
    1）枚举值 \n
    2）枚举值的各自占比 \n

    3. 通用 \n
    1）字段（列）含义 \n
    2）字段是否存在缺失值 \n
    3）字段是否有异常值 \n
    :param data:
    :return: 字段统计信息
    """
    pd_data = pd.DataFrame(data)
    big_var_column = []  # 统计大方差字段

    print(f"\n总行数：{data.shape[0]} | 总列数：{data.shape[1]} \n")

    for column in pd_data.columns:
        column_info = ""
        column_info += f"字段：{column} \n"
        column_info += f"\t 字段类型：{pd_data[column].dtype} \n"
        column_info += f"\t 是否存在缺失值：{pd_data[column].isnull().any()} \n"
        column_info += f"\t 枚举值个数：{len(pd_data[column].unique())} \n"

        # 统计枚举值信息
        if len(pd_data[column].unique()) < 10 or pd_data[column].dtype in ["object", ]:
            column_info += f"\t 枚举值：{pd_data[column].unique()} \n"
            column_info += f"\t 枚举值个数：{pd_data[column].value_counts()} \n"
            column_info += f"\t 枚举值占比：{pd_data[column].value_counts() / len(pd_data[column])} \n"

        # 统计数值型信息
        if pd_data[column].dtype in ["int64", "float64"]:
            column_info += f"\t 最大值：{pd_data[column].max()} \n"
            column_info += f"\t 最小值：{pd_data[column].min()} \n"
            column_info += f"\t 均值：{pd_data[column].mean()} \n"
            column_info += f"\t 方差：{pd_data[column].var()} \n"
        else:
            column_info += f"\t 最大值：{pd_data[column].max()} \n"

        # 统计大方差字段
        if pd_data[column].dtype in ["int64", "float64"]:
            var_value = pd_data[column].var()
            if  var_value >= min_var:
                big_var_column.append((column, var_value))

        print(column_info)

    return big_var_column
