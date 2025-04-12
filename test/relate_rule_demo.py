from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
"""
关联规则测试代码
"""

# 定义数据集
data = {
    "ID": [1, 2, 3, 4, 5, 6],
    "Onion": [1, 0, 0, 1, 1, 1],
    "Potato": [1, 1, 0, 1, 1, 1],
    "Burger": [1, 1, 0, 0, 1, 1],
    "Milk": [0, 1, 1, 1, 0, 1],
    "Beer": [0, 0, 1, 0, 1, 0],
}

df = pd.DataFrame(data)
print(df)

data_columns = ["Onion", "Potato", "Burger", "Milk", "Beer"]
related_rules = apriori(df[data_columns], min_support=0.5, use_colnames=True)
print(f"related_rules: \n{related_rules}")

sorted_rules = association_rules(related_rules, metric="lift", min_threshold=1)
print(f"sorted_rules: \n{sorted_rules}")
