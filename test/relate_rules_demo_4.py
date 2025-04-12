from my_models.related_rule import MyRelatedRuleModel
import pandas as pd
import numpy as np
def load_data_set():
    """
    加载数据集
    :return:
    """
    data_set = [
        ['1', '3', '4'],
        ['2', '3', '5'],
        ['1', '2', '3', '5'],
        ['2', '5'],
    ]
    return data_set

if "__main__" == __name__:
    data = load_data_set()
    print(f"data: {data}")

    data = pd.DataFrame(data, dtype=str)

    a = ('1', )
    b = ('1', '2', '3', None)
    if set(a).issubset(set(b)):
        print(f"a in b")

    relate_rule_model = MyRelatedRuleModel()
    relate_rule_model.build_related_rule(data, metric="support", min_threshold=0.5)
    print(f"relate_rule_model: {relate_rule_model}")
    relate_rule_model.print()
