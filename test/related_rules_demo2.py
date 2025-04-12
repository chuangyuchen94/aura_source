from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
pd.options.display.max_colwidth = 100

retail = {
    "ID": [1, 2, 3, 4, 5, 6],
    "Basket" : [
        ["Beer", "Diaper", "Pretzels", "Chips", "Aspirin"],
        ["Diaper", "Beer", "Chips", "Lotion", "Juice", "BabyFood", "Milk"],
        ["Soda", "Chips", "Milk"],
        ["Soup", "Beer", "Diaper", "Milk", "IceCream"],
        ["Soda", "Coffee", "Milk", "Bread"],
        ["Beer", "Chips"],
    ]
}

retail = pd.DataFrame(retail)
print(retail)

split_str = ","
retail_basket = retail.Basket.str.join(split_str)
print(f"retail_basket \n{retail_basket}")

retail_basket_data = retail_basket.str.get_dummies(split_str)
print(f"retail_basket_data \n{retail_basket_data}")

retail_set = apriori(retail_basket_data, min_support=0.5, use_colnames=True)
print(f"retail_set \n{retail_set}")
retail_rule = association_rules(retail_set, metric="lift", min_threshold=1)
with pd.option_context(
    'display.max_columns', None,
    'display.max_rows', None,
    'display.max_colwidth', None,
    'display.width', 1000
):
    print(f"retail_rule \n{retail_rule}")