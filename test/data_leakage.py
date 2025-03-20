import pandas as pd

credit_card_data = pd.read_csv("../data/AER_credit_card_data.csv", true_values=["yes"], false_values=["no"])
print(credit_card_data.head())
print(credit_card_data.columns)
print(f"credit_card_data shape:{credit_card_data.shape}")

y = credit_card_data.card
X = credit_card_data.drop(labels=["card"], axis=1)
print(f"X shape:{X.shape}")