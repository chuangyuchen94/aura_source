
def create_dataset():
    data_set = [
        [0, 0, 0, 0, "no"],
        [0, 0, 0, 1, "no"],
        [0, 1, 0, 1, "yes"],
        [0, 1, 1, 0, "yes"],
        [0, 0, 0, 0, "no"],
        [1, 0, 0, 0, "no"],
        [1, 0, 0, 1, "no"],
        [1, 1, 1, 1, "yes"],
        [1, 0, 1, 2, "yes"],
        [1, 0, 1, 2, "yes"],
        [2, 0, 1, 2, "yes"],
        [2, 0, 1, 1, "yes"],
        [2, 1, 0, 1, "yes"],
        [2, 1, 0, 2, "yes"],
        [2, 0, 0, 0, "no"],
    ]

    labels = ["F1-AGE", "F2-WORK", "F3-HOME", "F4-LOAN"]

    return data_set, labels

