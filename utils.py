import pandas as pd
from sklearn.utils import shuffle


def reduce_dataset(path):
    data = pd.read_csv(f"{path}")
    data = data[["content", "retweets"]]
    data = shuffle(data)
    data.to_csv("data/reduced_data.csv", index=False)
