import pandas as pd


def train_val_split():
    df = pd.read_csv("../data/labels.csv")
    df[:9000].to_csv("../data/train.csv", index=False)
    df[9000:10000].to_csv("../data/val.csv", index=False)


if __name__ == "__main__":
    train_val_split()
