import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("tofu_full_train.csv")
    print(df.head())
    print(len(df))

    df["label"] = (df.index // 20).astype(int)
    
    print(df.head(25))
    df.to_csv("tofu_labeled_train.csv", index=True)