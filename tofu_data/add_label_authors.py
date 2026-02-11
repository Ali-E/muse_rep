import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("authors_paragraphs.csv")
    print(df.head())
    print(len(df))

    df["label"] = df["id"]
    
    print(df.head(25))
    df.to_csv("authors_paragraphs_labeled.csv", index=False)