from typing import Any, List

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from utils import save_model, text2embeddings


def main() -> None:
    col_text = "text"
    cols_annotated_cnt = [
        "Not Toxic",
        "Hard to Say",
        "Toxic",
        "Very Toxic",
    ]
    data = pd.read_csv("data/train/subset.csv")
    data = data.assign(
        toxic_level=data[cols_annotated_cnt]
        .idxmax(axis=1)
        .map({name: i for i, name in enumerate(cols_annotated_cnt)})
    )

    embeddings = text2embeddings(data[col_text].tolist())
    X = embeddings
    y = data[["toxic_level"]].to_numpy()

    model = SGDClassifier()
    model.fit(X, y)
    save_model(model, "./data/models/model.pkl")

    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    print(acc)


if __name__ == "__main__":
    main()
