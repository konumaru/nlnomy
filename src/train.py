import re
import string
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.datasets import make_multilabel_classification
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def text2embeddings(
    sentence: str, model_name_or_path: str = "roberta-base"
) -> List[float]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)

    encoded_input = tokenizer(
        sentence,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(
        model_output, encoded_input["attention_mask"]
    )

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


def main() -> None:
    data = pd.read_csv(
        "data/toxic-comment-classification-challenge/train.csv", nrows=10
    )

    col_text = "comment_text"
    cols_target = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    data[col_text].fillna("unknown", inplace=True)
    data[col_text] = data[col_text].str.replace("\n", " ")
    print(data.head())

    input_text = data[col_text].tolist()[0]
    print(input_text)

    embeddings = text2embeddings(input_text)
    print(embeddings)

    # X, y = make_multilabel_classification(
    #     n_classes=len(cols_target), random_state=0
    # )
    # clf = MultiOutputClassifier(LogisticRegression())
    # clf.fit(X, y)

    # preds = np.array(clf.predict_proba(X[-10:]))[:, :, 1].T
    # print(pd.DataFrame(preds, columns=cols_target))


if __name__ == "__main__":
    main()
