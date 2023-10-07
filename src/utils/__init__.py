import pickle
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def save_model(model: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> Any:
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def moderate_content(text: str, model_path: str) -> str:
    model = load_model(model_path)

    annotated_labels = [
        "Not Toxic",
        "Hard to Say",
        "Toxic",
        "Very Toxic",
    ]
    embeddings = text2embeddings([text])
    pred = model.predict(embeddings)[0]
    return annotated_labels[pred]


def mean_pooling(model_output, attention_mask) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


@torch.no_grad()
def text2embeddings(
    sentences: List[str],
    model_name_or_path: str = "rinna/japanese-roberta-base",
    batch_size: int = 16,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)

    embeddings = []
    for i in range(0, len(sentences), batch_size):
        encoded_input = tokenizer(
            sentences[i : i + batch_size],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(
            model_output,
            encoded_input["attention_mask"],
        )
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        embeddings.append(sentence_embeddings)

    results = torch.cat(embeddings, dim=0)
    return results.numpy()
