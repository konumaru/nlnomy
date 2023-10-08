from .feature import text2embeddings
from .io import load_model, save_model

__all__ = ["text2embeddings", "load_model", "save_model"]


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
