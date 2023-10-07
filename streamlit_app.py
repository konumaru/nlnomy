import os
from typing import List

import pandas as pd
import streamlit as st

from src.utils import load_model, moderate_content, save_model, text2embeddings


def text_moderation() -> None:
    st.subheader("Text Moderation")

    if "moderation" not in st.session_state:
        st.session_state["moderation"] = [
            ["底知れないカリスマ性があるよな", "Not Toxic"],
            ["土人は怖いな", "Toxic"],
        ]

    if "is_retrained" in st.session_state:
        model_filepath = "./data/models/retrained_model.pkl"
        st.write("Retrained model is used!")
    else:
        model_filepath = "./data/models/model.pkl"

    user_input = st.text_area("検証したい文章を入力してください")
    if st.button("検証する"):
        moderated_result = moderate_content(user_input, model_filepath)
        st.session_state["moderation"].append([user_input, moderated_result])

    df = pd.DataFrame(
        st.session_state["moderation"],
        columns=["InputText", "ToxicLevel"],
    )
    st.dataframe(df.iloc[::-1], hide_index=True, width=800)


def retraining_model(sentences: List[str], toxic_levels: List[str]) -> None:
    model = load_model("./data/models/model.pkl")
    cols_annotated = {
        level: i
        for i, level in enumerate(
            [
                "Not Toxic",
                "Hard to Say",
                "Toxic",
                "Very Toxic",
            ]
        )
    }

    X = text2embeddings(sentences)
    y = [cols_annotated[level] for level in toxic_levels]

    model.partial_fit(X, y)
    save_model(model, "./data/models/retrained_model.pkl")


def annotation() -> None:
    st.subheader("Annotation")

    if "annotation" not in st.session_state:
        st.session_state["annotation"] = []

    user_input = st.text_area("登録したい文章を入力してください")
    toxic_level = st.radio(
        "Set label of toxic level 👇",
        reversed(["Not Toxic", "Hard to Say", "Toxic", "Very Toxic"]),
    )

    if st.button("登録する"):
        st.session_state["annotation"].append([user_input, toxic_level])

    annotated_data = pd.DataFrame(
        st.session_state["annotation"],
        columns=["InputText", "ToxicLevel"],
    )
    st.data_editor(
        annotated_data,
        hide_index=True,
        width=800,
    )

    if st.button("再学習する"):
        retraining_model(
            annotated_data["InputText"].tolist(),
            annotated_data["ToxicLevel"].tolist(),
        )
        st.session_state["is_retrained"] = True
        st.write("Complete retraining!")


def main() -> None:
    st.title("nlnomy")

    text_moderation()
    annotation()


if __name__ == "__main__":
    main()
