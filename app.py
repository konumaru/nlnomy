import pandas as pd
import streamlit as st

from src.utils import load_model, moderate_content, text2embeddings


def main() -> None:
    st.title("Text Moderation")

    user_input = st.text_area("検証したい文章を入力してください")

    st.session_state.data = [
        ["底知れないカリスマ性があるよな", "Not Toxic"],
        ["土人は怖いな", "Toxic"],
    ]

    if st.button("検証する"):
        moderated_result = moderate_content(
            user_input, "./data/models/model.pkl"
        )
        st.session_state.data.append([user_input, moderated_result])

    df = pd.DataFrame(st.session_state.data, columns=["InputText", "Result"])
    st.dataframe(df.iloc[::-1], hide_index=True, width=800)


if __name__ == "__main__":
    main()
