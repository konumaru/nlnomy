import pandas as pd
import streamlit as st

def main():
    if 'data' not in st.session_state:
        st.session_state.data = []

    st.title('テキストコンテンツモデレーションデモ')

    user_input = st.text_area('テキストを入力してください:')

    if st.button('モデレーション'):
        st.session_state.data.append(user_input)

        moderated_text = moderate_content(user_input)
        st.write('モデレーション結果:', moderated_text)
    
    if user_input:
        df = pd.DataFrame(st.session_state.data, columns=['入力されたテキスト'])
        st.dataframe(df.iloc[::-1], hide_index=True)

def moderate_content(text):
    # ここで機械学習モデルを使ってテキストのモデレーションを行う
    # デモのため、ここではシンプルな例を示します
    if 'badword' in text:
        return '不適切なコンテンツが検出されました'
    else:
        return '問題なし'

if __name__ == '__main__':
    main()
