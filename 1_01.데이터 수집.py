import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="데이터 수집",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",  # wide 또는 centered 중 선택
    initial_sidebar_state="auto",  # auto, collapsed, expanded 중 선택
    
)



st.title(':open_file_folder: 데이터를 업로드해주세요.')
st.write('데이터를 업로드합니다. 직접 파일을 업로드하거나 아래 예제 버튼을 클릭해주세요.')
uploaded_file = st.file_uploader('CSV파일을 업로드해주세요.')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.to_pickle('my_dataframe.pkl')
    st.write('데이터가 업로드 되었습니다!', ':arrow_left: :blue[<b>02.데이터 확인 및 탐색 페이지</b>]로 이동해주세요.',  unsafe_allow_html=True)
