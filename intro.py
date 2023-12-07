import streamlit as st


st.set_page_config(
    page_title="머신러닝 웹",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",  # wide 또는 center 중 선택
    initial_sidebar_state="auto",  # auto, collapsed, expanded 중 선택
    
)



image_path = 'mlflow-2.png' # 이미지의 경로
st.title(':rainbow: :rainbow[Data Analyze & ML Web]')
st.write('데이터를 업로드하고 시각화 후 머신러닝으로 모델을 학습시켜보는 웹사이트 입니다.')
st.write('''그림은 머신러닝의 전반적인 흐름입니다. 
        그림에 나와있는 흐름이 왼쪽 사이드바에 순서대로 표현되어 있으니, 순서에 맞춰 잘따라와주세요. 시작은 데이터 업로드 부터입니다.''')
st.image(image_path, caption='사진출저:ml-ops.org', width=900)




