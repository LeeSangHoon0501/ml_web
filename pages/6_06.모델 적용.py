import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import streamlit as st
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib



st.set_page_config(
    page_title="머신러닝",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",  # wide 또는 center 중 선택
    initial_sidebar_state="auto",  # auto, collapsed, expanded 중 선택
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'

x_train_final = pd.read_pickle('x_train_final.pkl')
y_train_final = pd.read_pickle('y_train_final.pkl')
x_test = pd.read_pickle('x_test.pkl')
y_test = pd.read_pickle('y_test.pkl')

feature_data = pd.read_pickle('feature_data.pkl')

loaded_model = joblib.load('model.joblib')


if st.session_state.poly_degree is not None:
    pf = PolynomialFeatures(degree=st.session_state.poly_degree)
    x_test_poly = pf.fit_transform(x_test)
    y_test_pred = loaded_model.predict(x_test_poly)

elif st.session_state.poly_degree is None:
    y_test_pred = loaded_model.predict(x_test)

def evalulate():
    evalulate_type = {
    '평가지표를 선택해주세요': 'placeholder',
    'MAE(Mean Absolute Error)': 'MAE',
    'MSE(Mean Squared Error)': 'MSE',
    'RMSE(Root MeanSquared Error)': 'RMSE',
    '결정 계수(R-squared)': 'R-squared'}

    def evaluate_type(select):
        if select != 'RMSE':
            test_value = select(y_test, y_test_pred)
            st.write('Test', test_evalulate_select, ':%.4f'  % test_value)
        elif select == 'RMSE':
            test_value = select(y_test, y_test_pred)
            test_rmse =  np.sqrt(test_value)
            st.write('Train', test_evalulate_select, ':%.4f'  % test_rmse)

    test_evalulate_select = st.selectbox('모델의 평가지표를 선택해주세요.', list(evalulate_type.keys()))
    if evalulate_type[test_evalulate_select] == 'placeholder':
        st.write('평가지표가 선택되지 않았습니다.')
    elif evalulate_type[test_evalulate_select] == 'MAE':
        evaluate_type(mean_absolute_error)
    elif evalulate_type[test_evalulate_select] == 'MSE':
        evaluate_type(mean_squared_error)
    elif evalulate_type[test_evalulate_select] == 'RMSE':
        evaluate_type(mean_squared_error)
    elif evalulate_type[test_evalulate_select] == 'R-squared':
        evaluate_type(r2_score)




def classification_evalulate():
    # 정확도 계산
    accuracy = accuracy_score(y_test, y_test_pred)
    st.write("정확도:%.4f" %accuracy)

    # # 혼동 행렬 (Confusion Matrix) 출력
    # confusion_mat = confusion_matrix(y_test, y_test_pred)
    # st.write("혼동 행렬:")
    # st.write(confusion_mat)

    # # 분류 보고서 (Classification Report) 출력
    # class_report = classification_report(y_test, y_test_pred)
    # st.write("분류 보고서:")
    # st.write(class_report)




st.header(':desktop_computer: 훈련된 모델의 적용')
st.divider()

col1, col2 = st.columns([1, 2])
with col1:
    st.write(':heavy_check_mark: <b>훈련된 모델의 평가</b>', unsafe_allow_html=True)
    evalulate()
# classification_evalulate()
with col2:
    st.write(':heavy_check_mark: <b>예측한데이터와 훈련데이터의 비교</b>', unsafe_allow_html=True)
    x_test_data = st.selectbox('Feature를 선택해주세요', list(x_test.columns))

    fig, ax = plt.subplots()
    sns.scatterplot(x=x_test[x_test_data], y=y_test, data=x_test, label='test')
    sns.scatterplot(x=x_test[x_test_data], y=y_test_pred, data=x_test, label='test_pred')  
    ax.set_xlabel(x_test_data)
    ax.set_ylabel(y_test.name)
    ax.set_title(f'"{y_test.name} - {x_test_data}" Scatter Plot')
    ax.legend()
    st.pyplot(fig)