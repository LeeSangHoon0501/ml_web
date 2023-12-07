import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import joblib
from sklearn.model_selection import KFold




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
x_train = pd.read_pickle('x_train.pkl')
y_train = pd.read_pickle('y_train.pkl')
x_val = pd.read_pickle('x_val.pkl')
y_val = pd.read_pickle('y_val.pkl')



feature_data = pd.read_pickle('feature_data.pkl')



y_train_pred = None
y_val_pred = None

def linear_Regression_type(linear_type):
    global y_train_pred, y_val_pred, weight, bias
    linear_type.fit(x_train_final, y_train_final)
    weight = np.array(linear_type.coef_)
    bias = np.array(linear_type.intercept_)
    y_train_pred = linear_type.predict(x_train_final)
    y_val_pred = linear_type.predict(x_val)
    joblib.dump(linear_type, 'model.joblib')

    return y_train_pred, y_val_pred, weight, bias

def non_linear_Regression_type(non_linear_type):
    global y_train_pred, y_val_pred, weight
    non_linear_type.fit(x_train_final, y_train_final)
    y_train_pred = non_linear_type.predict(x_train_final)
    y_val_pred = non_linear_type.predict(x_val)
    weight = 'None'
    joblib.dump(non_linear_type, 'model.joblib')

    return y_train_pred, y_val_pred, weight

def evaluate():
    evalulate_type = {
    '평가지표를 선택해주세요': 'placeholder',
    'MAE(Mean Absolute Error)': 'MAE',
    'MSE(Mean Squared Error)': 'MSE',
    'RMSE(Root MeanSquared Error)': 'RMSE',
    '결정 계수(R-squared)': 'R-squared'}
    
    def evaluate_type(select):
        if select != 'RMSE':
            train_value = select(y_train_final, y_train_pred)
            val_value = select(y_val, y_val_pred)
            st.write('Train', evalulate_select, ':%.4f'  % train_value)
            st.write('Val', evalulate_select, ':%.4f'  % val_value)
        elif select == 'RMSE':
            train_value = select(y_train_final, y_train_pred)
            val_value = select(y_val, y_val_pred)
            train_rmse =  np.sqrt(train_value)
            val_rmse =  np.sqrt(val_value)
            st.write('Train', evalulate_select, ':%.4f'  % train_rmse)
            st.write('Val', evalulate_select, ':%.4f'  % val_rmse)


    col1, col2 = st.columns([1, 1])
    with col1:
        evalulate_select = st.selectbox('모델의 평가지표를 선택해주세요.', list(evalulate_type.keys()))
        
    if evalulate_type[evalulate_select] == 'placeholder':
        st.write('평가지표가 선택되지 않았습니다.')
    elif evalulate_type[evalulate_select] == 'MAE':
        evaluate_type(mean_absolute_error)
    elif evalulate_type[evalulate_select] == 'MSE':
        evaluate_type(mean_squared_error)
    elif evalulate_type[evalulate_select] == 'RMSE':
        evaluate_type(mean_squared_error)
    elif evalulate_type[evalulate_select] == 'R-squared':
        evaluate_type(r2_score)    

        


y_vali_pred = None
def kfold_evaluate(classification_model):
    global x_tr, y_tr, x_vali, y_vali, y_vali_pred, y_tr_pred
    val_score = []
    num_fold = 1
    kfold_n_value = st.slider('폴드 갯수를 설정해주세요', 2, 10, 2)
    kfold = KFold(n_splits= kfold_n_value, shuffle=True, random_state=12)    
    #훈련용 데이터와 검증용 데이터의 행 인덱스를 각 Fold별로 구분하여 생성
    for tr_idx, val_idx in kfold.split(x_train, y_train):
        # 훈련용 데이터와 검증용 데이터를 행 인덱스 기준으로 추출
        x_tr, x_vali = x_train.iloc[tr_idx, :], x_train.iloc[val_idx, :]
        y_tr, y_vali = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        classification_model.fit(x_tr, y_tr)
        y_tr_pred = classification_model.predict(x_tr)
        y_vali_pred = classification_model.predict(x_vali)
        vali_acc = accuracy_score(y_vali, y_vali_pred)
        st.write('%d. Fold Accuracy: %.4f' %(num_fold, vali_acc))
        val_score.append(vali_acc)
        num_fold += +1
        joblib.dump(classification_model, 'model.joblib')
    mean_score = np.mean(vali_acc)    
    st.write('평균 Accuracy:', np.round(mean_score, 4))    
        
    return x_tr, y_tr, x_vali, y_vali, y_vali_pred, y_tr_pred    




ml_type = None
learn_type = None
st.header(':desktop_computer: 머신러닝시작해 봅니다.')
st.divider()
st.write(':heavy_check_mark: <b>머신러닝의 종류를 선택하세요.</b>', unsafe_allow_html=True)
col1, col2, = st.columns([1, 10])
with col1:
    st.write('<b>학습의 종류</b>', unsafe_allow_html=True)
with col2:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button('지도학습'):
            st.session_state.learn_type = '지도학습'
    with col2:
        if st.button('비지도 학습'):
            st.session_state.learn_type = '비지도학습'
    with col3:
        if st.button('강화학습'):
            st.session_state.learn_type = '강화학습'

try: learn_type = st.session_state.learn_type
except AttributeError:
    st.warning('학습의 종류가 선택되지 않았습니다.')

if learn_type == '지도학습':        
    col1, col2, = st.columns([1, 10])
    with col1:
        st.write('<b>학습의 목적</b>', unsafe_allow_html=True)
    with col2:
        col1, col2= st.columns([1, 1])
        with col1:
            if st.button('예측(Regression)'):
                st.session_state.ml_type = 'Regression'
        with col2:
            if st.button('분류(Classfication)'):
                st.session_state.ml_type = 'Classfication'
    st.markdown('<hr>', unsafe_allow_html=True)
    try: ml_type = st.session_state.ml_type
    except AttributeError:
        st.warning('머신러닝의 학습 유형이 선택되지 않았습니다.')
    
if ml_type == 'Regression':
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write(':heavy_check_mark: <b>회귀(예측)모델을 훈련하고, 성능을 평가해봅니다.</b>', unsafe_allow_html=True)
        regression_type= {        
            '회귀의 종류를 선택해주세요': 'placeholder',
            '선형회귀(Linear)': 'Linear',
            '비선형회귀(Non-Linear)': 'Non_Linear',
        }
        st.session_state.poly_degree = None
        weight = None
        bias = None
        weight_pf = None
        bias_pf = None
        feature_importances = None
        regression_type_select = st.selectbox('회귀의 종류를 선택해주세요', list(regression_type.keys()))
        if regression_type[regression_type_select] == 'Linear': 
            linear_model_type = {
            '모델의 종류를 선택해주세요': 'placeholder',
            '선형회귀모델(LinearRegression)': 'LinearRegression',
            '리드지모델(Ridge)': 'Ridge', 
            '라쏘모델(Lasso)': 'Lasso', 
            '엘라스틱넷(ElasticNet))': 'ElasticNet'}

            linear_model_select = st.selectbox('사용할 선형회귀모델을 선택해주세요', list(linear_model_type.keys()))
            if linear_model_type[linear_model_select] ==  'LinearRegression':
                lr = LinearRegression()
                linear_Regression_type(lr)


            elif linear_model_type[linear_model_select] == 'Ridge':
                rdg_alpha_value = st.slider('알파 값을 설정해주세요', 0.0, 100.0, 1.0, 0.1)  # 알파 값을 사용자로부터 입력받음
                rdg = Ridge(alpha=rdg_alpha_value)
                linear_Regression_type(rdg)
                
            elif linear_model_type[linear_model_select] ==  'Lasso':
                las_alpha_value = st.slider('알파 값을 설정해주세요', 0.0, 100.0, 1.0, 0.1)  # 알파 값을 사용자로부터 입력받음
                las = Lasso(alpha=las_alpha_value)
                linear_Regression_type(las)
                
            elif linear_model_type[linear_model_select] ==  'ElasticNet':
                els_alpha_value = st.slider('알파 값을 설정해주세요', 0.1, 10.0, 1.0)  # 알파 값을 사용자로부터 입력받음
                l1_ratio_value = st.slider('L1 비율 값을 설정해주세요', 0.1, 1.0, 0.1)  # L1으 비율 값을 사용자로부터 입력받음
                els = ElasticNet(alpha=els_alpha_value, l1_ratio=l1_ratio_value)
                linear_Regression_type(els)
                

        elif regression_type[regression_type_select] == 'Non_Linear':
            non_linear_model_type = {
                '모델의 종류를 선택해주세요': 'placeholder',
                '다항회귀(PolynomialFeature)': 'PolynomialFeature',
                '의사결정나무(DecisionTreeRegressor)': 'Decision',
                '랜덤포레스트(RandomForestRegressor)': 'Random',}

            non_linear_model_select = st.selectbox('사용할 비선형회귀모델을 선택해주세요', list(non_linear_model_type.keys()))
            if non_linear_model_type[non_linear_model_select] == 'PolynomialFeature':
                st.session_state.poly_degree = st.slider('항의 갯수를 설정해주세요', 0, 15, 1)
                lr = LinearRegression()
                pf = PolynomialFeatures(degree=st.session_state.poly_degree)
                x_train_poly = pf.fit_transform(x_train_final)
                x_train_poly_df = pd.DataFrame(x_train_poly)
                x_val_poly = pf.fit_transform(x_val)
                lr.fit(x_train_poly, y_train_final)
                weight_pf = np.array(lr.coef_)
                bias_pf = np.array(lr.intercept_)
                y_train_pred = lr.predict(x_train_poly)
                y_val_pred = lr.predict(x_val_poly)
                joblib.dump(lr, 'model.joblib')
            elif non_linear_model_type[non_linear_model_select] == 'Decision':
                dtr_depth_value = st.slider('트리의 깊이를 설정해주세요', 1, 20, 1)
                dtr = DecisionTreeRegressor(random_state=12, max_depth= dtr_depth_value)
                non_linear_Regression_type(dtr)


            elif non_linear_model_type[non_linear_model_select] == 'Random':
                rfr_depth_value = st.slider('트리의 깊이를 설정해주세요', 1, 20, 1)
                rfr = RandomForestRegressor(random_state=12, max_depth= rfr_depth_value)
                non_linear_Regression_type(rfr)
    with col2:
        st.write(':heavy_check_mark: <b>훈련된 모델의 가중치와 바이어스값</b>', unsafe_allow_html=True)
        if weight is None and weight_pf is None:
            st.warning('모델이 선택되지 않았습니다.')
        elif weight_pf is not None:    
            result_dataframe = pd.DataFrame({'Feature': x_train_poly_df.columns, 'Weight': weight_pf, 'Bias': bias_pf})
            st.write(result_dataframe)
        elif weight_pf is None:
            result_dataframe = pd.DataFrame({'Feature': x_train_final.columns, 'Weight': weight, 'Bias': bias})
            st.write(result_dataframe)

            
    st.divider()
    st.write(':heavy_check_mark: <b>훈련된 모델의 평가</b>', unsafe_allow_html=True)
    if y_val_pred is not None:
        evaluate()
    else:
        st.warning("모델을 먼저 훈련하십시오.") 
        

    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(':heavy_check_mark: <b>훈련된 데이터의 시각화</b>', unsafe_allow_html=True)
        x_val_data = st.selectbox('Feature를 선택해주세요', list(x_val.columns))

    if y_val_pred is not None:    
        fig_1, ax = plt.subplots()
        sns.scatterplot(x=x_train_final[x_val_data], y=y_train_final, data=x_train_final, label='train')
        sns.scatterplot(x=x_train_final[x_val_data], y=y_train_pred, data=x_train_final, label='train_pred')
        ax.set_xlabel(x_val_data)
        ax.set_ylabel(y_train_final.name)
        ax.set_title(f'"{y_train_final.name} - {x_val_data}" Train Data')
        ax.legend()
        fig_2, ax = plt.subplots()
        sns.scatterplot(x=x_val[x_val_data], y=y_val, data=x_val, label='val')
        sns.scatterplot(x=x_val[x_val_data], y=y_val_pred, data=x_val, label='val_pred')
        ax.set_xlabel(x_val_data)
        ax.set_ylabel(y_val.name)
        ax.set_title(f'"{y_val.name} - {x_val_data}" Val Data')
        ax.legend()
        col1, col2 = st.columns([1, 1])
        with col1:
            st.pyplot(fig_1)
        with col2:
            st.pyplot(fig_2)        

    else:
        st.warning("모델을 먼저 훈련하십시오.")


elif ml_type == 'Classfication':
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(':heavy_check_mark: <b>분류모델을 훈련하고, 성능을 평가해봅니다.</b>', unsafe_allow_html=True)        
        nonlinear_model_type = {
        '모델의 종류를 선택해주세요': 'placeholder',
        'KNN(K-Nearest-Neighbors)': 'KNN',
        '의사결정나무(DecisionTree)': 'DecisionTree',
        '랜덤 포레스트(RandomForest)': 'RandomForest',
        'SVM(서포트 벡터 머신)': 'SVM',  
        '로지스틱 회귀(Logistic)': 'Logistic'}

        non_linear_model_select = st.selectbox('사용할 비선형회귀모델을 선택해주세요', list(nonlinear_model_type.keys()))

        if nonlinear_model_type[non_linear_model_select] == 'KNN':
            k_value = st.slider('K값을 설정해주세요', 1, 10, 1) 
            knn = KNeighborsClassifier(n_neighbors = k_value)
        elif nonlinear_model_type[non_linear_model_select] == 'DecisionTree':
            dtc_depth_value = st.slider('트리의 깊이를 설정해주세요', 1, 20, 1)
            dtc = DecisionTreeClassifier(random_state=12, max_depth= dtc_depth_value)
        elif nonlinear_model_type[non_linear_model_select] == 'RandomForest':
            rfc_depth_value = st.slider('트리의 깊이를 설정해주세요', 1, 20, 1)
            rfc = RandomForestClassifier(random_state=12, max_depth= rfc_depth_value)
        elif nonlinear_model_type[non_linear_model_select] == 'SVM':
            svc = SVC(kernel='rbf')          
        elif nonlinear_model_type[non_linear_model_select] == 'Logistic':
            lrc = LogisticRegression()
    with col2:
        st.write(':heavy_check_mark: <b>K_Fold를 통해 모델의 성능을 평가해봅니다.</b>', unsafe_allow_html=True)
        if nonlinear_model_type[non_linear_model_select] == 'placeholder':
            st.warning('모델을 선택해주세요.')
        elif nonlinear_model_type[non_linear_model_select] == 'KNN':
            kfold_evaluate(knn)
        elif nonlinear_model_type[non_linear_model_select] == 'DecisionTree':
            kfold_evaluate(dtc)
        elif nonlinear_model_type[non_linear_model_select] == 'RandomForest':
            kfold_evaluate(rfc)
        elif nonlinear_model_type[non_linear_model_select] == 'SVM':
            kfold_evaluate(svc)            
        elif nonlinear_model_type[non_linear_model_select] == 'Logistic':
            kfold_evaluate(lrc)  



    st.divider()        
    st.write(':heavy_check_mark: <b>훈련한 데이터 시각화</b>', unsafe_allow_html=True)
    if y_vali_pred is not None:
        x_vali_data = st.selectbox('Feature를 선택해주세요', list(x_vali.columns))
        fig_1, ax = plt.subplots()
        sns.scatterplot(x=y_tr, y=x_tr[x_vali_data], data=x_tr, label='train')
        sns.scatterplot(x=y_tr_pred, y=x_tr[x_vali_data], data=x_tr, label='train_pred')
        ax.set_xlabel(y_val.name)
        ax.set_ylabel(x_vali_data)
        ax.legend()
        ax.set_title(f'"{y_val.name} - {x_vali_data}" Plot')
        fig_2, ax = plt.subplots()
        sns.scatterplot(x=y_vali, y=x_vali[x_vali_data], data=x_vali, label='val')
        sns.scatterplot(x=y_vali_pred, y=x_vali[x_vali_data], data=x_vali, label='val_pred')
        ax.set_xlabel(y_val.name)
        ax.set_ylabel(x_vali_data)
        ax.legend()
        ax.set_title(f'"{y_val.name} - {x_vali_data}" Plot')
        col1, col2 = st.columns([1, 1])
        with col1:
            st.pyplot(fig_1)
        with col2:
            st.pyplot(fig_2)

    else:
        st.warning("예측한 데이터가 아직 없습니다. 모델을 먼저 훈련하십시오.")




if learn_type == '비지도학습':      
    col1, col2, = st.columns([1, 10])
    with col1:
        st.write('<b>학습의 목적</b>', unsafe_allow_html=True)
    with col2:
        col1, col2= st.columns([1, 1])
        with col1:
            if st.button('차원축소'):
                st.session_state.ml_type = 'Dimension_reduction'
        with col2:
            if st.button('군집화'):
                st.session_state.ml_type = 'Clustering'
    st.markdown('<hr>', unsafe_allow_html=True)
    try: ml_type = st.session_state.ml_type
    except AttributeError:
        st.warning('머신러닝의 학습 유형이 선택되지 않았습니다.')

if ml_type == 'Clustering':
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(':heavy_check_mark: <b>K-평균 군집분석에서 적절한 군집수 찾기.</b>', unsafe_allow_html=True)
        st.write('적절한 군집수 찾기')  
        if ml_type == 'Clustering':
            k_count = st.slider('적절한 군집 영역을 설정해주세요.', 1, 20, 1)
            ks = range(1, k_count)
            inertias = []
            kmc_data ={'x':ks, 'y': inertias} 

            for k in ks:
                model = KMeans(n_clusters=k)
                model.fit(feature_data)
                inertias.append(model.inertia_)
            
    with col2:
        fig, ax = plt.subplots()
        sns.lineplot(x = 'x', y = 'y', data=kmc_data, marker='o')
        ax.set_xlabel('numner of clusters. k')
        ax.set_ylabel('inertia')
        ax.set_title('inertia vs k Line Plot')
        st.pyplot(fig)

    st.divider()
    st.write(':heavy_check_mark: <b>K-평균 군집분석 결과</b>', unsafe_allow_html=True)
    st.write('K-평균 군집분석을 통한 결과')
    kmc_k_value = st.number_input('적절한 군집 갯수를 입력해주세요', 1, 100, 1)
    n_init_value = st.number_input('inital centroid를 몇번 샘플링 할건지 입력해주세요.', 1, 1000, 1)
    iter_value = st.number_input('KMean을 몇번 수행할건지 입력해주세요.', 1, 2000, 1)
    ks = range(1, k_count)
    kmc = KMeans(n_clusters=kmc_k_value, n_init = n_init_value, max_iter = iter_value, algorithm='auto', random_state=12)
    #생성한 모델로 학습
    kmc.fit(feature_data)
    centers = kmc.cluster_centers_
    pred = kmc.predict(feature_data)
    joblib.dump(kmc, 'model.joblib')

    #원래 데이터에 예측된 군집 붙이기
    clust_df = feature_data.copy()
    clust_df['clust'] = pred

    clust_x_data = st.selectbox('x축 값을 입력해주세요', feature_data.columns)
    clust_y_data = st.selectbox('y축 값을 입력해주세요', feature_data.columns)
    st.write(len(feature_data.columns))
    i = 0
    j = 0
    for i in range(len(feature_data.columns)):
        if clust_x_data == feature_data.columns[i]:
            center_x_col_num = i
    for j in range(len(feature_data.columns)):        
        if clust_y_data == feature_data.columns[j]:
            center_y_col_num = j

    fig, ax = plt.subplots()
    plt.scatter(centers[:, center_x_col_num], centers[:, center_y_col_num], c='black', alpha=0.8, s=150)
    sns.scatterplot(x=clust_x_data, y=clust_y_data, data=feature_data, hue=kmc.labels_, palette='coolwarm')
    st.pyplot(fig)



