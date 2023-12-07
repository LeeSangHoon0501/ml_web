import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_pickle('my_dataframe.pkl')
num_data_property = pd.read_pickle('num_data_proprety.pkl')

st.set_page_config(
    page_title="데이터 전처리",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",  # wide 또는 centered 중 선택
    initial_sidebar_state="auto",  # auto, collapsed, expanded 중 선택
    
)


st.header(':twisted_rightwards_arrows: 머신러닝을 하기전 데이터를 처리합니다.')    
tab1, tab2, tab3, tab4, tab5 = st.tabs(['필요한 컬럼선택', '데이터 인코딩', '타겟 및 피처 추출','피처 스케일링', '훈련데이터와 테스트 데이터 분할'])

with tab1:
    columns_select = st.multiselect('데이터 분석에 필요 없는 컬럼은 삭제해주세요.', df.columns, default=list(df.columns))
    df_columns_select = df.loc[:, columns_select] 
    st.write(df_columns_select)

with tab2:
    # 토글 상태 업데이트
    col1, col2 = st.columns([1, 1])
    with col1:
        empty_value_handle = st.toggle('범주형 문자열들을 숫자레이블로 변경합니다.')
    with col2:
        empty_rows_remove = st.toggle('Nan값이 있는 행을 삭제합니다.')

    encoder = None
    str_encoder_data = []
    if empty_value_handle:
        for column in df_columns_select.columns:
            if df_columns_select[column].dtype not in ['int64', 'float64']:
                encoder = LabelEncoder()
                df_columns_select[column] = encoder.fit_transform(df[column])
                str_encoder_dic = {column: encoder.classes_ 
                }
                str_encoder_data.append(str_encoder_dic)     

    if empty_rows_remove:
        df_columns_select = df_columns_select.dropna()
        df_columns_select = df_columns_select.reset_index(drop=True)
            
    col3, col4 = st.columns([1, 1])
    with col3:
        st.write('<b>선택한 컬럼</b>', unsafe_allow_html=True)
        st.write(df_columns_select)
    with col4:
        if encoder is not None:
            st.write('<b>숫자로 레이블로 인코딩된 컬럼</b>', unsafe_allow_html=True)
            for i in range(len(str_encoder_data)):
                st.write(pd.DataFrame(str_encoder_data[i]).T)
            
    st.write('컬럼의 갯수:', {len(df_columns_select.columns)}, '열의 갯수:', {len(df_columns_select.index)})

feature_data = None
with tab3:
    df_columns_select.insert(0, 'Target_Select...', None)
    target = st.selectbox('Target으로 할 column을 선택하세요', df_columns_select.columns)

    if target == 'Target_Select...':
        st.write('')
    elif target != 'Target_Select...':   
        target_data = df_columns_select.loc[:, target]
        feature_data = df_columns_select.drop([target, 'Target_Select...'], axis=1)
        feature_select = st.multiselect('Feature로 할 column을 선택하세요.(여러가지)', feature_data.columns)
        feature_data = df_columns_select[feature_select]
    col1, col2 = st.columns([3, 5])
    with col1:
        if target == 'Target_Select...':
            st.write('타겟이 설정되지 않았습니다.')
        elif target != 'Target_Select...':
            st.write('Target 데이터')
            st.write(target_data.head())
            with col2:
                if feature_select is None:
                    st.write('피쳐가 설정되지 않았습니다.')
                elif feature_select is not None:
                    st.write('Feature 데이터')
                    st.write(feature_data.head())



    

with tab4:
    def normalize(normalize_type):
        if len(scaler_column_select) == 0:
            st.write('')
        else:    
            st.write(f'{select_normalization}으로 정규화된 데이터')
            feature_data[scaler_column_select] = normalize_type.fit_transform(feature_data[scaler_column_select])
            st.write(feature_data)


            
    normalization_type = {
    '정규화 종류를 선택해주세요': 'placeholder',
    '정규화 하지 않음': 'NonNormalization',
    '표준화(Standardization)': 'StandardScaler', 
    '최대최소스케일링(Min-Max)': 'MinMaxScaler', 
    '로버스터 스케일링(Robust Scaling)': 'RobustScaler', 
    '정규화(Normalization)': 'Normalizer'}

    select_normalization = st.selectbox('원하는 스케일링 타입을 선택하세요', list(normalization_type.keys()))


    if normalization_type[select_normalization] == 'placeholder':
        st.write('')
    elif normalization_type[select_normalization] != 'placeholder':
        if normalization_type[select_normalization] == 'NonNormalization':
            st.write('정규화 되지 않은 데이터', feature_data)
        else:
            scaler_column_select = st.multiselect('정규화할 컬럼을 선택해주세요.(선택한 컬럼만 정규화가 됩니다.)', feature_data.columns)
            if normalization_type[select_normalization] == 'StandardScaler':
                normalize(StandardScaler())
            elif normalization_type[select_normalization] == 'MinMaxScaler':
                normalize(MinMaxScaler())
            elif normalization_type[select_normalization] == 'RobustScaler':
                normalize(RobustScaler())
            elif normalization_type[select_normalization] == 'Normalizer':
                normalize(Normalizer())

        
    if feature_data is not None:        
        feature_data.to_pickle('feature_data.pkl')    

with tab5:
    if feature_data is not None:
        on = st.toggle('데이터를 셔플합니다.')
        ratio = st.slider('학습데이터와 데스트 데이터의 분할 비율을 설정해주세요.', 10, 90, 20, 10)
        x_train, x_test, y_train, y_test = train_test_split(feature_data, target_data, test_size= ratio/100, shuffle=on, random_state=12)

        ratio = st.slider('훈련데이터와 검증데이터의 분할 비율을 설정해세요.', 10, 90, 20, 10)
        x_train_final, x_val, y_train_final, y_val = train_test_split(x_train, y_train, test_size= ratio/100, shuffle=on, random_state=12)        
        
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
        with col1:
            st.write('훈련 데이터 피처의 shape:', x_train_final.shape)
            st.write(x_train.head())
        with col2:
            st.write('훈련 데이터 타겟의 shape:', y_train_final.shape)
            st.write(y_train.head())
        with col3:
            st.write('검증 데이터 피처의 shape:', x_val.shape)
            st.write(x_val.head())                        
        with col4:
            st.write('검증 데이터 타겟의 shape:', y_val.shape)
            st.write(y_val.head())
        with col5:
            st.write('테스트 데이터 피처의 shape:', x_test.shape)
            st.write(x_test.head())                        
        with col6:
            st.write('테스트 데이터 타겟의 shape:', y_test.shape)
            st.write(y_test.head())



        x_train.to_pickle('x_train.pkl')
        y_train.to_pickle('y_train.pkl')
        x_train_final.to_pickle('x_train_final.pkl')
        y_train_final.to_pickle('y_train_final.pkl')
        x_val.to_pickle('x_val.pkl')
        y_val.to_pickle('y_val.pkl')
        x_test.to_pickle('x_test.pkl')
        y_test.to_pickle('y_test.pkl')


        
    else:
        st.write("아직 데이터가 나눠지지 않았습니다.")


