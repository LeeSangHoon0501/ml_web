import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import missingno as msno

st.set_page_config(
    page_title="데이터 탐색",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",  # wide 또는 centered 중 선택
    initial_sidebar_state="auto",  # auto, collapsed, expanded 중 선택
    
)


df= pd.read_pickle('my_dataframe.pkl')

st.title(':chart_with_downwards_trend: 업로드된 데이터를 확인 및 탐색')

tab1, tab2, tab3, tab4 = st.tabs(['업로드한 데이터표(DataTable)', '데이터탐색(Data Exploration)','결측값 확인', '데이터간 상관관계분석(Correlation)'])


df= pd.read_pickle('my_dataframe.pkl')

with tab1:
    st.write(df)


with tab2:
    st.write('컬럼의 갯수 = ', len(df.columns), '열의 갯수 = ', len(df.index))
    num_data_properties = []  # 데이터 프레임의 특성을 저장할 리스트를 초기화합니다.
    str_data_properties = []

    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:    
            column_data = {
            '컬럼': column,
            '데이터타입': df[column].dtype,
            '최대값': df[column].max(),
            '최소값': df[column].min(),
            '중앙값': df[column].median(), 
            '평균값': df[column].mean(),
            '표준편차': df[column].std()}
            num_data_properties.append(column_data)

        elif df[column].dtype not in ['int64', 'float64']:
            str_value = df[column].drop_duplicates().reset_index(drop=True).to_string(index=True)
            column_data = {
            '컬럼': column,
            '데이터타입': df[column].dtype,
            '포함된 데이터':str_value}

            str_data_properties.append(column_data)


    st.write(':white_check_mark: <b>숫자로 구성된 데이터의 탐색</b>', unsafe_allow_html=True)
    num_data_property = pd.DataFrame(num_data_properties)
    st.write(num_data_property)

    st.write(':white_check_mark: <b>문자로 구성된 데이터의 탐색</b>', unsafe_allow_html=True)
    str_data_property = pd.DataFrame(str_data_properties)
    if not str_data_property.empty:
        st.write(str_data_property)
        st.write('(포함된 데이터를 더블클릭하면 세부적으로 볼 수 있습니다.)')
    else:
        st.write("문자로 구성된 데이터가 없습니다.")        

with tab3:
    miss_fig, ax = plt.subplots(figsize=(8, 4))
    msno.bar(df, color=(0.7, 0.2, 0.2), ax=ax, fontsize=7)
    st.pyplot(miss_fig)


with tab4:
    corr_feature_select = st.multiselect('상관관계를 분석할 컬럼을 선택하세요.(숫자 데이터를 가진 컬럼만 분석합니다.)', num_data_property)
    corr_select_data = df[corr_feature_select]
    corr_data= corr_select_data.corr()
    

    if not corr_data.empty:
        corr_data = corr_select_data.corr()

        st.write("선택한 특성(열)에 대한 상관관계")
        st.write(corr_data)

    else:
        st.warning("선택한 특성(열)이 없습니다. 하나 이상의 특성을 선택해주세요.")

num_data_property.to_pickle('num_data_proprety.pkl')


