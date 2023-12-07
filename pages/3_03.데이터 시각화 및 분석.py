import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns


# 데이터 불러오기
df = pd.read_pickle('my_dataframe.pkl')
num_data_property = pd.read_pickle('num_data_proprety.pkl')

st.set_page_config(
    page_title="데이터 시각화 및 분석",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",  # wide 또는 center 중 선택
    initial_sidebar_state="auto",  # auto, collapsed, expanded 중 선택
    )

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'


st.header('	:bar_chart: 업로드한 데이터를 시각화 하고 분석해봅니다.')
st.markdown('<hr>', unsafe_allow_html=True)
df.insert(0, '값을 선택해주세요', None)
col1, col2 = st.columns([1, 2])
fig = None

with col1:
    st.write(':heavy_check_mark: <b>그래프의 종류 선택하기</b>', unsafe_allow_html=True)

    graph_type = {
        '그래프의 형태를 선택해주세요.': 'placeholder',
        '관계형': 'Relational',
        '분포형': 'Distribution',
        '범주형': 'Categorical',
        '행렬형': 'Matrix'
        }
    selected_graph_type = st.selectbox('그래프의 형태 선택', list(graph_type.keys()))

    if graph_type[selected_graph_type] == 'Relational':
        rel_plot_type = {
            '선택해주세요':'placeholder',
            '산점도': 'scatter',
            '라인': 'line'
            }
        selected_rel_plot_type = st.selectbox('그래프의 종류를 선택해주세요.', list(rel_plot_type.keys()))


        if rel_plot_type[selected_rel_plot_type] == 'scatter':
            x_data = st.selectbox('x축 값을 입력해주세요', df.columns)
            y_data = st.selectbox('y축 값을 입력해주세요', df.columns)
            if x_data == df.columns[0] and y_data == df.columns[0]:
                st.warning('x, y축의 값을 선택해주세요.')
            elif x_data != df.columns[0] and y_data == df.columns[0]:    
                st.warning('y축의 값을 선택해주세요.')
            elif x_data == df.columns[0] and y_data != df.columns[0]:    
                st.warning('x축의 값을 선택해주세요.')
            elif x_data != df.columns[0] and y_data != df.columns[0]:    
                fig, ax = plt.subplots(constrained_layout=True)
                sns.scatterplot(x=x_data, y=y_data, data=df)
                # sns.regplot(x=x_data, y=y_data, data=df, ax=ax)
                ax.set_xlabel(x_data)
                ax.set_ylabel(y_data)
                ax.set_title(f'{y_data} - {x_data} Scatter Plot')

        elif rel_plot_type[selected_rel_plot_type] == 'line':
            x_data = st.selectbox('x축 값을 입력해주세요', df.columns)
            y_data = st.selectbox('y축 값을 입력해주세요', df.columns)
            if x_data == df.columns[0] and y_data == df.columns[0]:
                st.warning('x, y축의 값을 선택해주세요.')
            elif x_data != df.columns[0] and y_data == df.columns[0]:    
                st.warning('y축의 값을 선택해주세요.')
            elif x_data == df.columns[0] and y_data != df.columns[0]:    
                st.warning('x축의 값을 선택해주세요.')
            elif x_data != df.columns[0] and y_data != df.columns[0]:    
                fig, ax = plt.subplots(constrained_layout=True)
                sns.lineplot(x=x_data, y=y_data, data=df)
                ax.set_xlabel(x_data)
                ax.set_ylabel(y_data)
                ax.set_title(f'{y_data} vs {x_data} Line Plot')

    elif graph_type[selected_graph_type] == 'Distribution':
        dis_plot_type = {
            '선택해주세요':'placeholder',
            '히스토그램':'hist',
            '밀도그래프': 'kde',
            'ECDFplot':'ecdf',
            'rugplot':'rug',
            }

        selected_dis_plot_type = st.selectbox('그래프의 종류를 선택해주세요.', list(dis_plot_type.keys()))
        if dis_plot_type[selected_dis_plot_type] == 'hist':
            hist_data = st.selectbox('분포를 보고 싶은 Feature를 선택하세요', df.columns)
            fig, ax = plt.subplots(constrained_layout=True)
            sns.histplot(x=hist_data, data=df)
            ax.set_xlabel(hist_data)
            ax.set_title(f'{hist_data} Histogram Plot')

        elif dis_plot_type[selected_dis_plot_type] == 'kde':
            kde_data = st.selectbox('분포를 보고 싶은 Feature를 선택하세요', df.columns)
            fig, ax = plt.subplots(constrained_layout=True)
            sns.kdeplot(x=kde_data, data=df)
            ax.set_xlabel(kde_data)
            ax.set_title(f'{kde_data} KDE Plot')
        elif dis_plot_type[selected_dis_plot_type] == 'ecdf':
            ecdf_data = st.selectbox('분포를 보고 싶은 Feature를 선택하세요', df.columns)
            fig, ax = plt.subplots(constrained_layout=True)
            sns.ecdfplot(x=ecdf_data, data=df)
            ax.set_xlabel(ecdf_data)
            ax.set_title(f'{ecdf_data} KDE Plot')
        elif dis_plot_type[selected_dis_plot_type] == 'rug':
            rug_data = st.selectbox('분포를 보고 싶은 Feature를 선택하세요', df.columns)
            fig, ax = plt.subplots(constrained_layout=True)
            sns.rugplot(x=rug_data, data=df)
            ax.set_xlabel(rug_data)
            ax.set_title(f'{rug_data} KDE Plot')

    elif graph_type[selected_graph_type] == 'Categorical':
        kind_list = ['선택해주세요.', 'strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', 'count']
        kind_cat = st.selectbox('그래프 유형을 선택하세요', kind_list)
        if kind_cat != kind_list[0]:
            hue_cat = st.selectbox('색조(hue) 열을 입력해주세요', df.columns)
            x_cat = st.selectbox('x축 값을 입력해주세요', df.columns)
            y_cat = st.selectbox('y축 값을 입력해주세요', df.columns)

            if hue_cat == df.columns[0]:
                st.warning('Hue값을 선택해주세요.')
            elif hue_cat != df.columns[0]:    
                if x_cat == df.columns[0] and y_cat == df.columns[0]:
                    st.warning('x, y축의 값을 선택해주세요.')
                elif x_cat != df.columns[0] and y_cat == df.columns[0]:    
                    st.warning('y축의 값을 선택해주세요.')
                elif x_cat == df.columns[0] and y_cat != df.columns[0]:    
                    st.warning('x축의 값을 선택해주세요.')
                elif x_cat != df.columns[0] and y_cat != df.columns[0]:    
                    fig = sns.catplot(x=x_cat, y=y_cat, hue=hue_cat, kind=kind_cat, data=df)


    if graph_type[selected_graph_type] == 'Matrix':
        matrix_list = ['선택해주세요', 'heatmap', 'clustermap']
        matrix_type = st.selectbox('그래프의 종류를 선택해주세요', matrix_list)

        if matrix_type == 'heatmap':
            heatmap_cols = st.multiselect('히트맵으로 시각화할 열을 선택하세요', num_data_property)
            heatmap_data = df[heatmap_cols]
            sns.set(font_scale=1.2)
            if len(heatmap_data.columns) < 2:
                st.warning('시각화할 열을 2개 이상 선택해주세요.')
            else:    
                fig, ax = plt.subplots(constrained_layout=True)
                sns.heatmap(heatmap_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                ax.set_title('Correlation Heatmap')

        elif matrix_type == 'clustermap':
            clustermap_cols = st.multiselect('클러스터맵으로 시각화할 열을 선택하세요', num_data_property)
            clustermap_data = df[clustermap_cols]
            if len(clustermap_data.columns) < 2:
                st.warning('시각화할 열을 2개 이상 선택해주세요.')
            else:    
                sns.set(font_scale=1.2)
                fig = sns.clustermap(clustermap_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
                






with col2:
    st.write(':heavy_check_mark: <b>시각화 그래프</b>', unsafe_allow_html=True)
    if fig is None:
        st.warning('그래프의 종류 및 축의 값을 선택해주세요.')
    elif fig is not None:
        st.pyplot(fig)
        