import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# source .venv/bin/activate
# streamlit run dashboard.py

# Функция загрузки и предобработки данных
@st.cache_data
def load_data():
    # Загрузка данных
    df = sns.load_dataset('titanic')

    # Предобработка данных
    df = df.drop(['embarked', 'who', 'adult_male', 'deck', 'embark_town', 'alive'], axis=1)
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].apply(lambda x: np.log(x) if x > 0 else 0)
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})

    return df

# Загрузка данных
df = load_data()

# Интерфейс пользователя
st.title('📊 Анализ данных Титаника')
st.markdown("---")

# Боковая панель с фильтрами
st.sidebar.header('Параметры анализа')
age_range = st.sidebar.slider(
    'Диапазон возраста',
    min_value=0,
    max_value=100,
    value=(0, 100)
)

selected_class = st.sidebar.multiselect(
    'Класс пассажиров',
    options=['First', 'Second', 'Third'],
    default=['First', 'Second', 'Third']
)

# Применение фильтров
filtered_data = df[
    (df['age'] >= age_range[0]) &
    (df['age'] <= age_range[1]) &
    (df['pclass'].isin([1 if c == 'First' else 2 if c == 'Second' else 3 for c in selected_class]))
]

# Основная панель
st.header('Визуализация данных')
chart_type = st.selectbox(
    'Выберите тип графика',
    ['Гистограмма возрастов', 'Ящик с усами (Fare/Class)', 'Корреляционная матрица', 'Анализ выбросов']
)

# Построение графиков
if chart_type == 'Гистограмма возрастов':
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_data['age'], bins=20, kde=True, color='skyblue')
    plt.title('Распределение возраста пассажиров')
    st.pyplot(fig)

elif chart_type == 'Ящик с усами (Fare/Class)':
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x='pclass',
        y='fare',
        data=filtered_data,
        palette='pastel',
        width=0.5
    )
    plt.title('Распределение стоимости билетов по классам')
    plt.ylabel('Логарифм стоимости билета')
    st.pyplot(fig)

elif chart_type == 'Корреляционная матрица':
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        filtered_data.corr(numeric_only=True),
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=0.5
    )
    plt.title('Корреляции между признаками')
    st.pyplot(fig)

elif chart_type == 'Анализ выбросов':
    # Анализ выбросов в стоимости билетов
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df['fare'], palette='pastel')
    plt.title('Выбросы в стоимости билетов')
    st.pyplot(fig)

    # Анализ выбросов в возрасте
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df['age'], palette='pastel')
    plt.title('Выбросы в возрасте пассажиров')
    st.pyplot(fig)

    # Функция поиска выбросов по межквартильному размаху (IQR)
    def detect_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    outliers_fare = detect_outliers(df, 'fare')
    outliers_age = detect_outliers(df, 'age')

    st.subheader("Выбросы в стоимости билетов")
    st.dataframe(outliers_fare)

    st.subheader("Выбросы в возрасте пассажиров")
    st.dataframe(outliers_age)

# Отображение статистики
st.markdown("---")
st.header('Базовая статистика')
st.dataframe(filtered_data.describe(), use_container_width=True)

# Отображение сырых данных
if st.checkbox('Показать исходные данные'):
    st.subheader('Исходные данные')
    st.dataframe(filtered_data, use_container_width=True)
