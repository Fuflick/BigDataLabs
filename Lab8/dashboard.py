import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    value=(0, 80)  # ✅ Закрыта скобка
)

selected_class = st.sidebar.multiselect(
    'Класс пассажиров',
    options=['First', 'Second', 'Third'],
    default=['First', 'Second', 'Third']  # ✅ Убрана лишняя запятая
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
    ['Гистограмма возрастов', 'Ящик с усами (Fare/Class)', 'Корреляционная матрица']
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
        filtered_data.corr(),
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=0.5
    )
    plt.title('Корреляции между признаками')
    st.pyplot(fig)

# Отображение статистики
st.markdown("---")
st.header('Базовая статистика')
st.dataframe(filtered_data.describe(), use_container_width=True)

# Отображение сырых данных
if st.checkbox('Показать исходные данные'):
    st.subheader('Исходные данные')
    st.dataframe(filtered_data, use_container_width=True)  # ✅ Убрана лишняя `/`
