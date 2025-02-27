import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_decision_regions

st.title('Предсказание реальной или фальшивой банкноты')

# Загрузка данных
file_path = "https://raw.githubusercontent.com/Muhammad03jon/Olimov-M-homework-5/refs/heads/master/data_banknote_authentication.txt"
df = pd.read_csv(file_path, sep=",", header=None)
df.columns = ["variance", "skewness", "curtosis", "entropy", "class"]

# Отображение данных
with st.expander('Просмотр данных'):
    st.dataframe(df.head())

# Разделение на X и y
X_raw = df.drop('class', axis=1)
y_raw = df['class']

# Выбор модели
model_choice = st.sidebar.selectbox("Выберите модель", ['KNN', 'Логистическая регрессия', 'Дерево решений'])

# Настройка гиперпараметров
if model_choice == 'KNN':
    n_neighbors = st.sidebar.slider("Число соседей (K)", 1, 15, 3)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
elif model_choice == 'Логистическая регрессия':
    max_iter = st.sidebar.slider("Максимальное число итераций", 100, 1000, 500)
    model = LogisticRegression(max_iter=max_iter)
elif model_choice == 'Дерево решений':
    max_depth = st.sidebar.slider("Глубина дерева", 1, 20, 5)
    model = DecisionTreeClassifier(max_depth=max_depth)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.3, random_state=42)

# Стандартизация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Метрики качества
st.subheader("Метрики качества модели")
st.text(classification_report(y_test, y_pred))

# Матрица ошибок
st.subheader("Матрица ошибок")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Важность признаков (для дерева решений)
if model_choice == 'Дерево решений':
    st.subheader("Важность признаков")
    feature_importances = pd.Series(model.feature_importances_, index=X_raw.columns)
    fig, ax = plt.subplots()
    feature_importances.sort_values().plot(kind='barh', ax=ax)
    st.pyplot(fig)

# Выбор случайного образца
st.sidebar.subheader("Введите признаки вручную или выберите случайный образец")
use_random_sample = st.sidebar.checkbox("Использовать случайный образец")
if use_random_sample:
    sample = df.sample(1).iloc[0]
else:
    sample = {
        "variance": st.sidebar.slider("Variance", float(df["variance"].min()), float(df["variance"].max()), float(df["variance"].mean())),
        "skewness": st.sidebar.slider("Skewness", float(df["skewness"].min()), float(df["skewness"].max()), float(df["skewness"].mean())),
        "curtosis": st.sidebar.slider("Curtosis", float(df["curtosis"].min()), float(df["curtosis"].max()), float(df["curtosis"].mean())),
        "entropy": st.sidebar.slider("Entropy", float(df["entropy"].min()), float(df["entropy"].max()), float(df["entropy"].mean())),
    }
    sample = pd.Series(sample)

# Предсказание на выбранном образце
sample_scaled = scaler.transform(sample.values.reshape(1, -1))
prediction = model.predict(sample_scaled)[0]
st.subheader("Результат предсказания")
st.write(f"Модель предсказала: {'Настоящая банкнота' if prediction == 0 else 'Фальшивая банкнота'}")

# Загрузка пользовательского файла
st.subheader("Загрузите CSV-файл для предсказания на новых данных")
file_upload = st.file_uploader("Выберите файл CSV", type=["csv"])
if file_upload is not None:
    user_df = pd.read_csv(file_upload)
    st.dataframe(user_df.head())
    user_scaled = scaler.transform(user_df)
    user_predictions = model.predict(user_scaled)
    st.subheader("Результаты предсказания")
    user_df["Предсказанный класс"] = user_predictions
    st.dataframe(user_df)
