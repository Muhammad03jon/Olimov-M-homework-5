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

st.title('🔍 Предсказание реальной или фальшивой банкноты')

# Загрузка данных
file_path = "https://raw.githubusercontent.com/Muhammad03jon/Olimov-M-homework-5/refs/heads/master/data_banknote_authentication.txt"
df = pd.read_csv(file_path, sep=",", header=None)
df.columns = ["variance", "skewness", "curtosis", "entropy", "class"]

st.subheader('📊 Данные')
st.dataframe(df.head())

# Разделение на признаки и целевую переменную
X_raw = df.drop('class', axis=1)
y_raw = df['class']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Выбор модели
st.sidebar.header("⚙️ Выбор модели и гиперпараметров")
model_choice = st.sidebar.selectbox("Выберите модель", ["KNN", "Logistic Regression", "Decision Tree"])

if model_choice == "KNN":
    n_neighbors = st.sidebar.slider("Число соседей", 1, 20, 3)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
elif model_choice == "Logistic Regression":
    max_iter = st.sidebar.slider("Число итераций", 100, 1000, 500)
    model = LogisticRegression(max_iter=max_iter)
elif model_choice == "Decision Tree":
    max_depth = st.sidebar.slider("Максимальная глубина", 1, 20, 5)
    model = DecisionTreeClassifier(max_depth=max_depth)

# Обучение модели
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Выборка для предсказания
st.sidebar.header("🔢 Введите признаки для предсказания")
variance = st.sidebar.slider("Variance", float(df["variance"].min()), float(df["variance"].max()), float(df["variance"].mean()))
skewness = st.sidebar.slider("Skewness", float(df["skewness"].min()), float(df["skewness"].max()), float(df["skewness"].mean()))
curtosis = st.sidebar.slider("Curtosis", float(df["curtosis"].min()), float(df["curtosis"].max()), float(df["curtosis"].mean()))
entropy = st.sidebar.slider("Entropy", float(df["entropy"].min()), float(df["entropy"].max()), float(df["entropy"].mean()))

sample = np.array([[variance, skewness, curtosis, entropy]])
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
st.sidebar.write("**Предсказание:** ", "Настоящая банкнота" if prediction[0] == 0 else "Фальшивая банкнота")

# Визуализация метрик качества
st.subheader("📊 Оценка модели")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

st.text("Отчет по классификации:")
st.text(classification_report(y_test, y_pred))

# Важность признаков
if model_choice == "Decision Tree":
    st.subheader("🔬 Важность признаков")
    feature_importances = model.feature_importances_
    fig, ax = plt.subplots()
    sns.barplot(x=X_raw.columns, y=feature_importances, ax=ax)
    ax.set_title("Важность признаков")
    st.pyplot(fig)

# Загрузка CSV
st.subheader("📥 Загрузите файл для предсказания")
uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])
if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    new_data_scaled = scaler.transform(new_data)
    predictions = model.predict(new_data_scaled)
    st.write("Предсказания:")
    st.write(predictions)
