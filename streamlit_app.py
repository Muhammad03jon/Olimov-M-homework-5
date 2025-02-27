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
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from mlxtend.plotting import plot_decision_regions

st.title('Предсказание реальной или фальшивой банкноты')

file_path = "https://raw.githubusercontent.com/Muhammad03jon/Olimov-M-homework-5/refs/heads/master/data_banknote_authentication.txt"

df = pd.read_csv(file_path, sep=",", header=None)
df.columns = ["variance", "skewness", "curtosis", "entropy", "class"]

# Разделение данных
X = df.drop(columns=['class'])
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Выбор модели
st.sidebar.header("Выберите модель")
model_choice = st.sidebar.selectbox("Модель", ["KNN", "Логистическая регрессия", "Дерево решений"])

def train_model(model_name):
    if model_name == "KNN":
        n_neighbors = st.sidebar.slider("Количество соседей", 1, 20, 3)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_name == "Логистическая регрессия":
        max_iter = st.sidebar.slider("Максимальное количество итераций", 100, 1000, 300)
        model = LogisticRegression(max_iter=max_iter)
    else:
        max_depth = st.sidebar.slider("Максимальная глубина", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth)
    return model

# Обучение
model = train_model(model_choice)
model.fit(X_train_scaled, y_train)

# Предсказания
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

# Оценка модели
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

st.subheader("Оценка модели")
st.write(f"**Accuracy (Train):** {train_acc:.4f}")
st.write(f"**Accuracy (Test):** {test_acc:.4f}")
st.write(f"**ROC AUC (Train):** {train_auc:.4f}")
st.write(f"**ROC AUC (Test):** {test_auc:.4f}")

# ROC-кривая
fig, ax = plt.subplots()
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
ax.plot(fpr, tpr, label=f"{model_choice} (AUC = {test_auc:.2f})")
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC-кривая")
ax.legend()
st.pyplot(fig)
