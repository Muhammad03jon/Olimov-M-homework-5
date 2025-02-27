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
from mlxtend.plotting import plot_decision_regions

st.title('💵 Предсказание реальной или фальшивой банкноты')

# Загрузка данных
file_path = "https://raw.githubusercontent.com/Muhammad03jon/Olimov-M-homework-5/refs/heads/master/data_banknote_authentication.txt"
df = pd.read_csv(file_path, sep=",", header=None)
df.columns = ["variance", "skewness", "curtosis", "entropy", "class"]

with st.expander('📂 Исходные данные'):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("X (Признаки)")
        X_raw = df.drop('class', axis=1)
        st.dataframe(X_raw)
    with col2:
        st.subheader("y (Целевая переменная)")
        y_raw = df['class']
        st.dataframe(y_raw.to_frame())

# Ввод данных пользователем
with st.sidebar:
    st.header("🎛 Введите признаки: ")
    use_random_sample = st.checkbox("📌 Использовать случайный образец")
    if use_random_sample:
        random_sample = df.sample(1).iloc[0]
        variance, skewness, curtosis, entropy = random_sample[:4]
    else:
        variance = st.slider('Variance', float(df["variance"].min()), float(df["variance"].max()), float(df["variance"].mean()))
        skewness = st.slider('Skewness', float(df["skewness"].min()), float(df["skewness"].max()), float(df["skewness"].mean()))
        curtosis = st.slider('Curtosis', float(df["curtosis"].min()), float(df["curtosis"].max()), float(df["curtosis"].mean()))
        entropy = st.slider('Entropy', float(df["entropy"].min()), float(df["entropy"].max()), float(df["entropy"].mean()))
    
    data = {"variance": variance, "skewness": skewness, "curtosis": curtosis, "entropy": entropy}
    st.write("**Выбранные значения:**", data)

# Визуализация данных
st.subheader("📊 Анализ данных")
fig, axes = plt.subplots(4, 2, figsize=(12, 20))
for i, col in enumerate(["variance", "skewness", "curtosis", "entropy"]):
    sns.histplot(df[col], ax=axes[i, 0], bins=30, kde=False, color='skyblue')
    axes[i, 0].set_title(f"Гистограмма: {col}")
    sns.kdeplot(data=df, x=col, hue='class', fill=True, ax=axes[i, 1], palette='Set1', alpha=0.5)
    axes[i, 1].set_title(f"Плотность: {col} по классам")
plt.subplots_adjust(wspace=0.4, hspace=0.4)
st.pyplot(fig)

# Обучение моделей
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
log_reg = LogisticRegression(max_iter=565)
log_reg.fit(X_train_scaled, y_train)
d_tree = DecisionTreeClassifier(max_depth=5)
d_tree.fit(X_train_scaled, y_train)

# Предсказание
sample_df = pd.DataFrame([data])
sample_scaled = scaler.transform(sample_df)  # Исправлено

knn_pred = knn.predict(sample_scaled)[0]
log_reg_pred = log_reg.predict(sample_scaled)[0]
d_tree_pred = d_tree.predict(sample_scaled)[0]

st.subheader("🔮 Результаты предсказания")
col1, col2, col3 = st.columns(3)
col1.metric("KNN", "Реальная" if knn_pred == 0 else "Фальшивая")
col2.metric("Logistic Regression", "Реальная" if log_reg_pred == 0 else "Фальшивая")
col3.metric("Decision Tree", "Реальная" if d_tree_pred == 0 else "Фальшивая")

# Границы решений
st.subheader("📍 Границы решений моделей")
X_array, y_array = X_train_scaled.iloc[:, :2].to_numpy(), y_train.to_numpy()
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, clf, title in zip(axes, [knn, log_reg, d_tree], ['KNN', 'Logistic Regression', 'Decision Tree']):
    plot_decision_regions(X_array, y_array, clf=clf, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('variance')
    ax.set_ylabel('skewness')
plt.tight_layout()
st.pyplot(fig)
