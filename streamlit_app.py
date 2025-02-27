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

# Заголовок приложения
st.title('Предсказание реальной или фальшивой банкноты')

# Загрузка данных
file_path = "https://raw.githubusercontent.com/Muhammad03jon/Olimov-M-homework-5/refs/heads/master/data_banknote_authentication.txt"
df = pd.read_csv(file_path, sep=",", header=None)
df.columns = ["variance", "skewness", "curtosis", "entropy", "class"]

# Отображение данных
with st.expander('Данные'):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("X (Признаки)")
        X_raw = df.drop('class', axis=1)
        st.dataframe(X_raw.style.set_properties(**{'background-color': '#f0f2f6', 'color': 'black'}))

    with col2:
        st.subheader("y (Цель)")
        y_raw = df['class']
        st.dataframe(y_raw.to_frame().style.set_properties(**{'background-color': '#e8f4ea', 'color': 'black'}))

# Ввод признаков через боковую панель
with st.sidebar:
    st.header("Введите признаки:")
    
    # Переключатель для случайного образца
    use_random_sample = st.checkbox("Использовать случайный образец")

    if use_random_sample:
        random_sample = df.sample(1).iloc[0]
        variance = random_sample["variance"]
        skewness = random_sample["skewness"]
        curtosis = random_sample["curtosis"]
        entropy = random_sample["entropy"]
        st.write("Выбраны случайные значения:")
    else:
        # Ввод значений через слайдеры
        variance = st.slider('Variance', float(df["variance"].min()), float(df["variance"].max()), float(df["variance"].mean()))
        skewness = st.slider('Skewness', float(df["skewness"].min()), float(df["skewness"].max()), float(df["skewness"].mean()))
        curtosis = st.slider('Curtosis', float(df["curtosis"].min()), float(df["curtosis"].max()), float(df["curtosis"].mean()))
        entropy = st.slider('Entropy', float(df["entropy"].min()), float(df["entropy"].max()), float(df["entropy"].mean()))

    data = {
        "variance": variance,
        "skewness": skewness,
        "curtosis": curtosis,
        "entropy": entropy
    }

    # Отображение выбранных значений
    st.write(f"Выбранные значения:\n- Variance: {variance}\n- Skewness: {skewness}\n- Curtosis: {curtosis}\n- Entropy: {entropy}")

st.subheader("📊 Анализ данных")

# Графики для всех признаков: гистограммы и графики плотности
st.subheader("Гистограммы и графики плотности для всех признаков")

fig, axes = plt.subplots(4, 2, figsize=(12, 20))

for i, col in enumerate(["variance", "skewness", "curtosis", "entropy"]):
    # Гистограмма
    ax_hist = axes[i, 0]
    sns.histplot(df[col], ax=ax_hist, bins=30, kde=False, color='skyblue', alpha=0.6)
    ax_hist.set_title(f"Гистограмма: {col}")
    ax_hist.set_xlabel(col)
    ax_hist.set_ylabel("Частота")

    # График плотности
    ax_kde = axes[i, 1]
    sns.kdeplot(data=df, x=col, hue='class', fill=True, ax=ax_kde, palette='Set1', alpha=0.5)
    ax_kde.set_title(f"Плотность распределения: {col} по классам")
    ax_kde.set_xlabel(col)
    ax_kde.set_ylabel("Плотность")

# Увеличиваем отступы между графиками
plt.subplots_adjust(wspace=0.4, hspace=0.4)
st.pyplot(fig)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.3, random_state=42)

# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)

# Обучение моделей
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

log_reg = LogisticRegression(max_iter=565)
log_reg.fit(X_train_scaled, y_train)

d_tree = DecisionTreeClassifier(max_depth=5)
d_tree.fit(X_train_scaled, y_train)

# Визуализация границ решений
X_array = X_train_scaled.to_numpy() 
y_array = y_train.to_numpy()

classifiers = [knn, log_reg, d_tree]
titles = ['KNeighborsClassifier', 'Logistic Regression', 'Decision Tree']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, clf, title in zip(axes, classifiers, titles):
    plot_decision_regions(X_array, y_array, clf=clf, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('variance')
    ax.set_ylabel('skewness')

plt.tight_layout()
st.pyplot(fig)

# Предсказание
if st.button("Предсказать"):
    input_data = np.array([[variance, skewness, curtosis, entropy]])
    input_scaled = scaler.transform(input_data)  # Применение стандартизации

    # Предсказание с помощью каждой из моделей
    prediction_knn = knn.predict(input_scaled)[0]
    prediction_log_reg = log_reg.predict(input_scaled)[0]
    prediction_dtree = d_tree.predict(input_scaled)[0]

    # Отображение результатов предсказания
    st.subheader("Результаты предсказания:")
    st.write(f"KNeighborsClassifier: {'Фальшивая' if prediction_knn == 0 else 'Настоящая'} банкнота")
    st.write(f"Logistic Regression: {'Фальшивая' if prediction_log_reg == 0 else 'Настоящая'} банкнота")
    st.write(f"Decision Tree: {'Фальшивая' if prediction_dtree == 0 else 'Настоящая'} банкнота")
