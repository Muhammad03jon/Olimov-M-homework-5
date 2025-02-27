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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

st.title('💵 Предсказание реальной или фальшивой банкноты')

# Загрузка данных
file_path = "https://raw.githubusercontent.com/Muhammad03jon/Olimov-M-homework-5/refs/heads/master/data_banknote_authentication.txt"
df = pd.read_csv(file_path, sep=",", header=None)
df.columns = ["variance", "skewness", "curtosis", "entropy", "class"]

with st.expander('Исходные данные'):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("X (Признаки)")
        X_raw = df.drop('class', axis=1)
        st.dataframe(X_raw)
    with col2:
        st.subheader("y (Целевая переменная)")
        y_raw = df['class']
        st.dataframe(y_raw.to_frame())

# Ввод данных
with st.sidebar:
    st.header("Выбор модели и гиперпараметров")
    model_choice = st.selectbox("Выберите модель:", ["KNN", "Логистическая регрессия", "Дерево решений"])

    if model_choice == "KNN":
        n_neighbors = st.slider('n_neighbors', 1, 20, 3)
    elif model_choice == "Дерево решений":
        max_depth = st.slider('max_depth', 1, 10, 5)
    elif model_choice == "Логистическая регрессия":
        solver = st.selectbox("Выберите алгоритм оптимизации:", ["liblinear", "lbfgs"])
        C = st.slider('C (обратная регуляризация)', 0.01, 10.0, 1.0)

    use_random_sample = st.checkbox("📌 Использовать случайный образец")
    if use_random_sample:
        random_sample = df.sample(1).iloc[0]
        variance, skewness, curtosis, entropy = random_sample[:4]
    else:
        variance = st.slider('Variance', float(df["variance"].min()), float(df["variance"].max()), float(df["variance"].mean()))
        skewness = st.slider('Skewness', float(df["skewness"].min()), float(df["skewness"].max()), float(df["skewness"].mean()))
        curtosis = st.slider('Curtosis', float(df["curtosis"].min()), float(df["curtosis"].max()), float(df["curtosis"].mean()))
        entropy = st.slider('Entropy', float(df["entropy"].min()), float(df["entropy"].max()), float(df["entropy"].mean()))

    st.write("**Выбранные значения:**")
    st.write(f"Variance: {variance}")
    st.write(f"Skewness: {skewness}")
    st.write(f"Curtosis: {curtosis}")
    st.write(f"Entropy: {entropy}")

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

#  Выбор моделей
if model_choice == "KNN":
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
elif model_choice == "Логистическая регрессия":
    model = LogisticRegression(solver=solver, C=C, max_iter=565)
elif model_choice == "Дерево решений":
    model = DecisionTreeClassifier(max_depth=max_depth)

# Обучение модели
model.fit(X_train_scaled, y_train)

# Кнопка предсказаний
if st.button("Предсказать"):
    data = {"variance": variance, "skewness": skewness, "curtosis": curtosis, "entropy": entropy}
    sample_df = pd.DataFrame([data])
    sample_scaled = scaler.transform(sample_df)

    prediction = model.predict(sample_scaled)[0]

    st.subheader("🔮 Результаты предсказания")
    st.metric("Предсказание", "Реальная" if prediction == 0 else "Фальшивая")

# Тестирование на новых данных
st.header("Тестирование на новых данных")
uploaded_file = st.file_uploader("Загрузите CSV файл с новыми банкнотами", type="csv")
if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    new_data_scaled = scaler.transform(new_data)
    new_predictions = model.predict(new_data_scaled)
    st.write("Предсказания для новых данных:")
    st.dataframe(pd.DataFrame(new_predictions, columns=["Prediction"]))

# Визуализация важности признаков для дерева решений
if model_choice == "Дерево решений":
    importance = model.feature_importances_
    fig, ax = plt.subplots()
    ax.barh(X_raw.columns, importance)
    ax.set_xlabel('Важность признаков')
    ax.set_title('Важность признаков для дерева решений')
    st.pyplot(fig)

# Графики метрик качества моделей
if st.button("Показать метрики качества"):
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Accuracy
    accuracy_train = (y_pred_train == y_train).mean()
    accuracy_test = (y_pred_test == y_test).mean()
    
    # Метрики качества
    report_train = classification_report(y_train, y_pred_train, output_dict=True)
    report_test = classification_report(y_test, y_pred_test, output_dict=True)

    st.subheader("📊 Метрики качества модели")
    
    # Таблица с метриками
    metrics_df = pd.DataFrame({
        'Metric': ['Precision (Train)', 'Recall (Train)', 'F1-score (Train)', 'Accuracy (Train)',
                   'Precision (Test)', 'Recall (Test)', 'F1-score (Test)', 'Accuracy (Test)'],
        'Value': [
            report_train['1']['precision'], report_train['1']['recall'], report_train['1']['f1-score'], accuracy_train,
            report_test['1']['precision'], report_test['1']['recall'], report_test['1']['f1-score'], accuracy_test
        ]
    })
    st.table(metrics_df)

    # Матрица ошибок
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Предсказание')
    ax.set_ylabel('Истинные значения')
    ax.set_title('Матрица ошибок')
    st.pyplot(fig)

    # ROC AUC
    y_scores = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    st.pyplot(fig)
