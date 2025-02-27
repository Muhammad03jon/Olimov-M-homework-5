import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Предсказание реальной или фальшивой банкноты')

file_path = "https://raw.githubusercontent.com/Muhammad03jon/Olimov-M-homework-5/refs/heads/master/data_banknote_authentication.txt"

df = pd.read_csv(file_path, sep=",", header=None)
df.columns = ["variance", "skewness", "curtosis", "entropy", "class"]

with st.expander('Data'):
    col1, col2 = st.columns(2)  # Две колонки

    with col1:
        st.subheader("X (Features)")
        X_raw = df.drop('class', axis=1)
        st.dataframe(X_raw.style.set_properties(**{'background-color': '#f0f2f6', 'color': 'black'}))

    with col2:
        st.subheader("y (Target)")
        y_raw = df['class']
        st.dataframe(y_raw.to_frame().style.set_properties(**{'background-color': '#e8f4ea', 'color': 'black'}))

with st.sidebar:
    st.header("Введите признаки: ")

    if st.button("🎲 Выбрать случайный образец"):
        random_sample = df.sample(1).iloc[0]
        variance = random_sample["variance"]
        skewness = random_sample["skewness"]
        curtosis = random_sample["curtosis"]
        entropy = random_sample["entropy"]
    else:
        variance = st.slider('Variance', float(df["variance"].min()), float(df["variance"].max()), float(df["variance"].mean()))
        skewness = st.slider('Skewness', float(df["skewness"].min()), float(df["skewness"].max()), float(df["skewness"].mean()))
        curtosis = st.slider('Curtosis', float(df["curtosis"].min()), float(df["curtosis"].max()), float(df["curtosis"].mean()))
        entropy = st.slider('Entropy', float(df["entropy"].min()), float(df["entropy"].max()), float(df["entropy"].mean()))

    st.write(f"Выбранные значения:\n- Variance: {variance}\n- Skewness: {skewness}\n- Curtosis: {curtosis}\n- Entropy: {entropy}")
