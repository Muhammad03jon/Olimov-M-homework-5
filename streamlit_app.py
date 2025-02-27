import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Предсказание реальной или фальшивой банкноты')

file_path = "https://raw.githubusercontent.com/Muhammad03jon/Olimov-M-homework-5/refs/heads/master/data_banknote_authentication.txt"

with open(file_path, "r") as file:
    lines = file.readlines()

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
        y_raw = df.class
        st.dataframe(y_raw.to_frame().style.set_properties(**{'background-color': '#e8f4ea', 'color': 'black'}))


