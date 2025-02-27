import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Предсказание реальной или фальшивой банкноты')

file_path = "data_banknote_authentication.txt"

with open(file_path, "r") as file:
    lines = file.readlines()

df = pd.read_csv(file_path, sep=",", header=None)
df.columns = ["variance", "skewness", "curtosis", "entropy", "class"]

