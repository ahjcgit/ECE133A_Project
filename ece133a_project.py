# -*- coding: utf-8 -*-
"""ECE133A_Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13ENFnhEFtHiV2Yuxh3HlG21-0PAS8A7I
"""
import pandas as pd
import numpy as np

df = pd.read_csv("./OnlineNewsPopularity.csv")

# Preprocessing & Clean-up
url_array = df.iloc[:, 0].values  # extract news urls mapped by index
df = df.iloc[:, 1:]  # remove first column urls
df = df.iloc[:, 1:]  # remove second column timedelta

missing_vals = df.isnull().sum().sum()
# print(f"total missing values: {missing_vals}")
df_mod = df.dropna().reset_index(drop=True)  # remove rows with missing or null values

m_p1 = df_mod.to_numpy()  # phase 1 matrix


print("Dim: ", m_p1.shape)
print(url_array)
print(m_p1)

