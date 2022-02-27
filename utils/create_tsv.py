# coding=utf-8
import pandas as pd

df = pd.DataFrame({"query": ["abc", "ghj"],
                   "doc": ["dfsdf", "dfeew"],
                   "label": [1, 0]})
df.to_csv("E:\python_code\esim\input\\train.tsv", sep="\t", header=None, index=None)