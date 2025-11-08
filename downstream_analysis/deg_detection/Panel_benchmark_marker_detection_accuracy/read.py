import numpy as np
import pandas as pd
data_all = []

for i in range(1,10):
    df = pd.read_csv("./detection_accuracy/" + str(i) + ".csv")
    df = df.set_index("Unnamed: 0")
    data_all.append(df.mean(axis=1))
data_all_1 = pd.concat(data_all,axis=1)
data_all_1["Row_Sum"] = data_all_1.sum(axis=1)

data_sorted = data_all_1.sort_values(by="Row_Sum", ascending=False)

data_sorted_reset = data_sorted.reset_index()

print(data_sorted_reset)
data_sorted_reset.to_csv("full_result.csv")