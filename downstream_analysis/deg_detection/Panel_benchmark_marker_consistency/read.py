import numpy as np
import pandas as pd


data_all = []
data_all_values = np.zeros((12,12))
for i in range(1,10):
    df = pd.read_csv("/ailab/user/chenpengan/fanyimin/retrieval_clean_codebase/DEG_analysis/Panel_benchmark_marker_consistency/marker_overlap/" + str(i) + ".csv")
    df = df.set_index("Unnamed: 0")
    data_all.append(df)
    data_all_values += df.values
data_all_values = data_all_values / 9
data_all_values = pd.DataFrame(data_all_values)
data_all_values.columns = df.columns
data_all_values.index = df.index
data_all_values.to_csv("full_result_marker_overlap.csv")


data_all = []
data_all_values = np.zeros((12,12))
for i in range(1,10):
    df = pd.read_csv("/ailab/user/chenpengan/fanyimin/retrieval_clean_codebase/DEG_analysis/Panel_benchmark_marker_consistency/deg_overlap/" + str(i) + ".csv")
    df = df.set_index("Unnamed: 0")
    data_all.append(df)
    data_all_values += df.values
data_all_values = data_all_values / 9
data_all_values = pd.DataFrame(data_all_values)
data_all_values.columns = df.columns
data_all_values.index = df.index
data_all_values.to_csv("full_result_deg_overlap.csv")

data_all = []
data_all_values = np.zeros(12)
for i in range(1,10):
    df = pd.read_csv("/ailab/user/chenpengan/fanyimin/retrieval_clean_codebase/DEG_analysis/Panel_benchmark_marker_consistency/deg_voting/" + str(i) + ".csv")
    df = df.set_index("Unnamed: 0")
    data_all.append(df)
    data_all_values += df.values.flatten()
data_all_values = data_all_values / 9
data_all_values = pd.DataFrame(data_all_values)
data_all_values.index = df.index
breakpoint()
data_all_values.to_csv("full_result_deg_voting.csv")