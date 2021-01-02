import numpy as np
import pandas as pd


def count_trace(
    azure_file_path="azurefunctions-dataset2019/",
    sample_trace="sampled_invocation_traces.csv",
    max_timestep=60
):  
    trace_dict = {}

    df = pd.read_csv(azure_file_path + sample_trace)

    drop_list = [0, 1]
    for i in range(max_timestep, 1440):
        drop_list.append(i+1)

    df.drop(df.columns[drop_list], axis=1)

    df.to_csv(azure_file_path + "count_" + sample_trace, index=False)
