import numpy as np
import pandas as pd


def regulate_invocation_traces(
    azure_file_path="azurefunctions-dataset2019/",
    sample_trace="sampled_invocation_traces.csv",
    max_timestep=60
):  
    df = pd.read_csv(azure_file_path + sample_trace)

    drop_list = ["HashOwner", "HashApp"]
    for i in range(max_timestep, 1440):
        drop_list.append("{}".format(i+1))

    new_df = df.drop(drop_list, axis=1)

    new_df.to_csv(azure_file_path + "regulated_" + sample_trace, index=False)

if __name__ == "__main__":
    regulate_invocation_traces(
        azure_file_path="azurefunctions-dataset2019/",
        sample_trace="sampled_invocation_traces.csv",
        max_timestep=60
    )
