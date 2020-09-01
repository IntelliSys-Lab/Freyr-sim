import numpy as np
import pandas as pd



#
# Import Azure Functions traces
#
def load_from_azure(
    azure_file_path="azurefunctions-dataset2019/"
):     
    csv_suffix = ".csv"

    # Memory traces
    memory_traces = []
    n_memory_files = ["01", "02", "03", "04", "05", "06", "07", \
        "08", "09", "10", "11", "12"]
    memory_prefix = "app_memory_percentiles.anon.d"

    for n in n_memory_files:
        df = pd.read_csv(azure_file_path + memory_prefix + n + csv_suffix)
        for _, row in df.iterrows():
            memory_traces.append(row)

    # Duration traces
    duration_traces = []
    n_duration_files = ["01", "02", "03", "04", "05", "06", "07", \
        "08", "09", "10", "11", "12", "13", "14"]
    duration_prefix = "function_durations_percentiles.anon.d"

    for n in n_duration_files:
        df = pd.read_csv(azure_file_path + duration_prefix + n + csv_suffix)
        for _, row in df.iterrows():
            duration_traces.append(row)

    # Invocation traces
    invocation_traces = []
    n_invocation_files = ["01", "02", "03", "04", "05", "06", "07", \
        "08", "09", "10", "11", "12", "13", "14"]
    invocation_prefix = "invocations_per_function_md.anon.d"

    for n in n_invocation_files:
        df = pd.read_csv(azure_file_path + invocation_prefix + n + csv_suffix)
        for _, row in df.iterrows():
            invocation_traces.append(row)

    return memory_traces, duration_traces, invocation_traces


def characterize_memory_distribution(
    memory_traces,
    max_memory=1536,
    interval=128
):
    # Memory levels: e.g. max_memory=1536, interval=128
    # [0, 128), [128, 256), [256, 384), ..., [1408, 1536]
    n_levels = int(max_memory / interval)
    dist = []
    for _ in range(n_levels):
        dist.append([])

    for trace in memory_traces:
        dist_index = int(trace["AverageAllocatedMb"] / interval)
        if dist_index > n_levels:
            dist_index = n_levels

        dist[dist_index].append(trace)

    return dist


def sample_from_distribution(
    memory_dist,
    n_apps=4
):
    # Calculate total number of applications
    total_apps = 0
    for level in memory_dist:
        total_apps = total_apps + len(level)

    # Calculate sampling number of each level, at least one sample
    sample_num_list = []
    for level in memory_dist:
        sample_num = int(n_apps * len(level) / total_apps)
        if sample_num < 1:
            sample_num = 1

        sample_num_list.append(sample_num)

    # Random sample for each level
    app_hash_list = []
    for i in range(len(sample_num_list)):
        app_index_list = np.random.choice(
            a=len(memory_dist[i]), 
            size=sample_num_list[i],
            replace=False
        )

        for index in app_index_list:
            app_hash_list.append(memory_dist[i][index]["HashApp"])

    return app_hash_list
        

def save_sampled_traces(
    memory_traces,
    duration_traces,
    invocation_traces,
    app_hash_list,
    save_file_path="azurefunctions-dataset2019/"
):
    # Gather information and save as csv files
    sampled_memory_traces = []
    sampled_duration_traces = []
    sampled_invocation_traces = []

    # First visit, which is time-efficient
    for app_hash in app_hash_list:
        # Save sampled memory traces
        for trace in memory_traces:
            if app_hash == trace["HashApp"]:
                sampled_memory_traces.append(trace)
                break

        # Save sampled duration traces
        function_hash_list = []
        for trace in duration_traces:
            if app_hash == trace["HashApp"] and \
                trace["HashFunction"] not in function_hash_list:
                function_hash_list.append(trace["HashFunction"])
                sampled_duration_traces.append(trace)

        # Save sampled invocation traces
        function_hash_list = []
        for trace in invocation_traces:
            if app_hash == trace["HashApp"] and \
                trace["HashFunction"] not in function_hash_list:
                function_hash_list.append(trace["HashFunction"])
                sampled_invocation_traces.append(trace)

    pd.DataFrame(sampled_memory_traces).to_csv(
        save_file_path + "sampled_memory_traces.csv",
        index=False
    )
    pd.DataFrame(sampled_duration_traces).to_csv(
        save_file_path + "sampled_duration_traces.csv",
        index=False
    )
    pd.DataFrame(sampled_invocation_traces).to_csv(
        save_file_path + "sampled_invocation_traces.csv",
        index=False
    )

                

if __name__ == "__main__":
    azure_file_path = "azurefunctions-dataset2019/"
    n_apps = 4

    memory_traces, duration_traces, invocation_traces = load_from_azure(azure_file_path)
    dist = characterize_memory_distribution(
        memory_traces,
        max_memory=1536,
        interval=256
    )
    app_hash_list = sample_from_distribution(dist, n_apps)

    print("Start saving...")
    save_sampled_traces(
        memory_traces=memory_traces,
        duration_traces=duration_traces,
        invocation_traces=invocation_traces,
        app_hash_list=app_hash_list,
        save_file_path=azure_file_path
    )
    print("Finish saving!")

