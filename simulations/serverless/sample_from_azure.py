import numpy as np
import pandas as pd
import os
import stat
import sys
from glob import glob


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
        df = pd.read_csv(azure_file_path + memory_prefix + n + csv_suffix, index_col="HashApp")
        memory_traces.append(df)

    memory_df = pd.concat(memory_traces)
    memory_df = memory_df[~memory_df.index.duplicated()]
    memory_dict = memory_df.to_dict('index')

    # Duration traces
    duration_traces = []
    n_duration_files = ["01", "02", "03", "04", "05", "06", "07", \
        "08", "09", "10", "11", "12", "13", "14"]
    duration_prefix = "function_durations_percentiles.anon.d"

    for n in n_duration_files:
        df = pd.read_csv(azure_file_path + duration_prefix + n + csv_suffix, index_col="HashFunction")
        duration_traces.append(df)

    duration_df = pd.concat(duration_traces)
    duration_df = duration_df[~duration_df.index.duplicated()]
    duration_dict = duration_df.to_dict('index')

    # Invocation traces
    invocation_traces = []
    n_invocation_files = ["01", "02", "03", "04", "05", "06", "07", \
        "08", "09", "10", "11", "12", "13", "14"]
    invocation_prefix = "invocations_per_function_md.anon.d"

    for n in n_invocation_files:
        df = pd.read_csv(azure_file_path + invocation_prefix + n + csv_suffix, index_col="HashFunction")
        invocation_traces.append(df)

    invocation_df = pd.concat(invocation_traces)
    invocation_df = invocation_df[~invocation_df.index.duplicated()]
    invocation_dict = invocation_df.to_dict('index')

    return memory_dict, duration_dict, invocation_dict

#
# Sample from trigger distribution 
#

def sample_from_trigger_dist(
    memory_dict,
    duration_dict,
    invocation_dict,
    trigger_dist,
    max_function=50,
    max_load_per_trace=600,
    max_timestep=60,
    save_file_path="azurefunctions-dataset2019/",
    save_file_id=""
):
    sampled_memory_traces = {}
    sampled_duration_traces = {}
    sampled_invocation_traces = {}

    # Classify based on trigger types
    trigger_dict = {}

    for func_hash in invocation_dict.keys():
        trace = invocation_dict[func_hash]
        trigger = trace["Trigger"]
        if trigger not in trigger_dict:
            trigger_dict[trigger] = []

        trace["HashFunction"] = func_hash
        trigger_dict[trigger].append(trace)

    # Random sample
    for trigger in trigger_dict.keys():
        # Apply load limit
        index_list = []

        done = False
        while done is False:
            index = np.random.randint(len(trigger_dict[trigger]))
            invoke_list = []
            trace = trigger_dict[trigger][index]
            for timestep in range(max_timestep):
                invoke_list.append(int(trace["{}".format(timestep+1)]))

            if np.sum(invoke_list) <= max_load_per_trace and index not in index_list:
                index_list.append(index)

            if len(index_list) >= int(max_function * trigger_dist[trigger]) + 1:
                done = True

        for index in index_list:
            trace = trigger_dict[trigger][index]
            func_hash = trace["HashFunction"]
            app_hash = trace["HashApp"]

            # Save sampled invocation traces
            sampled_invocation_traces[func_hash] = trace

    # Save sampled duration traces
    for func_hash in sampled_invocation_traces.keys():
        if func_hash in duration_dict:
            trace = duration_dict[func_hash]
            trace["HashFunction"] = func_hash
            sampled_duration_traces[func_hash] = trace

    # Regularize duration and invocation traces
    while len(sampled_duration_traces) != len(sampled_invocation_traces):
        if len(sampled_duration_traces) < len(sampled_invocation_traces):
            redundant_list = []
            for func_hash in sampled_invocation_traces.keys():
                if func_hash not in sampled_duration_traces:
                    redundant_list.append(func_hash)

            for func_hash in redundant_list:
                sampled_invocation_traces.pop(func_hash)

        elif len(sampled_duration_traces) > len(sampled_invocation_traces):
            redundant_list = []
            for func_hash in sampled_duration_traces.keys():
                if func_hash not in sampled_invocation_traces:
                    redundant_list.append(func_hash)
                
                for func_hash in redundant_list:
                    sampled_duration_traces.pop(func_hash)

    # Save sampled memory traces 
    redundant_list = []
    for func_hash in sampled_duration_traces.keys():
        app_hash = sampled_duration_traces[func_hash]["HashApp"]
        if app_hash not in memory_dict:
            redundant_list.append(func_hash)
        else:
            trace = memory_dict[app_hash]
            trace["HashApp"] = app_hash
            sampled_memory_traces[app_hash] = trace

    for func_hash in redundant_list:
        sampled_invocation_traces.pop(func_hash)
        sampled_duration_traces.pop(func_hash)

    # Write to CSV
    new_memory_df = pd.DataFrame.from_records(list(sampled_memory_traces.values()))
    new_duration_df = pd.DataFrame.from_records(list(sampled_duration_traces.values()))
    drop_list = [] # Drop timesteps that are beyond max timestep
    for i in range(max_timestep, 1440):
        drop_list.append("{}".format(i+1))
    new_invocation_df = pd.DataFrame.from_records(list(sampled_invocation_traces.values())).drop(drop_list, axis=1)

    new_memory_df.to_csv(
        save_file_path + "sampled_memory_traces_{}.csv".format(save_file_id),
        index=False
    )
    new_duration_df.to_csv(
        save_file_path + "sampled_duration_traces_{}.csv".format(save_file_id),
        index=False
    )
    new_invocation_df.to_csv(
        save_file_path + "sampled_invocation_traces_{}.csv".format(save_file_id),
        index=False
    )

#
# Clean old sample files
#

def clean_old_samples(
    dir_name="azurefunctions-dataset2019/",
    file_pattern="sampled_*.csv"
):
    for file_name in glob(os.path.join(dir_name, file_pattern)):
        try:
            os.remove(file_name)
        except EnvironmentError:
            print("Demand permission to {}".format(file_name))
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)


if __name__ == "__main__":
    azure_file_path = "azurefunctions-dataset2019/"
    max_workload = 10
    max_function = 100
    max_timestep = 600
    max_load_per_trace = 10000
    trigger_dist = {
        "http": 0.359,
        "queue": 0.335,
        "event": 0.247,
        "orchestration": 0.023,
        "timer": 0.02,
        "storage": 0.007,
        "others": 0.01
    }

    print("Clean old sample files...")
    clean_old_samples(dir_name=azure_file_path, file_pattern="sampled_*.csv")

    print("Loading Azure Functions traces...")
    memory_dict, duration_dict, invocation_dict = load_from_azure(azure_file_path)
    
    print("Sampling from trigger distribution...")
    for i in range(max_workload):
        print("Sampling {} workload...".format(i))
        sample_from_trigger_dist(
            memory_dict=memory_dict,
            duration_dict=duration_dict,
            invocation_dict=invocation_dict,
            trigger_dist=trigger_dist,
            max_function=max_function,
            max_timestep=max_timestep,
            max_load_per_trace=max_load_per_trace,
            save_file_path="azurefunctions-dataset2019/",
            save_file_id=i
        )

    print("Sampling finished!")

