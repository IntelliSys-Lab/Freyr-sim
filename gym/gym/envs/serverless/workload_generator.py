import pandas as pd
import heapq
import itertools
import numpy as np
import gym
from gym.envs.serverless.faas_utils import Function, Profile, EventPQ
from gym.envs.serverless.faas_params import FunctionParameters, EventPQParameters


class WorkloadGenerator():
    """
    Generate workfloads
    """
    def __init__(
        self,
        exp_id,
        cpu_cap_per_function,
        memory_cap_per_function,
        memory_mb_limit,
        azure_file_path="../../simulation/serverless/azurefunctions-dataset2019/"
    ):
        self.azure_file_path = azure_file_path
        self.exp_id = exp_id
        self.cpu_cap_per_function = cpu_cap_per_function
        self.memory_cap_per_function = memory_cap_per_function
        self.memory_mb_limit = memory_mb_limit

    def generate_params(
        self,
        memory_traces_file,
        duration_traces_file,
        invocation_traces_file
    ):
        memory_traces = pd.read_csv(self.azure_file_path + memory_traces_file, index_col="HashFunction")
        memory_traces_dict = memory_traces.to_dict('index')
        duration_traces = pd.read_csv(self.azure_file_path + duration_traces_file, index_col="HashFunction")
        duration_traces_dict = duration_traces.to_dict('index')
        invocation_traces = pd.read_csv(self.azure_file_path + invocation_traces_file, index_col="HashFunction")
        invocation_traces_dict = invocation_traces.to_dict('index')
        
        cpu_level = self.memory_mb_limit / self.cpu_cap_per_function
        memory_level = self.memory_mb_limit / self.memory_cap_per_function

        function_params_dict = {}
        for function_hash in invocation_traces_dict.keys():
            function_params_dict[function_hash] = {}
            memory_trace = memory_traces_dict[function_hash]
            duration_trace = duration_traces_dict[function_hash]
            invocation_trace = invocation_traces_dict[function_hash]

            # Memory
            function_params_dict[function_hash]["ideal_memory"] = np.clip(int(memory_trace["Ideal"]), 1, self.memory_mb_limit)
            function_params_dict[function_hash]["ideal_cpu"] = np.clip(int(memory_trace["Ideal"]/cpu_level), 1, self.cpu_cap_per_function)
            function_params_dict[function_hash]["memory_least_hint"] = 1
            function_params_dict[function_hash]["cpu_least_hint"] = 1
            function_params_dict[function_hash]["memory_user_defined"] = np.clip(int(memory_trace["AverageAllocatedMb"]/memory_level), 1, self.memory_cap_per_function)
            function_params_dict[function_hash]["cpu_user_defined"] = np.clip(int(memory_trace["AverageAllocatedMb"]/cpu_level), 1, self.cpu_cap_per_function)
            function_params_dict[function_hash]["memory_cap_per_function"] = self.memory_cap_per_function
            function_params_dict[function_hash]["cpu_cap_per_function"] = self.cpu_cap_per_function
            function_params_dict[function_hash]["memory_mb_limit"] = self.memory_mb_limit

            # Duration
            function_params_dict[function_hash]["max_duration"] = int(duration_trace["percentile_Average_100"]/1000) + 1 
            function_params_dict[function_hash]["min_duration"] = int(duration_trace["percentile_Average_1"]/1000) + 1 
            function_params_dict[function_hash]["cold_start_time"] = int(duration_trace["Minimum"]/1000) + 1 
            
            # Max timeout limit
            # Reference: https://docs.microsoft.com/en-us/azure/azure-functions/functions-scale
            if invocation_trace["Trigger"] == "http":
                function_params_dict[function_hash]["timeout"] = 230
            else:
                function_params_dict[function_hash]["timeout"] = 600

        # Create Profile paramters
        profile_params = {}
        hash_value = 0
        for function_hash in function_params_dict.keys():
            function_params = FunctionParameters(
                ideal_cpu=function_params_dict[function_hash]["ideal_cpu"], 
                ideal_memory=function_params_dict[function_hash]["ideal_memory"],
                max_duration=function_params_dict[function_hash]["max_duration"],
                min_duration=function_params_dict[function_hash]["min_duration"],
                cpu_least_hint=function_params_dict[function_hash]["cpu_least_hint"],
                memory_least_hint=function_params_dict[function_hash]["memory_least_hint"],
                memory_mb_limit=function_params_dict[function_hash]["memory_mb_limit"],
                timeout=function_params_dict[function_hash]["timeout"],
                cpu_cap_per_function=function_params_dict[function_hash]["cpu_cap_per_function"],
                memory_cap_per_function=function_params_dict[function_hash]["memory_cap_per_function"],
                cpu_user_defined=function_params_dict[function_hash]["cpu_user_defined"],
                memory_user_defined=function_params_dict[function_hash]["memory_user_defined"],
                function_id=function_hash,
                hash_value=hash_value,
                cold_start_time=function_params_dict[function_hash]["cold_start_time"],
                k=2
            )

            profile_params[function_params.function_id] = function_params
            hash_value = hash_value + 1

        # Create discrete events based on invocation traces
        event_pq_params = EventPQParameters(azure_invocation_traces=invocation_traces.reset_index())

        return profile_params, event_pq_params

    def generate_workload(self):
        # Generate parameters
        profile_params, event_pq_params = self.generate_params(
            memory_traces_file="sampled_memory_traces_{}.csv".format(self.exp_id),
            duration_traces_file="sampled_duration_traces_{}.csv".format(self.exp_id),
            invocation_traces_file="sampled_invocation_traces_{}.csv".format(self.exp_id)
        )

        # Generate profile
        function_profile = {}
        
        for function_id in profile_params.keys():
            param = profile_params[function_id]
            function = Function(param)

            # Initially defined by users
            function.set_function(
                cpu=param.cpu_user_defined, 
                memory=param.memory_user_defined
            )
            function.set_baseline()
            function_profile[function_id] = function
        
        profile = Profile(function_profile=function_profile)

        # Generate event pq
        invocation_traces = event_pq_params.azure_invocation_traces
        max_timestep = len(invocation_traces.columns) - 2

        pq = []
        counter = itertools.count()
        for timestep in range(max_timestep):
            for _, row in invocation_traces.iterrows():
                function_id = row["HashFunction"]
                invoke_num = row["{}".format(timestep+1)]
                for _ in range(invoke_num):
                    heapq.heappush(pq, (timestep, next(counter), function_id))

        event_pq = EventPQ(
            pq=pq,
            max_timestep=max_timestep
        )

        return profile, event_pq
    