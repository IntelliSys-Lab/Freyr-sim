import sys
sys.path.append("../../gym")
import scipy.stats as stats
import pandas as pd
import numpy as np
import gym
from gym.envs.serverless.faas_utils import Function, Profile, Timetable
from gym.envs.serverless.faas_params import FunctionParameters, TimetableParameters
import params


class WorkloadGenerator():
    """
    Generate workloads running in FaaSEnv
    """
    def ensure_params(self):
        hash_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Generate profile parameters
        ET_Image_Resizing_params = FunctionParameters(
            function_id="ET_Image_Resizing",
            ideal_cpu=4, 
            ideal_memory=3, 
            ideal_duration=1, 
            cpu_least_hint=1, 
            memory_least_hint=1,
            timeout=60,
            cpu_cap_per_function=8,
            memory_cap_per_function=8,
            hash_value=hash_value.pop(),
            cold_start_time=0.5,
        )
        ET_Streaming_Analytics_params = FunctionParameters(
            function_id="ET_Streaming_Analytics",
            ideal_cpu=4, 
            ideal_memory=4, 
            ideal_duration=1, 
            cpu_least_hint=1, 
            memory_least_hint=1,
            timeout=60,
            cpu_cap_per_function=8,
            memory_cap_per_function=8,
            hash_value=hash_value.pop(),
            cold_start_time=1,
        )
        ET_Email_Gen_params = FunctionParameters(
            function_id="ET_Email_Gen",
            ideal_cpu=4, 
            ideal_memory=4, 
            ideal_duration=2, 
            cpu_least_hint=1, 
            memory_least_hint=1,
            timeout=60,
            cpu_cap_per_function=8,
            memory_cap_per_function=8,
            hash_value=hash_value.pop(),
            cold_start_time=1,
        )
        ET_Stock_Analysis_params = FunctionParameters(
            function_id="ET_Stock_Analysis",
            ideal_cpu=4, 
            ideal_memory=3, 
            ideal_duration=3, 
            cpu_least_hint=1, 
            memory_least_hint=1,
            timeout=60,
            cpu_cap_per_function=8,
            memory_cap_per_function=8,
            hash_value=hash_value.pop(),
            cold_start_time=1,
        )
        ET_File_Encrypt_params = FunctionParameters(
            function_id="ET_File_Encrypt",
            ideal_cpu=4, 
            ideal_memory=4, 
            ideal_duration=4, 
            cpu_least_hint=1, 
            memory_least_hint=1,
            timeout=60,
            cpu_cap_per_function=8,
            memory_cap_per_function=8,
            hash_value=hash_value.pop(),
            cold_start_time=1.5,
        )
        ET_Sentiment_Review_params = FunctionParameters(
            function_id="ET_Sentiment_Review",
            ideal_cpu=4, 
            ideal_memory=3, 
            ideal_duration=5, 
            cpu_least_hint=1, 
            memory_least_hint=1,
            timeout=60,
            cpu_cap_per_function=8,
            memory_cap_per_function=8,
            hash_value=hash_value.pop(),
            cold_start_time=2,
        )
        MP_Nearest_Neighbor_params = FunctionParameters(
            function_id="MP_Nearest_Neighbor",
            ideal_cpu=6, 
            ideal_memory=6, 
            ideal_duration=10, 
            cpu_least_hint=1, 
            memory_least_hint=1,
            timeout=600,
            cpu_cap_per_function=8,
            memory_cap_per_function=8,
            hash_value=hash_value.pop(),
            cold_start_time=3,
        ) 
        MP_Comp_Fluid_Dynamics_params = FunctionParameters(
            function_id="MP_Comp_Fluid_Dynamics",
            ideal_cpu=8, 
            ideal_memory=4, 
            ideal_duration=21, 
            cpu_least_hint=1, 
            memory_least_hint=1,
            timeout=600,
            cpu_cap_per_function=8,
            memory_cap_per_function=8,
            hash_value=hash_value.pop(),
            cold_start_time=5,
        )
        MP_Sorting_params = FunctionParameters(
            function_id="MP_Sorting",
            ideal_cpu=8, 
            ideal_memory=4, 
            ideal_duration=45, 
            cpu_least_hint=1, 
            memory_least_hint=1,
            timeout=600,
            cpu_cap_per_function=8,
            memory_cap_per_function=8,
            hash_value=hash_value.pop(),
            cold_start_time=5,
        ) 
        MP_Matrix_Multiply_params = FunctionParameters(
            function_id="MP_Matrix_Multiply",
            ideal_cpu=8, 
            ideal_memory=6, 
            ideal_duration=20, 
            cpu_least_hint=1, 
            memory_least_hint=1,
            timeout=600,
            cpu_cap_per_function=8,
            memory_cap_per_function=8,
            hash_value=hash_value.pop(),
            cold_start_time=4,
        )

        # Create Profile paramters
        profile_params = {}
        profile_params["ET_Image_Resizing"] = ET_Image_Resizing_params
        profile_params["ET_Streaming_Analytics"] = ET_Streaming_Analytics_params
        profile_params["ET_Email_Gen"] = ET_Email_Gen_params
        profile_params["ET_Stock_Analysis"] = ET_Stock_Analysis_params
        profile_params["ET_File_Encrypt"] = ET_File_Encrypt_params
        profile_params["ET_Sentiment_Review"] = ET_Sentiment_Review_params
        profile_params["MP_Nearest_Neighbor"] = MP_Nearest_Neighbor_params
        profile_params["MP_Comp_Fluid_Dynamics"] = MP_Comp_Fluid_Dynamics_params
        profile_params["MP_Sorting"] = MP_Sorting_params
        profile_params["MP_Matrix_Multiply"] = MP_Matrix_Multiply_params
        
        # Generate timetable parameters
        mod = [1, 1, 1, 1, 2, 2, 4, 4, 8, 10]
        
        timetable_params = TimetableParameters(
            max_timestep=60, 
            distribution_type="mod",
            mod_factors=mod
        )
            
        return profile_params, timetable_params
    
    def azure_params(
        self,
        max_timestep,
        azure_file_path,
        memory_traces_file,
        duration_traces_file,
        invocation_traces_file
    ):
        memory_traces = pd.read_csv(azure_file_path + memory_traces_file)
        duration_traces = pd.read_csv(azure_file_path + duration_traces_file)
        invocation_traces = pd.read_csv(azure_file_path + invocation_traces_file)

        cpu_cap_per_function = params.cpu_cap_per_function
        memory_cap_per_function = params.memory_cap_per_function
        # 1536 is the max memory allowed by Azure Functions 
        level = 1536 / memory_cap_per_function

        function_params_dict = {}

        # Retrieve function hash and its corresponding application hash
        for _, row in duration_traces.iterrows():
            function_hash = row["HashFunction"]
            app_hash = row["HashApp"]

            function_params_dict[function_hash] = {}
            function_params_dict[function_hash]["HashApp"] = app_hash

        for function_hash in function_params_dict.keys():
            for _, row in memory_traces.iterrows():
                if row["HashApp"] == function_params_dict[function_hash]["HashApp"]:
                    function_params_dict[function_hash]["ideal_memory"] = np.clip(int(row["AverageAllocatedMb_pct100"]/level) + 1, 1, cpu_cap_per_function)
                    function_params_dict[function_hash]["ideal_cpu"] = np.clip(int(row["AverageAllocatedMb_pct100"]/level) + 1, 1, memory_cap_per_function)
                    function_params_dict[function_hash]["memory_least_hint"] = 1
                    function_params_dict[function_hash]["cpu_least_hint"] = 1
                    function_params_dict[function_hash]["memory_user_defined"] = np.clip(int(row["AverageAllocatedMb_pct1"]/level) + 1, 1, cpu_cap_per_function)
                    function_params_dict[function_hash]["cpu_user_defined"] = np.clip(int(row["AverageAllocatedMb_pct1"]/level) + 1, 1, memory_cap_per_function)
                    function_params_dict[function_hash]["memory_cap_per_function"] = cpu_cap_per_function
                    function_params_dict[function_hash]["cpu_cap_per_function"] = memory_cap_per_function
                    break

            for _, row in duration_traces.iterrows():
                if row["HashFunction"] == function_hash:
                    # Millisec to sec
                    function_params_dict[function_hash]["max_duration"] = int(row["percentile_Average_100"]/1000) + 1 
                    function_params_dict[function_hash]["min_duration"] = int(row["percentile_Average_1"]/1000) + 1 
                    function_params_dict[function_hash]["cold_start_time"] = int(row["Minimum"]/1000) + 1 

            for _, row in invocation_traces.iterrows():
                if row["HashFunction"] == function_hash:
                    # Max timeout limit
                    # Reference: https://docs.microsoft.com/en-us/azure/azure-functions/functions-scale
                    if row["Trigger"] == "http":
                        function_params_dict[function_hash]["timeout"] = 230 
                    else:
                        function_params_dict[function_hash]["timeout"] = 600

        # Create Profile paramters and sequence dictionary
        profile_params = {}
        sequence_dict = {}
        for application_id in memory_traces["HashApp"]:
            if application_id not in sequence_dict.keys():
                sequence_dict[application_id] = []

        hash_value = 0
        for function_hash in function_params_dict.keys():
            function_params = FunctionParameters(
                ideal_cpu=function_params_dict[function_hash]["ideal_cpu"], 
                ideal_memory=function_params_dict[function_hash]["ideal_memory"],
                max_duration=function_params_dict[function_hash]["max_duration"],
                min_duration=function_params_dict[function_hash]["min_duration"],
                cpu_least_hint=function_params_dict[function_hash]["cpu_least_hint"],
                memory_least_hint=function_params_dict[function_hash]["memory_least_hint"],
                timeout=function_params_dict[function_hash]["timeout"],
                cpu_cap_per_function=function_params_dict[function_hash]["cpu_cap_per_function"],
                memory_cap_per_function=function_params_dict[function_hash]["memory_cap_per_function"],
                cpu_user_defined=function_params_dict[function_hash]["cpu_user_defined"],
                memory_user_defined=function_params_dict[function_hash]["memory_user_defined"],
                application_id=function_params_dict[function_hash]["HashApp"],
                function_id=function_hash,
                hash_value=hash_value,
                cold_start_time=function_params_dict[function_hash]["cold_start_time"],
                k=2
            )

            profile_params[function_params.function_id] = function_params
            sequence_dict[function_params.application_id].append(function_params.function_id)
            hash_value = hash_value + 1

        # Add sequence inforation to each entry function
        for application_id in sequence_dict.keys():
            total_sequence = sequence_dict[application_id]
            if len(total_sequence) > 1:
                entry = total_sequence[0]
                profile_params[entry].sequence = total_sequence[1:]

        # Create timetable based on invocation traces
        timetable_params = TimetableParameters(
            max_timestep=max_timestep, 
            distribution_type="azure",
            azure_invocation_traces=invocation_traces
        )

        return profile_params, timetable_params

    def generate_profile(self, profile_params):
        function_profile = {}
        
        for function_id in profile_params.keys():
            param = profile_params[function_id]
            function = Function(param)

            # function.set_function(
            #     cpu=param.cpu_cap_per_function, 
            #     memory=param.memory_cap_per_function
            # ) # Initially over-provision

            function.set_function(
                cpu=param.cpu_user_defined, 
                memory=param.memory_user_defined
            ) # Initially user defined

            # function.set_function(
            #     cpu=param.cpu_least_hint, 
            #     memory=param.memory_least_hint
            # ) # Initially set as hinted
            
            function_profile[function_id] = function
        
        profile = Profile(function_profile=function_profile)
        return profile
    
    def mod_distribution(
        self, 
        profile,
        timetable_params
    ):
        max_timestep = timetable_params.max_timestep
        mod_factors = timetable_params.mod_factors
        
        function_profile_list = list(profile.get_function_profile().keys())
        timetable_list = []
        
        for timestep_i in range(max_timestep):
            timestep = {}
            
            for function_i, factor in enumerate(mod_factors):
                function_id = function_profile_list[function_i]
                if timestep_i % factor == 0:
                    timestep[function_id] = 1
                else:
                    timestep[function_id] = 0

            timetable_list.append(timestep)

        timetable = Timetable(timetable_list)
        return timetable
    
    def poisson_distribution(
        self,
        profile,
        timetable_params
    ):
        max_timestep = timetable_params.max_timestep
        mu = timetable_params.poisson_mu
        
        function_profile_list = list(profile.get_function_profile().keys())
        timetable_list = []
        poisson_time_list = []
        
        for _ in function_profile_list:
            poisson_time_list.append(
                stats.poisson.rvs(
                    mu=mu, 
                    size=max_timestep
                )
            )
        
        for timestep_i in range(max_timestep):
            timestep = {}
            
            for function_i, invoke_num in enumerate(poisson_time_list):
                function_id = function_profile_list[function_i]
                timestep[function_id] = invoke_num
                
            timetable_list.append(timestep)
            
        timetable = Timetable(timetable_list)
        return timetable
    
    def bernoulli_distribution(
        self,
        profile,
        timetable_params
    ):
        max_timestep = timetable_params.max_timestep
        p = timetable_params.bernoulli_p
        
        function_profile_list = list(profile.get_function_profile().keys())
        timetable_list = []
        bernoulli_time_list = []
        
        for _ in function_list:
            bernoulli_time_list.append(
                stats.bernoulli.rvs(
                    p=p, 
                    size=max_timestep
                )
            )
        
        for timestep_i in range(max_timestep):
            timestep = {}
            
            for function_i, invoke_num in enumerate(bernoulli_time_list):
                function_id = function_profile_list[function_i]
                timestep[function_id] = invoke_num
                
            timetable_list.append(timestep)
            
        timetable = Timetable(timetable_list)
        return timetable

    def azure_distribution(
        self,
        profile,
        timetable_params
    ):
        max_timestep = timetable_params.max_timestep
        invocation_traces = timetable_params.azure_invocation_traces

        function_profile_list = list(profile.get_function_profile().keys())
        timetable_list = []

        for timestep_i in range(max_timestep):
            timestep = {}

            for _, row in invocation_traces.iterrows():
                function_id = row["HashFunction"]
                invoke_num = row["{}".format(timestep_i+1)]
                timestep[function_id] = invoke_num

            timetable_list.append(timestep)
        
        timetable = Timetable(timetable_list)
        return timetable

    def generate_timetable(
        self, 
        profile,
        timetable_params
    ):
        if timetable_params.distribution_type == "mod":
            timetable = self.mod_distribution(profile, timetable_params)
        elif timetable_params.distribution_type == "bernoulli":
            timetable = self.bernoulli_distribution(profile, timetable_params)
        elif timetable_params.distribution_type == "poisson":
            timetable = self.poisson_distribution(profile, timetable_params)
        elif timetable_params.distribution_type == "azure":
            timetable = self.azure_distribution(profile, timetable_params)
        
        return timetable
    
    def generate_workload(
        self, 
        default="azure",
        profile_params=None, 
        timetable_params=None,
        max_timestep=600,
        azure_file_path="azurefunctions-dataset2019/",
        memory_traces_file="simple_memory_traces.csv",
        duration_traces_file="simple_duration_traces.csv",
        invocation_traces_file="simple_invocation_traces.csv"
    ):
        if default == "ensure":
            default_profile_params, default_timetable_params = self.ensure_params()
        elif default == "azure":
            default_profile_params, default_timetable_params = self.azure_params(
                max_timestep=max_timestep,
                azure_file_path=azure_file_path,
                memory_traces_file=memory_traces_file,
                duration_traces_file=duration_traces_file,
                invocation_traces_file=invocation_traces_file
            )

        if profile_params is None:
            profile_params = default_profile_params
        profile = self.generate_profile(profile_params)
            
        if timetable_params is None:
            timetable_params = default_timetable_params
        timetable = self.generate_timetable(profile, timetable_params)
            
        return profile, timetable