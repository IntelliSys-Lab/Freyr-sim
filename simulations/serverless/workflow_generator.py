import sys
sys.path.append("../../gym")
import scipy.stats as stats
import pandas as pd
import gym
from gym.envs.serverless.faas_utils import Function, Application, Profile, Timetable
from gym.envs.serverless.faas_params import FunctionParameters, TimetableParameters


class WorkflowGenerator():
    """
    Generate workflows running on FaaSEnv
    """
    def ensure_params(self):
        # Generate profile parameters
        ET_Image_Resizing_params = FunctionParameters(
            ideal_cpu=4, 
            ideal_memory=3, 
            ideal_duration=1, 
            cpu_least_hint=3, 
            memory_least_hint=1,
            timeout=5,
            cpu_cap_per_function=32,
            memory_cap_per_function=45,
        )
        ET_Streaming_Analytics_params = FunctionParameters(
            ideal_cpu=5, 
            ideal_memory=2, 
            ideal_duration=1, 
            cpu_least_hint=4, 
            memory_least_hint=1,
            timeout=5,
            cpu_cap_per_function=32,
            memory_cap_per_function=45,
        )
        ET_Email_Gen_params = FunctionParameters(
            ideal_cpu=10, 
            ideal_memory=4, 
            ideal_duration=1, 
            cpu_least_hint=7, 
            memory_least_hint=2,
            timeout=5,
            cpu_cap_per_function=32,
            memory_cap_per_function=45,
        )
        ET_Stock_Analysis_params = FunctionParameters(
            ideal_cpu=9, 
            ideal_memory=3, 
            ideal_duration=1, 
            cpu_least_hint=5, 
            memory_least_hint=2,
            timeout=5,
            cpu_cap_per_function=32,
            memory_cap_per_function=45,
        )
        ET_File_Encrypt_params = FunctionParameters(
            ideal_cpu=11, 
            ideal_memory=5, 
            ideal_duration=1, 
            cpu_least_hint=6, 
            memory_least_hint=2,
            timeout=5,
            cpu_cap_per_function=32,
            memory_cap_per_function=45,
        )
        ET_Sentiment_Review_params = FunctionParameters(
            ideal_cpu=12, 
            ideal_memory=8, 
            ideal_duration=2, 
            cpu_least_hint=6, 
            memory_least_hint=4,
            timeout=5,
            cpu_cap_per_function=32,
            memory_cap_per_function=46,
        )
        MP_Nearest_Neighbor_params = FunctionParameters(
            ideal_cpu=32, 
            ideal_memory=12, 
            ideal_duration=9, 
            cpu_least_hint=22, 
            memory_least_hint=8,
            timeout=30,
            cpu_cap_per_function=32,
            memory_cap_per_function=45,
        ) 
        MP_Comp_Fluid_Dynamics_params = FunctionParameters(
            ideal_cpu=32, 
            ideal_memory=16, 
            ideal_duration=21, 
            cpu_least_hint=29, 
            memory_least_hint=8,
            timeout=60,
            cpu_cap_per_function=32,
            memory_cap_per_function=45,
        )
        MP_Sorting_params = FunctionParameters(
            ideal_cpu=32, 
            ideal_memory=28, 
            ideal_duration=45, 
            cpu_least_hint=29, 
            memory_least_hint=16,
            timeout=60,
            cpu_cap_per_function=32,
            memory_cap_per_function=45,
        ) 
        MP_Matrix_Multiply_params = FunctionParameters(
            ideal_cpu=32, 
            ideal_memory=6, 
            ideal_duration=20, 
            cpu_least_hint=28, 
            memory_least_hint=2,
            timeout=60,
            cpu_cap_per_function=32,
            memory_cap_per_function=45,
        )
        
        function_params = [
            ET_Image_Resizing_params, 
            ET_Streaming_Analytics_params,
            ET_Email_Gen_params,
            ET_Stock_Analysis_params,
            ET_File_Encrypt_params,
            ET_Sentiment_Review_params,
            MP_Nearest_Neighbor_params,
            MP_Comp_Fluid_Dynamics_params,
            MP_Sorting_params,
            MP_Matrix_Multiply_params
        ]
        application_params = []
        
        profile_params = [function_params, application_params]
        
        # Generate timetable parameters
        mod = [2, 2, 2, 2, 2, 5, 30, 30, 30, 30]
        
        timetable_params = TimetableParameters(
            max_timestep=200, 
            distribution_type="mod",
            mod_factors=mod
        )
            
        return profile_params, timetable_params
    
    def azure_params(
        self,
        azure_file_path="azurefunctions-dataset2019/",
        memory_traces_file="sampled_memory_traces.csv",
        duration_traces_file="sampled_duration_traces.csv",
        invocation_traces_file="sampled_invocation_traces.csv"
    ):
        memory_traces = pd.read_csv(azure_file_path + memory_traces_file)
        duration_traces = pd.read_csv(azure_file_path + duration_traces_file)
        invocation_traces = pd.read_csv(azure_file_path + invocation_traces_file)

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
                    function_params_dict[function_hash]["ideal_memory"] = int(45*row["AverageAllocatedMb_pct100"]/1536)
                    function_params_dict[function_hash]["ideal_cpu"] = int(32*row["AverageAllocatedMb_pct100"]/1536)
                    function_params_dict[function_hash]["memory_least_hint"] = int(32*row["AverageAllocatedMb_pct1"]/1536)
                    function_params_dict[function_hash]["cpu_least_hint"] = int(32*row["AverageAllocatedMb_pct1"]/1536)
                    function_params_dict[function_hash]["memory_cap_per_function"] = 32
                    function_params_dict[function_hash]["cpu_cap_per_function"] = 45
                    break

            for _, row in duration_traces.iterrows():
                if row["HashFunction"] == function_hash:
                    function_params_dict[function_hash]["ideal_duration"] = int(row["Average"]/1000) + 1 # Millisec to sec

            for _, row in invocation_traces.iterrows():
                if row["HashFunction"] == function_hash:
                    # Reference: 
                    # https://docs.microsoft.com/en-us/azure/azure-functions/functions-scale
                    if row["Trigger"] == "http":
                        function_params_dict[function_hash]["timeout"] = 230 
                    else:
                        function_params_dict[function_hash]["timeout"] = 300
        
        # Create Profile paramters
        function_params = []
        application_params = []

        for function_hash in function_params_dict.keys():
            function = FunctionParameters(
                ideal_cpu=function_params_dict[function_hash]["ideal_cpu"], 
                ideal_memory=function_params_dict[function_hash]["ideal_memory"],
                ideal_duration=function_params_dict[function_hash]["ideal_duration"],
                cpu_least_hint=function_params_dict[function_hash]["cpu_least_hint"],
                memory_least_hint=function_params_dict[function_hash]["memory_least_hint"],
                timeout=function_params_dict[function_hash]["timeout"],
                cpu_cap_per_function=function_params_dict[function_hash]["cpu_cap_per_function"],
                memory_cap_per_function=function_params_dict[function_hash]["memory_cap_per_function"],
                application_id=function_params_dict[function_hash]["HashApp"],
                function_id=function_hash
            )
            
            function_params.append(function)

        app_hash_list = memory_traces["HashApp"]
        for _ in app_hash_list:
            application_params.append([])

        for i in range(len(function_params)):
            for j in range(len(app_hash_list)):
                if function_params[i].application_id == app_hash_list[j]:
                    application_params[j].append(i)

        profile_params = [function_params, application_params]

        # Create timetable based on invocation traces
        timetable_params = TimetableParameters(
            max_timestep=1440, 
            distribution_type="azure",
            azure_invocation_traces=invocation_traces
        )

        return profile_params, timetable_params

    def generate_profile(self, profile_params):
        function_params = profile_params[0]
        application_params = profile_params[1]
        
        function_list = []
        
        for param in function_params:
            function = Function(param)
            # function.set_function(
            #     cpu=param.cpu_cap_per_function, 
            #     memory=param.memory_cap_per_function
            # ) # Initially over-provision

            function.set_function(
                cpu=param.cpu_least_hint, 
                memory=param.memory_least_hint
            ) # Initially set as hinted
            
            function_list.append(function)
        
        application_list = []
        if len(application_params) != 0:
            for group in application_params:
                functions = []
                for function_index in group:
                    functions.append(function_list[function_index])
                    
                application = Application(functions)
                application_list.append(application)
        
        profile = Profile(function_profile=function_list, 
                          application_profile=application_list
                          )
        return profile
    
    def mod_distribution(
        self, 
        profile,
        timetable_params
    ):
        max_timestep = timetable_params.max_timestep
        mod_factors = timetable_params.mod_factors
        
        function_list = profile.function_profile
        timetable_list = []
        
        for i in range(max_timestep):
            timestep = []
            
            for factor_i in range(len(mod_factors)):
                if i%mod_factors[factor_i] == 0:
                    timestep.append(function_list[factor_i].function_id)
                
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
        
        function_list = profile.function_profile
        timetable_list = []
        poisson_time_list = []
        
        for function in function_list:
            poisson_time_list.append(
                stats.poisson.rvs(
                    mu=mu, 
                    size=max_timestep
                )
            )
        
        for i in range(max_timestep):
            timestep = []
            
            for poisson_i in range(len(poisson_time_list)):
                for t in range(poisson_time_list[poisson_i][i]):
                    timestep.append(function_list[poisson_i].function_id)
                
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
        
        function_list = profile.function_profile
        timetable_list = []
        bernoulli_time_list = []
        
        for function in function_list:
            bernoulli_time_list.append(
                stats.bernoulli.rvs(
                    p=p, 
                    size=max_timestep
                )
            )
        
        for i in range(max_timestep):
            timestep = []
            
            for bernoulli_i in range(len(bernoulli_time_list)):
                for t in range(bernoulli_time_list[bernoulli_i][i]):
                    timestep.append(function_list[bernoulli_i].function_id)
                
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

        function_list = profile.function_profile
        timetable_list = []

        for i in range(max_timestep):
            timestep = []

            for _, row in invocation_traces.iterrows():
                for _ in range(row["{}".format(i+1)]):
                    timestep.append(row["HashFunction"])

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
    
    def generate_workflow(
        self, 
        default="azure",
        profile_params=None, 
        timetable_params=None
    ):
        if default == "ensure":
            default_profile_params, default_timetable_params = self.ensure_params()
        elif default == "azure":
            default_profile_params, default_timetable_params = self.azure_params()

        if profile_params is None:
            profile_params = default_profile_params
        profile = self.generate_profile(profile_params)
            
        if timetable_params is None:
            timetable_params = default_timetable_params
        timetable = self.generate_timetable(profile, timetable_params)
            
        return profile, timetable

