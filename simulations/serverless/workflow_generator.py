import sys
sys.path.append("../../gym")
import scipy.stats as stats
import gym
from gym.envs.serverless.faas_utils import Function, Application, Profile, Timetable
from gym.envs.serverless.faas_params import FunctionParameters, TimetableParameters


class WorkflowGenerator():
    """
    Generate workflows running on FaaSEnv
    """
    def default_params(self):
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
    
    def generate_profile(self, profile_params):
        function_params = profile_params[0]
        application_params = profile_params[1]
        
        function_list = []
        
        for param in function_params:
            function = Function(param)
#             function.set_function(
#                 cpu=param.cpu_cap_per_function, 
#                 memory=param.memory_cap_per_function
#                 ) # Initially over-provision

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
    
    def mod_distribution(self, 
                         profile,
                         timetable_params
                         ):
        max_timestep = timetable_params.max_timestep
        mod_factors = timetable_params.mod_factors
        
        funtion_list = profile.function_profile
        timetable_list = []
        
        for i in range(max_timestep):
            timestep = []
            
            for factor_i in range(len(mod_factors)):
                if i%mod_factors[factor_i] == 0:
                    timestep.append(funtion_list[factor_i].function_id)
                
            timetable_list.append(timestep)
            
        timetable = Timetable(timetable_list)
        return timetable
    
    def poisson_distribution(self,
                            profile,
                            timetable_params
                            ):
        max_timestep = timetable_params.max_timestep
        mu = timetable_params.poisson_mu
        
        funtion_list = profile.function_profile
        timetable_list = []
        poisson_time_list = []
        
        for function in profile.function_profile:
            poisson_time_list.append(stats.poisson.rvs(mu=mu, size=max_timestep))
        
        for i in range(max_timestep):
            timestep = []
            
            for poisson_i in range(len(poisson_time_list)):
                for t in range(poisson_time_list[poisson_i][i]):
                    timestep.append(funtion_list[poisson_i].function_id)
                
            timetable_list.append(timestep)
            
        timetable = Timetable(timetable_list)
        return timetable
    
    def bernoulli_distribution(self,
                               profile,
                               timetable_params
                               ):
        max_timestep = timetable_params.max_timestep
        p = timetable_params.bernoulli_p
        
        funtion_list = profile.function_profile
        timetable_list = []
        bernoulli_time_list = []
        
        for function in profile.function_profile:
            bernoulli_time_list.append(stats.bernoulli.rvs(p=p, size=max_timestep))
        
        for i in range(max_timestep):
            timestep = []
            
            for bernoulli_i in range(len(bernoulli_time_list)):
                for t in range(bernoulli_time_list[bernoulli_i][i]):
                    timestep.append(funtion_list[bernoulli_i].function_id)
                
            timetable_list.append(timestep)
            
        timetable = Timetable(timetable_list)
        return timetable
        
    def generate_timetable(self, 
                           profile,
                           timetable_params
                           ):
        if timetable_params.distribution_type == "mod":
            timetable = self.mod_distribution(profile, timetable_params)
        elif timetable_params.distribution_type == "bernoulli":
            timetable = self.bernoulli_distribution(profile, timetable_params)
        elif timetable_params.distribution_type == "poisson":
            timetable = self.poisson_distribution(profile, timetable_params)
        
        return timetable
    
    def generate_workflow(self, 
                          profile_params=None, 
                          timetable_params=None
                          ):
        default_profile_params, default_timetable_params = self.default_params()
        
        if profile_params is None:
            profile_params = default_profile_params
        profile = self.generate_profile(profile_params)
            
        if timetable_params is None:
            timetable_params = default_timetable_params
        timetable = self.generate_timetable(profile, timetable_params)
            
        return profile, timetable

