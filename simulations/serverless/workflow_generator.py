import sys
sys.path.append("../../gym")
import gym
from gym.envs.serverless.faas_utils import Function, Application, Profile, Timetable
from gym.envs.serverless.faas_params import FunctionParameters


class WorkflowGenerator():
    """
    Generate workflows running on FaaSEnv
    """
    
    def __init__(self, 
                 profile_params=None,
                 timetable_params=None,
                 ):
        self.profile_params = profile_params
        self.timetable_params = timetable_params
    
    def default_params(self):
        # Generate profile parameters
        function_params_1 = FunctionParameters(
            ideal_cpu=4, 
            ideal_memory=4, 
            ideal_duration=1, 
            cpu_least_hint=1, 
            memory_least_hint=1,
            timeout=60,
            cpu_cap_per_function=32,
            memory_cap_per_function=46
            )
        function_params_2 = FunctionParameters(
            ideal_cpu=16, 
            ideal_memory=16, 
            ideal_duration=5, 
            cpu_least_hint=8, 
            memory_least_hint=8,
            timeout=60,
            cpu_cap_per_function=32,
            memory_cap_per_function=46,
            )
        function_params_3 = FunctionParameters(
            ideal_cpu=24, 
            ideal_memory=32, 
            ideal_duration=30, 
            cpu_least_hint=12, 
            memory_least_hint=16,
            timeout=60,
            cpu_cap_per_function=32,
            memory_cap_per_function=46,
            ) 
        function_params_4 = FunctionParameters(
            ideal_cpu=32, 
            ideal_memory=46, 
            ideal_duration=50, 
            cpu_least_hint=16, 
            memory_least_hint=23,
            timeout=60,
            cpu_cap_per_function=32,
            memory_cap_per_function=46,
            )
        
        function_params = [
            function_params_1, 
            function_params_2, 
            function_params_3, 
            function_params_4
            ]
        application_params = [[0, 1], [2, 3]]
        
        profile_params = [function_params, application_params]
        
        # Generate timetable parameters
        # TODO
        timetable_params = []
            
        return profile_params, timetable_params
    
    def generate_profile(self):
        function_params = self.profile_params[0]
        application_params = self.profile_params[1]
        
        function_list = []
        
        for param in function_params:
            function = Function(param)
            function.set_function(
                cpu=param.cpu_cap_per_function, 
                memory=param.memory_cap_per_function
                ) # Initially over-provision
            
            function_list.append(function)
        
        application_list = []
        for group in application_params:
            functions = []
            for function_index in group:
                functions.append(function_list[function_index])
                
            application = Application(functions)
            application_list.append(application)
            
        profile = Profile(application_list, function_list)
        return profile
    
    # TODO
    def generate_timetable(self, profile):
        funtion_list = profile.function_profile
        
        timetable_list = []
        timesteps = 120
        
        for i in range(timesteps):
            time = []
            
            if i%1 == 0:
                time.append(funtion_list[0].function_id)
            if i%2 == 0:
                time.append(funtion_list[1].function_id)
            if i%8 == 0:
                time.append(funtion_list[2].function_id)
            if i%10 == 0:
                time.append(funtion_list[3].function_id)
                
            timetable_list.append(time)
            
        timetable = Timetable(timetable_list)
        
        return timetable
    
    def generate_workflow(self):
        default_profile_params, default_timetable_params = self.default_params()
        
        if self.profile_params is None:
            self.profile_params = default_profile_params
        profile = self.generate_profile()
            
        if self.timetable_params is None:
            self.timetable_params = default_timetable_params
        timetable = self.generate_timetable(profile)
            
        return profile, timetable

