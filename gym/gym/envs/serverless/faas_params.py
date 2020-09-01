import numpy as np


class EnvParameters():
    """
    Parameters used for generating FaaSEnv
    """
    def __init__(
        self,
        cpu_total=32*10,
        memory_total=45*10,
        cpu_cap_per_function=32,
        memory_cap_per_function=46
    ):
        self.cpu_total = cpu_total
        self.memory_total = memory_total
        self.cpu_cap_per_function = cpu_cap_per_function
        self.memory_cap_per_function = memory_cap_per_function

        
class FunctionParameters():
    """
    Parameters used for generating Function
    """
    def __init__(
        self,
        ideal_cpu=32,
        ideal_memory=45,
        ideal_duration=30,
        cpu_cap_per_function=32,
        memory_cap_per_function=45,
        cpu_least_hint=1,
        memory_least_hint=1,
        timeout=60,
        application_id=None,
        function_id=None
    ):
        self.ideal_cpu = ideal_cpu
        self.ideal_memory = ideal_memory
        self.ideal_duration = ideal_duration
        self.cpu_cap_per_function = cpu_cap_per_function
        self.memory_cap_per_function = memory_cap_per_function
        self.cpu_least_hint = cpu_least_hint
        self.memory_least_hint = memory_least_hint
        self.timeout = timeout
        self.application_id = application_id
        self.function_id = function_id

class TimetableParameters():
    """
    Parameters used for generating Timetable
    """
    def __init__(
        self,
        max_timestep=200,
        distribution_type="poisson",
        mod_factors=[1, 1, 1, 1, 1, 2, 5, 8, 10, 8],
        bernoulli_p=0.5,
        poisson_mu=0.8,
        azure_invocation_traces=None
    ):
        self.max_timestep = max_timestep
        self.distribution_type = distribution_type
        
        if distribution_type == "mod":
            self.mod_factors = mod_factors
        elif distribution_type == "bernoulli":
            self.bernoulli_p = bernoulli_p
        elif distribution_type == "poisson":
            self.poisson_mu = poisson_mu
        elif distribution_type == "azure":
            self.azure_invocation_traces = azure_invocation_traces

    
    
    