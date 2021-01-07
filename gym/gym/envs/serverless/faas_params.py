#
# Parameters
#

class EnvParameters():
    """
    Parameters used for generating FaaSEnv
    """
    def __init__(
        self,
        max_function=60,
        max_server=40,
        cluster_size=10,
        user_cpu_per_server=8,
        user_memory_per_server=8,
        keep_alive_window_per_server=60,
        cpu_cap_per_function=8,
        memory_cap_per_function=8,
        interval=1,
        timeout_penalty=60
    ):
        self.max_function = max_function
        self.max_server = max_server
        self.cluster_size = cluster_size
        self.user_cpu_per_server = user_cpu_per_server
        self.user_memory_per_server = user_memory_per_server
        self.keep_alive_window_per_server = keep_alive_window_per_server
        self.cpu_cap_per_function = cpu_cap_per_function
        self.memory_cap_per_function = memory_cap_per_function
        self.interval = interval
        self.timeout_penalty = timeout_penalty

        
class FunctionParameters():
    """
    Parameters used for generating Function
    """
    def __init__(
        self,
        ideal_cpu=8,
        ideal_memory=8,
        max_duration=1,
        min_duration=60,
        cpu_cap_per_function=8,
        memory_cap_per_function=8,
        cpu_least_hint=1,
        memory_least_hint=1,
        cpu_user_defined=4,
        memory_user_defined=4,
        timeout=60,
        application_id=None,
        function_id=None,
        hash_value=0,
        cold_start_time=1,
        k=2,
        sequence=None,
    ):
        self.ideal_cpu = ideal_cpu
        self.ideal_memory = ideal_memory
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.cpu_cap_per_function = cpu_cap_per_function
        self.memory_cap_per_function = memory_cap_per_function
        self.cpu_least_hint = cpu_least_hint
        self.memory_least_hint = memory_least_hint
        self.cpu_user_defined = cpu_user_defined
        self.memory_user_defined = memory_user_defined
        self.timeout = timeout
        self.application_id = application_id
        self.function_id = function_id
        self.hash_value = hash_value
        self.cold_start_time = cold_start_time
        self.k = k
        self.sequence = sequence
        

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

    
    
    
