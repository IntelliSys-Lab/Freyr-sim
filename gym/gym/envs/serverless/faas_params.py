#
# Parameters
#

class EnvParameters():
    """
    Parameters used for generating FaaSEnv
    """
    def __init__(
        self,
        max_function,
        max_server,
        cluster_size,
        user_cpu_per_server,
        user_memory_per_server,
        keep_alive_window_per_server,
        cpu_cap_per_function,
        memory_cap_per_function,
        interval,
        timeout_penalty
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
        ideal_cpu,
        ideal_memory,
        max_duration,
        min_duration,
        cpu_cap_per_function,
        memory_cap_per_function,
        cpu_least_hint,
        memory_least_hint,
        cpu_user_defined,
        memory_user_defined,
        timeout,
        hash_value,
        cold_start_time,
        k,
        application_id=None,
        function_id=None,
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
        max_timestep,
        distribution_type,
        mod_factors=None,
        bernoulli_p=None,
        poisson_mu=None,
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
            