#
# Parameters
#

class EnvParameters():
    """
    Parameters used for generating FaaSEnv
    """
    def __init__(
        self,
        cluster_size,
        user_cpu_per_server,
        user_memory_per_server,
        keep_alive_window,
        cpu_cap_per_function,
        memory_cap_per_function,
        memory_mb_limit,
        interval,
        fail_penalty
    ):
        self.cluster_size = cluster_size
        self.user_cpu_per_server = user_cpu_per_server
        self.user_memory_per_server = user_memory_per_server
        self.keep_alive_window = keep_alive_window
        self.cpu_cap_per_function = cpu_cap_per_function
        self.memory_cap_per_function = memory_cap_per_function
        self.memory_mb_limit = memory_mb_limit
        self.interval = interval
        self.fail_penalty = fail_penalty
        
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
        memory_mb_limit,
        cpu_least_hint,
        memory_least_hint,
        cpu_user_defined,
        memory_user_defined,
        timeout,
        hash_value,
        cold_start_time,
        k,
        function_id
    ):
        self.ideal_cpu = ideal_cpu
        self.ideal_memory = ideal_memory
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.cpu_cap_per_function = cpu_cap_per_function
        self.memory_cap_per_function = memory_cap_per_function
        self.memory_mb_limit = memory_mb_limit
        self.cpu_least_hint = cpu_least_hint
        self.memory_least_hint = memory_least_hint
        self.cpu_user_defined = cpu_user_defined
        self.memory_user_defined = memory_user_defined
        self.timeout = timeout
        self.function_id = function_id
        self.hash_value = hash_value
        self.cold_start_time = cold_start_time
        self.k = k


class WorkloadParameters():
    """
    Parameters used for workload configuration
    """
    def __init__(
        self,
        azure_file_path,
        exp_id
    ):
        self.azure_file_path = azure_file_path
        self.exp_id = exp_id

        
class EventPQParameters():
    """
    Parameters used for generating EventPQ
    """
    def __init__(
        self,
        azure_invocation_traces=None
    ):
        self.azure_invocation_traces = azure_invocation_traces
        