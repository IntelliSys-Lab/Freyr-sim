import numpy as np


class EnvParameters():
    """
    Parameters used by FaaSEnv
    """
    def __init__(self,
                 cpu_total=32*10,
                 memory_total=46*10,
                 cpu_cap_per_function=32,
                 memory_cap_per_function=46
                 ):
        self.cpu_total = cpu_total
        self.memory_total = memory_total
        self.cpu_cap_per_function = cpu_cap_per_function
        self.memory_cap_per_function = memory_cap_per_function
        
        
class FunctionParameters():
    """
    Parameters used by Function
    """
    def __init__(self,
                 ideal_cpu=32,
                 ideal_memory=46,
                 ideal_duration=50,
                 cpu_cap_per_function=32,
                 memory_cap_per_function=46,
                 cpu_least_hint=1,
                 memory_least_hint=1,
                 timeout=60
                 ):
        self.ideal_cpu = ideal_cpu
        self.ideal_memory = ideal_memory
        self.ideal_duration = ideal_duration
        self.cpu_cap_per_function = cpu_cap_per_function
        self.memory_cap_per_function = memory_cap_per_function
        self.cpu_least_hint = cpu_least_hint
        self.memory_least_hint = memory_least_hint
        self.timeout = timeout
        