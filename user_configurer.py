import numpy as np

class UserConfigurer():
    def get_configuration(invocation): # Type of invocation is a Request
        mem_mu = invocation.get_memory_peak() #max mem usage involved
        cpu_mu = invocation.get_cpu_peak()
        mem = np.random.normal(mem_mu, 128, 1)
        cpu =  np.random.normal(cpu_mu, 1, 1)
        return (mem, cpu)