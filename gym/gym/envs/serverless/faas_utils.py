import numpy as np
import uuid


class Function():
    """
    Function used by FaaSEnv
    """
    
    def __init__(self, 
                 ideal_cpu=2,
                 ideal_memory=46,
                 ideal_duration=50,
                 cpu_least_hint=1,
                 memory_least_hint=1,
                 timeout=60
                 ):
        self.function_id = uuid.uuid1()
        
        self.ideal_cpu = ideal_cpu
        self.ideal_memory = ideal_memory
        self.ideal_duration = ideal_duration
        self.cpu_least_hint = cpu_least_hint
        self.memory_least_hint = memory_least_hint
        self.timeout = timeout
        
        self.progress = 0
        self.waiting = 0
    
    def set_application_id(self, application_id):
        self.application_id = application_id
        
    def set_function(self, cpu=2, memory=46):
        self.cpu = cpu
        self.memory = memory
        
        # Calculate duration
        self.duration = self.ideal_duration * np.max([self.ideal_cpu, self.cpu])/self.cpu * np.max([self.ideal_memory, self.memory])/self.memory
            
    def step(self, in_registry):
        if in_registry is True:
            self.progress = self.progress + 1
        else:
            self.waiting = self.waiting + 1
        
        # Return status
        if self.progress + self.waiting >= self.timeout:
            return 2 # Timeout
        
        if self.progress >= self.duration:
            return 1 # Done
        else:
            return 0 # Undone


class Application():
    """
    Application used by FaaSEnv
    """
    def __init__(self, functions):
        self.application_id = uuid.uuid1()
        self.function_ids = []
        
        for function in functions:
            self.function_ids.append(function.function_id)
            function.set_application_id(self.application_id)
    