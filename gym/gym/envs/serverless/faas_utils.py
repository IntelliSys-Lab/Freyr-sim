import numpy as np
import copy as cp
import uuid



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
            return "timeout" # Timeout
        
        if self.progress >= self.duration:
            return "done" # Done
        else:
            return "undone" # Undone


class Request():
    """
    An invocation of a function
    """
    def __init__(self, function):
        self.profile = cp.deepcopy(function)
        self.request_id = uuid.uuid1()
        
    def step(self, in_registry):
        status = self.profile.step(in_registry)
        return status


class ResourcePattern():
    """
    A global view of the cluster resources at current timestep
    """
    
    def __init__(self, cluster_registry, cpu_total=0, memory_total=0):
        self.cluster_registry = cluster_registry
        self.cpu_total = cpu_total
        self.memory_total = memory_total
        
    def get_resources_total(self):
        return self.cpu_total, self.memory_total
    
    def get_resources_in_use(self):
        cpu_in_use, memory_in_use = self.cluster_registry.get_resources_in_use()
        return cpu_in_use, memory_in_use
    
    # Calculate available resources
    def get_resources_available(self):
        cpu_in_use, memory_in_use = self.cluster_registry.get_resources_in_use()
        cpu_available = self.cpu_total - cpu_in_use
        memory_available = self.memory_total - memory_in_use
        return cpu_available, memory_available
    
    # Check whether a given request is available
    def check_availablity(self, request):
        cpu_available, memory_available = self.get_resources_available()
        if cpu_available >= request.profile.cpu and memory_available >= request.profile.memory:
            return True
        else:
            return False
    
        
class Registry():
    """
    Where functions processed, used by FaaSEnv
    """
    
    def __init__(self, size=100):
        self.size = size
        self.in_registry = True
        self.registry = []
        
    def step(self):
        # Update registry
        request_done_or_timeout_list = []
        num_timeout = 0
        
        for request in self.registry:
            status = request.step(self.in_registry)
            # Remove finished functions if any
            if status == "done": # Done requests
                request_done_or_timeout_list.append(request)
            elif status == "timeout": # Timeout requests
                num_timeout = num_timeout + 1
                request_done_or_timeout_list.append(request)
                
        return request_done_or_timeout_list, num_timeout
                
    def reset(self):
        self.registry = []            
    
    def delete_requests(self, request_list):
        for request in request_list:
            self.registry.remove(request)
    
    def put_requests(self, request_list):
        if isinstance(request_list, Request):
            request = request_list
            request_chosen = cp.deepcopy(request)
            self.registry.append(request_chosen)
        else:
            for request in request_list:
                request_chosen = cp.deepcopy(request)
                self.registry.append(request_chosen)
                    
    def get_requests(self):
        return self.registry
                
    def get_size(self):
        return self.size
    
    def get_current_len(self):
        return len(self.registry)
    
    def get_resources_in_use(self):
        cpu_in_use, memory_in_use = 0, 0
        for request in self.registry:
            cpu_in_use = cpu_in_use + request.profile.cpu
            memory_in_use = memory_in_use + request.profile.memory
            
        return cpu_in_use, memory_in_use
    

class Queue():
    """
    Where functions waiting for entering registry, used by FaaSEnv
    """
    
    def __init__(self, size=114514):
        self.size = size
            
        self.in_registry = False
        self.queue = []
        
    def step(self):
        # Update queue
        request_timeout_list = []
        num_timeout = 0
        
        for request in self.queue:
            status = request.step(self.in_registry)
            # Remove finished functions if any
            if status == "timeout": # Timeout requests
                num_timeout = num_timeout + 1
                request_timeout_list.append(request)
                
        return request_timeout_list, num_timeout
    
    def reset(self):
        self.queue = []
    
    def delete_requests(self, request_list):
        for request in request_list:
            self.queue.remove(request)

    def put_requests(self, request_list):
        if isinstance(request_list, Request):
            request = request_list
            request_chosen = cp.deepcopy(request)
            self.queue.append(request_chosen)
        else:
            for request in request_list:
                request_chosen = cp.deepcopy(request)
                self.queue.append(request_chosen)
                    
    def get_requests(self):
        return self.queue
                
    def get_size(self):
        return self.size
                
    def get_current_len(self):
        return len(self.queue)
    
    def get_ready_quests(self, resource_pattern):
        request_ready_list = []
        for request in self.queue:
            if resource_pattern.check_availablity(request):
                if len(request_ready_list) == 0:
                    request_ready_list.append(request)
                else: # Sort by waiting time
                    is_inserted = False
                    for i in range(-1, -len(request_ready_list)-1, -1):
                        if request_ready_list[i].profile.waiting > request.profile.waiting:
                            request_ready_list.insert(i+1, request)
                            is_inserted = True
                            break
                    
                    if is_inserted is False:
                        request_ready_list.insert(0, request)
                        
        return request_ready_list
    
class Profile():
    """
    Record settings of any functions that submitted to FaaSEnv
    """
    
    def __init__(self, application_list, function_list):
        self.application_profile = application_list
        self.function_profile = function_list
        self.default_function_profile = function_list
        
    def put_application(self, application):
        self.application_profile.append(application)
        
    def put_function(self, function):
        self.function_profile.append(function)
        
    def reset(self):
        self.function_profile = cp.deepcopy(self.default_function_profile)
        
        
class Timetable():
    """
    Dictate which and when functions will be invoked by FaaSEnv
    """
    
    def __init__(self, timetable=[]):
        self.timetable = timetable
        self.size = len(self.timetable)
        
    def put_timestep(self, row):
        self.timetable.append(row)
        
    def get_timestep(self, timestep):
        if timestep >= len(self.timetable)-1:
            return None
        else:
            return self.timetable[timestep]
    
    def get_size(self):
        return self.size
    
    