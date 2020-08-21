import gym
from gym import spaces, logger
import numpy as np
import copy as cp


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
                 timeout=60):
        
        self.ideal_cpu = ideal_cpu
        self.ideal_memory = ideal_memory
        self.ideal_duration = ideal_duration
        self.cpu_least_hint = cpu_least_hint
        self.memory_least_hint = memory_least_hint
        self.timeout = timeout
        
        self.progress = 0
        self.waiting = 0
        
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
        
        
        
class FaaSEnv(gym.Env):
    """
    Function-as-a-Service environment.
    """
    
    def __init__(self,
                 repo,
                 timetable,
                 total_cpu=2*10,
                 total_memory=46*10
                 ):
        self.default_repo = repo
        self.timetable = timetable
        self.total_cpu = total_cpu
        self.total_memory = total_memory
        
        self.system_time = 0
        self.registry = []
        self.queue = []
        
        # Define action space
        action_space = []
        for function in self.default_repo:
            action_space.append(
                spaces.MultiDiscrete([2-function.cpu_least_hint+1, 46-function.memory_least_hint+1])
                )
            
        self.action_space = spaces.Tuple(action_space)
        
        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "available_resources": spaces.Box(low=np.array([0, 0]), high=np.array([2*10, 46*10]), dtype=np.int32),
                "undone_requests": spaces.Box(low=np.array([0,0]), high=np.array([100, 100]), dtype=np.int32),
            })
    
    # Update settings of functions
    def update_functions(self, action):
        for i in range(len(action)):
            next_cpu = self.repo[i].cpu_least_hint + action[i][0]
            next_memory = self.repo[i].memory_least_hint + action[i][1]
                    
            self.repo[i].set_function(cpu=next_cpu, memory=next_memory)
    
    # Update the cluster
    def update_cluster(self):
        num_timeout = 0
        
        # 1. Update registry
        request_done_or_timeout_list = []
        for request in self.registry:
            status = request.step(in_registry=True)
            # Remove finished functions if any
            if status == 1: # Done requests
                request_done_or_timeout_list.append(request)
            elif status == 2: # Timeout requests
                num_timeout = num_timeout + 1
                request_done_or_timeout_list.append(request)
        
        self.remove_requests(self.registry, request_done_or_timeout_list)
        
        # 2. Update queue
        request_timeout_list = []
        for request in self.queue:
            status = request.step(in_registry=False)
            # Remove finished functions if any
            if status == 2: # Timeout requests
                num_timeout = num_timeout + 1
                request_timeout_list.append(request)
        
        self.remove_requests(self.queue, request_timeout_list)
        
        # 3. Try to import queue if available
        request_ready_list = []
        for request in self.queue:
            if self.check_available(request):
                if len(request_ready_list) == 0:
                    request_ready_list.append(request)
                else: # Sort by waiting time
                    is_inserted = False
                    for i in range(-1, -len(request_ready_list)-1, -1):
                        if request_ready_list[i].waiting > request.waiting:
                            request_ready_list.insert(i+1, request)
                            is_inserted = True
                            break
                    
                    if is_inserted is False:
                        request_ready_list.insert(0, request)
                        
        # Copy chosen requests to registry and remove them from queue                   
        for request in request_ready_list:
            if self.check_available(request):
                request_chosen = cp.deepcopy(request)
                self.registry.append(request_chosen)
        
        self.remove_requests(self.queue, request_ready_list)
                
        # 4. Try to import timetable if not finished
        # Send requests to registry if queue is empty
        # Otherwise send them to queue
        if self.system_time < len(self.timetable):
            time = self.timetable[self.system_time]
            for i in range(len(time)):
                if time[i] == 1:
                    request = cp.deepcopy(self.repo[i]) 
                    if self.check_available(request) and len(self.queue) == 0:
                        self.registry.append(request)
                    else:
                        self.queue.append(request)
        
        return num_timeout
    
    """
    Utilities
    """
    # Calculate available resources
    def get_available(self):
        cpu_in_use, memory_in_use = 0, 0
        for request in self.registry:
            cpu_in_use = cpu_in_use + request.cpu
            memory_in_use = memory_in_use + request.memory
            
        available_cpu = self.total_cpu - cpu_in_use
        available_memory = self.total_memory - memory_in_use
        return available_cpu, available_memory
    
    # Check whether a given request is valid
    def check_available(self, request):
        available_cpu, available_memory = self.get_available()
        if available_cpu >= request.cpu and available_memory >= request.memory:
            return True
        else:
            return False
    
    def remove_requests(self, institution, list):
        for request in list:
            institution.remove(request)
        
    """
    Override
    """
    def render(self):
        logger.warn("Not implemented")
        pass
    
    def close(self):
        logger.warn("Not implemented")
        pass
    
    def step(self, action):
        self.system_time = self.system_time + 1
        self.update_functions(action)
        num_timeout = self.update_cluster()
        
        # Get observation for next state
        available_cpu, available_memory = self.get_available()
        observation = {
            "available_resources": [available_cpu, available_memory],
            "undone_requests": [len(self.registry), len(self.queue)],
        }
        
        # Calculate reward
        reward = -num_timeout*100
        for request in self.registry:
            reward = reward + -1/request.duration
        for request in self.queue:
            reward = reward + -1/request.duration
        
        # Done?
        if self.system_time >= len(self.timetable) and len(self.registry)+len(self.queue) == 0:
            done = True
        else:
            done = False
        
        # Information is null
        info = {}
        
        return observation, reward, done, info
    
    def reset(self):
        self.system_time = 0
        self.repo = cp.deepcopy(self.default_repo)
        
        self.registry = []
        self.queue = []
        
        available_cpu, available_memory = self.get_available()
        observation = {
            "available_resources": [available_cpu, available_memory],
            "undone_requests": [len(self.registry), len(self.queue)],
        }
        
        return observation
    
    
    