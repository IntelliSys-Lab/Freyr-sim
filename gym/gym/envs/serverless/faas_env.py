import gym
from gym import spaces, logger
import numpy as np
from gym.envs.serverless.faas_utils import Registry, Queue, ResourcePattern, Request, RequestRecord
        
        
class FaaSEnv(gym.Env):
    """
    Function-as-a-Service environment.
    """
    
    def __init__(self,
                 params,
                 profile,
                 timetable
                 ):
        self.params = params
        self.profile = profile
        self.timetable = timetable
        
        self.registry = Registry()
        self.queue = Queue()
        self.resource_pattern = ResourcePattern(
            cpu_total=self.params.cpu_total, 
            memory_total=self.params.memory_total, 
            cluster_registry=self.registry
            )
        self.request_record = RequestRecord(profile.function_profile)
        
        self.system_time = 0
        
        # Define action space
        self.action_space = spaces.Discrete(len(self.profile.function_profile)*4+1)
        
        # Define observation space
        cpu_total, memory_total = self.resource_pattern.get_resources_total()
        registry_size = self.registry.get_size()
        queue_size = self.queue.get_size()
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), # [available_cpu, available_memory, registry_current_len, queue_current_len]
            high=np.array([cpu_total, memory_total, registry_size, queue_size]),
            dtype=np.int32
            )
    
    #
    # Decode discrete action into resource change
    #
    def decode_action(self, action):
        function_index = int(action/4)
        resource = None
        adjust = 0
        
        if action%4 == 0:
            resource = 0 # CPU
            adjust = -1 # Decrease one slot
        elif action%4 == 1:
            resource = 0 # CPU
            adjust = 1 # Increase one slot
        elif action%4 == 2:
            resource = 0 # Memory
            adjust = -1 # Decrease one slot
        elif action%4 == 3:
            resource = 0 # Memory
            adjust = 1 # Increase one slot
        
        return function_index, resource, adjust
        
    #
    # Update settings of function profile based on given action
    #
    def update_function_profile(self, action):
        if isinstance(action, list): # WARNING! Only used by greedy provision!
            actions = action
            for act in actions:
                function_index, resource, adjust = self.decode_action(act)
                if self.profile.function_profile[function_index].validate_resource_adjust(resource, adjust) is True:
                    self.profile.function_profile[function_index].set_resource_adjust(resource, adjust)
            
            return True
        
        if action == self.action_space.n - 1: # Explicit invalid action
            return False
        else:
            function_index, resource, adjust = self.decode_action(action)
            if self.profile.function_profile[function_index].validate_resource_adjust(resource, adjust) is True:
                self.profile.function_profile[function_index].set_resource_adjust(resource, adjust)
                return True
            else:
                return False # Implicit invalid action

    #            
    # Update the cluster
    #
    def update_cluster(self):
        # 1. Update registry
        request_done_or_timeout_list, num_timeout_registry = self.registry.step()
        self.request_record.record(request_done_or_timeout_list)
        self.registry.delete_requests(request_done_or_timeout_list)
        
        # 2. Update queue
        request_timeout_list, num_timeout_queue = self.queue.step()
        self.request_record.record(request_timeout_list)
        self.queue.delete_requests(request_timeout_list)
        
        # 3. Try to import queue if available, copy chosen requests to registry and remove them from queue    
        request_ready_list = self.queue.get_ready_quests(self.resource_pattern)
        for request in request_ready_list:
            if self.resource_pattern.check_availablity(request):
                self.registry.put_requests(request)
                
        self.queue.delete_requests(request_ready_list)
                
        # 4. Try to import timetable if not finished
        # Send requests to registry if queue is empty
        # Otherwise send them to queue
        timestep = self.timetable.get_timestep(self.system_time)
        if timestep is not None:
            for function_id in timestep:
                for function in self.profile.function_profile:
                    if function_id == function.function_id:
                        request = Request(function)
                        if self.resource_pattern.check_availablity(request) and self.queue.get_current_len() == 0:
                            self.registry.put_requests(request)
                        else:
                            self.queue.put_requests(request)
        
        return num_timeout_registry + num_timeout_queue
    
    #
    # Get observation for next timestep
    #
    def get_observation(self):
        cpu_available, memory_available = self.resource_pattern.get_resources_available()
        registry_current_len = self.registry.get_current_len()
        queue_current_len = self.queue.get_current_len()
        
        observation = np.array([cpu_available, memory_available, registry_current_len, queue_current_len])
        
        return observation
    #
    # Calculate reward for current timestep
    #
    def get_reward(self, num_timeout):
        reward = -num_timeout*100
        for request in self.registry.get_requests():
            reward = reward + -1/request.profile.duration
        for request in self.queue.get_requests():
            reward = reward + -1/request.profile.duration
            
        return reward
    
    #
    # Get done for current timestep
    #
    def get_done(self):
        done = False
        registry_current_len = self.registry.get_current_len()
        queue_current_len = self.queue.get_current_len()
        if self.system_time >= self.timetable.get_size() and registry_current_len+queue_current_len == 0:
            done = True
            
        return done
    
    #
    # Get info for current timestep
    #
    def get_info(self):
        info = {
            "system_time": self.system_time,
            "avg_slow_down": self.request_record.get_avg_slow_down(),
            "avg_completion_time": self.request_record.get_avg_completion_time(),
            "timeout_num": self.request_record.get_timeout_num(),
            "request_record": self.request_record
            }
        
        return info
        
    """
    Override
    """
    def render(self):
        logger.warn("To do")
        pass
    
    def close(self):
        logger.warn("To do")
        pass
    
    def step(self, action=None):
        is_valid_action = self.update_function_profile(action)
        
        if is_valid_action is True:
            reward = 0
        else:
            self.system_time = self.system_time + 1
            num_timeout = self.update_cluster()
            reward = self.get_reward(num_timeout)
            
            # Reset resource adjust direction for each function 
            for function in self.profile.function_profile:
                function.reset_resource_adjust_direction()
            
        # Get observation for next state
        observation = self.get_observation()
        
        # Done?
        done = self.get_done()
        
        # Return system time
        info = self.get_info()
        
        return observation, reward, done, info
    
    def reset(self):
        self.system_time = 0
        
        self.profile.reset()
        self.registry.reset()
        self.queue.reset()
        self.request_record.reset()
        
        cpu_available, memory_available = self.resource_pattern.get_resources_available()
        registry_current_len = self.registry.get_current_len()
        queue_current_len = self.queue.get_current_len()
        
        observation = np.array([cpu_available, memory_available, registry_current_len, queue_current_len])
        
        return observation
    
    
    