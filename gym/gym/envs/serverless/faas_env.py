import gym
from gym import spaces, logger
import numpy as np
from gym.envs.serverless.faas_utils import Cluster, RequestRecord, Request
        
        
class FaaSEnv(gym.Env):
    """
    Function-as-a-Service environment.
    """
    
    def __init__(
        self,
        params,
        profile,
        timetable
    ):
        self.params = params
        self.profile = profile
        self.timetable = timetable
        
        self.cluster = Cluster(
            cluster_size=self.params.cluster_size,
            user_cpu_per_server=self.params.user_cpu_per_server,
            user_memory_per_server=self.params.user_memory_per_server,
            keep_alive_window_per_server=self.params.keep_alive_window_per_server
        )
        self.request_record = RequestRecord(self.profile.function_profile)
        
        self.system_time = 0
        
        # Define action space
        # Action space size: 4*m+1
        self.action_space = spaces.Discrete(4 * self.profile.get_function_profile_size() + 1)
        
        # Define observation space
        # Observation space size: 1 + 2 * n + 5 * m
        #
        # [
        #  num_in_fight_requests, 
        #  server_1_available_cpu, 
        #  server_1_available_memory,
        #  .
        #  .
        #  .
        #  server_n_available_cpu, 
        #  server_n_available_memory,
        #  function_1_cpu,
        #  function_1_memory,
        #  function_1_avg_interval,
        #  function_1_avg_completion_time,
        #  function_1_is_cold_start,
        #  .
        #  .
        #  .
        #  function_m_cpu,
        #  function_m_memory,
        #  function_m_avg_interval,
        #  function_m_avg_completion_time,
        #  function_m_is_cold_start
        # ]
        low = np.zeros(1 + 2 * self.cluster.get_cluster_size() + 5 * self.profile.get_function_profile_size(), dtype=np.float32)
        
        high_part_1 = np.array([10000])

        high_part_2 = []
        for server in self.cluster.get_server_pool():
            high_part_2.append(server.resource_manager.get_user_cpu())
            high_part_2.append(server.resource_manager.get_user_memory())
        high_part_2 = np.array(high_part_2)

        high_part_3 = []
        for function in self.profile.function_profile:
            high_part_3.append(self.params.cpu_cap_per_function)
            high_part_3.append(self.params.memory_cap_per_function)
            high_part_3.append(100)
            high_part_3.append(1000)
            high_part_3.append(1)
        high_part_3 = np.array(high_part_3)
        
        high = np.hstack((high_part_1, high_part_2, high_part_3))
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
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
            resource = 1 # Memory
            adjust = -1 # Decrease one slot
        elif action%4 == 3:
            resource = 1 # Memory
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
                # if self.profile.function_profile[function_index].validate_resource_adjust(resource, adjust) is True:
                #     self.profile.function_profile[function_index].set_resource_adjust(resource, adjust)
                self.profile.function_profile[function_index].set_resource_adjust(resource, adjust)
            
            return False
        
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
        # Try to import timetable if not finished
        timestep = self.timetable.get_timestep(self.system_time - 1)
        request_to_schedule_list = []

        if timestep is not None:
            for function_id in timestep:
                for function in self.profile.function_profile:
                    if function_id == function.function_id:
                        request = Request(
                            function=function, 
                            invoke_time=self.system_time,
                        )
                        request_to_schedule_list.append(request)
                        break

        self.request_record.put_requests(request_to_schedule_list)
        self.cluster.schedule(self.system_time, request_to_schedule_list)
        request_to_remove_list, num_timeout = self.cluster.step(self.system_time)
        self.request_record.update_request(request_to_remove_list)

        return num_timeout

    def get_info_dict(self):
        info_dict = {}
        for function in self.profile.function_profile:
            info_dict[function.function_id] = {}
            info_dict[function.function_id]["avg_completion_time"] = function.get_avg_completion_time()

        return info_dict
    
    #
    # Get observation for next timestep
    #
    def get_observation(self):
        num_undone_requests = self.request_record.get_undone_size()
        observation_part_1 = np.array([num_undone_requests])

        observation_part_2 = []
        for server in self.cluster.get_server_pool():
            available_cpu, available_memory = server.resource_manager.get_resources_available()
            observation_part_2.append(available_cpu)
            observation_part_2.append(available_memory)
        observation_part_2 = np.array(observation_part_2)

        observation_part_3 = []
        for function in self.profile.function_profile:
            function_id = function.get_function_id()

            observation_part_3.append(function.cpu)
            observation_part_3.append(function.memory)
            observation_part_3.append(
                self.request_record.get_avg_interval_per_function(self.system_time, function_id)
            )
            observation_part_3.append(
                self.request_record.get_avg_completion_time_per_function(function.function_id)
            )
            observation_part_3.append(
                self.request_record.get_is_cold_start_per_function(function_id)
            )
        observation_part_3 = np.array(observation_part_3)
        
        observation = np.hstack((observation_part_1, observation_part_2, observation_part_3))
        
        return observation
    #
    # Calculate reward for current timestep
    #
    def get_reward(self, num_timeout):
        # Timeout penalty
        reward = -num_timeout * self.params.timeout_penalty 

        # Reward of completion time 
        for request in self.request_record.get_success_request_record():
            if request.get_done_time() == self.system_time:
                reward = reward + -request.get_completion_time()
        for request in self.request_record.get_timeout_request_record():
            if request.get_done_time() == self.system_time:
                reward = reward + -request.get_completion_time()
            
        return reward
    
    #
    # Get done for current timestep
    #
    def get_done(self):
        done = False
        if self.system_time >= self.timetable.get_size() and self.request_record.get_undone_size() == 0:
            done = True
            
        return done
    
    #
    # Get info for current timestep
    #
    def get_info(self):
        info = {
            "system_time": self.system_time,
            "avg_completion_time": self.request_record.get_avg_completion_time(),
            "timeout_num": self.request_record.get_timeout_size(),
            "request_record": self.request_record,
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
            self.system_time = self.system_time + self.params.interval
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
        self.cluster.reset()
        self.request_record.reset()
        
        observation = self.get_observation()
        
        return observation
    
    
    
