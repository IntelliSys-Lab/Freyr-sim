import sys
import gym
from gym import spaces, logger
import numpy as np
import torch
from gym.envs.serverless.faas_utils import Request, Cluster, RequestRecord, ResourceUtilsRecord
        
        
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

        # Set up system time 
        self.system_time = 0

        # Set up request record
        self.request_record = RequestRecord(self.profile.get_function_profile())

        # Set up resource utils record
        self.resource_utils_record = ResourceUtilsRecord(self.cluster.get_cluster_size())
        
        # Define action space
        # Action space size: 4 * f + 1
        #
        # [
        #  void, 
        #  function_1_decrease_cpu, 
        #  function_1_increase_cpu,
        #  function_1_decrease_memory,
        #  function_1_increase_memory,
        #  .
        #  .
        #  .
        #  function_f_decrease_cpu, 
        #  function_f_increase_cpu,
        #  function_f_decrease_memory,
        #  function_f_increase_memory,
        # ]
        self.action_dim = 4 * self.params.max_function + 1
        
        # Define observation space
        # Observation space size: 1 + 2 * n + 8 * m
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
        #  function_1_avg_interval,
        #  function_1_avg_completion_time,
        #  function_1_is_cold_start,
        #  function_1_cpu,
        #  function_1_memory,
        #  function_1_cpu_direction,
        #  function_1_memory_direction,
        #  function_1_invoke_num,
        #  .
        #  .
        #  .
        #  function_m_avg_interval,
        #  function_m_avg_completion_time,
        #  function_m_is_cold_start
        #  function_m_cpu,
        #  function_m_memory,
        #  function_m_cpu_direction,
        #  function_m_memory_direction,
        #  function_m_invoke_num,
        # ]
        self.observation_dim = 1 + 2 * self.params.max_server + 8 * self.params.max_function
    
    #
    # Decode discrete action into resource change
    #
    def decode_action(self, action):
        next_timestep = self.timetable.get_timestep(self.system_time)
        function_invoke_list = []
        for function_id in next_timestep.keys():
            invoke_num = next_timestep[function_id]
            if invoke_num > 0:
                function_invoke_list.append(function_id)

        function_index = int(action/4)
        if function_index >= len(function_invoke_list): # Implicit invalid action
            function_id = None
            resource = None
            adjust = None
            print("Exceed function invoke list!")
            print("action: {}".format(action))
            print("function_invoke_list len: {}, function_index: {}".format(len(function_invoke_list), function_index))
            sys.exit()
        else:
            function_id = function_invoke_list[function_index]
            adjust = 0
            
            if action % 4 == 0:
                resource = "cpu" # CPU
                adjust = -1 # Decrease one slot
            elif action % 4 == 1:
                resource = "cpu" # CPU
                adjust = 1 # Increase one slot
            elif action % 4 == 2:
                resource = "memory" # Memory
                adjust = -1 # Decrease one slot
            elif action % 4 == 3:
                resource = "memory" # Memory
                adjust = 1 # Increase one slot
        
        return function_id, resource, adjust
        
    #
    # Update settings of function profile based on given action
    #
    def update_function_profile(self, action):
        function_profile = self.profile.get_function_profile()

        # Greedy RM inputs an action map
        if isinstance(action, dict): 
            action_map = action 
            for function_id in action_map.keys():
                new_cpu = action_map[function_id]["cpu"]
                new_memory = action_map[function_id]["memory"]
                function = function_profile[function_id]
                function.set_function(new_cpu, new_memory)

                # Set the sequence members as well if it is a function sequence
                if function.get_sequence() is not None:
                    sequence = function.get_sequence()
                    for member_id in sequence:
                        function_profile[member_id].set_function(new_cpu, new_memory)
            
            # Always invalid actions
            return False
        else:
            if action == self.action_dim - 1: # Explicit invalid action
                return False
            else:
                function_id, resource, adjust = self.decode_action(action)

                if function_id is None:
                    return False # Implicit invalid action

                next_timestep = self.timetable.get_timestep(self.system_time)
                if next_timestep[function_id] == 0:
                    return False # Implicit invalid action

                function = function_profile[function_id]
                if function.validate_resource_adjust(resource, adjust) is True:
                    function.set_resource_adjust(resource, adjust)
                    return True
                else:
                    return False # Implicit invalid action

    #            
    # Update the cluster
    #
    def update_cluster(self):
        function_profile = self.profile.get_function_profile()

        # Try to import timetable if not finished
        timestep = self.timetable.get_timestep(self.system_time - 1)
        request_to_schedule_list = []

        if timestep is not None:
            for function_id in timestep.keys():
                function = function_profile[function_id]
                invoke_num = timestep[function_id]
                for _ in range(invoke_num):
                    request = Request(
                        function=function, 
                        invoke_time=self.system_time,
                    )
                    request_to_schedule_list.append(request)

        self.request_record.put_requests(request_to_schedule_list)
        self.cluster.schedule(self.system_time, request_to_schedule_list)
        request_to_remove_list, num_timeout = self.cluster.step(self.system_time)
        self.request_record.update_requests(request_to_remove_list)

        return num_timeout

    #
    # Update resource utilization record
    #
    def update_resource_utils(self):
        for index, server in enumerate(self.cluster.get_server_pool()):
            server_id = "server{}".format(index)
            cpu_util, memory_util = server.resource_manager.get_resource_utils()
            self.resource_utils_record.put_resource_utils(server_id, cpu_util, memory_util)

    def get_resource_utils_record(self):
        self.resource_utils_record.calculate_avg_resource_utils()
        return self.resource_utils_record.get_resource_utils_record()

    def get_function_throughput(self):
        throughput = self.request_record.get_success_size() + self.request_record.get_timeout_size()
        return throughput

    def get_function_dict(self):
        next_timestep = self.timetable.get_timestep(self.system_time)
        function_profile = self.profile.get_function_profile()
        function_dict = {}
        for function_id in function_profile.keys():
            function = function_profile[function_id]
            function_dict[function_id] = {}

            avg_completion_time = self.request_record.get_avg_completion_time_per_function(function_id)
            avg_interval = self.request_record.get_avg_interval_per_function(function_id)
            cpu = function.get_cpu()
            memory = function.get_memory()
            total_sequence_size = function.get_total_sequence_size()

            is_success = False
            i = 1
            while i <= self.request_record.get_total_size_per_function(function_id):
                request = self.request_record.get_total_request_record_per_function(function_id)[-i]
                if request.get_status() == "success":
                    is_success = True
                    break

                i = i + 1

            if next_timestep is not None:
                invoke_num = next_timestep[function_id]
            else:
                invoke_num = 0

            # Prioritize failed functions
            if is_success is False:
                priority = -10e8
            else:
                priority = - avg_completion_time * invoke_num * total_sequence_size

            function_dict[function_id]["avg_completion_time"] = avg_completion_time
            function_dict[function_id]["avg_interval"] = avg_interval
            function_dict[function_id]["cpu"] = cpu
            function_dict[function_id]["memory"] = memory
            function_dict[function_id]["total_sequence_size"] = total_sequence_size
            function_dict[function_id]["is_success"] = is_success
            function_dict[function_id]["invoke_num"] = invoke_num
            function_dict[function_id]["priority"] = priority

        return function_dict
    
    #
    # Get observation for next timestep
    #
    def get_observation(self):
        function_profile = self.profile.get_function_profile()
        next_timestep = self.timetable.get_timestep(self.system_time)

        observation = np.zeros(self.observation_dim)

        # Init mask, always unmask action void
        mask = np.ones(self.action_dim) * -10e8
        mask[-1] = 0

         # Number of undone requests
        observation[0] = self.request_record.get_undone_size()

        # Available cpu and memory per server
        base_server = 0
        for i, server in enumerate(self.cluster.get_server_pool()):
            available_cpu, available_memory = server.resource_manager.get_resources_available()
            observation[base_server + 2*i + 1] = available_cpu
            observation[base_server + 2*i + 2] = available_memory

        # Information of functions
        if next_timestep is not None:
            base_function = 2*self.params.max_server
            offset_function = 0
            for function_id in next_timestep.keys():
                if next_timestep[function_id] > 0:
                    function = function_profile[function_id]
                    cpu_cap_per_function = function.params.cpu_cap_per_function
                    cpu_least_hint = function.params.cpu_least_hint
                    memory_cap_per_function = function.params.memory_cap_per_function
                    memory_least_hint = function.params.memory_least_hint
                    
                    avg_interval = self.request_record.get_avg_interval_per_function(function_id)
                    avg_completion_time = self.request_record.get_avg_completion_time_per_function(function.function_id)
                    is_cold_start = self.request_record.get_is_cold_start_per_function(function_id)
                    cpu = function.get_cpu()
                    memory = function.get_memory()
                    cpu_adjust_direction = function.get_resource_adjust_direction("cpu")
                    memory_adjust_direction = function.get_resource_adjust_direction("memory")
                    invoke_num = next_timestep[function_id]
                    
                    observation[base_function + 8*offset_function + 1] = avg_interval
                    observation[base_function + 8*offset_function + 2] = avg_completion_time
                    observation[base_function + 8*offset_function + 3] = is_cold_start
                    observation[base_function + 8*offset_function + 4] = cpu
                    observation[base_function + 8*offset_function + 5] = memory
                    observation[base_function + 8*offset_function + 6] = cpu_adjust_direction
                    observation[base_function + 8*offset_function + 7] = memory_adjust_direction
                    observation[base_function + 8*offset_function + 8] = invoke_num

                    # Unmask action cpu access
                    if cpu_adjust_direction == 0: 
                        mask[4*offset_function + 0] = 0 # Decrease one CPU slot
                        mask[4*offset_function + 1] = 0 # Increase one CPU slot
                    elif cpu_adjust_direction == -1 and cpu > cpu_least_hint:
                        mask[4*offset_function + 0] = 0 # Decrease one CPU slot
                    elif cpu_adjust_direction == 1 and cpu < cpu_cap_per_function:
                        mask[4*offset_function + 1] = 0 # Increase one CPU slot

                    # Unmask action memory access
                    if memory_adjust_direction == 0: 
                        mask[4*offset_function + 2] = 0 # Decrease one memory slot
                        mask[4*offset_function + 3] = 0 # Increase one memory slot
                    if memory_adjust_direction == -1 and memory > memory_least_hint:
                        mask[4*offset_function + 2] = 0 # Decrease one memory slot
                    elif memory_adjust_direction == 1 and memory < memory_cap_per_function:
                        mask[4*offset_function + 3] = 0 # Increase one memory slot

                    offset_function = offset_function + 1
        
        observation = torch.Tensor(observation).unsqueeze(0)
        mask = torch.Tensor(mask).unsqueeze(0)

        return observation, mask

    #
    # Calculate reward for current timestep
    #
    def get_reward(self, num_timeout):
        reward = 0

        # Timeout penalty
        reward = reward + -(num_timeout * self.params.timeout_penalty)

        # Reward of completion time
        reward = reward + -self.request_record.get_current_completion_time(self.system_time)

        # Discounted by square root of throughput
        throughput = np.max([np.sqrt(self.get_function_throughput()), 1])
        reward = reward / throughput
            
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
        total_available_cpu, total_available_memory = self.cluster.get_total_available_resources()

        info = {
            "system_time": self.system_time,
            "avg_completion_time": self.request_record.get_avg_completion_time(),
            "timeout_num": self.request_record.get_timeout_size(),
            "request_record": self.request_record,
            "function_dict": self.get_function_dict(),
            "function_throughput": self.get_function_throughput(),
            "total_available_cpu": total_available_cpu,
            "total_available_memory": total_available_memory
        }

        if self.get_done() is True:
            info["resource_utils_record"] = self.get_resource_utils_record()
        
        return info
        
    """
    Overwrite gym functions
    """

    def render(self):
        logger.warn("To do")
        pass
    
    def close(self):
        logger.warn("To do")
        pass
    
    def step(self, action=None):
        function_profile = self.profile.get_function_profile()
        is_valid_action = self.update_function_profile(action)
        
        if is_valid_action is True:
            reward = 0
        else:
            # Time proceeds
            self.system_time = self.system_time + self.params.interval

            # Update the cluster
            num_timeout = self.update_cluster()

            # Update resource utilization record
            self.update_resource_utils()

            # Calculate reward
            reward = self.get_reward(num_timeout)
            
            # Reset resource adjust direction for each function 
            for function_id in function_profile.keys():
                function = function_profile[function_id]
                function.reset_resource_adjust_direction()
            
        # Get observation for next state
        observation, mask = self.get_observation()
        
        # Done?
        done = self.get_done()
        
        # Return info
        info = self.get_info()
        
        return observation, mask, reward, done, info
    
    def reset(self):
        self.system_time = 0
        
        self.profile.reset()
        self.cluster.reset()
        self.request_record.reset()
        self.resource_utils_record.reset()
        
        observation, mask = self.get_observation()
        
        return observation, mask
    
    
    
