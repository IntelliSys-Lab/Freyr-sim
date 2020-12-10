import gym
from gym import spaces, logger
import numpy as np
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
        # Action space size: 4*m+1
        self.action_space = spaces.Discrete(4 * self.profile.get_size() + 1)
        
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
        low = np.zeros(1 + 2 * self.cluster.get_cluster_size() + 5 * self.profile.get_size(), dtype=np.float32)
        
        high_part_1 = np.array([10000])

        high_part_2 = []
        for server in self.cluster.get_server_pool():
            high_part_2.append(server.resource_manager.get_user_cpu())
            high_part_2.append(server.resource_manager.get_user_memory())
        high_part_2 = np.array(high_part_2)

        high_part_3 = []
        function_profile = self.profile.get_function_profile()
        for _ in function_profile.keys():
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
        function_profile = self.profile.get_function_profile()
        function_profile_list = list(function_profile.keys())

        function_index = int(action/4)
        function_id = function_profile_list[function_index]
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
        
        return function_id, resource, adjust
        
    #
    # Update settings of function profile based on given action
    #
    def update_function_profile(self, action):
        function_profile = self.profile.get_function_profile()

        if isinstance(action, list): # WARNING! Only used by greedy provision!
            actions = action 
            for act in actions:
                function_id, resource, adjust = self.decode_action(act)
                function = function_profile[function_id]
                # if function_profile[function_id].validate_resource_adjust(resource, adjust) is True:
                #     function_profile[function_id].set_resource_adjust(resource, adjust)
                function.set_resource_adjust(resource, adjust)

                # Set the sequence members as well if it is a function sequence
                if function.get_sequence() is not None:
                    sequence = function.get_sequence()
                    for member_id in sequence:
                        function_profile[member_id].set_resource_adjust(resource, adjust)
            
            return False
        
        if action == self.action_space.n - 1: # Explicit invalid action
            return False
        else:
            function_id, resource, adjust = self.decode_action(action)
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
        function_profile = self.profile.get_function_profile()
        function_dict = {}
        for function_id in function_profile.keys():
            function = function_profile[function_id]
            function_dict[function_id] = {}

            avg_completion_time = self.request_record.get_avg_completion_time_per_function(function_id)
            avg_interval = self.request_record.get_avg_interval_per_function(self.system_time, function_id)
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

            function_dict[function_id]["avg_completion_time"] = avg_completion_time
            function_dict[function_id]["avg_interval"] = avg_interval
            function_dict[function_id]["cpu"] = cpu
            function_dict[function_id]["memory"] = memory
            function_dict[function_id]["total_sequence_size"] = total_sequence_size
            function_dict[function_id]["is_success"] = is_success

        return function_dict
    
    #
    # Get observation for next timestep
    #
    def get_observation(self):
        function_profile = self.profile.get_function_profile()

        num_undone_requests = self.request_record.get_undone_size()
        observation_part_1 = np.array([num_undone_requests])

        observation_part_2 = []
        for server in self.cluster.get_server_pool():
            available_cpu, available_memory = server.resource_manager.get_resources_available()
            observation_part_2.append(available_cpu)
            observation_part_2.append(available_memory)
        observation_part_2 = np.array(observation_part_2)

        observation_part_3 = []
        for function_id in function_profile.keys():
            function = function_profile[function_id]

            observation_part_3.append(function.get_cpu())
            observation_part_3.append(function.get_memory())
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
        reward = 0

        # Timeout penalty
        reward = reward + -(num_timeout * self.params.timeout_penalty)

        # Reward of completion time
        reward = reward + -self.request_record.get_current_completion_time(self.system_time)

        # Discounted by throughput
        throughput = np.max([self.get_function_throughput(), 1])
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
        info = {
            "system_time": self.system_time,
            "avg_completion_time": self.request_record.get_avg_completion_time(),
            "timeout_num": self.request_record.get_timeout_size(),
            "request_record": self.request_record,
            "function_dict": self.get_function_dict(),
            "function_throughput": self.get_function_throughput(),
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
        observation = self.get_observation()
        
        # Done?
        done = self.get_done()
        
        # Return info
        info = self.get_info()
        
        return observation, reward, done, info
    
    def reset(self):
        self.system_time = 0
        
        self.profile.reset()
        self.cluster.reset()
        self.request_record.reset()
        
        observation = self.get_observation()
        
        return observation
    
    
    
