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
        
        # Define action space size
        self.action_dim_cpu = self.params.user_cpu_per_server
        self.action_dim_memory = self.params.user_memory_per_server
        
        # Define observation space size
        #
        # State for cpu: 1 + n + 4
        # [
        #  num_in_fight_requests, 
        #  server_1_available_cpu, 
        #  .
        #  .
        #  .
        #  server_n_available_cpu, 
        #  next_function_avg_cpu,
        #  next_function_avg_interval,
        #  next_function_avg_completion_time,
        #  next_function_is_cold_start,
        # ]
        #
        # State for memory: 1 + n + 4
        # [
        #  num_in_fight_requests, 
        #  server_1_available_memory,
        #  .
        #  .
        #  .
        #  server_n_available_memory,
        #  next_function_avg_memory,
        #  next_function_avg_interval,
        #  next_function_avg_completion_time,
        #  next_function_is_cold_start,
        # ]
        self.observation_dim_cpu = 1 + self.cluster.get_cluster_size() + 4
        self.observation_dim_memory = 1 + self.cluster.get_cluster_size() + 4
    
    #
    # Update settings of function profile based on given action map
    #
    def update_function_profile(
        self, 
        function_id, 
        action_cpu,
        action_memory
    ):
        function_profile = self.profile.get_function_profile()
        function = function_profile[function_id]
        function.set_function(cpu=action_cpu, memory=action_memory)
        
    #            
    # Update the cluster
    #
    def update_cluster(self, function_id):
        function_profile = self.profile.get_function_profile()

        # Try to import timetable if not finished
        timestep = self.timetable.get_timestep(self.system_time - 1)
        request_to_schedule_list = []

        if timestep is not None:
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

    #
    # Proceed the cluster
    #
    def proceed_cluster(self):
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
            avg_cpu = self.request_record.get_avg_cpu_per_function(function_id)
            avg_memory = self.request_record.get_avg_memory_per_function(function_id)
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
            function_dict[function_id]["avg_cpu"] = avg_cpu
            function_dict[function_id]["avg_memory"] = avg_memory
            function_dict[function_id]["total_sequence_size"] = total_sequence_size
            function_dict[function_id]["is_success"] = is_success

        return function_dict
    
    #
    # Get observations for next timestep
    #
    def get_observation(self, function_id):
        function_profile = self.profile.get_function_profile()

        num_undone_requests = self.request_record.get_undone_size()
        observation_num_undone_requests = np.array([num_undone_requests])

        observation_cluster_cpu = []
        observation_cluster_memory = []
        for server in self.cluster.get_server_pool():
            available_cpu, available_memory = server.resource_manager.get_resources_available()
            observation_cluster_cpu.append(available_cpu)
            observation_cluster_memory.append(available_memory)
        observation_cluster_cpu = np.array(observation_cluster_cpu)
        observation_cluster_memory = np.array(observation_cluster_memory)

        observation_next_function_cpu = []
        observation_next_function_memory = []
        avg_cpu = self.request_record.get_avg_cpu_per_function(function_id)
        avg_memory = self.request_record.get_avg_memory_per_function(function_id)
        avg_interval = self.request_record.get_avg_interval_per_function(self.system_time, function_id)
        avg_completion_time = self.request_record.get_avg_completion_time_per_function(function_id)
        is_cold_start = self.request_record.get_is_cold_start_per_function(function_id)

        observation_next_function_cpu.append(avg_cpu)
        observation_next_function_cpu.append(avg_interval)
        observation_next_function_cpu.append(avg_completion_time)
        observation_next_function_cpu.append(is_cold_start)
        observation_next_function_memory.append(avg_memory)
        observation_next_function_memory.append(avg_interval)
        observation_next_function_memory.append(avg_completion_time)
        observation_next_function_memory.append(is_cold_start)

        observation_next_function_cpu = np.array(observation_next_function_cpu)
        observation_next_function_memory = np.array(observation_next_function_memory)
        
        observation_cpu = np.hstack((observation_num_undone_requests, observation_cluster_cpu, observation_next_function_cpu))
        observation_memory = np.hstack((observation_num_undone_requests, observation_cluster_memory, observation_next_function_memory))
        
        return observation_cpu, observation_memory
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
    
    def step(
        self, 
        time_proceed,
        function_id=None,
        action_cpu=None,
        action_memory=None
    ):
        function_profile = self.profile.get_function_profile()

        if function_id is not None:
            if action_cpu is not None and action_memory is not None:
                # Update function profile
                self.update_function_profile(function_id, action_cpu, action_memory)

            # Update the cluster
            self.update_cluster(function_id)

        if time_proceed is True:
            # Time proceeds
            self.system_time = self.system_time + self.params.interval

            # Proceed the cluster
            num_timeout = self.proceed_cluster()

            # Update resource utilization record
            self.update_resource_utils()

            # Calculate reward
            reward = self.get_reward(num_timeout)
        else:
            # During bursts at the same timestep
            reward = 0
            
        # Done?
        done = self.get_done()
        
        # Return info
        info = self.get_info()
        
        return reward, done, info
    
    def reset(self):
        self.system_time = 0
        
        self.profile.reset()
        self.cluster.reset()
        self.request_record.reset()
        
