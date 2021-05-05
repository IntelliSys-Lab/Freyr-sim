import sys
import gym
from gym import spaces, logger
import numpy as np
import torch
from gym.envs.serverless.faas_utils import SystemTime, Request, Cluster, RequestRecord, ResourceUtilsRecord
from gym.envs.serverless.workload_generator import WorkloadGenerator

        
class FaaSEnv(gym.Env):
    """
    Function-as-a-Service environment.
    """
    
    def __init__(
        self,
        workload_params,
        env_params
    ):
        self.workload_params = workload_params
        self.env_params = env_params
        
        # Set up workloads
        self.workload_generator = WorkloadGenerator(
            exp_id=self.workload_params.exp_id,
            azure_file_path=self.workload_params.azure_file_path,
            cpu_cap_per_function=self.env_params.cpu_cap_per_function,
            memory_cap_per_function=self.env_params.memory_cap_per_function,
            memory_mb_limit=self.env_params.memory_mb_limit
        )
        self.profile, self.event_pq = self.workload_generator.generate_workload()
        
        # Set up cluster
        self.cluster = Cluster(
            cluster_size=self.env_params.cluster_size,
            schedule_step_size=7,
            user_cpu_per_server=self.env_params.user_cpu_per_server,
            user_memory_per_server=self.env_params.user_memory_per_server,
            keep_alive_window=self.env_params.keep_alive_window
        )

        # Set up system time module
        self.system_time = SystemTime(self.env_params.interval)

        # Set up request record
        self.request_record = RequestRecord(self.profile.get_function_profile())

        # Set up resource utils record
        self.resource_utils_record = ResourceUtilsRecord(self.cluster.get_cluster_size())
        
    #
    # Decode action
    #
    
    def decode_action(self, index):
        if index is not None:
            action = {}
            action["cpu"] = int(index / self.env_params.memory_cap_per_function) + 1
            action["memory"] = int(index % self.env_params.memory_cap_per_function) + 1
        else:
            action = None

        return action

    #
    # Encode action
    #
    
    def encode_action(self, action):
        return (action["cpu"] - 1) * self.env_params.memory_cap_per_function + action["memory"] - 1

    #
    # Update settings of function profile based on given action
    #

    def update_function_profile(self, next_function_id, action):
        if action is not None:
            function = self.profile.get_function_profile()[next_function_id]
            function.set_function(cpu=action["cpu"], memory=action["memory"])

    #            
    # Update the cluster
    #
    
    def update_cluster(self, next_function_id=None):
        function_profile = self.profile.get_function_profile()

        # Import next event
        if next_function_id is not None:
            function = function_profile[next_function_id]
            request = Request(
                function=function, 
                invoke_time=self.system_time.get_system_runtime(),
            )

            self.request_record.put_requests(request)
            self.cluster.schedule(request)
            
        request_to_remove_list, num_timeout = self.cluster.step(self.system_time.get_system_runtime())
        self.request_record.update_requests(request_to_remove_list)

        return num_timeout

    #
    # Update resource utilization record
    #

    def update_resource_utils(self):
        for index, server in enumerate(self.cluster.get_server_pool()):
            server_id = "server{}".format(index)
            cpu_util, memory_util = server.manager.get_resource_utils()
            self.resource_utils_record.put_resource_utils(server_id, cpu_util, memory_util)

    def get_resource_utils_record(self):
        self.resource_utils_record.calculate_avg_resource_utils()
        return self.resource_utils_record.get_resource_utils_record()

    def get_function_throughput(self):
        return self.request_record.get_success_size() + self.request_record.get_timeout_size()

    #
    # Observation space size: (15 + 2) for each, in total n * m
    #
    # [
    #  [num_inflight_requests, 
    #   server_in_use_cpu, 
    #   server_in_use_memory, 
    #   server_available_cpu, 
    #   server_available_memory,
    #   function_avg_execution_time,
    #   function_avg_waiting_time,
    #   function_avg_interval,
    #   function_avg_invoke_num,
    #   function_avg_completion_time_slo,
    #   function_avg_cpu_slo,
    #   function_avg_memory_slo,
    #   function_cpu_user_defined,
    #   function_memory_user_defined,
    #   function_baseline_duration,
    #   cpu_choice_1,
    #   memory_choice_1],
    #  ...,
    # [
    #  [num_inflight_requests, 
    #   server_in_use_cpu, 
    #   server_in_use_memory, 
    #   server_available_cpu, 
    #   server_available_memory,
    #   function_avg_execution_time,
    #   function_avg_waiting_time,
    #   function_avg_interval,
    #   function_avg_invoke_num,
    #   function_avg_completion_time_slo,
    #   function_avg_cpu_slo,
    #   function_avg_memory_slo,
    #   function_cpu_user_defined,
    #   function_memory_user_defined,
    #   function_baseline_duration,
    #   cpu_choice_n,
    #   memory_choice_m],
    # ]

    def get_observation(
        self,
        next_function_id
    ):
        function = self.profile.get_function_profile()[next_function_id]
        
        # Init observation
        n_undone_request = self.request_record.get_undone_size()
        server_in_use_cpu, server_in_use_memory = self.cluster.get_total_in_use_resources()
        server_available_cpu, server_available_memory = self.cluster.get_total_available_resources()
        function_avg_interval = self.request_record.get_avg_interval_per_function(next_function_id)
        function_avg_execution_time = self.request_record.get_avg_execution_time_per_function(next_function_id)
        function_avg_waiting_time = self.request_record.get_avg_waiting_time_per_function(next_function_id)
        function_avg_invoke_num = self.request_record.get_avg_invoke_num_per_function(next_function_id, self.system_time.get_system_runtime())
        function_avg_completion_time_slo = self.request_record.get_avg_completion_time_slo_per_function(next_function_id)
        function_avg_cpu_slo = self.request_record.get_avg_cpu_slo_per_function(next_function_id)
        function_avg_memory_slo = self.request_record.get_avg_memory_slo_per_function(next_function_id)
        function_cpu_user_defined = function.get_cpu_user_defined()
        function_memory_user_defined = function.get_memory_user_defined()
        function_baseline_duration = function.get_baseline_duration()

        state_batch = []
        for cpu in range(1, self.env_params.cpu_cap_per_function + 1):
            for memory in range(1, self.env_params.memory_cap_per_function + 1):
                state = []
                state.append(n_undone_request)
                state.append(server_in_use_cpu)
                state.append(server_in_use_memory)
                state.append(server_available_cpu)
                state.append(server_available_memory)
                state.append(function_avg_execution_time)
                state.append(function_avg_waiting_time)
                state.append(function_avg_interval)
                state.append(function_avg_invoke_num)
                state.append(function_avg_completion_time_slo)
                state.append(function_avg_cpu_slo)
                state.append(function_avg_memory_slo)
                state.append(function_cpu_user_defined)
                state.append(function_memory_user_defined)
                state.append(function_baseline_duration)
                state.append(cpu)
                state.append(memory)

                state_batch.append(state)

        observation = torch.Tensor(state_batch)

        # Init mask
        if self.request_record.get_success_size_per_function(next_function_id) == 0 and \
        self.request_record.get_timeout_size_per_function(next_function_id) == 0:
            mask = torch.ones(observation.size(0)) * -10e8
            index_user_defined = self.encode_action(
                {"cpu": function_cpu_user_defined, "memory": function_memory_user_defined}
            )
            mask[index_user_defined] = 0
        else:
            mask = torch.zeros(observation.size(0))

        observation = observation.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return observation, mask

    #
    # Calculate reward for current timestep
    #
    def get_reward(
        self, 
        interval,
        num_timeout=None
    ):
        reward = 0

        # Timeout penalty
        if num_timeout is not None:
            reward = reward - num_timeout * self.env_params.fail_penalty

        # SLO penalty
        for request in self.request_record.get_undone_request_record():
            baseline_duration = request.profile.get_baseline_duration()
            cpu_slo = request.get_cpu_slo()
            memory_slo = request.get_memory_slo()
            # reward = reward -  cpu_slo * memory_slo / baseline_duration
            reward = reward -  1 / baseline_duration

        reward = interval * reward

        return reward
    
    #
    # Get done for current timestep
    #
    def get_done(self):
        if self.event_pq.is_empty() is True and \
        self.system_time.get_system_runtime() > self.event_pq.get_max_timestep() - 1 and \
        self.request_record.get_undone_size() == 0:
            return True
        else:
            return False
    
    #
    # Get info for current timestep
    #
    def get_info(self):
        total_available_cpu, total_available_memory = self.cluster.get_total_available_resources()

        info = {
            "system_time": self.system_time.get_system_runtime(),
            "avg_completion_time_slo": self.request_record.get_avg_completion_time_slo(),
            "avg_completion_time": self.request_record.get_avg_completion_time(),
            "timeout_num": self.request_record.get_timeout_size(),
            "request_record": self.request_record,
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
    
    def step(
        self, 
        current_timestep,
        current_function_id,
        action
    ):
        function_profile = self.profile.get_function_profile()
        
        # Get next event
        if self.event_pq.is_empty() is False:
            # Get temporal rewards
            reward = 0
            system_step = self.system_time.get_system_runtime()
            while system_step < current_timestep:
                self.system_time.step()
                reward = reward + self.get_reward(interval=self.system_time.get_default_interval())

                self.update_resource_utils()
                num_timeout = self.update_cluster()
                system_step = self.system_time.get_system_runtime()

            # Go to next event
            reward = reward + self.get_reward(interval=0)
            self.update_function_profile(current_function_id, action)
            num_timeout = self.update_cluster(current_function_id)

            # Get next event
            next_timestep, next_function_id = self.event_pq.get_event()

            # Get observation for next state
            observation, mask = self.get_observation(next_function_id=next_function_id)
        else:
            # Retrieve tail rewards
            reward = 0
            while self.request_record.get_undone_size() > 0:
                self.system_time.step()
                reward = reward + self.get_reward(interval=self.system_time.get_default_interval())
                self.update_resource_utils()
                num_timeout = self.update_cluster()
            
            observation = None
            mask = None
            next_timestep = None
            next_function_id = None

       # Done?
        done = self.get_done()

        # Return info
        info = self.get_info()
        
        return observation, mask, reward, done, info, next_timestep, next_function_id
    
    def reset(self):
        self.system_time.reset()
        self.profile.reset()
        self.event_pq.reset()
        self.request_record.reset()
        self.resource_utils_record.reset()
        
        next_timestep, next_function_id = self.event_pq.get_event()
        observation, mask = self.get_observation(next_function_id=next_function_id)
        
        return observation, mask, next_timestep, next_function_id
