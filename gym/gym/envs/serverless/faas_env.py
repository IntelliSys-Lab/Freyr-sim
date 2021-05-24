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
        return (action["cpu"] - 1) * self.env_params.memory_cap_per_function + action["memory"]

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
        good_slo, bad_slo, total_duration_slo = self.request_record.update_requests(request_to_remove_list)

        return good_slo, bad_slo, total_duration_slo

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

    def get_mask(
        self, 
        next_function_id,
        cpu_range,
        memory_range
    ):
        batch_size = self.env_params.cpu_cap_per_function * self.env_params.memory_cap_per_function
        mask = torch.zeros(batch_size)

        if len(cpu_range) == 1:
            for i in range(mask.size(0)):
                cpu = self.decode_action(i)["cpu"]
                if cpu != cpu_range[0]:
                    mask[i] = -1e8
        else:
            for i in range(mask.size(0)):
                cpu = self.decode_action(i)["cpu"]
                if cpu < cpu_range[0] or cpu > cpu_range[1]:
                    mask[i] = -1e8

        if len(memory_range) == 1:
            for i in range(mask.size(0)):
                mem = self.decode_action(i)["memory"]
                if mem != memory_range[0]:
                    mask[i] = -1e8
        else:
            for i in range(mask.size(0)):
                mem = self.decode_action(i)["memory"]
                if mem < memory_range[0] or mem > memory_range[1]:
                    mask[i] = -1e8

        return mask

    def get_observation(
        self,
        next_function_id
    ):
        function = self.profile.get_function_profile()[next_function_id]
        
        # Init observation
        n_undone_request = self.request_record.get_undone_size()
        server_available_cpu, server_available_memory = self.cluster.get_total_available_resources()
        function_avg_interval = self.request_record.get_avg_interval_per_function(next_function_id)
        function_avg_invoke_num = self.request_record.get_avg_invoke_num_per_function(next_function_id, self.system_time.get_system_runtime())
        function_avg_cpu_peak = self.request_record.get_avg_cpu_peak_per_function(next_function_id)
        function_avg_memory_peak = self.request_record.get_avg_memory_peak_per_function(next_function_id)
        function_avg_duration = self.request_record.get_avg_duration_per_function(next_function_id)
        function_baseline = function.get_baseline()

        state_batch = []
        for cpu in range(1, self.env_params.cpu_cap_per_function + 1):
            for memory in range(1, self.env_params.memory_cap_per_function + 1):
                state = []
                state.append(n_undone_request)
                state.append(server_available_cpu)
                state.append(server_available_memory)
                state.append(function_avg_interval)
                state.append(function_avg_invoke_num)
                state.append(function_avg_cpu_peak)
                state.append(function_avg_memory_peak)
                state.append(function_avg_duration)
                state.append(function_baseline)
                state.append(cpu)
                state.append(memory)

                state_batch.append(state)

        observation = torch.Tensor(state_batch)

        # Init mask
        cpu_cap_per_function = self.env_params.cpu_cap_per_function
        memory_cap_per_function = self.env_params.memory_cap_per_function
        cpu_user_defined = function.get_cpu_user_defined()
        memory_user_defined = function.get_memory_user_defined()
        
        last_request = self.request_record.get_last_done_request_per_function(next_function_id)
        is_safeguard = False

        if last_request is None or last_request.get_status() == "timeout":
            cpu_range = [cpu_user_defined]
            memory_range = [memory_user_defined]
            is_safeguard = True
            # if last_request is None:
                # print("{} first request, safeguard activate".format(next_function_id))
            # elif last_request.get_is_success() is False:
                # print("{} last request failed, safeguard activate".format(next_function_id))
        else:
            last_cpu_alloc = last_request.get_cpu()
            last_mem_alloc = last_request.get_memory()
            last_cpu_peak = last_request.get_cpu_peak()
            last_mem_peak = last_request.get_memory_peak()
            recent_cpu_peak = self.request_record.get_recent_cpu_peak_per_function(next_function_id)
            recent_memory_peak = self.request_record.get_recent_memory_peak_per_function(next_function_id)

            if last_cpu_peak / cpu_user_defined <= 0.9: # Over-provisioned
                if last_cpu_peak / last_cpu_alloc >= 0.9: # Usage spike
                    cpu_range = [cpu_user_defined]
                    is_safeguard = True
                    # print("{}, last_cpu_peak {}, last_cpu_alloc {}, cpu safeguard activate".format(next_function_id, last_cpu_peak, last_cpu_alloc))
                else:
                    cpu_range = [min(int(recent_cpu_peak) + 1, cpu_user_defined), cpu_user_defined]
            else: # Under-provisioned
                cpu_range = [min(int(recent_cpu_peak) + 1, cpu_cap_per_function), cpu_cap_per_function]
                    
            if last_mem_peak / memory_user_defined <= 0.9: # Over-provisioned
                if last_mem_peak / last_mem_alloc >= 0.9: # Usage spike
                    memory_range = [memory_user_defined]
                    is_safeguard = True
                    # print("{}, last_mem_peak {}, last_mem_alloc {}, memory safeguard activate".format(next_function_id, last_mem_peak, last_mem_alloc))
                else:
                    memory_range = [min(int(recent_memory_peak) + 1, memory_user_defined), memory_user_defined]
            else: # Under-provisioned
                memory_range = [min(int(recent_memory_peak) + 1, memory_cap_per_function), memory_cap_per_function]

            # if len(cpu_range) > 1 and cpu_range[0] >= cpu_range[1]:
            #     print("last_cpu_peak: {}, last_cpu_alloc: {}, cpu_user_defined: {}".format(last_cpu_peak, last_cpu_alloc, cpu_user_defined))
            #     print("cpu_range: {} - {}".format(cpu_range[0], cpu_range[1]))
            # if len(memory_range) > 1 and memory_range[0] >= memory_range[1]:
            #     print("last_mem_peak: {}, last_mem_alloc: {}, memory_user_defined: {}".format(last_mem_peak, last_mem_alloc, memory_user_defined))
            #     print("memory_range: {} - {}".format(memory_range[0], memory_range[1]))

        mask = self.get_mask(
            next_function_id=next_function_id,
            cpu_range=cpu_range,
            memory_range=memory_range
        )

        observation = observation.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return observation, mask

    #
    # Calculate reward for current timestep
    #
    def get_reward(
        self, 
        good_slo,
        bad_slo,
        total_duration_slo
    ):
        if self.get_function_throughput() == 0:
            reward = - total_duration_slo
        else:
            reward = - total_duration_slo / self.get_function_throughput() ** (1/3)

        # Constant summary on good and bad decisions
        reward = reward + good_slo - bad_slo

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
            "request_record": self.request_record,
            "function_throughput": self.get_function_throughput(),
            "total_available_cpu": total_available_cpu,
            "total_available_memory": total_available_memory
        }

        if self.get_done() is True:
            info["avg_duration_slo"] = self.request_record.get_avg_duration_slo()
            info["avg_harvest_cpu_percent"] = self.request_record.get_avg_harvest_cpu_percent()
            info["avg_harvest_memory_percent"] = self.request_record.get_avg_harvest_memory_percent()
            info["slo_violation_percent"] = self.request_record.get_slo_violation_percent()
            info["acceleration_pecent"] = self.request_record.get_acceleration_pecent()
            info["timeout_num"] = self.request_record.get_timeout_size()
            info["request_record"] = self.request_record
            # info["avg_interval"] = self.request_record.get_avg_interval()
        
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
                good_slo, bad_slo, total_duration_slo = self.update_cluster()
                reward = reward + self.get_reward(
                    good_slo=good_slo,
                    bad_slo=bad_slo,
                    total_duration_slo=total_duration_slo
                )
                
                system_step = self.system_time.get_system_runtime()

            # Go to next event
            good_slo, bad_slo, total_duration_slo = self.update_cluster(current_function_id)
            self.update_function_profile(current_function_id, action)

            # Get next event
            next_timestep, next_function_id = self.event_pq.get_event()

            # Get observation for next state
            observation, mask = self.get_observation(next_function_id=next_function_id)

            # Get rewards
            reward = reward + self.get_reward(
                good_slo=good_slo,
                bad_slo=bad_slo,
                total_duration_slo=total_duration_slo
            )
        else:
            # Retrieve tail rewards
            reward = 0
            while self.get_done() is False:
                self.system_time.step()
                good_slo, bad_slo, total_duration_slo = self.update_cluster()
                reward = reward + self.get_reward(
                    good_slo=good_slo,
                    bad_slo=bad_slo,
                    total_duration_slo=total_duration_slo
                )
            
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
