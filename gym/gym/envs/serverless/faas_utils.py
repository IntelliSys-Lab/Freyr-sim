import numpy as np
import copy as cp
import uuid
import queue
import random
import functools
import heapq


@functools.total_ordering
class Prioritize:
    """
    Wrapper for non-comparative objects
    """
    def __init__(self, priority, item):
        self.priority = priority
        self.item = item

    def __eq__(self, other):
        return self.priority == other.priority

    def __lt__(self, other):
        return self.priority < other.priority


class Function():
    """
    Function used by FaaSEnv
    """
    
    def __init__(self, params):
        self.params = params
        
        if self.params.function_id != None:
            self.function_id = self.params.function_id
        else:
            self.function_id = uuid.uuid1()

        self.hash_value = self.params.hash_value

    def set_function(self, cpu, memory):
        self.cpu = cpu
        self.memory = memory
        
        # Calculate duration
        # Assume CPU has k times more impact on duration than memory
        cpu_duration = (self.params.max_duration - self.params.min_duration) * self.params.k / (self.params.k + 1)
        if self.cpu >= self.params.ideal_cpu:
            cpu_delay_factor = 0
        else:
            cpu_delay_factor = (self.params.ideal_cpu - self.cpu) / (self.params.ideal_cpu - self.params.cpu_least_hint)
        cpu_duration_increment = cpu_duration * cpu_delay_factor

        memory_duration = (self.params.max_duration - self.params.min_duration) * 1 / (self.params.k + 1)
        memory_mb = self.memory * self.params.memory_mb_limit / self.params.memory_cap_per_function
        least_memory_mb = self.params.memory_least_hint * self.params.memory_mb_limit / self.params.memory_cap_per_function
        if memory_mb >= self.params.ideal_memory:
            memory_delay_factor = 0
        else:
            memory_delay_factor = (self.params.ideal_memory - memory_mb) / (self.params.ideal_memory - least_memory_mb)
        memory_duration_increment = memory_duration * memory_delay_factor

        self.duration = self.params.min_duration + cpu_duration_increment + memory_duration_increment

        # print("ideal_memory: {}, memory_mb: {}".format(self.params.ideal_memory, memory_mb))
        # print("cpu_delay_factor: {}".format(cpu_delay_factor))
        # print("cpu_duration_increment: {}".format(cpu_duration_increment))
        # print("memory_delay_factor: {}".format(memory_delay_factor))
        # print("memory_duration_increment: {}".format(memory_duration_increment))

    def set_baseline(self):
        self.baseline = cp.deepcopy(self.duration)

    def get_function_id(self):
        return self.function_id

    def get_hash_value(self):
        return self.hash_value

    def get_cpu(self):
        return self.cpu

    def get_memory(self):
        return self.memory

    def get_ideal_cpu(self):
        return self.params.ideal_cpu

    def get_ideal_memory(self):
        return self.params.ideal_memory

    def get_cpu_user_defined(self):
        return self.params.cpu_user_defined

    def get_memory_user_defined(self):
        return self.params.memory_user_defined

    def get_min_duration(self):
        return self.params.min_duration
    
    def get_baseline(self):
        return self.baseline


class Request():
    """
    An invocation of a function
    """
    def __init__(
        self, 
        function, 
        invoke_time,
    ):
        self.profile = cp.deepcopy(function)
        self.request_id = uuid.uuid1()
        
        self.progress = 0
        self.waiting = 0
        self.status = "undone"

        self.invoke_time = invoke_time
        self.done_time = 0

        self.is_cold_start = None

    def set_is_cold_start(self, is_cold_start):
        self.is_cold_start = is_cold_start

        # Add cold start overhead
        if self.is_cold_start is True:
            self.profile.duration = self.profile.duration + self.profile.params.cold_start_time

    def get_cpu(self):
        return self.profile.get_cpu()

    def get_memory(self):
        return self.profile.get_memory()

    def get_cpu_user_defined(self):
        return self.profile.get_cpu_user_defined()

    def get_memory_user_defined(self):
        return self.profile.get_memory_user_defined()

    def get_function_id(self):
        return self.profile.get_function_id()

    def get_request_id(self):
        return self.request_id

    def get_invoke_time(self):
        return self.invoke_time

    def get_done_time(self):
        return self.done_time

    def get_waiting_time(self):
        return self.waiting

    def get_progress_time(self):
        return self.progress

    def get_duration_slo(self):
        return self.progress / self.profile.get_baseline()

    def get_completion_time(self):
        return self.progress + self.waiting

    def get_completion_time_slo(self):
        return self.get_completion_time() / self.profile.get_baseline()

    def get_cpu_peak(self):
        return min(self.get_cpu(), self.profile.get_ideal_cpu())

    def get_memory_peak(self):
        return min(self.get_memory(), self.profile.get_ideal_memory())
    
    def get_status(self):
        return self.status

    def get_is_cold_start(self):
        return self.is_cold_start
    
    def step(self, system_time, in_registry):
        # In Registry
        if in_registry is True:
            self.progress = self.progress + 1
            
            # Check status
            if self.progress >= self.profile.params.timeout:
                self.status = "timeout" # Timeout
                self.done_time = system_time
            
            if self.progress >= self.profile.duration:
                self.status = "success" # Done
                self.done_time = system_time
        # In Queue
        else:
            self.waiting = self.waiting + 1
        
        return self.status


class RequestRecord():
    """
    Recording of requests both in total and per Function
    """

    def __init__(self, function_profile):
        # General records
        self.total_request_record = []
        self.success_request_record = []
        self.undone_request_record = []
        self.timeout_request_record = []

        # Records per function
        self.total_request_record_per_function = {}
        self.success_request_record_per_function = {}
        self.undone_request_record_per_function = {}
        self.timeout_request_record_per_function = {}

        for function_id in function_profile.keys():
            self.total_request_record_per_function[function_id] = []
            self.success_request_record_per_function[function_id] = []
            self.undone_request_record_per_function[function_id] = []
            self.timeout_request_record_per_function[function_id] = []

    def put_requests(self, request_list):
        if isinstance(request_list, Request):
            request = request_list
            function_id = request.get_function_id()
            status = request.get_status()

            self.total_request_record.append(request)
            self.total_request_record_per_function[function_id].append(request)

            if status == "success":
                self.success_request_record.append(request)
                self.success_request_record_per_function[function_id].append(request)
            elif status == "undone":
                self.undone_request_record.append(request)
                self.undone_request_record_per_function[function_id].append(request)
            elif status == "timeout":
                self.timeout_request_record.append(request)
                self.timeout_request_record_per_function[function_id].append(request)
        else:
            for request in request_list:
                function_id = request.get_function_id()
                status = request.get_status()

                self.total_request_record.append(request)
                self.total_request_record_per_function[function_id].append(request)

                if status == "success":
                    self.success_request_record.append(request)
                    self.success_request_record_per_function[function_id].append(request)
                elif status == "undone":
                    self.undone_request_record.append(request)
                    self.undone_request_record_per_function[function_id].append(request)
                elif status == "timeout":
                    self.timeout_request_record.append(request)
                    self.timeout_request_record_per_function[function_id].append(request)

    def update_requests(self, done_request_list):
        good_slo = 0
        bad_slo = 0
        total_duration_slo = 0

        for request in done_request_list:
            function_id = request.get_function_id()
            status = request.get_status()

            duration_slo = request.get_duration_slo()
            if duration_slo < 1:
                good_slo = good_slo + (1 - duration_slo)
            elif duration_slo > 1:
                bad_slo = bad_slo + (duration_slo - 1) * (request.get_cpu_user_defined() + request.get_memory_user_defined())

            if status == "success":
                self.success_request_record.append(request)
                self.success_request_record_per_function[function_id].append(request)
            elif status == "timeout":
                self.timeout_request_record.append(request)
                self.timeout_request_record_per_function[function_id].append(request)
            
            self.undone_request_record.remove(request)
            self.undone_request_record_per_function[function_id].remove(request)

        return good_slo, bad_slo, total_duration_slo

    def get_total_size(self):
        return len(self.total_request_record)

    def get_undone_size(self):
        return len(self.undone_request_record)

    def get_success_size(self):
        return len(self.success_request_record)

    def get_timeout_size(self):
        return len(self.timeout_request_record)

    def get_last_done_request_per_function(self, function_id):
        last_request = None
        for request in reversed(self.total_request_record_per_function[function_id]):
            if request.get_status() != "undone":
                last_request = request
                break

        return last_request

    def get_avg_duration_slo(self):
        request_num = 0
        total_duration_slo = 0

        for request in self.success_request_record:
            request_num = request_num + 1
            total_duration_slo = total_duration_slo + request.get_duration_slo()

        for request in self.timeout_request_record:
            request_num = request_num + 1
            total_duration_slo = total_duration_slo + request.get_duration_slo()

        if request_num == 0:
            avg_duration_slo = 0
        else:
            avg_duration_slo = total_duration_slo / request_num

        return avg_duration_slo

    def get_avg_completion_time_slo(self):
        request_num = 0
        total_completion_time_slo = 0

        for request in self.success_request_record:
            request_num = request_num + 1
            total_completion_time_slo = total_completion_time_slo + request.get_completion_time_slo()

        for request in self.timeout_request_record:
            request_num = request_num + 1
            total_completion_time_slo = total_completion_time_slo + request.get_completion_time_slo()

        if request_num == 0:
            avg_completion_time_slo = 0
        else:
            avg_completion_time_slo = total_completion_time_slo / request_num

        return avg_completion_time_slo

    def get_avg_completion_time(self):
        request_num = 0
        total_completion_time = 0

        for request in self.success_request_record:
            request_num = request_num + 1
            total_completion_time = total_completion_time + request.get_completion_time()

        for request in self.timeout_request_record:
            request_num = request_num + 1
            total_completion_time = total_completion_time + request.get_completion_time()
        
        if request_num == 0:
            avg_completion_time = 0
        else:
            avg_completion_time = total_completion_time / request_num

        return avg_completion_time

    def get_total_size_per_function(self, function_id):
        total_size_per_function = len(self.total_request_record_per_function[function_id])
        return total_size_per_function

    def get_undone_size_per_function(self, function_id):
        undone_size_per_function = len(self.undone_request_record_per_function[function_id])
        return undone_size_per_function

    def get_success_size_per_function(self, function_id):
        success_size_per_function = len(self.success_request_record_per_function[function_id])
        return success_size_per_function

    def get_timeout_size_per_function(self, function_id):
        timeout_size_per_function = len(self.timeout_request_record_per_function[function_id])
        return timeout_size_per_function

    def get_avg_cpu_peak_per_function(self, function_id):
        request_num = 0
        total_cpu_peak = 0

        for request in self.success_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_cpu_peak = total_cpu_peak + request.get_cpu_peak()

        for request in self.timeout_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_cpu_peak = total_cpu_peak + request.get_cpu_peak()
        
        if request_num == 0:
            avg_cpu_peak_per_function = 0
        else:
            avg_cpu_peak_per_function = total_cpu_peak / request_num

        return avg_cpu_peak_per_function

    def get_avg_memory_peak_per_function(self, function_id):
        request_num = 0
        total_memory_peak = 0

        for request in self.success_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_memory_peak = total_memory_peak + request.get_memory_peak()

        for request in self.timeout_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_memory_peak = total_memory_peak + request.get_memory_peak()
        
        if request_num == 0:
            avg_memory_peak_per_function = 0
        else:
            avg_memory_peak_per_function = total_memory_peak / request_num

        return avg_memory_peak_per_function

    def get_recent_cpu_peak_per_function(self, function_id):
        recent_cpu_peak = 0
        for request in reversed(self.success_request_record_per_function[function_id]):
            if request.get_cpu_peak() > recent_cpu_peak:
                recent_cpu_peak = request.get_cpu_peak()

        return recent_cpu_peak

    def get_recent_memory_peak_per_function(self, function_id):
        recent_memory_peak = 0
        for request in reversed(self.success_request_record_per_function[function_id]):
            if request.get_memory_peak() > recent_memory_peak:
                recent_memory_peak = request.get_memory_peak()

        return recent_memory_peak

    def get_avg_cpu_slo_per_function(self, function_id):
        request_num = 0
        total_cpu_slo = 0

        for request in self.success_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_cpu_slo = total_cpu_slo + request.get_cpu_slo()

        for request in self.timeout_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_cpu_slo = total_cpu_slo + request.get_cpu_slo()
        
        if request_num == 0:
            avg_cpu_slo_per_function = 0
        else:
            avg_cpu_slo_per_function = total_cpu_slo / request_num

        return avg_cpu_slo_per_function

    def get_avg_memory_per_function(self, function_id):
        request_num = 0
        total_memory = 0

        for request in self.success_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_memory = total_memory + request.get_memory()

        for request in self.timeout_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_memory = total_memory + request.get_memory()
        
        if request_num == 0:
            avg_memory_per_function = 0
        else:
            avg_memory_per_function = total_memory / request_num

        return avg_memory_per_function

    def get_avg_memory_slo_per_function(self, function_id):
        request_num = 0
        total_memory_slo = 0

        for request in self.success_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_memory_slo = total_memory_slo + request.get_memory_slo()

        for request in self.timeout_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_memory_slo = total_memory_slo + request.get_memory_slo()
        
        if request_num == 0:
            avg_memory_slo_per_function = 0
        else:
            avg_memory_slo_per_function = total_memory_slo / request_num

        return avg_memory_slo_per_function

    def get_avg_duration_per_function(self, function_id):
        request_num = 0
        total_duration = 0

        for request in self.success_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_duration = total_duration + request.get_progress_time()

        for request in self.timeout_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_duration = total_duration + request.get_progress_time()
        
        if request_num == 0:
            avg_duration_per_function = 0
        else:
            avg_duration_per_function = total_duration / request_num

        return avg_duration_per_function

    def get_avg_duration_slo_per_function(self, function_id):
        request_num = 0
        total_duration_slo = 0

        for request in self.success_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_duration_slo = total_duration_slo + request.get_duration_slo()

        for request in self.timeout_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_duration_slo = total_duration_slo + request.get_duration_slo()
        
        if request_num == 0:
            avg_duration_slo_per_function = 0
        else:
            avg_duration_slo_per_function = total_duration_slo / request_num

        return avg_duration_slo_per_function

    def get_avg_waiting_time_per_function(self, function_id):
        request_num = 0
        total_waiting_time = 0

        for request in self.success_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_waiting_time = total_waiting_time + request.get_waiting_time()

        for request in self.timeout_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_waiting_time = total_waiting_time + request.get_waiting_time()
        
        if request_num == 0:
            avg_waiting_time_per_function = 0
        else:
            avg_waiting_time_per_function = total_waiting_time / request_num

        return avg_waiting_time_per_function

    def get_avg_completion_time_slo_per_function(self, function_id):
        request_num = 0
        total_completion_time_slo = 0

        for request in self.success_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_completion_time_slo = total_completion_time_slo + request.get_completion_time_slo()

        for request in self.timeout_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_completion_time_slo = total_completion_time_slo + request.get_completion_time_slo()
        
        if request_num == 0:
            avg_completion_time_slo_per_function = 0
        else:
            avg_completion_time_slo_per_function = total_completion_time_slo / request_num

        return avg_completion_time_slo_per_function

    def get_avg_completion_time_per_function(self, function_id):
        request_num = 0
        total_completion_time = 0

        for request in self.success_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_completion_time = total_completion_time + request.get_completion_time()

        for request in self.timeout_request_record_per_function[function_id]:
            request_num = request_num + 1
            total_completion_time = total_completion_time + request.get_completion_time()
        
        if request_num == 0:
            avg_completion_time_per_function = 0
        else:
            avg_completion_time_per_function = total_completion_time / request_num

        return avg_completion_time_per_function

    def get_avg_interval_per_function(self, function_id):
        total_interval = 0
        num = 0
        for i, request in enumerate(self.total_request_record_per_function[function_id]):
            if i < len(self.total_request_record_per_function[function_id]) - 1:
                next_request = self.total_request_record_per_function[function_id][i+1]
                interval = next_request.get_invoke_time() - request.get_invoke_time()

                if interval > 0:
                    total_interval = total_interval + interval
                    num = num + 1

        if num == 0:
            avg_interval_per_function = 0
        else:
            avg_interval_per_function = total_interval / num

        return avg_interval_per_function

    def get_avg_invoke_num_per_function(self, function_id, system_time):
        if system_time == 0:
            avg_invoke_num_per_function = 0
        else:
            avg_invoke_num_per_function = self.get_total_size_per_function(function_id) / system_time

        return avg_invoke_num_per_function

    def get_avg_harvest_cpu_percent(self):
        request_num = 0
        total_harvest_cpu_percent = 0

        for request in self.success_request_record:
            cpu = request.get_cpu()
            cpu_user_defined = request.get_cpu_user_defined()

            if cpu < cpu_user_defined:
                request_num = request_num + 1
                total_harvest_cpu_percent = total_harvest_cpu_percent + (cpu_user_defined - cpu) / cpu_user_defined

        if request_num == 0:
            avg_harvest_cpu_percent = 0
        else:
            avg_harvest_cpu_percent = total_harvest_cpu_percent / request_num
        
        return avg_harvest_cpu_percent

    def get_avg_harvest_memory_percent(self):
        request_num = 0
        total_harvest_memory_percent = 0

        for request in self.success_request_record:
            memory = request.get_memory()
            memory_user_defined = request.get_memory_user_defined()

            if memory < memory_user_defined:
                request_num = request_num + 1
                total_harvest_memory_percent = total_harvest_memory_percent + (memory_user_defined - memory) / memory_user_defined

        if request_num == 0:
            avg_harvest_memory_percent = 0
        else:
            avg_harvest_memory_percent = total_harvest_memory_percent / request_num
        
        return avg_harvest_memory_percent

    def get_slo_violation_percent(self):
        request_num = 0

        for request in self.success_request_record:
            if request.get_duration_slo() > 1.15:
                request_num = request_num + 1

        return request_num / len(self.success_request_record)

    def get_acceleration_pecent(self):
        request_num = 0

        for request in self.success_request_record:
            if request.get_duration_slo() < 1:
                request_num = request_num + 1

        return request_num / len(self.success_request_record)

    def get_total_request_record(self):
        return self.total_request_record

    def get_success_request_record(self):
        return self.success_request_record

    def get_undone_request_record(self):
        return self.undone_request_record

    def get_timeout_request_record(self):
        return self.timeout_request_record

    def get_total_request_record_per_function(self, function_id):
        return self.total_request_record_per_function[function_id]

    def get_success_request_record_per_function(self, function_id):
        return self.success_request_record_per_function[function_id]

    def get_undone_request_record_per_function(self, function_id):
        return self.undone_request_record_per_function[function_id]

    def get_timeout_request_record_per_function(self, function_id):
        return self.timeout_request_record_per_function[function_id]

    def reset(self):
        self.total_request_record = []
        self.success_request_record = []
        self.undone_request_record = []
        self.timeout_request_record = []

        for function_id in self.total_request_record_per_function.keys():
            self.total_request_record_per_function[function_id] = []
            self.success_request_record_per_function[function_id] = []
            self.undone_request_record_per_function[function_id] = []
            self.timeout_request_record_per_function[function_id] = []


class ResourceUtilsRecord():
    """
    Recording of CPU and memory utilizations per server in sec
    """

    def __init__(self, n_server):
        self.n_server = n_server

        self.resource_utils_record = {}

        for i in range(self.n_server):
            server = "server{}".format(i)
            self.resource_utils_record[server] = {}
            self.resource_utils_record[server]["cpu_util"] = []
            self.resource_utils_record[server]["memory_util"] = []
            self.resource_utils_record[server]["avg_cpu_util"] = 0
            self.resource_utils_record[server]["avg_memory_util"] = 0
        
        self.resource_utils_record["avg_server"] = {}
        self.resource_utils_record["avg_server"]["cpu_util"] = []
        self.resource_utils_record["avg_server"]["memory_util"] = []
        self.resource_utils_record["avg_server"]["avg_cpu_util"] = 0
        self.resource_utils_record["avg_server"]["avg_memory_util"] = 0

    def put_resource_utils(self, server, cpu_util, memory_util):
        self.resource_utils_record[server]["cpu_util"].append(cpu_util)
        self.resource_utils_record[server]["memory_util"].append(memory_util)

    def calculate_avg_resource_utils(self):
        for i in range(self.n_server):
            server = "server{}".format(i)
            self.resource_utils_record[server]["avg_cpu_util"] = np.mean(self.resource_utils_record[server]["cpu_util"])
            self.resource_utils_record[server]["avg_memory_util"] = np.mean(self.resource_utils_record[server]["memory_util"])

        for timestep in range(len(self.resource_utils_record["server0"]["cpu_util"])):
            cpu_util_tmp_list = []
            memory_util_tmp_list = []
            
            for i in range(self.n_server):
                if i == 0:
                    cpu_util_tmp_list = []
                    memory_util_tmp_list = []

                server = "server{}".format(i)
                cpu_util_tmp_list.append(self.resource_utils_record[server]["cpu_util"][timestep])
                memory_util_tmp_list.append(self.resource_utils_record[server]["memory_util"][timestep])

            self.resource_utils_record["avg_server"]["cpu_util"].append(np.mean(cpu_util_tmp_list))
            self.resource_utils_record["avg_server"]["memory_util"].append(np.mean(memory_util_tmp_list))

        self.resource_utils_record["avg_server"]["avg_cpu_util"] = np.mean(self.resource_utils_record["avg_server"]["cpu_util"])
        self.resource_utils_record["avg_server"]["avg_memory_util"] = np.mean(self.resource_utils_record["avg_server"]["memory_util"])
    
    def get_resource_utils_record(self):
        return self.resource_utils_record

    def reset(self):
        self.resource_utils_record = {}
        
        for i in range(self.n_server):
            server = "server{}".format(i)
            self.resource_utils_record[server] = {}
            self.resource_utils_record[server]["cpu_util"] = []
            self.resource_utils_record[server]["memory_util"] = []
            self.resource_utils_record[server]["avg_cpu_util"] = 0
            self.resource_utils_record[server]["avg_memory_util"] = 0

        self.resource_utils_record["avg_server"] = {}
        self.resource_utils_record["avg_server"]["cpu_util"] = []
        self.resource_utils_record["avg_server"]["memory_util"] = []
        self.resource_utils_record["avg_server"]["avg_cpu_util"] = 0
        self.resource_utils_record["avg_server"]["avg_memory_util"] = 0


class Cache():
    """
    Temporarily keeps function instances alive to enable warm-start
    """

    def __init__(self, keep_alive_window):
        self.keep_alive_window = keep_alive_window
        self.cache = {}
    
    def step(self, request_list, system_time):
        # Terminate function instances that exceed keep alive window
        terminate_list = []
        for function_id in self.cache.keys():
            cache_per_function = self.cache[function_id]
            for request in cache_per_function:
                if system_time - request.get_done_time() > self.keep_alive_window:
                    terminate_list.append(request)

        self.delete_requests(terminate_list)

        # Add new alive function instances
        self.put_requests(request_list)

    def reset(self):
        self.cache = {}

    def delete_requests(self, request_list):
        if isinstance(request_list, Request):
            request = request_list
            function_id = request.get_function_id()
            self.cache[function_id].remove(request)
        else:
            for request in request_list:
                function_id = request.get_function_id()
                self.cache[function_id].remove(request)
    
    def put_requests(self, request_list):
        if isinstance(request_list, Request):
            request = request_list
            function_id = request.get_function_id()
            if function_id not in self.cache:
                self.cache[function_id] = []
            self.cache[function_id].append(request)
        else:
            for request in request_list:
                function_id = request.get_function_id()
                if function_id not in self.cache:
                    self.cache[function_id] = []
                self.cache[function_id].append(request)

    def get_cache(self):
        return self.cache

    def get_resources_in_use(self):
        cpu_in_use = 0
        memory_in_use = 0
        for function_id in self.cache.keys():
            for request in self.cache[function_id]:
                cpu_in_use = cpu_in_use + request.get_cpu()
                memory_in_use = memory_in_use + request.get_memory()
            
        return cpu_in_use, memory_in_use

    def get_n_alive_instances(self, function_id):
        if function_id in self.cache:
            n_alive_instances = len(self.cache[function_id])
        else:
            n_alive_instances = 0

        return n_alive_instances

    def get_warm_start_instance(self, request_to_schedule):
        function_id = request_to_schedule.get_function_id()
        warm_start_request_to_remove = None
        min_done_time = 114514

        if self.get_n_alive_instances(function_id) > 0:
            for request in self.cache[function_id]:
                if request.get_cpu() > request_to_schedule.get_cpu() and \
                request.get_memory() > request_to_schedule.get_memory() and \
                min_done_time > request.get_done_time():
                    warm_start_request_to_remove = request
                    min_done_time = request.get_done_time()

        return warm_start_request_to_remove
    
    def get_cold_start_queue(self):
        cold_start_queue = queue.PriorityQueue()
        for function_id in self.cache.keys():
            for request in self.cache[function_id]:
                cold_start_queue.put(Prioritize(-request.get_done_time(), request))

        return cold_start_queue


class Registry():
    """
    Registry processes functions
    """
    
    def __init__(self):
        self.registry = []
        
    def step(self, system_time):
        # Update registry
        request_done_or_timeout_list = []
        num_timeout = 0
        
        for request in self.registry:
            status = request.step(in_registry=True, system_time=system_time)
            # Remove finished functions if any
            if status == "success": # Done 
                request_done_or_timeout_list.append(request)
            elif status == "timeout": # Timeout 
                request_done_or_timeout_list.append(request)
                num_timeout = num_timeout + 1

        self.delete_requests(request_done_or_timeout_list)
        return request_done_or_timeout_list, num_timeout
                
    def reset(self):
        self.registry = []            
    
    def delete_requests(self, request_list):
        if isinstance(request_list, Request):
            request = request_list
            self.registry.remove(request)
        else:
            for request in request_list:
                self.registry.remove(request)
    
    def put_requests(self, request_list):
        if isinstance(request_list, Request):
            request = request_list
            self.registry.append(request)
        else:
            for request in request_list:
                self.registry.append(request)
                    
    def get_registry(self):
        return self.registry
                
    def get_current_size(self):
        return len(self.registry)
    
    def get_resources_in_use(self):
        cpu_in_use = 0
        memory_in_use = 0
        for request in self.registry:
            cpu_in_use = cpu_in_use + request.get_cpu()
            memory_in_use = memory_in_use + request.get_memory()
            
        return cpu_in_use, memory_in_use
    

class Queue():
    """
    Queue temporarily stores function invocations
    """
    
    def __init__(self):
        self.queue = []
        
    def step(self, system_time):
        # Update queue
        request_timeout_list = []
        num_timeout = 0
        
        for request in self.queue:
            status = request.step(in_registry=True, system_time=system_time)
            # Remove finished functions if any
            if status == "timeout": # Timeout requests
                request_timeout_list.append(request)
                num_timeout = num_timeout + 1

        self.delete_requests(request_timeout_list)
        return request_timeout_list, num_timeout
    
    def reset(self):
        self.queue = []
    
    def delete_requests(self, request_list):
        if isinstance(request_list, Request):
            request = request_list
            self.queue.remove(request)
        else:
            for request in request_list:
                self.queue.remove(request)

    def put_requests(self, request_list):
        if isinstance(request_list, Request):
            request = request_list
            self.queue.append(request)
        else:
            for request in request_list:
                self.queue.append(request)
                    
    def get_queue(self):
        return self.queue
                
    def get_current_size(self):
        return len(self.queue)


class Manager():
    """
    A manager that manages function life cycles in each Server
    """
    
    def __init__(
        self, 
        user_cpu, 
        user_memory,
        keep_alive_window
    ):
        self.registry = Registry()
        self.cache = Cache(keep_alive_window=keep_alive_window)
        self.queue = Queue()
        self.user_cpu = user_cpu
        self.user_memory = user_memory

    def put_requests(self, request):
        if self.check_availability(request) is True and self.queue.get_current_size() == 0:
            # Determine if it's a cold start or warm start before entering in registry
            self.check_cold_start(request)
            self.registry.put_requests(request)
        else:
            self.queue.put_requests(request)

    def get_user_cpu(self):
        return self.user_cpu

    def get_user_memory(self):
        return self.user_memory

    def get_resources_total(self):
        return self.user_cpu, self.user_memory
    
    def get_registry_resources_in_use(self):
        registry_cpu_in_use, registry_memory_in_use = self.registry.get_resources_in_use()
        return registry_cpu_in_use, registry_memory_in_use

    def get_cache_resources_in_use(self):
        cache_cpu_in_use, cache_memory_in_use = self.cache.get_resources_in_use()
        return cache_cpu_in_use, cache_memory_in_use
    
    # Calculate available resources
    def get_resources_available(self):
        cpu_in_use, memory_in_use = self.get_registry_resources_in_use()
        user_cpu, user_memory = self.get_resources_total()
        cpu_available = user_cpu - cpu_in_use
        memory_available = user_memory - memory_in_use
        return cpu_available, memory_available

    # Calculate resource utilizations
    def get_resource_utils(self):
        cpu_in_use, memory_in_use = self.get_registry_resources_in_use()
        user_cpu, user_memory = self.get_resources_total()
        cpu_util = cpu_in_use / user_cpu
        memory_util = memory_in_use / user_memory

        return cpu_util, memory_util

    # Check whether a given request is available
    def check_availability(self, request_to_schedule):
        cpu_available, memory_available = self.get_resources_available()
        if cpu_available >= request_to_schedule.get_cpu() and memory_available >= request_to_schedule.get_memory():
            return True
        else:
            return False

    # Check if a request is cold start
    def check_cold_start(self, request_to_schedule):
        # Check if there will be a warm start by any chances
        warm_start_request_to_remove = self.cache.get_warm_start_instance(request_to_schedule)

        # Check if any other instances in cache should be terminated when cold start
        if warm_start_request_to_remove is None:
            cold_start_queue = self.cache.get_cold_start_queue()
            registry_cpu_in_use, registry_memory_in_use = self.get_registry_resources_in_use()
            cache_cpu_in_use, cache_memory_in_use = self.get_cache_resources_in_use()

            # Terminate instances to release resources
            terminate_list = []
            potential_available_cpu, potential_available_memory = 0, 0
            while request_to_schedule.get_cpu() + registry_cpu_in_use + cache_cpu_in_use - potential_available_cpu > self.user_cpu or \
                request_to_schedule.get_memory() + registry_memory_in_use + cache_memory_in_use - potential_available_memory > self.user_memory:
                request = cold_start_queue.get().item
                terminate_list.append(request)
                potential_available_cpu = potential_available_cpu + request.get_cpu()
                potential_available_memory = potential_available_memory + request.get_memory()

            self.cache.delete_requests(terminate_list)
            request_to_schedule.set_is_cold_start(True)
        # Warm start
        else:
            self.cache.delete_requests(warm_start_request_to_remove)
            request_to_schedule.set_is_cold_start(False)
    
    # Try to import queue to registry if available
    def try_import_queue_to_registry(self):
        request_ready_queue = queue.PriorityQueue()
        request_to_remove_list = []
        for request in self.queue.get_queue():
            if self.check_availability(request) is True:
                request_ready_queue.put(Prioritize(-request.get_waiting_time(), request))
        
        while request_ready_queue.empty() is False:
            request = request_ready_queue.get().item
            if self.check_availability(request):
                # Determine if it's a cold start or warm start before entering in registry
                self.check_cold_start(request)
                self.registry.put_requests(request)
                request_to_remove_list.append(request)

        self.queue.delete_requests(request_to_remove_list)

    def step(self, system_time):
        # Update registry
        registry_request_list, num_timeout_registry = self.registry.step(system_time)
        
        # Update cache
        self.cache.step(request_list=registry_request_list, system_time=system_time)
        
        # Update queue
        queue_request_list, num_timeout_queue = self.queue.step(system_time)
        
        # Try to import queue if available   
        self.try_import_queue_to_registry()

        # Aggregate results
        request_to_update_list = registry_request_list + queue_request_list
        num_timeout = num_timeout_registry + num_timeout_queue

        return request_to_update_list, num_timeout

    def reset(self):
        self.registry.reset()
        self.cache.reset()
        self.queue.reset()
                        

class Server():
    """
    A physical server which contains a collection of CPU and memory resource respectively
    """

    def __init__(
        self, 
        user_cpu, 
        user_memory,
        keep_alive_window
    ):
        self.server_id = uuid.uuid1()

        self.user_cpu = user_cpu
        self.user_memory = user_memory
        self.keep_alive_window = keep_alive_window

        self.manager = Manager(
            user_cpu=self.user_cpu,
            user_memory=self.user_memory,
            keep_alive_window=self.keep_alive_window
        )

    def check_availability(self, request):
        return self.manager.check_availability(request)

    def put_requests(self, request):
        self.manager.put_requests(request)

    def step(self, system_time):
        request_list, num_timeout = self.manager.step(system_time)
        return request_list, num_timeout

    def reset(self):
        self.manager.reset()


class Cluster():
    """
    A cluster that consists of multiple physical servers
    """

    def __init__(
        self,
        cluster_size,
        schedule_step_size,
        user_cpu_per_server,
        user_memory_per_server,
        keep_alive_window
    ):
        self.cluster_size = cluster_size
        self.schedule_step_size = schedule_step_size
        self.user_cpu_per_server = user_cpu_per_server
        self.user_memory_per_server = user_memory_per_server
        self.keep_alive_window = keep_alive_window

        self.server_pool = []
        for i in range(self.cluster_size):
            server = Server(
                user_cpu=self.user_cpu_per_server,
                user_memory=self.user_memory_per_server,
                keep_alive_window=self.keep_alive_window
            )
            self.server_pool.append(server)

    def get_cluster_size(self):
        return self.cluster_size

    def get_server_pool(self):
        return self.server_pool

    def get_total_in_use_resources(self):
        total_in_use_cpu = 0
        total_in_use_memory = 0

        for server in self.server_pool:
            in_use_cpu, in_use_memory = server.manager.get_registry_resources_in_use()

            total_in_use_cpu = total_in_use_cpu + in_use_cpu
            total_in_use_memory = total_in_use_memory + in_use_memory

        return total_in_use_cpu, total_in_use_memory

    def get_total_available_resources(self):
        total_available_cpu = 0
        total_available_memory = 0

        for server in self.server_pool:
            available_cpu, available_memory = server.manager.get_resources_available()
            total_available_cpu = total_available_cpu + available_cpu
            total_available_memory = available_memory + available_memory

        return total_available_cpu, total_available_memory

    def get_n_alive_instances(self, function_id):
        n_alive_instances = 0
        for server in self.server_pool:
            n_alive_instances = n_alive_instances + server.manager.get_n_alive_instances(function_id)

        return n_alive_instances

    def step(self, system_time):
        total_request_list = []
        total_num_timeout = 0
        for server in self.server_pool:
            request_list, num_timeout = server.step(system_time)
            total_request_list = total_request_list + request_list
            total_num_timeout = total_num_timeout + num_timeout

        return total_request_list, total_num_timeout

    # A hashing-greedy algorithm based on OpenWhisk load balancer
    # https://github.com/apache/openwhisk/blob/master/core/controller/src/main/scala/org/apache/openwhisk/core/loadBalancer/ShardingContainerPoolBalancer.scala
    def schedule(self, request):
        is_scheduled = False
        home_server_index = request.profile.get_hash_value() % self.cluster_size

        # Seek in steps
        i = 0
        while i <= self.cluster_size - 1:
            home_server = self.server_pool[home_server_index]
            if home_server.check_availability(request) is True:
                home_server.put_requests(request)
                is_scheduled = True
                break
            else:
                home_server_index = (home_server_index + self.schedule_step_size) % self.cluster_size
                i = i + 1

        # If none of servers has available resources, randome pick one
        if is_scheduled is False:
            random_index = random.randint(0, self.cluster_size - 1)
            random_server = self.server_pool[random_index]
            random_server.put_requests(request)

    def reset(self):
        for server in self.server_pool:
            server.reset()
    
    
class Profile():
    """
    Record settings of any functions that submitted to FaaSEnv
    """
    
    def __init__(
        self, 
        function_profile
    ):
        self.function_profile = function_profile
        self.default_function_profile = cp.deepcopy(function_profile)

    def put_function(self, function):
        function_id = function.get_function_id()
        self.function_profile[function_id] = function

    def get_function_profile(self):
        return self.function_profile

    def get_size(self):
        return len(self.function_profile)

    def reset(self):
        self.function_profile = cp.deepcopy(self.default_function_profile)
        
        
class EventPQ():
    """
    A priority queue of events, dictates which and when function will be invoked
    """
    
    def __init__(
        self, 
        pq,
        max_timestep
    ):
        self.pq = pq
        self.default_pq = cp.deepcopy(pq)
        self.max_timestep = max_timestep
        
    def get_event(self):
        if self.is_empty() is True:
            return None, None
        else:
            (timestep, counter, function_id) = heapq.heappop(self.pq)
            return timestep, function_id
    
    def get_current_size(self):
        return len(self.pq)

    def get_total_size(self):
        return len(self.default_pq)

    def get_max_timestep(self):
        return self.max_timestep

    def is_empty(self):
        if len(self.pq) == 0:
            return True
        else:
            return False

    def reset(self):
        self.pq = cp.deepcopy(self.default_pq)


class SystemTime():
    """
    Time module for the whole environment
    """
    
    def __init__(self, default_interval):
        self.system_runtime = 0
        self.default_interval = default_interval

    def get_system_runtime(self):
        return self.system_runtime

    def get_default_interval(self):
        return self.default_interval

    def step(self):
        self.system_runtime = self.system_runtime + self.default_interval

    def reset(self):
        self.system_runtime = 0
