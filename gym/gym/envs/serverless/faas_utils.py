import numpy as np
import copy as cp
import uuid
import queue
import random
import functools


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


class Application():
    """
    Application used by FaaSEnv
    """
    def __init__(self, functions):
        if functions[0].application_id != None:
            self.application_id = functions[0].application_id
        else:
            self.application_id = uuid.uuid1()
        
        self.function_ids = []
            
        for function in functions:
            function.set_application_id(self.application_id)
            self.function_ids.append(function.function_id)
            

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
        self.application_id = self.params.application_id
        self.resource_adjust_direction = [0, 0] # [cpu, memory]
    
    def set_application_id(self, application_id):
        self.application_id = application_id
        
    def set_function(self, cpu=1, memory=1):
        self.cpu = cpu
        self.memory = memory
        
        # Calculate duration
        self.duration = self.params.ideal_duration * np.max([self.params.ideal_cpu, self.cpu])/self.cpu * np.max([self.params.ideal_memory, self.memory])/self.memory
    
    def get_function_id(self):
        return self.function_id

    def get_hash_value(self):
        return self.hash_value

    def get_cpu(self):
        return self.cpu

    def get_memory(self):
        return self.memory

    def set_resource_adjust(self, resource, adjust):
        # Adjust resources
        next_cpu = self.cpu
        next_memory = self.memory
        
        if resource == 0:
            if adjust == 1:
                if next_cpu < self.params.cpu_cap_per_function:
                    next_cpu = next_cpu + 1
            else:
                if next_cpu > self.params.cpu_least_hint:
                    next_cpu = next_cpu - 1
        else:
            if adjust == 1:
                if next_memory < self.params.memory_cap_per_function:
                    next_memory = next_memory + 1
            else:
                if next_memory > self.params.memory_least_hint:
                    next_memory = next_memory - 1
            
        self.set_function(next_cpu, next_memory)
        
        # Set resource adjust direction if not touched yet
        if self.resource_adjust_direction[resource] == 0:
            self.resource_adjust_direction[resource] = adjust

    def validate_resource_adjust(self, resource, adjust): 
        if resource == 0:
            if adjust == 1:
                if self.cpu == self.params.cpu_cap_per_function: # Implicit invalid action: reach cpu cap
                    return False 
            else:
                if self.cpu == self.params.cpu_least_hint: # Implicit invalid action: reach cpu least hint
                    return False 
        else:
            if adjust == 1:
                if self.memory == self.params.memory_cap_per_function: # Implicit invalid action: reach memory cap
                    return False 
            else:
                if self.memory == self.params.memory_least_hint: # Implicit invalid action: reach memory least hint
                    return False 
        
        if self.resource_adjust_direction[resource] == 0: # Not touched yet
            return True   
        else:
            if self.resource_adjust_direction[resource] == adjust: # Correct direction as usual
                return True
            else: # Implicit invalid action: wrong direction
                return False
            
    def reset_resource_adjust_direction(self):
        self.resource_adjust_direction = [0, 0]


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

    def get_completion_time(self):
        return self.progress + self.waiting
    
    def get_slow_down(self):
        return self.get_completion_time() / self.profile.duration

    def get_status(self):
        return self.status

    def get_is_cold_start(self):
        return self.is_cold_start
    
    def step(self, system_time, in_registry):
        # In Registry
        if in_registry is True:
            self.progress = self.progress + 1
            
            # Check status
            if self.progress + self.waiting >= self.profile.params.timeout:
                self.status = "timeout" # Timeout
                self.done_time = system_time
            
            if self.progress >= self.profile.duration:
                self.status = "success" # Done
                self.done_time = system_time
        # In Queue
        else:
            self.waiting = self.waiting + 1
        
            # Check status
            if self.progress + self.waiting >= self.profile.params.timeout:
                self.status = "timeout" # Timeout
                self.done_time = system_time
            
        return self.status


class RequestRecord():
    """
    Recording of either done or undone requests per Function
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

        for function in function_profile:
            function_id = function.get_function_id()
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

    def update_request(self, done_request_list):
        for request in done_request_list:
            function_id = request.get_function_id()
            status = request.get_status()

            if status == "success":
                self.success_request_record.append(request)
                self.success_request_record_per_function[function_id].append(request)
            elif status == "timeout":
                self.timeout_request_record.append(request)
                self.timeout_request_record_per_function[function_id].append(request)
            
            self.undone_request_record.remove(request)
            self.undone_request_record_per_function[function_id].remove(request)

    def get_total_size(self):
        total_size = len(self.total_request_record)
        return total_size

    def get_undone_size(self):
        undone_size = len(self.undone_request_record)
        return undone_size

    def get_success_size(self):
        success_size = len(self.success_request_record)
        return success_size

    def get_timeout_size(self):
        timeout_size = len(self.timeout_request_record)
        return timeout_size

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

    def get_avg_interval(self, system_time):
        if system_time == 0:
            avg_interval = 0
        else:
            avg_interval = len(self.total_request_record) / system_time

        return avg_interval

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

    def get_avg_interval_per_function(self, system_time, function_id):
        if system_time == 0:
            avg_interval_per_function = 0
        else:
            avg_interval_per_function = len(self.total_request_record_per_function[function_id]) / system_time

        return avg_interval_per_function

    def get_is_cold_start_per_function(self, function_id):
        if self.get_total_size_per_function(function_id) == 0:
            is_cold_start = True
        else:
            is_cold_start = self.total_request_record_per_function[function_id][-1].get_is_cold_start()

        if is_cold_start is False:
            return 0
        else:
            return 1

    def get_total_request_record(self):
        return self.total_request_record

    def get_success_request_record(self):
        return self.success_request_record

    def get_undone_request_record(self):
        return self.undone_request_record

    def get_timeout_request_record(self):
        return self.timeout_request_record

    def get_total_request_record_per_function(self):
        return self.total_request_record

    def get_success_request_record_per_function(self):
        return self.success_request_record

    def get_undone_request_record_per_function(self):
        return self.undone_request_record

    def get_timeout_request_record_per_function(self):
        return self.timeout_request_record

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


class Cache():
    """
    Temporarily keeps function instances alive to enable warm-start
    """

    def __init__(
        self,
        keep_alive_window=60
    ):
        self.keep_alive_window = keep_alive_window
        self.cache = []
    
    def step(self, request_list, system_time):
        # Terminate function instances that exceed keep alive window
        terminate_list = []
        for request in self.cache:
            if system_time - request.get_invoke_time() > self.keep_alive_window:
                terminate_list.append(request)

        self.delete_requests(terminate_list)

        # Add new alive function instances
        self.put_requests(request_list)

    def reset(self):
        self.cache = []

    def delete_requests(self, request_list):
        if isinstance(request_list, Request):
            request = request_list
            self.cache.remove(request)
        else:
            for request in request_list:
                self.cache.remove(request)
    
    def put_requests(self, request_list):
        if isinstance(request_list, Request):
            request = request_list
            self.cache.append(request)
        else:
            for request in request_list:
                self.cache.append(request)

    def get_cache(self):
        return self.cache

    def get_resources_in_use(self):
        cpu_in_use = 0
        memory_in_use = 0
        for request in self.cache:
            cpu_in_use = cpu_in_use + request.get_cpu()
            memory_in_use = memory_in_use + request.get_memory()
            
        return cpu_in_use, memory_in_use


class Registry():
    """
    Where functions processed, used by FaaSEnv
    """
    
    def __init__(
        self, 
        registry_size=100
    ):
        self.registry_size = registry_size
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
                
    def get_size(self):
        return self.registry_size
    
    def get_current_len(self):
        return len(self.registry)
    
    def get_resources_in_use(self):
        cpu_in_use = 0
        memory_in_use = 0
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
                
    def get_size(self):
        return self.size
                
    def get_current_len(self):
        return len(self.queue)


class ResourceManager():
    """
    A resource manager to manage CPU and memory resources, each Server posseses one
    """
    
    def __init__(
        self, 
        user_cpu=8, 
        user_memory=8,
        keep_alive_window=60
    ):
        self.registry = Registry()
        self.cache = Cache(keep_alive_window=keep_alive_window)
        self.queue = Queue()
        self.user_cpu = user_cpu
        self.user_memory = user_memory

    def put_requests(self, system_time, request):
        if self.check_availability(request) is True and self.queue.get_current_len() == 0:
            # Determine if it's a cold start or warm start before entering in registry
            self.check_cold_start(system_time, request)
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
        cpu_available = self.user_cpu - cpu_in_use
        memory_available = self.user_memory - memory_in_use
        return cpu_available, memory_available
    
    # Check whether a given request is available
    def check_availability(self, request_to_schedule):
        cpu_available, memory_available = self.get_resources_available()
        if cpu_available >= request_to_schedule.get_cpu() and memory_available >= request_to_schedule.get_memory():
            return True
        else:
            return False

    # Check if a request is cold start
    def check_cold_start(self, system_time, request_to_schedule):
        warm_start_request_to_remove = None
        cold_start_queue = queue.PriorityQueue()

        # Retrieve available instance if warm start
        for request in self.cache.get_cache():
            if request.get_function_id() == request_to_schedule.get_function_id() and \
                request.get_cpu() == request_to_schedule.get_cpu() and \
                    request.get_memory() == request_to_schedule.get_memory():
                    warm_start_request_to_remove = request
                    break
            else:
                alive_time = system_time - request.get_invoke_time()
                cold_start_queue.put(Prioritize(-alive_time, request))

        # Check if any other instances in cache should be terminated when cold start
        if warm_start_request_to_remove is None:
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
    def try_import_queue_to_registry(self, system_time):
        request_ready_queue = queue.PriorityQueue()
        request_to_remove_list = []
        for request in self.queue.get_queue():
            if self.check_availability(request) is True:
                request_ready_queue.put(Prioritize(-request.get_waiting_time(), request))

        while request_ready_queue.empty() is False:
            request = request_ready_queue.get().item
            if self.check_availability(request):
                # Determine if it's a cold start or warm start before entering in registry
                self.check_cold_start(system_time, request)
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
        self.try_import_queue_to_registry(system_time)

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
        user_cpu=8, 
        user_memory=8,
        keep_alive_window=60
    ):
        self.server_id = uuid.uuid1()

        self.user_cpu = user_cpu
        self.user_memory = user_memory
        self.keep_alive_window = keep_alive_window

        self.resource_manager = ResourceManager(
            user_cpu=self.user_cpu,
            user_memory=self.user_memory,
            keep_alive_window=self.keep_alive_window
        )

    def check_availability(self, request):
        return self.resource_manager.check_availability(request)

    def put_requests(self, system_time, request):
        self.resource_manager.put_requests(system_time, request)

    def step(self, system_time):
        request_list, num_timeout = self.resource_manager.step(system_time)
        return request_list, num_timeout

    def reset(self):
        self.resource_manager.reset()


class Cluster():
    """
    A cluster that consists of multiple physical servers
    """

    def __init__(
        self,
        cluster_size=10,
        schedule_step_size=3,
        user_cpu_per_server=8,
        user_memory_per_server=8,
        keep_alive_window_per_server=60
    ):
        self.cluster_size = 10
        self.schedule_step_size = schedule_step_size
        self.user_cpu_per_server = user_cpu_per_server
        self.user_memory_per_server = user_memory_per_server
        self.keep_alive_window_per_server = keep_alive_window_per_server

        self.server_pool = []
        for i in range(self.cluster_size):
            server = Server(
                user_cpu=self.user_cpu_per_server,
                user_memory=self.user_memory_per_server,
                keep_alive_window=self.keep_alive_window_per_server
            )
            self.server_pool.append(server)

    def get_cluster_size(self):
        return self.cluster_size

    def get_server_pool(self):
        return self.server_pool

    def step(self, system_time):
        total_request_list = []
        total_num_timeout = 0
        for server in self.server_pool:
            request_list, num_timeout = server.step(system_time)
            total_request_list = total_request_list + request_list
            total_num_timeout = total_num_timeout + num_timeout

        return total_request_list, total_num_timeout

    # A hashing-greedy algorithm based on OpenWhisk scheduling algorithm
    # Reference: 
    # https://github.com/apache/openwhisk/blob/master/core/controller/src/main/scala/org/apache/openwhisk/core/loadBalancer/ShardingContainerPoolBalancer.scala
    def schedule(self, system_time, request_list):
        for request in request_list:
            is_scheduled = False
            home_server_index = request.profile.get_hash_value() % self.cluster_size

            # Seek in step
            i = 0
            while i <= self.cluster_size - 1:
                home_server = self.server_pool[home_server_index]
                if home_server.check_availability(request) is True:
                    home_server.put_requests(system_time, request)
                    is_scheduled = True
                    break
                else:
                    home_server_index = (home_server_index + self.schedule_step_size) % self.cluster_size
                    i = i + 1

            # If none of servers has available resources, randome pick one
            if is_scheduled is False:
                random_index = random.randint(0, self.cluster_size - 1)
                random_server = self.server_pool[random_index]
                random_server.put_requests(system_time, request)

    def reset(self):
        for server in self.server_pool:
            server.reset()
    
    
class Profile():
    """
    Record settings of any functions that submitted to FaaSEnv
    """
    
    def __init__(
        self, 
        function_profile, 
        application_profile
    ):
        self.application_profile = application_profile
        self.function_profile = function_profile
        self.default_function_profile = function_profile
        
    def put_application(self, application):
        self.application_profile.append(application)
        
    def put_function(self, function):
        self.function_profile.append(function)

    def get_function_profile(self):
        return self.function_profile

    def get_application_profile_size(self):
        return self.application_profile

    def get_function_profile_size(self):
        return len(self.function_profile)

    def get_application_profile_size(self):
        return len(self.application_profile)
        
    def reset(self):
        self.function_profile = cp.deepcopy(self.default_function_profile)
        
        
class Timetable():
    """
    Dictate which and when functions will be invoked by FaaSEnv
    """
    
    def __init__(
        self, 
        timetable=[]
    ):
        self.timetable = timetable
        self.size = len(self.timetable)
        
    def put_timestep(self, row):
        self.timetable.append(row)
        
    def get_timestep(self, timestep):
        if timestep >= len(self.timetable):
            return None
        else:
            return self.timetable[timestep]
    
    def get_size(self):
        return self.size

    