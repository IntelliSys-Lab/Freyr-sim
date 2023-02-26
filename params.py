import torch.nn as nn

# Environment parameters
CLUSTER_SIZE = 30
USER_CPU_PER_SERVER = 32
USER_MEMORY_PER_SERVER = 32
CPU_LEAST_PER_FUNCTION = 1
CPU_CAP_PER_FUNCTION = 8
MEMORY_LEAST_PER_FUNCTION = 1
MEMORY_CAP_PER_FUNCTION = 8
MEMORY_MB_LIMIT = 1024
SAFEGUARD_THRESHOLD = 0.9
KEEP_ALIVE_WINDOW = 10
ENV_INTERVAL_LIMIT = 1
FAIL_PENALTY = 0
FUNCTION_CONFIG = {
    0: {
        "ideal_memory": 8,
        "memory_user_defined": 1,
        "ideal_cpu": 8,
        "cpu_user_defined": 1,
        "max_duration": 100,
        "min_duration": 20,
        "cold_start_time": 1,
        "timeout": 114514
    },
    1: {
        "ideal_memory": 8,
        "memory_user_defined": 1,
        "ideal_cpu": 8,
        "cpu_user_defined": 1,
        "max_duration": 200,
        "min_duration": 40,
        "cold_start_time": 1,
        "timeout": 114514
    },
    2: {
        "ideal_memory": 8,
        "memory_user_defined": 1,
        "ideal_cpu": 8,
        "cpu_user_defined": 1,
        "max_duration": 160,
        "min_duration": 40,
        "cold_start_time": 1,
        "timeout": 114514
    },
    3: {
        "ideal_memory": 8,
        "memory_user_defined": 1,
        "ideal_cpu": 8,
        "cpu_user_defined": 1,
        "max_duration": 600,
        "min_duration": 100,
        "cold_start_time": 1,
        "timeout": 114514
    },
    4: {
        "ideal_memory": 8,
        "memory_user_defined": 1,
        "ideal_cpu": 8,
        "cpu_user_defined": 1,
        "max_duration": 300,
        "min_duration": 60,
        "cold_start_time": 1,
        "timeout": 114514
    },
    5: {
        "ideal_memory": 8,
        "memory_user_defined": 1,
        "ideal_cpu": 8,
        "cpu_user_defined": 1,
        "max_duration": 250,
        "min_duration": 50,
        "cold_start_time": 1,
        "timeout": 114514
    },
    6: {
        "ideal_memory": 8,
        "memory_user_defined": 1,
        "ideal_cpu": 8,
        "cpu_user_defined": 1,
        "max_duration": 180,
        "min_duration": 60,
        "cold_start_time": 1,
        "timeout": 114514
    },
    7: {
        "ideal_memory": 8,
        "memory_user_defined": 1,
        "ideal_cpu": 8,
        "cpu_user_defined": 1,
        "max_duration": 360,
        "min_duration": 60,
        "cold_start_time": 1,
        "timeout": 114514
    },
    8: {
        "ideal_memory": 8,
        "memory_user_defined": 1,
        "ideal_cpu": 8,
        "cpu_user_defined": 1,
        "max_duration": 550,
        "min_duration": 150,
        "cold_start_time": 5,
        "timeout": 114514
    },
    9: {
        "ideal_memory": 8,
        "memory_user_defined": 1,
        "ideal_cpu": 8,
        "cpu_user_defined": 1,
        "max_duration": 80,
        "min_duration": 10,
        "cold_start_time": 1,
        "timeout": 114514
    },
}

# Training parameters
AZURE_FILE_PATH = "azurefunctions-dataset2019/"
EXP_TRAIN = [2, 4, 6, 8, 10]
# EXP_TRAIN = [0]
EXP_EVAL = [0]
MAX_EPISODE_TRAIN = 500
MAX_EPISODE_EVAL = 1
USE_INCREMENTAL = True
USE_ATTN = True
STATE_DIM = 10
EMBED_DIM = 32
NUM_HEADS = 1
ACTION_DIM = 64
HIDDEN_DIMS = [64, 64]
LEARNING_RATE = 3e-4
ACTIVATION = nn.Tanh()
DISCOUNT_FACTOR = 0.99
PPO_CLIP = 0.2
PPO_EPOCH = 4
VALUE_LOSS_COEF = 0.05
ENTROPY_COEF = 0.01
MODEL_SAVE_PATH = "ckpt/"
MODEL_NAME = "max_reward.ckpt"


class EnvParameters():
    """
    Parameters used for generating FaaSEnv
    """
    def __init__(
        self,
        cluster_size,
        user_cpu_per_server,
        user_memory_per_server,
        keep_alive_window,
        cpu_cap_per_function,
        memory_cap_per_function,
        memory_mb_limit,
        interval,
        fail_penalty
    ):
        self.cluster_size = cluster_size
        self.user_cpu_per_server = user_cpu_per_server
        self.user_memory_per_server = user_memory_per_server
        self.keep_alive_window = keep_alive_window
        self.cpu_cap_per_function = cpu_cap_per_function
        self.memory_cap_per_function = memory_cap_per_function
        self.memory_mb_limit = memory_mb_limit
        self.interval = interval
        self.fail_penalty = fail_penalty
        
class FunctionParameters():
    """
    Parameters used for generating Function
    """
    def __init__(
        self,
        ideal_cpu,
        ideal_memory,
        max_duration,
        min_duration,
        cpu_cap_per_function,
        memory_cap_per_function,
        # memory_mb_limit,
        cpu_least_hint,
        memory_least_hint,
        cpu_user_defined,
        memory_user_defined,
        timeout,
        hash_value,
        cold_start_time,
        k,
        function_id
    ):
        self.ideal_cpu = ideal_cpu
        self.ideal_memory = ideal_memory
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.cpu_cap_per_function = cpu_cap_per_function
        self.memory_cap_per_function = memory_cap_per_function
        # self.memory_mb_limit = memory_mb_limit
        self.cpu_least_hint = cpu_least_hint
        self.memory_least_hint = memory_least_hint
        self.cpu_user_defined = cpu_user_defined
        self.memory_user_defined = memory_user_defined
        self.timeout = timeout
        self.function_id = function_id
        self.hash_value = hash_value
        self.cold_start_time = cold_start_time
        self.k = k


class WorkloadParameters():
    """
    Parameters used for workload configuration
    """
    def __init__(
        self,
        azure_file_path,
        exp_id
    ):
        self.azure_file_path = azure_file_path
        self.exp_id = exp_id

        
class EventPQParameters():
    """
    Parameters used for generating EventPQ
    """
    def __init__(
        self,
        azure_invocation_traces=None
    ):
        self.azure_invocation_traces = azure_invocation_traces
        