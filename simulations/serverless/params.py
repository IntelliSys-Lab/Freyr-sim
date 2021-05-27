# Environment parameters
CLUSTER_SIZE = 30
USER_CPU_PER_SERVER = 8
USER_MEMORY_PER_SERVER = 32
CPU_CAP_PER_FUNCTION = 8
MEMORY_CAP_PER_FUNCTION = 8
MEMORY_MB_LIMIT = 512
KEEP_ALIVE_WINDOW = 10
ENV_INTERVAL_LIMIT = 1
FAIL_PENALTY = 0

# Training parameters
AZURE_FILE_PATH = "azurefunctions-dataset2019/"
EXP_TRAIN = [i for i in range(250)]
EXP_EVAL = [0]
MAX_EPISODE_TRAIN = 1
MAX_EPISODE_EVAL = 1
STATE_DIM = 11
ACTION_DIM = 1
HIDDEN_DIMS = [32, 16]
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 1
PPO_CLIP = 0.2
PPO_EPOCH = 4
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
MODEL_SAVE_PATH = "ckpt/"
# MODEL_NAME = "max_episode.ckpt"
MODEL_NAME = "min_avg_duration_slo.ckpt"
