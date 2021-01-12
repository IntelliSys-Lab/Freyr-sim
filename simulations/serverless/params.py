#
# Experimental settings
#

workload_type = "azure"
max_workload = 1
max_episode = 1000
hidden_dims = [64, 32, 16]
learning_rate = 0.001
discount_factor = 1
ppo_clip = 0.2
ppo_epoch = 4
value_loss_coef = 0.5
entropy_coef = 0.01
model_save_path = "ckpt/best_model.pth"
max_timestep = 60
max_function = 110
max_server = 10
cluster_size = 10
user_cpu_per_server = 64
user_memory_per_server = 64
keep_alive_window_per_server = 10
cpu_cap_per_function = 16
memory_cap_per_function = 16
interval = 1
timeout_penalty = 600
