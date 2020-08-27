import sys
sys.path.append("../../gym")
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.serverless.faas_params import EnvParameters
from logger import Logger
from plotter import Plotter
from workflow_generator import WorkflowGenerator


#
# Encode sequential resource changes into discrete actions
#
def encode_action(function_profile, resource_adjust_list):
    actions = []
    
    for function in function_profile:
        for key in resource_adjust_list.keys():
            if function.function_id == key:
                index = function_profile.index(function)
                
                if resource_adjust_list[key][0] != -1:
                    adjust_cpu = index*4 + resource_adjust_list[key][0]
                    actions.append(adjust_cpu)
                if resource_adjust_list[key][1] != -1:
                    adjust_memory = index*4 + resource_adjust_list[key][1]
                    actions.append(adjust_memory)
                    
    return actions
                    

# Set up logger
logger_wrapper = Logger("greedy_provision")
logger = logger_wrapper.get_logger()

# Generate workflow
workflow_generator = WorkflowGenerator()
profile, timetable = workflow_generator.generate_workflow()

# Set paramters for FaaSEnv
env_params = EnvParameters(
    cpu_total=32*10,
    memory_total=46*10,
    cpu_cap_per_function=32,
    memory_cap_per_function=46
    )

# Make environment
env = gym.make("FaaS-v0", params=env_params, profile=profile, timetable=timetable)
env.seed(114514) # Reproducible, policy gradient has high variance

max_episode = 300
reward_trend = []
avg_slow_down_trend = []
timeout_num_trend = []

# Start random provision
for episode in range(max_episode):
    observation = env.reset()
    reward_sum = 0
    actual_time = 0
    system_time = 0
    
    action = env.action_space.n - 1
    
    while True:
        actual_time = actual_time + 1
        observation, reward, done, info = env.step(action)
        
        if system_time < info["system_time"]:
            system_time = info["system_time"]
            record = info["request_record"].request_per_function_record
            
            #
            # Greedy resource adjustment
            #
            
            # Record latest slow down for each function at each system timestep
            latest_slow_down_record = {}
            for function in profile.function_profile:
                latest_slow_down_record[function.function_id] = 1.0
                
            # Adjustment for each function
            resource_adjust_list = {}
            for function in profile.function_profile:
                resource_adjust_list[function.function_id] = []
            
            # Update latest slow down for each function
            for id in record.keys():
                if len(record[id]) == 0: # No request finished for this function
                    resource_adjust_list[id] = [-1, -1] # Hold 
                else:
                    latest_request = record[id][-1]
                    
                    if latest_request.status == "timeout": # Increase if timeout
                        latest_slow_down_record[id] = 2.0 # Timeout penalty
                    else: # Increase if slow down gets worse
                        latest_slow_down_record[id] = latest_request.get_slow_down()
            
            # Assign resource adjusts. 
            # Functions that have latest slow down over avg get increase
            # Otherwise decrease
            avg_slow_down = np.mean(list(latest_slow_down_record.values()))
            for id in latest_slow_down_record.keys():
                if latest_slow_down_record[id] >= avg_slow_down:
                    resource_adjust_list[id] = [1, 3] # Increase one slot for CPU and memory
                else:
                    resource_adjust_list[id] = [0, 2] # Decrease one slot for CPU and memory
            
            action = encode_action(profile.function_profile, resource_adjust_list)
        else:
            action = env.action_space.n - 1
            
        logger.debug("")
        logger.debug("Actual timestep {}".format(actual_time))
        logger.debug("System timestep {}".format(system_time))
        logger.debug("Take action: {}".format(action))
        logger.debug("Observation: {}".format(observation))
        logger.debug("Reward: {}".format(reward))
        
        reward_sum = reward_sum + reward
        
        if done:
            avg_slow_down = info["avg_slow_down"]
            timeout_num = info["timeout_num"]
            
            logger.info("")
            logger.info("**********")
            logger.info("**********")
            logger.info("**********")
            logger.info("")
            logger.info("Episode {} finished after:".format(episode))
            logger.info("{} actual timesteps".format(actual_time))
            logger.info("{} system timesteps".format(system_time))
            logger.info("Total reward: {}".format(reward_sum))
            logger.info("Avg slow down: {}".format(avg_slow_down))
            logger.info("Timeout num: {}".format(timeout_num))
            
            reward_trend.append(reward_sum)
            avg_slow_down_trend.append(avg_slow_down)
            timeout_num_trend.append(timeout_num)
            
            break

# Plot each episode 
plotter = Plotter()
# ploter.plot_save("Greedy", reward_trend, avg_slow_down_trend, timeout_num_trend)
plotter.plot_show(reward_trend, avg_slow_down_trend, timeout_num_trend)

