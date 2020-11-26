import sys
from numpy import save
sys.path.append("../../gym")
import numpy as np
import matplotlib.pyplot as plt
import gym
from logger import Logger
from plotter import Plotter
from utils import log_trends, log_resource_utils, log_function_throughput


#
# Encode sequential resource changes into discrete actions
#
def encode_action(function_profile, resource_adjust_list):
    actions = []
    
    for index, function in enumerate(function_profile):
        function_id = function.get_function_id()

        if resource_adjust_list[function_id][0] != -1:
            adjust_cpu = index*4 + resource_adjust_list[function_id][0]
            actions.append(adjust_cpu)
        if resource_adjust_list[function_id][1] != -1:
            adjust_memory = index*4 + resource_adjust_list[function_id][1]
            actions.append(adjust_memory)
                    
    return actions
   

#
# Naive greedy provision strategy
#                 
def greedy_rm(
    profile,
    timetable,
    env_params,
    logger_wrapper,
    max_episode=500,
    plot_prefix_name="Greedy",
    save_plot=False,
    show_plot=True
):
    # Set up logger
    logger = logger_wrapper.get_logger("greedy_provision")
    
    # Make environment
    env = gym.make("FaaS-v0", params=env_params, profile=profile, timetable=timetable)
    env.seed(114514) # Reproducible, policy gradient has high variance
    
    reward_trend = []
    avg_slow_down_trend = []
    avg_completion_time_trend = []
    timeout_num_trend = []
    avg_completion_time_per_function_trend = {}
    for function in profile.get_function_profile():
        function_id = function.get_function_id()
        avg_completion_time_per_function_trend[function_id] = []
    
    # Start random provision
    for episode in range(max_episode):
        observation = env.reset()
        reward_sum = 0
        actual_time = 0
        system_time = 0
        
        action = env.action_space.n - 1

        function_throughput_list = []
        
        while True:
            actual_time = actual_time + 1
            next_observation, reward, done, info = env.step(action)
            
            if system_time < info["system_time"]:
                system_time = info["system_time"]
                record = info["request_record"].request_per_function_record
                
                #
                # Greedy resource adjustment: completion time decay
                #
                
                # Record last two completion time for each function and its decay at each system timestep
                completion_time_decay_record = {}
                for function in profile.function_profile:
                    completion_time_decay_record[function.function_id] = 1.0
                    
                # Adjustment for each function
                resource_adjust_list = {}
                for function in profile.function_profile:
                    resource_adjust_list[function.function_id] = []
                
                # Update completion time decay for each function
                for id in record.keys():
                    if len(record[id]) <= 1: # No request finished or no old request for this function
                        resource_adjust_list[id] = [-1, -1] # Hold 
                    else:
                        old_request = record[id][-2]
                        new_request = record[id][-1]

                        if new_request.status == "timeout" or old_request.status == "timeout": 
                            completion_time_decay_record[id] = 114514.0 # Timeout penalty
                        else: 
                            # Update decay
                            completion_time_decay_record[id] = new_request.get_completion_time() / old_request.get_completion_time()

                # Assign resource adjusts. 
                # Functions that have decay (latest completion time) / (previous completion time)
                # over avg get increase, otherwise decrease
                decay_list = []
                for id in completion_time_decay_record.keys():
                    decay_list.append(completion_time_decay_record[id])

                decay_avg = np.mean(decay_list)

                for id in completion_time_decay_record.keys():
                    if completion_time_decay_record[id] >= decay_avg:
                        resource_adjust_list[id] = [1, 3] # Increase one slot for CPU and memory
                    else:
                        resource_adjust_list[id] = [0, 2] # Decrease one slot for CPU and memory
                
                action = encode_action(profile.function_profile, resource_adjust_list)

            logger.debug("")
            logger.debug("Actual timestep {}".format(actual_time))
            logger.debug("System timestep {}".format(system_time))
            logger.debug("Take action: {}".format(action))
            logger.debug("Observation: {}".format(observation))
            logger.debug("Reward: {}".format(reward))
            
            reward_sum = reward_sum + reward
            
            if done:
                avg_slow_down = info["avg_slow_down"]
                avg_completion_time = info["avg_completion_time"]
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
                logger.info("Avg slowdown: {}".format(avg_slow_down))
                logger.info("Avg completion time: {}".format(avg_completion_time))
                logger.info("Timeout num: {}".format(timeout_num))
                
                reward_trend.append(reward_sum)
                avg_slow_down_trend.append(avg_slow_down)
                avg_completion_time_trend.append(avg_completion_time)
                timeout_num_trend.append(timeout_num)

                # Log average completion time per function
                request_record = info["request_record"]
                for function_id in avg_completion_time_per_function_trend.keys():
                    avg_completion_time_per_function_trend[function_id].append(
                        request_record.get_avg_completion_time_per_function(function_id)
                    )

                # Log resource utilization 
                resource_utils_record = info["resource_utils_record"]
                log_resource_utils(
                    logger_wrapper=logger_wrapper,
                    rm_name=rm, 
                    overwrite=False, 
                    episode=episode, 
                    resource_utils_record=resource_utils_record
                )

                # Log function throughput
                log_function_throughput(
                    logger_wrapper=logger_wrapper,
                    rm_name=rm, 
                    overwrite=False, 
                    episode=episode, 
                    function_throughput_list=function_throughput_list
                )
                
                break
            
            observation = next_observation
    
    # Plot each episode 
    plotter = Plotter()
    
    if save_plot is True:
        plotter.plot_save(
            prefix_name=plot_prefix_name, 
            reward_trend=reward_trend, 
            avg_slow_down_trend=avg_slow_down_trend, 
            avg_completion_time_trend=avg_completion_time_trend,
            timeout_num_trend=timeout_num_trend
        )
    if show_plot is True:
        plotter.plot_show(
            reward_trend=reward_trend, 
            avg_slow_down_trend=avg_slow_down_trend, 
            avg_completion_time_trend=avg_completion_time_trend,
            timeout_num_trend=timeout_num_trend,
        )
        
    # Log trends
    log_trends(
        logger_wrapper=logger_wrapper,
        rm_name=rm,
        overwrite=False,
        reward_trend=reward_trend,
        avg_completion_time_trend=avg_completion_time_trend,
        avg_completion_time_per_function_trend=avg_completion_time_per_function_trend,
        timeout_num_trend=timeout_num_trend,
        loss_trend=None,
    )