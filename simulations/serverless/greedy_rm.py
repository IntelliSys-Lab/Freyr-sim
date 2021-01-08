import sys
sys.path.append("../../gym")
import numpy as np
import queue
import matplotlib.pyplot as plt
import gym

from gym.envs.serverless.faas_utils import Prioritize
from logger import Logger
from plotter import Plotter
from utils import log_trends, log_resource_utils, log_function_throughput


#
# Naive greedy provision strategy
#                 
def greedy_rm(
    profile,
    timetable,
    env_params,
    logger_wrapper,
    max_episode=10,
    save_plot=False,
    show_plot=True
):
    rm = "GreedyRM"
    function_profile = profile.get_function_profile()

    # Set up logger
    logger = logger_wrapper.get_logger(rm, True)
    
    # Make environment
    env = gym.make("FaaS-v0", params=env_params, profile=profile, timetable=timetable)
    env.seed(114514) # Reproducible, policy gradient has high variance
    
    reward_trend = []
    avg_completion_time_trend = []
    timeout_num_trend = []
    avg_completion_time_per_function_trend = {}
    for function_id in function_profile.keys():
        avg_completion_time_per_function_trend[function_id] = []
    
    # Start random provision
    for episode in range(max_episode):
        observation, mask = env.reset()
        reward_sum = 0
        actual_time = 0
        system_time = 0

        function_throughput_list = []
        
        action = {}
        
        episode_done = False
        while episode_done is False:
            actual_time = actual_time + 1
            next_observation, next_mask, reward, done, info = env.step(action)
            
            if system_time < info["system_time"]:
                system_time = info["system_time"]
                function_throughput_list.append(info["function_throughput"])
                request_record = info["request_record"]
                function_dict = info["function_dict"]

                #
                # Greedy resource adjustment
                #

                function_profile = profile.get_function_profile()

                total_available_cpu = info["total_available_cpu"]
                total_available_memory = info["total_available_memory"]

                function_queue = queue.PriorityQueue()
                action_map = {}
                for function_id in function_dict.keys():
                    function_stats = function_dict[function_id]
                    
                    # Only adjust incoming invocations
                    if function_stats["invoke_num"] > 0:
                        action_map[function_id] = {}
                        action_map[function_id]["cpu"] = function_stats["cpu"]
                        action_map[function_id]["memory"] = function_stats["memory"]

                        priority = function_stats["priority"]
                        function_queue.put(Prioritize(priority, function_id))

                while total_available_cpu > 0 and total_available_memory > 0 and function_queue.empty() is False:
                    function_id = function_queue.get().item
                    function_stats = function_dict[function_id]
                    function = function_profile[function_id]
                    total_sequence_size = function_stats["total_sequence_size"]
                    invoke_num = function_stats["invoke_num"]
                    
                    # Allocate full resources to failed functions immediately
                    if function_stats["is_success"] is False:
                        action_map[function_id]["cpu"] = env.params.cpu_cap_per_function
                        action_map[function_id]["memory"] = env.params.memory_cap_per_function
                    else:
                        new_cpu = np.clip(
                            function_stats["cpu"] + 1, 
                            function.params.cpu_least_hint,
                            function.params.cpu_cap_per_function
                        )
                        new_memory = np.clip(
                            function_stats["memory"] + 1, 
                            function.params.memory_least_hint,
                            function.params.memory_cap_per_function
                        )

                        if total_available_cpu - new_cpu * invoke_num * total_sequence_size > 0 and \
                            total_available_memory - new_memory * invoke_num * total_sequence_size > 0:
                            action_map[function_id]["cpu"] = new_cpu
                            action_map[function_id]["memory"] = new_memory
                            total_available_cpu = total_available_cpu - new_cpu * invoke_num * total_sequence_size
                            total_available_memory = total_available_memory - new_memory * invoke_num * total_sequence_size

                action = action_map

            logger.debug("")
            logger.debug("Actual timestep {}".format(actual_time))
            logger.debug("System timestep {}".format(system_time))
            logger.debug("Take action: {}".format(action))
            logger.debug("Observation: {}".format(observation))
            logger.debug("Reward: {}".format(reward))
            
            reward_sum = reward_sum + reward
            
            if done:
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
                logger.info("Avg completion time: {}".format(avg_completion_time))
                logger.info("Timeout num: {}".format(timeout_num))
                
                reward_trend.append(reward_sum)
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
                
                episode_done = True
            
            observation = next_observation
            mask = next_mask
    
    # Plot each episode 
    plotter = Plotter()
    
    if save_plot is True:
        plotter.plot_save(
            prefix_name=rm, 
            reward_trend=reward_trend, 
            avg_completion_time_trend=avg_completion_time_trend,
            timeout_num_trend=timeout_num_trend
        )
    if show_plot is True:
        plotter.plot_show(
            reward_trend=reward_trend, 
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