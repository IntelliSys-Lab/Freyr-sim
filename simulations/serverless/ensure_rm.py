import sys
sys.path.append("../../gym")
import numpy as np
import queue
import matplotlib.pyplot as plt
import gym
from logger import Logger
from plotter import Plotter
from utils import Prioritize, log_trends, log_resource_utils, log_function_throughput


#
# ENSURE provision
#                 
def ensure_rm(
    profile,
    timetable,
    env_params,
    logger_wrapper,
    max_episode=10,
    save_plot=False,
    show_plot=True
):
    rm = "EnsureRM"
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

        # ENSURE parameter
        num_update_threshold_dict = {}
        for function_id in function_profile.keys():
            num_update_threshold_dict[function_id] = 0
        
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
                # ENSURE dynamic CPU adjustment
                #

                total_available_cpu = info["total_available_cpu"]
                total_available_memory = info["total_available_memory"]

                # Paramters
                window_size = 10
                latency_threshold = 1.10

                function_queue = queue.PriorityQueue()
                rebalance_list = []
                action_map = {}

                for function_id in function_dict.keys():
                    function = function_profile[function_id]
                    function_stats = function_dict[function_id]
                    action_map[function_id] = {}
                    action_map[function_id]["cpu"] = function_stats["cpu"]
                    action_map[function_id]["memory"] = function_stats["memory"]

                    # Classify the function
                    if function_stats["avg_completion_time"] > 5: # MP
                        function_stats["max_cpu"] = env_params.cpu_cap_per_function
                        function_stats["num_update_threshold"] = 3
                        function_stats["cpu_step"] = 0.5
                    else: # ET
                        function_stats["max_cpu"] = env_params.cpu_cap_per_function
                        function_stats["num_update_threshold"] = 5
                        function_stats["cpu_step"] = 1

                    # Only evaluate incoming invocations
                    if function_stats["invoke_num"] > 0:
                        priority = function_stats["priority"]
                        function_queue.put(Prioritize(priority, function_id))
                    else:
                        # Otherwise add to rebalance list
                        if num_update_threshold_dict[function_id] >= function_stats["num_update_threshold"]:
                            rebalance_list.append(function_id)

                rebalance = False

                while function_queue.empty() is False:
                    function_id = function_queue.get().item
                    function_stats = function_dict[function_id]
                    function = function_profile[function_id]
                    total_sequence_size = function_stats["total_sequence_size"]
                    invoke_num = function_stats["invoke_num"]
                    
                    # If reach threshold of updates
                    current_update = num_update_threshold_dict[function_id]
                    if current_update >= function_stats["num_update_threshold"]:
                        # Monitor via a moving window
                        request_window = request_record.get_total_request_record_per_function(function_id)[-window_size:]
                        total_completion_time_in_window = 0
                        for request in request_window:
                            total_completion_time_in_window = total_completion_time_in_window + request.get_completion_time()

                        avg_completion_time_in_window = total_completion_time_in_window / window_size

                        # If performance degrade
                        if avg_completion_time_in_window / function.params.min_duration >= latency_threshold:
                            # Increment one step if possible
                            new_cpu = np.clip(
                                function_stats["cpu"] + function_stats["cpu_step"], 
                                function.params.cpu_least_hint,
                                function_stats["max_cpu"]
                            )
                            action_map[function_id]["cpu"] = new_cpu

                            # Otherwise exceed capacity, rebalance from other functions that haven't reached update threshold
                            if total_available_cpu - new_cpu * invoke_num * total_sequence_size >= 0:
                                total_available_cpu = total_available_cpu - new_cpu * invoke_num * total_sequence_size
                            else: 
                                rebalance = True
                    
                        num_update_threshold_dict[function_id] = 0
                    else:
                        num_update_threshold_dict[function_id] = current_update + invoke_num
                
                # Rebalance if needed
                if rebalance is True:
                    for function_id in rebalance_list:
                        function_stats = function_dict[function_id]
                        action_map[function_id]["cpu"] = function_stats["cpu"] - function_stats["cpu_step"]

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
                logger.info("Running {}".format(rm))
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