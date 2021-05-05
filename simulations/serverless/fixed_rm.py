import sys
sys.path.append("../../gym")
import gym

from gym.envs.serverless.faas_params import WorkloadParameters, EnvParameters
from logger import Logger
from utils import log_trends, log_resource_utils, log_function_throughput
import params


#
# Fixed provision strategy
#
def fixed_rm(
    logger_wrapper
):
    rm = "FixedRM"

    # Set up logger
    logger = logger_wrapper.get_logger(rm, True)
    
    # Start
    episode = 0
    for exp_id in params.EXP_EVAL:
        # Set paramters for workloads
        workload_params = WorkloadParameters(
            azure_file_path=params.AZURE_FILE_PATH,
            exp_id=exp_id
        )

        # Set paramters for Environment
        env_params = EnvParameters(
            cluster_size=params.CLUSTER_SIZE,
            user_cpu_per_server=params.USER_CPU_PER_SERVER,
            user_memory_per_server=params.USER_MEMORY_PER_SERVER,
            keep_alive_window=params.KEEP_ALIVE_WINDOW,
            cpu_cap_per_function=params.CPU_CAP_PER_FUNCTION,
            memory_cap_per_function=params.MEMORY_CAP_PER_FUNCTION,
            memory_mb_limit=params.MEMORY_MB_LIMIT,
            interval=params.ENV_INTERVAL_LIMIT,
            fail_penalty=params.FAIL_PENALTY
        )

        # Set up environment
        env = gym.make(
            "FaaS-v0", 
            workload_params=workload_params,
            env_params=env_params
        )
        env.seed(114514)

        # Trends recording
        reward_trend = []
        avg_completion_time_slo_trend = []
        avg_completion_time_trend = []
        timeout_num_trend = []
        avg_trends_per_function = {}
        for function_id in env.profile.get_function_profile().keys():
            avg_trends_per_function[function_id] = {}
            avg_trends_per_function[function_id]["avg_completion_time"] = []
            avg_trends_per_function[function_id]["avg_completion_time_slo"] = []
            avg_trends_per_function[function_id]["avg_cpu_slo"] = []
            avg_trends_per_function[function_id]["avg_memory_slo"] = []

        for episode_per_exp in range(params.MAX_EPISODE_EVAL):
            # Record total number of events
            total_events = env.event_pq.get_total_size()

            # Reset logger, env, agent
            logger = logger_wrapper.get_logger(rm, False)
            observation, mask, current_timestep, current_function_id = env.reset()

            actual_time = 0
            system_time = 0
            reward_sum = 0

            function_throughput_list = []

            action = None

            episode_done = False
            while episode_done is False:
                actual_time = actual_time + 1
                next_observation, next_mask, reward, done, info, next_timestep, next_function_id = env.step(
                    current_timestep=current_timestep,
                    current_function_id=current_function_id,
                    action=env.decode_action(action)
                )

                if system_time < info["system_time"]:
                    system_time = info["system_time"]
                    function_throughput_list.append(info["function_throughput"])

                logger.debug("")
                logger.debug("Actual timestep {}".format(actual_time))
                logger.debug("System timestep {}".format(system_time))
                logger.debug("Take action: {}".format(action))
                logger.debug("Observation: {}".format(observation))
                logger.debug("Reward: {}".format(reward))
            
                reward_sum = reward_sum + reward
            
                if done:
                    if system_time < info["system_time"]:
                        system_time = info["system_time"]

                    avg_completion_time_slo = info["avg_completion_time_slo"]
                    avg_completion_time = info["avg_completion_time"]
                    timeout_num = info["timeout_num"]
                
                    logger.info("")
                    logger.info("**********")
                    logger.info("**********")
                    logger.info("**********")
                    logger.info("")
                    logger.info("Running {}".format(rm))
                    logger.info("Exp {}, Episode {} finished".format(exp_id, episode))
                    logger.info("{} actual timesteps".format(actual_time))
                    logger.info("{} system timesteps".format(system_time))
                    logger.info("Total events: {}".format(total_events))
                    logger.info("Total reward: {}".format(reward_sum))
                    logger.info("Avg completion time SLO: {}".format(avg_completion_time_slo))
                    logger.info("Avg completion time: {}".format(avg_completion_time))
                    logger.info("Timeout num: {}".format(timeout_num))
                    logger.info("")
                
                    reward_trend.append(reward_sum)
                    avg_completion_time_slo_trend.append(avg_completion_time_slo)
                    avg_completion_time_trend.append(avg_completion_time)
                    timeout_num_trend.append(timeout_num)
                    
                    request_record = info["request_record"]
                    # Log average trends per function
                    for function_id in avg_trends_per_function.keys():
                        avg_trends_per_function[function_id]["avg_completion_time"].append(request_record.get_avg_completion_time_per_function(function_id))
                        avg_trends_per_function[function_id]["avg_completion_time_slo"].append(request_record.get_avg_completion_time_slo_per_function(function_id))
                        avg_trends_per_function[function_id]["avg_cpu_slo"].append(request_record.get_avg_cpu_slo_per_function(function_id))
                        avg_trends_per_function[function_id]["avg_memory_slo"].append(request_record.get_avg_memory_slo_per_function(function_id))

                    # Log resource utilization 
                    resource_utils_record = info["resource_utils_record"]
                    log_resource_utils(
                        overwrite=False, 
                        rm_name=rm, 
                        exp_id=exp_id,
                        logger_wrapper=logger_wrapper,
                        episode=episode, 
                        resource_utils_record=resource_utils_record
                    )

                    # Log function throughput
                    log_function_throughput(
                        overwrite=False, 
                        rm_name=rm, 
                        exp_id=exp_id,
                        logger_wrapper=logger_wrapper,
                        episode=episode, 
                        function_throughput_list=function_throughput_list
                    )
                    
                    episode_done = True
            
                observation = next_observation
                mask = next_mask
                current_timestep = next_timestep
                current_function_id = next_function_id

            episode = episode + 1
    
        # Log trends
        log_trends(
            overwrite=False,
            rm_name=rm,
            exp_id=exp_id,
            logger_wrapper=logger_wrapper,
            reward_trend=reward_trend,
            avg_completion_time_slo_trend=avg_completion_time_slo_trend,
            avg_completion_time_trend=avg_completion_time_trend,
            avg_trends_per_function=avg_trends_per_function,
            timeout_num_trend=timeout_num_trend
        )
    