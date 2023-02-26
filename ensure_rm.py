from env import Environment
from params import WorkloadParameters, EnvParameters
from logger import Logger
from utils import log_trends, log_function_throughput, export_csv_percentile, export_csv_per_invocation
import params


#
# ENSURE RM
#
def ensure_rm(logger_wrapper):
    rm = "ENSURE_RM"

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
        env = Environment(
            workload_params=workload_params,
            env_params=env_params
        )

        # Trends recording
        reward_trend = []
        avg_duration_slo_trend = []
        avg_harvest_cpu_percent_trend = []
        avg_harvest_memory_percent_trend = []
        slo_violation_percent_trend = []
        acceleration_pecent_trend = []
        timeout_num_trend = []

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

            action = {}

            # ENSURE util
            update_threshold_dict = {}
            for function_id in env.profile.get_function_profile().keys():
                update_threshold_dict[function_id] = 0

            episode_done = False
            while episode_done is False:
                actual_time = actual_time + 1
                next_observation, next_mask, reward, done, info, next_timestep, next_function_id = env.step(
                    current_timestep=current_timestep,
                    current_function_id=current_function_id,
                    action=action
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

                if done is False:
                    request_record = info["request_record"]
                    total_available_cpu = info["total_available_cpu"]
                    total_available_memory = info["total_available_memory"]

                    #
                    # ENSURE dynamic CPU adjustment
                    #
                
                    window_size = 3
                    latency_threshold = 1.10

                    # Classify the function
                    cpu_max = env.env_params.cpu_cap_per_function
                    if request_record.get_avg_completion_time_per_function(next_function_id) > 5: # MP
                        num_update_threshold = 5
                        cpu_step = 1
                    else: # ET
                        num_update_threshold = 3
                        cpu_step = 2

                    # If the function reaches threshold of updates
                    if update_threshold_dict[function_id] >= num_update_threshold:
                        # Monitor via a moving window
                        request_window = request_record.get_last_n_done_request_per_function(next_function_id, window_size)
                        if len(request_window) > 0:
                            total_completion_time_in_window = 0
                            for request in request_window:
                                total_completion_time_in_window = total_completion_time_in_window + request.get_completion_time()
                            avg_completion_time_in_window = total_completion_time_in_window / window_size

                            # If performance degrade, increase its CPU allocation based on step
                            if total_available_cpu > cpu_step:
                                if avg_completion_time_in_window / env.profile.get_function_profile()[next_function_id].params.min_duration >= latency_threshold:
                                    action["cpu"] = min(request_window[0].get_cpu() + cpu_step, env.env_params.cpu_cap_per_function)
                                else:
                                    action = {} # No increase
                            else: # If reaches capacity, rebalance CPU from other functions
                                action["cpu"] = max(request_window[0].get_cpu() - cpu_step, 1)
                        else:
                            action = {} # No increase

                        update_threshold_dict[function_id] = 0
                    else:
                        update_threshold_dict[function_id] = update_threshold_dict[function_id]  + 1
                else:
                    if system_time < info["system_time"]:
                        system_time = info["system_time"]

                    avg_duration_slo = info["avg_duration_slo"]
                    avg_harvest_cpu_percent = info["avg_harvest_cpu_percent"]
                    avg_harvest_memory_percent = info["avg_harvest_memory_percent"]
                    slo_violation_percent = info["slo_violation_percent"]
                    acceleration_pecent = info["acceleration_pecent"]
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
                    logger.info("Avg relative duration: {}".format(avg_duration_slo))
                    logger.info("Avg harvest CPU percent: {}".format(avg_harvest_cpu_percent))
                    logger.info("Avg harvest memory percent: {}".format(avg_harvest_memory_percent))
                    logger.info("SLO violation percent: {}".format(slo_violation_percent))
                    logger.info("Acceleration pecent: {}".format(acceleration_pecent))
                    logger.info("Timeout num: {}".format(timeout_num))
                    logger.info("")
                
                    reward_trend.append(reward_sum)
                    avg_duration_slo_trend.append(avg_duration_slo)
                    avg_harvest_cpu_percent_trend.append(avg_harvest_cpu_percent)
                    avg_harvest_memory_percent_trend.append(avg_harvest_memory_percent)
                    slo_violation_percent_trend.append(slo_violation_percent)
                    acceleration_pecent_trend.append(acceleration_pecent)
                    timeout_num_trend.append(timeout_num)
                    
                    request_record = info["request_record"]

                    # Log function throughput
                    log_function_throughput(
                        overwrite=False, 
                        rm_name=rm, 
                        exp_id=exp_id,
                        logger_wrapper=logger_wrapper,
                        episode=episode, 
                        function_throughput_list=function_throughput_list
                    )

                    # Export csv per invocation
                    csv_cpu_pos_rfet, csv_cpu_zero_rfet, csv_cpu_neg_rfet, csv_memory_pos_rfet, csv_memory_zero_rfet, csv_memory_neg_rfet = request_record.get_csv_per_invocation()
                    export_csv_per_invocation(
                        rm_name=rm,
                        exp_id=exp_id,
                        episode=episode,
                        csv_cpu_pos_rfet=csv_cpu_pos_rfet,
                        csv_cpu_zero_rfet=csv_cpu_zero_rfet,
                        csv_cpu_neg_rfet=csv_cpu_neg_rfet,
                        csv_memory_pos_rfet=csv_memory_pos_rfet,
                        csv_memory_zero_rfet=csv_memory_zero_rfet,
                        csv_memory_neg_rfet=csv_memory_neg_rfet
                    )

                    # Export csv percentile
                    export_csv_percentile(
                        rm_name=rm,
                        exp_id=exp_id,
                        episode=episode,
                        csv_percentile=request_record.get_csv_percentile()
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
            avg_duration_slo_trend=avg_duration_slo_trend,
            avg_harvest_cpu_percent_trend=avg_harvest_cpu_percent_trend,
            avg_harvest_memory_percent_trend=avg_harvest_memory_percent_trend,
            slo_violation_percent_trend=slo_violation_percent_trend,
            acceleration_pecent_trend=acceleration_pecent_trend,
            timeout_num_trend=timeout_num_trend
        )


if __name__ == "__main__":
    logger_wrapper = Logger()
    ensure_rm(logger_wrapper=logger_wrapper)
