import sys
sys.path.append("../../gym")
import gym

from gym.envs.serverless.faas_params import WorkloadParameters, EnvParameters
from logger import Logger
from utils import log_trends, log_function_throughput, export_csv_percentile, export_csv_per_invocation
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
            
                if done:
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
                    export_csv_per_invocation(
                        rm_name=rm,
                        exp_id=exp_id,
                        episode=episode,
                        csv_per_invocation=request_record.get_csv_per_invocation()
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
    