import sys
sys.path.append("../../gym")
import time
import gym

from logger import Logger
from ppo2_agent import PPO2Agent
from gym.envs.serverless.faas_params import WorkloadParameters, EnvParameters
from utils import log_trends, log_function_throughput, log_per_function, log_per_invocation
import params



def lambda_rm_eval(
    logger_wrapper
):
    rm = "LambdaRM_eval"

    # Set up logger
    logger = logger_wrapper.get_logger(rm, True)
    
    # Set up policy gradient agent
    agent = PPO2Agent(
        observation_dim=params.STATE_DIM,
        action_dim=params.ACTION_DIM,
        hidden_dims=params.HIDDEN_DIMS,
        learning_rate=params.LEARNING_RATE,
        discount_factor=params.DISCOUNT_FACTOR,
        ppo_clip=params.PPO_CLIP,
        ppo_epoch=params.PPO_EPOCH,
        value_loss_coef=params.VALUE_LOSS_COEF,
        entropy_coef=params.ENTROPY_COEF
    )

    # Restore checkpoint model
    agent.load(params.MODEL_SAVE_PATH + params.MODEL_NAME)

    # Start training
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

        avg_duration_slo_per_function = {}
        avg_harvest_cpu_per_function = {}
        avg_harvest_memory_per_function = {}
        avg_reduced_duration_per_function = {}
        for function_id in env.profile.get_function_profile().keys():
            avg_duration_slo_per_function[function_id] = []
            avg_harvest_cpu_per_function[function_id] = []
            avg_harvest_memory_per_function[function_id] = []
            avg_reduced_duration_per_function[function_id] = []

        for episode_per_exp in range(params.MAX_EPISODE_EVAL):
            # Record total number of events
            total_events = env.event_pq.get_total_size()

            # Reset logger, env, agent
            logger = logger_wrapper.get_logger(rm, False)
            observation, mask, current_timestep, current_function_id = env.reset()
            agent.reset()

            actual_time = 0
            system_time = 0
            reward_sum = 0

            function_throughput_list = []

            episode_done = False
            while episode_done is False:
                # before_eval = int(round(time.time() * 1000))
                action, _, value_pred, log_prob = agent.choose_action(observation, mask)
                # after_eval = int(round(time.time() * 1000))
                # print("Eval overhead: {}".format(after_eval - before_eval))

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
                    
                    reward_trend.append(reward_sum)
                    avg_duration_slo_trend.append(avg_duration_slo)
                    avg_harvest_cpu_percent_trend.append(avg_harvest_cpu_percent)
                    avg_harvest_memory_percent_trend.append(avg_harvest_memory_percent)
                    slo_violation_percent_trend.append(slo_violation_percent)
                    acceleration_pecent_trend.append(acceleration_pecent)
                    timeout_num_trend.append(timeout_num)
                    
                    request_record = info["request_record"]

                    # Log per function
                    for function_id in avg_duration_slo_per_function.keys():
                        avg_duration_slo_per_function[function_id].append(request_record.get_avg_duration_slo_per_function(function_id))
                        avg_harvest_cpu_per_function[function_id].append(request_record.get_avg_harvest_cpu_per_function(function_id))
                        avg_harvest_memory_per_function[function_id].append(request_record.get_avg_harvest_memory_per_function(function_id))
                        avg_reduced_duration_per_function[function_id].append(request_record.get_reduced_duration_per_function(function_id))
                    
                    # Log function throughput
                    log_function_throughput(
                        overwrite=False, 
                        rm_name=rm, 
                        exp_id=exp_id,
                        logger_wrapper=logger_wrapper,
                        episode=episode, 
                        function_throughput_list=function_throughput_list
                    )

                    # Log per invocation
                    log_per_invocation(
                        overwrite=False,
                        rm_name=rm,
                        exp_id=exp_id,
                        logger_wrapper=logger_wrapper,
                        episode=episode, 
                        duration_slo_per_invocation=request_record.get_duration_slo_per_invocation(),
                        harvest_cpu_per_invocation=request_record.get_harvest_cpu_per_invocation(),
                        harvest_memory_per_invocation=request_record.get_harvest_memory_per_invocation(),
                        reduced_duration_per_invocation=request_record.get_reduced_duration_per_invocation()
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

        # Log per function
        log_per_function(
            overwrite=False, 
            rm_name=rm, 
            exp_id=exp_id,
            logger_wrapper=logger_wrapper,
            avg_duration_slo_per_function=avg_duration_slo_per_function,
            avg_harvest_cpu_per_function=avg_harvest_cpu_per_function,
            avg_harvest_memory_per_function=avg_harvest_memory_per_function,
            avg_reduced_duration_per_function=avg_reduced_duration_per_function
        )
    