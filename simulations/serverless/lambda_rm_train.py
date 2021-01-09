import sys
sys.path.append("../../gym")
import matplotlib.pyplot as plt
import numpy as np
import torch
import multiprocessing
import gym

from gym.envs.serverless.faas_params import EnvParameters
from workload_generator import WorkloadGenerator
from logger import Logger
from plotter import Plotter
from ppo2_agent import PPO2Agent
from utils import log_trends


#
# Generating workloads via multiprocessing
#

def batch_workload(
    workload_id,
    workload_type,
    max_timestep,
    max_function,
    max_server,
    cluster_size,
    user_cpu_per_server,
    user_memory_per_server,
    keep_alive_window_per_server,
    cpu_cap_per_function,
    memory_cap_per_function,
    interval,
    timeout_penalty,
    result_dict
):
    # Set up workload generator
    workload_generator = WorkloadGenerator()

    azure_file_path="azurefunctions-dataset2019/"
    memory_traces_file="sampled_memory_traces_{}.csv".format(workload_id)
    duration_traces_file="sampled_duration_traces_{}.csv".format(workload_id)
    invocation_traces_file="sampled_invocation_traces_{}.csv".format(workload_id)

    profile, timetable = workload_generator.generate_workload(
        default=workload_type,
        max_timestep=max_timestep,
        azure_file_path=azure_file_path,
        memory_traces_file=memory_traces_file,
        duration_traces_file=duration_traces_file,
        invocation_traces_file=invocation_traces_file
    )

    # Set paramters 
    env_params = EnvParameters(
        max_function=max_function,
        max_server=max_server,
        cluster_size=cluster_size,
        user_cpu_per_server=user_cpu_per_server,
        user_memory_per_server=user_memory_per_server,
        keep_alive_window_per_server=keep_alive_window_per_server,
        cpu_cap_per_function=cpu_cap_per_function,
        memory_cap_per_function=memory_cap_per_function,
        interval=interval,
        timeout_penalty=timeout_penalty
    )

    result_dict[workload_id]["profile"] = profile
    result_dict[workload_id]["timetable"] = timetable
    result_dict[workload_id]["env_params"] = env_params

def generate_workload_dict(
    workload_type,
    max_workload,
    max_timestep, 
    max_function,
    max_server,
    cluster_size,
    user_cpu_per_server,
    user_memory_per_server,
    keep_alive_window_per_server,
    cpu_cap_per_function,
    memory_cap_per_function,
    interval,
    timeout_penalty
):  
    # Init workload dict
    workload_dict = {}

    # Set up multithreading
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    jobs = []

    for workload_id in range(max_workload):
        result_dict[workload_id] = manager.dict()
        
        p = multiprocessing.Process(
            target=batch_workload,
            args=(
                workload_id, 
                workload_type, 
                max_timestep, 
                max_function,
                max_server,
                cluster_size,
                user_cpu_per_server,
                user_memory_per_server,
                keep_alive_window_per_server,
                cpu_cap_per_function,
                memory_cap_per_function,
                interval,
                timeout_penalty,
                result_dict,
            )
        )
        jobs.append(p)
        p.start()
    
    for p in jobs:
        p.join()
    
    # Retrieve workloads from result dict
    for workload_id in result_dict.keys():
        workload_dict[workload_id] = {}
        workload_dict[workload_id]["profile"] = result_dict[workload_id]["profile"]
        workload_dict[workload_id]["timetable"] = result_dict[workload_id]["timetable"]
        workload_dict[workload_id]["env_params"] = result_dict[workload_id]["env_params"]

    return workload_dict
        
#
# Batch training via torch multiprocessing
#

def batch_training(
    workload_id,
    env,
    agent,
    result_dict
):
    observation, mask = env.reset()
    agent.reset()

    actual_time = 0
    system_time = 0
    reward_sum = 0

    observation_history = []
    mask_history = []
    action_history = []
    reward_history = []
    value_history = []
    log_prob_history = []

    episode_done = False
    while episode_done is False:
        actual_time = actual_time + 1
        action, value_pred, log_prob = agent.choose_action(observation, mask)
        next_observation, next_mask, reward, done, info = env.step(action)

        # Detach tensors
        observation = observation.detach()
        mask = mask.detach()
        action = action.detach()
        value_pred = value_pred.detach()
        log_prob = log_prob.detach()

        observation_history.append(observation)
        mask_history.append(mask)
        action_history.append(action)
        reward_history.append(reward)
        value_history.append(value_pred)
        log_prob_history.append(log_prob)

        if system_time < info["system_time"]:
            system_time = info["system_time"]

        reward_sum = reward_sum + reward
        
        if done:
            result_dict[workload_id]["actual_time"] = actual_time
            result_dict[workload_id]["system_time"] = system_time
            result_dict[workload_id]["reward_sum"] = reward_sum
            result_dict[workload_id]["avg_completion_time"] = info["avg_completion_time"]
            result_dict[workload_id]["timeout_num"] = info["timeout_num"]

            result_dict[workload_id]["observation_history"] = observation_history
            result_dict[workload_id]["mask_history"] = mask_history
            result_dict[workload_id]["action_history"] = action_history
            result_dict[workload_id]["reward_history"] = reward_history
            result_dict[workload_id]["value_history"] = value_history
            result_dict[workload_id]["log_prob_history"]= log_prob_history
            
            episode_done = True
        
        observation = next_observation
        mask = next_mask

#
# Process all data collected from the result dict
#

def process_result_dict(result_dict):
    actual_time_batch = []
    system_time_batch = []
    reward_sum_batch = []
    avg_completion_time_batch = []
    timeout_num_batch = []

    observation_history_batch = []
    mask_history_batch = []
    action_history_batch = []
    reward_history_batch = []
    value_history_batch = []
    log_prob_history_batch = []
    
    for workload_id in result_dict.keys():
        actual_time_batch.append(result_dict[workload_id]["actual_time"])
        system_time_batch.append(result_dict[workload_id]["system_time"])
        reward_sum_batch.append(result_dict[workload_id]["reward_sum"])
        avg_completion_time_batch.append(result_dict[workload_id]["avg_completion_time"])
        timeout_num_batch.append(result_dict[workload_id]["timeout_num"])

        observation_history_batch.append(torch.cat(result_dict[workload_id]["observation_history"], dim=0))
        mask_history_batch.append(torch.cat(result_dict[workload_id]["mask_history"], dim=0))
        action_history_batch.append(torch.cat(result_dict[workload_id]["action_history"], dim=0))
        reward_history_batch.append(result_dict[workload_id]["reward_history"])
        value_history_batch.append(torch.cat(result_dict[workload_id]["value_history"]).squeeze())
        log_prob_history_batch.append(torch.cat(result_dict[workload_id]["log_prob_history"], dim=0))

    actual_time = np.mean(actual_time_batch)
    system_time = np.mean(system_time_batch)
    reward_sum = np.mean(reward_sum_batch)
    avg_completion_time = np.mean(avg_completion_time_batch)
    timeout_num = np.mean(timeout_num_batch)

    return actual_time, system_time, reward_sum, avg_completion_time, timeout_num, \
        observation_history_batch, mask_history_batch, action_history_batch, reward_history_batch, value_history_batch, log_prob_history_batch

#
# Policy gradient provision strategy
#

def lambda_rm_train(
    workload_dict,
    logger_wrapper,
    max_episode,
    observation_dim,
    action_dim,
    hidden_dims,
    learning_rate,
    discount_factor,
    ppo_clip,
    ppo_epoch,
    value_loss_coef,
    entropy_coef,
    model_save_path,
    save_plot=False,
    show_plot=True,
):
    rm = "LambdaRM_train"

    # Set up logger
    logger = logger_wrapper.get_logger(rm, True)

    # Make env and result dict
    env_dict = {}
    for workload_id in workload_dict.keys():
        workload = workload_dict[workload_id]
        profile = workload["profile"]
        timetable = workload["timetable"]
        env_params = workload["env_params"]

        # Make environment
        env = gym.make("FaaS-v0", params=env_params, profile=profile, timetable=timetable)
        env.seed(114514) # Reproducible, policy gradient has high variance
        env_dict[workload_id] = env

    # Set up policy gradient agent
    agent = PPO2Agent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        ppo_clip=ppo_clip,
        ppo_epoch=ppo_epoch,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef
    )
    
    # Trends recording
    reward_trend = []
    avg_completion_time_trend = []
    timeout_num_trend = []
    loss_trend = []

    # Record max sum rewards
    max_reward_sum = -10e8
    
    # Start training
    for episode in range(max_episode):
        # Set up multithreading
        manager = torch.multiprocessing.Manager()
        result_dict = manager.dict()
        batch = []

        for workload_id in env_dict.keys():
            env = env_dict[workload_id]
            result_dict[workload_id] = manager.dict()
            
            p = torch.multiprocessing.Process(
                target=batch_training,
                args=(workload_id, env, agent, result_dict,)
            )
            batch.append(p)
            p.start()

        for p in batch:
            p.join()

        # Process results
        actual_time, system_time, reward_sum, avg_completion_time, timeout_num, \
            observation_history_batch, mask_history_batch, action_history_batch, reward_history_batch, value_history_batch, log_prob_history_batch \
                = process_result_dict(result_dict)

        loss = agent.update(
            observation_history_batch=observation_history_batch,
            mask_history_batch=mask_history_batch,
            action_history_batch=action_history_batch,
            reward_history_batch=reward_history_batch,
            value_history_batch=value_history_batch,
            log_prob_history_batch=log_prob_history_batch
        )

        # Save the best model
        if max_reward_sum < reward_sum:
            max_reward_sum = reward_sum
            agent.save(model_save_path)
        
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
        logger.info("Loss: {}".format(loss))
        
        reward_trend.append(reward_sum)
        avg_completion_time_trend.append(avg_completion_time)
        timeout_num_trend.append(timeout_num)
        loss_trend.append(loss)

    # Plot each episode 
    plotter = Plotter()
    
    if save_plot is True:
        plotter.plot_save(
            prefix_name=rm, 
            reward_trend=reward_trend, 
            avg_completion_time_trend=avg_completion_time_trend,
            timeout_num_trend=timeout_num_trend, 
            loss_trend=loss_trend
        )
    if show_plot is True:
        plotter.plot_show(
            reward_trend=reward_trend, 
            avg_completion_time_trend=avg_completion_time_trend, 
            timeout_num_trend=timeout_num_trend, 
            loss_trend=loss_trend
        )

    # Log trends
    log_trends(
        logger_wrapper=logger_wrapper,
        rm_name=rm,
        overwrite=False,
        reward_trend=reward_trend,
        avg_completion_time_trend=avg_completion_time_trend,
        timeout_num_trend=timeout_num_trend,
        loss_trend=loss_trend,
    )


if __name__ == "__main__":
    # Prevent torch.multiprocessing from deadlocks: https://github.com/pytorch/pytorch/issues/48382
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        print("Unable to set_start_method('spawn')!")

    # Training paramters
    workload_type = "azure"
    max_workload = 1
    max_episode = 1000
    hidden_dims = [32, 16]
    learning_rate = 0.001
    discount_factor = 1
    ppo_clip = 0.2
    ppo_epoch = 5
    value_loss_coef = 0.5
    entropy_coef = 0.01
    model_save_path = "ckpt/best_model.pth"
    max_timestep = 60
    max_function = 200
    max_server = 20
    cluster_size = 10
    user_cpu_per_server = 64
    user_memory_per_server = 64
    keep_alive_window_per_server = 60
    cpu_cap_per_function = 8
    memory_cap_per_function = 8
    interval = 1
    timeout_penalty = 600

    # Set up logger wrapper
    logger_wrapper = Logger()

    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")
    print("Generating workloads...")
    workload_dict = generate_workload_dict(
        workload_type=workload_type, 
        max_workload=max_workload,
        max_timestep=max_timestep, 
        max_function=max_function,
        max_server=max_server,
        cluster_size=cluster_size,
        user_cpu_per_server=user_cpu_per_server,
        user_memory_per_server=user_memory_per_server,
        keep_alive_window_per_server=keep_alive_window_per_server,
        cpu_cap_per_function=cpu_cap_per_function,
        memory_cap_per_function=memory_cap_per_function,
        interval=interval,
        timeout_penalty=timeout_penalty
    )

    # Start training
    observation_dim = 1 + 2 * max_server + 8 * max_function
    action_dim = 4 * max_function + 1

    print("")
    print("Start training...")
    lambda_rm_train(
        workload_dict=workload_dict,
        logger_wrapper=logger_wrapper,
        max_episode=max_episode,
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        ppo_clip=ppo_clip,
        ppo_epoch=ppo_epoch,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef,
        model_save_path=model_save_path,
        save_plot=True,
        show_plot=False,
    )

    print("")
    print("Training finished!")