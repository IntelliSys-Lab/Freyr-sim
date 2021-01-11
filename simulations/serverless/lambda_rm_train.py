import sys
sys.path.append("../../gym")
import matplotlib.pyplot as plt
import numpy as np
import torch
import gym

from gym.envs.serverless.faas_params import EnvParameters
from workload_generator import WorkloadGenerator
from logger import Logger
from plotter import Plotter
from ppo2_agent import PPO2Agent
from utils import log_trends

import params


#
# Policy gradient provision strategy
#

def lambda_rm_train(
    profile,
    timetable,
    env_params,
    logger_wrapper,
    max_episode,
    hidden_dims,
    learning_rate,
    discount_factor,
    ppo_clip,
    ppo_epoch,
    value_loss_coef,
    entropy_coef,
    model_save_path,
    save_plot,
    show_plot,
):
    rm = "LambdaRM_train"
    function_profile = profile.get_function_profile()

    # Set up logger
    logger = logger_wrapper.get_logger(rm, True)
    
    # Make environment
    env = gym.make("FaaS-v0", params=env_params, profile=profile, timetable=timetable)
    env.seed(114514) # Reproducible, policy gradient has high variance
    
    # Set up policy gradient agent
    agent = PPO2Agent(
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
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
    avg_completion_time_per_function_trend = {}
    for function_id in function_profile.keys():
        avg_completion_time_per_function_trend[function_id] = []

    # Record max sum rewards
    max_reward_sum = -10e8
    
    # Start random provision
    for episode in range(max_episode):
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

            observation_history.append(observation)
            mask_history.append(mask)
            action_history.append(action)
            reward_history.append(reward)
            value_history.append(value_pred)
            log_prob_history.append(log_prob)
            
            if system_time < info["system_time"]:
                system_time = info["system_time"]
                
            logger.debug("")
            logger.debug("Actual timestep {}".format(actual_time))
            logger.debug("System timestep {}".format(system_time))
            logger.debug("Take action: {}".format(action))
            logger.debug("Observation: {}".format(observation))
            logger.debug("Reward: {}".format(reward))
            
            reward_sum = reward_sum + reward
            
            if done:
                loss = agent.update(
                    observation_history=torch.cat(observation_history, dim=0),
                    mask_history=torch.cat(mask_history, dim=0),
                    action_history=torch.cat(action_history, dim=0),
                    reward_history=reward_history,
                    value_history=torch.cat(value_history).squeeze(),
                    log_prob_history=torch.cat(log_prob_history, dim=0)
                )
                avg_completion_time = info["avg_completion_time"]
                timeout_num = info["timeout_num"]

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
        avg_completion_time_per_function_trend=avg_completion_time_per_function_trend,
        timeout_num_trend=timeout_num_trend,
        loss_trend=loss_trend,
    )


if __name__ == "__main__":
    # Set up logger wrapper
    logger_wrapper = Logger()

    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")
    print("Generating workloads...")

    workload_generator = WorkloadGenerator()
    profile, timetable = workload_generator.generate_workload(
        default=params.workload_type,
        max_timestep=60,
        azure_file_path="azurefunctions-dataset2019/",
        memory_traces_file="sampled_memory_traces_0.csv",
        duration_traces_file="sampled_duration_traces_0.csv",
        invocation_traces_file="sampled_invocation_traces_0.csv"
    )

    env_params = EnvParameters(
        max_function=params.max_function,
        max_server=params.max_server,
        cluster_size=params.cluster_size,
        user_cpu_per_server=params.user_cpu_per_server,
        user_memory_per_server=params.user_memory_per_server,
        keep_alive_window_per_server=params.keep_alive_window_per_server,
        cpu_cap_per_function=params.cpu_cap_per_function,
        memory_cap_per_function=params.memory_cap_per_function,
        timeout_penalty=params.timeout_penalty,
        interval=params.interval,
    )

    print("")
    print("Start training...")

    lambda_rm_train(
        profile=profile,
        timetable=timetable,
        env_params=env_params,
        max_episode=params.max_episode,
        hidden_dims=params.hidden_dims,
        learning_rate=params.learning_rate,
        discount_factor=params.discount_factor,
        ppo_clip=params.ppo_clip,
        ppo_epoch=params.ppo_epoch,
        value_loss_coef=params.value_loss_coef,
        entropy_coef=params.entropy_coef,
        model_save_path=params.model_save_path,
        save_plot=True,
        show_plot=False,
        logger_wrapper=logger_wrapper
    )

    print("")
    print("Training finished!")
