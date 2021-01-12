import sys
sys.path.append("../../gym")
import matplotlib.pyplot as plt
import gym
from logger import Logger
from plotter import Plotter
from ppo2_agent import PPO2Agent
from utils import log_trends, log_resource_utils, log_function_throughput


<<<<<<< HEAD
=======

#
# Policy gradient provision strategy
#

>>>>>>> 396338cac6ea37244761b01a938a2a8d4a56f49c
def lambda_rm_eval(
    profile,
    timetable,
    env_params,
    logger_wrapper,
<<<<<<< HEAD
    max_episode=10,
    hidden_dims=[32, 16],
    learning_rate=0.001,
    discount_factor=1,
    ppo_clip=0.2,
    ppo_epoch=5,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    checkpoint_path="ckpt/best_model.pth",
    save_plot=False,
    show_plot=True,
=======
    max_episode,
    hidden_dims,
    learning_rate,
    discount_factor,
    ppo_clip,
    ppo_epoch,
    value_loss_coef,
    entropy_coef,
    checkpoint_path,
    save_plot,
    show_plot,
>>>>>>> 396338cac6ea37244761b01a938a2a8d4a56f49c
):
    rm = "LambdaRM_eval"
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
        entropy_coef=entropy_coef,
    )

    # Restore checkpoint model
    agent.load(checkpoint_path)
    
    # Trends recording
    reward_trend = []
    avg_completion_time_trend = []
    timeout_num_trend = []
    loss_trend = []
    avg_completion_time_per_function_trend = {}
    for function_id in function_profile.keys():
        avg_completion_time_per_function_trend[function_id] = []
    
    # Start random provision
    for episode in range(max_episode):
        observation, mask = env.reset()
        agent.reset()

        actual_time = 0
        system_time = 0
        reward_sum = 0

        function_throughput_list = []
        
        episode_done = False
        while episode_done is False:
            actual_time = actual_time + 1
            action, value_pred, log_prob = agent.choose_action(observation, mask)
            next_observation, next_mask, reward, done, info = env.step(action)

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
            timeout_num_trend=timeout_num_trend, 
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
