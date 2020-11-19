import sys
sys.path.append("../../gym")
import matplotlib.pyplot as plt
import gym
from logger import Logger
from plotter import Plotter



#
# Fixed provision strategy
#
def fixed_rm(
    profile,
    timetable,
    env_params,
    logger_wrapper,
    max_episode=500,
    save_plot=False,
    show_plot=True
):
    rm = "FixedRM"

    # Set up logger
    logger = logger_wrapper.get_logger(rm, True)
    
    # Make environment
    env = gym.make("FaaS-v0", params=env_params, profile=profile, timetable=timetable)
    env.seed(114514) # Reproducible, policy gradient has high variance
    
    reward_trend = []
    avg_slow_down_trend = []
    avg_completion_time_trend = []
    timeout_num_trend = []
    
    # Start random provision
    for episode in range(max_episode):
        observation = env.reset()
        reward_sum = 0
        actual_time = 0
        system_time = 0
        
        action = env.action_space.n - 1
        while True:
            actual_time = actual_time + 1
            observation, reward, done, info = env.step(action)
            
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
                
                break
    
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

    